
from functools import partial
import os
import shutil
import sys
from timeit import default_timer as timer
import traceback
from typing import NamedTuple, Tuple

import chex
import hydra
import jax
import logging
import numpy as np

import flax
from flax import struct
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import imageio
import jax.numpy as jnp
from omegaconf import OmegaConf
import optax
import orbax.checkpoint as ocp
from purejaxrl.wrappers import LogEnvState
from tensorboardX import SummaryWriter

from gen_env.evo.individual import Individual
from evo_accel import EvoState
from gen_env.configs.config import RLConfig
from gen_env.envs.play_env import GenEnvParams, GenEnvState, PlayEnv
from gen_env.utils import gen_rand_env_params, init_base_env, init_config
from il_player_jax import init_bc_agent
from purejaxrl.experimental.s5.wrappers import LogWrapper
from pcgrl_utils import get_rl_ckpt_dir, get_network
from rl_player_jax import init_checkpointer, restore_checkpoint
from utils import evaluate_on_env_params, get_rand_train_envs, init_il_config, init_rl_config, load_elite_envs
logging.getLogger('jax').setLevel(logging.INFO)


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: GenEnvState
    evo_state: EvoState
    last_obs: jnp.ndarray
    train_env_params: GenEnvParams
    curr_env_param_idxs: chex.Array
    val_env_params: GenEnvParams
    # rng_act: jnp.ndarray
#   ep_returns: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    # rng_act: jnp.ndarray


def render(train_env_params, env: LogWrapper, cfg, network, network_params, runner_state):

    rng = jax.random.PRNGKey(cfg.seed)
    n_train_envs = train_env_params.rule_dones.shape[0]
    rng_reset = jax.random.split(rng, n_train_envs)

    # This is just for reference
    dummy_env_params = jax.tree.map(lambda x: x[0], train_env_params)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(rng_reset, train_env_params)

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if cfg.random_agent:
            action = env.action_space(dummy_env_params).sample(rng_act)
        else:
            # obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, n_train_envs)
        # obs, env_state, reward, done, info = env.step(
        #     rng_step, env_state, action[..., 0], env_params
        # )
        env_state: GenEnvState
        obs, env_state, reward, done, info, params = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))(
            rng_step, env_state, action, train_env_params, train_env_params

        )
        # frames = jax.vmap(env.render, in_axes=(0))(env_state, env_params)
        # frame = env.render(env_state)
        rng = jax.random.split(rng)[0]
        # Can't concretize these values inside jitted function (?)
        # So we add the stats on cpu later (below)
        # frame = render_stats(env, env_state, frame)
        return (rng, obs, env_state), (env_state, reward, done, info)

    step_env = jax.jit(step_env, backend='cpu')
    print('Scanning episode steps:')
    _, (states, rewards, dones, infos) = jax.lax.scan(
        step_env, (rng, obs, env_state), None,
        length=1*env.max_episode_steps)

    jax.tree.map(lambda x: jax.device_put(x, jax.devices('cpu')[0]), states)

    states = jax.device_put(states, jax.devices('cpu')[0])
    train_env_params = jax.device_put(train_env_params, jax.devices('cpu')[0])
    
    print('Rendering gifs:')
    # Since we can't jit our render function (yet)
    frames = []
    for ep_i in range(states.env_state.ep_rew.shape[1]):
        ep_frames = []
        env_params_i = jax.tree.map(lambda x: x[ep_i], train_env_params)
        for step_i in range(states.env_state.ep_rew.shape[0]):
            state_i = jax.tree.map(lambda x: x[step_i, ep_i], states.env_state)
            ep_frames.append(env.render(state_i, env_params_i, mode='rgb_array'))
        frames.append(ep_frames)
        # Print reward
        env_params_i: GenEnvParams
        # train_elite_i: IndividualPlaytraceData = jax.tree_map(lambda x: x[ep_i], train_elites)
        print(f'Rendered ep {ep_i}.')

        frame_shapes = [frame.shape for frame in ep_frames]
        max_frame_w, max_frame_h = max(frame_shapes, key=lambda x: x[0])[0], \
            max(frame_shapes, key=lambda x: x[1])[1]
        # Pad frames to be same size
        new_ep_frames = []
        for frame in ep_frames:
            frame = np.pad(frame, ((0, max_frame_w - frame.shape[0]),
                                      (0, max_frame_h - frame.shape[1]),
                                      (0, 0)), constant_values=0)
            # frame[:, :, 3] = 255
            new_ep_frames.append(frame)
        ep_frames = new_ep_frames

        gif_name = f"{cfg._log_dir_rl}/anim_update-{runner_state.update_i}_ep-{ep_i}" + \
            f"{('_randAgent' if cfg.random_agent else '')}.gif"
        vid_name = gif_name[:-4] + ".mp4"
        imageio.v3.imwrite(
            gif_name,
            ep_frames,
            duration=100,
            loop=0,
        )
        # imageio.mimwrite(vid_name, ep_frames, fps=10, quality=8, macro_block_size=1)



def make_train(cfg: RLConfig, init_runner_state: RunnerState, il_params, checkpoint_manager, train_env_params: GenEnvParams,
               val_env_params: GenEnvParams, network, env: LogWrapper):
    cfg.NUM_UPDATES = (
        cfg.total_timesteps // cfg.num_steps // cfg.n_envs
    )
    cfg.MINIBATCH_SIZE = (
        cfg.n_envs * cfg.num_steps // cfg.NUM_MINIBATCHES
    )


    def linear_schedule(count):
        frac = (
            1.0
            - (count // (cfg.NUM_MINIBATCHES * cfg.update_epochs))
            / cfg.NUM_UPDATES
        )
        return cfg["LR"] * frac

    def train(rng, cfg: RLConfig, train_env_params: GenEnvParams, val_env_params: GenEnvParams):

        rng, _rng = jax.random.split(rng)

        # Print number of learnable parameters in the network
        if cfg.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(cfg.MAX_GRAD_NORM),
                optax.adam(cfg.lr, eps=1e-5),
            )
        runner_state = init_runner_state
        steps_prev_complete = checkpoint_manager.latest_step()
        steps_prev_complete = 0 if steps_prev_complete is None else steps_prev_complete
        if runner_state.update_i > 0:

            assert il_params is None
            steps_remaining = cfg.total_timesteps - steps_prev_complete
            cfg.NUM_UPDATES = int(
                steps_remaining // cfg.num_steps // cfg.n_envs)

            # TODO: Overwrite certain config values
        
        if il_params is not None:
            assert init_runner_state.update_i == 0
            train_state = TrainState.create(
                apply_fn=network.apply,
                params={'params': il_params},
                tx=tx,
            )
            runner_state = runner_state.replace(
                train_state=train_state,
            )

        network_params = runner_state.train_state.params

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)

        # Note that there may be duplicates in this first generation due to sampling with replacement above.
        evo_env_params = runner_state.evo_state.env_params
        n_envs = evo_env_params.map.shape[0]

        evo_state = EvoState(env_params=evo_env_params, top_fitness=jnp.zeros(n_envs))

        if cfg.evo_freq != -1:
            # To deal with mismatched shapes after calling _update_step
            train_env_params = evo_state.env_params

        render(train_env_params, env, cfg, network, network_params, runner_state)

    return lambda rng: train(rng, cfg, train_env_params, val_env_params)


@hydra.main(version_base=None, config_path='gen_env/configs', config_name='enjoy_config')
def main(cfg: RLConfig):
    # Try/except to avoid submitit-launcher-plugin swallowing up our error tracebacks.
    try:
        _main(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def _main(cfg: RLConfig):
    init_config(cfg)
    latest_evo_gen = init_il_config(cfg)
    latest_il_update_step = init_rl_config(cfg, latest_evo_gen)
    print(f"Logging RL results to: {cfg._log_dir_rl}")

    # Need to do this before setting up RL checkpoint manager so that it doesn't refer to old checkpoints.
    if cfg.overwrite and os.path.exists(cfg._log_dir_rl):
        shutil.rmtree(cfg._log_dir_rl)

    env, env_params = init_base_env(cfg)

    if cfg.load_il:
        _rng, train_state, t, il_checkpoint_manager = init_bc_agent(cfg, env)
        il_params = train_state.params
        assert not os.path.exists(cfg._log_dir_rl)
    else:
        il_params = None

    rng = jax.random.PRNGKey(cfg.seed)

    train_elites, val_elites = load_elite_envs(cfg._log_dir_common, latest_evo_gen)

    # train_env_params = jax.tree.map(lambda x: x[:cfg.n_envs], train_elites.env_params)
    val_env_params = val_elites.env_params

    # Then we load the latest gen
    if cfg.load_gen is None and cfg.load_game is None:
        # In this case, we generate random (probably garbage) environments upon which to begin training.
        train_env_params = jax.vmap(gen_rand_env_params, in_axes=(None, 0, None, None))(
            cfg, jax.random.split(rng, cfg.n_envs), env_params, env.game_def)
            
    else:
        train_env_params = train_elites.env_params

    if cfg.n_train_envs != -1:
        train_env_params = jax.tree.map(lambda x: x[-cfg.n_train_envs:], train_env_params)
    if cfg.n_val_envs != -1:
        val_env_params = jax.tree.map(lambda x: x[-cfg.n_val_envs:], val_env_params)

    del train_elites, val_elites

    checkpoint_manager, runner_state, network, env, env_params = init_checkpointer(
        cfg, train_env_params=train_env_params, val_env_params=val_env_params
    )
    if checkpoint_manager.latest_step() is None:
        progress_csv_path = os.path.join(cfg._log_dir_rl, "progress.csv")
        assert not os.path.exists(progress_csv_path), "Progress csv already exists, but have no checkpoint to restore " +\
            "from. Run with `overwrite=True` to delete the progress csv."
        # Create csv for logging progress
        with open(os.path.join(cfg._log_dir_rl, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")
    
    else:
        runner_state = restore_checkpoint(checkpoint_manager, runner_state, cfg)
    
    train = make_train(cfg, runner_state, il_params, checkpoint_manager, train_env_params, val_env_params,
                                   network=network, env=env)
    out = train(rng)


if __name__ == "__main__":
    main()