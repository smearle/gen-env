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
from gen_env.games import GAMES
from gen_env.rules import compile_rule
from purejaxrl.wrappers import LogEnvState
from tensorboardX import SummaryWriter

from gen_env.evo.individual import Individual
from evo_accel import EvoState, apply_evo, distribute_evo_envs_to_train, gen_discount_factors_matrix
from gen_env.configs.config import RLConfig
from gen_env.envs.play_env import GenEnvParams, GenEnvState, PlayEnv
from gen_env.utils import gen_rand_env_params, init_base_env, init_config
from il_player_jax import init_bc_agent
from purejaxrl.experimental.s5.wrappers import LogWrapper
from pcgrl_utils import get_rl_ckpt_dir, get_network
from utils import evaluate_on_env_params, get_rand_train_envs, init_il_config, init_rl_config, load_elite_envs, save_train_env_params
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


def log_callback(metric, steps_prev_complete, cfg: RLConfig, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]
                                    ] * cfg.n_envs
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]

    if len(timesteps) > 0:
        t = timesteps[0]
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()
        print(f"global step={t}; episodic return mean: {ep_return_mean} " + \
            f"max: {ep_return_max}, min: {ep_return_min}")
        ep_length = (metric["returned_episode_lengths"]
                        [metric["returned_episode"]].mean())

        # Add a row to csv with ep_return
        with open(os.path.join(cfg._log_dir_rl,
                                "progress.csv"), "a") as f:
            f.write(f"{t},{ep_return_mean}\n")

        writer.add_scalar("rl/ep_return", ep_return_mean, t)
        writer.add_scalar("rl/ep_return_max", ep_return_max, t)
        writer.add_scalar("rl/ep_return_min", ep_return_min, t)
        writer.add_scalar("rl/ep_length", ep_length, t)
        fps = (t - steps_prev_complete) / (timer() - train_start_time)
        writer.add_scalar("rl/fps", fps, t)

        print(f"fps: {fps}")


def make_train(cfg: RLConfig, init_runner_state: RunnerState, il_params, checkpoint_manager, train_env_params: GenEnvParams,
               val_env_params: GenEnvParams, network, env: LogWrapper):
    cfg.NUM_UPDATES = (
        cfg.total_timesteps // cfg.num_steps // cfg.n_envs
    )
    cfg.MINIBATCH_SIZE = (
        cfg.n_envs * cfg.num_steps // cfg.NUM_MINIBATCHES
    )
    # _, base_env_params = init_base_env(cfg)
    # dummy_env_params = jax.tree.map(lambda x: x[0], train_env_params)
    # env_r, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)
    # env = LogWrapper(env_r)
    # env_r.init_graphics()

    evo_individual = Individual(cfg, env.tiles)
    discount_factors_matrix = gen_discount_factors_matrix(cfg.GAMMA, max_episode_steps=env.max_episode_steps)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (cfg.NUM_MINIBATCHES * cfg.update_epochs))
            / cfg.NUM_UPDATES
        )
        return cfg["LR"] * frac

    def train(rng, cfg: RLConfig, train_env_params: GenEnvParams, val_env_params: GenEnvParams):
        train_start_time = timer()

        # INIT NETWORK
        # network = get_network(env, dummy_env_params, cfg)

        rng, _rng = jax.random.split(rng)
        # init_x = env.gen_dummy_obs(dummy_env_params)
        # init_x = env.observation_space(env_params).sample(_rng)[None]
        # network_params = network.init(_rng, init_x)
        # print(network.subnet.tabulate(_rng, init_x.map, init_x.flat))
        # print(network.subnet.tabulate(_rng, init_x, jnp.zeros((init_x.shape[0], 0))))

        # Create a tensorboard writer
        writer = SummaryWriter(cfg._log_dir_rl)

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
        # reset_rng = jax.random.split(_rng, cfg.n_envs)
        # Apply pmap
        # vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, 0))
        # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))

        # Sample n_envs many random indices between in the range of train_env_params.shape[0] without replacement
        # curr_env_params = get_rand_train_envs(train_env_params, cfg.n_envs, _rng)
        # obsv, env_state = vmap_reset_fn(reset_rng, curr_env_params)
        obsv, env_state = runner_state.last_obs, runner_state.env_state

        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, 0))

        rng, _rng = jax.random.split(rng)

        # Note that there may be duplicates in this first generation due to sampling with replacement above.
        evo_env_params = runner_state.evo_state.env_params
        n_envs = evo_env_params.map.shape[0]

        evo_state = EvoState(env_params=evo_env_params, top_fitness=jnp.zeros(n_envs))

        if cfg.evo_freq != -1:
            # To deal with mismatched shapes after calling _update_step
            train_env_params = evo_state.env_params

        # exp_dir = get_exp_dir(config)

        _log_callback = partial(log_callback, cfg=cfg, writer=writer,
                               train_start_time=train_start_time,
                               steps_prev_complete=steps_prev_complete)
        
        evolved_env_params_dir = os.path.join(cfg._log_dir_rl, "evolved_envs")
        os.makedirs(evolved_env_params_dir, exist_ok=True)
        _save_train_env_params = partial(save_train_env_params, log_dir=evolved_env_params_dir)


        def save_checkpoint(runner_state, info, steps_prev_complete):
            try:
                timesteps = info["timestep"][info["returned_episode"]
                                             ] * cfg.n_envs
            except jax.errors.NonConcreteBooleanIndexError:
                return
            for t in timesteps:
                if t > 0:
                    latest_ckpt_step = checkpoint_manager.latest_step()
                    if (latest_ckpt_step is None or
                            t - latest_ckpt_step >= cfg.ckpt_freq):
                        print(f"Saving checkpoint at step {t}")
                        ckpt = {'runner_state': runner_state,
                                # 'config': OmegaConf.to_container(cfg), 
                                'step_i': t}
                        # ckpt = {'step_i': t}
                        checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))
                    break


        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RunnerState, unused):
                train_state, env_state, evo_state, curr_env_param_idxs, last_obs, rng, update_i = (
                    runner_state.train_state, runner_state.env_state,
                    runner_state.evo_state,
                    runner_state.curr_env_param_idxs,
                    runner_state.last_obs,
                    runner_state.rng, runner_state.update_i,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                # Squash the gpu dimension (network only takes one batch dimension)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, cfg.n_envs)

                # TODO: Move this inside env_step (in case of long rollouts) and make sure no slowdown results.
                # Randomly sample new environments from the training set to be used in case of reset during rollout.
                next_env_params = get_rand_train_envs(train_env_params, cfg.n_envs, rng)
                curr_env_params = jax.tree.map(lambda x: x[curr_env_param_idxs], train_env_params)
                # next_env_params = curr_env_params
                # next_env_params = runner_state.train_env_params

                # Note that we are mapping across environment parameters as well to train on different environments
                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))
                # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
                obsv, env_state, reward, done, info, curr_env_param_idxs = vmap_step_fn(
                    rng_step, env_state, action, curr_env_params, next_env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = runner_state.replace(
                    train_state=train_state, env_state=env_state, last_obs=obsv, rng=rng,
                    update_i=update_i, evo_state=evo_state, curr_env_param_idxs=curr_env_param_idxs)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, evo_state = \
                runner_state.train_state, \
                runner_state.env_state, \
                runner_state.last_obs, runner_state.rng, runner_state.evo_state
            
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + cfg.GAMMA * \
                        next_value * (1 - done) - value
                    gae = (
                        delta
                        + cfg.GAMMA * cfg.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        # obs = traj_batch.obs[None]
                        pi, value = network.apply(params, traj_batch.obs)
                        # action = traj_batch.action.reshape(pi.logits.shape[:-1])
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-cfg.CLIP_EPS, cfg.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses,
                                              value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - cfg.CLIP_EPS,
                                1.0 + cfg.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + cfg.VF_COEF * value_loss
                            - cfg.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    # jax.debug.print("total_loss={total_loss}", total_loss=total_loss)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = \
                    update_state
                rng, _rng = jax.random.split(rng)
                batch_size = cfg.MINIBATCH_SIZE * cfg.NUM_MINIBATCHES
                assert (
                    batch_size == cfg.num_steps * cfg.n_envs
                ), "batch size must be equal to number of steps * number " + \
                    "of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [cfg.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch,
                                advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, cfg.update_epochs
            )
            train_state = update_state[0]
            
            # take the elite params, mutate some new offspring, evaluate 
            # everything, and keep the best
            # NOTE: If you vmap the train function, both of these branches will (most probably)
            # be evaluated each time.                  
            if cfg.evo_freq != -1:
                do_evo = runner_state.update_i % cfg.evo_freq == 0
                def apply_multiple_evo(evo_state):
                    def evo_step(i, evo_state):
                        return apply_evo(
                            rng=rng, env=env, ind=evo_individual, evo_state=evo_state,
                            network_params=network_params, network=network,
                            cfg=cfg, discount_factor_matrix=discount_factors_matrix
                        )
                    
                    return jax.lax.fori_loop(0, cfg.n_evo_gens, evo_step, evo_state)

                evo_state: EvoState = jax.lax.cond(
                    do_evo,
                    lambda: apply_multiple_evo(evo_state),
                    lambda: evo_state
                )

                next_train_env_params = evo_state.env_params

                # do_save = runner_state.update_i % cfg.evo_save_freq == 0
                # jax.lax.cond(
                #     do_save,
                #     lambda: jax.debug.callback(_save_train_env_params, evo_state.env_params, runner_state.update_i),
                #     lambda: None,
                # )
            else:
                # If we're not evolving the envs, just keep the same training set.
                next_train_env_params = train_env_params


            if cfg.val_freq != -1:
                do_eval = runner_state.update_i % cfg.val_freq == 0
                jax.lax.cond(do_eval,
                             lambda: evaluate_on_env_params(rng, cfg, env, val_env_params, network.apply,
                                                            train_state.params, runner_state.update_i,
                                          writer, mode='rl'),
                             lambda: None)

            metric = traj_batch.info
            rng = update_state[-1]

            jax.debug.callback(_log_callback, metric)
            jax.debug.callback(save_checkpoint, runner_state,
                               metric, steps_prev_complete)


            runner_state = runner_state.replace(
                train_state=train_state, env_state=env_state, last_obs=last_obs, rng=rng,
                update_i=runner_state.update_i+1, evo_state=evo_state, train_env_params=next_train_env_params)

            return runner_state, metric

        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, cfg.NUM_UPDATES
        )

        jax.debug.callback(save_checkpoint, runner_state,
                           metric, steps_prev_complete)

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, cfg, train_env_params, val_env_params)


def init_checkpointer(config: RLConfig, train_env_params: GenEnvParams, val_env_params: GenEnvParams):
    # This will not affect training, just for initializing dummy env etc. to load checkpoint.
    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_rl_ckpt_dir(config)
    # Get absolute path
    ckpt_dir = os.path.join(os.getcwd(), ckpt_dir)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    # env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env, env_params = init_base_env(config)
    # env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, _rng = jax.random.split(rng)
    network = get_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    # init_x = env.observation_space(env_params).sample(_rng)[None, ]
    network_params = network.init(_rng, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, 0))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
    curr_env_params = get_rand_train_envs(train_env_params, config.n_envs, _rng)
    curr_env_param_idxs = curr_env_params.env_idx
    obsv, env_state = vmap_reset_fn(
        reset_rng, 
        curr_env_params, 
    )
    n_train_envs = train_env_params.rule_dones.shape[0]
    # evo_env_params = jax.tree.map(lambda x: jnp.array([x for _ in range(n_train_envs)]), env_params)
    # evo_state = EvoState(env_params=train_env_params, top_fitness=jnp.full(config.evo_pop_size, -jnp.inf))
    evo_state = EvoState(env_params=train_env_params, top_fitness=jnp.full(n_train_envs, -jnp.inf))
    runner_state = RunnerState(train_state=train_state, env_state=env_state, last_obs=obsv,
                               train_env_params=train_env_params, val_env_params=val_env_params,
                               # ep_returns=jnp.full(config.num_envs, jnp.nan), 
                               rng=rng, update_i=0, evo_state=evo_state,
                               curr_env_param_idxs=curr_env_param_idxs,
                            )
    options = ocp.CheckpointManagerOptions(
        max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(ckpt_dir), 
        # orbax.checkpoint.PyTreeCheckpointer(), 
        options=options)

    return checkpoint_manager, runner_state, network, env, env_params


def restore_checkpoint(checkpoint_manager, runner_state, config):
    steps_prev_complete = checkpoint_manager.latest_step()
    cfg_dict = config.__dict__
    # for k, v in cfg_dict.items():
    #     cfg_dict[k] = None
    items = {'runner_state': runner_state, 
            #  'config': cfg_dict, 
             'step_i': 0}
    ckpt = checkpoint_manager.restore(steps_prev_complete, args=ocp.args.StandardRestore(items))
    runner_state = ckpt['runner_state']
    return runner_state
    

@hydra.main(version_base=None, config_path='gen_env/configs', config_name='rl')
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

    game_def = GAMES[cfg.game].make_env()
    for rule in game_def.rules:
        rule.n_tile_types = len(game_def.tiles)
        rule = compile_rule(rule)

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


    if cfg.load_gen is None:

        if not cfg.load_game:
            # In this case, we generate random (probably garbage) environments upon which to begin training.
            train_env_params = jax.vmap(gen_rand_env_params, in_axes=(None, 0, None, None))(
                cfg, jax.random.split(rng, cfg.n_envs), env_params, game_def)
            val_env_params = jax.vmap(gen_rand_env_params, in_axes=(None, 0, None, None))(
                cfg, jax.random.split(rng, cfg.n_val_envs), env_params, game_def)
        else:
            # This effectively loads up a sample (with potential repetition) of predefined maps
            train_env_params = get_rand_train_envs(env_params, cfg.n_envs, rng, replace=True)
            val_env_params = get_rand_train_envs(env_params, cfg.n_val_envs, rng, replace=True)
            
    else:
        train_elites, val_elites, test_elites = load_elite_envs(cfg, latest_evo_gen)

        # train_env_params = jax.tree.map(lambda x: x[:cfg.n_envs], train_elites.env_params)
        val_env_params = val_elites.env_params
        train_env_params = train_elites.env_params

        del train_elites, val_elites, test_elites

    if cfg.n_train_envs != -1:
        train_env_params = jax.tree.map(lambda x: x[-cfg.n_train_envs:], train_env_params)
    if cfg.n_val_envs != -1:
        val_env_params = jax.tree.map(lambda x: x[-cfg.n_val_envs:], val_env_params)

    checkpoint_manager, runner_state, network, env, env_params = init_checkpointer(
        cfg, train_env_params=train_env_params, val_env_params=val_env_params
    )
    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
        progress_csv_path = os.path.join(cfg._log_dir_rl, "progress.csv")
        assert not os.path.exists(progress_csv_path), "Progress csv already exists, but have no checkpoint to restore " +\
            "from. Run with `overwrite=True` to delete the progress csv."
        # Create csv for logging progress
        with open(os.path.join(cfg._log_dir_rl, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")
    
    else:
        runner_state = restore_checkpoint(checkpoint_manager, runner_state, cfg)
    
    train_jit = jax.jit(make_train(cfg, runner_state, il_params, checkpoint_manager, train_env_params, val_env_params,
                                   network=network, env=env))
    out = train_jit(rng)


if __name__ == "__main__":
    main()
