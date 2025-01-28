import os

import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from evo_accel import distribute_evo_envs_to_train
from gen_env.configs.config import EnjoyConfig
# from envs.pcgrl_env import PCGRLEnv, render_stats
from gen_env.envs.play_env import GenEnvParams, GenEnvState, PlayEnv
from gen_env.evo.individual import IndividualPlaytraceData
from gen_env.utils import pad_frames


# @hydra.main(version_base=None, config_path='gen_env/configs/', config_name='enjoy_xlife')
# def main_enjoy(cfg: EnjoyConfig):
#     init_config(cfg)
#     latest_gen = init_il_config(cfg)
#     init_rl_config(cfg, latest_gen)

#     train_elites: IndividualPlaytraceData
#     train_elites, val_elites, test_elites = load_elite_envs(cfg, latest_gen)
#     # Select random elites to run inference on
#     idxs = jax.random.permutation(jax.random.PRNGKey(cfg.seed), jnp.arange(train_elites.fitness.shape[0]))[:cfg.n_eps]

#     # We'll render on these different maps/rules
#     env_params_v = jax.tree.map(lambda x: x[idxs], train_elites.env_params)
#     val_params_v = val_elites.env_params

#     # This is just for reference
#     dummy_env_params = jax.tree.map(lambda x: x[0], env_params_v)


#     # Convenienve HACK so that we can render progress without stopping training. Uncomment this or 
#     # set JAX_PLATFORM_NAME=cpu in your terminal environment before running this script to run it on cpu.
#     # WARNING: Be sure to set it back to gpu before training again!
#     # os.system("export JAX_PLATFORM_NAME=cpu")

#     if not cfg.random_agent:
#         checkpoint_manager, restored_ckpt = init_checkpointer(cfg, env_params_v, val_params_v)
#         runner_state: RunnerState = restored_ckpt['runner_state']
#         env_params_v = jax.tree.map(lambda x: x[idxs], runner_state.train_env_params)
#         # env_params_v = jax.tree.map(lambda x: x[:cfg.n_eps], runner_state.evo_state.env_params)
#         network_params = runner_state.train_state.params
#     elif not os.path.exists(cfg._log_dir_rl):
#         os.makedirs(cfg._log_dir_rl)

#     env: PlayEnv
#     env, _ = init_base_env(cfg)
#     # env.prob.init_graphics()
#     network = get_network(env, dummy_env_params, cfg)

def main_enjoy(train_env_params, env, cfg, network, network_params, runner_state):

    rng = jax.random.PRNGKey(cfg.seed)
    n_train_envs = train_env_params.rule_dones.shape[0]
    rng_reset = jax.random.split(rng, cfg.n_eps)

    # This is just for reference
    dummy_env_params = jax.tree.map(lambda x: x[0], train_env_params)

    # frz_map = jnp.zeros(env.map_shape, dtype=jnp.int8)
    # frz_map = frz_map.at[7, 3:-3].set(1)
    # env.queue_frz_map(frz_map)

    # obs, env_state = env.reset(rng, env_params)

    obs, env_state = jax.vmap(env.reset, in_axes=(0, 0))(rng_reset, env_params_v)
    # As above, but explicitly jit

    def step_env(carry, _):
        rng, obs, env_state = carry
        rng, rng_act = jax.random.split(rng)
        if cfg.random_agent:
            action = env.action_space(dummy_env_params).sample(rng_act)
        else:
            # obs = jax.tree_map(lambda x: x[None, ...], obs)
            action = network.apply(network_params, obs)[
                0].sample(seed=rng_act)
        rng_step = jax.random.split(rng, cfg.n_eps)
        # obs, env_state, reward, done, info = env.step(
        #     rng_step, env_state, action[..., 0], env_params
        # )
        env_state: GenEnvState
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))(
            rng_step, env_state, action, env_state.params, env_params_v

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

    # Bring things onto the cpu to make rendering faster
    # jax.device_put(states, jax.devices('cpu'))
    jax.tree.map(lambda x: jax.device_put(x, jax.devices('cpu')[0]), states)

    # for ep_i in range(states.ep_rew.shape[1]):
    #     train_elite_i: IndividualPlaytraceData = jax.tree_map(lambda x: x[ep_i], train_elites)
    #     # print(f'Ep {ep_i}. RL reward: {states.ep_rew[:, ep_i].sum()}. Search reward: {train_elite_i.rew_seq.sum()}')
    #     print(f'Ep {ep_i}. RL reward: {states.ep_rew[:, ep_i].sum()}')

    states = jax.device_put(states, jax.devices('cpu')[0])
    
    print('Rendering gifs:')
    # Since we can't jit our render function (yet)
    frames = []
    for ep_i in range(states.ep_rew.shape[1]):
        ep_frames = []
        for step_i in range(states.ep_rew.shape[0]):
            state_i = jax.tree.map(lambda x: x[step_i, ep_i], states)
            env_params_i = jax.tree.map(lambda x: x[ep_i], env_params_v)
            ep_frames.append(env.render(state_i, env_params_i, mode='rgb_array'))
        frames.append(ep_frames)
        # Print reward
        env_params_i: GenEnvParams
        train_elite_i: IndividualPlaytraceData = jax.tree_map(lambda x: x[ep_i], train_elites)
        print(f'Rendered ep {ep_i}.')

    # frames = np.array(frames)

    # frames = frames.reshape((config.n_eps*env.max_steps, *frames.shape[2:]))

    # assert len(frames) == config.n_eps * env.max_episode_steps, \
    #     "Not enough frames collected"
    # assert frames.shape[0] == cfg.n_eps and frames.shape[1] == env.max_episode_steps, \
    #     "`frames` has wrong shape"


    # Save gifs.
    # for ep_i in range(cfg.n_eps):
        # ep_frames = frames[ep_is*env.max_steps:(ep_is+1)*env.max_steps]
        # ep_frames = frames[ep_i]

        ep_frames = pad_frames(ep_frames)

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



if __name__ == '__main__':
    main_enjoy()
