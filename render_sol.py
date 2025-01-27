from timeit import default_timer as timer

import hydra
import imageio
import jax

from gen_env.utils import init_base_env, pad_frames, save_video
from gen_env.configs.config import EvoConfig
from search_agent import bfs, batched_bfs


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def render_sol(cfg: EvoConfig):
    key = jax.random.PRNGKey(1)
    env, params = init_base_env(cfg)
    best_state_actions = None
    best_reward = 0
    # while best_reward == 0:
    env, params = init_base_env(cfg)
    obs, state = env.reset_env(key=key, params=params)
    best_state_actions, best_reward, n_iter_best, n_iter = \
        bfs(env, state, params=params, max_steps=50_000)
    print(f"Found best solution after {n_iter_best} iterations with {best_reward} reward. Searched for {n_iter} iterations total.")
    # Render the solution
    frames = []
    if best_state_actions is not None:
        (final_state, action_seq) = best_state_actions[0]
        # env.set_state(state)
        frame = env.render(mode='rgb_array', state=state, params=params)
        frames.append(frame)
        for action in action_seq:
            obs, state, reward, done, info = \
                env.step_env(key=key, action=action, state=state, params=params)
            frame = env.render(mode='rgb_array', state=state, params=params)
            frames.append(frame)
    
    # Save the solution as a video
    frames = pad_frames(frames)
    # save_video(frames, f"sol_{cfg.game}.mp4", fps=10)
    gif_name = f"sol_{cfg.game}.gif"
    imageio.v3.imwrite(
        gif_name,
        frames,
        duration=100,
        loop=0,
    )


if __name__ == '__main__':
    render_sol()