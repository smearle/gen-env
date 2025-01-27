from timeit import default_timer as timer

import hydra
import jax

from gen_env.utils import init_base_env, init_config
from gen_env.configs.config import EvoConfig
from search_agent import batched_bfs

n_episodes = 10
n_steps = 10000

# def main(exp_id='0', overwrite=False, load=False, multi_proc=False, render=False):
@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def profile(cfg: EvoConfig):
    init_config(cfg)
    env, params = init_base_env(cfg)
    start_time = timer()
    total_reset_time = 0
    key = jax.random.PRNGKey(0)
    for i in range(n_episodes):
        reset_start = timer()
        key = jax.random.split(key)[0]
        obs, state = env.reset(key=key, params=params)
        total_reset_time = timer() - reset_start
        for i in range(n_steps):
            key = jax.random.split(key)[0]
            obs, state, reward, done, info = \
                env.step(key, action=env.action_space.sample(), state=state,
                         params=params, reset_params=params)
            if cfg.render:
                env.render(state=state, params=params)
    done_time = timer()
    n_total_steps = n_episodes * n_steps
    print(f"overall FPS: {n_total_steps / (done_time - start_time)}")
    print(f"reset FPS: {n_episodes / total_reset_time}")
    print(f"step FPS: {n_total_steps / (done_time - start_time - total_reset_time)}")

if __name__ == '__main__':
    # render_sol()
    profile()