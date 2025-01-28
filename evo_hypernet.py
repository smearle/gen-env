from functools import partial
import os
import hydra
import jax

from gen_env.configs.config import EvoConfig
from gen_env.envs.play_env import GameDef
from gen_env.games import GAMES
from gen_env.utils import gen_rand_env_params, init_base_env, init_config


def step_env_random(carry, _, env):
    key, obs, state, env_params = carry
    key = carry[0]
    key, _ = jax.random.split(key)
    action = jax.random.randint(key, 1, 0, 6)
    obs, state, reward, done, info, env_params_idx = env.step(key, state, action, env_params, env_params) 
    return (key, obs, state, env_params), state

def eval_random_ep(key, params, env):
    _step_env_random = partial(step_env_random, env=env)
    obs, state = env.reset(key, params) 
    (_, _, _, _), states = jax.lax.scan(_step_env_random, (key, obs, state, params), None, env.max_episode_steps)
    return states

def eval_random(key, params, env, n_eps=100):
    key = jax.random.split(key, n_eps)
    return jax.vmap(
        eval_random_ep,
        in_axes=(0, None, None)
    )(
        key, params, env
    )


def evaluate_solution(key, params, env) -> Tuple[float, float]:
    """
    Evaluate the genotype in the environment.
    Return (fitness, descriptor).
    """
    states = eval_random(key, params, env, n_eps=5)
    breakpoint()
    fitness = 0

    return fitness


@hydra.main(version_base=None, config_path='gen_env/configs', config_name='me')
def main(cfg: EvoConfig):
    init_config(cfg)
    key = jax.random.PRNGKey(0)
    save_dir = (os.path.join(cfg._log_dir_common, 'hypernet'))
    archive_path = os.path.join(save_dir, 'archive.npz')

    # 1) Initialize environment
    env, base_params = init_base_env(cfg)
    game_def: GameDef = GAMES[cfg.game].make_env()

    gen_key = jax.random.split(key, cfg.evo_pop_size)
    rand_params = jax.vmap(gen_rand_env_params, in_axes=(None, 0, None, None))(
        cfg, gen_key, base_params, game_def,
    )
    breakpoint()

if __name__ == '__main__':
    main()