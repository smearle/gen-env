from functools import partial
import os
import pickle

from evaluate import eval_noop, eval_random
import hydra
import jax
from jax import numpy as jnp

from gen_env.configs.config import EvoConfig
from gen_env.envs.play_env import PlayEnv
from gen_env.evo.individual import IndividualPlaytraceData
from gen_env.utils import init_base_env, init_config
from utils import load_elite_envs


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def main(cfg: EvoConfig):
    init_config(cfg)
    train_elites, val_elites = load_elite_envs(cfg._log_dir_common, cfg.load_gen)
    return compute_noop_rewards(cfg, train_elites, val_elites)


def compute_noop_rewards(cfg: EvoConfig, train_elites: IndividualPlaytraceData,
                         val_elites: IndividualPlaytraceData):
    env, dummy_params = init_base_env(cfg)
    env: PlayEnv
    _eval_elite_noop = partial(eval_noop, env=env)
    _eval_elite_random = partial(eval_random, env=env)
    rng = jax.random.PRNGKey(cfg.seed)
    new_elite_sets = []
    for e in [train_elites, val_elites]:
        e: IndividualPlaytraceData
        n_elites = e.env_params.rule_dones.shape[0]
        e_params = e.env_params

        # Backward compatibility HACK
        # e_params = e_params.replace(rew_scale=jnp.ones((n_elites,)), rew_bias=jnp.zeros((n_elites,)))
        # TODO: May need to batch this vmapping to prevent OOM
        noop_ep_rewards = jax.vmap(_eval_elite_noop, in_axes=(0))(e_params)
        e = e.replace(env_params=e.env_params.replace(noop_ep_rew=noop_ep_rewards))
        states, random_ep_reward_means, random_ep_reward_stds, random_ep_reward_maxs = \
            jax.vmap(_eval_elite_random, in_axes=(0))(e_params)
        e = e.replace(env_params=e.env_params.replace(random_ep_rew=random_ep_reward_means))
        e = e.replace(env_params=e.env_params.replace(search_ep_rew=e.rew_seq.sum(1)))

        # stacked_ep_rews = jnp.stack([e.env_params.noop_ep_rew, e.env_params.random_ep_rew, e.env_params.search_ep_rew[:, 0]], axis=1)
        stacked_ep_rews = jnp.stack([e.env_params.noop_ep_rew, random_ep_reward_maxs], axis=1)

        best_ep_rew = jnp.max(stacked_ep_rews, axis=1)
        worst_ep_rew = jnp.min(stacked_ep_rews, axis=1)

        reward_bias = -random_ep_reward_means
        e = e.replace(env_params=e.env_params.replace(rew_bias=reward_bias))
        # Note that since we have integer rewards, this will always be a decimal
        # reward_scale = 1 / random_ep_reward_stds
        reward_scale = 1 / (best_ep_rew - worst_ep_rew)
        reward_scale = jnp.where(jnp.isinf(reward_scale), 1, reward_scale)
        e = e.replace(env_params=e.env_params.replace(rew_scale=reward_scale))
        
        new_elite_sets.append(e)

    return new_elite_sets


if __name__ == '__main__':
    main()