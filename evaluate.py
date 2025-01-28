from functools import partial
import json
import os
import pickle

import jax
import numpy as np

from gen_env.configs.config import ILConfig
from gen_env.evo.individual import IndividualPlaytraceData
from utils import load_elite_envs


def step_env_noop(carry, _, env):
    rng = jax.random.PRNGKey(0)  # inconsequential
    # Hardcoded to select a rotation action
    action = env.ROTATE_LEFT_ACTION
    obs, state, env_params = carry
    obs, state, reward, done, info, env_params_idx = env.step(rng, state, action, env_params, env_params) 
    return (obs, state, env_params), reward


def step_env_random(carry, _, env):
    rng = jax.random.PRNGKey(0)  # inconsequential
    # Hardcoded to select a rotation action
    action = env.action_space.sample()
    obs, state, env_params = carry
    obs, state, reward, done, info, env_params_idx = env.step(rng, state, action, env_params, env_params) 
    return (obs, state, env_params), reward


def step_env_nn(carry, _, env, env_params, network_params, apply_fn):
    # Hardcoded to select a rotation action
    obs, state, rng = carry
    rng, _ = jax.random.split(rng)
    obs = jax.tree.map(lambda x: x[None], obs)
    dist, _ = apply_fn(network_params, obs)
    action = dist.sample(seed=rng)
    obs, state, reward, done, info, env_params_idx = env.step(rng, state, action, env_params, env_params) 
    return (obs, state, rng), reward


def step_env_render_nn(carry, _, env, env_params, network_params, apply_fn):
    # Hardcoded to select a rotation action
    obs, state, rng = carry
    rng, _ = jax.random.split(rng)
    obs = jax.tree.map(lambda x: x[None], obs)
    dist, _ = apply_fn(network_params, obs)
    action = dist.sample(seed=rng)
    obs, state, reward, done, info, env_params_idx = env.step(rng, state, action, env_params, env_params) 
    return (obs, state, rng), state


def render_elite_nn(env_params, env, apply_fn, network_params, n_eps=100):
    rng = jax.random.PRNGKey(0)  # we can get away with this here
    _step_env_render_nn = partial(step_env_render_nn, env=env, network_params=network_params, apply_fn=apply_fn, env_params=env_params)
    obs, state = env.reset(rng, env_params) 
    _, states = jax.lax.scan(_step_env_render_nn, (obs, state, rng), None, env.max_episode_steps * n_eps)
    return states

    
def eval_env_nn(env_params, env, apply_fn, network_params, n_eps=100):
    rng = jax.random.PRNGKey(0)  # we can get away with this here
    _step_env_nn = partial(step_env_nn, env=env, network_params=network_params, apply_fn=apply_fn, env_params=env_params)
    obs, state = env.reset(rng, env_params) 
    _, rewards = jax.lax.scan(_step_env_nn, (obs, state, rng), None, env.max_episode_steps * n_eps)
    ep_reward = rewards.sum()
    return ep_reward


def eval_noop(params, env):
    rng = jax.random.PRNGKey(0)  # inconsequential
    _step_env_noop = partial(step_env_noop, env=env)
    obs, state = env.reset(rng, params) 
    _, rewards = jax.lax.scan(_step_env_noop, (obs, state, params), None, env.max_episode_steps)
    ep_reward = rewards.sum()
    return ep_reward


def eval_random(params, env, n_eps=100):
    rng = jax.random.PRNGKey(0)  # inconsequential
    _step_env_random = partial(step_env_random, env=env)
    obs, state = env.reset(rng, params) 
    (_, states, _), rewards = jax.lax.scan(_step_env_random, (obs, state, params), None, env.max_episode_steps * n_eps)
    ep_reward_mean = rewards.mean()
    ep_reward_std = rewards.std()
    ep_reward_max = rewards.max()
    return states, ep_reward_mean, ep_reward_std, ep_reward_max


def eval_nn(cfg: ILConfig, latest_gen: int, env, apply_fn, network_params, algo):
    _eval_elite_nn = partial(eval_env_nn, env=env, apply_fn=apply_fn, network_params={'params': network_params})

    log_dir = getattr(cfg, f'_log_dir_{algo}')

    # Load the transitions from the training set
    train_elites, val_elites = load_elite_envs(cfg._log_dir_common, latest_gen)
    test_elites_path = os.path.join('saves', 'test_envs', 'test_envs.pkl')
    with open(test_elites_path, 'rb') as f:
        test_elites = pickle.load(f)
        # HACK
        test_elites = test_elites.replace(
            env_params=test_elites.env_params.replace(
                env_idx=test_elites.env_params.env_idx.squeeze(1),
                rew_scale=test_elites.env_params.rew_scale.squeeze(1),
                rew_bias=test_elites.env_params.rew_bias.squeeze(1)))

    e_stats = {}
    for name, e in zip(['train', 'val', 'test'], [train_elites, val_elites, test_elites]):
        e: IndividualPlaytraceData
        n_elites = e.env_params.rule_dones.shape[0]
        e_params = e.env_params
        nn_rewards = jax.vmap(_eval_elite_nn, in_axes=(0))(e_params)
        e_stats[f'{name}_returns'] = np.array(nn_rewards).tolist()
        e_stats[f'{name}_mean'] = nn_rewards.mean().item()
        e_stats[f'{name}_std'] = nn_rewards.std().item()
        e_stats[f'{name}_max'] = nn_rewards.max().item()
        e_stats[f'{name}_min'] = nn_rewards.min().item()

    # Save stats to disk as json
    print(f'Saving stats to {log_dir}')
    with open(os.path.join(log_dir, f"nn_stats.json"), 'w') as f:
        json.dump(e_stats, f, indent=4)


def render_nn(cfg: ILConfig, latest_gen: int, env, apply_fn, network_params, algo):
    _render_elite_nn = partial(render_elite_nn, env=env, apply_fn=apply_fn, network_params={'params': network_params})

    log_dir = getattr(cfg, f'_log_dir_{algo}')

    # Load the transitions from the training set
    train_elites, val_elites = load_elite_envs(cfg._log_dir_common, latest_gen)

    e_stats = {}
    for name, e in zip(['train', 'val', 'test'], [train_elites, val_elites, test_elites]):
        e: IndividualPlaytraceData
        n_elites = e.env_params.rule_dones.shape[0]
        e_params = e.env_params
        nn_states = jax.vmap(_render_elite_nn, in_axes=(0))(e_params)

        nn_states_lst = [jax.tree.map(lambda x: x[i], nn_states) for i in range(n_elites)]

        for elite_states in nn_states_lst:
            elite_states_lst = [jax.tree.map(lambda x: x[i], elite_states) for i in range(elite_states.reward.shape[0])]

            for i, state in enumerate(elite_states_lst):
                frame = env.render(state)
                breakpoint()
