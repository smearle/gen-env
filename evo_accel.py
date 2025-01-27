'''
get the fitness of the evolved frz map (or other thingys we want to evolve)
'''
from functools import partial
import math
import os
import distrax
from flax import linen as nn
from flax import struct
from typing import Optional
import chex
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.configs.config import RLConfig
from gen_env.envs.play_env import PlayEnv, GenEnvParams
from gen_env.evo.individual import Individual


def fill_row_rolled(i, row, n_rows):
    rolled = jnp.roll(row, shift=i)
    return jnp.where(jnp.arange(n_rows) < i, 0, rolled)


def gen_discount_factors_matrix(gamma, max_episode_steps):
    '''
    Generate a discount factor matrix for each timestep in the episode
    '''
    discount_factors = jnp.power(gamma, jnp.arange(max_episode_steps))
    matrix = jax.vmap(fill_row_rolled, in_axes=(0, None, None))(
        jnp.arange(max_episode_steps), discount_factors, max_episode_steps
    )
    return matrix


def distribute_evo_envs_to_train(cfg: RLConfig, evo_env_params: GenEnvParams):
    n_reps = max(1, cfg.n_envs // cfg.evo_pop_size)
    return jax.tree_map(lambda x: jnp.concatenate([x for _ in range(n_reps)])
                        [:cfg.n_envs], evo_env_params)



def step_env_evo_eval(carry, _, env, cfg, all_env_params, network):
    rng, obs, env_state, curr_env_param_idxs, network_params = carry
    curr_env_params = jax.tree.map(lambda x: x[curr_env_param_idxs], all_env_params)
    rng, _rng = jax.random.split(rng)

    pi: distrax.Categorical
    pi, value = network.apply(network_params, obs)
    action = pi.sample(seed=rng)
    # action_r = jnp.full(action_r.shape, 0) # FIXME dumdum Debugging evo 

    rng_step = jax.random.split(_rng, cfg.n_envs)

    # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
    vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))
    # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
    obs, env_state, reward, done, info, curr_env_param_idxs = vmap_step_fn(
                    rng_step, env_state, action,
                    curr_env_params, curr_env_params)

    return (rng, obs, env_state, curr_env_param_idxs, network_params),\
        (env_state, reward, done, info, value)


@struct.dataclass # need to make a carrier for for the fitness to the tensorboard logging? hmm unnecessary
class EvoState:
    top_fitness: Optional[chex.Array] = None
    env_params: Optional[GenEnvParams] = None


def apply_evo(rng, env: PlayEnv, ind: Individual, evo_state: EvoState, network_params, network,
              cfg: RLConfig, discount_factor_matrix):
    '''
    - copy and mutate the environments
    - get the fitness of the envs
    - rank the envs based on the fitness
    - discard the worst envs and return the best
    '''
    network: nn.Module
    evo_env_params = evo_state.env_params
    rng, _rng = jax.random.split(rng)

    # TODO: In case of very large initial train populations, we may not want to re-evaluate everything here.
    n_envs =  evo_env_params.map.shape[0]

    evo_rng = jax.random.split(_rng, n_envs)
    mutate_fn = jax.vmap(ind.mutate, in_axes=(0, 0, 0, None))

    
    maps, ruless = mutate_fn(evo_rng, evo_env_params.map, evo_env_params.rules, env.tiles)
    new_env_params = evo_env_params.replace(map=maps, rules=ruless)
    all_env_params = jax.tree_map(lambda x, y: jnp.concatenate([x, y],axis=0), evo_env_params, new_env_params)

    # Weird but necessary HACK to make vstacking work below. Should probably not be using vstack then?
    # all_env_params = all_env_params.replace(
    #     rew_bias=jnp.concat(all_env_params.rew_bias),
    #     rew_scale=jnp.concat(all_env_params.rew_scale),
    # )

    n_candidate_envs = all_env_params.map.shape[0]

    n_eps = 1

    _step_env_evo_eval = partial(step_env_evo_eval, env=env, cfg=cfg, all_env_params=all_env_params, network=network)
 
    def eval_params(carry, unused):
        all_env_params, network_params, next_env_idxs = carry
        eval_rng = jax.random.split(rng, cfg.n_envs)

        curr_env_params = jax.tree.map(lambda x: x[next_env_idxs], all_env_params)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
                eval_rng, curr_env_params
        )

        _, (states, rewards, dones, infos, values) = jax.lax.scan(
            _step_env_evo_eval, (rng, obsv, env_state, next_env_idxs, network_params),
            None, n_eps*env.max_episode_steps)

        n_steps = rewards.shape[0]

        # Truncate the discount factor matrix in case the episode terminated 
        # early. Add empty batch dimension for broadcasting.
        discount_mat = discount_factor_matrix[:n_steps][..., None]

        # Tile along new 0th axis
        rewards_mat = jnp.tile(rewards[None], (n_steps, 1, 1))
        discounted_rewards_mat = rewards_mat * discount_mat
        returns = discounted_rewards_mat.sum(axis=1)

        vf_errs = jnp.abs(returns - values)

        # In this case, evolution aims to increase the degree to which the agent overestimates its success. If the agent
        # is frozen, reward should decrease.
        # vf_errs = returns - values

        fits = vf_errs.sum(axis=0)

        next_env_idxs += cfg.n_envs
        
        return (all_env_params, network_params, next_env_idxs), (fits, states)

    
    n_eval_batches = math.ceil(n_candidate_envs / cfg.n_envs)

    # fits, states = eval_params(all_env_params, network_params, n_eps)    
    next_env_idxs = jnp.arange(cfg.n_envs)

    _, (fits, states) = jax.lax.scan(
        eval_params, (all_env_params, network_params, next_env_idxs),
        None, n_eval_batches
    )

    fits = fits.reshape(-1)[:n_candidate_envs]
    fits = fits.reshape((-1, n_candidate_envs)).mean(axis=0)
    # sort the top frz maps based on the fitness
    # Get indices of the top 5 largest elements
    top_indices = jnp.argpartition(-fits, cfg.evo_pop_size)[:n_envs] # We negate arr to get largest elements
    # top = frz_maps[:2 * config.evo_pop_size][top_indices]
    elite_params = jax.tree_map(lambda x: x[top_indices], all_env_params)
    
    top_fitnesses = fits[top_indices]
    # evo_writer = SummaryWriter(os.path.join(get_exp_dir(config), "evo"))
    # jax.debug.breakpoint()
    # evo_writer.add_scalar("fitness", top_fitnesses.mean(0), runner_state.update_i)
    return EvoState(top_fitness=top_fitnesses, env_params=elite_params) # here do I need to init an empty one and evo_state.replace(top_fitness=top_fitnesses, frz_map=top) ?
    # return top
    
    
