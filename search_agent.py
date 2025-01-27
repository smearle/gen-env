import copy
import dataclasses
from functools import partial
from math import inf
from pdb import set_trace as TT
import random
from typing import Iterable

import chex
from fire import Fire
from flax import struct
import hydra
import gym
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.games import (hamilton, maze, maze_backtracker, maze_npc, power_line, sokoban)
from gen_env.envs.play_env import GenEnvParams, GenEnvState, PlayEnv, Rule, TileType
from gen_env.utils import init_base_env
from utils import stack_leaves


# For debugging. Render every environment state that is visited during search.
RENDER = False

FRONTIER_VMAP_SIZE = 1_000

@struct.dataclass
class SearchNode:
    state: GenEnvState
    action_seq: Iterable[int]
    done: bool


@partial(jax.jit, static_argnums=(0,))
def expand_frontier(env: PlayEnv, next_frontier: SearchNode, params: GenEnvParams,
                    possible_actions: jnp.ndarray):

    def apply_action(key, state, action, params, action_seq):
        state, obs, s_rew, done, info = env.step_env(key=key, action=action, state=state, params=params)
        action_seq = action_seq.at[state.n_step-1].set(action)
        return SearchNode(state, action_seq, done)

    def expand_frontier_node(key, state, action_seq):
        step_key = jax.random.split(key, possible_actions.shape[0])
        nodes = jax.vmap(
            apply_action, 
            in_axes=(None, None, 0, None, None)
        )(
            step_key,
            state,
            possible_actions,
            params,
            action_seq,
        )
        return nodes

    frontier = jax.vmap(
        expand_frontier_node, 
        in_axes=(None, 0, 0)
    )(
        jax.random.PRNGKey(0),
        next_frontier.state,
        next_frontier.action_seq
    )

    # Flatten the frontier/possible-actions dimensions so the first dimension corresponds to newly generated states
    frontier = jax.tree_map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), frontier)

    return frontier


def batched_bfs(env: PlayEnv, state: GenEnvState, params: GenEnvParams,
          max_steps: int = inf, render: bool = RENDER, max_episode_steps: int = 100):

    key = jax.random.PRNGKey(0)
    action_seq = jnp.empty(max_episode_steps, dtype=jnp.int32)
    frontier = SearchNode(state, action_seq, 0)
    # Add a batch dimension to the frontier
    frontier = jax.tree_map(lambda x: jnp.expand_dims(x, 0), frontier)

    queued_frontier = frontier # HACK to give it some initial value. We will end up searching this state twice.
                               # How can we initialize it with dimension 0 of size 0?
    # queued_frontier = 

    frontier_size = 1
    next_frontier = frontier

    visited = {}
    best_state_actions = None
    best_reward = -inf
    n_iter = 0
    n_iter_best = 0

    possible_actions = jnp.array(list(range(env.action_space.n)), dtype=jnp.int32)
    n_actions = possible_actions.shape[0]

    while n_iter == 0 or next_frontier.state.ep_rew.shape[0] > 0:

        # Expand the frontier
        # print(f'pre-expand frontier')
        frontier = expand_frontier(env, next_frontier, params, possible_actions)
        # print(f'post-expand frontier')

        # Filter out any states that are already in the visited dict
        idxs_to_keep = []
        idxs_to_delete = []

        for i in range(frontier.done.shape[0]):
            # Can't have expanded more states than this
            if i == frontier_size * n_actions:
                idxs_to_delete += list(range(i, frontier.done.shape[0]))
                break
            state = jax.tree_map(lambda x: x[i], frontier.state)
            hashed_state = hash(env, state)
            # print(f'hashed state {hashed_state}')
            if hashed_state not in visited or frontier.state.ep_rew[i] > visited[hashed_state]:
                if not frontier.done[i]:
                    idxs_to_keep.append(i)
                else:
                    idxs_to_delete.append(i)
                    # print(f'Deleting state because done')
                visited[hashed_state] = frontier.state.ep_rew[i]
                if frontier.state.ep_rew[i] > best_reward:
                    print(f'New best reward {frontier.state.ep_rew[i]}')
                    best_state_actions = (state, frontier.action_seq[i])
                    best_reward = frontier.state.ep_rew[i]
                    n_iter_best = n_iter
            else:
                # print('deleting state because already visited')
                idxs_to_delete.append(i)


        # FIXME:Remove all things from the frontier that we've already visited
        # frontier = jax.tree_map(lambda x: x[jnp.array(idxs_to_keep, dtype=jnp.int32)], frontier) 
        frontier = jax.tree_map(
            lambda x: jnp.delete(x, 
                                 jnp.array(idxs_to_delete, dtype=jnp.int32),
                                 axis=0),
            frontier)

        # Combine with the queued frontier
        total_frontier = jax.tree_map(lambda x, y: jnp.concatenate((x, y)), frontier, queued_frontier)

        # Break if we have nothing left
        if total_frontier.done.shape[0] == 0:
            break

        best_idxs = jnp.argpartition(total_frontier.state.ep_rew, 
                                     -min(total_frontier.done.shape[0], FRONTIER_VMAP_SIZE))
        # Flip the order so the best idxs are first
        best_idxs = best_idxs[::-1]

        frontier_size = min(len(best_idxs), FRONTIER_VMAP_SIZE)

        n_iter += frontier_size
        if n_iter > max_steps:
            break

        # Pad this out with the first idx to be size of FRONTIER_VMAP_SIZE (if 
        # necessary)
        best_idxs = jnp.pad(best_idxs, 
                            (0, max(0, FRONTIER_VMAP_SIZE - frontier_size)),
                            mode='constant', constant_values=best_idxs[0])

        # Exclude idxs beyond FRONTIER_VMAP_SIZE (if necessary)
        best_idxs = best_idxs[:FRONTIER_VMAP_SIZE]

        next_frontier = jax.tree_map(lambda x: x[best_idxs], total_frontier)

        # The new queued_frontier is everything other than the best_idxs
        queued_frontier = jax.tree_map(lambda x: jnp.delete(x, best_idxs, axis=0), total_frontier)
        
        print(f'iter {n_iter}, frontier size {frontier.done.shape} queued frontier size {queued_frontier.done.shape}')


    return best_state_actions, best_reward, n_iter_best, n_iter

    
def bfs(env: PlayEnv, state: GenEnvState, params: GenEnvParams,
          max_steps: int = inf, render: bool = RENDER, max_episode_steps: int = 100, n_best_to_keep: int = 1):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # height, width = env.height, env.width
    # state = env.get_state()
    key = jax.random.PRNGKey(0)
    frontier = [(state, [], 0)]
    visited = {hash(env, state): -inf}
    # visited = {env.player_pos: state}
    # visited = {type(env).hashable(state): state}
    # visited_0 = [state]
    best_state_actionss = [None] * n_best_to_keep
    best_rewards = [-inf] * n_best_to_keep
    n_iter_bests = [0] * n_best_to_keep
    n_iter = 0

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = list(range(env.action_space.n))
    else:
        raise NotImplementedError

    while len(frontier) > 0:
        n_iter += 1
        if n_iter > max_steps:
            break
        # parent_state, parent_action_seq, parent_rew = frontier.pop(0)
        # Find the idx of the best state in the frontier
        best_idx = np.argmax([f[2] for f in frontier])
        parent_state, parent_action_seq, parent_rew = frontier.pop(best_idx)
        
        # FIXME: Redundant, remove me
        # env.set_state(parent_state)

        # visited[env.player_pos] = env.get_state()
        # if type(env).hashable(parent_state) in visited:
            # continue
        # visited[hash(env, parent_state)] = parent_rew
        # print(visited.keys())
        random.shuffle(possible_actions)
        for action in possible_actions:
            # env.set_state(parent_state)
            # print('set frontier state')
            obs, state, rew, done, info = \
                env.step_env(key=key, action=action, state=parent_state, params=params)
            child_rew = state.ep_rew.item()
            if render:
                env.render()
            # print(f'action: {action}')

            # FIXME: Redundant, remove me

            # map_arr = state['map_arr']
            action_seq = parent_action_seq + [action]
            # if env.player_pos in visited:
            hashed_state = hash(env, state)
            # if hashed_state in visited and child_rew > visited[hashed_state]:
            #     breakpoint()
            if (hashed_state in visited) and (child_rew <= visited[hashed_state]):
                # print(f'already visited {hashed_state}')
                continue
            # visited[env.player_pos] = state
            # visited[tuple(action_seq)] = state
            visited[hashed_state] = child_rew
            # print(len(visited))
            # visited_0.append(state)
            # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
            # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                # TT()
            for i in range(n_best_to_keep):
                if child_rew > best_rewards[i]:
                    best_state_actionss = best_state_actionss[:i] + [(state, action_seq)] + best_state_actionss[i+1:]
                    best_rewards = best_rewards[:i] + [child_rew] + best_rewards[i+1:]
                    n_iter_bests = n_iter_bests[:i] + [n_iter] + n_iter_bests[i+1:]
                    # print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
                    break
            if not jnp.all(done):
                # Add this state to the frontier so can we can continue searching from it later
                frontier.append((state, action_seq, child_rew))

    return best_state_actionss, best_rewards, n_iter_bests, n_iter

    
@struct.dataclass
class FrontierState:
    params: GenEnvParams
    env_state: GenEnvState
    action_seq: chex.Array
    reward: int
    done: chex.Array


@partial(jax.jit, static_argnames=('env'))
def expand_frontier_action(env: PlayEnv, frontier: FrontierState, params: GenEnvParams, action: int):
    print('tracing expand_frontier_action')
    key = jax.random.PRNGKey(0)
    parent_state = frontier.env_state
    obs, state, rew, done, info = \
        env.step_env(key=key, action=action, state=parent_state, params=frontier.params)
    child_rew = state.ep_rew
    action_seq = frontier.action_seq.at[state.n_step-1].set(action)
    next_frontier = FrontierState(
        params=params,
        env_state=state,
        action_seq=action_seq,
        reward=child_rew,
        done=done,
    )
    return next_frontier

def bfs_multi_env(env: PlayEnv, state: GenEnvState, params: GenEnvParams,
          max_steps: int = inf, max_episode_steps: int = 100, n_best_to_keep: int = 1):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # print('Tracing bfs_multi_env')
    n_envs = int(params.map.shape[0])
    key = jax.random.PRNGKey(0)
    state_is = [jax.tree.map(lambda x: x[i], state) for i in range(n_envs)]
    params_is = [jax.tree.map(lambda x: x[i], params) for i in range(n_envs)]
    frontier_lst = [[FrontierState(
        params=params_i,
        env_state=state_i,
        # action_seq=jnp.full((n_envs, max_episode_steps), fill_value=-1),
        action_seq=jnp.full((max_episode_steps,), fill_value=-1),
        # reward=jnp.empty(n_envs),
        # done = jnp.zeros(n_envs, dtype=jnp.bool),
        done = False,
        reward=0,
    )] for params_i, state_i in zip(params_is, state_is)]
    visited = [{hash(env, jax.tree.map(lambda x: x[i], state)): -inf} for i in range(n_envs)]
    # best_action_seqs = [[None] * n_best_to_keep] * n_envs
    best_action_seqs = jnp.full((n_envs, n_best_to_keep, max_episode_steps), fill_value=-1)
    best_rewards = jnp.full((n_envs, n_best_to_keep), fill_value=-jnp.inf)
    n_iter_bests = jnp.zeros((n_envs, n_best_to_keep))
    n_iter = 0

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = np.array(list(range(env.action_space.n)))
    else:
        raise NotImplementedError


    def expand_frontier_env(frontier: FrontierState, params: GenEnvParams):
        # vmapping over actions
        f = jax.vmap(expand_frontier_action, in_axes=(None, None, None, 0))(env, frontier, params, possible_actions)
        return f

    def expand_frontier(frontier: FrontierState, params: GenEnvParams):
        # vmapping over envs
        return jax.vmap(expand_frontier_env, in_axes=(0, 0))(frontier, params)

    _expand_frontier = jax.jit(
        partial(expand_frontier, params=params)
    )

    # while jnp.any(frontier) > 0:
    # while len(frontier_lst) > 0:
    while n_iter < max_steps:
        print('expanding')
        n_iter += 1
        print(n_iter)
        # if n_iter > max_steps:
        #     break
        # Find the idx of the best state in the frontier
        best_idxs = [jnp.argmax(jnp.array([f.reward for f in env_frontier])) for env_frontier in frontier_lst]

        frontier = [frontier_lst[i].pop(j) for i, j in enumerate(best_idxs)]
        # Now turn this into a single pytree
        frontier = stack_leaves(frontier)

        # (n_envs, n_expansions/actions, ...)
        next_frontier = _expand_frontier(frontier)

        print('hashing')
        next_frontier_lst = [[] for _ in range(n_envs)]
        for env_i in range(n_envs):
            visited_i = visited[env_i]
            for action_i in range(next_frontier.done.shape[1]):
                frontier_i = jax.tree.map(lambda x: x[env_i, action_i], next_frontier)
                reward_i = frontier_i.reward
                hashed_state = hash(env, frontier_i.env_state)
                # if hashed_state in visited and child_rew > visited[hashed_state]:
                #     breakpoint()
                if (hashed_state in visited_i) and (reward_i <= visited_i[hashed_state]):
                    # print(f'already visited {hashed_state}')
                    continue
                # visited[env.player_pos] = state
                # visited[tuple(action_seq)] = state
                visited_i[hashed_state] = reward_i
                # print(len(visited))
                # visited_0.append(state)
                # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
                # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
                    # TT()
        
                next_frontier_lst[env_i].append(frontier_i)

        print('getting baest') 
        for env_i in range(n_envs):
            frontier_env_i = next_frontier_lst[env_i]
            for frontier_i in frontier_env_i:
                reward_i = frontier_i.reward
                for i in range(n_best_to_keep):
                    if reward_i > best_rewards[env_i][i]:
                        # best_action_seqs[env_i] = best_action_seqs[env_i][:i] + [(frontier_i, frontier_i.action_seq)] + \
                        #     best_action_seqs[env_i][i+1:]
                        new_best_action_seqs = best_action_seqs.at[env_i, i:i+1].set(frontier_i.action_seq)
                        new_best_action_seqs = new_best_action_seqs.at[env_i, i+1:].set(best_action_seqs[env_i, i:-1])
                        best_action_seqs = new_best_action_seqs
                        # best_rewards = best_rewards[env_i][:i] + [reward_i] + best_rewards[env_i][i+1:]
                        new_best_rewards = best_rewards.at[env_i, i:i+1].set(reward_i)
                        new_best_rewards = new_best_rewards.at[env_i, i+1:].set(best_rewards[env_i, i:-1])
                        best_rewards = new_best_rewards
                        # n_iter_bests = n_iter_bests[env_i][:i] + [n_iter] + n_iter_bests[env_i][i+1:]
                        new_n_iter_bests = n_iter_bests.at[env_i, i].set(n_iter)
                        new_n_iter_bests = new_n_iter_bests.at[env_i, i+1:].set(n_iter_bests[env_i, i:-1])
                        n_iter_bests = new_n_iter_bests
                        # print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
                        break
                if not frontier_i.done:
                    # Add this state to the frontier so can we can continue searching from it later
                    frontier_lst[env_i].append(frontier_i)
            

    return best_action_seqs, best_rewards, n_iter_bests, n_iter


# @partial(jax.jit, static_argnames=('env', 'max_steps', 'max_episode_steps', 'n_best_to_keep'))
def bfs_multi_env_nohash(env: PlayEnv, state: GenEnvState, params: GenEnvParams,
          max_steps: int = inf, max_episode_steps: int = 100, n_best_to_keep: int = 1):
    """Apply a search algorithm to find the sequence of player actions leading to the highest possible reward."""
    # print('Tracing bfs_multi_env')
    n_envs = int(params.map.shape[0])
    key = jax.random.PRNGKey(0)
    state_is = [jax.tree.map(lambda x: x[i], state) for i in range(n_envs)]
    params_is = [jax.tree.map(lambda x: x[i], params) for i in range(n_envs)]
    frontier_lst = [[FrontierState(
        params=params_i,
        env_state=state_i,
        # action_seq=jnp.full((n_envs, max_episode_steps), fill_value=-1),
        action_seq=jnp.full((max_episode_steps,), fill_value=-1),
        # reward=jnp.empty(n_envs),
        # done = jnp.zeros(n_envs, dtype=jnp.bool),
        done = False,
        reward=0,
    )] for params_i, state_i in zip(params_is, state_is)]
    visited = [{hash(env, jax.tree.map(lambda x: x[i], state)): -inf} for i in range(n_envs)]
    # best_action_seqs = [[None] * n_best_to_keep] * n_envs
    best_action_seqs = jnp.full((n_envs, n_best_to_keep, max_episode_steps), fill_value=-1)
    best_rewards = jnp.full((n_envs, n_best_to_keep), fill_value=-jnp.inf)
    n_iter_bests = jnp.zeros((n_envs, n_best_to_keep))
    n_iter = 0

    if isinstance(env.action_space, gym.spaces.Discrete):
        possible_actions = np.array(list(range(env.action_space.n)))
    else:
        raise NotImplementedError


    def expand_frontier_env(frontier: FrontierState, params: GenEnvParams):
        # vmapping over actions
        f = jax.vmap(expand_frontier_action, in_axes=(None, None, None, 0))(env, frontier, params, possible_actions)
        return f

    def expand_frontier(frontier: FrontierState, params: GenEnvParams):
        # vmapping over envs
        return jax.vmap(expand_frontier_env, in_axes=(0, 0))(frontier, params)

    _expand_frontier = jax.jit(partial(expand_frontier, params=params))

    # while jnp.any(frontier) > 0:
    # while len(frontier_lst) > 0:
    while n_iter < max_steps:
        n_iter += 1
        # if n_iter > max_steps:
        #     break
        # Find the idx of the best state in the frontier
        best_idxs = [jnp.argmax(jnp.array([f.reward for f in env_frontier])) for env_frontier in frontier_lst]

        frontier = [frontier_lst[i].pop(j) for i, j in enumerate(best_idxs)]
        # Now turn this into a single pytree
        print(len(frontier))
        frontier = stack_leaves(frontier)

        # (n_envs, n_expansions/actions, ...)
        next_frontier = _expand_frontier(frontier)
        breakpoint()

        next_frontier_lst = [[] for _ in range(n_envs)]
        for env_i in range(n_envs):
            visited_i = visited[env_i]
            for action_i in range(next_frontier.done.shape[1]):
                frontier_i = jax.tree.map(lambda x: x[env_i, action_i], next_frontier)
                # reward_i = frontier_i.reward
        #         hashed_state = hash(env, frontier_i.env_state)
        #         # if hashed_state in visited and child_rew > visited[hashed_state]:
        #         #     breakpoint()
        #         if (hashed_state in visited_i) and (reward_i <= visited_i[hashed_state]):
        #             # print(f'already visited {hashed_state}')
        #             continue
        #         # visited[env.player_pos] = state
        #         # visited[tuple(action_seq)] = state
        #         visited_i[hashed_state] = reward_i
        #         # print(len(visited))
        #         # visited_0.append(state)
        #         # print([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()])
        #         # if not np.all([np.any(state['map_arr'] != s['map_arr']) for s in visited.values()]):
        #             # TT()
        
                next_frontier_lst[env_i].append(frontier_i)
        # print('hashed')

        
        for env_i in range(n_envs):
            frontier_env_i = next_frontier_lst[env_i]
            for frontier_i in frontier_env_i:
                reward_i = frontier_i.reward
                for i in range(n_best_to_keep):
                    if reward_i > best_rewards[env_i][i]:
                        # best_action_seqs[env_i] = best_action_seqs[env_i][:i] + [(frontier_i, frontier_i.action_seq)] + \
                        #     best_action_seqs[env_i][i+1:]
                        new_best_action_seqs = best_action_seqs.at[env_i, i:i+1].set(frontier_i.action_seq)
                        new_best_action_seqs = new_best_action_seqs.at[env_i, i+1:].set(best_action_seqs[env_i, i:-1])
                        best_action_seqs = new_best_action_seqs
                        # best_rewards = best_rewards[env_i][:i] + [reward_i] + best_rewards[env_i][i+1:]
                        new_best_rewards = best_rewards.at[env_i, i:i+1].set(reward_i)
                        new_best_rewards = new_best_rewards.at[env_i, i+1:].set(best_rewards[env_i, i:-1])
                        best_rewards = new_best_rewards
                        # n_iter_bests = n_iter_bests[env_i][:i] + [n_iter] + n_iter_bests[env_i][i+1:]
                        new_n_iter_bests = n_iter_bests.at[env_i, i].set(n_iter)
                        new_n_iter_bests = new_n_iter_bests.at[env_i, i+1:].set(n_iter_bests[env_i, i:-1])
                        n_iter_bests = new_n_iter_bests
                        # print(f'found new best: {best_reward} at {n_iter_best} iterations step {state.n_step} action sequence length {len(action_seq)}')
                        break
                if not frontier_i.done:
                    # Add this state to the frontier so can we can continue searching from it later
                    frontier_lst[env_i].append(frontier_i)
            

    return best_action_seqs, best_rewards, n_iter_bests, n_iter


def hash(env: PlayEnv, state):
    return env.hashable(state)



# def main(game=maze, height=10, width=10, render=False):
@hydra.main(version_base="1.3", config_path="gen_env/configs", config_name='evo')
def main(cfg):
    if isinstance(game, str):
        game = globals()[game]

    # env: PlayEnv = game.make_env()
    env, params = init_base_env(cfg)
    obs, state = env.reset()
    while True:
        sol = batched_bfs(env, state=state, params=params, render=False)

if __name__ == "__main__":
    main()
