import copy
from enum import unique
from functools import partial
import glob
import os
from pdb import set_trace as TT
import pickle
import random
import shutil
from typing import Iterable, List

import chex
from einops import rearrange
from flax import struct
import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np
# import pool from ray
# from ray.util.multiprocessing import Pool
from multiprocessing import Pool
from tensorboardX import SummaryWriter

from gen_env.configs.config import EvoConfig
from gen_env.games import GAMES
from gen_env.envs.play_env import GenEnvParams, GenEnvState, PlayEnv
from gen_env.evo.eval import evaluate_multi, evaluate
from gen_env.evo.individual import Individual, IndividualData, IndividualPlaytraceData, hash_individual
from gen_env.rules import compile_rule
from gen_env.utils import gen_rand_env_params, init_base_env, init_config
from gen_env.evo.individual import Individual, IndividualData, hash_individual
from search_agent import bfs_multi_env
from utils import concatenate_leaves, stack_leaves



@struct.dataclass
class Playtrace:
    obs_seq: List[np.ndarray]
    action_seq: List[int]
    reward_seq: List[float]
    done_seq: chex.Array


def collect_elites(cfg: EvoConfig, max_episode_steps: int):

    # If overwriting, or elites have not previously been aggregated, then collect all unique games.
    # if cfg.overwrite or not os.path.isfile(unique_elites_path):
    # Aggregate all playtraces into one file
    elite_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
    # Get the highest generation number
    gen_nums = [int(f.split('-')[-1].split('.')[0]) for f in elite_files]
    latest_gen = max(gen_nums)
    # An elite is a set of game rules, a game map, and a solution/playtrace
    # elite_hashes = set()
    elites = {}
    n_evaluated = 0
    for f in elite_files:
        save_dict = np.load(f, allow_pickle=True)['arr_0'].item()
        elites_i = save_dict['elites']
        for elite in elites_i:
            elite: IndividualData
            n_evaluated += 1
            e_hash = hash_individual(elite)
            if e_hash not in elites or elites[e_hash].fitnesses.item() < elite.fitness[0].item():
                if not hasattr(elite, 'fitnesses'):
                    breakpoint()

                # HACK which kind of runs counter to naming
                elite = elite.replace(fitnesses=jnp.array(elite.fitness[0]))
                action_seq = jnp.pad(jnp.array(elite.action_seq[0]),
                                     (0, max_episode_steps + 1 - len(elite.action_seq[0])),
                                     constant_values=-1)
                elite = elite.replace(action_seqs=action_seq)

                elites[e_hash] = elite
    print(f"Aggregated {len(elites)} unique elites from {n_evaluated} evaluated individuals.")
    # Replay episodes, recording obs and rewards and attaching to individuals
    env, env_params = init_base_env(cfg)
    elites = list(elites.values())
    n_elites = len(elites)

    elites_v = stack_leaves(elites)

    vid_dir = os.path.join(cfg._log_dir_evo, 'debug_videos')
    os.makedirs(vid_dir, exist_ok=True)
    # Replay the episode, storing the obs and action sequences to the elite.

    # def _replay_episode(carry, i):
    def _replay_episode(elite):
        # elite = elites_jnp[i]
        playtrace, frames = replay_episode_jax(cfg, env, elite, record=False, best_i=0)

        return None, IndividualPlaytraceData(
            env_params=elite.env_params,
            fitness=elite.fitnesses,
            action_seq=playtrace.action_seq,
            obs_seq=playtrace.obs_seq,
            rew_seq=playtrace.reward_seq,
            done_seq=playtrace.done_seq,
        )
    
    # _, elites = jax.lax.scan(_replay_episode, None, jnp.arange(len(elites)), length=len(elites))
    # Actually, we can vmap this
    _, playtraces = jax.vmap(_replay_episode, in_axes=(0))(elites_v)

    # for e_idx, elite in enumerate(elites):
    # #     # assert elite.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    #     playtrace, frames = replay_episode_jax(cfg, env, elite, record=False, best_i=0)
    #     # playtrace, frames = replay_episode(cfg, env, elite, record=False, best_i=0)

    #     elites[e_idx] = IndividualPlaytraceData(
    #         env_params=elite.env_params,
    #         fitness=elite.fitnesses[0],
    #         action_seq=playtrace.action_seq,
    #         obs_seq=playtrace.obs_seq,
    #         rew_seq=playtrace.reward_seq,
    #     )

    #     # Will only have returned frames in case of funky error, for debugging
    #     if frames is not None:
    #         breakpoint()
    #         imageio.mimsave(os.path.join(vid_dir, f"elite-{e_idx}_fitness-{elite.fitnesses[0]}.mp4"), frames, fps=10)
    #         frames_2 = replay_episode(cfg, env, elite, record=False)
    #         imageio.mimsave(os.path.join(vid_dir, f"elite-{e_idx}_fitness-{elite.fitnesses[0]}_take2.mp4"), frames_2, fps=10)
    #         breakpoint()


    # Sort elites by increasing fitness

    if not os.path.isdir(cfg._log_dir_player_common):
        os.mkdir(cfg._log_dir_player_common)

    train_elites, val_elites, test_elites = split_elites(cfg, playtraces)
    # Save elites to file
    # np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_train_elites.npz'), train_elites)
    # User pickle instead
    with open(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_train_elites.pkl'), 'wb') as f:
        pickle.dump(train_elites, f)
    # np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_val_elites.npz'), val_elites)
    with open(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_val_elites.pkl'), 'wb') as f:
        pickle.dump(val_elites, f)
    # np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_test_elites.npz'), test_elites)
    with open(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_test_elites.pkl'), 'wb') as f:
        pickle.dump(test_elites, f)

    # Save unique elites to npz file
    # If not overwriting, load existing elites
    # else:
    #     # Load elites from file
    #     elites = np.load(unique_elites_path, allow_pickle=True)['arr_0']

    # if not os.path.isdir(cfg._log_dir_player_common):
    #     os.mkdir(cfg._log_dir_playecommonv)

    # Additionally save elites to workspace directory for easy access for imitation learning
    # np.savez(unique_elites_path, elites)

def split_elites(cfg: EvoConfig, playtraces: Playtrace):
    """ Split elites into train, val and test sets."""
    # playtraces.sort(key=lambda x: x.fitness, reverse=True)
    # Sort 
    sorted_idxs = jnp.argsort(playtraces.fitness, axis=0)[:, 0]
    playtraces = jax.tree.map(lambda x: x[sorted_idxs], playtraces)

    n_elites = sorted_idxs.shape[0]
    # n_train = int(n_elites * .8)
    # n_val = int(n_elites * .1)
    # n_test = n_elites - n_train - n_val

    # Sample train/val/test sets from elites with a range of fitness values. Every `n`th elite is sampled.
    # This ensures that the train/val/test sets are diverse. No elites can be in multiple sets.
    # train_elites = []
    # val_elites = []
    # test_elites = []
    # for i in range(n_elites):
    #     if i % 10 == 0:
    #         val_elites.append(playtraces[i])
    #     elif (i + 1) % 10 == 0:
    #         test_elites.append(playtraces[i])
    #     else:
    #         train_elites.append(playtraces[i])
    val_idxs = jnp.arange(0, n_elites, 10)
    test_idxs = jnp.arange(9, n_elites, 10)
    # train_idxs = jnp.array([i for i in range(n_elites) if i not in val_idxs and i not in test_idxs])
    # More efficient:
    train_idxs = jnp.setdiff1d(jnp.arange(n_elites), jnp.concatenate([val_idxs, test_idxs])) 
    
    val_elites = jax.tree.map(lambda x: x[val_idxs], playtraces)
    test_elites = jax.tree.map(lambda x: x[test_idxs], playtraces)
    train_elites = jax.tree.map(lambda x: x[train_idxs], playtraces)

    n_train = len(train_idxs)
    n_val = len(val_idxs)
    n_test = len(test_idxs)

    # train_elites = elites[:n_train]
    # val_elites = elites[n_train:n_train+n_val]
    # test_elites = elites[n_train+n_val:]
    print(f"Split {n_elites} elites into {n_train} train, {n_val} val, {n_test} test.")
    return train_elites, val_elites, test_elites


def replay_episode_jax(cfg: EvoConfig, env: PlayEnv, elite: IndividualData, 
                   record: bool = False, best_i: int = 0):
    """Re-play the episode, recording observations and rewards (for imitation learning)."""
    # FIXME: This is super slow! Maybe better to do a scan over max_episode_steps, then slice away invalid moves?

    # print(f"Fitness: {elite.fitness}")
    # action_seq = elite.action_seqs[best_i]
    action_seq = elite.action_seq
    # action_seq_jnp = jnp.array(action_seq)
    params = elite.env_params
    # load_game_to_env(env, elite)
    # env.queue_games([elite.map.copy()], [elite.rules.copy()])
    key = jax.random.PRNGKey(0)
    init_obs, state = env.reset_env(key=key, params=params)
    # print(f"Initial state reward: {state.ep_rew}")
    # assert env.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    # Debug: interact after episode completes (investigate why episode ends early)
    # env.render(mode='pygame')
    # while True:
    #     env.tick_human()
    if record:
        frames = [env.render(mode='rgb_array', state=state, params=params)]
    if cfg.render:
        env.render(mode='human', state=state)
    done = False
    i = 0

    def step_while(carry):
        rng, obs, state, rew, done, i = carry
        return i < len(action_seq)

    def step_env(carry, _):
    # def step_env(carry):
        # rng, obs, state, rew, done, i = carry
        rng, state, i = carry
        action = action_seq[i]
        obs, state, reward, done, info = env.step_env(key, state=state, action=action, params=params)    

        # Put a fake done here in case we have fewer actions than max_episode_steps. For IL dataset, just in case (?)
        done = jax.lax.select(action_seq[i+1] == -1, True, done)

        i += 1
        rng, _ = jax.random.split(rng)
        # return (rng, obs, state, reward, done, i)
        return (rng, state, i), (obs, state, reward, done)

    _, (obs_seq, state, rew_seq, done_seq) = jax.lax.scan(step_env, (key, state, 0), None, length=env.max_episode_steps)
    # rng, obs_seq, state, rew_seq, done, i = jax.lax.while_loop(step_while, step_env, (key, obs, state, 0, False, 0))

    init_obs = jax.tree.map(lambda x: x[None], init_obs)
    obs_seq = concatenate_leaves((obs_seq, init_obs))
    rew_seq = concatenate_leaves((rew_seq, jnp.array([0.0])))
    done_seq = concatenate_leaves((done_seq, jnp.array([False])))

    playtrace = Playtrace(obs_seq=obs_seq, action_seq=action_seq,
                          reward_seq=rew_seq, done_seq=done_seq)
    if record:
        return playtrace, frames
    return playtrace, None


def replay_episode(cfg: EvoConfig, env: PlayEnv, elite: IndividualData, 
                   record: bool = False, best_i: int = 0):
    """Re-play the episode, recording observations and rewards (for imitation learning)."""
    # print(f"Fitness: {elite.fitness}")
    action_seq = elite.action_seq[best_i]
    params = elite.env_params
    # load_game_to_env(env, elite)
    obs_seq = []
    rew_seq = []
    done_seq = []
    # env.queue_games([elite.map.copy()], [elite.rules.copy()])
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key=key, params=params)
    # print(f"Initial state reward: {state.ep_rew}")
    # assert env.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    # Debug: interact after episode completes (investigate why episode ends early)
    # env.render(mode='pygame')
    # while True:
    #     env.tick_human()
    obs_seq.append(obs)
    if record:
        frames = [env.render(mode='rgb_array', state=state, params=params)]
    if cfg.render:
        env.render(mode='human', state=state)
    done = False
    i = 0
    while not done:
        if i >= len(action_seq):
            # FIXME: Problem with player death...?
            print('Warning: action sequence too short. Ending episode before env is done. Probably because of cap on '
                  'search iterations.')
            # breakpoint()
            # if not record:
            #     print('Replaying again, rendering this time')
            #     return replay_episode(cfg, env, elite, record=True)
            break
        obs, state, reward, done, info = env.step_env(key, state=state, action=action_seq[i], params=params)
        done_seq.append(done)
        # print(state.ep_rew)
        obs_seq.append(obs)
        rew_seq = rew_seq + [reward]
        if record:
            frames.append(env.render(mode='rgb_array', state=state, params=params))
        if cfg.render:
            env.render(mode='human', state=state, params=params)
        i += 1
    if i < len(action_seq):
        # FIXME: Problem with player death...?
        # raise Exception("Action sequence too long.")
        print('Warning: action sequence too long.')
        if not record:
            print('Replaying again, rendering this time')
            return replay_episode(cfg, env, elite, record=True, best_i=best_i)
        # breakpoint()
    playtrace = Playtrace(obs_seq=obs_seq, action_seq=action_seq,
                          reward_seq=rew_seq, done_seq=done_seq)
    if record:
        return playtrace, frames
    return playtrace, None

def mutate_action_seq(rng, n_actions, action_seq: chex.Array):
    # Generate new random action sequence
    new_action_seq = jax.random.randint(jax.random.PRNGKey(0), (len(action_seq),), 0, n_actions)
    # Sample probability of masking tiles from a uniform distribution
    mask_prob = jax.random.exponential(rng, shape=(1,))
    mask_vals = jax.random.uniform(rng, shape=(len(action_seq),))
    action_seq = jnp.where(mask_vals < mask_prob, new_action_seq, action_seq)
    return action_seq


def step_evolve_action_seqs(env: PlayEnv, params: GenEnvParams, action_seqs: chex.Array, fits: chex.Array, 
                            pop_size: int):
    # Take top pop_size/2 action sequences
    elite_inds = jnp.argpartition(fits, -pop_size)[-pop_size:]
    elite_action_seqs = action_seqs[elite_inds]
    elite_fits = fits[elite_inds]
    mut_rng = jax.random.split(jax.random.PRNGKey(0), pop_size)
    offspring_action_seqs = jax.vmap(mutate_action_seq, in_axes=(0, None, 0))(
        mut_rng, env.num_actions, elite_action_seqs
    )
    offspring_action_seq_fits = jax.vmap(play_action_seq, in_axes=(None, None, 0))(env, params, offspring_action_seqs)
    all_fits = jnp.concatenate([elite_fits, offspring_action_seq_fits])
    all_action_seqs = jnp.concatenate([elite_action_seqs, offspring_action_seqs])
    return all_action_seqs, all_fits


def play_action_seq(env: PlayEnv, params: GenEnvParams, action_seq: chex.Array):
    rng = jax.random.PRNGKey(0)
    # use vmap
    obs, state = env.reset_env(key=rng, params=params)
    # for i in range(env.max_episode_steps):
    #     action = action_seq[i]
    #     obs, state, reward, done, info = env.step_env(reset_rng, state, action, params)
    #     state: GenEnvState
    def step_env(carry, i):
        rng, state = carry
        action = action_seq[i]
        obs, state, reward, done, info = env.step_env(rng, state, action, params)
        # obs, state, reward, done, info = jax.vmap(env.step_env, in_axes=(None, 0, 0, None))(
        #     rng, state, action, params
        # )
        return (rng, state), reward

    (rng, state), reward = jax.lax.scan(step_env, (rng, state), jnp.arange(env.max_episode_steps))
    return reward.sum()


@struct.dataclass
class ActionSeqEvoState:
    best_rew: int
    best_i: int
    best_action_seq: chex.Array


def evaluate_evo(key, params, trg_n_iter,
                 env: PlayEnv, pop_size):
    action_seqs = jax.random.randint(key, (pop_size * 2, env.max_episode_steps), 0, env.num_actions)
    rews = jax.vmap(play_action_seq, in_axes=(None, None, 0))(env, params, action_seqs)
    _step_evolve_action_seqs = partial(step_evolve_action_seqs, env=env, pop_size=pop_size)
    best_rew = -jnp.inf
    best_i = -1
    evo_state = ActionSeqEvoState(best_rew, best_i, jnp.full(env.max_episode_steps, fill_value=-1))
    def step_eval_evo(carry, i):
        action_seqs, rews, evo_state = carry
        action_seqs, rews = _step_evolve_action_seqs(action_seqs=action_seqs, fits=rews, pop_size=pop_size, params=params)
        gen_best_rew_idx = rews.argmax()
        gen_best_rew = rews[gen_best_rew_idx]
        gen_best_action_seq = action_seqs[gen_best_rew_idx]
        is_new_best = gen_best_rew > evo_state.best_rew
        evo_state = jax.lax.cond(
            is_new_best,
            lambda: ActionSeqEvoState(gen_best_rew, i, gen_best_action_seq),
            lambda: evo_state
        )
        return (action_seqs, rews, evo_state), None

    (action_seqs, rews, evo_state), _ = jax.lax.scan(
        step_eval_evo, (action_seqs, rews, evo_state), jnp.arange(trg_n_iter)
    )

    # Fitness of environment params is how recently evolution found a new best action sequence
    return evo_state

@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def main(cfg: EvoConfig):
    # _step_evolve_action_seqs = partial(step_evolve_action_seqs, env=env, pop_size=cfg.evo_pop_size)

    init_config(cfg)
    vid_dir = os.path.join(cfg._log_dir_evo, 'videos')
    
    overwrite, n_proc, render = cfg.overwrite, cfg.n_proc, cfg.render

    if overwrite:
        # Use input to overwrite
        # ovr_bool = input(f"Directory {cfg._log_dir_evo} already exists. Overwrite? (y/n)")
        # if ovr_bool == 'y':
        shutil.rmtree(cfg._log_dir_evo, ignore_errors=True)
        # else:
            # return

    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    # if cfg.record:
    #     cfg.evaluate=True
    load = not overwrite
    if cfg.collect_elites:
        collect_elites(cfg, max_episode_steps=cfg.max_episode_steps)
        return
    loaded = False
    if os.path.isdir(cfg._log_dir_evo):
        ckpt_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
        if len(ckpt_files) == 0:
            print(f'No checkpoints found in {cfg._log_dir_evo}. Starting from scratch')
        elif load:
            if cfg.load_gen is not None:
                save_file = os.path.join(cfg._log_dir_evo, f'gen-{int(cfg.load_gen)}.npz')
            else:
                # Get `gen-xxx.npz` with largest `xxx`
                save_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
                save_file = max(save_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

            # HACK to load trained run after refactor
            # from gen_env import evo
            # from gen_env import configs
            # from gen_env import tiles, rules
            # import sys
            # individual = evo.individual
            # sys.modules['individual'] = individual
            # sys.modules['evo'] = evo
            # sys.modules['configs'] = configs
            # sys.modules['tiles'] = tiles
            # sys.modules['rules'] = rules
            # end HACK

            save_dict = np.load(save_file, allow_pickle=True)['arr_0'].item()
            n_gen = save_dict['n_gen']
            elite_inds = save_dict['elites']
            trg_n_iter = save_dict['trg_n_iter']
            pop_size = len(elite_inds)
            loaded = True
            print(f"Loaded {len(elite_inds)} elites from {save_file} at generation {n_gen}.")
        elif not overwrite:
            print(f"Directory {cfg._log_dir_evo} already exists. Use `--overwrite=True` to overwrite.")
            return
        else:
            shutil.rmtree(cfg._log_dir_il, ignore_errors=True)
    if not loaded:
        pop_size = cfg.evo_pop_size
        trg_n_iter = 1_000 # Max number of iterations while searching for solution. Will increase during evolution
        os.makedirs(cfg._log_dir_evo, exist_ok=True)

    env, base_params = init_base_env(cfg)
    _evaluate_evo = partial(evaluate_evo, env=env, pop_size=cfg.evo_pop_size)
    env.tiles
    ind = Individual(cfg, env.tiles)
    key = jax.random.PRNGKey(0)
    env_state, obs = env.reset(key=key, params=base_params)
    # if num_proc > 1:
    #     envs, params = zip(*[init_base_env(cfg) for _ in range(num_proc)])
    #     breakpoint()
        # envs = [init_base_env(cfg) for _ in range(num_proc)]

    if cfg.evaluate:
        # breakpoint()
        print(f"Elites at generation {n_gen}:")
        eval_elites(cfg, env, elite_inds, n_gen=n_gen, vid_dir=vid_dir)
        return

    fixed_tile_nums = np.array([t.num if t.num is not None else -1 for t in env.tiles])

    game_def = GAMES[cfg.game].make_env()
    for rule in game_def.rules:
        rule.n_tile_types = len(game_def.tiles)
        rule = compile_rule(rule)

    # Initial population
    if not loaded:
        n_gen = 0
        tiles = env.tiles
        rules = base_params.rules
        # rule_rewards = base_params.rules.reward
        map = base_params.map

        # offspring_params = []
        # for _ in range(pop_size):
        #     key, _ = jax.random.split(key)
        #     o_params = gen_rand_env_params(cfg, key, game_def, rules)
        #     # o_map, o_rules = ind.mutate(key=key, map=map, rules=rules, 
        #     #                         tiles=tiles)
        #     # o_params = base_params.replace(map=o_map, rules=o_rules)
        #     offspring_params.append(o_params)
    
        # Generate offspring params using vmap
        rng_o = jax.random.split(key, pop_size)
        key, _ = jax.random.split(key)
        offspring_params = jax.vmap(gen_rand_env_params, in_axes=(None, 0, None, None))(
            cfg, rng_o, game_def, rules
        )
        # fitnesses, action_seqs = _evaluate_evo(key, env, offspring_params, trg_n_iter, pop_size)
        rng_o = jax.random.split(key, pop_size)
        evo_state = jax.vmap(_evaluate_evo, in_axes=(0, 0, None))(rng_o, offspring_params, trg_n_iter)
        print(f'searched for {trg_n_iter}')
        fitnesses, action_seqs = evo_state.best_i, evo_state.best_action_seq
        offspring_inds = []
        # for o_i, o_params in enumerate(offspring_params):
        #     o_ind = IndividualData(env_params=o_params, fitnesses=fitnesses[o_i], action_seqs=action_seqs[o_i])
        #     offspring_inds.append(o_ind)

        # elite_inds = offspring_inds
        elite_inds = IndividualData(
            env_params=offspring_params,
            fitness=fitnesses,
            action_seq=action_seqs
        )

    # Training loop
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=cfg._log_dir_evo)
    for n_gen in range(n_gen, 10000):
        # parents = np.random.choice(elite_inds, size=cfg.batch_size, replace=True)
        # parents = np.random.choice(elite_inds, size=cfg.evo_pop_size, replace=True)
        # offspring_inds = []
        # if n_proc == 1:
        #     for p_ind in parents:
        #         p_params = p_ind.env_params
        #         # o: Individual = copy.deepcopy(p)
        #         key, _ = jax.random.split(key)
        #         map, rules = ind.mutate(key, p_params.map, p_params.rules, env.tiles)
        #         o_params = p_params.replace(map=map, rules=rules)
        #         fitnesses, action_seqs = evaluate(key, env, o_params, render, trg_n_iter)
        #         o_ind = IndividualData(env_params=o_params, fitnesses=fitnesses, action_seqs=action_seqs)
        #         offspring_inds.append(o_ind)
        maps, rules = jax.vmap(ind.mutate, in_axes=(0, 0, 0, None))(
            jax.random.split(key, pop_size), elite_inds.env_params.map, elite_inds.env_params.rules, env.tiles
        )
        offspring_params = elite_inds.env_params.replace(map=maps, rules=rules)
        evo_state = jax.vmap(_evaluate_evo, in_axes=(0, 0, None))(rng_o, offspring_params, trg_n_iter)
        print(f'searched for {trg_n_iter}')
        fitnesses, action_seqs = evo_state.best_i, evo_state.best_action_seq
        offspring_inds = IndividualData(
            env_params=offspring_params,
            fitness=fitnesses,
            action_seq=action_seqs
        )
        # else:
        #     with Pool(processes=n_proc) as pool:
        #         offspring_inds = multiproc_eval_offspring(p_params, env.tiles)
        # elite_inds = np.concatenate((elite_inds, offspring_inds))
        elite_inds = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), elite_inds, offspring_inds)
        # Discard the weakest.
        # for e in elite_inds:
        #     if e.fitnesses[0] is None:
        #         raise ValueError("Fitness is None.")
        elite_idxs = jnp.argpartition(-elite_inds.fitness, cfg.evo_pop_size)[:cfg.evo_pop_size]
        elite_inds = jax.tree.map(lambda x: x[elite_idxs], elite_inds)
        max_fit = jnp.max(elite_inds.fitness)
        mean_fit = np.mean(elite_inds.fitness)
        min_fit = min(elite_inds.fitness) 
        # Log stats to tensorboard.
        writer.add_scalar('fitness/best', max_fit, n_gen)
        writer.add_scalar('fitness/mean', mean_fit, n_gen)
        writer.add_scalar('fitness/min', min_fit, n_gen)
        # Print stats about elites.
        print(f"Generation {n_gen}")
        print(f"Best fitness: {max_fit}")
        print(f"Average fitness: {mean_fit}")
        print(f"Median fitness: {np.median(elite_inds.fitness)}")
        print(f"Worst fitness: {min_fit}")
        print(f"Standard deviation: {np.std(elite_inds.fitness)}")
        print()
        # Increment trg_n_iter if the best fitness is within 10 of it.
        # if max_fit > trg_n_iter - 10:
        if max_fit > trg_n_iter * 0.5:
            # trg_n_iter *= 2
            trg_n_iter += 200
        if n_gen % cfg.save_freq == 0: 
            # Save the elites.
            np.savez(os.path.join(cfg._log_dir_evo, f"gen-{n_gen}"),
            # np.savez(os.path.join(log_dir, "elites"), 
                {
                    'n_gen': n_gen,
                    'elites': elite_inds,
                    'trg_n_iter': trg_n_iter
                })
            # Save the elite's game mechanics to a yaml
            # elite_games_dir = os.path.join(cfg._log_dir_evo, "elite_games")
            # if not os.path.isdir(elite_games_dir):
            #     os.mkdir(os.path.join(cfg._log_dir_evo, "elite_games"))
            # for i, e in enumerate(elite_inds):
            #     ind.save(os.path.join(elite_games_dir, f"{i}.yaml"))
        if cfg.eval_freq != -1 and n_gen % cfg.eval_freq == 0:
            eval_elites(cfg, env, elite_inds, n_gen=n_gen, vid_dir=vid_dir)


def eval_elites(cfg: EvoConfig, env: PlayEnv, elites: Iterable[IndividualData], n_gen: int, vid_dir: str):
    """ Evaluate elites."""
    # Sort elites by fitness.
    elites = sorted(elites, key=lambda e: e.fitness[0], reverse=True)
    for e_idx, e in enumerate(elites[:10]):
        for best_i in range(len(e.fitness)):
            print(f"Trace {best_i}, actions: {e.action_seq[best_i]}")
            playtraces, frames = replay_episode(cfg, env, e, record=cfg.record, best_i=best_i)
            if cfg.record:
                # imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.gif"), frames, fps=10)
                # Save as mp4
                # imageio.mimsave(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
                imageio.mimsave(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_trace-{best_i}_fitness-{e.fitness[best_i]}.mp4"), frames, fps=10)
                # Save elite as yaml
                # ind.save(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.yaml"))


if __name__ == '__main__':
    # We use CPU multiprocessing to do search with hash tables (which are not
    # easily implemented in jax), so we need to set the jax platform to CPU
    # (otherwise we'll run into multiprocessing errors).
    jax_platform_name = os.system(' echo $JAX_PLATFORM_NAME')
    os.system('export JAX_PLATFORM_NAME=cpu')
    main()
    os.system(f'export JAX_PLATFORM_NAME={jax_platform_name}')
