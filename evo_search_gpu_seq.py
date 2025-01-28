import copy
from enum import unique
import glob
import os
from pdb import set_trace as TT
import pickle
import random
import shutil
import sys
from timeit import default_timer as timer
import traceback
from typing import Iterable, List

import chex
from compute_noop_rewards import compute_noop_rewards
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
from gen_env.envs.play_env import GenEnvParams, PlayEnv
from gen_env.evo.eval import evaluate_multi, evaluate
from gen_env.evo.individual import Individual, IndividualData, IndividualPlaytraceData, hash_individual
from gen_env.rules import compile_rule
from gen_env.utils import gen_rand_env_params, init_base_env, init_config
from gen_env.evo.individual import Individual, IndividualData, hash_individual, hash_env
from utils import concatenate_leaves, pad_frames, stack_leaves



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
    if cfg.load_gen is None:
        load_gen = max(gen_nums)
    else:
        load_gen = cfg.load_gen
    elite_files_gns = [(ef, gn) for gn, ef in zip(gen_nums, elite_files) if gn <= load_gen]
    # An elite is a set of game rules, a game map, and a solution/playtrace
    # elite_hashes = set()
    elites = {}
    n_evaluated = 0
    n_filtered = 0
    for f, gn in elite_files_gns:
        save_dict = np.load(f, allow_pickle=True)['arr_0'].item()
        elites_i = save_dict['elites']
        for elite in elites_i:
            elite: IndividualData
            n_evaluated += 1
            e_hash = hash_env(elite.env_params)
            if elite.fitness[0].item() < 2:
                n_filtered += 1
                continue
            if e_hash not in elites or elites[e_hash].fitness.item() < elite.fitness[0].item():
                # HACK which kind of runs counter to naming
                elite = elite.replace(fitness=jnp.array(elite.fitness[0]))
                action_seq = jnp.pad(jnp.array(elite.action_seq[0]),
                                     (0, max_episode_steps + 1 - len(elite.action_seq[0])),
                                     constant_values=-1)
                elite = elite.replace(action_seq=action_seq)
                elites[e_hash] = elite
    print(f"Aggregated {len(elites)} unique elites from {n_evaluated} evaluated individuals. Filtered {n_filtered} " + \
          f"elites with fitness < 2.")
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
            fitness=elite.fitness,
            action_seq=playtrace.action_seq,
            obs_seq=playtrace.obs_seq,
            rew_seq=playtrace.reward_seq,
            done_seq=playtrace.done_seq,
        )
    
    # _, elites = jax.lax.scan(_replay_episode, None, jnp.arange(len(elites)), length=len(elites))
    # Actually, we can vmap this
    elites_v = elites_v.replace(
        env_params=elites_v.env_params.replace(
            rew_scale=jnp.ones((n_elites,)), rew_bias=jnp.zeros((n_elites,))
        )
    )
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

    playtraces = playtraces.replace(
        env_params=playtraces.env_params.replace(
            rew_scale=1/playtraces.rew_seq.sum(1), rew_bias=jnp.zeros((n_elites,))
        )
    )

    train_elites, val_elites = split_elites(cfg, playtraces)
    print(f"Saving collected elites in {cfg._log_dir_common}")

    # new_elite_sets= compute_noop_rewards(cfg, train_elites, val_elites, test_elites)
    # Save elite files under names above
    with open(os.path.join(cfg._log_dir_common, f"gen-{load_gen}_filtered_train_elites.pkl"), 'wb') as f:
        pickle.dump(train_elites, f)
    with open(os.path.join(cfg._log_dir_common, f"gen-{load_gen}_filtered_val_elites.pkl"), 'wb') as f:
        pickle.dump(val_elites, f)
    
    # Save elites to file
    # np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_train_elites.npz'), train_elites)
    # User pickle instead
    # with open(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_filtered_train_elites.pkl'), 'wb') as f:
    #     pickle.dump(train_elites, f)
    # # np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_val_elites.npz'), val_elites)
    # with open(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_filtered_val_elites.pkl'), 'wb') as f:
    #     pickle.dump(val_elites, f)
    # # np.savez(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_test_elites.npz'), test_elites)
    # with open(os.path.join(cfg._log_dir_common, f'gen-{latest_gen}_filtered_test_elites.pkl'), 'wb') as f:
    #     pickle.dump(test_elites, f)

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
    # test_idxs = jnp.arange(9, n_elites, 10)
    # train_idxs = jnp.array([i for i in range(n_elites) if i not in val_idxs and i not in test_idxs])
    # More efficient:
    train_idxs = jnp.setdiff1d(jnp.arange(n_elites), jnp.concatenate([val_idxs])) 
    
    val_elites = jax.tree.map(lambda x: x[val_idxs], playtraces)
    # test_elites = jax.tree.map(lambda x: x[test_idxs], playtraces)
    train_elites = jax.tree.map(lambda x: x[train_idxs], playtraces)

    n_train = len(train_idxs)
    n_val = len(val_idxs)
    # n_test = len(test_idxs)

    # train_elites = elites[:n_train]
    # val_elites = elites[n_train:n_train+n_val]
    # test_elites = elites[n_train+n_val:]
    print(f"Split {n_elites} elites into {n_train} train, {n_val} val.")
    return train_elites, val_elites


def replay_episode_jax(cfg: EvoConfig, env: PlayEnv, elite: IndividualData, 
                   record: bool = False, best_i: int = 0):
    """Re-play the episode, recording observations and rewards (for imitation learning)."""
    # FIXME: This is super slow! Maybe better to do a scan over max_episode_steps, then slice away invalid moves?

    # print(f"Fitness: {elite.fitness}")
    # action_seq = elite.action_seq[best_i]
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
        done = jax.lax.select(action_seq[i+1] == -1, jnp.array([True]), jnp.array([done]))

        i += 1
        rng, _ = jax.random.split(rng)
        # return (rng, obs, state, reward, done, i)
        return (rng, state, i), (obs, state, reward, done)

    _, (obs_seq, state, rew_seq, done_seq) = jax.lax.scan(step_env, (key, state, jnp.array([0])), None, length=env.max_episode_steps)
    # rng, obs_seq, state, rew_seq, done, i = jax.lax.while_loop(step_while, step_env, (key, obs, state, 0, False, 0))

    init_obs = jax.tree.map(lambda x: x[None], init_obs)
    obs_seq = concatenate_leaves((obs_seq, init_obs))
    rew_seq = concatenate_leaves((rew_seq, jnp.array([0.0])))
    done_seq = concatenate_leaves((done_seq, jnp.array([[False]])))

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


@hydra.main(version_base='1.3', config_path="gen_env/configs", config_name="evo")
def main(cfg: EvoConfig):
    # Try/except to avoid submitit-launcher-plugin swallowing up our error tracebacks.
    try:
        _main(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

def _main(cfg: EvoConfig):
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
        trg_n_iter = 200 # Max number of iterations while searching for solution. Will increase during evolution
        os.makedirs(cfg._log_dir_evo, exist_ok=True)

    env, base_params = init_base_env(cfg)
    env.tiles
    ind = Individual(cfg, env.tiles)
    key = jax.random.PRNGKey(cfg.evo_seed)
    # env_state, obs = env.reset(key=key, params=base_params)

    if cfg.evaluate:
        # breakpoint()
        print(f"Elites at generation {n_gen}:")
        eval_elites(cfg, env, elite_inds, n_gen=n_gen, vid_dir=vid_dir)
        return

    if cfg.render_all:    
        save_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
        for save_file in save_files:
            save_dict = np.load(save_file, allow_pickle=True)['arr_0'].item()
            n_gen = save_dict['n_gen']
            elite_inds = save_dict['elites']
            trg_n_iter = save_dict['trg_n_iter']
            eval_elites(cfg, env, elite_inds, n_gen=n_gen, vid_dir=vid_dir, overwrite=False)
        return

    # Expand rules as given in definition into version including sub-rules (i.e. rotations)
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

        offspring_params = []
        for _ in range(pop_size):
            key, _ = jax.random.split(key)
            o_params = gen_rand_env_params(cfg, key, base_params, game_def)
            offspring_params.append(o_params)

        offspring_inds = []
        for o_params in offspring_params:
            if not cfg.domain_randomize:
                fitnesses, action_seqs = evaluate(key, env, o_params, render, trg_n_iter)
            else:
                fitnesses = jnp.zeros((1,))
                action_seqs = jnp.zeros((1, cfg.max_episode_steps))
            o_ind = IndividualData(env_params=o_params, fitness=fitnesses, action_seq=action_seqs)
            offspring_inds.append(o_ind)

        elite_inds = offspring_inds

    elites = set({hash_env(ind.env_params) for ind in elite_inds})

    # Training loop
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=cfg._log_dir_evo)
    for n_gen in range(n_gen, 10000):
        print(f"{len(elites)} unique elites")
        start_time = timer()
        # parents = np.random.choice(elite_inds, size=cfg.batch_size, replace=True)
        parents = np.random.choice(elite_inds, size=cfg.evo_pop_size, replace=True)
        offspring_inds = []
        for p_ind in parents:
            p_params = p_ind.env_params
            # o: Individual = copy.deepcopy(p)
            key, _ = jax.random.split(key)
            map, rules = ind.mutate(key, p_params.map, p_params.rules, env.tiles)
            o_params = p_params.replace(map=map, rules=rules)
            o_hash = hash_env(o_params)
            # if o_hash in elites:
            #     print('skipping repeat env params')
            #     continue
            elites.add(o_hash)
            if not cfg.domain_randomize:
                fitnesses, action_seqs = evaluate(key, env, o_params, render, trg_n_iter)
            else:
                fitnesses = jnp.zeros((1,))
                action_seqs = jnp.zeros((1, cfg.max_episode_steps))
            o_ind = IndividualData(env_params=o_params, fitness=fitnesses, action_seq=action_seqs)
            offspring_inds.append(o_ind)

        elite_inds = np.concatenate((elite_inds, offspring_inds))
        # Discard the weakest.
        for e in elite_inds:
            if e.fitness[0] is None:
                raise ValueError("Fitness is None.")
        elite_idxs = np.argpartition(np.array([o.fitness[0] for o in elite_inds]), cfg.evo_pop_size)[:cfg.evo_pop_size]
        elite_inds = np.delete(elite_inds, elite_idxs)
        fits = [e.fitness[0] for e in elite_inds]
        max_fit = max(fits)
        mean_fit = np.mean(fits)
        min_fit = min(fits) 
        # Log stats to tensorboard.
        writer.add_scalar('fitness/best', max_fit, n_gen)
        writer.add_scalar('fitness/mean', mean_fit, n_gen)
        writer.add_scalar('fitness/min', min_fit, n_gen)
        # Print stats about elites.
        print(f"Generation {n_gen}")
        print(f"Best fitness: {max_fit}")
        print(f"Average fitness: {mean_fit}")
        print(f"Median fitness: {np.median(fits)}")
        print(f"Worst fitness: {min_fit}")
        print(f"Standard deviation: {np.std(fits)}")
        print(f"FPS: {trg_n_iter * cfg.evo_pop_size / (timer() - start_time)}")
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


def eval_elites(cfg: EvoConfig, env: PlayEnv, elites: Iterable[IndividualData], n_gen: int, vid_dir: str,
                overwrite=True):
    """Render elite environment-solution pairs."""
    # Sort elites by fitness.
    elites = sorted(elites, key=lambda e: e.fitness[0], reverse=True)
    for e_idx, e in enumerate(elites[:10]):
        for best_i in range(len(e.fitness)):
            vid_path = os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_trace-{best_i}_fitness-{e.fitness[best_i]}.mp4")
            if not overwrite and os.path.isfile(vid_path):
                continue
            print(f"Trace {best_i}, actions: {e.action_seq[best_i]}")
            playtraces, frames = replay_episode(cfg, env, e, record=cfg.record, best_i=best_i)
            frames = pad_frames(frames)
            if cfg.record:
                # imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.gif"), frames, fps=10)
                # Save as mp4
                # imageio.mimsave(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
                imageio.mimsave(vid_path, frames, fps=10)
                # Save elite as yaml
                # ind.save(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.yaml"))


if __name__ == '__main__':
    # We use CPU multiprocessing to do search with hash tables (which are not
    # easily implemented in jax), so we need to set the jax platform to CPU
    # (otherwise we'll run into multiprocessing errors).
    # NOTE: Actually, seems like jax uses the GPU anyway :D Let's not touch this for now haha.
    jax_platform_name = os.system(' echo $JAX_PLATFORM_NAME')
    os.system('export JAX_PLATFORM_NAME=cpu')
    main()
    os.system(f'export JAX_PLATFORM_NAME={jax_platform_name}')