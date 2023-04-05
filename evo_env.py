import copy
from enum import unique
import glob
import os
from pdb import set_trace as TT
import random
import shutil
from typing import Iterable

# from fire import Fire
from einops import rearrange
import hydra
import imageio
import numpy as np
# import pool from ray
# from ray.util.multiprocessing import Pool
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
import yaml

from gen_env.configs.config import Config
from gen_env.games import GAMES
from gen_env.envs.play_env import PlayEnv
from gen_env.evo.eval import load_game_to_env, evaluate_multi, evaluate
from gen_env.evo.individual import Individual
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TileSet, TileType
from gen_env.utils import validate_config



# @dataclass
# class Playtrace:
#     obs_seq: List[np.ndarray]
#     action_seq: List[int]
#     reward_seq: List[float]

def collect_elites(cfg: Config):

    # If overwriting, or elites have not previously been aggregated, then collect all unique games.
    # if cfg.overwrite or not os.path.isfile(unique_elites_path):
    # Aggregate all playtraces into one file
    elite_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
    # An elite is a set of game rules, a game map, and a solution/playtrace
    # elite_hashes = set()
    elites = {}
    n_evaluated = 0
    for f in elite_files:
        save_dict = np.load(f, allow_pickle=True)['arr_0'].item()
        elites_i = save_dict['elites']
        for elite in elites_i:
            n_evaluated += 1
            e_hash = elite.hashable()
            if e_hash not in elites or elites[e_hash].fitness < elite.fitness:
                elites[e_hash] = elite
    print(f"Aggregated {len(elites)} unique elites from {n_evaluated} evaluated individuals.")
    # Replay episodes, recording obs and rewards and attaching to individuals
    env = init_base_env(cfg)
    elites = list(elites.values())

    # Replay the episode, storing the obs and action sequences to the elite.
    for elite in elites:
    #     # assert elite.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
        frames = replay_episode(cfg, env, elite)

    # Sort elites by increasing fitness

    if not os.path.isdir(cfg._log_dir_player_common):
        os.mkdir(cfg._log_dir_player_common)

    train_elites, val_elites, test_elites = split_elites(cfg, elites)
    # Save elites to file
    np.savez(os.path.join(cfg._log_dir_common, 'train_elites.npz'), train_elites)
    np.savez(os.path.join(cfg._log_dir_common, 'val_elites.npz'), val_elites)
    np.savez(os.path.join(cfg._log_dir_common, 'test_elites.npz'), test_elites)

    # Save unique elites to npz file
    # If not overwriting, load existing elites
    # else:
    #     # Load elites from file
    #     elites = np.load(unique_elites_path, allow_pickle=True)['arr_0']

    # if not os.path.isdir(cfg._log_dir_player_common):
    #     os.mkdir(cfg._log_dir_playecommonv)

    # Additionally save elites to workspace directory for easy access for imitation learning
    # np.savez(unique_elites_path, elites)

def split_elites(cfg: Config, elites: Iterable[Individual]):
    """ Split elites into train, val and test sets."""
    elites.sort(key=lambda x: x.fitness, reverse=True)

    n_elites = len(elites)
    # n_train = int(n_elites * .8)
    # n_val = int(n_elites * .1)
    # n_test = n_elites - n_train - n_val

    # Sample train/val/test sets from elites with a range of fitness values. Every `n`th elite is sampled.
    # This ensures that the train/val/test sets are diverse. No elites can be in multiple sets.
    train_elites = []
    val_elites = []
    test_elites = []
    for i in range(n_elites):
        if i % 10 == 0:
            val_elites.append(elites[i])
        elif (i + 1) % 10 == 0:
            test_elites.append(elites[i])
        else:
            train_elites.append(elites[i])

    n_train = len(train_elites)
    n_val = len(val_elites)
    n_test = len(test_elites)

    # train_elites = elites[:n_train]
    # val_elites = elites[n_train:n_train+n_val]
    # test_elites = elites[n_train+n_val:]
    print(f"Split {n_elites} elites into {n_train} train, {n_val} val, {n_test} test.")
    return train_elites, val_elites, test_elites

def replay_episode(cfg: Config, env: PlayEnv, elite: Individual):
    """Re-play the episode, recording observations and rewards (for imitation learning)."""
    # print(f"Fitness: {elite.fitness}")
    # print(f"Action sequence: {elite.action_seq}")
    load_game_to_env(env, elite)
    obs_seq = []
    rew_seq = []
    env.queue_games([elite.map.copy()], [elite.rules.copy()])
    obs = env.reset()
    # assert env.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    # Debug: interact after episode completes (investigate why episode ends early)
    # env.render(mode='pygame')
    # while True:
    #     env.tick_human()
    obs_seq.append(obs)
    if cfg.record:
        frames = [env.render(mode='rgb_array')]
    if cfg.render:
        env.render(mode='human')
    done = False
    i = 0
    while not done:
        if i >= len(elite.action_seq):
            # print('Warning: action sequence too short. Ending episode before env is done. This must mean no solution was found.')
            # breakpoint()
            break
        obs, reward, done, info = env.step(elite.action_seq[i])
        obs_seq.append(obs)
        rew_seq = rew_seq + [reward]
        if cfg.record:
            frames.append(env.render(mode='rgb_array'))
        if cfg.render:
            env.render(mode='human')
        i += 1
    if i < len(elite.action_seq):
        raise Exception("Action sequence too long.")
        # print('Warning: action sequence too long.')
        # breakpoint()
    elite.obs_seq = obs_seq
    elite.rew_seq = rew_seq
    if cfg.record:
        return frames


# def main(exp_id='0', overwrite=False, load=False, multi_proc=False, render=False):
@hydra.main(version_base='1.3', config_path="configs", config_name="evo")
def main(cfg: Config):
    validate_config(cfg)
    vid_dir = os.path.join(cfg._log_dir_evo, 'videos')
    
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    overwrite, num_proc, render = cfg.overwrite, cfg.n_proc, cfg.render
    # if cfg.record:
    #     cfg.evaluate=True
    load = not overwrite
    if cfg.collect_elites:
        collect_elites(cfg)
        return
    loaded = False
    if os.path.isdir(cfg._log_dir_evo):
        if load:
            if cfg.load_gen is not None:
                save_file = os.path.join(cfg._log_dir_evo, f'gen-{int(cfg.load_gen)}.npz')
            else:
                # Get `gen-xxx.npz` with largest `xxx`
                save_files = glob.glob(os.path.join(cfg._log_dir_evo, 'gen-*.npz'))
                save_file = max(save_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))

            # HACK to load trained run after refactor
            import evo
            import sys
            individual = evo.individual
            sys.modules['individual'] = individual
            # end HACK

            save_dict = np.load(save_file, allow_pickle=True)['arr_0'].item()
            n_gen = save_dict['n_gen']
            elites = save_dict['elites']
            trg_n_iter = save_dict['trg_n_iter']
            pop_size = len(elites)
            loaded = True
            print(f"Loaded {len(elites)} elites from {save_file} at generation {n_gen}.")
        elif not overwrite:
            print(f"Directory {cfg._log_dir_evo} already exists. Use `--overwrite=True` to overwrite.")
            return
        else:
            shutil.rmtree(cfg._log_dir_il, ignore_errors=True)
    if not loaded:
        pop_size = cfg.batch_size
        trg_n_iter = 10_000 # Max number of iterations while searching for solution. Will increase during evolution
        os.makedirs(cfg._log_dir_evo, exist_ok=True)

    env = init_base_env(cfg)
    env.reset()
    if num_proc > 1:
        envs = [init_base_env(cfg) for _ in range(num_proc)]

    if cfg.evaluate:
        # breakpoint()
        print(f"Elites at generation {n_gen}:")
        eval_elites(cfg, env, elites, n_gen=n_gen, vid_dir=vid_dir)

    def multiproc_eval_offspring(offspring):
        eval_offspring = []
        while len(offspring) > 0:
            eval_offspring += pool.map(evaluate_multi, [(env, ind, render, trg_n_iter) for env, ind in zip(envs, offspring)])
            offspring = offspring[cfg.n_proc:]
        return eval_offspring

    # Initial population
    if not loaded:
        n_gen = 0
        tiles = env.tiles
        rules = env.rules
        map = env.map
        ind = Individual(cfg, tiles, rules, map)
        offspring = []
        for _ in range(pop_size):
            o = copy.deepcopy(ind)
            o.mutate()
            offspring.append(o)
        if num_proc == 1:
            for o in offspring:
                o = evaluate(env, o, render, trg_n_iter)
        else:
            with Pool(processes=num_proc) as pool:
                offspring = multiproc_eval_offspring(offspring)
        elites = offspring

    # Training loop
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=cfg._log_dir_evo)
    for n_gen in range(n_gen, 10000):
        parents = np.random.choice(elites, size=cfg.batch_size, replace=True)
        offspring = []
        for p in parents:
            o: Individual = copy.deepcopy(p)
            o.mutate()
            offspring.append(o)
        if num_proc == 1:
            for o in offspring:
                o = evaluate(env, o, render, trg_n_iter)
        else:
            with Pool(processes=num_proc) as pool:
                offspring = multiproc_eval_offspring(offspring)

        elites = np.concatenate((elites, offspring))
        # Discard the weakest.
        for e in elites:
            if o.fitness is None:
                raise ValueError("Fitness is None.")
        elite_idxs = np.argpartition(np.array([o.fitness for o in elites]), cfg.batch_size)[:cfg.batch_size]
        elites = np.delete(elites, elite_idxs)
        fits = [e.fitness for e in elites]
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
        print()
        # Increment trg_n_iter if the best fitness is within 10 of it.
        # if max_fit > trg_n_iter - 10:
        if max_fit > trg_n_iter * 0.5:
            trg_n_iter *= 2
            # trg_n_iter += 1000
        if n_gen % cfg.save_freq == 0: 
            # Save the elites.
            np.savez(os.path.join(cfg._log_dir_evo, f"gen-{n_gen}"),
            # np.savez(os.path.join(log_dir, "elites"), 
                {
                    'n_gen': n_gen,
                    'elites': elites,
                    'trg_n_iter': trg_n_iter
                })
            # Save the elite's game mechanics to a yaml
            elite_games_dir = os.path.join(cfg._log_dir_evo, "elite_games")
            if not os.path.isdir(elite_games_dir):
                os.mkdir(os.path.join(cfg._log_dir_evo, "elite_games"))
            for i, e in enumerate(elites):
                e.save(os.path.join(elite_games_dir, f"{i}.yaml"))

        if n_gen % cfg.eval_freq == 0:
            eval_elites(cfg, env, elites, n_gen=n_gen, vid_dir=vid_dir)

def eval_elites(cfg: Config, env: PlayEnv, elites: Iterable[Individual], n_gen: int, vid_dir: str):
    # Evaluate elites
    for e_idx, e in enumerate(elites[:10]):
        frames = replay_episode(cfg, env, e)
        if cfg.record:
            # imageio.mimsave(os.path.join(log_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.gif"), frames, fps=10)
            # Save as mp4
            imageio.mimsave(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.mp4"), frames, fps=10)
            # Save elite as yaml
            e.save(os.path.join(vid_dir, f"gen-{n_gen}_elite-{e_idx}_fitness-{e.fitness}.yaml"))


if __name__ == '__main__':
    main()
