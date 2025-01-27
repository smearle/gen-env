"""This is a simple example demonstrating how to clone the behavior of an expert.
Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
from functools import partial
import glob
import json
import os
import shutil

import gym
import hydra
import imageio
import numpy as np
import pandas as pd
import pickle
import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

import imitation
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import TransitionsMinimal
from imitation.data.wrappers import RolloutInfoWrapper

from gen_env.configs.config import EvoConfig
from gen_env.games import maze
from gen_env.envs.play_env import PlayEnv
from gen_env.utils import init_base_env, load_game_to_env, init_evo_config


def evaluate_policy_on_elites(cfg: EvoConfig, policy, env, elites, name):
    vid_dir = os.path.join(cfg._log_dir_il, "videos")
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    net_rew = 0
    for e_idx, elite in enumerate(elites):
        print(f"Evaluating elite {e_idx} of {len(elites)}")
        rew, frames = evaluate_policy(cfg, policy, env, elite)
        net_rew += rew
        if cfg.record:
            # Save frames to video
            imageio.mimsave(os.path.join(vid_dir, f"{name}_elite-{e_idx}.mp4"), frames, fps=10)
    return net_rew / len(elites)

def evaluate_policy(cfg, policy, env: PlayEnv, individual):
    load_game_to_env(env, individual)
    # assert individual.map[4].sum() == 0, "Force tile should not be present in map"
    env.queue_games([individual.map.copy()], [individual.rules.copy()])
    state, obs = env.reset()
    frames = None
    if cfg.record:
        frames = [env.render(mode="rgb_array", state=state)]
    done = False
    total_reward = 0
    while not done:
        action, _ = policy.predict(obs, deterministic=False)
        state, obs, reward, done, info = env.step(action, state)
        if cfg.record:
            frames.append(env.render(mode="rgb_array", state=state))
        total_reward += reward
    # print(f"Reward: {total_reward}")
    return total_reward, frames

# TODO: Should maybe just be save/loading policy instead of entire trainer(?)
def save(cfg: EvoConfig, bc_trainer: bc.BC, batch_i, epoch_i):
    policy_filepath = os.path.join(cfg._log_dir_il, "policy")
    logger_filepath = os.path.join(cfg._log_dir_il, "logger.pkl")

    # Create a backup of the logger
    if os.path.exists(logger_filepath):
        os.rename(logger_filepath, logger_filepath + ".bak")

    # Create a backup of the policy
    if os.path.exists(policy_filepath):
        os.rename(policy_filepath, policy_filepath + ".bak")

    # Save logger
    # with open(logger_filepath, "wb") as f:
    #     pickle.dump(bc_trainer._bc_logger._logger, f)

    # Save transitions and bc_trainer with pickle
    # pickle.dump(bc_trainer, open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "wb"))
    bc_trainer.save_policy(os.path.join(cfg._log_dir_il, "policy"))

    # Save current epoch number and batch number as json
    with open(os.path.join(cfg._log_dir_il, "stats.json"), "w") as f:
        json.dump({
            "epoch_i": epoch_i,
            "batch_i": batch_i,
            "tb_step": bc_trainer._bc_logger._tensorboard_step
        }, f)

    # # Delete backup of the logger
    # if os.path.exists(logger_file_path + ".bak"):
    #     os.remove(logger_file_path + ".bak")

    # # Delete backup of the policy
    # if os.path.exists(policy_filepath + ".bak"):
    #     os.remove(policy_filepath + ".bak")

def load(cfg: EvoConfig, policy_path: str):
    # Load transitions and bc_trainer
    # bc_trainer = pickle.load(open(os.path.join(cfg.log_dir, "bc_trainer.pkl"), "rb"))
    # bc_trainer._bc_logger._logger = pickle.load(open(os.path.join(cfg.log_dir, "logger.pkl"), "rb"))

    # Load logger
    # logger = pickle.load(open(os.path.join(cfg._log_dir_il, "logger.pkl"), "rb"))

    # Load policy
    policy = bc.reconstruct_policy(policy_path)
    # Load stats
    stats = json.load(open(os.path.join(cfg._log_dir_il, "stats.json"), "r"))
    batch_i, epoch_i, tb_i = stats["batch_i"], stats["epoch_i"], stats["tb_step"]
    return policy, batch_i, epoch_i, tb_i


@hydra.main(version_base="1.3", config_path="gen_env/configs", config_name="il")
def main(cfg: EvoConfig):
    init_evo_config(cfg)

    # glob files of form `gen-XX*elites.npz` and get highest gen number
    gen_files = glob.glob(os.path.join(cfg._log_dir_common, "gen-*_elites.npz"))
    gen_nums = [int(os.path.basename(f).split("_")[0].split("-")[1]) for f in gen_files]
    latest_gen = max(gen_nums)

    cfg._log_dir_il += f"_env-evo-gen-{latest_gen}"

    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env = init_base_env(cfg)
    rng = np.random.default_rng(cfg.env_exp_id)

    if cfg.overwrite:
        if os.path.exists(cfg._log_dir_il):
            shutil.rmtree(cfg._log_dir_il)

    if not os.path.exists(cfg._log_dir_il):
        os.makedirs(cfg._log_dir_il)

    # Initialize tensorboard logger
    writer = th.utils.tensorboard.SummaryWriter(cfg._log_dir_il)

    # HACK to load trained run after refactor
    # import sys
    # from gen_env import evo, configs, tiles, rules
    # sys.modules['evo'] = evo
    # sys.modules['configs'] = configs
    # sys.modules['tiles'] = tiles
    # sys.modules['rules'] = rules
    # end HACK

    # elites = np.load(os.path.join(cfg.log_dir_evo, "unique_elites.npz"), allow_pickle=True)['arr_0']
    train_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.npz"), allow_pickle=True)['arr_0']
    val_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.npz"), allow_pickle=True)['arr_0']
    test_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.npz"), allow_pickle=True)['arr_0']

    transitions_path = os.path.join(cfg._log_dir_il, "transitions.npz")

    policy_kwargs = dict(net_arch=[64, 64], observation_space=env.observation_space, action_space=env.action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,)

    # if cfg.overwrite or not os.path.exists(os.path.join(cfg.log_dir_common, "transitions.npz")):
    if True:
        # progress_df = pd.DataFrame()
        # cfg.overwrite = True
        obs = []
        acts = []
        for elite in train_elites:
            # obs_seq = [ob for ob in elite.obs_seq[:-1]]
            obs_seq = elite.obs_seq[:-1]
            if len(obs_seq) != len(elite.action_seq):
                print('Warning: obs_seq and action_seq have different lengths.')
                elite.action_seq = elite.action_seq[:len(obs_seq)]
            assert len(obs_seq) == len(elite.action_seq)
            obs.extend(obs_seq)
            acts.extend(elite.action_seq)
        obs = np.array(obs)
        acts = np.array(acts)
        infos = np.array([{} for _ in range(obs.shape[0])])
        assert obs.shape[0] == acts.shape[0]
        transitions = TransitionsMinimal(obs=obs, acts=acts, infos=infos)
        print(f"Loaded {transitions.obs.shape[0]} transitions from {len(train_elites)} playtraces.")
        # Save the transitions with pickle
        np.savez(transitions_path, {
            'obs': transitions.obs,
            'acts': transitions.acts,
            'infos': transitions.infos,
            })

        # policy=None


    # else:
    #     # p, Nonerogress_df = pd.read_csv(os.path.join(cfg.log_dir, "progress.csv"))
    #     transitions = np.load(os.path.join(cfg.log_dir_common, "transitions.npz"), allow_pickle=True)['arr_0'].item()
    #     transitions = TransitionsMinimal(
    #         obs=transitions['obs'],
    #         acts=transitions['acts'],
    #         infos=transitions['infos'],
    #     )

    policy_path = os.path.join(cfg._log_dir_il, "policy")
    if cfg.overwrite or not os.path.exists(policy_path):
        epoch_i = 0
        batch_i = 0
        policy = MlpPolicy(**policy_kwargs)


    else:
        # FIXME: Not reloading optimizer???
        policy, batch_i, epoch_i, tb_i = load(cfg, policy_path)

    custom_logger = imitation.util.logger.configure(
        os.path.join(cfg._log_dir_il, "logs"),
        ["stdout", "tensorboard"],
    )

    bc_trainer = bc.BC(
        policy=policy,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        custom_logger=custom_logger,
    )

    # HACK
    assert not hasattr(bc_trainer._bc_logger, '_current_batch')
    bc_trainer._bc_logger._current_batch = batch_i
    # Note that `_current_epoch` already exists.
    bc_trainer._bc_logger._current_epoch = epoch_i
    # END HACK

    if cfg.overwrite:
        # save(cfg, bc_trainer, epoch_i)
        save(cfg, bc_trainer, batch_i, epoch_i=epoch_i)
    
    else:
        bc_trainer._bc_logger._tensorboard_step = tb_i
        # bc_trainer._bc_logger._tensorboard_step = curr_batch

    # reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"epoch-{epoch_i}")
    # reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, val_elites[-10:], name=f"batch-{curr_batch}")
    # print(f"Reloaded epoch {epoch_i}.\nReward before imitation learning: {reward}")
    # print(f"Reloaded batch {curr_batch}.\nReward before imitation learning: {reward}")
    # with open(os.path.join(cfg.log_dir, f"epoch-{epoch_i}_reward.txt"), "w") as f:
    # with open(os.path.join(cfg.log_dir_il, f"epoch-{curr_batch}_reward.txt"), "w") as f:
        # f.write(str(reward))


    def on_epoch_end(bc_trainer, cfg, base_n_epoch):
        epoch_i = base_n_epoch + bc_trainer._bc_logger._current_epoch
        save(cfg, bc_trainer, batch_i, epoch_i)

    def on_batch_end(bc_trainer: imitation.algorithms.bc.BC,
                    cfg: EvoConfig, base_n_batch, base_n_epoch, env, val_elites, writer):
        batch_i = base_n_batch + bc_trainer._bc_logger._current_batch
        epoch_i = base_n_epoch + bc_trainer._bc_logger._current_epoch
        if batch_i % (cfg.save_freq) == 0:
            print(f"Saving at batch {batch_i}")
            save(cfg, bc_trainer, batch_i, epoch_i=epoch_i)
            print(f"Saved at batch {batch_i}")
        if batch_i % (cfg.eval_freq) == 0:
            print(f"Evaluating at batch {batch_i}")
            val_reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, np.random.choice(val_elites, 20), 
                                                   name=f"batch-{batch_i}")
            print(f"Val reward at IL batch {batch_i}: {val_reward}")
            # bc_trainer._logger.record("val_reward", val_reward, bc_trainer._bc_logger._current_batch)
            writer.add_scalar("bc2/val_reward", val_reward, batch_i)
        bc_trainer._bc_logger._current_batch += 1

    # on_epoch_end = partial(on_epoch_end, bc_trainer=bc_trainer, cfg=cfg, base_n_epoch=epoch_i)
    on_batch_end = partial(on_batch_end, bc_trainer=bc_trainer, cfg=cfg, base_n_batch=batch_i, base_n_epoch=epoch_i, env=env, val_elites=val_elites, writer=writer)

    print("Training a policy using Behavior Cloning")
    # n_train_epochs = cfg.n_epochs - epoch_i
    n_train_batches = int(cfg.n_il_batches - batch_i)
    if n_train_batches > 0:
        bc_trainer.train(
            n_epochs=None, 
            n_batches=n_train_batches, 
            # on_epoch_end=on_epoch_end, 
            on_batch_end=on_batch_end
        )

        batch_i = batch_i + bc_trainer._bc_logger._current_batch
        epoch_i = epoch_i + bc_trainer._bc_logger._current_epoch
        save(cfg, bc_trainer, batch_i, epoch_i=epoch_i)

        # reward = evaluate_policy_on_elites(cfg, bc_trainer.policy, env, elites[-10:], name=f"batch-{cfg.n_il_batches}")
        # print(f"Reward after imitation learning: {reward}")
        # with open(os.path.join(cfg.log_dir_il, f"batch-{cfg.n_il_batches}_reward.txt"), "w") as f:
        #     f.write(str(reward))
        
        # Access the logged data
        # logger = bc_trainer.logger
        # bc_log_dir = logger.dir
        # Load progress.csv
        # progress_df_new = pd.read_csv(os.path.join(bc_log_dir, "progress.csv"), index_col=False)
        # Add to existing dataframe
        # breakpoint()
        # progress_df = pd.concat([progress_df, progress_df_new], ignore_index=True)

        # Add number of rows in saved df to row indices of new df
        # progress_df_new.index += progress_df.shape[0]

        # Save dataframe
        # progress_df.to_csv(os.path.join(cfg.log_dir, "progress.csv"))

    # policy = bc_trainer.policy

    # Save the state dict
    # th.save(policy.state_dict(), os.path.join(cfg._log_dir_il, "policy_il.pt"))




if __name__ == "__main__":
    main()