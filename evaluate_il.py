
from functools import partial
import json
import os
import shutil

import hydra
import jax
import numpy as np

from evaluate import eval_nn, eval_nn, render_nn
from gen_env.configs.config import ILConfig
from gen_env.evo.individual import IndividualPlaytraceData
from gen_env.utils import init_base_env, init_config
from il_player_jax import init_bc_agent
from utils import init_il_config


@hydra.main(config_path="gen_env/configs", config_name="il")
def eval_il(cfg: ILConfig, render=False):
    init_config(cfg)
    latest_gen = init_il_config(cfg)
    assert latest_gen is not None, \
            "Must select an evo-gen from which to load playtraces for imitation learning when running IL script." +\
            "Set `load_gen=-1` to load latest generation for which playtraces have been aggregated."

    # Environment class doesn't matter and will be overwritten when we load in an individual.
    # env = maze.make_env(10, 10)
    env, env_params = init_base_env(cfg)
    # rng = np.random.default_rng(cfg.env_exp_id)
    rng, train_state, t, checkpoint_manager = init_bc_agent(cfg, env)
    apply_fn = train_state.apply_fn
    network_params = train_state.params

    if not render:
        eval_nn(cfg, latest_gen=latest_gen, env=env, apply_fn=apply_fn, network_params=network_params,
                        algo='il')
    else:
        render_nn(cfg, latest_gen=latest_gen, env=env, apply_fn=apply_fn, network_params=network_params,
                        algo='il')

        
if __name__ == '__main__':
    eval_il()