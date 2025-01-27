import argparse
from enum import Enum

import hydra
import jax
import numpy as np
import pygame

from gen_env.configs.config import EvoConfig
from gen_env.evo.individual import IndividualData
from gen_env.utils import init_base_env
from gen_env.envs.play_env import PlayEnv, Rule, TileType
from gen_env.games import *


@hydra.main(config_path="gen_env/configs", config_name="human")
def main(cfg: EvoConfig):
    game, map_shape = cfg.game, cfg.map_shape
    cfg.game = cfg.game
    if isinstance(game, str):
        if cfg.load_game is not None:
            # Then it's a filepath, so we are loading up an evolved game, saved to a yaml.
            fname = cfg.load_game
            env, params = init_base_env(cfg)
            individual = IndividualData.load(fname, cfg)
            params = individual.env_params
        else:
            env, params = init_base_env(cfg)
            # game_file = globals()[game]
            # game_def = game_file.make_env()
            # for rule in game_def.rules:
            #     rule.n_tile_types = len(game_def.tiles)
            #     rule.compile()
            # env = PlayEnv(
            #     cfg=cfg, height=map_shape[0], width=map_shape[1], 
            # )
            # [rule.compile() for rule in game_def.rules]
            # env: PlayEnv = game.make_env(height, width)
    else:
        env: PlayEnv = game.make_env()

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key=key, params=params)
    print(f'player pos {state.player_pos}')
    env.render(mode='pygame', state=state, params=params)
    done = False

    running = True

    while running:
        # env.step(env.action_space.sample())
        # env.render()
        key, subkey = jax.random.split(key)
        state = env.tick_human(key=subkey, state=state, params=params)

    pygame.quit()

if __name__ == "__main__":
    main()