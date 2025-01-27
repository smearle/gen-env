from dataclasses import dataclass
import math
import os
from typing import Tuple
import chex

import cv2
from einops import rearrange
import imageio
from jax import numpy as jnp
import jax
import numpy as np

from gen_env.configs.config import EvoConfig, MapElitesConfig
from gen_env.envs.play_env import GameDef, PlayEnv, SB3PlayEnv, GenEnvParams
from gen_env.evo.individual import IndividualData
from gen_env.games import GAMES
from gen_env.rules import RuleData, compile_rule, gen_rand_rule
from utils import stack_leaves


def init_config(cfg: EvoConfig):
    env_exp_name = (f"{cfg.game}_{'mutRule_' if cfg.mutate_rules else ''}{'mutMap_' if cfg.mutate_map else ''}" + 
        f's-{cfg.evo_seed}_' + f"exp-{cfg.env_exp_id}")

    # Get path to parent directory of this file
    grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg._log_dir_common = os.path.join(grandparent_dir, cfg.workspace, env_exp_name)

    player_exp_name = (f"player{'_hideRule' if cfg.hide_rules else ''}")

    cfg._log_dir_player_common = os.path.join(cfg._log_dir_common, player_exp_name)
    cfg._log_dir_rl = os.path.join(cfg._log_dir_player_common, cfg.runs_dir_rl)
    cfg._log_dir_il = os.path.join(cfg._log_dir_player_common, cfg.runs_dir_il)


    # cfg.log_dir_evo = os.path.join(cfg.workspace, cfg.runs_dir_evo, f"exp-{cfg.exp_id}")
    cfg._log_dir_evo = os.path.join(cfg._log_dir_common, cfg.runs_dir_evo)


def init_me_config(cfg: MapElitesConfig):
    cfg._log_dir_me = os.path.join(cfg._log_dir_common, 'me')

    
def pad_frames(frames):
    frame_shapes = [frame.shape for frame in frames]
    max_frame_w, max_frame_h = max(frame_shapes, key=lambda x: x[0])[0], \
        max(frame_shapes, key=lambda x: x[1])[1]
    # Pad frames to be same size
    new_frames = []
    for frame in frames:
        frame = np.pad(frame, ((0, max_frame_w - frame.shape[0]),
                                    (0, max_frame_h - frame.shape[1]),
                                    (0, 0)), constant_values=0)
        new_frames.append(frame)
    frames = new_frames
    return frames


def save_video(frames, video_path, fps=10):
    """Save a list of frames to a video file.
    Args:
        frames (list): list of frames to save
        video_path (str): path to save the video
        fps (int): frame rate of the video
    """
    imageio.mimwrite(video_path, frames, fps=25, quality=8, macro_block_size=1)


def gen_random_multihot_map(key: jax.random.PRNGKey, game_def: GameDef, map_shape):
    """Create random multi-hot map with uniform probability"""
    new_map = jax.random.bernouilli(map_shape)
    return new_map


def gen_random_int_map(key: jax.random.PRNGKey, game_def: GameDef, map_shape):
    """Generate frequency-based tiles with certain probabilities."""
    tile_probs = [tile.prob for tile in game_def.tiles]
    # int_map = np.random.choice(len(game_def.tiles), size=map_shape, p=tile_probs)
    int_map = jax.random.choice(key, len(game_def.tiles), shape=map_shape, p=jnp.array(tile_probs))
    # map_coords = np.argwhere(int_map != -1)
    map_coords = jnp.argwhere(int_map != -1, size=math.prod(map_shape))
    # Overwrite frequency-based tiles with tile-types that require fixed numbers of instances.
    n_fixed = sum([tile.num for tile in game_def.tiles if tile.num is not None])
    fixed_coords = map_coords[jax.random.choice(key, map_coords.shape[0], shape=(n_fixed,), replace=False)]
    i = 0
    for tile in game_def.tiles:
        if tile.prob == 0 and tile.num is not None:
            coord_list = fixed_coords[i: i + tile.num]
            # int_map[coord_list[:, 0], coord_list[:, 1]] = tile.idx
            int_map = int_map.at[coord_list[:, 0], coord_list[:, 1]].set(tile.idx)
            i += tile.num
    return map_to_onehot(int_map, game_def)


def map_to_onehot(int_map: np.ndarray, game_def: GameDef):
    map_arr = jnp.eye(len(game_def.tiles), dtype=np.int16)[int_map]
    map_arr = rearrange(map_arr, "h w c -> c h w")
    # self._update_player_pos(map_arr)
    # Activate parent/co-occuring tiles.
    for tile in game_def.tiles:
        coactive_tiles = tile.parents + tile.cooccurs
        if len(coactive_tiles) > 0:
            for cotile in coactive_tiles:
                # Activate parent channels of any child tiles wherever the latter are active.
                # map_arr[cotile.idx, map_arr[tile.idx] == 1] = 1
                map_arr = map_arr.at[cotile.idx, map_arr[tile.idx] == 1].set(1)
    # obj_set = {}
    return map_arr.astype(jnp.int16)


def init_base_env(cfg: EvoConfig, sb3=False) -> Tuple[PlayEnv, GenEnvParams]:
    game_def: GameDef = GAMES[cfg.game].make_env()
    if game_def.map is not None and len(np.array(game_def.map).shape) == 3:
        all_params = []
        # TODO: Could vectorize this. But not a priority, because this happens only once per train loop.
        for map_arr in game_def.map:
            game_def = game_def._replace(map=map_arr)
            env, params = init_base_env_single(cfg, game_def, sb3)
            all_params.append(params)
        # Stack all params
        all_params = stack_leaves(all_params)
        return env, all_params
    else:
        return init_base_env_single(cfg, game_def, sb3)
            

def init_base_env_single(cfg: EvoConfig, game_def: GameDef, sb3=False) -> Tuple[PlayEnv, GenEnvParams]:
    for rule in game_def.rules:
        rule.n_tile_types = len(game_def.tiles)
        rule = compile_rule(rule)
    if game_def.map is None:
        key = jax.random.PRNGKey(cfg.seed)
        map_arr = gen_random_int_map(key, game_def, cfg.map_shape).astype(jnp.int16)
    else:
        map_arr = map_to_onehot(game_def.map, game_def)
        map_arr = map_arr.astype(jnp.int16)
    # TODO: Flatten rule and subrule dimensions!
    rules_int = [rule.subrules_int for rule in game_def.rules]

    # FIXME: Below is misplaced. Needs to happen before subrules get rotated!
    # Find largest rule dimensions
    # max_rule_dims = np.max([rule.subrules_int.shape[-2:] for rule in game_def.rules], axis=0)
    # Pad rules with -1s to make them all the same size
    # rules_int = [np.pad(rule_int, ((0, 0), (0, 0), (0, 0), (0, max_rule_dims[0] - rule_int.shape[-2]), (0, max_rule_dims[1] - rule_int.shape[-1])), constant_values=-1) for rule_int in rules_int]

    # TODO: Deal with non-rotate/rotated mix of rules. Gotta flatten along subrule dimension basically
    rules_int = jnp.array(rules_int, dtype=jnp.int16)

    rule_rewards = jnp.array([rule.reward for rule in game_def.rules])
    rules = RuleData(rule=rules_int, reward=rule_rewards)
    rule_dones = jnp.array([rule.done for rule in game_def.rules], dtype=bool)
    player_placeable_tiles = \
        jnp.array([tile.idx for tile, placement_rule in game_def.player_placeable_tiles], dtype=int)
    if len(game_def.impassable_tiles) == 0:
        impassable_tiles = jnp.zeros((len(game_def.tiles),), dtype=bool)
    else:
        impassable_tiles = jnp.array([tile.idx for tile in game_def.impassable_tiles], dtype=int)
        impassable_tiles = jnp.eye(len(game_def.tiles), dtype=bool)[impassable_tiles].sum(axis=0)
    params = GenEnvParams(rules=rules, map=map_arr,
                        rule_dones=rule_dones,
                        player_placeable_tiles=player_placeable_tiles,
                        impassable_tiles=impassable_tiles,
                    )
    if not sb3:
        env = PlayEnv(
            cfg=cfg, height=cfg.map_shape[0], width=cfg.map_shape[1],
            game_def=game_def, params=params, max_episode_steps=cfg.max_episode_steps,
        )
    else:
        env = SB3PlayEnv(
            cfg=cfg, height=cfg.map_shape[0], width=cfg.map_shape[1],
            game_def=game_def, params=params,
        )
    # env = evo_base.make_env(10, 10)
    # env = maze.make_env(10, 10)
    # env = maze_for_evo.make_env(10, 10)
    # env = maze_spike.make_env(10, 10)
    # env = sokoban.make_env(10, 10)
    # env.search_tiles = [t for t in env.tiles]
    return env, params


def gen_rand_env_params(cfg: EvoConfig, rng: jax.random.PRNGKey, base_params: GenEnvParams, game_def: GameDef) \
        -> GenEnvParams:
    params = base_params
    if cfg.mutate_rules:
        rand_rules = gen_rand_rule(rng, base_params.rules.rule)
        # Generate random rewards per rule, in [-1, 1]
        rule_rewards = jax.random.randint(rng, (len(base_params.rules.reward),), minval=-1, maxval=2, dtype=jnp.int32)
        rules = RuleData(rule=rand_rules, reward=rule_rewards)
        # Make rules random
        rule_dones = jnp.zeros((len(base_params.rules.reward),), dtype=bool)
        params = params.replace(rules=rules, rule_dones=rule_dones)

    if cfg.mutate_map:
        map_arr = gen_random_int_map(rng, game_def, cfg.map_shape).astype(jnp.int16)
        params = params.replace(map=map_arr)
    return params

# def load_game_to_env(env: PlayEnv, individual: IndividualData):
#     env._map_queue = [individual.params.map,]
#     env.rules = individual.params.rules
#     env.tiles = individual.tiles
#     env._init_rules = individual.rules
#     params = get_params_from_individual(env, individual)
#     env.init_obs_space(params=params)
#     return env


# TODO: individual should basically be its own dataclass
# def get_params_from_individual(env: PlayEnv, individual: IndividualData):
#     params = GenEnvParams(rules=jnp.array([rule.subrules_int for rule in individual.rules], dtype=jnp.int16),
#                        map=individual.map,
#                        rule_rewards=jnp.array([rule.reward for rule in individual.rules]),
#                        rule_dones=jnp.array([rule.done for rule in individual.rules], dtype=bool),
#                        player_placeable_tiles=jnp.array([tile.idx for tile, placement_rule in env.game_def.player_placeable_tiles]))
#     return params
