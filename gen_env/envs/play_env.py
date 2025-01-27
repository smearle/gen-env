import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import math
import random
from timeit import default_timer as timer
from typing import Dict, Iterable, List, Optional, Tuple, Union

import chex
import cv2
from einops import rearrange, repeat
from flax import struct
import gym
from gym import spaces
import jax
from jax import numpy as jnp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from gen_env.configs.config import EvoConfig
from gen_env.events import Event, EventGraph
from gen_env.objects import ObjectType
from gen_env.rules import Rule, RuleData, RuleSet
from gen_env.tiles import TileNot, TilePlacement, TileSet, TileType
from gen_env.envs.utils import draw_triangle
from gen_env.variables import Variable

from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'


@dataclass
class GameDef:
    tiles: TileSet
    rules: RuleSet
    player_placeable_tiles: list
    search_tiles: Iterable[TileType]
    impassable_tiles: Iterable[TileType] = field(default_factory=lambda: [])
    map: Optional[np.ndarray] = None
    done_at_reward: Optional[int] = None


@struct.dataclass
class GenEnvParams:
    rules: RuleData
    rule_dones: chex.Array
    map: chex.Array
    player_placeable_tiles: chex.Array
    impassable_tiles: chex.Array

    env_idx: Optional[int] = None
    noop_ep_rew: Optional[int] = None
    random_ep_rew: Optional[int] = None
    search_ep_rew: Optional[int] = None
    best_rl_ep_rew: Optional[int] = None
    best_ep_rew: Optional[int] = None

    rew_bias: float = 0
    rew_scale: float = 1


@struct.dataclass
class GenEnvState(struct.PyTreeNode):
    n_step: int
    map: chex.Array
    # obj_set: Iterable
    player_rot: int
    player_pos: Tuple[int]
    ep_rew: int
    # params: GenEnvParams
    # queued_params: GenEnvParams
    rule_activations: chex.Array = None


@struct.dataclass
class GenEnvObs:
    map: chex.Array
    flat: chex.Array


class PlayEnv(gym.Env):
    placement_positions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
    tile_size = 32
    N_MOVE_ACTIONS = 4
    
    def __init__(self, width: int, height: int,
            game_def: GameDef,
            # tiles: Iterable[TileType], 
            params: GenEnvParams,
            # player_placeable_tiles: List[Tuple[TileType, TilePlacement]], 
            object_types: List[ObjectType] = [],
            # search_tiles: List[TileType] = None,
            events: Iterable[Event] = [],
            variables: Iterable[Variable] = [],
            done_at_reward: int = None,
            max_episode_steps: int = 100,
            cfg: EvoConfig = None,
        ):
        """_summary_

        Args:
            width (int): The width of the 2D game map.
            height (int): The height of the 2D game map.
            tiles (list): A list of TileType objects. Must include a tile-type with name `player`.
            rules (list): A list of Rule objects, between TileTypes.
            done_at_reward (int): Defaults to None. Otherwise, episode ends when reward reaches this number.
        """
        tiles, search_tiles, player_placeable_tiles = \
            game_def.tiles, game_def.search_tiles, game_def.player_placeable_tiles

        self.game_def = game_def

        # Just for rendering. Not jax-able!
        self.rules = game_def.rules

        self.cfg = cfg
        # rules_int = np.array([r.subrules_int for r in rules])
        rules_int = params.rules
        # self.default_params = EnvParams(rules=rules_int)
        self._rot_dirs = jnp.array([(0, -1), (1, 0), (0, 1), (-1, 0)])
        self.map_shape = np.array([len(tiles), width, height])
        assert width == height

        self.full_view_size = width - 1
        if cfg.obs_window == -1:
            self.view_size = self.full_view_size
        else:
            self.view_size = cfg.obs_window

        # Which game in the game_archive are we loading next?
        self._game_idx = 0

        # FIXME: too hardcoded (for maze_for_evo) rn
        self._n_fixed_rules = 0

        self.ep_rew = jnp.array([0])
        self._done = False
        if search_tiles is None:
            self._search_tiles = tiles
        else:
            self._search_tiles = search_tiles
        self._search_tile_idxs = np.array([tile.idx for tile in self._search_tiles])
        self.event_graph = EventGraph(events)
        self._has_applied_rule = False
        self.n_step = 0
        self.max_episode_steps = max_episode_steps
        self.w, self.h = width, height
        self.tiles = tiles
        # [setattr(tile, 'idx', i) for i, tile in enumerate(tiles)]
        tiles_by_name = {t.name: t for t in tiles}
        # Assuming here that we always have player and floor...
        self.player_idx = tiles_by_name['player'].idx if 'player' in tiles_by_name else 0
        self.tile_probs = [tile.prob for tile in tiles]
        # Add white for background when rendering individual tile-channel images.
        self.tile_colors = np.array([tile.color for tile in tiles] + [[255,255,255]], dtype=np.uint8)
        # Rules as they should be at the beginning of the episode (in case later events should change them)
        # self._init_rules = rules 
        # self.rules = copy.copy(rules)
        self.map: np.ndarray = None
        self.objects: Iterable[ObjectType.GameObject] = []
        self.player_pos: Tuple[int] = None
        self.player_force_arr: np.ndarray = None
        # No rotation
        # self.action_space = spaces.Discrete(4)
        # Rotation
        N_ACTIONS = PlayEnv.N_MOVE_ACTIONS + len(player_placeable_tiles)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.num_actions = N_ACTIONS
        self.build_hist: list = []
        # self.static_builds: np.ndarray = None
        self.variables = variables
        self.window = None
        self.rend_im: np.ndarray = None

        self.screen = None
        self._actions = []
        for tile, placement in player_placeable_tiles:
            if placement == TilePlacement.CURRENT:
                self._actions += [(tile.idx, 0)]
            elif placement == TilePlacement.ADJACENT:
                # No rotation
                # self._actions += [(tile.idx, i) for i in range(1, 5)]
                # Rotation
                self._actions += [tile.idx]
            else:
                raise Exception
        self._done_at_reward = done_at_reward
        self._map_queue = []
        self._rule_queue = []
        self._map_id = 0
        self.init_obs_space(params)

        # max_rule_shape = max([r._in_out.shape for r in self.rules])
        max_rule_shape = max(params.rules.rule.shape[-2:])
        self.map_padding = (max_rule_shape + 1) // 2
        self.ROTATE_LEFT_ACTION = 0

    def init_obs_space(self, params: GenEnvParams):
        # self.observation_space = spaces.Box(0, 1, (self.w, self.h, len(self.tiles)))
        # Dictionary observation space containing box 2d map and flat list of rules
        # Note that we assume rule in/outs are fixed in size
        # len_rule_obs = sum([len(rule.observe(len(self.tiles))) for rule in params.rules[self._n_fixed_rules:]])
        len_rule_obs = np.prod(params.rules.rule.shape)
        # Lazily flattening observations for now. It is a binary array
        # Only observe player patch and rotation for now
        # self.observation_space = spaces.Dict({
        #     'map': spaces.Box(0, 1, (self.view_size * 2 + 1, self.view_size * 2 + 1, len(self.tiles))),
        #     'player_rot': spaces.Discrete(4),
        #     'rules': spaces.Box(0, 1, (len_rule_obs * len(self.rules),))
        # })
        self.observation_space = spaces.MultiBinary((self.view_size * 2 + 1) * (self.view_size * 2 + 1) * len(self.tiles) + 4 + len_rule_obs + 2)

    def queue_games(self, maps: Iterable[np.ndarray], rules: Iterable[np.ndarray]):
        self._map_queue = maps
        self._rule_queue = rules

    def _update_player_pos(self, state: GenEnvState):
        player_pos = jnp.argwhere(state.map[self.player_idx] == 1)
        return state.replace(player_pos=player_pos)
        # if self.player_pos.shape[0] < 1:
        #     self.player_pos = None
        #     return
        #     # TT()
        # if self.player_pos.shape[0] > 1:
        #     raise Exception("More than one player on map.")
        # assert self.player_pos.shape[0] == 1
        # self.player_pos = tuple(self.player_pos[0])
        
    def _update_cooccurs(self, map_arr: np.ndarray):
        for tile_type in self.tiles:
            if tile_type.cooccurs:
                for cooccur in tile_type.cooccurs:
                    map_arr[cooccur.idx, map_arr[tile_type.idx] == 1] = 1

    def _update_inhibits(self, map_arr: np.ndarray):
        for tile_type in self.tiles:
            # print(f"tile_type: {tile_type.name}")
            if tile_type.inhibits:
                # print(f"tile_type.inhibits: {tile_type.inhibits}")
                for inhibit in tile_type.inhibits:
                    # print(f"inhibit: {inhibit.name}")
                    map_arr[inhibit.idx, map_arr[tile_type.idx] == 1] = 0

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: GenEnvParams):
        self._done = False
        self.unwrapped._done = False
        rules, map_arr = params.rules, jnp.array(params.map, dtype=jnp.int16)

        # if len(self._rule_queue) > 0:
        #     self._game_idx = self._game_idx % len(self._rule_queue)
            # self.rules = copy.copy(self._rule_queue[self._game_idx])
            # self.map = copy.copy(self._map_queue[self._game_idx])
            # self._game_idx += 1
            # rules = queued_params.rules
            # map_arr = queued_params.map_arr 

        self.ep_rew = jnp.array([0])
        self._has_applied_rule = False
        # Reset rules.
        # self.rules = copy.copy(self._init_rules)
        # Reset variables.
        [v.reset() for v in self.variables]
        self.event_graph.reset()
        self.n_step = 0
        # self._last_reward = 0
        self._reward = 0.0
        # if len(self._map_queue) == 0:
        #     map_arr = self.gen_random_map()
        # else:
        #     map_arr = self._map_queue[self._map_id]
        #     # obj_set = {}
        # feeld mouse    self._map_id = (self._map_id + 1) % len(self._map_queue)
        self.player_rot = 0
        player_pos = jnp.argwhere(map_arr[self.player_idx] == 1, size=1)[0]
        state = GenEnvState(
            n_step=self.n_step, map=map_arr, #, obj_set=obj_set,
            player_rot=jnp.array([0]), ep_rew=jnp.array([0.0]),
            player_pos=player_pos,
            # params=params,
            # FIXME: padding is hard-coded here. Not a huge deal but will result in rendering issues if map/rule shape changes.
            rule_activations=jnp.zeros((len(rules.rule), map_arr.shape[1]+4, map_arr.shape[2]+4)),
        )
        # self._set_state(env_state)
        obs = self.get_obs(state, params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: GenEnvState,
        action: Union[int, float],
        params: GenEnvParams,
        reset_params: GenEnvParams,
    ) -> Tuple[chex.Array, GenEnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        # if params is None:
        #     params = self.default_params

        # This feels kind of sketchy but it does the trick        
        # reset_params = state.queued_params

        action = action.astype(jnp.int32)
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key, action=action, state=state, params=params
        )
        obs_re, state_re = self.reset_env(key_reset, reset_params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        # obs = jax.lax.select(done, obs_re, obs_st)
        # Generalizing this to flax dataclass observations
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st
        )
        params = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), reset_params, params
        )

        # Print action and donw
        # jax.debug.print("action {action} done {done}", action=action, done=done)
        return obs, state, reward, done, info, params.env_idx


    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[GenEnvParams] = None,
    ) -> Tuple[chex.Array, GenEnvState]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key, params)
        return obs, state


    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
            key: chex.PRNGKey,
            state: GenEnvState,
            action: int,
            params: GenEnvParams
        ):
        # TODO: Only pass global variable object to event graph.
        # self.event_graph.tick(self)
        state = self.act(action=action, state=state, params=params)
        state, reward, done = self.tick(state, params)
        reward = (reward + params.rew_bias / self.max_episode_steps) * params.rew_scale
        n_step = state.n_step + 1
        ep_rew = state.ep_rew + reward
        state = state.replace(n_step=n_step, ep_rew=ep_rew)
        obs = self.get_obs(state, params)
        info = {}
        return (
            jax.lax.stop_gradient(obs), 
            jax.lax.stop_gradient(state),
            reward,
            done,
            info
        )

    def step_classic(self, action: chex.Array, state: GenEnvState):
        # TODO: Only pass global variable object to event graph.
        self.event_graph.tick(self)
        self.act(action)
        reward = self.tick()
        n_step = state.n_step + 1
        if self._done:
            pass
            # print('done at step')
        ep_rew = state.ep_rew + reward
        self.ep_rew = ep_rew
        self.n_step = n_step
        state = GenEnvState(
            n_step=n_step,
            map_arr=self.map,
            # obj_set=self.objects,
            player_rot=self.player_rot,
            ep_rew=ep_rew
        )
        return state, self.get_obs(), reward, self._done, {}

    def act(self, action: int, state: GenEnvState, params: GenEnvParams):
        # If action is 0 or 1, rotate player.
        # Otherwise, move player to adjacent tile according to rotation.
        # move_coeff = jnp.where(jnp.where(action < 2, 0, action) == 2, -1, 1)
        rot_diffs = jnp.array([1, -1, 0, 0, 0])
        rot_diff = rot_diffs[action]
        acts_to_move_coeffs = jnp.array([0, 0, 1, -1, 0])
        place_tiles = jnp.array([0, 0, 0, 0, 1])

        # player_rot = ((state.player_rot + rot_diff) % 4).item()
        # Above but jax-compatible
        player_rot = jnp.mod(state.player_rot + rot_diff, 4).astype(jnp.int32)

        move_coeff = acts_to_move_coeffs[action]
        place_tile = place_tiles[action]
        new_pos = state.player_pos + move_coeff * self._rot_dirs[player_rot][0]
        new_pos = new_pos % jnp.array(state.map.shape[1:])
        new_map = state.map
        player_pos = state.player_pos
        new_map = new_map.at[self.player_idx, player_pos[0], player_pos[1]].set(0)
        # Apply new position if within bounds.
        # player_pos = jax.lax.select(
        #     jnp.all(new_pos >= 0) & jnp.all(new_pos < self.map_shape[1:]),
        #     new_pos,
        #     player_pos,
        # )

        new_tile = new_map[:, new_pos[0], new_pos[1]]
        is_passable = jnp.logical_not(jnp.any(new_tile * params.impassable_tiles))
        player_pos = jax.lax.select(is_passable, new_pos, player_pos)

        # Actually print the values in the jax array
        # print(f"player_pos: {player_pos}")
        # But also inside compiled jax code
        # print(f"player_pos: {player_pos[0]}, {player_pos[1]}")

        # Hackish way
        n_prev = PlayEnv.N_MOVE_ACTIONS
        if len(params.player_placeable_tiles) > 0:
            trg_pos = player_pos + self._rot_dirs[player_rot][0]
            trg_pos = trg_pos % jnp.array(state.map.shape[1:])
            new_map = new_map.at[params.player_placeable_tiles[jnp.int16(action) - n_prev], trg_pos[0], trg_pos[1]].set(place_tile)

        new_map = new_map.at[self.player_idx, player_pos[0], player_pos[1]].set(1)
        state = state.replace(player_rot=player_rot, player_pos=player_pos,
                              map=new_map)
        return state

    def get_obs(self, state: GenEnvState, params: GenEnvParams):
        # return self.observe_map()
        # return {
        #     'map': self.observe_map(),
        #     'rules': self.observe_rules(),
        #     'player_rot': np.eye(4)[self.player_rot].astype(np.float32),
        # }
        map_obs = self.observe_map(state.map, state.player_pos)
        if self.cfg.obs_rew_norm:
            rew_norm_obs = jnp.array([params.rew_bias / self.max_episode_steps, params.rew_scale]).flatten()
        else:
            rew_norm_obs = jnp.zeros((2,))
        flat_obs = jnp.concatenate((
            jnp.eye(4)[state.player_rot].astype(jnp.float32).flatten(),
            self.observe_rules(params).flatten(),
            rew_norm_obs,
        ))
        return GenEnvObs(map_obs, flat_obs)
        
    def gen_dummy_obs(self, params: GenEnvParams):
        map_obs = self.observe_map(jnp.zeros(self.map_shape), (0, 0))
        flat_obs = jnp.concatenate((
            jnp.eye(4)[0].astype(jnp.float32).flatten(),
            self.observe_rules(params).flatten(),
            jnp.zeros((2,))
        ))
        return GenEnvObs(map_obs[None], flat_obs[None])

    def repair_map(key, map: chex.Array, fixed_tile_nums: chex.Array):
        # map is (n_tiles, w, h)
        # fixed_num_tiles = [t for t in tiles if t.num is not None]
        # free_num_tile_idxs = [t.idx for t in tiles if t.num is None]

        # padding the map at the right sides allows us to deal with ``none''
        # indices of (-1,-1,) in the result of jnp.argmax
        pad_map = jnp.pad(map, ((0, 0), (0, 1), (0, 1)))
        tile_nums = map.sum(axis=(1, 2))
        # How many to add/delete
        n_add_delete = fixed_tile_nums - tile_nums
        # set -1 in n_add wehere fixed_tile_nums is -1
        n_add_delete = jnp.where(fixed_tile_nums == -1, 0, n_add_delete)
        n_add = jnp.where(n_add_delete > 0, n_add_delete, 0)
        n_delete = jnp.where(n_add_delete < 0, -n_add_delete, 0)

        # generate random noise between 0 and .5 to add to the map, used for
        # randomly ranking/ordering which tiles to add/delete

        # flatten the map to 1d
        # noised_map = rearrange(noised_map, 'c h w -> c (h w)')
        # sorted_idxs = jnp.argsort(noised_map, axis=1, kind='stable')

        # Function to add activations
        def add_activations(key, channel, n):
            flat_channel = channel.ravel()
            zero_indices = jnp.where(flat_channel == 0)[0]
            selected_indices = jax.random.choice(key, zero_indices, shape=(n,), replace=False)
            flat_channel = flat_channel.at[selected_indices].set(1)
            return flat_channel.reshape(channel.shape)

        # Function to delete activations
        def delete_activations(key, channel, n):
            flat_channel = channel.ravel()
            one_indices = jnp.where(flat_channel == 1)[0]
            selected_indices = jax.random.choice(key, one_indices, shape=(n,), replace=False)
            flat_channel = flat_channel.at[selected_indices].set(0)
            return flat_channel.reshape(channel.shape)

        # Processing each channel
        for i in range(map.shape[0]):
            if n_add[i] > 0:
                key, subkey = jax.random.split(key)
                map = map.at[i].set(add_activations(subkey, map[i], n_add[i]))
            elif n_delete[i] > 0:
                key, subkey = jax.random.split(key)
                map = map.at[i].set(delete_activations(subkey, map[i], n_delete[i]))

        return map



        # n_add = max(0, tile_nums.sum() - sum([t.num for t in fixed_tile_nums]))

        # For tile types with fixed numbers, make sure this many occur
        # for tile in fixed_num_tiles:
        #     # If there are too many, remove some
        #     # print(f"Checking {tile.name} tiles")
        #     idxs = np.where(map[tile.idx] == 1)[0]
        #     # print(f"Found {len(idxs)} {tile.name} tiles")
        #     if len(idxs) > tile.num:
        #         # print(f'Found too many {tile.name} tiles, removing some')
        #         for idx in idxs[tile.num:]:
        #             map.flat[idx] = np.random.choice(free_num_tile_idxs)
        #         # print(f'Removed {len(idxs) - tile.num} tiles')
        #         assert len(np.where(map == tile.idx)[0]) == tile.num
        #     elif len(idxs) < tile.num:
        #         # FIXME: Not sure if this is working
        #         net_idxs = []
        #         chs_i = 0
        #         # np.random.shuffle(free_num_tile_idxs)
        #         freeze_num_tile_idxs = jax.random.shuffle(key, free_num_tile_idxs)
        #         while len(net_idxs) < tile.num:
        #             # Replace only 1 type of tile (weird)
        #             idxs = np.where(map.flat == free_num_tile_idxs[chs_i])[0]
        #             net_idxs += idxs.tolist()
        #             chs_i += 1
        #             if chs_i >= len(free_num_tile_idxs):
        #                 print(f"Warning: Not enough tiles to mutate into {tile.name} tiles")
        #                 break
        #         idxs = np.array(net_idxs[:tile.num])
        #         for idx in idxs:
        #             map.flat[idx] = tile.idx
        #         assert len(np.where(map == tile.idx)[0]) == tile.num
        # for tile in fixed_num_tiles:
        #     assert len(np.where(map == tile.idx)[0]) == tile.num
        # return rearrange(np.eye(len(tiles), dtype=np.int16)[map], 'h w c -> c h w')

    def observe_map(self, map_arr, player_pos):
        obs = rearrange(map_arr, 'b h w -> h w b')
        # Pad map to view size.
        obs = jnp.pad(obs, (
            (self.full_view_size, self.full_view_size),
            (self.full_view_size, self.full_view_size), (0, 0)), 'constant')
        # Crop map to player's view.
        # if self.player_pos is not None:
        # TODO: Rotate observation?
        x, y = player_pos
        obs = jax.lax.dynamic_slice(obs, (x, y, 0), (2 * self.full_view_size + 1, 2 * self.full_view_size + 1, len(self.tiles)))
        if self.cfg.obs_window != -1:
            lpad = (self.full_view_size - self.cfg.obs_window) // 2
            rpad = self.full_view_size - self.cfg.obs_window - lpad
            new_obs = jnp.zeros_like(obs)
            new_obs = new_obs.at[:, lpad:-rpad, lpad:-rpad].set(
                obs[:, lpad:-rpad, lpad:-rpad])
            obs = new_obs
            
        # obs = obs[x: x + 2 * self.view_size + 1,
        #           y: y + 2 * self.view_size + 1]
        n_tiles = self.map_shape[0]
        assert obs.shape == (2 * self.full_view_size + 1, 2 * self.full_view_size + 1, len(self.tiles))
        return obs.astype(np.float32)

    def observe_rules(self, params: GenEnvParams):
        # Hardcoded for maze_for_evo to ignore first 2 (unchanging) rules
        # if self._n_fixed_rules == len(self.rules):
        #     return np.zeros((0,), dtype=np.float32)
        # rule_obs = np.concatenate([r.observe(n_tiles=len(self.tiles)) for r in self.rules[self._n_fixed_rules:]])
        rule_obs = params.rules.rule
        rule_reward_obs = params.rules.reward
        rule_obs = jnp.concatenate((rule_obs.flatten(), rule_reward_obs), axis=-1)
        rule_obs = jax.lax.select(
            self.cfg.hide_rules,
            jnp.zeros_like(rule_obs),
            rule_obs
        )

        # if self.cfg.hide_rules:
        #     # Hide rules
        #     rule_obs = np.zeros_like(rule_obs)
        
        return rule_obs.astype(np.float32)

    def render_cell(self, cell, tile_size):
        cell_im = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        k_idxs = np.argwhere(cell == 1)[::-1]
        n_tiles_ij = len(k_idxs)
        for k_i, k in enumerate(k_idxs):
            # Each tile being stacked has reduced width
            tile_im = self.tile_colors[k]
            # tile_k_width = int(tile_size * (n_tiles - n_tiles_ij) / n_tiles)
            margin_ki = tile_size // 2
            a0 = b0 = int(k_i / n_tiles_ij * margin_ki)
            a1 = b1 = -a0 if k_i > 0 else tile_size
            cell_im[a0:a1, b0:b1] = tile_im
        return cell_im

    def render_flat_map(self, map, tile_size):
        # Flat render of all tiles
        # For each cell on the map, render the tile with lowe
        row_ims = []
        for i in range(map.shape[1]):
            col_ims = []
            for j in range(map.shape[2]):
                cell_im = self.render_cell(map[:, i, j], tile_size)
                col_ims.append(cell_im)
            row_ims.append(np.concatenate(col_ims, axis=1))
        flat_im = np.concatenate(row_ims, axis=0)
        flat_im = np.pad(flat_im, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        return flat_im
    
    def render(self, state: GenEnvState, params: GenEnvParams, mode='human'):
        font = ImageFont.load_default()
        tile_size = self.tile_size
        
        RENDER_LAYERS = False

        if not RENDER_LAYERS:
            tile_size = 64
            tile_ims = self.render_flat_map(state.map, tile_size)
        
        else:
            # self.rend_im = np.zeros_like(self.int_map)
            # Create an int map where the first tiles in `self.tiles` take priority.
            int_map = np.full(state.map.shape[1:], dtype=np.int16, fill_value=-1)
            tile_ims = []
            for tile in self.tiles[::-1]:
                # if tile.color is not None:
                int_map[np.array(state.map)[tile.idx] == 1] = tile.idx
                tile_map = np.where(state.map[tile.idx] == 1, tile.idx, -1)
                # If this is the player, render as a triangle according to its rotation
                tile_im = self.tile_colors[tile_map]
                # Pad the tile image and add text to the bottom
                tile_im = repeat(tile_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
                tile_im = np.pad(tile_im, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                # Get text as rgb np array
                text = tile.name
                # Draw text on image
                # font = ImageFont.truetype("arial.ttf", 20)
                # Get font available on mac 
                img_pil = Image.fromarray(tile_im)
                draw = ImageDraw.Draw(img_pil)
                draw.text((10, 10), text, font=font, fill=(255, 255, 255, 0))
                tile_im = np.array(img_pil)
                tile_ims.append(tile_im)

            # Extra one for player pos
            tile_im = np.zeros_like(self.tile_colors[tile_map]) + 255
            tile_im = repeat(tile_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
            tile_im = np.pad(tile_im, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            tile_im = draw_triangle(tile_im, state.player_pos, state.player_rot, tile.color, tile_size)
            img_pil = Image.fromarray(tile_im)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), 'player pos', font=font, fill=(255, 255, 255, 0))
            tile_im = np.array(img_pil)
            tile_ims.append(tile_im)

            flat_im = self.render_flat_map(state.map, tile_size)
            # tile_im = self.tile_colors[int_map]
            # tile_im = repeat(tile_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
            # tile_im = np.pad(tile_im, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
            tile_ims = [flat_im] + tile_ims[::-1]

            map_h, map_w = tile_ims[0].shape[:-1]

            # Add empty images to the end of the tile images to fill out the grid
            # Find the smallest square number greater than or equal to n_tiles
            # Reshape the tile images into a grid
            tile_ims = np.array(tile_ims)
            n_ims = tile_ims.shape[0]
            n_ims_sqrt = int(np.ceil(np.sqrt(n_ims)))
            n_ims_sqrt2 = n_ims_sqrt ** 2
            n_empty_ims = n_ims_sqrt2 - n_ims
            empty_im = np.zeros((map_h, map_w, 3), dtype=np.uint8)
            empty_ims = [empty_im] * n_empty_ims
            tile_ims = np.concatenate([tile_ims, empty_ims])
            tile_ims = tile_ims.reshape(n_ims_sqrt, n_ims_sqrt, map_h, map_w, 3)
            # Add padding between tiles
            pw = 2
            tile_ims = np.pad(tile_ims, ((0, 0), (0, 0), (pw, pw), (pw, pw), (0, 0)), mode='constant', constant_values=0)

            # Concatenate the tile images into a single image
            tile_ims = rearrange(tile_ims, 'n1 n2 h w c -> (n1 h) (n2 w) c')

        # Below the image, add a row of text showing episode/cumulative reward
        # Add padding below the image
        tile_ims = np.pad(tile_ims, ((0, 30), (0, 0), (0, 0)), mode='constant', constant_values=0)
        text = f'Reward: {state.ep_rew}'
        # Paste text
        img_pil = Image.fromarray(tile_ims)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), text, font=font, fill=(255, 255, 255, 0))
        tile_ims = np.array(img_pil)

        tile_size = 16

        # On a separate canvas, visualize rules.
        # Visualize each rule's in_out pattern using grids of tiles
        rule_ims = []
        # for rule in self.rules:
        for rule_i, (rule_int, rule) in enumerate(zip(params.rules.rule, self.rules)):
            # print(rule_int.shape)
            # Select the first rotation-variant (subrule) of the rule
            in_out = rule_int[0]
            # Get the tile images corresponding to the in_out pattern
            p_ims = []
            i, o = in_out
            for p in (i, o):
                row_ims = []
                for j in range(p.shape[1]):
                    col_ims = []
                    for k in range(p.shape[2]):
                        cell_im = self.render_cell(p[:, j, k], tile_size)
                        col_ims.append(cell_im)
                    row_ims.append(np.concatenate(col_ims, axis=1))
                p_im = np.concatenate(row_ims, axis=0)
                p_ims.append(p_im)
            # Concatenate the input and output images into a single image, with padding and a left-right arrow in between
            p_ims = np.array(p_ims)
            p_ims = np.pad(p_ims, ((0, 0), (0, 0), (0, 30), (0, 0)), mode='constant', constant_values=0)
            # arrow_im = np.zeros((map_h, 30, 3), dtype=np.uint8)
            # arrow_im[:, 15, :] = 255
            # arrow_im[15, :, :] = 255
            # p_ims = np.concatenate([p_ims, arrow_im], axis=3)
            p_ims = np.concatenate(p_ims, axis=1)
            # Add padding below the image
            p_ims = np.pad(p_ims, ((50, 30), (0, 0), (0, 0)), mode='constant', constant_values=0)
            # Paste text
            text = f'Rule {rule.name}'
            # If the rule has non-zero reward, add the reward to the text
            r_reward = params.rules.reward[rule_i]
            if r_reward != 0:
                text += f'\nReward: {r_reward}'
            img_pil = Image.fromarray(p_ims)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), text, font=font, fill=(255, 255, 255, 0))
            p_ims = np.array(img_pil)

            # Underneath the rule, add a grid the same size as the map, which lights up wherever the rule has been 
            # applied.
            binary_rule_activ_map = state.rule_activations[rule_i]
            # This has shape (1, map_h, map_w)
            # Transform the binary rule activations into an image (inactive is red, active is green), with tiles of the
            # width below.
            tile_width = 10
            # Repeat each element to increase resolution
            tiled_activ_map = np.repeat(np.repeat(binary_rule_activ_map, tile_width, axis=0), tile_width, axis=1)

            # Map the binary values to colors: 1 (active) to green, 0 (inactive) to red
            color_map = np.zeros((*tiled_activ_map.shape, 3), dtype=np.uint8)  # Prepare an array for RGB values

            # NOTE: Hardcoded, assumes subrules are rotations (and thus that there are 4 possible values in the map)
            color_map[tiled_activ_map == 1] = [0, 255*1/4, 0]  # Active: Green
            color_map[tiled_activ_map == 2] = [0, 255*1/2, 0]  # Active: Green
            color_map[tiled_activ_map == 3] = [0, 255*3/4, 0]  # Active: Green
            color_map[tiled_activ_map == 4] = [0, 255, 0]  # Active: Green
            color_map[tiled_activ_map == 0] = [255, 0, 0]  # Inactive: Red
            
            # Pad the color map to match the size of the rule image
            pad_h = max(0, p_ims.shape[0] - color_map.shape[0])
            pad_w = max(0, p_ims.shape[1] - color_map.shape[1])
            color_map = np.pad(color_map, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
            pad_h = max(0, color_map.shape[0] - p_ims.shape[0])
            pad_w = max(0, color_map.shape[1] - p_ims.shape[1])
            p_ims = np.pad(p_ims, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

            p_ims = np.concatenate([p_ims, color_map], axis=0)
            rule_ims.append(p_ims)

        # Get the shape of the largest rule, pad other rules to match
        max_h = max([im.shape[0] for im in rule_ims])
        max_w = max([im.shape[1] for im in rule_ims])
        for i, im in enumerate(rule_ims):
            h, w = im.shape[:2]
            pad_h = max_h - h
            pad_w = max_w - w
            rule_ims[i] = np.pad(im, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        # Assert all rule images have the same shape
        assert len(set([im.shape for im in rule_ims])) == 1

        rule_ims = np.array(rule_ims)
        n_ims = rule_ims.shape[0]
        n_ims_sqrt = int(np.ceil(np.sqrt(n_ims)))
        n_ims_sqrt2 = n_ims_sqrt ** 2
        n_empty_ims = n_ims_sqrt2 - n_ims
        empty_im = np.zeros(rule_ims[0].shape, dtype=np.uint8)
        if n_empty_ims > 0:
            empty_ims = [empty_im] * n_empty_ims
            rule_ims = np.concatenate([rule_ims, empty_ims])
        rule_ims = rule_ims.reshape(n_ims_sqrt, n_ims_sqrt, *rule_ims[0].shape)
        # rule_ims = np.concatenate(rule_ims, axis=1)
        rule_ims = rearrange(rule_ims, 'n1 n2 h w c -> (n1 h) (n2 w) c')

        # Pad rules below to match the height of the tile images
        h, w = tile_ims.shape[:2]
        pad_h = max(0, h - rule_ims.shape[0])
        rule_ims = np.pad(rule_ims, ((0, pad_h), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Or pad tile images below to match the height of the rule images
        pad_h2 = max(0, rule_ims.shape[0] - h)
        tile_ims = np.pad(tile_ims, ((0, pad_h2), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Add the rules im to the right of the tile im, with padding between
        tile_ims = np.concatenate([tile_ims, rule_ims], axis=1)

        self.rend_im = tile_ims

        # self.rend_im = np.concatenate([self.rend_im, tiles_rend_im], axis=0)
        # self.rend_im = repeat(self.rend_im, 'h w -> h w 3')
        # self.rend_im = repeat(self.rend_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        # b0 = self.build_hist[0]
        # for b1 in self.build_hist[1:]:
            # x0, x1 = sorted([b0[0], b1[0]])
            # y0, y1 = sorted([b0[1], b1[1]])
            # self.rend_im[
                # x0 * tile_size + tile_size // 2 - pw: x1 * tile_size + tile_size // 2 + pw,
                # y0 * tile_size + tile_size // 2 - pw: y1 * tile_size + tile_size // 2 + pw] = [0, 1, 0]
            # b0 = b1
        # self.rend_im *= 255
        if mode == "human":
            rend_im = self.rend_im.copy()
            # rend_im[:, :, (0, 2)] = self.rend_im[:, :, (2, 0)]
            if self.window is None:
                self.window = cv2.namedWindow('Generated Environment', cv2.WINDOW_NORMAL)
            cv2.imshow('Generated Environment', rend_im)
            cv2.waitKey(1)
            return
        elif mode == "rgb_array":
            return self.rend_im
        if mode == "pygame":
            import pygame
            # Map human-input keys to action indices. Here we assume the first 4 actions correspond to player navigation 
            # (i.e. placement of `force` at adjacent tiles).
            self.keys_to_acts = {
                pygame.K_LEFT: 0,
                pygame.K_RIGHT: 1,
                pygame.K_UP: 2,
                pygame.K_DOWN: 3,
                pygame.K_q: 4,
            }
            self.rend_im = np.flip(self.rend_im, axis=0)
            # Rotate to match pygame
            self.rend_im = np.rot90(self.rend_im, k=-1)

            # Scale up the image by 2
            win_shape = tuple((np.array(self.rend_im.shape)[:-1][::-1] * self.cfg.window_scale).astype(int))
            self.rend_im = cv2.resize(self.rend_im, win_shape)

            if self.screen is None:
                pygame.init()
                # Flip image to match pygame coordinate system
                # Set up the drawing window to match size of rend_im
                self.screen = pygame.display.set_mode((self.rend_im.shape[0], self.rend_im.shape[1]))
                # self.screen = pygame.display.set_mode([(len(self.tiles)+1)*self.h*GenEnv.tile_size, self.w*GenEnv.tile_size])
            pygame_render_im(self.screen, self.rend_im)
            return
        else:
            cv2.imshow('Generated Environment', self.rend_im)
            cv2.waitKey(1)
            # return self.rend_im

    def tick(self, state: GenEnvState, params: GenEnvParams):
        # self._last_reward = self._reward
        # for obj in self.objects:
        #     obj.tick(self)
        map_arr, reward, done, sr_activs, has_applied_rule, rule_time_ms\
            = apply_rules(state.map, params, self.map_padding)
        map_arr = map_arr.astype(jnp.int16)
        # if self._done_at_reward is not None:
        #     done = done or reward == self._done_at_reward
        done = done | (state.n_step >= self.max_episode_steps) | \
            (jnp.sum(map_arr[self.player_idx]) == 0)
        player_pos = jnp.argwhere(map_arr[self.player_idx] == 1, size=1)[0]

        # We're just using these to visualize where a given rule is triggered.
        # Sum over rotation axis.
        rule_activations = sr_activs.sum(axis=1)
        # Remove these other axes... whatever they are lol.
        rule_activations = rule_activations[:, 0, 0]

        state = state.replace(player_pos=player_pos, map=map_arr, rule_activations=rule_activations,)
        # map_arr = jax.lax.cond(
        #     not done,
        #     lambda map_arr: self._compile_map(map_arr),
        #     lambda map_arr: map_arr,
        # )
        return state, reward, done

    def _remove_additional_players(self, map_arr: chex.Array):
        # Remove additional players
        player_pos = jnp.argwhere(map_arr[self.player_idx] == 1, size=1)[0]
        if player_pos.shape[0] > 1:
            for i in range(1, player_pos.shape[0]):
                # Remove redundant players
                self.map[self.player_idx, player_pos[i][0], player_pos[i][1]] = 0
                # rand_tile_type = random.randint(0, len(self.tiles) - 2)
                # Activate the next possible tile so that we know at least one tile is active
                rand_tile_type = 1
                rand_tile_type = rand_tile_type if rand_tile_type < self.player_idx else rand_tile_type + 1
                self.map[rand_tile_type, player_pos[i][0], player_pos[i][1]] = 1
        elif player_pos.shape[0] == 0:
            # Get random x y position
            x, y = np.random.randint(0, self.map.shape[1]), np.random.randint(0, self.map.shape[2])
            # Set random tile to be player
            self.map[self.player_idx, x, y] = 1

    def _compile_map(self, map_arr: chex.Array):
        map_arr = self._remove_additional_players(map_arr)
        map_arr = self._update_player_pos(map_arr)
        map_arr = self._update_cooccurs(map_arr)
        map_arr = self._update_inhibits(map_arr)
        return map_arr

    def tick_human(self, key: jax.random.PRNGKey, state: GenEnvState, params: GenEnvParams):
        import pygame
        done = False
        # If there is no player, take any action to tick the environment (e.g. during level-generation).
        if state.player_pos is None:
            action = 0
            state, obs, rew, done, info = self.step(key=key, action=action,
                                             state=state, params=params)
            self.render(mode='pygame', state=state, params=params)
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key in self.keys_to_acts:
                    action = self.keys_to_acts[event.key]
                    obs, state, rew, done, info = \
                        self.step_env(key=key, action=action, state=state, params=params)
                        # self.step(key=key, action=action, state=state,
                        #           params=params)
                    state: GenEnvState

                    self.render(mode='pygame', state=state, params=params)
                    # if self._last_reward != self._reward:
                    print(f"Step: {state.n_step}, Reward: {state.ep_rew}")
                elif event.key == pygame.K_x:
                    done = True
            if done:
                map_arr = gen_random_map(key, self.game_def, self.cfg.map_shape)
                params = params.replace(map=map_arr)
                obs, state = self.reset_env(key=key, params=params)
                done = False
                self.render(mode='pygame', state=state, params=params)
        return state


    # def get_state(self):
    #     return GenEnvState(n_step=self.n_step, map_arr=self.map.copy(),
    #                     # obj_set=self.objects,
    #         player_rot=self.player_rot, ep_rew=self.ep_rew)

    # def set_state(self, state: GenEnvState):
    #     state = copy.deepcopy(state)
    #     self._set_state(state)
    #     # TODO: setting variables and event graph.

    def hashable(self, state: GenEnvState):
        # assert hash(state['map_arr'].tobytes()) == hash(state['map_arr'].tobytes())
        search_state = state.map[self._search_tile_idxs]
        player_rot = state.player_rot
        # Uniquely hash based on player rotation and search tile states
        return hash((player_rot.item(), search_state.astype(bool).tobytes()))

    def _hashable(self, state: GenEnvState):
        # assert hash(state['map_arr'].tobytes()) == hash(state['map_arr'].tobytes())
        search_state = state.map[self._search_tile_idxs]
        player_rot = state.player_rot
        # Uniquely hash based on player rotation and search tile states
        return hash((player_rot.item(), search_state.astype(bool).tobytes()))


    def _set_state(self, state: GenEnvState):
        map_arr = state.map
        self.n_step = state.n_step
        self.map = map_arr
        # self.objects = obj_set
        self.height, self.width = self.map.shape[1:]
        self.player_rot = state.player_rot
        self.ep_rew = state.ep_rew
        self._compile_map()


class SB3PlayEnv(PlayEnv):
    def reset(self, *args, **kwargs):
        state, obs = super().reset(*args, **kwargs)
        return obs

    def step(self, *args, **kwargs):
        state = self.get_state()
        state, obs, rew, done, info = super().step(*args, **kwargs, state=state)
        return obs, rew, done, info

        
def apply_subrule(map: np.ndarray, subrule_int: np.ndarray):
    # Apply, e.j., rotations of the base rule
    # Add one output channel
    subrule_int = rearrange(subrule_int, 'iop (o i) h w -> iop o i h w', o=1)
    inp, outp = subrule_int

    # Pad the map, wrapping around the edges
    pad_width = 1
    # Make it toroidal
    # Use jax to apply a convolution to the map
    sr_activs = jax.lax.conv(map, inp, window_strides=(1, 1), padding='SAME')
    # How many tiles are expected in the input pattern. Not summing absence of tiles here
    inp_posi = jnp.clip(inp, 0, 1)
    n_constraints = inp_posi.sum()
    # Identify positions at which all constraints were met
    sr_activs = (sr_activs == n_constraints).astype(jnp.float32)

    # jax.debug.print('sr_activs {sr_activs}', sr_activs=sr_activs)
    # if sr_activs.sum() > 0 and rule.reward > 0:
    #     jax.debug.breakpoint()

    # Note that this can have values of `-1` to remove tiles
    outp = rearrange(outp, 'o i h w -> i o h w')

    # Need to flip along height/width dimensions for transposed convolution to work as expected
    outp = jnp.flip(outp, 2)
    outp = jnp.flip(outp, 3)

    # jax.debug.print('outp {outp}', outp=outp)

    # Now paste the output pattern wherever input is active
    out_map = jax.lax.conv_transpose(sr_activs, outp, (1, 1), 'SAME',
                                        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))
    # jax.debug.print('out_map {out_map}', out_map=out_map)
    
    # Crop out_map to be the height and width as next_map, cropping more on the right/lower side if uneven
    # crop_shapes = (out_map.shape[2] - next_map.shape[2]) / 2, (out_map.shape[3] - next_map.shape[3]) / 2
    # crop_shapes = math.ceil(crop_shapes[0]), math.ceil(crop_shapes[1])
    # out_map = out_map[:, :, crop_shapes[0]: crop_shapes[0] + next_map.shape[2], crop_shapes[1]:crop_shapes[1]+ next_map.shape[3]]

    # jax.debug.breakpoint()

    # if sr_activs.sum() > 0:
    #     has_applied_rule = True
    #     # breakpoint()
    
    return out_map, sr_activs

    # DEPRECATED approach
    # xys = np.indices((h + map_padding, w + map_padding))
    # xys = rearrange(xys, 'xy h w -> (h w) xy')
    # if rule.random:
    #     np.random.shuffle(xys)
    #     # print(f'y: {y}')
    # for (x, y) in xys:
    #     match = True
    #     for subp in inp:
    #         if subp.shape[0] + x > map.shape[1] or subp.shape[1] + y > map.shape[2]:
    #             match = False
    #             break
    #         if not match:
    #             break
    #         for i in range(subp.shape[0]):
    #             if not match:
    #                 break
    #             for j in range(subp.shape[1]):
    #                 tile = subp[i, j]
    #                 if tile is None:
    #                     continue
    #                 if map[tile.get_idx(), x + i, y + j] != tile.trg_val:
    #                     match = False
    #                     break
    #     if match:
    #         # print(f'matched rule {rule.name} at {x}, {y}')
    #         # print(f'rule has input \n{inp}\n and output \n{out}')
    #         [f() for f in rule.application_funcs]
    #         [blocked_rules.add(r) for r in rule.inhibits]
    #         for r in rule.children:
    #             if r in rules_set:
    #                 continue
    #             rules_set.add(r)
    #             rules.append(r)
    #         reward += rule.reward
    #         done = done or rule.done
    #         for k, subp in enumerate(out):
    #             for i in range(subp.shape[0]):
    #                 for j in range(subp.shape[1]):
    #                     # Remove the corresponding tile in the input pattern if one exists.
    #                     in_tile = inp[k, i, j]
    #                     if in_tile is not None:
    #                         # Note that this has no effect when in_tile is a NotTile.
    #                         next_map[in_tile.get_idx(), x + i, y + j] = 0
    #                     out_tile = subp[i, j]
    #                     if out_tile is None:
    #                         continue
    #                     # if out_tile.get_idx() == -1:
    #                     #     breakpoint()
    #                     # if out_tile.get_idx() == 7:
    #                     #     breakpoint()
    #                     next_map[out_tile.get_idx(), x + i, y + j] = 1
    #         n_rule_applications += 1
    #         if n_rule_applications >= rule.max_applications:
    #             # print(f'Rule {rule.name} exceeded max applications')
    #             break
        
    # else:
    #     continue

    # Will break the subrule loop if we have broken the board-scanning loop.
    # break

VMAP = True
    
def apply_rule(map: chex.Array, subrules_int: chex.Array, rule_reward: float, done: bool, random: bool,
               map_padding: int):
    map = map.astype(jnp.float32)
    subrules_int = subrules_int.astype(jnp.float32)

    # rule = rules.pop(0)
    # if rule in blocked_rules:
    #     continue
    n_rule_applications = 0
    # if not hasattr(rule, 'subrules'):
    #     print("Missing `rule.subrules`. Maybe you have not called `rule.compile`? You will need to do this manually" +
    #         "if the rule is not included in a ruleset.")
    # subrules = rule.subrules
    # breakpoint()
    # if random:
        # Apply rotations of base rule in a random order.
        # np.random.shuffle(subrules_int)
    # if not VMAP:
    #     out_map = np.zeros_like(map)
    #     for subrule_int in subrules_int:
    #         out_map_i, sr_activs = apply_subrule(map, subrule_int)
    #         out_map += out_map_i
    #         has_applied_rule = sr_activs.sum() > 0
    #         reward += reward * sr_activs.sum()
    #         done = done or np.any(sr_activs * done)
    #         # next_map += out_map
    # else:
    out_maps, sr_activs = jax.vmap(apply_subrule, (None, 0))(map, subrules_int)

    # For computing reward, zero out the right/bottom most columns/rows of the map
    # so that rule applications applied to the ``repeated'' edges are not counted
    # twice.
    sr_activs = sr_activs.at[:, :, :, :map_padding].set(0)
    sr_activs = sr_activs.at[:, :, :, -map_padding:].set(0)
    sr_activs = sr_activs.at[:, :, :, :, :map_padding].set(0)
    sr_activs = sr_activs.at[:, :, :, :, -map_padding:].set(0)

    out_map = out_maps.sum(axis=0)
    done = jnp.any(sr_activs * done)
    reward = rule_reward * sr_activs.sum()
    has_applied_rule = sr_activs.sum() > 0
    return out_map, done, reward, sr_activs, has_applied_rule


def apply_rules(map: np.ndarray, params: GenEnvParams, map_padding: int):
    """Apply rules to a one-hot encoded map state, to return a mutated map.

    Args:
        map (np.ndarray): A one-hot encoded map representing the game state.
        rules (List[Rule]): A list of rules for mutating the onehot-encoded map.
    """
    # Start a timer for the rule application.
    start = timer()

    # print(map)
    # rules = copy.copy(rules)
    # rules_set = set(rules)
    # print([r.name for r in rules])
    h, w = map.shape[1:]
    # map = map.astype(jnp.int8)
    map = toroidal_pad(map, map_padding)
    done = False
    reward = 0.0
    # These rules may become blocked when other rules are activated.
    blocked_rules = set({})

    # Add a batch channel to the map
    map = rearrange(map, 'c h w -> () c h w')
    next_map = map.copy()
    has_applied_rule = False
    done = False

    # subrules_ints = np.array([r.subrules_int for r in rules])
    subrules_ints = params.rules.rule
    rule_rewards = params.rules.reward
    dones = params.rule_dones

    # if not VMAP:
    #     for rule in rules:
    #     # while len(rules) > 0:
    #         # out_maps, dones, rewards, has_applied_rules = jax.vmap(apply_rule, (None, 0))(map, rules)
    #         out_map, r_done, r_reward, r_has_applied_rule = apply_rule(
    #             map, rule.subrules_int, rule.reward, rule.done)
    #         next_map += out_map
    #         done = done or r_done
    #         reward += r_reward
    #         has_applied_rule = has_applied_rule or r_has_applied_rule
    # else:
    out_maps, r_dones, r_rewards, r_sr_activs, r_has_applied_rules = jax.vmap(apply_rule, (None, 0, 0, 0, None, None))(
        map, subrules_ints, rule_rewards, dones, False, map_padding)
    # jax.debug.print('r_rewards {r_rewards}', r_rewards=r_rewards)
    next_map += out_maps.sum(axis=0)
    next_map = jnp.clip(next_map, 0, 1)
    done = jnp.any(r_dones)
    reward = r_rewards.sum()
    has_applied_rule = jnp.any(r_has_applied_rules)
                            
    next_map = jnp.array(next_map, dtype=jnp.int16)[0]
    # Remove padding.
    next_map = next_map[:, map_padding:-map_padding, map_padding:-map_padding]
    time_ms=(timer() - start) * 1000
    return next_map, reward, done, r_sr_activs, has_applied_rule, time_ms

def hash_rules(rules):
    """Hash a list of rules to a unique value.

    Args:
        rules (List[Rule]): A list of rules to hash.

    Returns:
        int: A unique hash value for the rules.
    """
    rule_hashes = [r.hashable() for r in rules]
    return hash(tuple(rule_hashes))

import pygame
def pygame_render_im(screen, img):
    surf = pygame.surfarray.make_surface(img)
    # Fill the background with white
    # screen.fill((255, 255, 255))
    screen.blit(surf, (0, 0))
    # Flip the display
    pygame.display.flip()


def toroidal_pad(map, pad_width=1):
    """
    Pads a 3D map (with channel as the first dimension) to make it toroidal (like in Pac-Man),
    wrapping the last two dimensions.
    
    Args:
    - map (jax.numpy.ndarray): A 3D array with shape (n_channels, height, width).
    - pad_width (int): The width of the padding.
    
    Returns:
    - jax.numpy.ndarray: The padded, toroidal map.
    """

    if map.ndim != 3 or map.shape[1] != map.shape[2]:
        raise ValueError("Input map must be a 3D array with the last two dimensions being square.")

    # Padding the map normally
    padded_map = jnp.pad(map, ((0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')

    # Wrapping the top to the bottom and the bottom to the top for each channel
    padded_map = padded_map.at[:, :pad_width, pad_width:-pad_width].set(map[:, -pad_width:, :])
    padded_map = padded_map.at[:, -pad_width:, pad_width:-pad_width].set(map[:, :pad_width, :])

    # Wrapping the left to the right and the right to the left for each channel
    padded_map = padded_map.at[:, pad_width:-pad_width, :pad_width].set(map[:, :, -pad_width:])
    padded_map = padded_map.at[:, pad_width:-pad_width, -pad_width:].set(map[:, :, :pad_width])

    # Handling the corners for each channel
    padded_map = padded_map.at[:, :pad_width, :pad_width].set(map[:, -pad_width:, -pad_width:])  # Top-left to bottom-right
    padded_map = padded_map.at[:, -pad_width:, :pad_width].set(map[:, :pad_width, -pad_width:])  # Bottom-left to top-right
    padded_map = padded_map.at[:, :pad_width, -pad_width:].set(map[:, -pad_width:, :pad_width])  # Top-right to bottom-left
    padded_map = padded_map.at[:, -pad_width:, -pad_width:].set(map[:, :pad_width, :pad_width])  # Bottom-right to top-left

    return padded_map