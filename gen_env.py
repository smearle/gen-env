import copy
from dataclasses import dataclass
from enum import Enum
import random
import time
from typing import Dict, Iterable, List, Tuple

import cv2
from einops import rearrange, repeat
import gym
from gym import spaces
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pygame

from events import Event, EventGraph
from objects import ObjectType
from rules import Rule
from tiles import TileNot, TilePlacement, TileType
from utils import draw_triangle
from variables import Variable


@dataclass
class GenEnvState:
    n_step: int
    map_arr: np.ndarray
    obj_set: Iterable


class GenEnv(gym.Env):
    placement_positions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
    tile_size = 16
    view_size = 3
    
    def __init__(self, width: int, height: int,
            tiles: Iterable[TileType], 
            rules: List[Rule], 
            player_placeable_tiles: List[Tuple[TileType, TilePlacement]], 
            object_types: List[ObjectType] = [],
            search_tiles: List[TileType] = None,
            events: Iterable[Event] = [],
            variables: Iterable[Variable] = [],
            done_at_reward: int = None,
            max_episode_steps: int = 100
        ):
        """_summary_

        Args:
            width (int): The width of the 2D game map.
            height (int): The height of the 2D game map.
            tiles (list): A list of TileType objects. Must include a tile-type with name `player`.
            rules (list): A list of Rule objects, between TileTypes.
            done_at_reward (int): Defaults to None. Otherwise, episode ends when reward reaches this number.
        """
        self._ep_rew = 0
        self._done = False
        if search_tiles is None:
            self._search_tiles = tiles
        else:
            self._search_tiles = search_tiles
        self._search_tile_idxs = np.array([tile.idx for tile in self._search_tiles])
        self.event_graph = EventGraph(events)
        self.n_step = 0
        self.max_episode_steps = max_episode_steps
        self.w, self.h = width, height
        self.tiles = tiles
        # [setattr(tile, 'idx', i) for i, tile in enumerate(tiles)]
        tiles_by_name = {t.name: t for t in tiles}
        # Assuming here that we always have player and floor...
        self.player_idx = tiles_by_name['player'].idx
        self.tile_probs = [tile.prob for tile in tiles]
        # Add white for background when rendering individual tile-channel images.
        self.tile_colors = np.array([tile.color for tile in tiles] + [[255,255,255]], dtype=np.uint8)
        self._init_rules = rules
        self.rules = copy.copy(rules)
        self.map: np.ndarray = None
        self.objects: Iterable[ObjectType.GameObject] = []
        # self.observation_space = spaces.Box(0, 1, (self.w, self.h, len(self.tiles)))
        # Dictionary observation space containing box 2d map and flat list of rules
        len_rule_obs = len(self.rules[0].observe())
        self.observation_space = spaces.Dict({
            'map': spaces.Box(0, 1, (self.w, self.h, len(self.tiles))),
            'rules': spaces.Box(0, 1, (len_rule_obs * len(self.rules),))
        })
        self.player_pos: Tuple[int] = None
        self.player_force_arr: np.ndarray = None
        # No rotation
        # self.action_space = spaces.Discrete(4)
        # Rotation
        self.action_space = spaces.Discrete(3)
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
        self._map_id = 0

    def queue_maps(self, maps: Iterable[np.ndarray]):
        self._map_queue = maps

    def _update_player_pos(self, map_arr):
        self.player_pos = np.argwhere(map_arr[self.player_idx] == 1)
        if self.player_pos.shape[0] < 1:
            self.player_pos = None
            return
            # TT()
        if self.player_pos.shape[0] > 1:
            raise Exception("More than one player on map.")
        # assert self.player_pos.shape[0] == 1
        self.player_pos = tuple(self.player_pos[0])
        
    def _update_cooccurs(self, map_arr: np.ndarray):
        for tile_type in self.tiles:
            if tile_type.cooccurs:
                for cooccur in tile_type.cooccurs:
                    map_arr[cooccur.idx, map_arr[tile_type.idx] == 1] = 1

    def _update_inhibits(self, map_arr: np.ndarray):
        for tile_type in self.tiles:
            if tile_type.inhibits:
                for inhibit in tile_type.inhibits:
                    map_arr[inhibit.idx, map_arr[tile_type.idx] == 1] = 0

    def gen_random_map(self):
        # Generate frequency-based tiles with certain probabilities.
        int_map = np.random.choice(len(self.tiles), size=(self.w, self.h), p=self.tile_probs)
        map_coords = np.argwhere(int_map != -1)
        # Overwrite frequency-based tiles with tile-types that require fixed numbers of instances.
        n_fixed = sum([tile.num for tile in self.tiles if tile.num is not None])
        fixed_coords = map_coords[np.random.choice(map_coords.shape[0], size=n_fixed, replace=False)]
        i = 0
        for tile in self.tiles:
            if tile.prob == 0 and tile.num is not None:
                coord_list = fixed_coords[i: i + tile.num]
                int_map[coord_list[:, 0], coord_list[:, 1]] = tile.idx
                i += tile.num
        map_arr = np.eye(len(self.tiles), dtype=np.uint8)[int_map]
        map_arr = rearrange(map_arr, "h w c -> c h w")
        self._update_player_pos(map_arr)
        # Activate parent/co-occuring tiles.
        for tile in self.tiles:
            coactive_tiles = tile.parents + tile.cooccurs
            if len(coactive_tiles) > 0:
                for cotile in coactive_tiles:
                    # Activate parent channels of any child tiles wherever the latter are active.
                    map_arr[cotile.idx, map_arr[tile.idx] == 1] = 1
        return map_arr

    def reset(self):
        self._ep_rew = 0
        # Reset rules.
        self.rules = copy.copy(self._init_rules)
        # Reset variables.
        [v.reset() for v in self.variables]
        self.event_graph.reset()
        self.n_step = 0
        self._last_reward = 0
        self._reward = 0
        if len(self._map_queue) == 0:
            map_arr = self.gen_random_map()
        else:
            map_arr = self._map_queue[self._map_id]
            self._map_id = (self._map_id + 1) % len(self._map_queue)
        self._set_state(GenEnvState(n_step=self.n_step, map_arr=map_arr, obj_set={}))
        self.player_pos = np.argwhere(map_arr[self.player_idx] == 1)[0]
        self.player_rot = 0
        self._rot_dirs = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)])
        obs = self.get_obs()
        return obs

    def step(self, action):
        # TODO: Only pass global variable object to event graph.
        self.event_graph.tick(self)
        self.act(action)
        reward = self.tick()
        self.n_step += 1
        if self._done:
            pass
            # print('done at step')
        self._ep_rew += reward
        return self.get_obs(), reward, self._done, {}

    def act(self, action):
        if self.player_pos is None:
            return
        # if action >= len(self._actions):
        #     return
        if action < 2:
            rot = self.player_rot + (1 if action == 0 else -1)
            self.player_rot = rot % 4
            return
        # No rotation
        # new_tile, placement_id = self._actions[action]
        # pos = self.player_pos + self.placement_positions[placement_id]

        # Rotation
        new_tile = self._actions[action - 2]
        pos = self.player_pos + self._rot_dirs[self.player_rot]

        # Do not place anything over the edge of the map. Should we wrap by default instead?
        if np.any(pos < 0) or pos[0] >= self.w or pos[1] >= self.h:
            return
        self.map[new_tile, pos[0], pos[1]] = 1

    def get_obs(self):
        # return self.observe_map()
        return {
            'map': self.observe_map(),
            'rules': self.observe_rules(),
            'player_rot': np.eye(4)[self.player_rot].astype(np.float32),
        }

    def observe_map(self):
        obs = rearrange(self.map, 'b h w -> h w b')
        # Crop map to player's view.
        if self.player_pos is not None:
            x, y = self.player_pos
            obs = obs[x - self.view_size: x + self.view_size + 1,
                      y - self.view_size: y + self.view_size + 1]
        return obs.astype(np.float32)

    def observe_rules(self):
        return np.concatenate([r.observe() for r in self.rules])
    
    def render(self, mode='human'):
        font = ImageFont.load_default()
        tile_size = self.tile_size
        # self.rend_im = np.zeros_like(self.int_map)
        # Create an int map where the last tiles in `self.tiles` take priority.
        int_map = np.zeros(self.map[0].shape, dtype=np.uint8)
        tile_ims = []
        for tile in self.tiles:
            # if tile.color is not None:
            int_map[self.map[tile.idx] == 1] = tile.idx
            tile_map = np.where(self.map[tile.idx] == 1, tile.idx, -1)
            # If this is the player, render as a triangle according to its rotation
            if tile.name == 'player':
                tile_im = np.zeros_like(self.tile_colors[tile_map]) + 255
                tile_im = repeat(tile_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
                tile_im = np.pad(tile_im, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
                tile_im = draw_triangle(tile_im, self.player_pos, self.player_rot, tile.color, tile_size)
            else: 
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
        # Flat render of all tiles
        tile_im = self.tile_colors[int_map]
        tile_im = repeat(tile_im, f'h w c -> (h {tile_size}) (w {tile_size}) c')
        tile_im = np.pad(tile_im, ((30, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        tile_ims = [tile_im] + tile_ims

        map_h, map_w = tile_ims[0].shape[:-1]

        # Add empty images to the end of the tile images to fill out the grid
        n_tiles = len(tile_ims)
        # Find the smallest square number greater than or equal to n_tiles
        n_tiles_sqrt = int(np.ceil(np.sqrt(n_tiles)))
        n_tiles_sqrt2 = n_tiles_sqrt ** 2
        n_empty_tiles = n_tiles_sqrt2 - n_tiles
        empty_tile_im = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        empty_tile_ims = [empty_tile_im] * n_empty_tiles
        tile_ims += empty_tile_ims
        # Reshape the tile images into a grid
        tile_ims = np.array(tile_ims)
        tile_ims = tile_ims.reshape(n_tiles_sqrt, n_tiles_sqrt, map_h, map_w, 3)
        # Add padding between tiles
        pw = 2
        tile_ims = np.pad(tile_ims, ((0, 0), (0, 0), (pw, pw), (pw, pw), (0, 0)), mode='constant', constant_values=0)
        # Concatenate the tile images into a single image
        tile_ims = rearrange(tile_ims, 'n1 n2 h w c -> (n1 h) (n2 w) c')
        # Below the image, add a row of text showing episode/cumulative reward
        # Add padding below the image
        tile_ims = np.pad(tile_ims, ((0, 30), (0, 0), (0, 0)), mode='constant', constant_values=0)
        text = f'Reward: {self._ep_rew}'
        # Paste text
        img_pil = Image.fromarray(tile_ims)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), text, font=font, fill=(255, 255, 255, 0))
        tile_ims = np.array(img_pil)
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
            # Map human-input keys to action indices. Here we assume the first 4 actions correspond to player navigation 
            # (i.e. placement of `force` at adjacent tiles).
            self.keys_to_acts = {
                pygame.K_LEFT: 0,
                pygame.K_RIGHT: 1,
                pygame.K_UP: 2,
                # pygame.K_DOWN: 3,
                # pygame.K_q: 4,
            }
            self.rend_im = np.flip(self.rend_im, axis=0)
            # Rotate to match pygame
            self.rend_im = np.rot90(self.rend_im, k=-1)
            if self.screen is None:
                pygame.init()
                # Flip image to match pygame coordinate system
                # Set up the drawing window to match size of rend_im
                self.screen = pygame.display.set_mode([self.rend_im.shape[0], self.rend_im.shape[1]])
                # self.screen = pygame.display.set_mode([(len(self.tiles)+1)*self.h*GenEnv.tile_size, self.w*GenEnv.tile_size])
            pygame_render_im(self.screen, self.rend_im)
            return
        else:
            cv2.imshow('Generated Environment', self.rend_im)
            cv2.waitKey(1)
            # return self.rend_im

    def tick(self):
        self._last_reward = self._reward
        for obj in self.objects:
            obj.tick(self)
        self.map, self._reward, self._done = apply_rules(self.map, self.rules)
        if self._done_at_reward is not None:
            self._done = self._done or self._reward == self._done_at_reward
        # self._done = self._done or self.n_step >= self.max_episode_steps
        if not self._done:
            self._compile_map()
        return self._reward

    def _remove_additional_players(self):
        # Remove additional players
        player_pos = np.argwhere(self.map[self.player_idx] == 1)
        if player_pos.shape[0] > 1:
            for i in range(1, player_pos.shape[0]):
                # Set to random other tile
                self.map[self.player_idx, player_pos[i][0], player_pos[i][1]] = 0
                rand_tile_type = random.randint(0, len(self.tiles) - 2)
                rand_tile_type = rand_tile_type if rand_tile_type < self.player_idx else rand_tile_type + 1
                self.map[rand_tile_type, player_pos[i][0], player_pos[i][1]] = 1
        elif player_pos.shape[0] == 0:
            # Get random x y position
            x, y = np.random.randint(0, self.map.shape[1]), np.random.randint(0, self.map.shape[2])
            # Set random tile to be player
            self.map[self.player_idx, x, y] = 1

    def _compile_map(self):
        self._remove_additional_players()
        self._update_player_pos(self.map)
        self._update_cooccurs(self.map)
        self._update_inhibits(self.map)

    def tick_human(self):
        import pygame
        done = False
        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame.KEYDOWN:
                if event.key in self.keys_to_acts:
                    action = self.keys_to_acts[event.key]
                    obs, rew, done, info = self.step(action)

                    self.render(mode='pygame')
                    # if self._last_reward != self._reward:
                    print(f"Step: {self.n_step}, Reward: {self._reward}")
                elif event.key == pygame.K_x:
                    done = True
            if done:
                self.reset()
                done = False
                self.render(mode='pygame')

    def get_state(self):
        return GenEnvState(n_step=self.n_step, map_arr=self.map.copy(), obj_set=self.objects)

    def set_state(self, state: GenEnvState):
        state = copy.deepcopy(state)
        self._set_state(state)
        # TODO: setting variables and event graph.

    def hashable(self, state):
        # assert hash(state['map_arr'].tobytes()) == hash(state['map_arr'].tobytes())
        search_state = state.map_arr[self._search_tile_idxs]
        # search_state = state['map_arr']
        return hash(search_state.data.tobytes())


    def _set_state(self, state: GenEnvState):
        map_arr, obj_set = state.map_arr, state.obj_set
        self.map = map_arr
        self.objects = obj_set
        self.height, self.width = self.map.shape[1:]
        self._compile_map()


def apply_rules(map: np.ndarray, rules: List[Rule]):
    """Apply rules to a one-hot encoded map state, to return a mutated map.

    Args:
        map (np.ndarray): A one-hot encoded map representing the game state.
        rules (List[Rule]): A list of rules for mutating the onehot-encoded map.
    """
    # print(map)
    rules = copy.copy(rules)
    rules_set = set(rules)
    # print([r.name for r in rules])
    next_map = map.copy()
    done = False
    reward = 0
    h, w = map.shape[1:]
    # These rules may become blocked when other rules are activated.
    blocked_rules = set({})
    for rule in rules:
        if rule in blocked_rules:
            continue
        n_rule_applications = 0
        if not hasattr(rule, 'subrules'):
            print("Missing `rule.subrules`. Maybe you have not called `rule.compile`? You will need to do this manually" +
                "if the rule is not included in a ruleset.")
        subrules = rule.subrules
        if rule.random:
            # Apply rotations of base rule in a random order.
            np.random.shuffle(subrules)
        for subrule in rule.subrules:
            # Apply, e.g., rotations of the base rule
            inp, out = subrule
            xys = np.indices((h, w))
            xys = rearrange(xys, 'xy h w -> (h w) xy')
            if rule.random:
                np.random.shuffle(xys)
                # print(f'y: {y}')
            for (x, y) in xys:
                match = True
                for subp in inp:
                    if subp.shape[0] + x > h or subp.shape[1] + y > w:
                        match = False
                        break
                    if not match:
                        break
                    for i in range(subp.shape[0]):
                        if not match:
                            break
                        for j in range(subp.shape[1]):
                            tile = subp[i, j]
                            if tile is None:
                                continue
                            if map[tile.get_idx(), x + i, y + j] != tile.trg_val:
                                match = False
                                break
                if match:
                    # print(f'matched rule {rule.name} at {x}, {y}')
                    # print(f'rule has input \n{inp}\n and output \n{out}')
                    [f() for f in rule.application_funcs]
                    [blocked_rules.add(r) for r in rule.inhibits]
                    for r in rule.children:
                        if r in rules_set:
                            continue
                        rules_set.add(r)
                        rules.append(r)
                    reward += rule.reward
                    done = done or rule.done
                    for k, subp in enumerate(out):
                        for i in range(subp.shape[0]):
                            for j in range(subp.shape[1]):
                                # Remove the corresponding tile in the input pattern if one exists.
                                in_tile = inp[k, i, j]
                                if in_tile is not None:
                                    # Note that this has no effect when in_tile is a NotTile.
                                    next_map[in_tile.get_idx(), x + i, y + j] = 0
                                out_tile = subp[i, j]
                                if out_tile is None:
                                    continue
                                next_map[out_tile.get_idx(), x + i, y + j] = 1
                    n_rule_applications += 1
                    if n_rule_applications >= rule.max_applications:
                        # print(f'Rule {rule.name} exceeded max applications')
                        break
                
            else:
                continue

            # Will break the subrule loop if we have broken the board-scanning loop.
            break
                        
                    
    return next_map, reward, done


def pygame_render_im(screen, img):
    surf = pygame.surfarray.make_surface(img)
    # Fill the background with white
    # screen.fill((255, 255, 255))
    screen.blit(surf, (0, 0))
    # Flip the display
    pygame.display.flip()
