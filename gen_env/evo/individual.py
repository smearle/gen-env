import random
from typing import Dict, Iterable, List, Optional
import yaml

import chex
from einops import rearrange
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.envs.play_env import GenEnvParams, PlayEnv
from gen_env.configs.config import EvoConfig
from gen_env.rules import Rule, RuleData, RuleSet, mutate_rules
from gen_env.tiles import TileType, TileSet


@struct.dataclass
class IndividualData:
    env_params: GenEnvParams
    fitness: Iterable[float]
    # bc_0: float
    # bc_1: float
    action_seq: Iterable[chex.Array]
    # obs_seq: chex.Array


# Will contain only the best playtrace, for offline learning
@struct.dataclass
class IndividualPlaytraceData:
    env_params: GenEnvParams
    fitness: float
    action_seq: chex.Array
    obs_seq: chex.Array
    rew_seq: chex.Array
    done_seq: chex.Array

def hash_env(env_params):
    hashable_env = jax.tree_map(lambda x: np.array(x).tobytes(), env_params)
    flat_hashable_env: List = jax.tree_leaves(hashable_env)
    return hash(frozenset(flat_hashable_env))
    
def hash_individual(individual: IndividualData) -> int:
    hashable_elite = jax.tree_map(lambda x: np.array(x).tobytes(), individual)
    # Return flattened leaf nodes of pytree
    flat_hashable_elite: List = jax.tree_leaves(hashable_elite)
    return hash(frozenset(flat_hashable_elite))

def mutate_params(cfg, key, params: GenEnvParams):
    map, rules = mutate(cfg, key, params.map, params.rules)
    return params.replace(map=map, rules=rules)

def mutate(cfg, key, map, rules):
    rules = jax.lax.cond(cfg.mutate_rules, lambda key, rules: mutate_rules(key, rules), lambda _, __: rules, key, rules)
    map = jax.lax.cond(cfg.mutate_map, lambda key, map: mutate_map(key, map), lambda _, __: map, key, map)
    return map, rules

def mutate_map(key, map):
    flip_pct = jax.random.uniform(key, shape=(), minval=0.0, maxval=0.5)
    bit_flips = jax.random.bernoulli(key, p=flip_pct, shape=map.shape)

    # Mask out bit flips at `player_idx`        
    bit_flips = bit_flips.at[0].set(0)
    map = map.astype(jnp.int16)
    map = jnp.bitwise_xor(map, bit_flips)
    return map

class Individual():
    def __init__(self, cfg: EvoConfig, tiles: Iterable[TileType]):
        self.cfg = cfg
        # self.tiles = tiles
        # self.rules = rules
        # for rule in self.rules:
        #     rule.n_tile_types = len(self.tiles)
        #     rule.compile()
        # self.map = map
        self.fitness = None

        self.obs_seq = None
        self.action_seq = None
        self.reward_seq = None

        self.player_idx = None
        for t in tiles:
            if t.is_player:
                self.player_idx = t.idx
        assert self.player_idx == 0

    def mutate(self, key, map, rules, tiles):
        return mutate(self.cfg, key, map, rules)



    # TODO: bring this up to speed
    # def save(self, filename):
    #     # Save dictionary to yaml
    #     with open(filename, 'w') as f:
    #         d = {'tiles': [t.to_dict() for t in self.tiles], 'rules': [r.to_dict() for r in self.rules],
    #              'map': self.map.tolist()}
    #         yaml.safe_dump(d, f, indent=4, allow_unicode=False)

    def load(filename, cfg):
        # Load dictionary from yaml
        with open(filename, 'r') as f:
            d = yaml.safe_load(f)
            tiles = []
            for t_dict in d['tiles']:
                assert len(t_dict) == 1
                name = list(t_dict.keys())[0]
                t_dict = t_dict[name]
                t_dict.update({'name': name})
                tiles.append(TileType.from_dict(t_dict))
            tiles = TileSet(tiles)
            names_to_tiles = {t.name: t for t in tiles}
            rules = [Rule.from_dict(r, names_to_tiles=names_to_tiles) for r in d['rules']]
            for t in tiles:
                t.cooccurs = [names_to_tiles[c] for c in t.cooccurs]
                t.inhibits = [names_to_tiles[i] for i in t.inhibits]
            names_to_rules = {r.name: r for r in rules}
            for r in rules:
                r.children = [names_to_rules[c] for c in r.children]
                r.inhibits = [names_to_rules[i] for i in r.inhibits]
            rules = RuleSet(rules)
            map = np.array(d['map'])
        return IndividualData(tiles=tiles, rules=rules, cfg=cfg, map=map)

    # TODO: bring this up to speed
    # def hashable(self):
    #     rule_hashes = [r.hashable() for r in self.rules]
    #     rules_hash = hash((tuple(rule_hashes)))
    #     map_hash = hash(self.map.tobytes())
    #     return hash((rules_hash, map_hash))
