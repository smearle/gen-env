"""Rules determine a game's mechanics. They map input tile patterns to output patterns, rewards, done state (... and 
changes in global variables?).

During each step of the game engine, rules are applied in order, and the application of each rule involves scanning over
the tiles in the board in search of the relevant input pattern.

TODO: 
  - Add a `random_order` boolean value to rules that will result in scanning the board in a random order. THis will 
  allow, e.g., NPCs that move randomly on the board.
  - Re-implement some of MarkovJunior's level generation algos. Each game can optionally have a separate set of 
    map-generation rules so that we get, e.g., reasonable-looking mazes.


Implementation goal / design principle: compile rules as differential convolutional neural networks.

Assumptions to start:
- each rule will be applied in sequence. Whenever a rule is applied, we expect everything else on the map
to remain fixed. 
- two applications of the same rule will not overlap.
- don't worry about differentiability *too much* yet. E.g., can use step functions for now then scale things later to make it
  smooth/differentiable (?)

- weight of 2 from channel to itself at same (center) tile
- bias of -1 everywhere
- step function activations

  (This would any instances of A by 1 to the right at each tick.)
- [A, -] -> [-, A]: weight of 1 from channel to itself at adjacent tile
                    weight of -1 from channel to itself at center tile

  (i.e. player (A) moving toward user-input force (B))
- [A, B] -> [-, A]: weight of 1 from channel to itself at center tile (by default)
                    weight of 1/2 from A center to B right, 1/2 from B right to B right; and the -1/2 from both to A left  

- [A, -] -> [B, -]: weight of 2 from A center to B center
                    nothing else since empty tile in output does not replace anything
"""
from functools import partial
import math
from pdb import set_trace as TT
import random
from typing import Dict, Iterable

import chex
from einops import rearrange, reduce
from flax import struct
from jax import numpy as jnp
import jax
import numpy as np

from gen_env.objects import ObjectType
from gen_env.tiles import MJTile, TileSet, TileType


def tile_to_str(tile: TileType) -> str:
    return tile.name if tile is not None else tile

def tile_from_str(name: str, names_to_tiles: Dict[str, TileType]) -> TileType:
    return names_to_tiles[name] if not (name is None or name == 'None') else None

    
@struct.dataclass
class RuleData:
    rule: chex.Array
    reward: float


class Rule():
    """Rules match input to output patterns. When the environment ticks, the input patters are identified and replaced 
    by the output patterns. Each rule may consist of multiple sub-rules, any one of which can be applied. Each rule may
    consist of multiple subpatterns, which may all match in order for the subrule to be applied.
    Input and output are 2D patches of tiles.
    """
    def __init__(self, name: str, in_out: Iterable[TileType], rotate: bool = True, reward: int = 0, done: bool = False,
            random: bool = False, max_applications: int = np.inf, inhibits: Iterable = [], children: Iterable = [],
            application_funcs: Iterable = [], else_apply: Iterable = []):
        """Process the main subrule `in_out`, potentially rotating it to produce a set of subrules. Also convert 
        subrules from containing TileType objects to their indices for fast matching during simulation.

        Args:
            name (str): The rule's name. For human-readable purposes.
            in_out (Iterable[TileType]): A sub-rule with shape (2, n_subpatterns, h, w)
            rotate (bool): Whether subpatterns .
            else_apply (Iterable[Rule]): Rules to apply if this when cannot be applied.
        """
        self.name = name
        self._in_out = in_out
        self._rotate = rotate
        self.application_funcs = application_funcs
        self.children = children
        self.done = done
        self.inhibits = inhibits
        self.max_applications = max_applications
        self.random = random
        self.reward = reward
        self.else_apply = else_apply
        self.n_tile_types = None

    def from_dict(d, names_to_tiles):
        assert len(d) == 1
        name, d = list(d.items())[0]
        d['name'] = name
        in_out = np.stack((d['in_out']['in'], d['in_out']['out']), axis=0)
        _tile_from_str = partial(tile_from_str, names_to_tiles=names_to_tiles)
        in_out = np.vectorize(_tile_from_str)(in_out)
        d['in_out'] = in_out
        return Rule(**d)

    def to_dict(self):
        in_out = np.vectorize(tile_to_str)(self._in_out)
        inp, outp = in_out.tolist()
        # TODO: record application_funcs?
        return {
            self.name: {
                'in_out': {
                    'in': inp,
                    'out': outp,
                },
                'max_applications': self.max_applications,
                'rotate': self._rotate,
                'random': self.random,
                'reward': self.reward,
                'done': self.done,
                'inhibits': [t.name for t in self.inhibits],
                'children': [t.name for t in self.children],
            }
        }

    def observe(self, n_tiles):
        # FIXME: Need to observe reward, rotation, etc.
        # return np.array([self.done, self.reward / 3, self.max_applications / 11, self.random, self._rotate,])

        in_out_disc = np.vectorize(TileType.get_idx)(self._in_out)
        # in_out_disc = self.subrules_int

        # if in_out_disc.min() < 0:
        #     print('WARNING: negative tile index in rule observation. `TileNot` observations not supported. Hope you\'re not training on this!')
        in_out_disc = np.clip(in_out_disc, 0, n_tiles) 

        in_out_onehot = np.eye(n_tiles + 1)[in_out_disc]
        return in_out_onehot.flatten()

    def mutate_new(self, tiles, other_rules):
        # TODO: User GenEnv actions to modify map and rules (need to factor this out of GenEnv to be a static method)
        pass


    def hashable(self):
        int_in_out = np.vectorize(lambda x: x.get_idx() if x is not None else -1)(self._in_out)
        return hash(int_in_out.tobytes())


def gen_rand_rule(rng, like_rules: RuleData) -> chex.Array:
    # By default we generate rotation-invariant rules
    # Hackish: getting shape of (pre-rotation) rules by selecting a single rotation
    # (n_rules, n_rotations, in_out, n_tiles, height, width)
    rand_rule = jax.random.randint(rng, like_rules[:, 0:1, :, :, :, :].shape, minval=0, maxval=2, dtype=like_rules.dtype)

    # Cannot have any other tiles active in the input patch
    # rand_rule = rand_rule.at[:, :, 0].set(
    #     jnp.where(rand_rule[:, :, 0] == 0, -1, rand_rule[:, :, 0])
    # )

    # Restrict to one row/column
    rand_rule = rand_rule.at[:, :, :, :, 1:].set(0)

    # Remove input tiles unless otherwise specified
    rand_rule = rand_rule.at[:,:,1].set(rand_rule[:, :, 1] - rand_rule[:, :, 0])

    # rand_rule = rand_rule.at[:, :, 1, 0].set(rand_rule[:, :, 0, 0])

    # Ensure the player cannot be active in the rule TODO: do not hardcode player index
    rand_rule = rand_rule.at[:, :, :, 0].set(0)
    # Repeat rotations
    rule_rot_90 = jnp.rot90(rand_rule, k=1, axes=(-2, -1))
    rule_rot_180 = jnp.rot90(rand_rule, k=2, axes=(-2, -1))
    rule_rot_270 = jnp.rot90(rand_rule, k=3, axes=(-2, -1))
    rand_rule = jnp.concatenate([rand_rule, rule_rot_90, rule_rot_180, rule_rot_270], axis=1)
    return rand_rule


def mutate_rules(key, rules: RuleData):
    rule = rules.rule
    reward = rules.reward
    new_reward = jax.random.randint(key, shape=reward.shape, minval=-1.0, maxval=2.0)
    reward_mask = jax.random.bernoulli(key, p=0.3, shape=reward.shape)
    reward = jnp.where(reward_mask, new_reward, reward)

    pct_rules_to_mutate = jax.random.uniform(key, shape=(), minval=0.0, maxval=0.5)
    rule_mask = jax.random.bernoulli(key, p=pct_rules_to_mutate, shape=rule.shape)
    pct_tiles_to_mutate = jax.random.uniform(key, shape=(), minval=0.0, maxval=0.5)
    tile_mask = jax.random.bernoulli(key, p=pct_tiles_to_mutate, shape=rule.shape)
    tile_mask = tile_mask * rule_mask
    new_rule = gen_rand_rule(key, like_rules=rules.rule)
    rule = jnp.where(tile_mask, new_rule, rule)
    return RuleData(rule=rule, reward=reward)

def compile_rule(rule: Rule):
    """Generate a list of subrules resulting from rule (i.e. if applying rotation).
    """

    # (in_out, subpatterns, height, width)
    in_out = rule._in_out
    subrules = [in_out]
    # Only rotate if necessary.
    if rule._rotate and rule._in_out[0].shape != (1, 1):
        subrules += [np.rot90(in_out, k=1, axes=(2, 3)), np.rot90(in_out, k=2, axes=(2,3)), 
            np.rot90(in_out, k=3, axes=(2,3))]
    max_subrule_shape = np.array([sr.shape for sr in subrules]).max(axis=0)
    rule.subrules = subrules
    rule.subrules_int = [np.vectorize(TileType.get_idx)(sr) for sr in subrules]
    rule.subrules_int = \
        [np.pad(sr, [(0, max_subrule_shape[i] - sr.shape[i]) for i in range(len(max_subrule_shape))], constant_values=-1)
        for sr in rule.subrules_int]
    # Now shape: (n_subrules, in_out, rule_channels, height, width)

    # To convert to onehot, make Nones (-1) equal to 0. We'll give them their
    # own onehot channel for now (then remove it later).
    rule.subrules_int = (np.array(rule.subrules_int) + 1).astype(np.int8)

    # Take one-hot over channels
    # (n_subrules, in_out, rule_channels, height, width, tile_channels)
    rule.subrules_int = np.eye(rule.n_tile_types + 1)[rule.subrules_int][..., 1:]
    rule.subrules_int = rearrange(rule.subrules_int, 'n_subrules in_out rule_channels height width tile_channels -> ' +
                                    'n_subrules in_out rule_channels tile_channels height width')
    # Sum over rule channels.
    # NOTE: This assumes the rule int array is binary.
    rule.subrules_int = reduce(rule.subrules_int,
                                'n_subrules in_out rule_channels tile_channels height width -> ' +
                                'n_subrules in_out tile_channels height width', 'sum').astype(np.int8)
    
    # We may have redundant tiles in different rule channels in input or output pattern, so clip
    rule.subrules_int = np.clip(rule.subrules_int, 0, 1)

    # Always remove whatever was detected in the input pattern when
    # it is activated.
    # outp -= inp
    rule.subrules_int[:, 1] -= rule.subrules_int[:, 0]

    # Get max shape over subrules
    max_shape = np.array([sr.shape for sr in subrules]).max(axis=0)
    # Pad subrules to max shape
    padded_subrules = []
    # for sr in subrules:
    #     padded_sr_shape = [(0, max_shape[i] - sr.shape[i]) for i in range(len(max_shape))]
    #     padded_subrule = [np.pad(sr, padded_sr_shape) for sr in self.subrules_int]
    #     padded_subrules.append(padded_subrule)



def is_valid(in_out):
    """Accept new in-out rule only if it does not result in invalid player transformation.
    A rule is allowed to move or remove the player, but not create new players where previously
    there were none."""
    in_out_players = in_out == 0
    # return in_out_players.sum() == 0 or in_out_players[0].sum() == 1 and in_out_players[1].sum() <= 1
    return in_out_players.sum() == 0 | in_out_players[0].sum() == 1 & in_out_players[1].sum() == 0

class MJRule(Rule):
    def __init__(self, rule):
        i, o = rule.split('=')
        i, o = np.array(list(i)), np.array(list(o))
        in_out = rearrange([i, o], "io x -> io () () x")
        super().__init__(name=f"{i}_{o}", in_out=in_out,
            rotate=True, random=True)

class ObjectRule(Rule):
    def __init__(self, *args, offset=(0, 0), **kwargs):
        super().__init__(*args, **kwargs)
        # Expect an object in the first input subpattern at the offset coordinates.
        assert isinstance(self._in_out[0, 0, offset[0], offset[0]], ObjectType.GameObject)

    def compile(self):
        super().compile()

class RuleSet(list):
    def __init__(self, rules: Iterable[Rule]):
        super().__init__(rules)
        # [rule.compile() for rule in rules]


class MJRuleNode(RuleSet):
    def __init__(self, rules: Iterable[str]):
        for i, r in enumerate(rules):
            rules[i] = MJRule(r)
        super().__init__(rules)
        symbols = set([s for r in rules for s in r._in_out.flatten()])
        tile_dict = {}
        for s in symbols:
            num, prob = 0, 0
            if s == "B":
                prob = 1.0
            if s == "R":
                num = 1
            tile_dict[s] = MJTile(s, num=num, prob=prob)
        self.tiles = TileSet(list(tile_dict.values()))
        for r in rules:
            r._in_out = np.vectorize(tile_dict.get)(r._in_out)
            r.compile()

