from enum import IntEnum
from math import inf

import numpy as np

from gen_env.envs.play_env import GameDef, PlayEnv
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TileNot, TilePlacement, TileSet, TileType


def make_env():
    n_crates = 3
    force = TileType(name='force', prob=0, color='purple')
    # passable = TileType(name='passable', prob=0, color=None)
    # floor = TileType('floor', prob=0.8, color='white', parents=[passable])
    wall = TileType('wall', prob=0.2, color='black')
    floor = TileType('floor', prob=0.8, color='grey')
    player = TileType('player', prob=0, color='blue', num=1, cooccurs=[floor])
    crate = TileType('crate', num=n_crates, color='brown')  # Not passable.
    target = TileType('target', num=n_crates, color='green', cooccurs=[floor])


    tiles = TileSet([player, force, crate, target, wall, floor])
    maps = None
    maps = np.array([
        # [
        #     [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        #     [4, 0, 5, 5, 5, 5, 5, 5, 5, 4],
        #     [4, 4, 4, 4, 5, 5, 5, 5, 5, 4],
        #     [4, 5, 5, 5, 5, 5, 5, 5, 5, 4],
        #     [4, 5, 5, 5, 5, 4, 5, 5, 5, 4],
        #     [4, 5, 5, 4, 2, 4, 5, 5, 5, 4],
        #     [4, 5, 5, 5, 5, 4, 5, 5, 5, 4],
        #     [4, 5, 5, 5, 5, 5, 5, 4, 4, 4],
        #     [4, 5, 5, 5, 5, 5, 5, 5, 3, 4],
        #     [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        # ],
        [
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 4],
            [4, 5, 5, 4, 5, 5, 5, 4, 4, 4, 4],
            [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4],
            [4, 0, 5, 5, 4, 2, 5, 5, 4, 5, 4],
            [4, 4, 4, 5, 4, 5, 5, 5, 4, 5, 4],
            [4, 4, 5, 5, 4, 4, 4, 4, 4, 5, 4],
            [4, 4, 5, 5, 3, 5, 5, 5, 5, 5, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
    ])
    maps = maps[0]

    impassable_tiles = [wall, crate]
    search_tiles = [floor, wall, target, crate, player]

    done_at_reward = n_crates
    player_move = Rule(
        'player_move', 
        in_out=np.array(  [# Both input patterns must be present to activate the rule.
            [[[None, player, floor]],  # Player next to a passable/floor tile.
            [[None, None, force]], # A force is active on said passable tile.
            ],
            # Both changes are applied to the relevant channels, given by the respective input subpatterns.
            [[[None, None, player]],  # Player moves to target. No change at source.
            [[None, None, floor]],  # Force is removed from target tile.
            ],
        ]),
        rotate=True,
        max_applications=inf,
    )
    crate_on_target = Rule(
        'crate_on_target',
        in_out=np.array([
            [
                [[None, None, crate]],
                [[None, None, target]],
                # [[TileNot(force)]]  # Otherwise we can push a clone-crate off a target.
            ],
            [  # Kill target.
                [[None, None, wall]],
                [[None, None, None]],
                # [[None]],
            ]
        ]),
        # max_applications=inf,
        # FIXME: SHould be able to make this false
        rotate=True,
        reward=1,
        max_applications=inf,
    )
    # This prevents player from passing through crates when they are on targets (and against a wall).
    crate_kill_force = Rule(
        'crate_kill_force',
        in_out=np.array([
            [
                [[None, crate, None]],
                [[None, force, None]],
            ],
            [
                [[None, crate, None]],
                [[None, crate, None]],
            ]
        ]),
        max_applications=inf,
        rotate=True,
    )
    player_push_crate = Rule(
        'player_push_crate',
        in_out=np.array([
            [
                [[player, crate, floor]],  # Player next to force, then floor
                [[None, force, None]],  # Crate on same tile as force.
            ],
            [
                [[None, player, crate]],  # Player moves, force removed, crate overwrites floor.
                [[None, floor, None]],  # No effect.
            ]
        ]),
        max_applications=inf,
        # max_applications=1,
        rotate=True,
        reward=0,
        inhibits=[player_move, crate_kill_force],
    )
    # Order is important for movement/physics.
    rules = RuleSet([player_push_crate, crate_kill_force, player_move, crate_on_target])

    # return PlayEnv(10, 10, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)], 
    #     done_at_reward=n_crates, search_tiles=search_tiles)

    game_def = GameDef(
        tiles=tiles,
        rules=rules,
        player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
        done_at_reward=n_crates,
        search_tiles=search_tiles,
        impassable_tiles=impassable_tiles,
        map=maps,
    )
    return game_def
