from functools import partial
from pdb import set_trace as TT

import numpy as np

from gen_env.envs.play_env import GameDef
from gen_env.rules import Rule, RuleSet
from gen_env.tiles import TilePlacement, TileSet, TileType


def make_env():
    # force = TileType(name='force', num=0, color='purple')
    wall = TileType('wall', prob=0.1, color='brown')
    floor = TileType('floor', prob=0.9, color='black')
    player = TileType('player', num=1, color='blue', cooccurs=[floor])
    goal = TileType('goal', num=1, color='green', cooccurs=[floor])
    tiles = TileSet([player, goal, wall, floor])
    impassable_tiles = [wall]
    search_tiles = [floor, goal, player, wall]

    # player_move = Rule(
    #     'player_move', 
    #     in_out=np.array(  [# Both input patterns must be present to activate the rule.
    #         [
    #             [[player, floor]],  # Player next to a passable/floor tile.
    #             [[None, force]], # A force is active on said passable tile.
    #         ],
    #         # Both changes are applied to the relevant channels, given by the respective input subpatterns.
    #         [[[None, player]],  # Player moves to target. No change at source.
    #         [[None, None]],  # Force is removed from target tile.
    #         ],
    #     ]),
    #     rotate=True,)

    player_consume_goal = Rule(
        'player_consume_goal',
        in_out=np.array([
            [
                [[player]],  # Player and goal tile overlap.
                [[goal]],
            ],
            [
                [[player]],  # Player remains.
                [[None]],  # Goal is removed.
            ]
        ]),
        rotate=True,
        reward=1,
        done=True,
    )
    rules = RuleSet([player_consume_goal])
    # env = PlayEnv(height, width, tiles=tiles, rules=rules, player_placeable_tiles=[(force, TilePlacement.ADJACENT)],
    #     search_tiles=search_tiles)
    game_def = GameDef(
        tiles=tiles,
        rules=rules,
        player_placeable_tiles=[],
        search_tiles=search_tiles,
        impassable_tiles=impassable_tiles,
    )
    return game_def
