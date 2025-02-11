from gen_env.games import (
    blank_for_evo,
    dungeon,
    evo_base,
    hamilton, 
    lava_maze,
    loop_erased_walk,
    maze, 
    maze_for_evo,
    maze_for_evo_2,
    maze_backtracker, 
    maze_growth, 
    maze_npc, 
    maze_spike, 
    power_line, 
    rush_hour,
    sokoban,
    test_1x1_rules,
    )

GAMES = {
    'blank_for_evo': blank_for_evo,
    'dungeon': dungeon,
    'evo_base': evo_base,
    'hamilton': hamilton,
    'lava_maze': lava_maze,
    'maze': maze,
    'maze_for_evo': maze_for_evo,
    'maze_for_evo_2': maze_for_evo_2,
    'maze_backtracker': maze_backtracker,
    'maze_growth': maze_growth,
    'maze_npc': maze_npc,
    'maze_spike': maze_spike,
    'power_line': power_line,
    # 'rush_hour': rush_hour,
    'sokoban': sokoban,
    'test_1x1_rules': test_1x1_rules,
}

def make_env_rllib(env_config, make_env_func):
    return make_env_func(**env_config)
