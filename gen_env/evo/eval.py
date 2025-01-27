from timeit import default_timer as timer

import jax
from jax import numpy as jnp
import numpy as np

from gen_env.envs.play_env import GenEnvParams, PlayEnv
from gen_env.evo.individual import IndividualData
from search_agent import batched_bfs, bfs, bfs_multi_env

def evaluate_multi(args):
    return evaluate(*args)

def evaluate_DUMMY(env: PlayEnv, individual: IndividualData, render: bool, trg_n_iter: bool):
    n_players_in_rules = 0
    for rule in individual.rules:
        n_players_in_rules += np.sum(np.vectorize(lambda t: t is not None and t.is_player)(rule._in_out[1])) - \
            np.sum(np.vectorize(lambda t: t is not None and t.is_player)(rule._in_out[0]))
    individual.fitness = n_players_in_rules
    individual.action_seq = []
    return individual

def evaluate_multi(key: jax.random.PRNGKey, env: PlayEnv, env_params: GenEnvParams, trg_n_iter: int, n_envs: int):
    n_envs = env_params.map.shape[0]
    # obs, init_state = env.reset(key=key, params=env_params)
    _, init_state = jax.vmap(env.reset, in_axes=(0, 0))(jax.random.split(key, n_envs), env_params)
    best_action_seqs, best_rewards, n_iter_bests, n_iter = \
        bfs_multi_env(env, init_state, env_params, max_steps=trg_n_iter, n_best_to_keep=1)
    best_state_actions, best_reward, n_iter_best = best_action_seqs[0], best_rewards[0], n_iter_bests[0]
    action_seq = None
    # if best_state_actions is not None:
    #     (final_state, action_seq) = best_state_actions
    # TODO: dummy
    # fitness = best_reward
    fitnesses = n_iter_bests
    # if fitness == 1:
    #     fitnesses += n_iter / (trg_n_iter + 2)
    fitnesses = jnp.where(fitnesses == 1, fitnesses + n_iter / (trg_n_iter + 2), fitnesses)
    # fitness = len(action_seq) if action_seq is not None else 0
    # env_params.fitness = fitness
    # env_params.action_seq = action_seq
    print(f"Achieved fitnesses:\n{fitnesses.tolist()}\nat\n{n_iter_bests}\niterations with\n{best_rewards}\nreward. Searched for {n_iter} iterations total.")
    return fitnesses, best_action_seqs

def evaluate(key: jax.random.PRNGKey, env: PlayEnv, env_params: GenEnvParams, render: bool, trg_n_iter: int):
    start_time = timer()
    # load_game_to_env(env, env_params)
    # env.queue_games([env_params.map.copy()], [env_params.rules.copy()])
    # params = get_params_from_individual(env, individual)
    obs, init_state = env.reset(key=key, params=env_params)
    # Save the map after it having been cleaned up by the environment
    # individual.map = env.map.copy()
    # assert individual.map[4].sum() == 0, "Extra force tile!" # Specific to maze tiles only
    best_state_actionss, best_rewards, n_iter_bests, n_iter = \
        bfs(env, init_state, env_params, max_steps=trg_n_iter, n_best_to_keep=1)
    best_state_actions, best_reward, n_iter_best = best_state_actionss[0], best_rewards[0], n_iter_bests[0]
    action_seq = None
    if best_state_actions is not None:
        (final_state, action_seq) = best_state_actions
        if render:
            env.render(state=state)
            for action in action_seq:
                state, obs, reward, done, info = env.step(action, state)
                env.render()
    action_seqs = [e[1] for e in best_state_actionss if e is not None]
    # TODO: dummy
    # fitness = best_reward
    fitnesses = np.array(n_iter_bests)
    # if fitness == 1:
    #     fitnesses += n_iter / (trg_n_iter + 2)
    np.where(fitnesses == 1, fitnesses + n_iter / (trg_n_iter + 2), fitnesses)
    # fitness = len(action_seq) if action_seq is not None else 0
    # env_params.fitness = fitness
    # env_params.action_seq = action_seq
    fps = n_iter / (timer() - start_time)
    print(f"Achieved fitnesses {fitnesses.tolist()} at {n_iter_bests} iterations with {best_rewards} reward. Searched for {n_iter} iterations total. FPS: {fps}")
    return fitnesses, action_seqs