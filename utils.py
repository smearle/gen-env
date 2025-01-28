from functools import partial
import glob
import os
from typing import Tuple

import jax
from jax import numpy as jnp
import numpy as np

from gen_env.configs.config import EvoConfig, ILConfig, RLConfig
from gen_env.envs.play_env import GenEnvParams, PlayEnv
from gen_env.evo.individual import IndividualPlaytraceData
from purejaxrl.wrappers import LogEnvState

def pad_frames(frames):
    frame_shapes = [frame.shape for frame in frames]
    max_frame_w, max_frame_h = max(frame_shapes, key=lambda x: x[0])[0], \
        max(frame_shapes, key=lambda x: x[1])[1]
    # Pad frames to be same size
    new_ep_frames = []
    for frame in frames:
        frame = np.pad(frame, ((0, max_frame_w - frame.shape[0]),
                                  (0, max_frame_h - frame.shape[1]),
                                  (0, 0)), constant_values=0)
        new_ep_frames.append(frame)
    return new_ep_frames
        
# Function to stack leaves of PyTrees
def stack_leaves(trees):
    # Make sure each leaf is an array
    def to_array(x):
        x = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
        if x.shape == ():
            x = x.reshape(1)
        return x

    trees = [jax.tree.map(lambda x: to_array(x), tree) for tree in trees]

    # Flatten each tree
    flat_trees_treedefs = [jax.tree.flatten(tree) for tree in trees]
    flat_trees, treedefs = zip(*flat_trees_treedefs)

 
    # Concatenate the flattened lists
    concatenated_leaves = [jnp.stack(leaves) for leaves in zip(*flat_trees)]

    # Rebuild PyTree
    return jax.tree.unflatten(treedefs[0], concatenated_leaves)


def concatenate_leaves(trees):
    # Flatten each tree
    flat_trees_treedefs = [jax.tree.flatten(tree) for tree in trees]
    flat_trees, treedefs = zip(*flat_trees_treedefs)
 
    # Concatenate the flattened lists
    concatenated_leaves = [jnp.concatenate(leaves, axis=0) for leaves in zip(*flat_trees)]

    # Rebuild PyTree
    return jax.tree.unflatten(treedefs[0], concatenated_leaves)

def load_elite_envs(dir, latest_gen) -> Tuple[IndividualPlaytraceData]:
    # elites = np.load(os.path.join(cfg.log_dir_evo, "unique_elites.npz"), allow_pickle=True)['arr_0']
    # train_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.npz"), allow_pickle=True)['arr_0']
    # val_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.npz"), allow_pickle=True)['arr_0']
    # test_elites = np.load(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.npz"), allow_pickle=True)['arr_0']
    # load with pickle instead
    import pickle
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_train_elites.pkl"), 'rb') as f:
    with open(os.path.join(dir, f"gen-{latest_gen}_filtered_train_elites.pkl"), 'rb') as f:
        train_elites = pickle.load(f)
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_val_elites.pkl"), 'rb') as f:
    with open(os.path.join(dir, f"gen-{latest_gen}_filtered_val_elites.pkl"), 'rb') as f:
        val_elites = pickle.load(f)
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_test_elites.pkl"), 'rb') as f:
    # with open(os.path.join(cfg._log_dir_common, f"gen-{latest_gen}_filtered_test_elites.pkl"), 'rb') as f:
    #     test_elites = pickle.load(f)
    
    # Assign unique ids to each elite
    elites = []
    for e in [train_elites, val_elites]:
        e: IndividualPlaytraceData
        n_elites = e.env_params.rule_dones.shape[0]
        old_env_params_dict = e.env_params.__dict__
        old_env_params_dict.pop('env_idx')
        new_env_params = GenEnvParams(
            **e.env_params.__dict__,
            env_idx=jnp.arange(n_elites),
            # impassable_tiles=jnp.zeros((n_elites, e.env_params.map.shape[1],), dtype=bool),
        )
        e = e.replace(env_params=new_env_params)
        elites.append(e)

    # Sort train elites by fitness (seems to be already sorted but... just in case!)
    train_elites = elites[0]
    fit_idxs = jnp.argsort(train_elites.fitness, axis=0)
    train_elites = jax.tree_map(lambda x: x[fit_idxs[:, 0]], train_elites)
    elites[0] = train_elites

    return elites

        
def get_rand_train_envs(train_env_params: GenEnvParams, n_envs: int, rng: jax.random.PRNGKey, replace=False):
    """From available `train_env_params, randomly sample `n_envs` many environments. First, tile `train_env_params` if 
    there are not enough."""
    n_avail_train_envs = train_env_params.rule_dones.shape[0]
    n_reps = int(np.ceil(n_envs / n_avail_train_envs))
    train_env_params = jax.tree.map(lambda x: jnp.repeat(x, n_reps, axis=0), train_env_params)
    n_avail_train_envs = train_env_params.rule_dones.shape[0]
    rand_idxs = jax.random.choice(rng, n_avail_train_envs, (n_envs,), replace=replace)
    return jax.tree.map(lambda x: x[rand_idxs], train_env_params)


def eval_log_callback(metric, writer, t, mode, params_type):
    writer.add_scalar(f"{mode}/eval/{params_type}_ep_return", metric['mean_return'], t)
    writer.add_scalar(f"{mode}/eval/{params_type}_ep_return_max", metric['max_return'], t)
    writer.add_scalar(f"{mode}/eval/{params_type}_ep_return_min", metric['min_return'], t)


def evaluate_on_env_params(rng: jax.random.PRNGKey, cfg: RLConfig, env: PlayEnv, env_params: GenEnvParams,
                           network_apply_fn, network_params, update_i: int, writer, n_eps: int = 1, mode: str = "rl",
                           params_type: str = "val", search_rewards=None):

    def step_env(carry, _):
        env_state: LogEnvState
        rng, obs, env_state, curr_env_param_idxs, network_params = carry
        rng, _rng = jax.random.split(rng)

        next_params = get_rand_train_envs(env_params, cfg.n_envs, _rng)
        curr_env_params = jax.tree.map(lambda x: x[curr_env_param_idxs], env_params)

        pi, value = network_apply_fn(network_params, obs)
        action_r = pi.sample(seed=rng)
        rng_step = jax.random.split(_rng, cfg.n_envs)

        # rng_step_r = rng_step_r.reshape((config.n_gpus, -1) + rng_step_r.shape[1:])
        vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, 0, 0))
        # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
        obs, env_state, reward_r, done_r, info_r, curr_env_param_idxs = vmap_step_fn(
                        rng_step, env_state, action_r,
                        curr_env_params, next_params)
        
        return (rng, obs, env_state, curr_env_param_idxs, network_params),\
            (env_state, reward_r, done_r, info_r, value)

    eval_rng = jax.random.split(rng, cfg.n_envs)

    curr_env_params = get_rand_train_envs(env_params, cfg.n_envs, rng, replace=True)
    search_rewards = None if search_rewards is None else \
        jax.tree.map(lambda x: x[curr_env_params.env_idx], search_rewards)
    rng = jax.random.split(rng)[0]

    obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(
            eval_rng, curr_env_params
    )

    _, (states, rewards, dones, infos, values) = jax.lax.scan(
        step_env, (rng, obsv, env_state, curr_env_params.env_idx, network_params),
        None, n_eps*env.max_episode_steps)

    returns = rewards.sum(axis=0)

    _eval_log_callback = partial(eval_log_callback, writer=writer, mode=mode, params_type=params_type)

    metric = {
        'mean_return': returns.mean(),
        'max_return': returns.max(),
        'min_return': returns.min(),
    }

    jax.experimental.io_callback(_eval_log_callback, None, metric=metric,
    t=update_i)

    
def init_il_config(cfg: ILConfig):
    if cfg.load_gen == -1 or cfg.load_gen == None:
        # glob files of form `gen-XX*elites.npz` and get highest gen number
        gen_files = glob.glob(os.path.join(cfg._log_dir_common, "gen-*_elites.pkl"))
        gen_nums = [int(os.path.basename(f).split("_")[0].split("-")[1]) for f in gen_files]
        if len(gen_nums) == 0:
            raise FileExistsError(f"No elite files found in {cfg._log_dir_common}")
        latest_gen = max(gen_nums)

    else:
        latest_gen = cfg.load_gen

    cfg._log_dir_il = os.path.join(
        cfg._log_dir_il,
        f"_env-evo-gen-{latest_gen}_" + \
        f"lr-{cfg.il_lr}_" + \
        ('hideRules_' if cfg.hide_rules else '') + \
        f'obs_win-{cfg.obs_window}_' + \
        (f'noObsRewNorm_' if not cfg.obs_rew_norm else '') + \
        f"s-{cfg.il_seed}_" + \
        f"{cfg.il_exp_name}")
    cfg._il_ckpt_dir = os.path.abspath(os.path.join(cfg._log_dir_il, "ckpt"))

    return latest_gen


def init_rl_config(cfg: RLConfig, latest_evo_gen: int):
    cfg._n_gpus = jax.local_device_count()
    if cfg.load_il:
        il_ckpt_files = glob.glob(os.path.join(cfg._il_ckpt_dir, "[0-9]*"))
        update_steps = [os.path.basename(f) for f in il_ckpt_files]
        update_steps = [int(us) for us in update_steps if us.isnumeric()]
        latest_il_update_step = max(update_steps)
        # We will load the IL agent with the corresponding seed.
        cfg.il_seed = cfg.rl_seed
    else:
        latest_il_update_step = None
    cfg._log_dir_rl = os.path.join(cfg._log_dir_rl, 
        f'evogen-{cfg.load_gen}_accel-{cfg.evo_freq}_' + \
        (f'nGens-{cfg.n_evo_gens}_' if cfg.n_evo_gens != 1 else '') + \
        f'ilstep-{latest_il_update_step}_' + \
        (f'envs-{cfg.n_envs}_' if cfg.n_envs != 100 else '') + \
        f'tenvs-{cfg.n_train_envs}_' + \
        ('hideRules_' if cfg.hide_rules else '') + \
        f'obs_win-{cfg.obs_window}_' + \
        (f'noObsRewNorm_' if not cfg.obs_rew_norm else '') + \
        f's-{cfg.rl_seed}_{cfg.rl_exp_name}')
    return latest_il_update_step


def save_train_env_params(train_env_params, update_i, log_dir):
    save_path = os.path.join(log_dir, f"train_env_params_update_{update_i}")
    jnp.savez(save_path, train_env_params)
    # with open(save_path, "wb") as f:
    #     pickle.dump(train_env_params, f)
    print(f"Saved train_env_params to {save_path}")
