from functools import partial
import os
import pickle
import random
from typing import Any, Dict, Tuple

import zlib
import hydra
import imageio
import jax
from jax import numpy as jnp
import numpy as np

from gen_env.configs.config import EvoConfig, MapElitesConfig
from gen_env.envs.play_env import GameDef, GenEnvParams, GenEnvState
from gen_env.evo.individual import mutate, mutate_params
from gen_env.games import GAMES
from gen_env.utils import gen_rand_env_params, init_base_env, init_config
from utils import pad_frames, stack_leaves


def step_env_random(carry, _, env):
    key, obs, state, env_params = carry
    key = carry[0]
    key, _ = jax.random.split(key)
    action = jax.random.randint(key, 1, 0, 6)
    obs, state, reward, done, info, env_params_idx = env.step(key, state, action, env_params, env_params) 
    return (key, obs, state, env_params), state

def eval_random_ep(key, params, env):
    _step_env_random = partial(step_env_random, env=env)
    obs, state = env.reset(key, params) 
    (_, _, _, _), states = jax.lax.scan(_step_env_random, (key, obs, state, params), None, env.max_episode_steps)
    return states

def eval_random(key, params, env, n_eps=100):
    key = jax.random.split(key, n_eps)
    return jax.vmap(
        eval_random_ep,
        in_axes=(0, None, None)
    )(
        key, params, env
    )

def ca_to_bytes(ca_states):
    """
    ca_states is assumed to be a 2D array (time x space) of 0/1 for a 1D CA,
    or (time x height x width) of 0/1 for a 2D CA.
    This function flattens and packs bits into bytes.
    """
    # Flatten everything into one dimension of bits
    flat_bits = ca_states.reshape(-1).astype(np.uint8)
    # Pack bits into bytes. np.packbits will create 1 byte per 8 bits.
    packed = np.packbits(flat_bits)
    return packed.tobytes()

def compression_ratio(sequence_bytes):
    compressed = zlib.compress(sequence_bytes)
    ratio = len(compressed) / len(sequence_bytes)
    return ratio

def measure_lz_complexity(ca_states):
    data = ca_to_bytes(ca_states)
    return compression_ratio(data)


def shannon_entropy(binary_array):
    """
    binary_array: 1D array of bits {0,1}
    Returns the Shannon entropy in bits.
    """
    counts = np.bincount(binary_array, minlength=2)
    p = counts / np.sum(counts)
    # Avoid log2(0)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def ca_shannon_entropy(ca_states):
    """
    Flatten the CA states and compute Shannon entropy
    of the distribution of 0s and 1s.
    """
    flat_bits = ca_states.reshape(-1)
    return shannon_entropy(flat_bits)

def render_archive(cfg, save_dir, archive_path):
    env, base_params = init_base_env(cfg)
    with open(archive_path, 'rb') as f:
        archive = pickle.load(f)
    
    random.shuffle(archive)
    
    params_lst = [a[1] for a in archive if a is not None]
    descs_lst = [a[2] for a in archive if a is not None]
    fits_lst = [a[0] for a in archive if a is not None]
    params = stack_leaves(params_lst)
    key = jax.random.PRNGKey(0)
    states = jax.vmap(eval_random, in_axes=(None, 0, None, None))(
        key,
        params,
        env,
        1,
    )

    # Get rid of n_episode per params dimension
    states = jax.tree.map(lambda x: x[:, 0], states)

    states = jax.device_put(states, jax.devices('cpu')[0])
    params = jax.device_put(params, jax.devices('cpu')[0])

    for bin_i in range(states.ep_rew.shape[0]):
        print(f"Saving gif for bin {bin_i}")
        ep_frames = []
        params_i = jax.tree_map(lambda x: x[bin_i], params)
        for step_i in range(states.ep_rew.shape[1]):
            fit_i = fits_lst[bin_i]
            desc_i = descs_lst[bin_i]
            state_i = jax.tree.map(lambda x: x[bin_i, step_i], states)
            frame = env.render(state_i, params_i, mode='rgb_array')
            ep_frames.append(frame)
        frames = pad_frames(ep_frames)
        gif_name=os.path.join(save_dir, f"desc-{desc_i:.4f}_fit-{fit_i:.4f}.gif")
        imageio.v3.imwrite(
            gif_name,
            frames,
            duration=100,
            loop=0,
        )

@hydra.main(version_base=None, config_path='gen_env/configs', config_name='me')
def main(cfg: MapElitesConfig):
    """
    Turn the previous random evaluation into a 1D MAP-Elites run.
    We'll assume you specify in cfg:
      - cfg.n_initial
      - cfg.n_gen
      - cfg.bins
      - cfg.metric  (one of ["shannon", "lz"])
      - etc.
    """
    init_config(cfg)
    save_dir = os.path.join(cfg._log_dir_common, 'evo_compressibility')
    os.makedirs(save_dir, exist_ok=True)
    archive_path = os.path.join(save_dir, 'archive.npz')

    if cfg.evaluate:
        return render_archive(cfg, save_dir, archive_path)

    # 1) Initialize environment
    env, base_params = init_base_env(cfg)

    # 2) Decide how many bins we use for our 1D descriptor
    n_bins = cfg.bins
    # We store in each bin: (fitness, genotype, descriptor)
    # Initialize the archive with None
    archive = [None] * n_bins


    def descriptor_to_bin(descriptor_val: float) -> int:
        """
        Convert the descriptor value (entropy or LZ) to
        an integer bin index [0, n_bins - 1].
        We assume the descriptor is in [0,1]. If not, you might
        rescale or clamp as needed.
        """
        idx = int(descriptor_val * n_bins)
        # Make sure index is in range
        return min(max(idx, 0), n_bins-1)

    def evaluate_solution(params) -> Tuple[float, float]:
        """
        Evaluate the genotype in the environment.
        Return (fitness, descriptor).
        """
        states = eval_random(key, params, env, n_eps=5)
        vals = []
        for i in range(states.map.shape[0]):
            states_i = jax.tree.map(lambda x: x[i], states)
            ca_states = states_i.map
            # Behavior descriptor
            if cfg.metric == "lz":
                descriptor_val = measure_lz_complexity(ca_states)
            else:
                descriptor_val = ca_shannon_entropy(ca_states)
            vals.append(descriptor_val)
        mean_val = np.mean(vals)
        # std_val = np.std(val)

        fitness = jnp.std(states.ep_rew)

        return fitness, mean_val

    # --- 3) Initialization: Generate random solutions and fill the archive ---
    key = jax.random.PRNGKey(0)
    game_def: GameDef = GAMES[cfg.game].make_env()

    if os.path.isfile(archive_path):
        with open(archive_path, 'rb') as f:
            archive = pickle.load(f)

    else:
        for _ in range(cfg.n_initial):
            key, _ = jax.random.split(key)
            params = gen_rand_env_params(cfg, key, base_params, game_def)
            fit, desc = evaluate_solution(params)
            b = descriptor_to_bin(desc)

            # If this bin is empty or we found a better fitness, update
            if archive[b] is None or fit > archive[b][0]:
                archive[b] = (fit, params, desc)

    # --- 4) Main MAP-Elites loop ---
    for gen in range(cfg.n_gen):
        key, _ = jax.random.split(key)
        # a) Randomly select an existing elite
        existing_bins = [i for i, item in enumerate(archive) if item is not None]
        if not existing_bins:
            # If archive is empty, skip or re-initialize
            continue
        parent_bin = random.choice(existing_bins)
        parent_fitness, parent_params, parent_desc = archive[parent_bin]

        # b) Create variant via mutation
        child_params = mutate_params(cfg, key, parent_params)

        # c) Evaluate
        child_fitness, child_desc = evaluate_solution(child_params)
        print(f"child fitness: {child_fitness}, desc: {child_desc}")

        # d) Place in the correct bin if better
        child_bin = descriptor_to_bin(child_desc)
        if archive[child_bin] is None or child_fitness > archive[child_bin][0]:
            archive[child_bin] = (child_fitness, child_params, child_desc)

        # Optional: print some info or track stats
        if gen % 1 == 0:
            print(f"Generation {gen}: filled bins = "
                  f"{sum([1 for x in archive if x is not None])}")
        
        if gen % 1 == 0:
            with open(archive_path, 'wb') as f:
                pickle.dump(archive, f)

    # --- 5) Post-processing or results ---
    # Example: Print out final coverage and best bins
    filled_bins = [(i, x) for i, x in enumerate(archive) if x is not None]
    print(f"Filled {len(filled_bins)} / {n_bins} bins.")
    for b_idx, (fit, g, d) in filled_bins:
        print(f"Bin {b_idx}, fitness={fit:.3f}, descriptor={d:.3f}")

    

if __name__ == "__main__":
    main()