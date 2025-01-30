import os
from pathlib import Path
import pickle
import hydra
import jax

from gen_env.configs.config import RLConfig
from gen_env.evo.individual import hash_env
from gen_env.utils import init_config
from utils import init_il_config, init_rl_config, load_elite_envs, stack_leaves

test_dirs = [
    ("evosearch", "/scratch/se2161/autoverse/saves/blank_for_evo_mutRule_mutMap_s-1_exp-0"),
    # ("complexity", "/scratch/se2161/autoverse/saves/blank_for_evo_mutRule_mutMap_s-0_exp-0/evo_compressibility"),
]
test_gen = 25
save_dir = "test_envs"

def save_test_set(test_name, test_dir, train_elites, test_save_dir):
    """Save all test examples that are unique from the training set."""
    test_elites_a, test_elites_b = load_elite_envs(test_dir, test_gen)

    # Store hashes of everything that might be in the training set
    train_elite_hashes = set({})
    for i in range(train_elites.fitness.shape[0]):
        train_elite_i = jax.tree.map(lambda x: x[i], train_elites)
        env_params = train_elite_i.env_params
        e_hash = hash_env(env_params)
        train_elite_hashes.add(e_hash)

    # Now filter our test set so that it excludes envs which collide with the training set
    test_elites = {}
    for i in range(test_elites_a.fitness.shape[0]):
        test_elite_i = jax.tree.map(lambda x: x[i], test_elites_a)
        env_params = test_elite_i.env_params
        e_hash = hash_env(env_params)
        if e_hash not in train_elite_hashes:
            test_elites[e_hash] = test_elite_i
            
    print(f"Collected {len(test_elites)} unique envs for testing from {test_elites_a.fitness.shape[0]} total.")
    assert len(test_elites) > 0, "Found no unique elites w.r.t. training set!"
    test_elites = stack_leaves(list(test_elites.values()))
    test_elites_path = os.path.join(test_save_dir, 'test_envs.pkl')
    os.makedirs(test_save_dir, exist_ok=True)
    print(f"Saving test set to {test_elites_path}")

    with open(test_elites_path, 'wb') as f:
        pickle.dump(test_elites, f)

@hydra.main(config_path="gen_env/configs", config_name="rl", version_base="1.3")
def main(cfg: RLConfig):
    """For testing generalization of RL agents, we filter elites from some test set, keeping anything
    that is not also present in some training set"""
    init_config(cfg)
    latest_gen = init_il_config(cfg)
    init_rl_config(cfg, latest_gen)
    test_save_dir = os.path.join(Path(cfg._log_dir_evo).parent, save_dir)

    # Assume we have set `cfg.load_gen` to be the latest generation of environment evolution whose archive
    # we might be training on.
    train_elites, val_elites = load_elite_envs(cfg._log_dir_common, latest_gen)

    for test_name, test_dir in test_dirs:
        save_test_set(test_name, test_dir, train_elites, test_save_dir)


if __name__ == '__main__':
    main()