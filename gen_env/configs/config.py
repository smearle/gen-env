from dataclasses import dataclass
from typing import Optional, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from numpy import int32
from omegaconf import DictConfig


@dataclass
class EvoConfig:
    evo_seed: int = 0
    seed: int = 0
    domain_randomize: bool = False
    # map_width: int = 16
    map_shape: tuple = (16, 16)
    max_episode_steps: int = 100
    env_exp_id: int = 0
    player_exp_id: int = 0
    overwrite: bool = False
    n_proc: int = 1
    evo_pop_size: int = 40
    game: str = "blank_for_evo"
    mutate_rules: bool = True
    mutate_map: bool = True
    evaluate: bool = False
    eval_freq: int = -1
    save_freq: int = 1
    render: bool = False
    record: bool = True
    workspace: str = "saves"
    runs_dir_evo: str = "evo_env"
    runs_dir_rl: str = "rl_player"
    runs_dir_il: str = "il_player"
    # When collecting elite envs/playtraces from evolution, latest gen to include (if None, include all).
    load_gen: Optional[int] = None
    load_game: bool = False
    collect_elites: bool = False
    load_game: Optional[str] = None
    render_all: bool = False
    obs_rew_norm: bool = True
    obs_window: int = -1
    hide_rules: bool = False

    _log_dir_il: Optional[str] = None
    _log_dir_rl: Optional[str] = None
    _log_dir_evo: Optional[str] = None
    _log_dir_common: Optional[str] = None
    _log_dir_player_common: Optional[str] = None

    n_il_batches: float = 1_000_000

    load_il: bool = False
    load_rl: bool = False

    n_rl_iters: float = 1e6
    
    window_scale: float = 1.0

    activation: str = "relu"
    model: str = "conv"
    # Size of the receptive field to be fed to the action subnetwork.
    # vrf_size: Optional[int] = 31
    # Size of the receptive field to be fed to the value subnetwork.
    # arf_size: Optional[int] = 31
    # TODO: actually take arf and vrf into account in models, where possible
    act_shape: Tuple[int, int] = (1, 1)

    hidden_dims: Tuple[int, int] = (256, 256)

@dataclass
class ILConfig(EvoConfig):
    MAX_GRAD_NORM: float = 0.5
    il_seed: int = 0
    il_exp_name: str = "0"
    il_max_steps: int = 1_000_000
    il_batch_size: int = 4096
    il_tqdm: bool = True
    il_lr: float = 1.0e-4
    log_interval: int = 500
    eval_interval: int = 1000
    ckpt_interval: int = 1000
    render_freq: int = 1
    model: str = "conv"
    n_envs: int = 100
    obs_rew_norm: bool = True

    # In case we want to train only on a subset of the available training envs.
    n_train_envs: int = 100
    n_val_envs: int = 100

    _il_ckpt_dir: Optional[str] = None


@dataclass
class RLConfig(ILConfig):
    rl_seed: int = 0
    rl_exp_name: str = "0"
    lr: float = 1.0e-5
    num_steps: int = 200
    total_timesteps: int = int(1e9)
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    env_name: str = "xlife"
    ANNEAL_LR: bool = False
    DEBUG: bool = True
    exp_name: str = "0"
    problem: str = "binary"
    representation: str = "narrow"

    evo_freq: int = -1
    n_evo_gens: int = 1
    evo_save_freq: int = 10
    val_freq: int = 10
    n_envs: int = 1_000
    evo_pop_size: int = 10
    evo_mutate_prob: float = 0.1
    blank_env_start: bool = False

    render: bool = False
    n_render_eps: int = 1

    # change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.

    # static_tile_prob: Optional[float] = 0.0
    # n_freezies: int = 0
    # n_agents: int = 1
    # max_board_scans: float = 1.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 50

    """ DO NOT USE. WILL BE OVERWRITTEN. """
    _n_gpus: int = 1

    overwrite: bool = False

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1e6)
    render_freq: int = 100
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################


# @dataclass
# class TrainAccelConfig(TrainConfig):
#     evo_freq: int = 10
#     n_envs: int = 20
#     evo_pop_size: int = 10
#     evo_mutate_prob: float = 0.1


@dataclass
class EnjoyConfig(RLConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 10
    # Path to a .npz or .pkl file containing env params to render.
    envs_to_render: Optional[str] = None


# @dataclass
# class EnjoyAccelConfig(TrainAccelConfig, EnjoyConfig):
#     random_agent: bool = False
#     # How many episodes to render as gifs
#     n_eps: int = 10

@dataclass
class MapElitesConfig(EvoConfig):
    n_initial: int = 10
    n_gen: int = 1_000
    bins: int = 100
    metric: str = "lz"
    mutate_map: bool = True

    
@dataclass
class EvalConfig(RLConfig):
    reevaluate: bool = True
    random_agent: bool = False
    # In how many bins to divide up each control metric
    n_bins: int = 10
    n_envs: int = 200
    n_eps: int = 1


@dataclass
class ProfileEnvConfig(RLConfig):
    N_PROFILE_STEPS: int = 5000


@dataclass
class SweepConfig():
    algo: str = 'il'  # il, rl
    name: str = 'load_gen'
    mode: str = 'train'  # train, eval, plot, cross_eval
    slurm: bool = True
    skip_failures: bool = False


cs = ConfigStore.instance()
cs.store(name="base_config", node=EvoConfig)
# cs.store(name="rl_config_old", node=RLConfig)
cs.store(name="il_config", node=ILConfig)
# cs.store(name="train_xlife", node=TrainConfig)
cs.store(name="rl_config", node=RLConfig)
cs.store(name="me", node=MapElitesConfig)
cs.store(name="sweep_config", node=SweepConfig)
cs.store(name="enjoy_config", node=EnjoyConfig)
# cs.store(name="enjoy_accel_xlife", node=EnjoyAccelConfig)
cs.store(name="eval_xlife", node=EvalConfig)
cs.store(name="profile_xlife", node=ProfileEnvConfig)
