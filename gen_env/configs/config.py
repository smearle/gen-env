from dataclasses import dataclass
from typing import Optional, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from numpy import int32
from omegaconf import DictConfig


@dataclass
class GenEnvConfig:
    env_exp_id: int = 0
    player_exp_id: int = 0
    overwrite: bool = False
    n_proc: int = 4
    evo_batch_size: int = 40
    game: str = "blank_for_evo"
    mutate_rules: bool = True
    fix_map: bool = False
    evaluate: bool = False
    eval_freq: int = 1
    save_freq: int = 1
    render: bool = False
    record: bool = True
    workspace: str = "../gen-game-runs"
    runs_dir_evo: str = "evo_env"
    runs_dir_rl: str = "rl_player"
    runs_dir_il: str = "il_player"
    load_gen: Optional[int] = None
    collect_elites: bool = False
    load_game: Optional[str] = None

    _log_dir_il: Optional[str] = None
    _log_dir_rl: Optional[str] = None
    _log_dir_evo: Optional[str] = None
    _log_dir_common: Optional[str] = None
    _log_dir_player_common: Optional[str] = None

    n_il_batches: float = 1_000_000

    load_il: bool = False
    load_rl: bool = False

    n_rl_iters: float = 1e6

    hide_rules: bool = False
    
    map_shape: tuple = (10, 10)

    window_shape: tuple = (800, 800)


@dataclass
class RLConfig(GenEnvConfig):
    lr: float = 1.0e-4
    n_envs: int = 4
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "xlife"
    ANNEAL_LR: bool = False
    DEBUG: bool = True
    exp_name: str = "0"
    seed: int = 0
    problem: str = "binary"
    representation: str = "narrow"
    model: str = "conv"

    map_width: int = 16
    is_3d: bool = False
    # ctrl_metrics: Tuple[str] = ('diameter', 'n_regions')
    ctrl_metrics: Tuple[str] = ()
    # Size of the receptive field to be fed to the action subnetwork.
    vrf_size: Optional[int] = 31
    # Size of the receptive field to be fed to the value subnetwork.
    arf_size: Optional[int] = 31
    # TODO: actually take arf and vrf into account in models, where possible

    change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.
    act_shape: Tuple[int, int] = (1, 1)

    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1
    max_board_scans: float = 1.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 25

    """ DO NOT USE. WILL BE OVERWRITTEN. """
    exp_dir: Optional[str] = None
    n_gpus: int = 1

@dataclass
class TrainConfig(RLConfig):
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

@dataclass
class TrainAccelConfig(TrainConfig):
    evo_freq: int = 10
    evo_pop_size: int = 10
    evo_mutate_prob: float = 0.1


@dataclass
class EnjoyConfig(RLConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 10

    
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
class BatchConfig(TrainConfig):
    mode: str = 'train'
    slurm: bool = True


cs = ConfigStore.instance()
cs.store(name="base_config", node=GenEnvConfig)
cs.store(name="rl_config", node=RLConfig)
cs.store(name="train_xlife", node=TrainConfig)
cs.store(name="train_accel_xlife", node=TrainAccelConfig)
cs.store(name="enjoy_xlife", node=EnjoyConfig)
cs.store(name="eval_xlife", node=EvalConfig)
cs.store(name="profile_xlife", node=ProfileEnvConfig)
cs.store(name="batch_xlife", node=BatchConfig)
