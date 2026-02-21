from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for a ReDo DQN agent."""

    # Experiment settings
    exp_name: str = "ReDo DQN"
    tags: tuple[str, ...] | str | None = None
    seed: int = 0
    torch_deterministic: bool = True
    gpu: int | None = None
    track: bool = True
    wandb_project_name: str = "ML Projekt"
    wandb_entity: str | None = None
    capture_video: bool = False
    save_model: bool = False
    log_txt: bool = False
    log_txt_dir: str = "runs"
    log_txt_filename: str = "metrics.txt"

    # Environment settings
    env_id: str = "LunarLander-v3"
    env_preset: str = "auto"
    total_timesteps: int = 1_000_000  # total timesteps to train for
    num_envs: int = 1

    # DQN settings
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_rate: float = 1e-3
    adam_eps: float = 1.5 * 1e-4
    use_lecun_init: bool = False  # ReDO uses lecun_normal initializer, cleanRL uses the pytorch default (kaiming_uniform)
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1_000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.995
    learning_starts: int = 20000  # cleanRL default: 80000, theirs 20000, we use 20k for better benchmarking on the 1M timesteps setting
    train_frequency: int = 4  # cleanRL default: 4, theirs 1

    # ReDo settings
    enable_redo: bool = True
    redo_tau: float = 0.025  # 0.025 for default, else 0.1
    redo_tau_start: float | None = 0.05
    redo_tau_end: float | None = 0.01
    redo_tau_fraction: float = 1.0 # Fraction of training to anneal tau over (if redo_tau_start and redo_tau_end are set)
    redo_check_interval: int = 1000
    redo_bs: int = 64

    # BatchNorm settings
    use_batch_norm: bool = False
    bn_eps: float = 1e-5
    bn_momentum: float = 0.1

    # NaP (Normalize-and-Project) settings
    enable_nap: bool = False
    nap_norm_type: str = "layer"  # "layer" or "rms"
    nap_eps: float = 1e-8
    nap_affine: bool = True # If True, uses affine parameters in NaP layers
    nap_remove_bias: bool = False # If True, removes bias terms from layers with NaP
    nap_project_interval: int = 1
    nap_project_eps: float = 1e-12

    # Evaluation settings
    eval_interval: int = 50_000
    eval_episodes: int = 5
    eval_seed: int = 123
    eval_epsilon: float = 0.05
