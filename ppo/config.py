from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RewardWeights:
    """Weights for reward components in the locomotion task."""

    forward_velocity: float = 1.0
    survival: float = 0.2
    energy: float = 0.002
    fall_penalty: float = 40.0


@dataclass(slots=True)
class PPOTrainConfig:
    """Train-time configuration for PPO."""

    total_timesteps: int = 500_000
    target_speed_mps: float = 1.25
    max_episode_steps: int = 1000
    max_torque: float = 100.0
    model_dir: Path = Path("artifacts/models")
    log_dir: Path = Path("artifacts/logs")
    tensorboard_log_dir: Path = Path("artifacts/tensorboard")

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42

    reward: RewardWeights = field(default_factory=RewardWeights)
