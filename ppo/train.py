from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import PPOTrainConfig
from .env import EnvConfig, HumanoidLocomotionEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on humanoid locomotion task")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--run-name", type=str, default="ppo_humanoid", help="Output prefix")
    parser.add_argument(
        "--target-speed",
        type=float,
        default=None,
        help="Override target forward speed in m/s",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override max steps per episode",
    )
    return parser.parse_args()


def make_env(cfg: PPOTrainConfig):
    def _factory():
        env_cfg = EnvConfig(
            max_episode_steps=cfg.max_episode_steps,
            target_speed_mps=cfg.target_speed_mps,
            max_torque=cfg.max_torque,
            reward=cfg.reward,
        )
        env = HumanoidLocomotionEnv(render_mode=None, config=env_cfg)
        return Monitor(env)

    return _factory


def ensure_dirs(*dirs: Path):
    for folder in dirs:
        folder.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    cfg = PPOTrainConfig()
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.target_speed is not None:
        cfg.target_speed_mps = args.target_speed
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps

    ensure_dirs(cfg.model_dir, cfg.log_dir, cfg.tensorboard_log_dir)

    env = DummyVecEnv([make_env(cfg)])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        tensorboard_log=str(cfg.tensorboard_log_dir),
        seed=cfg.seed,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.total_timesteps, tb_log_name=args.run_name)

    model_path = cfg.model_dir / f"{args.run_name}.zip"
    model.save(str(model_path))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
