from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .env import EnvConfig, HumanoidLocomotionEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO locomotion policy")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved PPO model zip")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--target-speed", type=float, default=1.25, help="Override target speed")
    parser.add_argument("--max-episode-steps", type=int, default=1000, help="Episode step limit")
    parser.add_argument("--max-torque", type=float, default=100.0, help="Maximum torque")
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> HumanoidLocomotionEnv:
    env_cfg = EnvConfig(
        max_episode_steps=args.max_episode_steps,
        target_speed_mps=args.target_speed,
        max_torque=args.max_torque,
    )
    return HumanoidLocomotionEnv(render_mode=None, config=env_cfg)


def evaluate_episode(model: PPO, env: HumanoidLocomotionEnv, deterministic: bool) -> dict:
    obs, info = env.reset()
    done = False

    total_reward = 0.0
    steps = 0
    initial_x = float(obs[0])

    x_velocities: list[float] = []
    final_info: dict = {}

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        done = bool(terminated or truncated)

        x_velocities.append(float(info.get("base_lin_vel_x", 0.0)))
        final_info = info

    final_x = float(obs[0])
    forward_distance = final_x - initial_x
    avg_forward_velocity = float(np.mean(x_velocities)) if x_velocities else 0.0

    return {
        "episode_reward": total_reward,
        "episode_steps": steps,
        "forward_distance": forward_distance,
        "avg_forward_velocity": avg_forward_velocity,
        "fell": bool(final_info.get("fell", False)),
        "final_base_height": float(final_info.get("base_height", 0.0)),
    }


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    print(f"Torch version: {torch.__version__}")

    model = PPO.load(str(model_path))
    env = make_env(args)

    results = []
    try:
        for ep in range(args.episodes):
            result = evaluate_episode(model, env, deterministic=args.deterministic)
            results.append(result)

            print(f"\nEpisode {ep + 1}")
            print(f"  reward              : {result['episode_reward']:.2f}")
            print(f"  steps               : {result['episode_steps']}")
            print(f"  forward distance    : {result['forward_distance']:.3f} m")
            print(f"  avg forward velocity: {result['avg_forward_velocity']:.3f} m/s")
            print(f"  fell                : {result['fell']}")
            print(f"  final base height   : {result['final_base_height']:.3f}")

        rewards = np.array([r["episode_reward"] for r in results], dtype=np.float64)
        steps = np.array([r["episode_steps"] for r in results], dtype=np.float64)
        distances = np.array([r["forward_distance"] for r in results], dtype=np.float64)
        velocities = np.array([r["avg_forward_velocity"] for r in results], dtype=np.float64)
        falls = np.array([1.0 if r["fell"] else 0.0 for r in results], dtype=np.float64)

        print("\n========== Summary ==========")
        print(f"Episodes              : {args.episodes}")
        print(f"Mean reward           : {rewards.mean():.2f}")
        print(f"Mean steps            : {steps.mean():.2f}")
        print(f"Mean forward distance : {distances.mean():.3f} m")
        print(f"Mean avg fwd velocity : {velocities.mean():.3f} m/s")
        print(f"Fall rate             : {falls.mean() * 100:.1f}%")
        print("=============================")

    finally:
        env.close()


if __name__ == "__main__":
    main()