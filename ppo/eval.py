from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .env import HumanoidLocomotionEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO humanoid policy")
    parser.add_argument("--model", type=Path, required=True, help="Path to .zip PPO model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with PyBullet GUI while evaluating",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    render_mode = "human" if args.render else None
    env = HumanoidLocomotionEnv(render_mode=render_mode)
    model = PPO.load(str(args.model))

    returns: list[float] = []
    lengths: list[int] = []

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not done and not truncated and steps < args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

            returns.append(total_reward)
            lengths.append(steps)
            print(
                f"episode={episode} return={total_reward:.3f} steps={steps} "
                f"fell={info.get('fell', False)}"
            )

        print("\nEvaluation summary")
        print(f"episodes={len(returns)}")
        print(f"mean_return={np.mean(returns):.3f} +/- {np.std(returns):.3f}")
        print(f"mean_length={np.mean(lengths):.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
