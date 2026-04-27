from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from .env import HumanoidLocomotionEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained PPO policy in PyBullet GUI")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model zip")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = HumanoidLocomotionEnv(render_mode="human")
    model = PPO.load(str(args.model))

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            done = False
            truncated = False
            step = 0

            while not done and not truncated and step < args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = env.step(action)
                step += 1

            print(
                f"episode={episode} steps={step} fell={info.get('fell', False)} "
                f"height={info.get('base_height', 0.0):.3f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
