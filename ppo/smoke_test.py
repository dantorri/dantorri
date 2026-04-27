from __future__ import annotations

import argparse

import numpy as np

from .env import HumanoidLocomotionEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run random-action smoke test for env")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = HumanoidLocomotionEnv(render_mode=None)
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=episode)
            assert np.isfinite(obs).all(), "Reset observation contains NaNs/inf"

            total_reward = 0.0
            for step in range(args.steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                assert np.isfinite(obs).all(), f"Non-finite observation at step {step}"
                total_reward += reward
                if terminated or truncated:
                    break

            print(
                f"episode={episode} steps={step + 1} total_reward={total_reward:.3f} "
                f"last_height={info['base_height']:.3f} fell={info['fell']}"
            )

        print("Smoke test completed successfully.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
