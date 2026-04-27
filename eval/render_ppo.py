# from __future__ import annotations

# import os
# # Windows/OpenMP workaround must be set BEFORE importing torch
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import argparse
# import time
# from pathlib import Path

# import torch
# from stable_baselines3 import PPO

# from ppo.env import EnvConfig, HumanoidLocomotionEnv


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Render a trained PPO locomotion policy")
#     parser.add_argument(
#         "--model-path",
#         type=str,
#         required=True,
#         help="Path to a saved PPO model zip file",
#     )
#     parser.add_argument(
#         "--episodes",
#         type=int,
#         default=3,
#         help="Number of episodes to render",
#     )
#     parser.add_argument(
#         "--deterministic",
#         action="store_true",
#         help="Use deterministic actions",
#     )
#     parser.add_argument(
#         "--target-speed",
#         type=float,
#         default=1.25,
#         help="Override target speed",
#     )
#     parser.add_argument(
#         "--max-episode-steps",
#         type=int,
#         default=1000,
#         help="Episode step limit",
#     )
#     parser.add_argument(
#         "--max-torque",
#         type=float,
#         default=100.0,
#         help="Maximum torque passed into the env config",
#     )
#     parser.add_argument(
#         "--sleep",
#         type=float,
#         default=1.0 / 60.0,
#         help="GUI playback delay between policy steps",
#     )
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()

#     model_path = Path(args.model_path)
#     if not model_path.exists():
#         raise FileNotFoundError(f"Model not found: {model_path}")

#     print(f"Loading model from: {model_path}")
#     print(f"Torch version: {torch.__version__}")

#     env_cfg = EnvConfig(
#         max_episode_steps=args.max_episode_steps,
#         target_speed_mps=args.target_speed,
#         max_torque=args.max_torque,
#     )
#     env = HumanoidLocomotionEnv(render_mode="human", config=env_cfg)

#     model = PPO.load(str(model_path))

#     try:
#         for ep in range(args.episodes):
#             obs, info = env.reset()
#             done = False
#             total_reward = 0.0
#             steps = 0

#             print(f"\nStarting rendered episode {ep + 1}")

#             while not done:
#                 action, _ = model.predict(obs, deterministic=args.deterministic)
#                 obs, reward, terminated, truncated, info = env.step(action)

#                 total_reward += float(reward)
#                 steps += 1
#                 done = bool(terminated or truncated)

#                 # Slow the GUI enough for you to see it
#                 time.sleep(args.sleep)

#             print(f"Episode {ep + 1} finished")
#             print(f"  reward: {total_reward:.2f}")
#             print(f"  steps : {steps}")
#             print(f"  fell  : {info.get('fell', False)}")
#             print(f"  base height: {info.get('base_height', 0.0):.3f}")
#             print(f"  base vx    : {info.get('base_lin_vel_x', 0.0):.3f}")

#             time.sleep(1.5)

#     finally:
#         env.close()


# if __name__ == "__main__":
#     main()