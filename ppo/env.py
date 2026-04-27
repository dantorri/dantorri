from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import RewardWeights

try:
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - import-time dependency guard
    raise ImportError(
        "pybullet is required for ppo.env. Install dependencies from requirements.txt"
    ) from exc


@dataclass(slots=True)
class EnvConfig:
    max_episode_steps: int = 1000
    target_speed_mps: float = 1.25
    max_torque: float = 100.0
    simulation_timestep: float = 1.0 / 240.0
    frame_skip: int = 4
    min_torso_height: float = 0.75
    max_abs_roll: float = 1.2
    max_abs_pitch: float = 1.2
    reward: RewardWeights = field(default_factory=RewardWeights)


class HumanoidLocomotionEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium-compatible PyBullet environment for early PPO development."""

    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EnvConfig()

        connection_mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.config.simulation_timestep, physicsClientId=self.client)

        # Try this first. It usually fixes the "sideways humanoid" issue.
        self.initial_base_pos = [0.0, 0.0, 4.0]
        self.initial_base_orn = p.getQuaternionFromEuler([1.5708, 0.0, 0.0])

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            self.initial_base_pos,
            self.initial_base_orn,
            useFixedBase=False,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.client,
        )

        self.actuated_joint_indices = self._find_actuated_joints()

        # Disable default motor behavior so torque control is actually in charge
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.actuated_joint_indices),
            physicsClientId=self.client,
        )

        self.num_actions = len(self.actuated_joint_indices)
        self.num_obs = 13 + 2 * self.num_actions

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        self.step_count = 0

    def _find_actuated_joints(self) -> list[int]:
        joint_indices: list[int] = []
        for joint_idx in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            joint_info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client)
            joint_type = joint_info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                joint_indices.append(joint_idx)
        if not joint_indices:
            raise RuntimeError("No actuated joints found on loaded humanoid model.")
        return joint_indices

    def _get_obs(self) -> np.ndarray:
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        base_lin_vel, base_ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.client
        )

        joint_states = p.getJointStates(
            self.robot_id, self.actuated_joint_indices, physicsClientId=self.client
        )
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        return np.array(
            [
                *base_pos,
                *base_quat,
                *base_lin_vel,
                *base_ang_vel,
                *joint_positions,
                *joint_velocities,
            ],
            dtype=np.float32,
        )

    def _compute_reward_components(
        self, action: np.ndarray, base_lin_vel_x: float, fell: bool
    ) -> dict[str, float]:
        cfg = self.config

        speed_err = abs(cfg.target_speed_mps - base_lin_vel_x)
        forward_reward = float(cfg.reward.forward_velocity * np.exp(-speed_err))

        survival_reward = float(cfg.reward.survival)

        energy_penalty = float(
            cfg.reward.energy * np.sum(np.square(action * cfg.max_torque))
        )

        fall_penalty = float(cfg.reward.fall_penalty if fell else 0.0)

        total_reward = forward_reward + survival_reward - energy_penalty - fall_penalty

        return {
            "forward_reward": forward_reward,
            "survival_reward": survival_reward,
            "energy_penalty": energy_penalty,
            "fall_penalty": fall_penalty,
            "total_reward": float(total_reward),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        p.resetBasePositionAndOrientation(
            self.robot_id,
            self.initial_base_pos,
            self.initial_base_orn,
            physicsClientId=self.client,
        )
        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        roll, pitch, yaw = p.getEulerFromQuaternion(base_quat)

        print("RESET DEBUG")
        print("  base_pos :", base_pos)
        print("  roll     :", roll)
        print("  pitch    :", pitch)
        print("  yaw      :", yaw)
        
        p.resetBaseVelocity(
            self.robot_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=self.client
        )

        for joint_idx in self.actuated_joint_indices:
            p.resetJointState(self.robot_id, joint_idx, 0.0, 0.0, physicsClientId=self.client)

        # Re-disable motors after reset just to be safe
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0] * len(self.actuated_joint_indices),
            physicsClientId=self.client,
        )

        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        clipped_action = np.clip(action, -1.0, 1.0)
        torques = clipped_action * self.config.max_torque

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=torques.tolist(),
            physicsClientId=self.client,
        )

        for _ in range(self.config.frame_skip):
            p.stepSimulation(physicsClientId=self.client)

        self.step_count += 1

        obs = self._get_obs()

        base_pos, base_quat = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client
        )
        roll, pitch, yaw = p.getEulerFromQuaternion(base_quat)

        base_height = float(base_pos[2])
        base_lin_vel_x = float(obs[7])

        fell = (
            base_height < self.config.min_torso_height
            or abs(roll) > self.config.max_abs_roll
            or abs(pitch) > self.config.max_abs_pitch
        )

        terminated = fell
        truncated = self.step_count >= self.config.max_episode_steps

        reward_parts = self._compute_reward_components(
            clipped_action, base_lin_vel_x, fell
        )

        info = {
            "base_height": base_height,
            "base_lin_vel_x": base_lin_vel_x,
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "fell": fell,
            "step_count": self.step_count,
            **reward_parts,
        }

        return obs, reward_parts["total_reward"], terminated, truncated, info

    def close(self):
        if p.isConnected(physicsClientId=self.client):
            p.disconnect(physicsClientId=self.client)
