from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time

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
    spawn_base_height: float = 1.3
    reward: RewardWeights = field(default_factory=RewardWeights)


class HumanoidLocomotionEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium-compatible PyBullet humanoid locomotion environment."""

    metadata = {"render_modes": ["human", None], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EnvConfig()

        connection_mode = p.GUI if self.render_mode == "human" else p.DIRECT
        self.client = p.connect(connection_mode)

        self.plane_id: int | None = None
        self.robot_id: int | None = None
        self.actuated_joint_indices: list[int] = []

        self._reset_world()

        self.num_actions = len(self.actuated_joint_indices)
        self.num_obs = 13 + 2 * self.num_actions

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )

        self.step_count = 0

    def _reset_world(self) -> None:
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.config.simulation_timestep, physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            enableConeFriction=1,
            physicsClientId=self.client,
        )

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot_id = p.loadURDF(
            "humanoid/humanoid.urdf",
            [0.0, 0.0, self.config.spawn_base_height],
            [0.0, 0.0, 0.0, 1.0],
            useFixedBase=False,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
            physicsClientId=self.client,
        )

        self.actuated_joint_indices = self._find_actuated_joints()
        self._disable_default_motor_controls()

    def _find_actuated_joints(self) -> list[int]:
        if self.robot_id is None:
            raise RuntimeError("Robot is not loaded.")

        joint_indices: list[int] = []
        for joint_idx in range(p.getNumJoints(self.robot_id, physicsClientId=self.client)):
            joint_info = p.getJointInfo(self.robot_id, joint_idx, physicsClientId=self.client)
            joint_type = joint_info[2]
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                joint_indices.append(joint_idx)
        if not joint_indices:
            msg = "No actuated joints found on loaded humanoid model."
            raise RuntimeError(msg)
        return joint_indices

    def _disable_default_motor_controls(self) -> None:
        if self.robot_id is None:
            raise RuntimeError("Robot is not loaded.")

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.actuated_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(self.actuated_joint_indices),
            forces=[0.0] * len(self.actuated_joint_indices),
            physicsClientId=self.client,
        )

    def _get_obs(self) -> np.ndarray:
        if self.robot_id is None:
            raise RuntimeError("Robot is not loaded.")

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

    def _compute_reward(self, action: np.ndarray, base_lin_vel_x: float, fell: bool) -> float:
        cfg = self.config
        speed_err = abs(cfg.target_speed_mps - base_lin_vel_x)
        forward_reward = cfg.reward.forward_velocity * np.exp(-speed_err)

        energy_penalty = cfg.reward.energy * float(
            np.sum(np.square(action * cfg.max_torque))
        )
        reward = forward_reward + cfg.reward.survival - energy_penalty

        if fell:
            reward -= cfg.reward.fall_penalty
        return float(reward)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self._reset_world()
        self.step_count = 0

        obs = self._get_obs()
        return obs, {"reset": True}

    def step(self, action: np.ndarray):
        if self.robot_id is None:
            raise RuntimeError("Robot is not loaded.")

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

        if self.render_mode == "human":
            time.sleep(self.config.simulation_timestep * self.config.frame_skip)

        self.step_count += 1

        obs = self._get_obs()
        base_height = float(obs[2])
        base_lin_vel_x = float(obs[7])
        fell = base_height < self.config.min_torso_height or np.isnan(obs).any()

        terminated = fell
        truncated = self.step_count >= self.config.max_episode_steps

        reward = self._compute_reward(clipped_action, base_lin_vel_x, fell)
        info = {
            "base_height": base_height,
            "base_lin_vel_x": base_lin_vel_x,
            "fell": fell,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        if p.isConnected(physicsClientId=self.client):
            p.disconnect(physicsClientId=self.client)
