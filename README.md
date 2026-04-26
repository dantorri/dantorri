# Bipedal Locomotion with Reinforcement Learning

## Problem Statement
Walking on two legs is a difficult control problem, but solving it is essential for effective humanoid and terrain-capable robotics. This motion, called **bipedal locomotion**, is inherently unstable and requires continuous feedback control.

The challenge is amplified by:
- Nonlinear dynamics,
- High-dimensional state and action spaces, and
- Frequent contact changes with the ground.

Rather than hand-designing a controller, this project uses **reinforcement learning (RL)** in simulation. The agent learns to walk in a PyBullet environment by maximizing rewards for speed, stability, and energy efficiency.

## Approach
We simulate a humanoid robot from a URDF model in **PyBullet** and wrap the simulator in a **Gymnasium-compatible environment** for use with standard RL libraries.

### Observation space
The policy receives:
- Torso position and orientation,
- Linear and angular velocity,
- Joint angles, and
- Joint velocities.

### Action space
A continuous action vector specifies target joint torques for each actuated joint, subject to maximum torque limits.

### Reward design
The reward function combines:
1. Forward-velocity tracking toward a target speed and direction,
2. Survival bonus per timestep,
3. Penalty for energy consumption,
4. Large fall penalty with early episode termination.

Reward weights are treated as tunable hyperparameters.

## Algorithms Compared
This project compares two policy-gradient methods implemented in **Stable-Baselines3**:

- **PPO** (on-policy): uses a clipped surrogate objective for stable policy updates.
- **SAC** (off-policy actor-critic): maximizes expected return and policy entropy to improve exploration.

Both methods use MLP policy/value networks trained with gradient descent, linking directly to optimization methods covered in course material.

## Evaluation Metrics
We will compare PPO and SAC using:
- Cumulative reward,
- Forward distance traveled,
- Timesteps before falling,
- Energy cost per meter,
- Wall-clock training time and sample efficiency.

## Ethical Considerations
Key ethical risks include:

1. **Simulation-to-real transfer risk**: policies that are safe in simulation may fail unpredictably on hardware.
   - Mitigation: enforce torque/joint limits and safety constraints before real-world deployment.
2. **Potential misuse**: bipedal robots may be deployed for surveillance, raising privacy concerns.
3. **Socioeconomic impacts**: increased automation can contribute to job displacement in some sectors.

## Software and Compute
- **Software**: Python, PyBullet, Stable-Baselines3, Gymnasium, PyTorch, Matplotlib.
- **Compute**: personal machines with GPU acceleration.
- **Expected training budget**: approximately 1–5 million timesteps, typically several hours on modern GPUs.

## References
1. Coumans & Bai, *PyBullet Physics Simulation*.
2. Fujimoto et al., *TD3* (potential additional baseline).
3. Haarnoja et al., *Soft Actor-Critic (SAC)*.
4. Raffin et al., *Stable-Baselines3*.
5. Schulman et al., *Proximal Policy Optimization (PPO)*.
6. Sutton & Barto, *Reinforcement Learning: An Introduction*.

## Expected Deliverables
1. Trained PPO and SAC policies that achieve stable bipedal walking,
2. Training curves and comparative PPO vs. SAC analysis,
3. Reward ablation study isolating reward-component effects,
4. Video recordings of gait behavior at multiple training stages,
5. Final report with ethical discussion of sim-to-real deployment risk.
