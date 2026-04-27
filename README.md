# Bipedal Locomotion with Reinforcement Learning

## Current Workflow (Train 100k, Evaluate, Render)
This repository now supports an end-to-end PPO workflow:
1. Create/use Miniforge conda env `biped311`.
2. Run a smoke test to validate physics + control.
3. Train PPO for 100k timesteps.
4. Evaluate the saved model.
5. Render the learned gait in PyBullet GUI.

## Next Steps
See [`NEXT_STEPS.md`](NEXT_STEPS.md) for a concrete execution plan after initial setup.

## Setup (Miniforge / conda)
```bash
conda create -n biped311 python=3.11 -y
conda activate biped311
pip install -r requirements.txt
```

## 1) Smoke Test Environment
```bash
python -m ppo.smoke_test --episodes 2 --steps 300
```

## 2) Train PPO for 100k timesteps
```bash
python -m ppo.train --timesteps 100000 --run-name ppo_100k
```

Expected artifact:
- `artifacts/models/ppo_100k.zip`

## 3) Evaluate trained model
```bash
python -m ppo.eval --model artifacts/models/ppo_100k.zip --episodes 5 --max-steps 2000
```

## 4) Render trained policy
```bash
python -m ppo.render --model artifacts/models/ppo_100k.zip --episodes 2 --max-steps 2000
```

## Troubleshooting Humanoid Spawn / Rigid Behavior
If the humanoid is sideways, in-ground, or appears rigid:
- Pull latest code (`ppo/env.py`) where reset now rebuilds the simulation world each episode.
- Ensure default joint motors are disabled before torque control.
- Re-run smoke test before training.

## Project Overview
We train a bipedal humanoid controller in PyBullet using reinforcement learning. The current implementation focuses on PPO with configurable reward components for forward velocity, survival, energy usage, and fall penalties.

## Files
- `ppo/env.py` — Gymnasium-compatible PyBullet humanoid environment.
- `ppo/config.py` — training and reward hyperparameters.
- `ppo/train.py` — PPO training entrypoint.
- `ppo/smoke_test.py` — random-action environment verification.
- `ppo/eval.py` — deterministic model evaluation.
- `ppo/render.py` — GUI gait playback for trained models.
- `TESTING.md` — step-by-step local testing guide.
