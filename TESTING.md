# Local Testing Guide

## Pull latest changes
```bash
git fetch origin
git checkout work
git pull --rebase origin work
```

## Prepare conda environment
```bash
conda create -n biped311 python=3.11 -y
conda activate biped311
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step-by-step validation

### Step 1 — Smoke test env dynamics and control
```bash
python -m ppo.smoke_test --episodes 2 --steps 300
```

### Step 2 — Train PPO for 100k timesteps
```bash
python -m ppo.train --timesteps 100000 --run-name ppo_100k
```

### Step 3 — Evaluate saved policy
```bash
python -m ppo.eval --model artifacts/models/ppo_100k.zip --episodes 5 --max-steps 2000
```

### Step 4 — Render policy in GUI
```bash
python -m ppo.render --model artifacts/models/ppo_100k.zip --episodes 2 --max-steps 2000
```

## Expected outputs
- Model checkpoint: `artifacts/models/ppo_100k.zip`
- TensorBoard logs: `artifacts/tensorboard/`
- Console evaluation summary with mean return and episode length.

## Troubleshooting
- Missing packages: confirm `biped311` is active, then rerun `pip install -r requirements.txt`.
- Humanoid looks rigid/stuck: rerun smoke test and ensure latest `ppo/env.py` is present.
- GUI doesn’t open: verify local machine has display access for PyBullet GUI and try a local terminal session.
