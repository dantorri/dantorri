# Local Testing Guide

This guide shows how to pull the latest PPO scaffold and run your first local smoke test.

## 1) Pull the latest branch

If you already cloned the repo:

```bash
git fetch origin
git checkout work
git pull --rebase origin work
```

If your branch name is different, replace `work` with your branch.

## 2) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Quick sanity checks

```bash
python -m compileall ppo
python -m ppo.train --timesteps 5000 --run-name ppo_smoke
```

Expected result:
- Training logs print to console.
- A model file is written to `artifacts/models/ppo_smoke.zip`.

## 5) Optional: TensorBoard

```bash
tensorboard --logdir artifacts/tensorboard
```

Then open the shown local URL in your browser.

## Troubleshooting

- `ModuleNotFoundError: gymnasium` or `pybullet`:
  - Ensure your venv is active and rerun `pip install -r requirements.txt`.
- `No module named stable_baselines3`:
  - Verify pip installed into the same Python interpreter you are running.
- PyBullet GUI issues:
  - Current training entrypoint uses headless mode (`DIRECT`), so GUI is not required.
