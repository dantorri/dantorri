# Next Steps

## 1) Verify local environment
1. Activate conda env:
   ```bash
   conda activate biped311
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Confirm imports:
   ```bash
   python -c "import gymnasium, pybullet, stable_baselines3, torch; print('deps_ok')"
   ```

## 2) Run baseline checks
1. Smoke test:
   ```bash
   python -m ppo.smoke_test --episodes 2 --steps 300
   ```
2. 100k PPO training:
   ```bash
   python -m ppo.train --timesteps 100000 --run-name ppo_100k
   ```
3. Evaluate:
   ```bash
   python -m ppo.eval --model artifacts/models/ppo_100k.zip --episodes 10 --max-steps 2000
   ```
4. Render policy:
   ```bash
   python -m ppo.render --model artifacts/models/ppo_100k.zip --episodes 2 --max-steps 2000
   ```

## 3) Track metrics every run
Record for each run:
- Mean return,
- Episode length before falling,
- Average forward velocity,
- Training wall-clock time.

Store these in a CSV (`artifacts/logs/experiment_tracker.csv`) so PPO variants are comparable.

## 4) Stabilize training
Try one change at a time and re-run 100k:
- Lower `max_torque` (e.g., 60–80),
- Increase `survival` reward weight,
- Raise `fall_penalty`,
- Start with lower `target_speed_mps` (e.g., 0.6–0.9) then curriculum to 1.25.

## 5) Add robust evaluation protocol
- Evaluate each checkpoint with at least 10 episodes.
- Report mean ± std return.
- Keep random seeds fixed for fair comparisons.

## 6) Prepare SAC baseline
After PPO stabilizes:
- Add `sac/train.py` with same env and logging conventions,
- Match evaluation protocol,
- Compare PPO vs SAC by sample efficiency and final performance.

## 7) Deliverables pipeline
- Plot training curves (reward and episode length).
- Save evaluation summary tables.
- Capture short gait videos for early/mid/late training.
- Write ethical analysis section covering sim-to-real risk and misuse.
