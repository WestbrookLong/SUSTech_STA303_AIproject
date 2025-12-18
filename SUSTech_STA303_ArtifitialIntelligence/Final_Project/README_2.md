# README_2 – How to Train and Evaluate

This project trains RL agents on `CartPole-v1` (Gymnasium). Use the commands below to run training and evaluation, and see where outputs are saved.

## 1) Training (unified entry: `train.py`)

Run from the project root:

```bash
# Train with a specific algorithm and episodes
python train.py --algorithm ddqn --train-episodes 500
python train.py --algorithm ppo  --train-episodes 500
python train.py --algorithm sac  --train-episodes 500
```

Common flags:
- `--algorithm {dqn, ddqn, ppo, a2c, sac}`: pick the agent.
- `--train-episodes <int>`: number of training episodes.
- `--no-terminal-penalty`: disable -1 reward at episode end (optional).
- `--load-model <path>`: warm start from saved weights (optional).
- `--skip-eval`: train only, skip evaluation afterward.
- `--eval-episodes <int>`: evaluation episodes (if not skipped).
- `--eval-render`: render window during eval (can slow runs).
- `--eval-fps <int>`: FPS cap during rendering (default 60).

Outputs from training:
- Models: saved under `./models/` with names from `AGENT_REGISTRY`, e.g.:
  - `models/cartpole_dqn.torch`
  - `models/cartpole_double_dqn.torch`
  - `models/cartpole_ppo.torch`
  - `models/cartpole_a2c.torch`
  - `models/cartpole_sac.torch`
- Scores & plots: saved under `./scores/` with per-run timestamp and algorithm in the filename, e.g.:
  - `scores/CartPole-v1_ddqn_YYYYMMDD-HHMMSS_scores.csv/.png`
  - `scores/CartPole-v1_ddqn_YYYYMMDD-HHMMSS_solved.csv/.png` (when “solved”)
- Console output: prints per-episode score and, if applicable, exploration ε.

## 2) Evaluation (unified entry: `train.py`)

To evaluate a trained model (use `--skip-eval` during training if you prefer to eval later):

```bash
# Use default saved path for the algorithm
python train.py --algorithm ddqn --train-episodes 0 --skip-eval \
    && python train.py --algorithm ddqn --train-episodes 0 --eval-episodes 50 --eval-render
```

Or explicitly pass a model path and skip training:

```bash
python train.py --algorithm sac --train-episodes 0 --skip-eval
python train.py --algorithm sac --train-episodes 0 \
    --eval-episodes 50 --eval-render --load-model models/cartpole_sac.torch
```

Key flags for eval (same script):
- `--algorithm {dqn, ddqn, ppo, a2c, sac}`: choose agent for evaluation.
- `--eval-episodes <int>`: how many episodes to run.
- `--eval-render`: open a window to render (requires display).
- `--load-model <path>`: optional explicit model file; otherwise defaults to `./models/<algo>.torch`.

Evaluation outputs:
- Console: episode steps and average over eval episodes.
- No new model files are written unless training is also run.

## 3) Alternative unified scripts

- `total_train.py`: train only, same CLI style (`--algorithm`, `--episodes`, etc.); saves to `./models` and logs to `./scores` like `train.py`.
- `total_eval.py`: eval only, same registry and default model paths; logs results to console.
- `train_ppo.py`: PPO-specific trainer/evaluator (saves to `models/cartpole_ppo.torch` and logs to `scores/`).

## 4) Where files go (summary)

- Models: `./models/` (one file per algorithm; new runs overwrite the same filename for that algo).
- Scores & plots: `./scores/` (each run gets a unique timestamp+algorithm prefix; old runs are preserved).
- Hyperparameter sweeps: `script/hparam_sweep.py` reuses `train.py` and will overwrite the same model file per algorithm; scores are still timestamped in `./scores/`.

If you need to keep multiple model snapshots per run, rename or move files from `./models/` after training. The score plots are already unique per run. 
