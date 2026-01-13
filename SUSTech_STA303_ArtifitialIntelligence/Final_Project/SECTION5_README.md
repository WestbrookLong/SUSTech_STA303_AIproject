# Section 5: Offline RL & Imitation

This folder adds a full offline pipeline without touching the existing Section 1–4 code. Components live under `section5/` and `scripts/`.

## Quickstart (from project root)

1) **Collect datasets (D-Expert / D-Mixed)** using the trained SAC checkpoint:
```bash
python scripts/section5_collect_dataset.py --env_id CartPole-v1 --ckpt models/cartpole_sac_2.torch --out datasets/section5/cartpole_d_expert_old.pt --steps 50000 --prand 0.0
python scripts/section5_collect_dataset.py --env_id CartPole-v1 --ckpt models/cartpole_sac_2.torch --out datasets/section5/cartpole_d_mixed.pt  --steps 50000 --prand 0.5
```

2) **Train offline learners** (logs to `runs/section5/<algo>/...`):
```bash
python scripts/section5_train_bc.py   --dataset datasets/section5/cartpole_d_expert_old.pt --env_id CartPole-v1
python scripts/section5_train_awbc.py --dataset datasets/section5/cartpole_d_expert_old.pt --env_id CartPole-v1 --beta 2.0 --w_clip 20
python scripts/section5_train_cql.py  --dataset datasets/section5/cartpole_d_mixed.pt  --env_id CartPole-v1 --alpha_cql 1.0
```

3) **OOD evaluation** (larger reset angles). Pass the checkpoints produced in step 2:
```bash
python scripts/section5_eval_ood.py --bc_ckpt runs/section5/bc/cartpole_d_expert_old/bc_policy.pt \
    --awbc_ckpt runs/section5/awbc/cartpole_d_expert_old/awbc_policy.pt \
    --cql_ckpt runs/section5/cql/cartpole_d_mixed/cql_policy.pt \
    --theta_low -0.25 --theta_high 0.25 --episodes 20
```

4) **Plot learning curves & OOD bars**:
```bash
python scripts/section5_plot.py --log_root runs/section5 --output_dir runs/section5/figures
```

## What was added
- `section5/`: BC, AWBC, CQL implementations, datasets, utilities, and logging helpers.
- `scripts/`: runnable scripts for dataset collection, training, OOD eval, and plotting.
- Checkpoints + logs are stored under `runs/section5/...`; figures under `runs/section5/figures`. Datasets default to `datasets/section5/`.
