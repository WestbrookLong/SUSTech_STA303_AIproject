# Section 5 Offline RL 实验方案（可逐行复制执行）

> 运行环境：Windows PowerShell（或任意能运行 `python` 的终端）  
> 执行位置：请先进入 `AIproject/SUSTech_STA303_ArtifitialIntelligence/Final_Project`

---

## A. 单次完整复现（推荐第一次先跑这个）

### 0) 进入项目目录
```powershell
cd AIproject/SUSTech_STA303_ArtifitialIntelligence/Final_Project
```

### 1) 安装依赖（只需一次）
```powershell
pip install -r requirements.txt
```

### 2) 设置实验变量（可直接复制）
```powershell
$ENV_ID = "CartPole-v1"
$CKPT   = "models/cartpole_sac_2.torch"
$STEPS  = 50000
$GAMMA  = 0.99
$SEED   = 0

# 有 GPU 用 "cuda"，没有就改成 "cpu"
$DEVICE = "cuda"
```

### 3) 采集离线数据集（D-Expert / D-Mixed）
```powershell
python scripts/section5_collect_dataset.py --env_id $ENV_ID --ckpt $CKPT --out "datasets/section5/cartpole_d_expert_s$SEED.pt" --steps $STEPS --prand 0.0 --seed $SEED --gamma $GAMMA
python scripts/section5_collect_dataset.py --env_id $ENV_ID --ckpt $CKPT --out "datasets/section5/cartpole_d_mixed_s$SEED.pt"  --steps $STEPS --prand 0.5 --seed $SEED --gamma $GAMMA
```

### 4) 训练：BC / AWBC（用 D-Expert）
```powershell
python scripts/section5_train_bc.py   --dataset "datasets/section5/cartpole_d_expert_s$SEED.pt" --env_id $ENV_ID --seed $SEED --device $DEVICE
python scripts/section5_train_awbc.py --dataset "datasets/section5/cartpole_d_expert_s$SEED.pt" --env_id $ENV_ID --seed $SEED --device $DEVICE --beta 2.0 --w_clip 20
```

### 5) 训练：CQL（用 D-Mixed）
```powershell
python scripts/section5_train_cql.py --dataset "datasets/section5/cartpole_d_mixed_s$SEED.pt" --env_id $ENV_ID --seed $SEED --device $DEVICE --alpha_cql 1.0 --entropy_alpha 0.05
```

### 6) OOD 初始状态评估（建议按论文：BC/AWBC 用 Expert，CQL 用 Mixed）
```powershell
python scripts/section5_eval_ood.py --env_id $ENV_ID --episodes 50 --seed 321 --theta_low -0.25 --theta_high 0.25 --out "runs/section5/ood_results.json" --bc_ckpt "runs/section5/bc/cartpole_d_expert_s$SEED/bc_policy.pt" --awbc_ckpt "runs/section5/awbc/cartpole_d_expert_s$SEED/awbc_policy.pt" --cql_ckpt "runs/section5/cql/cartpole_d_mixed_s$SEED/cql_policy.pt"
```

### 7) 绘图（学习曲线 + OOD 柱状图）
```powershell
python scripts/section5_plot.py --log_root "runs/section5" --output_dir "runs/section5/figures" --ood_path "runs/section5/ood_results.json"
```

### 8) （可选）指定一个 `metrics.csv` 画 “主指标+loss”
```powershell
python -m section5.plot_metrics --csv "runs/section5/cql/cartpole_d_mixed_s$SEED/metrics.csv" --metric eval_return --loss_metric critic_loss --join --out "runs/section5/figures/cql_mixed_s$SEED_join.png" --smooth 5
```

输出位置：
- 数据集：`datasets/section5/*.pt`
- 日志+模型：`runs/section5/<algo>/<dataset_name>/*`
- 图：`runs/section5/figures/*`

---

## B. 多随机种子（用于论文/报告均值±方差）

> 说明：下面会一次性跑 `seed=0,1,2`。如果你只想跑一个种子，把 `$SEEDS` 改成 `@(0)` 即可。

### 1) 设置多 seed
```powershell
$ENV_ID = "CartPole-v1"
$CKPT   = "models/cartpole_sac_2.torch"
$STEPS  = 50000
$GAMMA  = 0.99
$SEEDS  = @(0,1,2)
$DEVICE = "cuda"   # 没GPU就改成 "cpu"
```

### 2) 采集数据（Expert + Mixed）
```powershell
foreach ($s in $SEEDS) {
  python scripts/section5_collect_dataset.py --env_id $ENV_ID --ckpt $CKPT --out "datasets/section5/cartpole_d_expert_s$s.pt" --steps $STEPS --prand 0.0 --seed $s --gamma $GAMMA
  python scripts/section5_collect_dataset.py --env_id $ENV_ID --ckpt $CKPT --out "datasets/section5/cartpole_d_mixed_s$s.pt"  --steps $STEPS --prand 0.5 --seed $s --gamma $GAMMA
}
```

### 3) 训练 BC / AWBC（Expert）
```powershell
foreach ($s in $SEEDS) {
  python scripts/section5_train_bc.py   --dataset "datasets/section5/cartpole_d_expert_s$s.pt" --env_id $ENV_ID --seed $s --device $DEVICE
  python scripts/section5_train_awbc.py --dataset "datasets/section5/cartpole_d_expert_s$s.pt" --env_id $ENV_ID --seed $s --device $DEVICE --beta 2.0 --w_clip 20
}
```

### 4) 训练 CQL（Mixed）
```powershell
foreach ($s in $SEEDS) {
  python scripts/section5_train_cql.py --dataset "datasets/section5/cartpole_d_mixed_s$s.pt" --env_id $ENV_ID --seed $s --device $DEVICE --alpha_cql 1.0 --entropy_alpha 0.05
}
```

### 5) OOD 评估（每个 seed 跑一次；结果追加到同一个 JSON）
```powershell
Remove-Item "runs/section5/ood_results.json" -ErrorAction SilentlyContinue
foreach ($s in $SEEDS) {
  python scripts/section5_eval_ood.py --env_id $ENV_ID --episodes 50 --seed (321 + $s) --theta_low -0.25 --theta_high 0.25 --out "runs/section5/ood_results.json" --bc_ckpt "runs/section5/bc/cartpole_d_expert_s$s/bc_policy.pt" --awbc_ckpt "runs/section5/awbc/cartpole_d_expert_s$s/awbc_policy.pt" --cql_ckpt "runs/section5/cql/cartpole_d_mixed_s$s/cql_policy.pt"
}
```

### 6) 绘图（曲线图 + OOD 柱状图）
```powershell
python scripts/section5_plot.py --log_root "runs/section5" --output_dir "runs/section5/figures" --ood_path "runs/section5/ood_results.json"
```

---

## C. 常用“改一改就能跑”的调参点（只改命令行参数）

```text
AWBC:
  --beta 1.0 / 2.0 / 3.0
  --w_clip 10 / 20 / 50

CQL:
  --alpha_cql 0.5 / 1.0 / 2.0
  --entropy_alpha 0.0 / 0.01 / 0.05

通用（训练更久/更稳）:
  --epochs 100 / 200
  --lr 1e-4
  --batch_size 128 / 256
```
