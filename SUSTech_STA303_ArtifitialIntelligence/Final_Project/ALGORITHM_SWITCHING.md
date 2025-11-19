# 在 `train.py` 中切换 DQN / PPO 指南

为方便直接使用 `train.py` 同时训练/评估 DQN 和 PPO，本仓库新增了统一的 agent 注册逻辑与命令行参数。本文说明需要变更的参数以及示例用法。

## 1. 核心参数

- `train(num_episodes=..., terminal_penalty=True, algorithm="dqn")`
  - `algorithm` 现支持 `"dqn"` 与 `"ppo"`，用于创建不同的 agent/模型路径。
  - 其他参数保持不变：`num_episodes` 控制训练轮数，`terminal_penalty` 控制是否在失败回合加 -1。
- `evaluate(model_path=None, algorithm="dqn", episodes=..., render=..., fps=...)`
  - `algorithm` 同样决定加载哪种 agent。
  - 若 `model_path` 为空，则自动从 `./models/cartpole_<algo>.torch` 读取对应权重。

## 2. 命令行使用

`train.py` 现在自带简单的 CLI，常用参数如下：

```
python train.py \
    --algorithm ppo \
    --train-episodes 800 \
    --no-terminal-penalty \
    --eval-episodes 50 \
    --eval-render
```

- `--algorithm {dqn,ppo}`：选择要训练/评估的智能体。
- `--train-episodes`：训练总回合数。
- `--no-terminal-penalty`：关闭 -1 终止惩罚（默认开启）。
- `--skip-eval`：仅训练，不做评估。
- `--eval-episodes` / `--eval-render` / `--eval-fps`：控制评估轮数、是否渲染、渲染帧率。

若只想训练 PPO 而不立刻评估，可执行：

```
python train.py --algorithm ppo --train-episodes 600 --skip-eval
```

训练完成后，再单独评估：

```
python train.py --algorithm ppo --skip-eval --train-episodes 0
python - <<'PY'
from train import evaluate
evaluate(algorithm="ppo", episodes=50, render=False)
PY
```

或直接调用 `total_eval.py` 统一入口也可以。

## 3. 作为模块调用

在其他脚本/Notebook 中，可直接传参调用：

```python
from train import train, evaluate

# 训练 PPO
train(num_episodes=500, terminal_penalty=True, algorithm="ppo")

# 评估已保存的 PPO 模型
evaluate(algorithm="ppo", episodes=20, render=False)
```

保持 `algorithm` 参数一致即可复用 `./models` 下按算法命名的 `.torch` 文件。

## 4. 新增逻辑概览

- `AGENT_REGISTRY`：维护 `{ "dqn": AgentEntry(...), "ppo": AgentEntry(...) }`，集中管理 Solver、配置、模型文件名。
- `_default_model_path(algorithm)`：根据算法生成 `models/cartpole_<algo>.torch`。
- 所有训练/评估逻辑通过查表自动选择对应 Agent，因此将来新增算法时，只需在注册表里再添加一项即可。

如需在 README 或其他地方展示新的运行方式，可参考此文档的示例命令。若想添加更多算法（如 A2C），先实现 `agents/cartpole_a2c.py`，再在 `AGENT_REGISTRY` 中注册即可继承同样的入口。 

