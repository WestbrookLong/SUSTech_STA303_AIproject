# 在 `train.py` 中切换 DQN / PPO / A2C

为方便直接使用 `train.py` 训练或评估不同算法，仓库提供了统一的 agent 注册表与命令行参数。下文描述如何切换到任意已注册的算法（目前支持 DQN、Double DQN、PPO、A2C）。

## 1. 核心函数参数

- `train(num_episodes=..., terminal_penalty=True, algorithm="dqn")`
  - `algorithm` 可以是 `"dqn"`, `"ddqn"`, `"ppo"`, `"a2c"`，用来实例化对应 agent 并决定模型保存文件名。
  - 其余参数保持不变：`num_episodes` 控制训练回合数，`terminal_penalty` 控制终止惩罚。
- `evaluate(model_path=None, algorithm="dqn", episodes=..., render=..., fps=...)`
  - `algorithm` 决定加载哪种 agent。
  - 若 `model_path` 为空，会从 `./models/cartpole_<algo>.torch` 读取默认文件。

## 2. 命令行用法

`train.py` 提供 CLI，可通过以下方式切换算法：

```bash
python train.py \
    --algorithm a2c \
    --train-episodes 800 \
    --load-model models/cartpole_a2c.torch \
    --no-terminal-penalty \
    --eval-episodes 50 \
    --eval-render
```

- `--algorithm {dqn,ddqn,ppo,a2c}`：指定当前训练/评估的智能体。
- `--load-model <path>`：在训练开始前加载已有权重，可用于热启动或继续训练。
- 其余参数（`--train-episodes`, `--no-terminal-penalty`, `--skip-eval`, `--eval-*`）与原先一致。

若只想训练（例如 PPO）而暂不评估：

```bash
python train.py --algorithm ppo --train-episodes 600 --skip-eval
```

训练结束后可单独评估（以下示例评估 A2C）：

```bash
python train.py --algorithm a2c --skip-eval --train-episodes 0
python - <<'PY'
from train import evaluate
evaluate(algorithm="a2c", episodes=50, render=False)
PY
```

或直接改用 `total_eval.py` 统一入口。

## 3. 作为模块调用

```python
from train import train, evaluate

# 训练 Double DQN
train(num_episodes=500, terminal_penalty=True, algorithm="ddqn")

# 评估已保存的 A2C 模型
evaluate(algorithm="a2c", episodes=20, render=False)
```

确保 `algorithm` 参数一致，即可复用 `./models` 下对应的 `.torch` 权重。

## 4. 注册表概览

- `AGENT_REGISTRY`：维护 `{ "dqn": AgentEntry(...), "ddqn": AgentEntry(...), "ppo": AgentEntry(...), "a2c": AgentEntry(...) }`，集中存放 solver、配置类和默认文件名。
- `_default_model_path(algorithm)`：根据算法自动生成 `models/cartpole_<algo>.torch`。
- 训练/评估逻辑统一通过注册表查找，从而可以轻松扩展。若未来添加其他算法，只需在 `agents/` 新增实现并在注册表补充一条记录即可使用。
