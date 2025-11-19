# AGENTS 开发规范

本规范用于指导 Codex（或其他贡献者）在 `agents/` 目录下实现新的智能体代码，并确保其风格与 `agents/cartpole_dqn.py` 保持一致，且能够无缝兼容 `train.py` 提供的训练/评估入口。

## 1. 目录与命名
- 所有新智能体文件必须放在 `agents/` 目录下，并使用 `cartpole_<algorithm>.py` 命名（例如 `cartpole_ppo.py`）。
- 每个文件应包含完整可运行的实现，不依赖未声明的全局状态。

## 2. 基本代码规范
- 遵循 `agents/cartpole_dqn.py` 的代码结构和风格：
  - 顶部集中定义默认超参数与常量。
  - 使用类型注解、`dataclass`（若需要配置对象）、以及清晰的 docstring。
  - 依赖 `numpy`, `torch`, `gymnasium` 等第三方库时，导入顺序与 DQN 文件保持一致，避免循环依赖。
- 遵循 PEP 8/PEP 257：4 空格缩进、下划线命名函数/变量、驼峰命名类名、模块级常量使用全大写。
- 重要逻辑（网络结构、损失计算、更新流程）需要提供精炼的注释，参考 `cartpole_dqn.py` 的写法。

## 3. 与 `train.py` 的接口约定
为保证 `train.py` 可以直接切换不同智能体，新的 solver 类必须遵循以下 API：

| 方法 | 签名 | 说明 |
| --- | --- | --- |
| `__init__(observation_space: int, action_space: int, cfg: Optional[YourConfig]=None)` | 接受观测/动作维度，并存储配置（可选）。 |
| `act(state_np: np.ndarray, evaluation_mode: bool = False) -> int` | 输入 `[1, obs_dim]`（或 `[obs_dim]`）的 numpy 状态，训练模式下允许探索，评估模式必须纯贪心/策略输出。 |
| `step(state, action, reward, next_state, done)` | 接受 `train.py` 传入的单步转移；内部负责记忆与学习。 |
| `save(path: str)` / `load(path: str)` | 读写模型权重，默认路径位于 `models/`。 |
| `update_target(...)` | 若算法无需目标网络，可实现空方法或提供适配逻辑；但函数必须存在以兼容 `train.py`。 |

实现时需保证：
- `train.py` 的 `terminal_penalty`、`reward` 修改逻辑无需在智能体内重新处理。
- `act` 和 `step` 能够处理 `numpy` 数组，内部再转换为 Torch 张量。
- 所有设备选择（CPU/GPU）逻辑参考 `cartpole_dqn.py`：读取自配置或自动检测 `torch.cuda.is_available()`。

## 4. 配置与超参数
- 建议使用与 DQN 相同的 `@dataclass Config` 模式，包含核心超参数、`device`、以及其他算法特有参数。
- 默认值应保证在 CartPole-v1 上可运行（无需额外传参即可训练）。

## 5. 与现有代码的互操作
- 新智能体如果需要在 `train.py` 中启用，应提供清晰的初始化方式（例如通过 `algorithm` 参数选择），并保证内部依赖（如 replay buffer、网络结构）只使用本文件或标准库/已安装库。
- 若引入新依赖，请更新 `requirements.txt` 并在 README/架构文档中注明用途。

## 6. 测试与自检
- 在提交新智能体之前，至少运行一次 `python train.py`（必要时调整 `num_episodes` 为较小值）以验证训练循环没有崩溃。
- 确保 `agent.save()` 生成的文件可被 `agent.load()` 成功加载并用于 `evaluate()`。

如需扩展规范（例如多算法路由、统一注册表等），请先在 PR/Issue 中说明需求，再更新本文件。

