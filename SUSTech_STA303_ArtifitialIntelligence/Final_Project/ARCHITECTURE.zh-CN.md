# 代码架构解析说明（Final_Project）

本文对 `Final_Project` 的整体代码结构、各模块/文件的职责、主要类与函数的功能、以及模块之间的通讯与数据流进行说明，便于后续维护与扩展。

## 目录结构概览

```
Final_Project/
├── agents/                 # 强化学习智能体实现（当前提供 DQN 基线）
│   └── cartpole_dqn.py     # CartPole 的 DQN 实现（含 Q 网络、经验回放、目标网络等）
├── assets/                 # 项目图片/素材
│   ├── cartpole_example.gif
│   └── cartpole_icon_web.png
├── models/                 # 训练后模型保存目录（.torch 权重文件）
├── scores/                 # 训练过程得分日志与图表
│   ├── score_logger.py     # 记录与绘图工具
│   ├── scores.csv          # 每回合分数（训练时自动生成/更新）
│   └── scores.png          # 分数曲线图（训练时自动生成/更新）
├── .github/workflows/      # CI（可选）
│   └── macos-ci.yml
├── LICENSE
├── README.md               # 项目使用说明与任务介绍
├── requirements.txt        # 依赖（torch、gymnasium、matplotlib 等）
└── train.py                # 训练与评估入口脚本
```

## 顶层入口：train.py

- 常量
  - `ENV_NAME = "CartPole-v1"`：环境名称。
  - `MODEL_DIR = "models"` / `MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_dqn.torch")`：模型保存位置。

- 函数：`train(num_episodes: int = 200, terminal_penalty: bool = True) -> DQNSolver`
  - 作用：完整训练循环。
    - 创建环境 `gym.make(ENV_NAME)` 与分数记录器 `ScoreLogger(ENV_NAME)`。
    - 通过环境空间自动推断 `obs_dim` 与 `act_dim`。
    - 构建智能体 `DQNSolver(obs_dim, act_dim, cfg=DQNConfig())`。
    - 按回合循环：
      - `state, info = env.reset(seed=run)`，将 `state` reshape 为 `(1, obs_dim)`。
      - 循环步进：
        1) `action = agent.act(state)`（带 ε-贪心探索）。
        2) `next_state_raw, reward, terminated, truncated, _ = env.step(action)`。
        3) 若 `terminal_penalty=True` 且 `done`，将 `reward` 设为 `-1.0`。
        4) `next_state = reshape(next_state_raw, (1, obs_dim))`。
        5) `agent.step(state, action, reward, next_state, done)`（内部记忆+学习）。
        6) `state = next_state`。
        7) 若 `done`：打印本回合信息并 `logger.add_score(steps, run)`。
    - 训练结束：`agent.save(MODEL_PATH)` 持久化模型。

- 函数：
  `evaluate(model_path: str | None = None, algorithm: str = "dqn", episodes: int = 5, render: bool = True, fps: int = 60)`
  - 作用：加载已训练模型并按贪心策略评估。
    - 自动从 `models/` 选择 `.torch` 文件（若 `model_path is None`）。
    - `env = gym.make(ENV_NAME, render_mode="human" if render else None)`。
    - 构建 `DQNSolver` 并 `agent.load(model_path)`。
    - 每个评估回合：用 `agent.act(state, evaluation_mode=True)`（纯贪心，不探索）直到终止，统计步数。
    - 按需 `time.sleep(1/fps)` 控制渲染速度，最终输出平均步数。

- 与其他模块的接口
  - 从 `agents.cartpole_dqn` 导入：`DQNSolver`, `DQNConfig`。
  - 从 `scores.score_logger` 导入：`ScoreLogger`。
  - 文件层面：读写 `models/*.torch`、`scores/*.csv|.png`。

## 智能体实现：agents/cartpole_dqn.py

- 顶部超参数（默认）
  - `GAMMA=0.99`, `LR=1e-3`, `BATCH_SIZE=32`, `MEMORY_SIZE=50_000`,
    `INITIAL_EXPLORATION_STEPS=1000`, `EPS_START=1.0`, `EPS_END=0.05`,
    `EPS_DECAY=0.995`, `TARGET_UPDATE_STEPS=500`。

- 类：`class QNet(nn.Module)`
  - 作用：Q 函数近似器（MLP）。
  - `__init__(obs_dim: int, act_dim: int)`：搭建网络 `Linear(obs_dim,64) + ReLU + Linear(64, act_dim)`，并对 `Linear` 使用 `xavier_uniform_` 初始化。
  - `forward(x: torch.Tensor) -> torch.Tensor`：输入 `[B, obs_dim]`，输出 `[B, act_dim]` 对应各动作的 Q 值。

- 类：`class ReplayBuffer`
  - 作用：经验回放缓冲区（FIFO deque，最大 `capacity`）。
  - `__init__(capacity: int)`：初始化 `deque(maxlen=capacity)`。
  - `push(s, a, r, s2, done)`：存储转移 `(s, a, r, s2, mask)`，其中 `mask = 0.0 if done else 1.0`；若 `s/s2` 为 `[1, obs_dim]` 自动 squeeze 为 `[obs_dim]`。
  - `sample(batch_size: int)`：均匀采样小批量，返回 `np.stack/np.array`：`(s, a, r, s2, m)`，形状分别为 `[B, obs_dim]`, `[B]`, `[B]`, `[B, obs_dim]`, `[B]`。
  - `__len__()`：当前存量。

- 数据类：`@dataclass class DQNConfig`
  - 作用：整理超参数，便于可视与传参。
  - 字段：`gamma, lr, batch_size, memory_size, initial_exploration, eps_start, eps_end, eps_decay, target_update, device`。
  - `device` 自动选择可用的 `cuda`，否则 `cpu`。

- 类：`class DQNSolver`
  - 作用：DQN 智能体（在线/目标 Q 网络、经验回放、ε-贪心、持久化）。
  - 构造：`__init__(observation_space: int, action_space: int, cfg: DQNConfig | None = None)`
    - 记录 `obs_dim/act_dim/cfg`，设置 `device = torch.device(cfg.device)`。
    - 创建 `self.online = QNet(...).to(device)` 和 `self.target = QNet(...).to(device)`，并 `update_target(hard=True)` 对齐参数。
    - `self.optim = Adam(self.online.parameters(), lr=cfg.lr)`；`self.memory = ReplayBuffer(cfg.memory_size)`。
    - 计数：`self.steps = 0`；探索率：`self.exploration_rate = cfg.eps_start`。
  - 行为选择：`act(state_np: np.ndarray, evaluation_mode: bool = False) -> int`
    - 训练模式：以概率 `ε` 随机动作，否则贪心 `argmax_a Q_online(s,a)`；评估模式：始终贪心。
    - 入参 `state_np` 期望为 `[1, obs_dim]`，内部自动处理 `[obs_dim]` 情况。
  - 记忆入栈：`remember(state, action, reward, next_state, done)`
    - 直接调用 `self.memory.push(...)`。
  - 学习入口：`step(state, action, reward, next_state, done)`
    - 先 `remember(...)`，再调用 `experience_replay()` 进行一次参数更新尝试。
  - 参数更新：`experience_replay()`
    - 仅当 `len(memory) >= max(cfg.batch_size, cfg.initial_exploration)` 时执行。
    - 采样批量并转为张量：`s_t, a_t, r_t, s2_t, m_t`（其中 `m_t` 为终止 mask）。
    - 当前 Q：`q_sa = online(s_t).gather(1, a_t)`；目标 Q：`q_next = target(s2_t).max(1)[0]`。
    - 目标值：`target = r_t + m_t * gamma * q_next`；损失：`MSE(q_sa, target)`；执行 `backward()` 与 `optim.step()`。
    - 衰减 ε：`_decay_eps()`；每隔 `cfg.target_update` 步执行 `update_target(hard=True)`。
  - 目标网络同步：`update_target(hard: bool = True, tau: float = 0.005)`
    - `hard=True`：直接拷贝；`hard=False`：Polyak 平滑更新，`p_t = (1-τ) p_t + τ p`。
  - 持久化：
    - `save(path: str)`：`torch.save({"online": state_dict, "target": state_dict, "cfg": cfg.__dict__}, path)`。
    - `load(path: str)`：`torch.load(..., map_location=self.device)`，恢复 online/target 权重（可按需恢复 cfg）。
  - 内部辅助：`_decay_eps()`：`exploration_rate = max(eps_end, exploration_rate * eps_decay)`，并 `self.steps += 1`。

- 与外部的接口与约定
  - `train.py` 仅依赖 `DQNSolver` 的公开 API：`act() / step() / save() / load() / update_target()`。
  - `act/step` 期望 `state` 形状为 `[1, obs_dim]`；`step` 使用标量 `reward: float` 与 `done: bool`。
  - 通过 `save/load` 与磁盘交互（`models/*.torch`）。

## 记录与可视化：scores/score_logger.py

- 常量
  - 目录与路径：`SCORES_DIR=./scores`，`scores.csv/png`、`solved.csv/png` 输出位置。
  - 判定阈值：`AVERAGE_SCORE_TO_SOLVE = 475`，`CONSECUTIVE_RUNS_TO_SOLVE = 100`（近 100 回合平均≥475 视为解题）。

- 类：`class ScoreLogger`
  - `__init__(env_name: str)`：初始化队列与输出目录，清理旧的 `scores.csv/png`。
  - `add_score(score: float, run: int)`：
    - 追加分数到 CSV，并更新分数曲线 PNG；
    - 维护最近 100 回合均值；若达到 `AVERAGE_SCORE_TO_SOLVE` 且样本数足够，生成 `solved.csv/png` 并打印达成信息。
  - `_save_png(input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend)`：
    - 读取 CSV，绘制分数曲线；可叠加最近 N 回合均值线、目标线与粗粒度趋势线；输出到 PNG。
  - `_save_csv(path, score: float)`：将单次分数追加写入 CSV。

- 与外部的接口
  - `train.py` 在每个回合结束调用 `logger.add_score(steps, run)`。
  - 文件层面：写 `scores/*.csv`，读回绘图并输出 `scores/*.png`。

## 模块间通讯与数据流

- 调用关系
  - `train.py` → `agents.cartpole_dqn.DQNSolver`：
    - 训练阶段：`act()`（动作选择）与 `step()`（记忆+学习）。
    - 训练结束：`save(MODEL_PATH)`；评估阶段：`load(MODEL_PATH)` 与 `act(..., evaluation_mode=True)`。
  - `train.py` → `scores.score_logger.ScoreLogger`：
    - 每回合结束：`add_score(steps, run)` 记录与绘图。

- 数据形状与约定
  - 观测 `state`：使用 `numpy.ndarray`，形状 `[1, obs_dim]`。
  - 动作 `action`：`int`（CartPole 离散动作 {0,1}）。
  - 奖励 `reward`：`float`；终止标记 `done`：`bool`。
  - 经验回放中 `mask`：`0.0 if done else 1.0`，用于构造 TD 目标。

- 文件 I/O
  - 模型：`models/cartpole_dqn.torch`（`torch.save`/`torch.load`）。
  - 分数：`scores/scores.csv`（追加写入）；图表：`scores/scores.png`、`scores/solved.png`。
  - 资源：`assets/` 仅用于 README 展示。

## 依赖与运行环境

- `requirements.txt`：
  - `torch==2.4.1`
  - `numpy>=1.26`
  - `gymnasium[classic-control]>=0.29`
  - `pygame>=2.5`
  - `matplotlib>=3.8`

- 运行流程（简述）
  1) `pip install -r requirements.txt`
  2) `python train.py` 进行训练并生成 `models/*.torch` 与 `scores/*`。
  3) 调用 `evaluate(...)`（或直接运行 `train.py` 中示例）进行评估与可视化。

## 扩展建议

- 可在 `agents/` 下新增 `cartpole_ppo.py`、`cartpole_actorcritic.py` 等文件，复用 `train.py` 入口（按 `algorithm` 参数路由）以便公平对比。
- 可尝试 Double DQN / Dueling DQN / 优先级回放 等改进，并复用 `ScoreLogger` 统一记录与绘图。

---

如需我补充英文版本或添加时序图/调用图，请告诉我。

