# CartPole Project – Algorithm Description and Implementation Notes

This document summarizes the design and implementation of the three main agents currently implemented in the project:

- Double DQN (`ddqn`) – value-based, off-policy
- Proximal Policy Optimization (`ppo`) – on-policy actor–critic
- Soft Actor-Critic (`sac`) – off-policy maximum-entropy actor–critic (discrete variant)

It also explains their key differences and why they behave differently under various hyperparameter and training settings.

---

## 1. Common Setup

### 1.1 Environment and Training Loop

- Environment: `CartPole-v1` from Gymnasium
  - Observation: 4D continuous state `[x, x_dot, theta, theta_dot]`
  - Action space: discrete `{0, 1}` (“push left / push right”)
  - Reward: `+1` for each time-step the pole stays balanced
- Entry-point: `train.py`
  - Creates a CartPole environment
  - Instantiates the chosen agent via `AGENT_REGISTRY` using the `algorithm` argument (`"dqn"`, `"ddqn"`, `"ppo"`, `"a2c"`, `"sac"`)
  - Main loop per episode:
    1. Reset env → get `state`
    2. `action = agent.act(state)`
    3. `next_state, reward, done = env.step(action)`
    4. `agent.step(state, action, reward, next_state, done)`
    5. Accumulate `steps` until `done`
    6. Call `ScoreLogger.add_score(steps, run)` to log and plot the episode result
  - After training: `agent.save(model_path)` persists weights and selected training state to `./models`.

### 1.2 Score Logging (Shared Across Algorithms)

- `scores/score_logger.py` defines `ScoreLogger`:
  - Logs per-episode scores into CSV
  - Produces per-run score and (optionally) “solved” plots
  - Each new training run now uses a timestamp- and algorithm-specific prefix:
    - `CartPole-v1_<algorithm>_<timestamp>_scores.csv / .png`
    - `CartPole-v1_<algorithm>_<timestamp>_solved.csv / .png` (when the environment is “solved”)
- This means all algorithms share a consistent logging interface and visual format, facilitating fair comparison.

---

## 2. Double DQN (`ddqn`)

### 2.1 High-Level Idea

Double DQN is a value-based off-policy algorithm that addresses the overestimation bias of standard DQN. It maintains:

- One online Q-network `Q_online(s, a)` (trainable)
- One target Q-network `Q_target(s, a)` (slow-moving copy)

The key difference vs. vanilla DQN is the target computation:

- Vanilla DQN: `max_a' Q_target(s', a')` both selects and evaluates the next action using the same network, which tends to overestimate values.
- Double DQN: uses `Q_online` to select the greedy action, and `Q_target` to evaluate that action:
  - `a*_next = argmax_a' Q_online(s', a')`
  - `Q_target_used = Q_target(s', a*_next)`

This simple change typically improves stability and reduces overoptimistic Q-values.

### 2.2 Implementation Details

File: `agents/cartpole_double_dqn.py`

- Architecture (`QNet`):
  - 2 hidden layers with ReLU activation (e.g., `128 → 128 → act_dim`)
  - Xavier initialization for weights
- Replay buffer:
  - `ReplayBuffer` stores tuples `(s, a, r, s', mask)` where `mask = 0` if terminal, `1` otherwise.
  - FIFO deque with max capacity `MEMORY_SIZE` (e.g. 50,000 transitions).
- Agent class: `DoubleDQNSolver`
  - Hyperparameters encapsulated in `DoubleDQNConfig`:
    - `gamma`, learning rate `lr`, `batch_size`, `memory_size`, `initial_exploration`, `eps_start`, `eps_end`, `eps_decay`, and `target_update` frequency.
  - Online and target networks: `self.online`, `self.target`
  - Optimizer: Adam on `self.online.parameters()`
  - Exploration: ε-greedy (`self.exploration_rate` decays multiplicatively to `eps_end`)
  - Training step:
    1. `remember(...)`: push transition into replay buffer
    2. `_experience_replay()` when enough samples collected:
       - Sample batch `B`
       - Compute `Q_online(s, a)` for taken actions
       - Double DQN target:
         - `a_next = argmax_a' Q_online(s', a')`
         - `Q_next = Q_target(s', a_next)`
         - `target = r + mask * gamma * Q_next`
       - Loss = MSE( `Q_online(s, a)`, `target` )
       - Backpropagate and update online network
       - Periodically update target network (`hard` copy every `target_update` steps)
  - Save/Load:
    - Saves both `online` and `target` state dicts, and also `exploration_rate` + `steps` to permit true continuation of training.

### 2.3 Characteristics

- Off-policy & replay-based → data-efficient; performance improves as buffer fills.
- Stable with target networks and Double Q targets.
- Exploration is controlled by explicit ε-schedule; behavior depends strongly on `EPS_START`, `EPS_END`, and `EPS_DECAY`.

---

## 3. Proximal Policy Optimization (`ppo`)

### 3.1 High-Level Idea

PPO is an on-policy policy-gradient method with a clipped surrogate objective. It maintains:

- A stochastic policy πθ(a|s) (actor)
- A state-value function Vϕ(s) (critic)

PPO’s key idea is to constrain each policy update to stay within a “trust region” using a clipped importance sampling ratio:

- Ratio `r_t(θ) = πθ(a_t|s_t) / πθ_old(a_t|s_t)`
- Objective `L_CLIP(θ) = E[ min( r_t(θ) A_t, clip(r_t(θ), 1 − ε, 1 + ε) A_t ) ]`

This clipping prevents excessively large policy updates, improving stability compared to naive policy gradients.

### 3.2 Implementation Details

File: `agents/cartpole_ppo.py`

- Architectures:
  - Policy (`PolicyNet`): MLP with Tanh activations, outputs logits over discrete actions.
  - Value (`ValueNet`): MLP with Tanh activations, outputs scalar V(s).
  - Both use orthogonal initialization for better gradient flow.
- Rollout buffer: `RolloutBuffer`
  - Stores sequences: `(state, action, reward, done, log_prob, value)`
  - After a rollout of length `ROLLOUT_LENGTH`, computes GAE (Generalized Advantage Estimation):
    - `advantages[t]` computed backward using `gamma` and `gae_lambda`
    - Returns = advantages + values
- `PPOSolver`:
  - Hyperparameters in `PPOConfig`:
    - `gamma`, `gae_lambda`, `clip_epsilon`, `rollout_length`, `ppo_epochs`, `batch_size`, `actor_lr`, `critic_lr`, `entropy_coef`, `value_coef`, `max_grad_norm`.
  - Training loop:
    1. `step()` collects transitions into the rollout buffer.
    2. Once `rollout_length` is reached:
       - Build batched tensors of states/actions/log_probs/returns/advantages.
       - Normalize advantages.
       - For multiple epochs: shuffle indices and iterate in mini-batches.
       - Compute:
         - New log-probs under current policy
         - Ratio `r_t`
         - Clipped surrogate loss
         - Value loss (MSE between predicted V(s) and returns)
         - Entropy bonus for exploration
       - Backpropagate combined loss and update actor/critic.

### 3.3 Characteristics

- On-policy: uses fresh data from the current policy; older data is discarded after each PPO update.
- Very stable for many environments with good hyperparameters, less prone to divergence than vanilla policy gradient.
- Exploration arises from stochastic policies (sampling actions from πθ) and entropy regularization rather than ε-greedy.

---

## 4. Soft Actor-Critic (`sac`) – Discrete Variant

### 4.1 High-Level Idea

SAC is an off-policy actor–critic method that maximizes a trade-off between expected return and policy entropy (“maximum entropy RL”):

- Objective: maximize `E[ Σ_t γ^t (r_t + α H(π(·|s_t))) ]`
  - α controls the strength of the entropy term.
- Uses two Q-networks and target networks for better stability and less overestimation (similar to Twin Delayed DDPG / Double Q-learning).

In continuous control, SAC usually uses Gaussian policies; here we implement a **discrete** version using a categorical actor for the CartPole action space.

### 4.2 Implementation Details

File: `agents/cartpole_sac.py`

- Architectures:
  - `PolicyNet`: outputs logits for each action; `Categorical(logits)` defines π(a|s).
  - `QNet`: approximates Q(s, a) for all actions; two instances `q1`, `q2` with targets `q1_target`, `q2_target`.
- Replay buffer:
  - Same structure as DDQN: transitions `(s, a, r, s', mask)` with a large capacity.
- `SACSolver`:
  - Hyperparameters `SACConfig`:
    - `gamma`, `batch_size`, `memory_size`, `initial_exploration`, `q_lr`, `pi_lr`, `alpha` (entropy temperature), `tau` (Polyak averaging coefficient), `target_update_interval`.
  - Critic update:
    1. Sample batch from replay buffer.
    2. Compute `logits_next = policy(s')`, `probs_next = softmax(logits_next)`, `log_probs_next = log_softmax(logits_next)`.
    3. Evaluate target Q-values:
       - `q1_next = q1_target(s')`, `q2_next = q2_target(s')`
       - `min_q_next = min(q1_next, q2_next)`
       - Soft value: `V(s') = Σ_a π(a|s')[ min_q_next(s', a) − α log π(a|s') ]`
    4. Target Q:
       - `Q_target = r + mask * γ * V(s')`
    5. Critic loss: sum of MSE losses for `q1` and `q2` vs `Q_target` for the sampled actions.
  - Policy update:
    - For current states `s`:
      - Get logits, `probs`, `log_probs`.
      - Evaluate `q1(s)`, `q2(s)` and `min_q(s)`.
      - Loss: `E_s[ Σ_a π(a|s) (α log π(a|s) − min_q(s,a)) ]`, encouraging high entropy and high Q-values.
  - Target update:
    - Soft update (Polyak averaging) every `target_update_interval` steps:
      - `θ_target ← (1 − τ) θ_target + τ θ` for both Q networks.
  - Off-policy:
    - Can reuse old transitions in the replay buffer; more sample-efficient than PPO in principle.

### 4.3 Characteristics

- Maximum-entropy objective encourages persistent exploration; policies remain stochastic even late in training.
- Twin Q-networks and soft updates mitigate overestimation and stabilize training.
- More hyperparameters (α, τ, target update frequency) introduce flexibility but also tuning complexity.

---

## 5. Key Differences Between Double DQN, PPO, and SAC

### 5.1 Value-Based vs Policy-Based vs Actor–Critic

- **Double DQN (DDQN)**:
  - Purely value-based.
  - Learns Q(s, a) and chooses actions via greedy/ε-greedy over Q.
  - Policy is implicit (greedy or ε-greedy w.r.t. Q).
- **PPO**:
  - Policy-based with value function baseline (actor–critic).
  - Learns a parameterized policy π(a|s) directly.
  - Value function is used only for variance reduction and advantage estimation.
- **SAC**:
  - Actor–critic with explicit policy and soft Q-functions.
  - Optimizes both value and entropy, balancing exploration and exploitation.

### 5.2 On-Policy vs Off-Policy

- DDQN:
  - Off-policy with replay buffer → transitions can be reused many times.
  - Sensitive to replay buffer size, sampling, and target update schedule.
- PPO:
  - On-policy; each batch of data is used only for a limited number of gradient steps (PPO epochs).
  - Data must be refreshed frequently, but learning updates are more “trust-region-like” and stable.
- SAC:
  - Off-policy with replay buffer; can reuse data across many updates like DDQN.
  - Also uses target networks and Polyak averaging for stability.

### 5.3 Exploration Mechanisms

- DDQN:
  - ε-greedy: choose random actions with probability ε, greedy otherwise.
  - Exploration decays over time according to `EPS_DECAY` and `EPS_END`.
  - Behavior strongly depends on how fast ε decays; decaying too quickly can cause premature convergence to suboptimal behaviors.
- PPO:
  - Stochastic policy: actions sampled from π(a|s), typically unimodal but not tied to ε.
  - Entropy bonus (if used) prevents the policy from collapsing too early.
  - Exploration is more state-dependent because π(a|s) learns structured stochasticity rather than uniform random noise.
- SAC:
  - Uses an explicit entropy term in its objective; high α encourages more randomness.
  - Even optimal policies remain stochastic; exploration persists even later in training.
  - Often more robust to local minima because the algorithm is encouraged to keep trying alternatives.

### 5.4 Stability and Sensitivity to Hyperparameters

- DDQN:
  - Sensitive to learning rate, batch size, and target update interval.
  - Too high learning rate or too frequent target updates can cause divergence or oscillations in Q-values.
  - Under CartPole (a relatively simple and well-behaved environment), modest settings usually work reasonably well.
- PPO:
  - Sensitive to `clip_epsilon`, GAE parameters (`gamma`, `gae_lambda`), and rollout length.
  - Too large clipping range → unstable updates; too small → slow learning.
  - Rollout length affects variance vs bias trade-off: short rollouts noisy but quick; long rollouts more stable but slower per iteration.
- SAC:
  - Sensitive to entropy temperature α, target update τ, batch size, and learning rates.
  - Too high α → overly random behavior, slower convergence; too low α → behavior closer to deterministic actor–critic, potentially losing SAC’s robustness benefits.
  - Needs enough batch size and replay buffer diversity to approximate the soft Bellman updates reliably.

### 5.5 Behavior Under Different Training Settings

1. **Number of training episodes**:
   - DDQN:
     - Initially learns slowly until replay buffer has enough variety; performance often has a “warm-up” phase.
     - Longer training allows Q-values to converge, but overfitting to noise can occur if learning rate is too high.
   - PPO:
     - Can improve steadily with more episodes as long as rollouts cover the state space and clipping is well-tuned.
     - Because it’s on-policy, more episodes directly translate to more effective gradient steps from fresh data.
   - SAC:
     - Benefits from longer training and larger replay; can keep improving as it collects more diverse data.
     - If α is well-chosen, training tends to be smooth and monotonic in performance.

2. **Entropy/exploration parameters**:
   - DDQN:
     - Slower ε decay → more exploration, less risk of local minima but slower exploitation.
     - Faster ε decay → faster exploitation but risk of “locking in” suboptimal strategies.
   - PPO:
     - Higher entropy coefficient (if used) → policies remain more stochastic; robust but may under-exploit.
     - Lower entropy coefficient → early convergence; can be good on CartPole but less robust in harder tasks.
   - SAC:
     - α controls the exploration–exploitation balance directly inside the objective.
     - Larger α → more randomness; smaller α → behavior more like a “deterministic” actor–critic.

3. **Reward shaping / terminal penalty**:
   - In this project, `terminal_penalty=True` optionally sets the terminal reward to `-1.0` at failure.
   - DDQN:
     - Negative terminal reward shapes Q-values to prefer longer episodes strictly; encourages early avoidance of failure states.
   - PPO & SAC:
     - The negative terminal reward influences value estimates and advantages; can sharpen the learning signal about failure states.
     - With entropy bonuses, algorithms might still occasionally explore risky actions, but will learn to avoid them more consistently under a strong terminal penalty.

---

## 6. Practical Recommendations for This Project

- **When to use Double DQN (ddqn)**:
  - You want a classic, widely-understood value-based baseline.
  - You care about sample-efficiency and compatibility with large replay buffers.
  - You are comfortable tuning ε-schedules and target network update frequencies.

- **When to use PPO (ppo)**:
  - You prefer a stable on-policy method with a strong theoretical foundation.
  - You want clear, episodic learning progress and don’t need to reuse old data heavily.
  - You are interested in policy-gradient behavior and advantage estimation.

- **When to use SAC (sac)**:
  - You want robust performance with explicit entropy regularization and twin Q networks.
  - You are willing to tolerate more hyperparameters and slightly higher computational cost per update.
  - You care about exploring the trade-offs between exploitation and persistent exploration in maximum-entropy RL.

For the project report, you can:

- Compare learning curves (`scores/*.png`) between DDQN, PPO, and SAC.
- Vary key hyperparameters (e.g., ε-decay, PPO clipping, SAC α) and discuss how learning speed, stability, and final performance change.
- Relate observed behaviors back to the algorithmic differences described above (value-based vs policy-based, on-policy vs off-policy, entropy vs ε-greedy, etc.).

