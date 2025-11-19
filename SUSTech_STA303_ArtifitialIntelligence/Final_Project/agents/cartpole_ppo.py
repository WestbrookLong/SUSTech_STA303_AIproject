"""
PyTorch PPO for CartPole (Gymnasium)
------------------------------------
- Shared with train.py via the same public API as DQNSolver
- Collects rollouts, computes GAE advantages, and updates policy/value heads
- Discrete action space handled through torch.distributions.Categorical
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------
# Default Hyperparameters
# -----------------------------
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ROLLOUT_LENGTH = 512
PPO_EPOCHS = 4
BATCH_SIZE = 128
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5


class PolicyNet(nn.Module):
    """Simple MLP policy that outputs action logits."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNet(nn.Module):
    """MLP value function approximator."""

    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RolloutBuffer:
    """Accumulates on-policy data and returns numpy batches with GAE."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.clear()

    def add(self, state, action, reward, done, log_prob, value):
        if len(self.states) >= self.capacity:
            # Should not happen if caller updates on schedule, but guard anyway
            return
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def build(self, last_value: float, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, ...]:
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        values_ext = np.append(values, last_value).astype(np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.int64)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        return states, actions, old_log_probs, returns, advantages

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []


@dataclass
class PPOConfig:
    """Keeps PPO hyperparameters tidy and discoverable."""

    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_epsilon: float = CLIP_EPSILON
    rollout_length: int = ROLLOUT_LENGTH
    ppo_epochs: int = PPO_EPOCHS
    batch_size: int = BATCH_SIZE
    actor_lr: float = ACTOR_LR
    critic_lr: float = CRITIC_LR
    entropy_coef: float = ENTROPY_COEF
    value_coef: float = VALUE_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOSolver:
    """
    Proximal Policy Optimization agent for CartPole.

    Public API mirrors DQNSolver so train.py can instantiate interchangeably.
    """

    def __init__(self, observation_space: int, action_space: int, cfg: PPOConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or PPOConfig()

        self.device = torch.device(self.cfg.device)
        self.policy = PolicyNet(self.obs_dim, self.act_dim).to(self.device)
        self.value_fn = ValueNet(self.obs_dim).to(self.device)

        self.actor_optim = optim.Adam(self.policy.parameters(), lr=self.cfg.actor_lr)
        self.critic_optim = optim.Adam(self.value_fn.parameters(), lr=self.cfg.critic_lr)

        self.rollout = RolloutBuffer(self.cfg.rollout_length)
        self.global_steps = 0

    # -----------------------------
    # Acting
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Samples an action from the current policy.
        In evaluation_mode, returns the greedy action (highest probability).
        """
        state_t = self._state_to_tensor(state_np)
        dist = self._dist(state_t)
        if evaluation_mode:
            action = int(torch.argmax(dist.probs, dim=1).item())
        else:
            action = int(dist.sample().item())
        return action

    # -----------------------------
    # Learning
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store transition and trigger PPO updates when rollout buffer is full."""
        state_vec = self._ensure_vector(state)
        next_state_vec = self._ensure_vector(next_state)

        log_prob, value = self._evaluate_state_action(state_vec, action)
        self.rollout.add(state_vec, action, reward, done, log_prob, value)
        self.global_steps += 1

        if len(self.rollout) >= self.cfg.rollout_length:
            last_value = 0.0 if done else self._value_estimate(next_state_vec)
            self._ppo_update(last_value)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str):
        """Save policy/value weights plus config dict for reproducibility."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_fn.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        """Load policy/value weights from disk."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.value_fn.load_state_dict(ckpt["value"])
        # Optional: restore cfg from ckpt["cfg"] if strict parity is needed

    def update_target(self, hard: bool = True, tau: float = 0.0):
        """PPO does not use a target network; method kept for interface parity."""
        _ = hard, tau

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _ppo_update(self, last_value: float):
        states, actions, old_log_probs, returns, advantages = self.rollout.build(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        num_steps = states_t.size(0)
        batch_size = min(self.cfg.batch_size, num_steps)

        for _ in range(self.cfg.ppo_epochs):
            permutation = torch.randperm(num_steps, device=self.device)
            for start in range(0, num_steps, batch_size):
                idx = permutation[start : start + batch_size]

                dist = self._dist(states_t[idx])
                log_probs = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.value_fn(states_t[idx]).squeeze(-1)
                value_loss = nn.functional.mse_loss(values_pred, returns_t[idx])

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_fn.parameters(), self.cfg.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

        self.rollout.clear()

    def _dist(self, state_t: torch.Tensor) -> Categorical:
        logits = self.policy(state_t)
        return Categorical(logits=logits)

    def _evaluate_state_action(self, state_vec: np.ndarray, action: int) -> Tuple[float, float]:
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._dist(state_t)
        action_t = torch.as_tensor([action], dtype=torch.int64, device=self.device)
        log_prob = dist.log_prob(action_t).item()
        value = self.value_fn(state_t).item()
        return log_prob, value

    def _value_estimate(self, state_vec: np.ndarray) -> float:
        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.value_fn(state_t).item())

    def _state_to_tensor(self, state_np: np.ndarray) -> torch.Tensor:
        state_vec = self._ensure_vector(state_np)
        return torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

    @staticmethod
    def _ensure_vector(state_np: np.ndarray) -> np.ndarray:
        state = np.asarray(state_np, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        return state

