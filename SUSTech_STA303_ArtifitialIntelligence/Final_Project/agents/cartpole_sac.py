"""
Soft Actor-Critic (SAC) for discrete CartPole.
----------------------------------------------
- Discrete SAC variant using a categorical policy.
- Public API matches other agents: act(), step(), save(), load(), update_target().
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# -----------------------------
# Default Hyperparameters
# -----------------------------
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 50_000
INITIAL_EXPLORATION_STEPS = 1_000
Q_LR = 3e-4
PI_LR = 3e-4
ALPHA = 0.2  # entropy temperature (fixed)
TARGET_UPDATE_TAU = 0.005
TARGET_UPDATE_INTERVAL = 1


class ReplayBuffer:
    """Standard FIFO replay buffer storing numpy transitions."""

    def __init__(self, capacity: int):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        s = np.asarray(s)
        s2 = np.asarray(s2)
        if s.ndim == 2 and s.shape[0] == 1:
            s = s.squeeze(0)
        if s2.ndim == 2 and s2.shape[0] == 1:
            s2 = s2.squeeze(0)
        self.buf.append((s, int(a), float(r), s2, 0.0 if done else 1.0))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, m = zip(*batch)
        return (
            np.stack(s, axis=0),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(s2, axis=0),
            np.array(m, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class PolicyNet(nn.Module):
    """Categorical policy π(a|s) via logits."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNet(nn.Module):
    """State-action value approximator Q(s, a) for all actions."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SACConfig:
    """Configuration object for SAC."""

    gamma: float = GAMMA
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    initial_exploration: int = INITIAL_EXPLORATION_STEPS
    q_lr: float = Q_LR
    pi_lr: float = PI_LR
    alpha: float = ALPHA
    tau: float = TARGET_UPDATE_TAU
    target_update_interval: int = TARGET_UPDATE_INTERVAL
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SACSolver:
    """
    Discrete Soft Actor-Critic agent compatible with train.py.

    Public API:
      - act(state_np, evaluation_mode=False) -> int
      - step(state, action, reward, next_state, done)
      - save(path), load(path), update_target()
    """

    def __init__(self, observation_space: int, action_space: int, cfg: SACConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or SACConfig()

        self.device = torch.device(self.cfg.device)
        self.policy = PolicyNet(self.obs_dim, self.act_dim).to(self.device)
        self.q1 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2 = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q1_target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.q2_target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.update_target(hard=True)

        self.q_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.cfg.q_lr,
        )
        self.pi_optim = optim.Adam(self.policy.parameters(), lr=self.cfg.pi_lr)

        self.memory = ReplayBuffer(self.cfg.memory_size)
        self.steps = 0

    # -----------------------------
    # Acting
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        state = np.asarray(state_np, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.policy(state_t)
            dist = Categorical(logits=logits)
            if evaluation_mode:
                action = int(torch.argmax(dist.probs, dim=1).item())
            else:
                action = int(dist.sample().item())
        return action

    # -----------------------------
    # Learning
    # -----------------------------
    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.memory.push(state, action, reward, next_state, done)
        self._update()

    def _update(self):
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self.steps += 1
            return

        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)

        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(m, dtype=torch.float32, device=self.device).unsqueeze(1)

        # --- Critic update ---
        with torch.no_grad():
            logits_next = self.policy(s2_t)
            # For discrete SAC, compute π(a|s') and log π(a|s') for all actions
            probs_next = torch.softmax(logits_next, dim=1)              # [B, A]
            log_probs_all = torch.log_softmax(logits_next, dim=1)       # [B, A]

            q1_next = self.q1_target(s2_t)
            q2_next = self.q2_target(s2_t)
            min_q_next = torch.min(q1_next, q2_next)                    # [B, A]

            v_next = (probs_next * (min_q_next - self.cfg.alpha * log_probs_all)).sum(dim=1, keepdim=True)
            target_q = r_t + m_t * self.cfg.gamma * v_next

        q1_values = self.q1(s_t).gather(1, a_t)
        q2_values = self.q2(s_t).gather(1, a_t)
        q1_loss = nn.functional.mse_loss(q1_values, target_q)
        q2_loss = nn.functional.mse_loss(q2_values, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optim.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=5.0)
        self.q_optim.step()

        # --- Policy update ---
        logits = self.policy(s_t)
        log_probs_all = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        q1_pi = self.q1(s_t)
        q2_pi = self.q2(s_t)
        min_q_pi = torch.min(q1_pi, q2_pi)

        inside = self.cfg.alpha * log_probs_all - min_q_pi
        policy_loss = (probs * inside).sum(dim=1).mean()

        self.pi_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=5.0)
        self.pi_optim.step()

        self.steps += 1
        if self.steps % self.cfg.target_update_interval == 0:
            self.update_target(hard=False, tau=self.cfg.tau)

    # -----------------------------
    # Target network update
    # -----------------------------
    def update_target(self, hard: bool = True, tau: float = 0.005):
        if hard:
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
        else:
            with torch.no_grad():
                for tgt, src in zip(self.q1_target.parameters(), self.q1.parameters()):
                    tgt.data.mul_(1 - tau).add_(tau * src.data)
                for tgt, src in zip(self.q2_target.parameters(), self.q2.parameters()):
                    tgt.data.mul_(1 - tau).add_(tau * src.data)

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "cfg": self.cfg.__dict__,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.steps = int(ckpt.get("steps", 0))
