"""
PyTorch Double DQN for CartPole (Gymnasium)
------------------------------------------
- Similar to the baseline DQN but uses Double Q-learning to reduce overestimation.
- Shares the same public API as other agents so it integrates with train.py directly.
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

GAMMA = 0.99
LR = 5e-5
BATCH_SIZE = 32
MEMORY_SIZE = 50_000
INITIAL_EXPLORATION_STEPS = 1_000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_STEPS = 500


class QNet(nn.Module):
    """Simple MLP for Q-value estimation."""

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
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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


@dataclass
class DoubleDQNConfig:
    gamma: float = GAMMA
    lr: float = LR
    batch_size: int = BATCH_SIZE
    memory_size: int = MEMORY_SIZE
    initial_exploration: int = INITIAL_EXPLORATION_STEPS
    eps_start: float = EPS_START
    eps_end: float = EPS_END
    eps_decay: float = EPS_DECAY
    target_update: int = TARGET_UPDATE_STEPS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleDQNSolver:
    """
    Double DQN agent with separate online/target networks.

    Public interface: act(), step(), save(), load(), update_target().
    """

    def __init__(self, observation_space: int, action_space: int, cfg: DoubleDQNConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or DoubleDQNConfig()

        self.device = torch.device(self.cfg.device)
        self.online = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.target = QNet(self.obs_dim, self.act_dim).to(self.device)
        self.update_target(hard=True)

        self.optim = optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.memory = ReplayBuffer(self.cfg.memory_size)
        self.steps = 0
        self.exploration_rate = self.cfg.eps_start

    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        if not evaluation_mode and np.random.rand() < self.exploration_rate:
            return random.randrange(self.act_dim)

        state = np.asarray(state_np, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.online(state_t)
            action = int(torch.argmax(q_values, dim=1).item())
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self._experience_replay()

    def _experience_replay(self):
        if len(self.memory) < max(self.cfg.batch_size, self.cfg.initial_exploration):
            self._decay_eps()
            return

        s, a, r, s2, m = self.memory.sample(self.cfg.batch_size)
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2_t = torch.as_tensor(s2, dtype=torch.float32, device=self.device)
        m_t = torch.as_tensor(m, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.online(s_t).gather(1, a_t)

        with torch.no_grad():
            next_actions = self.online(s2_t).argmax(dim=1, keepdim=True)
            next_q = self.target(s2_t).gather(1, next_actions)
            target = r_t + m_t * self.cfg.gamma * next_q

        loss = nn.functional.mse_loss(q_sa, target)

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optim.step()

        self._decay_eps()
        if self.steps % self.cfg.target_update == 0:
            self.update_target()

    def update_target(self, hard: bool = True, tau: float = 0.005):
        if hard:
            self.target.load_state_dict(self.online.state_dict())
        else:
            with torch.no_grad():
                for p_t, p in zip(self.target.parameters(), self.online.parameters()):
                    p_t.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path: str):
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])

    def _decay_eps(self):
        self.exploration_rate = max(self.cfg.eps_end, self.exploration_rate * self.cfg.eps_decay)
        self.steps += 1

