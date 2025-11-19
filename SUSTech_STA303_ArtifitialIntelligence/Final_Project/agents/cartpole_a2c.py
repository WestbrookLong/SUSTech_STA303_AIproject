"""
PyTorch Advantage Actor-Critic (A2C) for CartPole.
-------------------------------------------------
- Shares the same public API as other agents (act, step, save, load, update_target).
- Uses a shared backbone network with separate policy/logit and value heads.
- Performs on-policy TD(0) updates each environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------
# Default Hyperparameters
# -----------------------------
GAMMA = 0.99
LR = 3e-3
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5


class ActorCriticNet(nn.Module):
    """Shared MLP that outputs policy logits and state-value estimate."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


@dataclass
class A2CConfig:
    """Configuration values for the A2C solver."""

    gamma: float = GAMMA
    lr: float = LR
    value_coef: float = VALUE_COEF
    entropy_coef: float = ENTROPY_COEF
    max_grad_norm: float = MAX_GRAD_NORM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class A2CSolver:
    """
    Advantage Actor-Critic agent compatible with train.py.

    Public API:
        act(), step(), save(), load(), update_target()
    """

    def __init__(self, observation_space: int, action_space: int, cfg: A2CConfig | None = None):
        self.obs_dim = observation_space
        self.act_dim = action_space
        self.cfg = cfg or A2CConfig()

        self.device = torch.device(self.cfg.device)
        self.model = ActorCriticNet(self.obs_dim, self.act_dim).to(self.device)
        self.optim = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.steps = 0

    # -----------------------------
    # Acting
    # -----------------------------
    def act(self, state_np: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Select action via stochastic policy; greedy argmax when evaluation_mode=True.
        """
        state_t = self._state_to_tensor(state_np)
        logits, _ = self.model(state_t)
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
        """
        Perform one TD(0) update using the recent transition.
        """
        state_vec = self._ensure_vector(state)
        next_state_vec = self._ensure_vector(next_state)

        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = torch.as_tensor(next_state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor([action], dtype=torch.int64, device=self.device)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)
        mask = torch.tensor([0.0 if done else 1.0], dtype=torch.float32, device=self.device)

        logits, value = self.model(state_t)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()

        with torch.no_grad():
            _, next_value = self.model(next_state_t)
            target_value = reward_t + self.cfg.gamma * next_value.squeeze(-1) * mask

        advantage = target_value - value.squeeze(-1)
        policy_loss = -(log_prob * advantage.detach()).mean()
        value_loss = nn.functional.mse_loss(value.squeeze(-1), target_value)
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optim.step()
        self.steps += 1

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: str):
        torch.save(
            {
                "model": self.model.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

    def update_target(self, hard: bool = True, tau: float = 0.0):
        """A2C does not use target networks; method exists for interface parity."""
        _ = hard, tau

    # -----------------------------
    # Helpers
    # -----------------------------
    def _state_to_tensor(self, state_np: np.ndarray) -> torch.Tensor:
        state_vec = self._ensure_vector(state_np)
        return torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

    @staticmethod
    def _ensure_vector(state_np: np.ndarray) -> np.ndarray:
        state = np.asarray(state_np, dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        return state

