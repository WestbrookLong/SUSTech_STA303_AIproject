"""
Policy and Q-network definitions for Section 5 algorithms.
Supports discrete (categorical) and continuous (tanh-Gaussian) actions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def mlp(in_dim: int, out_dim: int, hidden_dim: int = 128, depth: int = 2, activation=nn.ReLU) -> nn.Sequential:
    layers = []
    last_dim = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, out_dim))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    return net


class DiscreteActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_dim=hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def act(self, obs_np, deterministic: bool = True) -> int:
        device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.sample(obs_t, deterministic=deterministic)
        return int(action.item())


class GaussianActor(nn.Module):
    """
    Tanh-squashed Gaussian policy for continuous actions.
    Actions are scaled to [low, high].
    """

    def __init__(self, obs_dim: int, act_dim: int, action_low, action_high, hidden_dim: int = 256, log_std_bounds=(-5, 2)):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden_dim=hidden_dim, depth=2)
        self.log_std_bounds = log_std_bounds
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        act_dim = out.shape[-1] // 2
        mean, log_std = out[..., :act_dim], out[..., act_dim:]
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        return mean, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        tanh_action = torch.tanh(action)
        log_prob = dist.log_prob(action) - torch.log(1 - tanh_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        scaled_action = self.action_low + (tanh_action + 1) * 0.5 * (self.action_high - self.action_low)
        return scaled_action, log_prob

    def act(self, obs_np, deterministic: bool = True):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.action_low.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.sample(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()


class DiscreteQNetwork(nn.Module):
    """Q(s, a) for all discrete actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden_dim=hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ContinuousQNetwork(nn.Module):
    """Q(s, a) for continuous actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden_dim=hidden_dim, depth=3)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
