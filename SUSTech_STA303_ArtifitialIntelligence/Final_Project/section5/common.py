"""
Shared utilities for Section 5 (offline RL + imitation).
- Seeding helpers
- Env creation
- Evaluation loop with optional success metric
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

# Gym success keys we may encounter
SUCCESS_KEYS = ("success", "is_success", "solved")


def set_seed_everywhere(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_env(env_id: str, seed: Optional[int] = None, render: bool = False) -> gym.Env:
    """Create a Gymnasium environment with optional seed and render mode."""
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def _success_flag(info: Dict[str, Any], terminated: bool, truncated: bool, steps: int, max_steps: Optional[int]) -> Optional[bool]:
    """Derive a success boolean if possible."""
    for key in SUCCESS_KEYS:
        if key in info:
            val = info[key]
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            try:
                return bool(int(val))
            except Exception:
                continue
    # For CartPole-like tasks, lasting the full horizon counts as success
    if max_steps is not None and truncated and not terminated and steps >= max_steps:
        return True
    return None


def evaluate_policy(
    policy: Any,
    env_id: str,
    episodes: int = 5,
    seed: int = 42,
    render: bool = False,
    deterministic: bool = True,
) -> Tuple[float, Optional[float]]:
    """
    Roll out a policy in the env and return (avg_return, success_rate or None).
    The policy is expected to expose `.act(obs, deterministic=True/False) -> action`.
    """
    env = make_env(env_id, seed=seed, render=render)
    max_steps = getattr(env.spec, "max_episode_steps", None)
    returns = []
    successes = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            action = policy.act(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            steps += 1
            if done:
                success = _success_flag(info, terminated, truncated, steps, max_steps)
                if success is not None:
                    successes.append(success)
            obs = next_obs
        returns.append(ep_ret)
    env.close()
    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = None
    if successes:
        success_rate = float(np.mean(successes))
    return avg_return, success_rate


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
