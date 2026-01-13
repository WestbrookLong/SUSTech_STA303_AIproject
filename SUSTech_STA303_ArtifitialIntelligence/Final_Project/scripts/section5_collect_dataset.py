"""
Collect offline datasets from a trained SAC (or other) policy.
Produces D-Expert (prand=0) or D-Mixed (prand>0) datasets.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List

import gymnasium as gym
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from agents.cartpole_sac import SACConfig, SACSolver  # noqa: E402
from section5 import DATA_DIR  # noqa: E402
from section5.common import ensure_dir, set_seed_everywhere  # noqa: E402


def collect_dataset(env_id: str, ckpt: str, out_path: str, steps: int, prand: float, seed: int, gamma: float):
    set_seed_everywhere(seed)
    ensure_dir(os.path.dirname(out_path))

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    is_discrete = hasattr(env.action_space, "n")
    if not is_discrete:
        raise ValueError("This script currently assumes a discrete action space (CartPole).")
    act_dim = env.action_space.n

    policy = SACSolver(obs_dim, act_dim, cfg=SACConfig())
    policy.load(ckpt)
    print(f"[Collect] Loaded expert policy from {ckpt}")

    observations: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    next_observations: List[np.ndarray] = []
    dones: List[bool] = []
    episode_starts: List[bool] = []
    episode_ids: List[int] = []
    timeouts: List[bool] = []

    obs, _ = env.reset(seed=seed)
    episode_id = 0
    start_flag = True
    total_steps = 0
    while total_steps < steps:
        if np.random.rand() < prand:
            action = env.action_space.sample()
        else:
            action = policy.act(obs, evaluation_mode=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(int(action))
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_obs, dtype=np.float32))
        dones.append(done)
        episode_starts.append(start_flag)
        episode_ids.append(episode_id)
        timeouts.append(bool(truncated))

        total_steps += 1
        if done:
            episode_id += 1
            obs, _ = env.reset(seed=seed + episode_id)
            start_flag = True
        else:
            obs = next_obs
            start_flag = False

    env.close()

    data = {
        "observations": torch.tensor(np.stack(observations), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "next_observations": torch.tensor(np.stack(next_observations), dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.bool),
        "episode_starts": torch.tensor(episode_starts, dtype=torch.bool),
        "episode_ids": torch.tensor(episode_ids, dtype=torch.long),
        "timeouts": torch.tensor(timeouts, dtype=torch.bool),
        "metadata": {
            "env_id": env_id,
            "seed": seed,
            "gamma": gamma,
            "prand": prand,
            "steps": total_steps,
            "policy_ckpt_path": os.path.abspath(ckpt),
            "timestamp": datetime.utcnow().isoformat(),
            "action_space": "discrete",
        },
    }
    torch.save(data, out_path)
    print(f"[Collect] Saved dataset with {total_steps} transitions to {out_path}")
    print(f"[Collect] Episodes: {episode_id + 1}, prand={prand}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect offline dataset from a trained SAC policy.")
    parser.add_argument("--env_id", default="CartPole-v1", help="Gymnasium environment id.")
    parser.add_argument("--ckpt", default=os.path.join("models", "cartpole_sac_2.torch"), help="Path to expert checkpoint.")
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "cartpole_d_expert.pt"), help="Output dataset path (.pt).")
    parser.add_argument("--steps", type=int, default=50_000, help="Number of transitions to collect.")
    parser.add_argument("--prand", type=float, default=0.0, help="Random action probability (0=expert, 0.5=mixed).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for env and sampling.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor stored in metadata.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_dataset(
        env_id=args.env_id,
        ckpt=args.ckpt,
        out_path=args.out,
        steps=args.steps,
        prand=args.prand,
        seed=args.seed,
        gamma=args.gamma,
    )
