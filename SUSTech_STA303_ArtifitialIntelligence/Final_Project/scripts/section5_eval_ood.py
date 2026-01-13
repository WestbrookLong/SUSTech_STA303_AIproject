"""
Evaluate BC/AWBC/CQL policies under in-distribution and OOD initial states.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from section5.awbc import load_policy as load_awbc_policy  # noqa: E402
from section5.bc import load_policy as load_bc_policy  # noqa: E402
from section5.cql import load_policy as load_cql_policy  # noqa: E402
from section5.common import set_seed_everywhere  # noqa: E402


class ResetRangeWrapper(gym.Wrapper):
    """Reset-time perturbation wrapper for CartPole-like tasks."""

    def __init__(self, env, theta_low: float, theta_high: float):
        super().__init__(env)
        self.theta_low = theta_low
        self.theta_high = theta_high

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            state = np.array(self.env.unwrapped.state, dtype=np.float32)
            if state.shape[0] >= 3:
                state[2] = np.random.uniform(self.theta_low, self.theta_high)
                self.env.unwrapped.state = state
                obs = state.copy()
                info["ood_reset"] = True
        except Exception as exc:  # pragma: no cover - fallback path
            info["ood_reset_error"] = str(exc)
        return obs, info


def evaluate(policy, env_id: str, episodes: int, seed: int) -> Tuple[float, float]:
    env = gym.make(env_id)
    max_steps = getattr(env.spec, "max_episode_steps", None)
    rewards = []
    successes = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            action = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            steps += 1
            if done:
                success = info.get("success") or info.get("is_success")
                if success is None and max_steps:
                    success = bool(truncated and not terminated and steps >= max_steps)
                if success is not None:
                    successes.append(bool(success))
        rewards.append(ep_ret)
    env.close()
    avg_ret = float(np.mean(rewards)) if rewards else 0.0
    success_rate = float(np.mean(successes)) if successes else 0.0
    return avg_ret, success_rate


def evaluate_pair(policy_loader, ckpt_path: str, env_id: str, episodes: int, seed: int, theta_low: float, theta_high: float):
    policy, ckpt_env = policy_loader(ckpt_path)
    env_id = ckpt_env or env_id
    id_return, id_success = evaluate(policy, env_id, episodes, seed)

    # OOD env
    env = ResetRangeWrapper(gym.make(env_id), theta_low=theta_low, theta_high=theta_high)
    rewards = []
    successes = []
    max_steps = getattr(env.spec, "max_episode_steps", None)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1_000 + ep)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            action = policy.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += float(reward)
            steps += 1
            if done:
                success = info.get("success") or info.get("is_success") or info.get("ood_reset")
                if success is None and max_steps:
                    success = bool(truncated and not terminated and steps >= max_steps)
                if success is not None:
                    successes.append(bool(success))
        rewards.append(ep_ret)
    env.close()
    ood_return = float(np.mean(rewards)) if rewards else 0.0
    ood_success = float(np.mean(successes)) if successes else 0.0
    return {
        "ckpt": ckpt_path,
        "env_id": env_id,
        "in_dist": {"avg_return": id_return, "success_rate": id_success},
        "ood": {"avg_return": ood_return, "success_rate": ood_success},
    }


def parse_args():
    parser = argparse.ArgumentParser(description="OOD evaluation for Section 5 policies.")
    parser.add_argument("--env_id", default="CartPole-v1", help="Environment id.")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per evaluation split.")
    parser.add_argument("--seed", type=int, default=321, help="Base seed.")
    parser.add_argument("--theta_low", type=float, default=-0.2, help="Lower bound for pole angle at reset.")
    parser.add_argument("--theta_high", type=float, default=0.2, help="Upper bound for pole angle at reset.")
    parser.add_argument("--bc_ckpt", default=None, help="Path to BC checkpoint.")
    parser.add_argument("--awbc_ckpt", default=None, help="Path to AWBC checkpoint.")
    parser.add_argument("--cql_ckpt", default=None, help="Path to CQL checkpoint.")
    parser.add_argument("--out", default=os.path.join("runs", "section5", "ood_results.json"), help="Where to append JSON results.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed_everywhere(args.seed)
    loaders = []
    if args.bc_ckpt:
        loaders.append(("bc", load_bc_policy, args.bc_ckpt))
    if args.awbc_ckpt:
        loaders.append(("awbc", load_awbc_policy, args.awbc_ckpt))
    if args.cql_ckpt:
        loaders.append(("cql", load_cql_policy, args.cql_ckpt))

    if not loaders:
        raise ValueError("Provide at least one checkpoint (--bc_ckpt/--awbc_ckpt/--cql_ckpt).")

    results = []
    for name, loader, ckpt in loaders:
        res = evaluate_pair(loader, ckpt, args.env_id, args.episodes, args.seed, args.theta_low, args.theta_high)
        res.update({"algorithm": name, "theta_range": [args.theta_low, args.theta_high], "timestamp": datetime.utcnow().isoformat()})
        print(f"[OOD] {name}: ID return={res['in_dist']['avg_return']:.2f}, OOD return={res['ood']['avg_return']:.2f}")
        results.append(res)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    existing: list = []
    if os.path.exists(args.out):
        try:
            with open(args.out, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.extend(results)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[OOD] Results appended to {args.out}")


if __name__ == "__main__":
    main()
