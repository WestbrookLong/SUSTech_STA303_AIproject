"""
Hyperparameter sweep script for CartPole agents.
------------------------------------------------
- Runs multiple training jobs over a predefined parameter grid.
- Does not modify existing project files; only uses train.train() and config classes.

Usage:
    python script/hparam_sweep.py --algorithm ddqn --episodes 300
    python script/hparam_sweep.py --algorithm ppo  --episodes 300
    python script/hparam_sweep.py --algorithm sac  --episodes 300
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Iterable, Tuple, Any, Callable

# Ensure project root (where train.py and agents/ live) is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import train
from agents.cartpole_double_dqn import DoubleDQNConfig
from agents.cartpole_ppo import PPOConfig
from agents.cartpole_sac import SACConfig


# -----------------------------
# Parameter grids per algorithm
# -----------------------------

@dataclass
class DDQNParamGrid:
    """Parameter grid for Double DQN (ddqn)."""

    gamma: Tuple[float, ...] = (0.99, 0.95)
    lr: Tuple[float, ...] = (1e-3, 5e-4)
    batch_size: Tuple[int, ...] = (32, 64)
    eps_decay: Tuple[float, ...] = (0.995, 0.99)


@dataclass
class PPOParamGrid:
    """Parameter grid for PPO (ppo)."""

    gamma: Tuple[float, ...] = (0.99, 0.97)
    gae_lambda: Tuple[float, ...] = (0.95, 0.9)
    clip_epsilon: Tuple[float, ...] = (0.2, 0.1)
    actor_lr: Tuple[float, ...] = (3e-4, 1e-4)
    critic_lr: Tuple[float, ...] = (1e-3, 5e-4)


@dataclass
class SACParamGrid:
    """Parameter grid for SAC (sac)."""

    gamma: Tuple[float, ...] = (0.99, 0.95)
    alpha: Tuple[float, ...] = (0.2, 0.05)
    q_lr: Tuple[float, ...] = (3e-4, 1e-4)
    pi_lr: Tuple[float, ...] = (3e-4, 1e-4)


def _iter_param_combinations(grid) -> Iterable[Dict[str, Any]]:
    """Yield dicts of all combinations from a dataclass grid."""
    grid_dict = asdict(grid)
    keys = list(grid_dict.keys())
    value_lists = [grid_dict[k] for k in keys]
    for values in product(*value_lists):
        yield dict(zip(keys, values))


def _get_grid_and_cfg(algorithm: str) -> Tuple[Callable[..., Any], Any]:
    algo = algorithm.lower()
    if algo == "ddqn":
        return DoubleDQNConfig, DDQNParamGrid()
    if algo == "ppo":
        return PPOConfig, PPOParamGrid()
    if algo == "sac":
        return SACConfig, SACParamGrid()
    raise ValueError(f"Unsupported algorithm for sweep: {algorithm}. Use one of: ddqn, ppo, sac.")


def run_sweep(algorithm: str, episodes: int):
    """Run train.train() once for every parameter combination in the grid."""
    algo = algorithm.lower()
    cfg_cls, grid = _get_grid_and_cfg(algo)

    # Access the registry entry used by train.py and temporarily override cfg_cls
    entry = train.AGENT_REGISTRY[algo]
    original_cfg_cls = entry.cfg_cls

    try:
        for idx, params in enumerate(_iter_param_combinations(grid), start=1):
            entry.cfg_cls = lambda **kw: cfg_cls(**{**params, **kw})

            print(
                f"\n=== Sweep run {idx} for algorithm={algo.upper()} ===\n"
                f"Hyperparameters: {params}\n"
                f"Episodes per run: {episodes}\n"
            )

            # Call into the existing training loop; terminal_penalty kept True by default.
            train.train(
                num_episodes=episodes,
                terminal_penalty=True,
                algorithm=algo,
            )
    finally:
        # Restore original factory to avoid side-effects if this module is reused
        entry.cfg_cls = original_cfg_cls


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep over CartPole agents.")
    parser.add_argument(
        "-a",
        "--algorithm",
        required=True,
        help="Algorithm to sweep over. Options: ddqn, ppo, sac",
    )
    parser.add_argument(
        "-n",
        "--episodes",
        type=int,
        required=True,
        help="Number of episodes per training run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args.algorithm, args.episodes)
