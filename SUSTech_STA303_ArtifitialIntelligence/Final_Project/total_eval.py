"""
Unified evaluation entry for CartPole agents (DQN, PPO, ...).
-----------------------------------------------------------
- Loads the requested agent weights and runs deterministic rollouts.
- Mirrors evaluate() logic from train.py/train_ppo.py but with algorithm switching.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Type

import gymnasium as gym
import numpy as np

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"


class AgentEntry:
    def __init__(self, solver_cls: Type, cfg_cls: Type, model_name: str):
        self.solver_cls = solver_cls
        self.cfg_cls = cfg_cls
        self.model_name = model_name


AGENT_REGISTRY: Dict[str, AgentEntry] = {
    "dqn": AgentEntry(DQNSolver, DQNConfig, "cartpole_dqn.torch"),
    "ppo": AgentEntry(PPOSolver, PPOConfig, "cartpole_ppo.torch"),
}


def _get_entry(algorithm: str) -> AgentEntry:
    key = algorithm.lower()
    if key not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[key]


def evaluate(algorithm: str = "dqn",
             model_path: str | None = None,
             episodes: int = 5,
             render: bool = False,
             fps: int = 60) -> List[int]:
    """Generic evaluation loop for the requested algorithm."""
    entry = _get_entry(algorithm)
    os.makedirs(MODEL_DIR, exist_ok=True)

    default_path = os.path.join(MODEL_DIR, entry.model_name)
    path = model_path or default_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' not found. Train {algorithm.upper()} first or pass --model-path.")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = entry.solver_cls(obs_dim, act_dim, cfg=entry.cfg_cls())
    agent.load(path)
    print(f"[Eval] Loaded {algorithm.upper()} weights from: {path}")

    scores: List[int] = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=50_000 + ep)
        state = np.reshape(state, (1, obs_dim))
        done = False
        steps = 0

        while not done:
            action = agent.act(state, evaluation_mode=True)
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.reshape(next_state, (1, obs_dim))
            steps += 1

            if dt > 0:
                time.sleep(dt)

        scores.append(steps)
        print(f"[Eval] Episode {ep}: steps={steps}")

    env.close()
    avg = float(np.mean(scores)) if scores else 0.0
    print(f"[Eval] Average over {episodes} episodes: {avg:.2f}")
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CartPole agents with a unified entry point.")
    parser.add_argument("-a", "--algorithm", default="dqn", help=f"Agent to evaluate. Options: {list(AGENT_REGISTRY.keys())}")
    parser.add_argument("-m", "--model-path", default=None, help="Path to .torch weights. Defaults to ./models/<algo>.torch")
    parser.add_argument("-n", "--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Enable human rendering window.")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS when rendering.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        algorithm=args.algorithm,
        model_path=args.model_path,
        episodes=args.episodes,
        render=args.render,
        fps=args.fps,
    )

