"""
Unified training entry for CartPole agents (DQN, PPO, ...).
---------------------------------------------------------
- Mirrors train.py/train_ppo.py loops but lets you switch algorithms via CLI/args.
- Saves models under ./models/ with algorithm-specific filenames.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Type

import gymnasium as gym
import numpy as np

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"


class AgentEntry:
    """Small helper container for registry entries."""

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


def train(algorithm: str = "dqn", num_episodes: int = 500, terminal_penalty: bool = True):
    """Generic training loop that instantiates the requested solver and runs CartPole episodes."""
    entry = _get_entry(algorithm)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = entry.solver_cls(obs_dim, act_dim, cfg=entry.cfg_cls())
    device = getattr(agent, "device", "cpu")
    print(f"[Train] Algorithm={algorithm.upper()} using device: {device}")

    for run in range(1, num_episodes + 1):
        state, info = env.reset(seed=run)
        state = np.reshape(state, (1, obs_dim))
        steps = 0

        while True:
            steps += 1
            action = agent.act(state)

            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if terminal_penalty and done:
                reward = -1.0

            next_state = np.reshape(next_state_raw, (1, obs_dim))
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                epsilon = getattr(agent, "exploration_rate", None)
                if epsilon is not None:
                    print(f"Run: {run}, Score: {steps}, Epsilon: {epsilon:.3f}")
                else:
                    print(f"Run: {run}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()
    model_path = os.path.join(MODEL_DIR, entry.model_name)
    agent.save(model_path)
    print(f"[Train] Saved {algorithm.upper()} model to {model_path}")
    return agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train CartPole agents with a unified entry point.")
    parser.add_argument("-a", "--algorithm", default="dqn", help=f"Agent to train. Options: {list(AGENT_REGISTRY.keys())}")
    parser.add_argument("-n", "--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--no-terminal-penalty", action="store_true", help="Disable the -1 reward when an episode ends.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        terminal_penalty=not args.no_terminal_penalty,
    )

