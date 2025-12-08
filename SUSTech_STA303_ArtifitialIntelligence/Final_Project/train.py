"""
CartPole Training & Evaluation (PyTorch + Gymnasium)
---------------------------------------------------
- Supports multiple agents (DQN, Double DQN, PPO, A2C, ...)
- Logs scores via ScoreLogger (PNG + CSV)
- Saves models under ./models/<algo>.torch

Student reading map:
  1) train(): env loop → agent.act() → env.step() → agent.step() [Encapsulated]
  2) evaluate(): loads saved model and runs agent.act(evaluation_mode=True)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Type

import gymnasium as gym
import numpy as np

from agents.cartpole_dqn import DQNSolver, DQNConfig
from agents.cartpole_double_dqn import DoubleDQNSolver, DoubleDQNConfig
from agents.cartpole_ppo import PPOSolver, PPOConfig
from agents.cartpole_a2c import A2CSolver, A2CConfig
from agents.cartpole_sac import SACSolver, SACConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"


class AgentEntry:
    """Registry entry storing solver/config classes and default filename."""

    def __init__(self, solver_cls: Type, cfg_cls: Type, model_name: str):
        self.solver_cls = solver_cls
        self.cfg_cls = cfg_cls
        self.model_name = model_name


AGENT_REGISTRY: Dict[str, AgentEntry] = {
    "dqn": AgentEntry(DQNSolver, DQNConfig, "cartpole_dqn.torch"),
    "ddqn": AgentEntry(DoubleDQNSolver, DoubleDQNConfig, "cartpole_double_dqn.torch"),
    "ppo": AgentEntry(PPOSolver, PPOConfig, "cartpole_ppo.torch"),
    "a2c": AgentEntry(A2CSolver, A2CConfig, "cartpole_a2c.torch"),
    "sac": AgentEntry(SACSolver, SACConfig, "cartpole_sac.torch"),
}


def _get_entry(algorithm: str) -> AgentEntry:
    key = algorithm.lower()
    if key not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[key]


def _default_model_path(algorithm: str) -> str:
    entry = _get_entry(algorithm)
    return os.path.join(MODEL_DIR, entry.model_name)


def train(num_episodes: int = 200,
          terminal_penalty: bool = True,
          algorithm: str = "dqn",
          load_path: str | None = None):
    """
    Main training loop (algorithm-agnostic):
      - Creates the environment and agent (DQN/PPO/etc)
      - For each episode:
          * Reset env → get initial state
          * Loop: select action, step environment, call agent.step()
          * Log episode score with ScoreLogger
      - Saves the trained model to disk
    """
    entry = _get_entry(algorithm)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)
    logger = ScoreLogger(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = entry.solver_cls(obs_dim, act_dim, cfg=entry.cfg_cls())
    device = getattr(agent, "device", "cpu")
    print(f"[Info] Training {algorithm.upper()} on device: {device}")
    if load_path:
        agent.load(load_path)
        print(f"[Info] Loaded weights from {load_path}")

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
            agent.step(state, action, reward, next_state, done)    #here we actually use experience replay to optimize the exsisting policy
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
    model_path = _default_model_path(algorithm)
    agent.save(model_path)
    print(f"[Train] {algorithm.upper()} model saved to {model_path}")
    return agent


def evaluate(model_path: str | None = None,
             algorithm: str = "dqn",
             episodes: int = 5,
             render: bool = True,
             fps: int = 60):
    """
    Evaluate a trained agent using deterministic actions.
    Args mirror train.py but now algorithm can be 'dqn', 'ppo', or 'a2c'.
    """
    entry = _get_entry(algorithm)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if model_path is None:
        model_path = _default_model_path(algorithm)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No saved {algorithm.upper()} model at '{model_path}'. Please train first or pass model_path."
            )
        print(f"[Eval] Using default {algorithm.upper()} model: {model_path}")
    else:
        print(f"[Eval] Using provided model: {model_path}")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = entry.solver_cls(obs_dim, act_dim, cfg=entry.cfg_cls())
    agent.load(model_path)
    print(f"[Eval] Loaded {algorithm.upper()} model from: {model_path}")

    scores = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=10_000 + ep)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/Evaluate CartPole agents (DQN, Double DQN, PPO, A2C, ...)")
    parser.add_argument("-a", "--algorithm", default="dqn", help=f"Agent type. Options: {list(AGENT_REGISTRY.keys())}")
    parser.add_argument("-n", "--train-episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--no-terminal-penalty", action="store_true", help="Disable -1 reward at episode end.")
    parser.add_argument("--load-model", default=None, help="Optional path to weights to load before training.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--eval-render", action="store_true", help="Render the environment during evaluation.")
    parser.add_argument("--eval-fps", type=int, default=60, help="FPS cap when rendering during evaluation.")
    args = parser.parse_args()

    train(
        num_episodes=args.train_episodes,
        terminal_penalty=not args.no_terminal_penalty,
        algorithm=args.algorithm,
        load_path=args.load_model,
    )
    if not args.skip_eval:
        evaluate(
            model_path=_default_model_path(args.algorithm),
            algorithm=args.algorithm,
            episodes=args.eval_episodes,
            render=args.eval_render,
            fps=args.eval_fps,
        )
        # 评估：python train.py -a ddqn -n 0 --eval-render --load-model models/cartpole_double_dqn.torch_2
        # 训练：python train.py --algorithm sac --train-episodes 1000 --skip-eval --load-model models/cartpole_double_dqn.torch_2
