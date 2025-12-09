"""
CartPole PPO Training & Evaluation (PyTorch + Gymnasium)
--------------------------------------------------------
- Mirrors train.py structure but instantiates the PPO solver
- Logs episode scores via ScoreLogger and saves weights to ./models/cartpole_ppo.torch
"""

from __future__ import annotations

import os
import time
from typing import List

import gymnasium as gym
import numpy as np

from agents.cartpole_ppo import PPOSolver, PPOConfig
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cartpole_ppo.torch")


def train(num_episodes: int = 500, terminal_penalty: bool = True) -> PPOSolver:
    """
    Training entry point for PPO:
      - Creates the environment and PPO agent
      - Runs main interaction loop, feeding transitions to agent.step()
      - Logs episode scores and saves the trained policy/value networks
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = gym.make(ENV_NAME)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    cfg = getattr(agent, "cfg", None)
    if cfg is not None and hasattr(cfg, "__dict__"):
        hparams_text = ", ".join(f"{k}={v}" for k, v in cfg.__dict__.items())
    else:
        hparams_text = None
    logger = ScoreLogger(ENV_NAME, algorithm="ppo", hparams=hparams_text)
    print(f"[Info] PPO using device: {agent.device}")

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
                print(f"Run: {run}, Score: {steps}")
                logger.add_score(steps, run)
                break

    env.close()
    agent.save(MODEL_PATH)
    print(f"[Train] PPO model saved to {MODEL_PATH}")
    return agent


def evaluate(model_path: str | None = None,
             episodes: int = 5,
             render: bool = True,
             fps: int = 60) -> List[int]:
    """
    Evaluate a trained PPO agent with deterministic (greedy) actions.
    Args mirror train.py's evaluate() but default to PPO weights.
    """
    if model_path is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".torch")]
        if not candidates:
            raise FileNotFoundError("No saved PPO model found. Please train first.")
        model_path = os.path.join(MODEL_DIR, candidates[0])
        print(f"[Eval] Using detected PPO model: {model_path}")
    else:
        print(f"[Eval] Using provided PPO model: {model_path}")

    render_mode = "human" if render else None
    env = gym.make(ENV_NAME, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = PPOSolver(obs_dim, act_dim, cfg=PPOConfig())
    agent.load(model_path)
    print(f"[Eval] Loaded PPO weights from: {model_path}")

    scores: List[int] = []
    dt = (1.0 / fps) if render and fps else 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=20_000 + ep)
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
    # ppo_agent = train(num_episodes=500, terminal_penalty=True)
    evaluate(model_path=MODEL_PATH, episodes=50, render=True, fps=60)
