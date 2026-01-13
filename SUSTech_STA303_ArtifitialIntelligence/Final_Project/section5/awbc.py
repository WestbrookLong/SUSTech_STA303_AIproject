"""
Advantage-Weighted Behavior Cloning (AWBC).
Weights log-likelihood by exp((G_t - b)/beta) with clipping.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from section5 import RUNS_DIR
from section5.bc import load_policy as load_bc_policy
from section5.common import ensure_dir, evaluate_policy, get_device, make_env, set_seed_everywhere
from section5.dataset import OfflineDataset
from section5.logger import MetricLogger
from section5.policies import DiscreteActor, GaussianActor


@dataclass
class AWBCConfig:
    env_id: str = "CartPole-v1"
    dataset_path: str = ""
    batch_size: int = 256
    epochs: int = 50
    lr: float = 3e-4
    hidden_dim: int = 128
    eval_interval: int = 5
    eval_episodes: int = 10
    beta: float = 1.0
    w_clip: float = 20.0
    seed: int = 42
    log_dir: str = os.path.join(RUNS_DIR, "awbc")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AWBCTrainer:
    def __init__(self, cfg: AWBCConfig):
        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.dataset = OfflineDataset.from_file(cfg.dataset_path)
        self.dataset.attach_returns(gamma=self.dataset.metadata.get("gamma", 0.99))
        env = make_env(cfg.env_id, seed=cfg.seed)
        self.env = env
        self.is_discrete = hasattr(env.action_space, "n")

        obs_dim = env.observation_space.shape[0]
        if self.is_discrete:
            act_dim = env.action_space.n
            self.policy = DiscreteActor(obs_dim, act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        else:
            low = env.action_space.low
            high = env.action_space.high
            act_dim = env.action_space.shape[0]
            self.policy = GaussianActor(obs_dim, act_dim, low, high, hidden_dim=cfg.hidden_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.run_dir = os.path.join(cfg.log_dir, os.path.basename(cfg.dataset_path).replace(".pt", ""))
        ensure_dir(self.run_dir)
        self.logger = MetricLogger(self.run_dir)
        self.ckpt_path = os.path.join(self.run_dir, "awbc_policy.pt")
        with open(os.path.join(self.run_dir, "run_config.json"), "w") as f:
            json.dump({"algorithm": "awbc", **asdict(cfg), "dataset_metadata": self.dataset.metadata}, f, indent=2)

    def _loss(self, batch) -> torch.Tensor:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        returns = batch["returns_to_go"].to(self.device)

        baseline = returns.mean()
        weights = torch.exp((returns - baseline) / self.cfg.beta)
        weights = torch.clamp(weights, max=self.cfg.w_clip)

        if self.is_discrete:
            logits = self.policy(obs)
            ce = F.cross_entropy(logits, actions.long(), reduction="none")
            loss = (weights * ce).mean()
        else:
            mean, log_std = self.policy(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            loss = -(weights * log_probs).mean()
        return loss

    def train(self) -> Tuple[float, float]:
        dataloader = DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        global_step = 0
        last_eval = (0.0, None)
        for epoch in range(1, self.cfg.epochs + 1):
            for batch in dataloader:
                loss = self._loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.optimizer.step()
                global_step += 1
                self.logger.log(global_step, {"train_loss": float(loss.item())})

            if epoch % self.cfg.eval_interval == 0:
                avg_return, success_rate = evaluate_policy(
                    self.policy, self.cfg.env_id, episodes=self.cfg.eval_episodes, seed=self.cfg.seed + epoch
                )
                metrics = {"eval_return": avg_return}
                if success_rate is not None:
                    metrics["eval_success"] = success_rate
                self.logger.log(global_step, metrics)
                last_eval = (avg_return, success_rate)

        self.save(self.ckpt_path)
        self.logger.close()
        return last_eval

    def save(self, path: str):
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "config": asdict(self.cfg),
                "metadata": self.dataset.metadata,
                "env_id": self.cfg.env_id,
                "is_discrete": self.is_discrete,
            },
            path,
        )
        print(f"[AWBC] Saved policy to {path}")


def load_policy(path: str, device: str | None = None):
    ckpt = torch.load(path, map_location=device or get_device())
    env_id = ckpt.get("env_id", "CartPole-v1")
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    device_final = torch.device(device) if device else get_device()
    if ckpt.get("is_discrete", True):
        act_dim = env.action_space.n
        policy = DiscreteActor(obs_dim, act_dim)
    else:
        act_dim = env.action_space.shape[0]
        policy = GaussianActor(obs_dim, act_dim, env.action_space.low, env.action_space.high)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device_final)
    policy.eval()
    return policy, env_id
