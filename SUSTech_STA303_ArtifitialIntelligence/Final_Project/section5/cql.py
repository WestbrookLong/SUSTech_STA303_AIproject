"""
Conservative Q-Learning (CQL) implementation for offline RL.
Uses double Q critics with a conservative penalty on out-of-distribution actions.
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
from section5.common import ensure_dir, evaluate_policy, get_device, make_env, set_seed_everywhere
from section5.dataset import OfflineDataset
from section5.logger import MetricLogger
from section5.policies import ContinuousQNetwork, DiscreteActor, DiscreteQNetwork, GaussianActor


@dataclass
class CQLConfig:
    env_id: str = "CartPole-v1"
    dataset_path: str = ""
    batch_size: int = 256
    epochs: int = 50
    lr: float = 3e-4
    hidden_dim: int = 128
    gamma: float = 0.99
    alpha_cql: float = 1.0  # weight for conservative penalty
    temperature: float = 0.8
    num_random_actions: int = 10
    entropy_alpha: float = 0.2
    target_update_interval: int = 1
    tau: float = 0.005
    eval_interval: int = 5
    eval_episodes: int = 10
    seed: int = 42
    log_dir: str = os.path.join(RUNS_DIR, "cql")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CQLTrainer:
    def __init__(self, cfg: CQLConfig):
        self.cfg = cfg
        set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.dataset = OfflineDataset.from_file(cfg.dataset_path)
        env = make_env(cfg.env_id, seed=cfg.seed)
        self.env = env
        self.is_discrete = hasattr(env.action_space, "n")
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n if self.is_discrete else env.action_space.shape[0]
        self.action_low = torch.as_tensor(env.action_space.low, device=self.device) if not self.is_discrete else None
        self.action_high = torch.as_tensor(env.action_space.high, device=self.device) if not self.is_discrete else None

        # Policy
        if self.is_discrete:
            self.policy = DiscreteActor(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q1 = DiscreteQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q2 = DiscreteQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q1_target = DiscreteQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q2_target = DiscreteQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
        else:
            self.policy = GaussianActor(
                self.obs_dim, self.act_dim, env.action_space.low, env.action_space.high, hidden_dim=cfg.hidden_dim
            ).to(self.device)
            self.q1 = ContinuousQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q2 = ContinuousQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q1_target = ContinuousQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)
            self.q2_target = ContinuousQNetwork(self.obs_dim, self.act_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)

        self.run_dir = os.path.join(cfg.log_dir, os.path.basename(cfg.dataset_path).replace(".pt", ""))
        ensure_dir(self.run_dir)
        self.logger = MetricLogger(self.run_dir)
        self.ckpt_path = os.path.join(self.run_dir, "cql_policy.pt")
        with open(os.path.join(self.run_dir, "run_config.json"), "w") as f:
            json.dump({"algorithm": "cql", **asdict(cfg), "dataset_metadata": self.dataset.metadata}, f, indent=2)

    def _critic_loss_discrete(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].long().to(self.device).unsqueeze(1)
        rewards = batch["rewards"].to(self.device).unsqueeze(1)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device).float().unsqueeze(1)

        with torch.no_grad():
            logits_next = self.policy(next_obs)
            probs_next = torch.softmax(logits_next, dim=1)
            log_probs_next = torch.log_softmax(logits_next, dim=1)
            q1_next = self.q1_target(next_obs)
            q2_next = self.q2_target(next_obs)
            min_q_next = torch.min(q1_next, q2_next)
            v_next = (probs_next * (min_q_next - self.cfg.entropy_alpha * log_probs_next)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.cfg.gamma * v_next

        q1_pred_all = self.q1(obs)
        q2_pred_all = self.q2(obs)
        q1_pred = q1_pred_all.gather(1, actions)
        q2_pred = q2_pred_all.gather(1, actions)
        bellman = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        cql1 = (torch.logsumexp(q1_pred_all / self.cfg.temperature, dim=1) * self.cfg.temperature - q1_pred.squeeze(1)).mean()
        cql2 = (torch.logsumexp(q2_pred_all / self.cfg.temperature, dim=1) * self.cfg.temperature - q2_pred.squeeze(1)).mean()
        critic_loss = bellman + self.cfg.alpha_cql * (cql1 + cql2)
        return critic_loss, bellman

    def _critic_loss_continuous(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).unsqueeze(1)
        next_obs = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device).float().unsqueeze(1)

        with torch.no_grad():
            next_actions, next_logp = self.policy.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = rewards + (1 - dones) * self.cfg.gamma * (min_q_next - self.cfg.entropy_alpha * next_logp.unsqueeze(1))

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        bellman = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        batch_size = obs.shape[0]
        num_rand = self.cfg.num_random_actions
        rand_actions = torch.empty((batch_size, num_rand, self.act_dim), device=self.device).uniform_(self.action_low, self.action_high)

        # current policy actions
        pi_actions, _ = self.policy.sample(obs)
        next_pi_actions, _ = self.policy.sample(next_obs)

        def _stack_q(q_net):
            q_rand = q_net(obs.unsqueeze(1).expand(-1, num_rand, -1).reshape(-1, self.obs_dim),
                           rand_actions.reshape(-1, self.act_dim)).view(batch_size, num_rand)
            q_pi = q_net(obs, pi_actions).unsqueeze(1)
            q_next = q_net(next_obs, next_pi_actions).unsqueeze(1)
            cat = torch.cat([q_rand, q_pi, q_next], dim=1)
            return torch.logsumexp(cat / self.cfg.temperature, dim=1) * self.cfg.temperature

        logsumexp1 = _stack_q(self.q1)
        logsumexp2 = _stack_q(self.q2)
        cql1 = (logsumexp1 - q1_pred.squeeze(1)).mean()
        cql2 = (logsumexp2 - q2_pred.squeeze(1)).mean()
        critic_loss = bellman + self.cfg.alpha_cql * (cql1 + cql2)
        return critic_loss, bellman

    def _actor_loss_discrete(self, obs: torch.Tensor) -> torch.Tensor:
        # Discrete SAC-style policy update (exact expectation over actions):
        #   J_pi = E_s[ Σ_a π(a|s) * (α * log π(a|s) - min(Q1,Q2)(s,a)) ]
        logits = self.policy(obs)
        log_probs = torch.log_softmax(logits, dim=1)  # [B, A]
        probs = torch.softmax(logits, dim=1)          # [B, A]

        q1 = self.q1(obs)
        q2 = self.q2(obs)
        min_q = torch.min(q1, q2)                     # [B, A]

        inside = self.cfg.entropy_alpha * log_probs - min_q
        return (probs * inside).sum(dim=1).mean()

    def _actor_loss_continuous(self, obs: torch.Tensor) -> torch.Tensor:
        actions, logp = self.policy.sample(obs)
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        min_q = torch.min(q1, q2)
        return (self.cfg.entropy_alpha * logp.unsqueeze(1) - min_q).mean()

    def train(self) -> Tuple[float, float]:
        dataloader = DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
        global_step = 0
        last_eval = (0.0, None)
        for epoch in range(1, self.cfg.epochs + 1):
            for batch in dataloader:
                if self.is_discrete:
                    critic_loss, bellman = self._critic_loss_discrete(batch)
                else:
                    critic_loss, bellman = self._critic_loss_continuous(batch)
                self.q_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 5.0)
                self.q_optim.step()

                obs = batch["observations"].to(self.device)
                if self.is_discrete:
                    actor_loss = self._actor_loss_discrete(obs)
                else:
                    actor_loss = self._actor_loss_continuous(obs)
                self.policy_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
                self.policy_optim.step()

                global_step += 1
                self.logger.log(
                    global_step,
                    {
                        "critic_loss": float(critic_loss.item()),
                        "bellman_loss": float(bellman.item()),
                        "actor_loss": float(actor_loss.item()),
                    },
                )

                if global_step % self.cfg.target_update_interval == 0:
                    with torch.no_grad():
                        for tgt, src in zip(self.q1_target.parameters(), self.q1.parameters()):
                            tgt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * src.data)
                        for tgt, src in zip(self.q2_target.parameters(), self.q2.parameters()):
                            tgt.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * src.data)

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
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "config": asdict(self.cfg),
                "metadata": self.dataset.metadata,
                "env_id": self.cfg.env_id,
                "is_discrete": self.is_discrete,
            },
            path,
        )
        print(f"[CQL] Saved policy to {path}")


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
