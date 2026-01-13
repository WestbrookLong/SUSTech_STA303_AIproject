"""
Offline dataset utilities for Section 5.
Datasets are stored as .pt files with fields:
  observations, actions, rewards, next_observations, dones, episode_starts
and a metadata dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


def compute_returns_to_go(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Compute discounted return-to-go for each timestep.
    Resets when dones[t] == True (end of episode).
    """
    returns = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.zeros(1, dtype=torch.float32)
    for t in range(len(rewards) - 1, -1, -1):
        if dones[t]:
            running = torch.zeros(1, dtype=torch.float32)
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


@dataclass
class OfflineBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    episode_starts: torch.Tensor
    returns_to_go: Optional[torch.Tensor] = None


class OfflineDataset(Dataset):
    """Lightweight Dataset wrapper around tensors saved in .pt files."""

    def __init__(self, batch: OfflineBatch, metadata: Optional[Dict[str, Any]] = None):
        self.batch = batch
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        episode_starts = data.get("episode_starts", None)
        if episode_starts is None:
            if "episode_ids" in data:
                episode_ids = data["episode_ids"].to(torch.long)
                episode_starts = torch.zeros_like(episode_ids, dtype=torch.bool)
                if episode_ids.numel() > 0:
                    episode_starts[0] = True
                    episode_starts[1:] = episode_ids[1:] != episode_ids[:-1]
            else:
                episode_starts = torch.zeros_like(data["dones"], dtype=torch.bool)
        else:
            episode_starts = episode_starts.to(torch.bool)
        batch = OfflineBatch(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            next_observations=data["next_observations"],
            dones=data["dones"],
            episode_starts=episode_starts,
            returns_to_go=data.get("returns_to_go"),
        )
        return cls(batch=batch, metadata=data.get("metadata", {}))

    @classmethod
    def from_file(cls, path: str):
        data = torch.load(path, map_location="cpu")
        return cls.from_dict(data)

    def attach_returns(self, gamma: float):
        if self.batch.returns_to_go is None:
            self.batch.returns_to_go = compute_returns_to_go(self.batch.rewards, self.batch.dones, gamma)

    def __len__(self):
        return self.batch.observations.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "observations": self.batch.observations[idx],
            "actions": self.batch.actions[idx],
            "rewards": self.batch.rewards[idx],
            "next_observations": self.batch.next_observations[idx],
            "dones": self.batch.dones[idx],
            "episode_starts": self.batch.episode_starts[idx],
        }
        if self.batch.returns_to_go is not None:
            item["returns_to_go"] = self.batch.returns_to_go[idx]
        return item

    @property
    def obs_dim(self) -> int:
        return int(self.batch.observations.shape[-1])

    @property
    def action_shape(self) -> Tuple[int, ...]:
        return tuple(self.batch.actions.shape[1:]) if self.batch.actions.ndim > 1 else ()

    @property
    def is_discrete(self) -> bool:
        # Metadata can store 'action_space': 'discrete'/'box'
        action_space = self.metadata.get("action_space")
        if action_space:
            return action_space == "discrete"
        # Fallback: int dtype implies discrete
        return self.batch.actions.dtype in (torch.int, torch.int32, torch.int64)
