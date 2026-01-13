"""
Train Conservative Q-Learning (CQL) on an offline dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from section5.cql import CQLConfig, CQLTrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train CQL on an offline dataset.")
    parser.add_argument("--dataset", required=True, help="Path to offline dataset (.pt).")
    parser.add_argument("--env_id", default="CartPole-v1", help="Environment id.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha_cql", type=float, default=1.0, help="Weight for conservative penalty.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature inside logsumexp.")
    parser.add_argument("--num_random_actions", type=int, default=10, help="Random actions for CQL penalty (continuous).")
    parser.add_argument("--entropy_alpha", type=float, default=0.2, help="Entropy temperature for policy loss.")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging factor.")
    parser.add_argument("--target_update_interval", type=int, default=1, help="Steps between target updates.")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--log_dir", default=None, help="Optional override for log directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CQLConfig(
        env_id=args.env_id,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        alpha_cql=args.alpha_cql,
        temperature=args.temperature,
        num_random_actions=args.num_random_actions,
        entropy_alpha=args.entropy_alpha,
        tau=args.tau,
        target_update_interval=args.target_update_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir or CQLConfig.log_dir,
    )
    trainer = CQLTrainer(cfg)
    avg_return, success_rate = trainer.train()
    print(f"[CQL] Done. Eval return={avg_return:.2f}, success={success_rate}")


if __name__ == "__main__":
    main()
