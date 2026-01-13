"""
Train Advantage-Weighted Behavior Cloning (AWBC) on an offline dataset.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from section5.awbc import AWBCConfig, AWBCTrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train AWBC on an offline dataset.")
    parser.add_argument("--dataset", required=True, help="Path to offline dataset (.pt).")
    parser.add_argument("--env_id", default="CartPole-v1", help="Environment id.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=1.0, help="Temperature in exp((G-b)/beta).")
    parser.add_argument("--w_clip", type=float, default=20.0, help="Clip for AWBC weights.")
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--log_dir", default=None, help="Optional override for log directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = AWBCConfig(
        env_id=args.env_id,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        w_clip=args.w_clip,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
        log_dir=args.log_dir or AWBCConfig.log_dir,
    )
    trainer = AWBCTrainer(cfg)
    avg_return, success_rate = trainer.train()
    print(f"[AWBC] Done. Eval return={avg_return:.2f}, success={success_rate}")


if __name__ == "__main__":
    main()
