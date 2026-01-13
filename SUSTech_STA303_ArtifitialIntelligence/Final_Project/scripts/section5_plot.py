"""
Plot learning curves and OOD bars for Section 5 experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_run(run_dir: str) -> Tuple[Dict, List[Tuple[float, float]]]:
    cfg_path = os.path.join(run_dir, "run_config.json")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if not (os.path.exists(cfg_path) and os.path.exists(metrics_path)):
        return {}, []
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    steps, returns = [], []
    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "eval_return" in row and row["eval_return"]:
                steps.append(float(row["step"]))
                returns.append(float(row["eval_return"]))
    return cfg, list(zip(steps, returns))


def _dataset_tag(cfg: Dict) -> str:
    meta = cfg.get("dataset_metadata", {})
    prand = meta.get("prand", None)
    if prand is None:
        name = os.path.basename(cfg.get("dataset_path", "dataset"))
        return "D-Mixed" if "mixed" in name.lower() else "D-Expert"
    return "D-Mixed" if prand > 0 else "D-Expert"


def plot_learning_curves(log_root: str, output_dir: str):
    curves = defaultdict(list)
    for algo in ("bc", "awbc", "cql"):
        algo_dir = os.path.join(log_root, algo)
        if not os.path.isdir(algo_dir):
            continue
        for run_name in os.listdir(algo_dir):
            run_dir = os.path.join(algo_dir, run_name)
            cfg, data = _load_run(run_dir)
            if not data:
                continue
            tag = _dataset_tag(cfg)
            curves[(algo, tag)].append((run_name, data))

    ensure_dir(output_dir)
    for tag in ("D-Expert", "D-Mixed"):
        plt.figure()
        for algo in ("bc", "awbc", "cql"):
            key = (algo, tag)
            if key not in curves:
                continue
            # Use the latest run
            run_name, data = sorted(curves[key], key=lambda x: x[0])[-1]
            steps, returns = zip(*data)
            plt.plot(steps, returns, label=f"{algo.upper()} ({run_name})")
        plt.xlabel("Training step")
        plt.ylabel("Avg return")
        plt.title(f"{tag} dataset learning curves")
        plt.legend()
        out_path = os.path.join(output_dir, f"learning_{tag.lower().replace('-', '_')}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path}")


def plot_ood_bars(ood_path: str, output_dir: str):
    if not os.path.exists(ood_path):
        print(f"[Plot] OOD results file not found: {ood_path}")
        return
    with open(ood_path, "r") as f:
        results = json.load(f)
    if not results:
        return
    algos = []
    id_returns = []
    ood_returns = []
    for entry in results:
        algos.append(entry.get("algorithm", "algo"))
        id_returns.append(entry["in_dist"]["avg_return"])
        ood_returns.append(entry["ood"]["avg_return"])

    x = range(len(algos))
    width = 0.35
    ensure_dir(output_dir)
    plt.figure()
    plt.bar([i - width / 2 for i in x], id_returns, width=width, label="In-distribution")
    plt.bar([i + width / 2 for i in x], ood_returns, width=width, label="OOD")
    plt.xticks(list(x), [a.upper() for a in algos])
    plt.ylabel("Average return")
    plt.title("OOD performance comparison")
    plt.legend()
    out_path = os.path.join(output_dir, "ood_bars.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {out_path}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Section 5 results.")
    parser.add_argument("--log_root", default=os.path.join("runs", "section5"), help="Root directory containing bc/awbc/cql runs.")
    parser.add_argument("--output_dir", default=os.path.join("runs", "section5", "figures"), help="Where to save figures.")
    parser.add_argument("--ood_path", default=os.path.join("runs", "section5", "ood_results.json"), help="Path to OOD results JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_learning_curves(args.log_root, args.output_dir)
    plot_ood_bars(args.ood_path, args.output_dir)


if __name__ == "__main__":
    main()
