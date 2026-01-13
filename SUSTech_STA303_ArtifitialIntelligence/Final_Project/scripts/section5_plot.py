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
import numpy as np  # noqa: E402


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


def _group_key(entry: Dict):
    env_id = entry.get("env_id", "env")
    theta = entry.get("theta_range", None)
    theta_key = None
    if isinstance(theta, (list, tuple)) and len(theta) == 2:
        theta_key = (float(theta[0]), float(theta[1]))
    return env_id, theta_key


def _safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in str(s))


def _format_theta(theta_key) -> str:
    if theta_key is None:
        return "theta_unknown"
    lo, hi = theta_key
    return f"theta_{lo:.3f}_{hi:.3f}".replace("-", "m").replace(".", "p")


def _algo_order(algo: str) -> int:
    order = {"bc": 0, "awbc": 1, "cql": 2}
    return order.get(algo.lower(), 999)


def plot_ood_bars(ood_path: str, output_dir: str, mode: str = "aggregate"):
    if not os.path.exists(ood_path):
        print(f"[Plot] OOD results file not found: {ood_path}")
        return
    with open(ood_path, "r") as f:
        results = json.load(f)
    if not results:
        return

    ensure_dir(output_dir)

    if mode == "raw":
        algos = []
        id_returns = []
        ood_returns = []
        for entry in results:
            algos.append(entry.get("algorithm", "algo"))
            id_returns.append(entry["in_dist"]["avg_return"])
            ood_returns.append(entry["ood"]["avg_return"])

        x = np.arange(len(algos))
        width = 0.35
        plt.figure(figsize=(max(8, len(algos) * 0.6), 4))
        plt.bar(x - width / 2, id_returns, width=width, label="In-distribution")
        plt.bar(x + width / 2, ood_returns, width=width, label="OOD")
        plt.xticks(x, [a.upper() for a in algos], rotation=45, ha="right")
        plt.ylabel("Average return")
        plt.title("OOD performance comparison (raw)")
        plt.legend()
        out_path = os.path.join(output_dir, "ood_bars_raw.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved {out_path}")
        return

    # Aggregate by (env_id, theta_range) first; within each group, aggregate by algorithm.
    groups = defaultdict(list)
    for entry in results:
        groups[_group_key(entry)].append(entry)

    # If multiple theta ranges exist, emit one figure per group.
    for (env_id, theta_key), entries in groups.items():
        algos = sorted({e.get("algorithm", "algo") for e in entries}, key=_algo_order)
        algos_kept = []
        id_means, id_stds, ood_means, ood_stds, ns = [], [], [], [], []
        for algo in algos:
            id_vals = [float(e["in_dist"]["avg_return"]) for e in entries if e.get("algorithm", "") == algo]
            ood_vals = [float(e["ood"]["avg_return"]) for e in entries if e.get("algorithm", "") == algo]
            if not id_vals or not ood_vals:
                continue
            algos_kept.append(algo)
            id_means.append(float(np.mean(id_vals)))
            id_stds.append(float(np.std(id_vals)))
            ood_means.append(float(np.mean(ood_vals)))
            ood_stds.append(float(np.std(ood_vals)))
            ns.append(len(id_vals))

        x = np.arange(len(id_means))
        width = 0.35
        plt.figure(figsize=(max(7, len(x) * 1.6), 4))
        plt.bar(x - width / 2, id_means, width=width, yerr=id_stds, capsize=4, label="In-distribution")
        plt.bar(x + width / 2, ood_means, width=width, yerr=ood_stds, capsize=4, label="OOD")
        plt.xticks(x, [a.upper() for a in algos_kept])
        plt.ylabel("Average return")
        theta_text = f"[{theta_key[0]:.3f}, {theta_key[1]:.3f}]" if theta_key is not None else "unknown"
        plt.title(f"OOD performance (env={env_id}, theta={theta_text}, n={max(ns) if ns else 0})")
        plt.legend()

        # Keep a stable filename when there's only one group.
        if len(groups) == 1:
            out_name = "ood_bars.png"
        else:
            out_name = f"ood_bars_{_safe_name(env_id)}_{_format_theta(theta_key)}.png"
        out_path = os.path.join(output_dir, out_name)
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
    parser.add_argument("--ood_mode", choices=["aggregate", "raw"], default="aggregate", help="OOD bar plot mode.")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_learning_curves(args.log_root, args.output_dir)
    plot_ood_bars(args.ood_path, args.output_dir, mode=args.ood_mode)


if __name__ == "__main__":
    main()
