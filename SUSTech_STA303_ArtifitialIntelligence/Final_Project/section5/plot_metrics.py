"""
Plot learning curves directly from Section 5 `metrics.csv`.

Usage examples:
  python -m section5.plot_metrics --csv runs/section5/bc/cartpole_d_expert/metrics.csv --out runs/section5/figures/bc_expert.png
  python -m section5.plot_metrics --csv runs/section5/bc/cartpole_d_expert/metrics.csv --csv runs/section5/cql/cartpole_d_expert/metrics.csv --metric eval_return --out runs/section5/figures/compare.png
  python -m section5.plot_metrics --csv runs/section5/cql/cartpole_d_expert/metrics.csv --out runs/section5/figures/cql_join.png --metric eval_return --join
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOSS_METRIC_CANDIDATES = ("train_loss", "critic_loss", "bellman_loss", "actor_loss")


def _safe_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    try:
        return float(x)
    except Exception:
        return None


def read_metrics_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_series(
    rows: Sequence[Dict[str, str]],
    x_key: str = "step",
    y_key: str = "eval_return",
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = _safe_float(row.get(x_key))
        y = _safe_float(row.get(y_key))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    cumsum = 0.0
    buf: List[float] = []
    for v in values:
        buf.append(float(v))
        cumsum += float(v)
        if len(buf) > window:
            cumsum -= buf.pop(0)
        out.append(cumsum / len(buf))
    return out


def plot_learning_curve(
    csv_paths: Sequence[str],
    out_path: str,
    metric: str = "eval_return",
    x_key: str = "step",
    smooth: int = 1,
    title: Optional[str] = None,
) -> str:
    if not csv_paths:
        raise ValueError("No CSV paths provided.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure()
    for csv_path in csv_paths:
        rows = read_metrics_csv(csv_path)
        xs, ys = extract_series(rows, x_key=x_key, y_key=metric)
        if not xs:
            raise ValueError(f"No usable data for metric='{metric}' in: {csv_path}")
        ys_smooth = moving_average(ys, smooth)
        label = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        plt.plot(xs, ys_smooth, label=label)

    plt.xlabel(x_key)
    plt.ylabel(metric)
    plt.title(title or f"{metric} vs {x_key}")
    if len(csv_paths) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_joined_metric_and_loss(
    csv_paths: Sequence[str],
    out_path: str,
    metric: str = "eval_return",
    loss_metric: Optional[str] = None,
    x_key: str = "step",
    smooth: int = 1,
    title: Optional[str] = None,
) -> str:
    """
    Plot the main metric and a loss curve into a single figure using a twin y-axis.
    Left axis: `metric`, Right axis: `loss_metric` (or auto-picked loss column).
    """
    if not csv_paths:
        raise ValueError("No CSV paths provided.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    colors = plt.rcParams.get("axes.prop_cycle", None)
    color_list = colors.by_key().get("color", []) if colors is not None else []
    if not color_list:
        color_list = ["C0", "C1", "C2", "C3", "C4", "C5"]

    fig, ax_metric = plt.subplots()
    ax_loss = ax_metric.twinx()

    metric_handles = []
    loss_handles = []
    for i, csv_path in enumerate(csv_paths):
        rows = read_metrics_csv(csv_path)

        xs_m, ys_m = extract_series(rows, x_key=x_key, y_key=metric)
        if not xs_m:
            raise ValueError(f"No usable data for metric='{metric}' in: {csv_path}")

        loss_key = loss_metric or _pick_loss_metric(rows)
        if loss_key is None:
            raise ValueError(f"No loss columns found in: {csv_path}. Candidates: {list(LOSS_METRIC_CANDIDATES)}")
        xs_l, ys_l = extract_series(rows, x_key=x_key, y_key=loss_key)
        if not xs_l:
            raise ValueError(f"No usable data for loss_metric='{loss_key}' in: {csv_path}")

        run_name = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        color = color_list[i % len(color_list)]

        (h_metric,) = ax_metric.plot(
            xs_m,
            moving_average(ys_m, smooth),
            color=color,
            label=f"{run_name}:{metric}",
        )
        (h_loss,) = ax_loss.plot(
            xs_l,
            moving_average(ys_l, smooth),
            color=color,
            linestyle="--",
            alpha=0.8,
            label=f"{run_name}:{loss_key}",
        )
        metric_handles.append(h_metric)
        loss_handles.append(h_loss)

    ax_metric.set_xlabel(x_key)
    ax_metric.set_ylabel(metric)
    ax_loss.set_ylabel(loss_metric or "loss")
    ax_metric.set_title(title or f"{metric} + loss vs {x_key}")

    handles = metric_handles + loss_handles
    labels = [h.get_label() for h in handles]
    ax_metric.legend(handles, labels, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _derive_loss_out_path(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    if not ext:
        return out_path + "_loss.png"
    return root + "_loss" + ext


def _pick_loss_metric(rows: Sequence[Dict[str, str]]) -> Optional[str]:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for metric in LOSS_METRIC_CANDIDATES:
        if metric in keys:
            return metric
    return None


def plot_loss_curve(
    csv_paths: Sequence[str],
    out_path: str,
    loss_metric: Optional[str] = None,
    x_key: str = "step",
    smooth: int = 1,
    title: Optional[str] = None,
) -> str:
    if not csv_paths:
        raise ValueError("No CSV paths provided.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure()
    plotted = 0
    for csv_path in csv_paths:
        rows = read_metrics_csv(csv_path)
        metric = loss_metric or _pick_loss_metric(rows)
        if metric is None:
            print(f"[Plot] Skip (no loss column): {csv_path}")
            continue

        xs, ys = extract_series(rows, x_key=x_key, y_key=metric)
        if not xs:
            print(f"[Plot] Skip (no data for '{metric}'): {csv_path}")
            continue

        ys_smooth = moving_average(ys, smooth)
        run_name = os.path.basename(os.path.dirname(csv_path)) or os.path.basename(csv_path)
        label = f"{run_name} ({metric})" if loss_metric is None else run_name
        plt.plot(xs, ys_smooth, label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError("No usable loss data found in provided CSVs.")

    plt.xlabel(x_key)
    plt.ylabel(loss_metric or "loss")
    plt.title(title or "Loss vs step")
    if plotted > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Plot learning curves from Section 5 metrics.csv.")
    p.add_argument("--csv", dest="csv_paths", action="append", required=True, help="Path to metrics.csv (repeatable).")
    p.add_argument("--out", required=True, help="Output image path (png).")
    p.add_argument("--metric", default="eval_return", help="Metric column name to plot (default: eval_return).")
    p.add_argument("--x_key", default="step", help="X-axis column name (default: step).")
    p.add_argument("--smooth", type=int, default=1, help="Moving average window (default: 1=no smoothing).")
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument("--join", action="store_true", help="Plot main metric and loss into a single figure.")
    p.add_argument("--loss_out", default=None, help="Optional output path for loss plot (default: derive from --out).")
    p.add_argument(
        "--loss_metric",
        default=None,
        help=f"Loss column to plot (default: auto from {list(LOSS_METRIC_CANDIDATES)}).",
    )
    p.add_argument("--no_loss_plot", action="store_true", help="Disable generating the extra loss plot.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.join:
        if args.no_loss_plot:
            raise ValueError("`--join` requires a loss plot; remove `--no_loss_plot`.")
        out_path = plot_joined_metric_and_loss(
            csv_paths=args.csv_paths,
            out_path=args.out,
            metric=args.metric,
            loss_metric=args.loss_metric,
            x_key=args.x_key,
            smooth=args.smooth,
            title=args.title,
        )
        print(f"[Plot] Saved {out_path}")
        return

    out_path = plot_learning_curve(
        csv_paths=args.csv_paths,
        out_path=args.out,
        metric=args.metric,
        x_key=args.x_key,
        smooth=args.smooth,
        title=args.title,
    )
    print(f"[Plot] Saved {out_path}")

    if not args.no_loss_plot:
        loss_out = args.loss_out or _derive_loss_out_path(args.out)
        loss_path = plot_loss_curve(
            csv_paths=args.csv_paths,
            out_path=loss_out,
            loss_metric=args.loss_metric,
            x_key=args.x_key,
            smooth=args.smooth,
            title=None,
        )
        print(f"[Plot] Saved {loss_path}")


if __name__ == "__main__":
    main()
