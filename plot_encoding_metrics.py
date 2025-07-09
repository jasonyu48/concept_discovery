#!/usr/bin/env python3
"""
Plot encoding metrics for experiments under a specific seed directory.

Usage:
    python plot_encoding_metrics.py --seed 0

The script expects the following directory structure (default results_path):
/home/jyu197/tdmpc2/tdmpc2/logs/cheetah-run/{seed}/{exp_name}/
    ├─ .hydra/config.yaml         # contains num_bins, q_sample_ratio, buffer_size
    ├─ encoding_monitor/monitoring_data.json  # contains lists of metrics, including
    │                                        # "est_dim_avg" and "rankme".
    └─ DecoderLoss.txt                       # text file with decoder loss values

For each experiment it extracts the *last* non-NaN value of the two metrics and
creates a scatter plot (est_dim_avg vs rankme). The resulting PNG is saved
in the same seed directory.
"""

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional
import csv

import matplotlib.pyplot as plt
import yaml

RESULTS_PATH = "/home/jyu197/tdmpc2/tdmpc2/logs/cheetah-run"


def _last_not_nan(values: List[Any]) -> Optional[float]:
    """Return the last element in *values* that is not NaN/None, or None."""
    for v in reversed(values):
        if v is None:
            continue
        # Handle numbers that might be NaN (float("nan"))
        if isinstance(v, (int, float)) and math.isnan(v):
            continue
        return float(v)
    return None


def _decoder_loss_avg(exp_path: str) -> Optional[float]:
    """Return the average of the last 10 (or fewer) decoder loss values.

    `DecoderLoss.txt` is expected to be a CSV-like file with two columns
    (e.g. "step,loss") and possibly a header row. We parse the last column as
    the loss value while gracefully skipping non-numeric rows (headers)."""

    loss_path = os.path.join(exp_path, "DecoderLoss.txt")
    if not os.path.isfile(loss_path):
        return None

    values: List[float] = []
    try:
        with open(loss_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    val = float(row[-1])  # use last column (loss)
                    values.append(val)
                except ValueError:
                    # Likely a header row like ["step", "loss"] – skip
                    continue
    except Exception as e:
        print(f"[WARN] Failed to read {loss_path}: {e}")
        return None

    if not values:
        return None

    tail = values[-10:]
    return sum(tail) / len(tail)


def _load_experiment(exp_path: str) -> Optional[Dict[str, Any]]:
    """Load config and monitoring data for a single experiment directory."""
    cfg_path = os.path.join(exp_path, ".hydra", "config.yaml")
    monitor_path = os.path.join(exp_path, "encoding_monitor", "monitoring_data.json")

    if not (os.path.isfile(cfg_path) and os.path.isfile(monitor_path)):
        return None

    # Load YAML config
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {cfg_path}: {e}")
        return None

    # Load monitoring metrics
    try:
        with open(monitor_path, "r") as f:
            monitor = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {monitor_path}: {e}")
        return None

    est_dim_avg = _last_not_nan(monitor.get("est_dim_avg", []))
    est_dim_p_avg = _last_not_nan(monitor.get("est_dim_p_avg", []))
    rankme = _last_not_nan(monitor.get("rankme", []))
    dec_loss_avg = _decoder_loss_avg(exp_path)

    # Need rankme and at least one of the est_dim metrics
    if rankme is None or (est_dim_avg is None and est_dim_p_avg is None):
        return None

    return {
        "num_bins": cfg.get("num_bins"),
        "q_sample_ratio": cfg.get("q_sample_ratio"),
        "buffer_size": cfg.get("buffer_size"),
        "est_dim_avg": est_dim_avg,
        "est_dim_p_avg": est_dim_p_avg,
        "rankme": rankme,
        "decoder_loss_avg": dec_loss_avg,
    }


def plot_rankme(seed: str) -> None:
    """Generate two plots: est_dim_avg vs rankme and est_dim_p_avg vs rankme."""
    seed_dir = os.path.join(RESULTS_PATH, str(seed))
    if not os.path.isdir(seed_dir):
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    xs_avg: List[float] = []
    xs_p: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    # Iterate over experiment subdirectories
    for exp_name in sorted(os.listdir(seed_dir)):
        exp_path = os.path.join(seed_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        data = _load_experiment(exp_path)
        if data is None:
            continue

        rankme = data["rankme"]
        ys.append(rankme)
        labels.append(exp_name)

        xs_avg.append(data.get("est_dim_avg"))
        xs_p.append(data.get("est_dim_p_avg"))

    if not ys:
        print(f"No valid encoding data found in {seed_dir}.")
        return

    # Plot est_dim_avg vs rankme if available
    if any(x is not None for x in xs_avg):
        plt.figure(figsize=(6, 4))
        plt.scatter(xs_avg, ys, c="tab:blue")
        for x, y, lbl in zip(xs_avg, ys, labels):
            plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
        plt.xlabel("est_dim_avg")
        plt.ylabel("rankme")
        plt.title(f"Encoding metrics (est_dim_avg vs rankme) for seed {seed}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_file = os.path.join(seed_dir, f"est_dimavg_vs_rankme_seed_{seed}.png")
        plt.savefig(out_file, dpi=150)
        print(f"Plot saved to {out_file}")

    # Plot est_dim_p_avg vs rankme if available
    if any(x is not None for x in xs_p):
        plt.figure(figsize=(6, 4))
        plt.scatter(xs_p, ys, c="tab:green")
        for x, y, lbl in zip(xs_p, ys, labels):
            plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)
        plt.xlabel("equation13")
        plt.ylabel("rankme")
        plt.title(" ")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        out_file = os.path.join(seed_dir, f"est_dim_p_avg_vs_rankme_seed_{seed}.png")
        plt.savefig(out_file, dpi=150)
        print(f"Plot saved to {out_file}")


# -----------------------------------------------------------------------------
# Decoder loss plots
# -----------------------------------------------------------------------------


def plot_decoder_loss(seed: str) -> None:
    """Plot decoder loss vs relevant x-axis depending on seed (2021 or 2022)."""
    seed_dir = os.path.join(RESULTS_PATH, str(seed))
    if not os.path.isdir(seed_dir):
        return

    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []

    for exp_name in sorted(os.listdir(seed_dir)):
        exp_path = os.path.join(seed_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        data = _load_experiment(exp_path)
        if data is None:
            continue

        loss_avg = data.get("decoder_loss_avg")
        if loss_avg is None:
            continue

        if seed == "2021":
            x_val = data.get("num_bins")
            x_label = "num_bins"
        elif seed == "2022":
            q_ratio = data.get("q_sample_ratio")
            buf_size = data.get("buffer_size")
            if q_ratio is None or buf_size is None:
                continue
            x_val = q_ratio * buf_size
            x_label = "q_sample_ratio * buffer_size"
        else:
            # Only seeds 2021 and 2022 are specified for decoder loss plots
            return

        if x_val is None:
            continue

        xs.append(x_val)
        ys.append(loss_avg)
        labels.append(exp_name)

    if not xs:
        print(f"No valid decoder loss data found in {seed_dir}.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, c="tab:red")

    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)

    plt.xlabel(x_label)
    plt.ylabel("Average decoder loss (last 10)")
    plt.title(f"Decoder loss vs {x_label} for seed {seed}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fname = ("numbins" if seed == "2021" else "ratio_bufsize")
    out_file = os.path.join(seed_dir, f"decoder_loss_vs_{fname}_seed_{seed}.png")
    plt.savefig(out_file, dpi=150)
    print(f"Decoder loss plot saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot encoding and decoder metrics for a seed's experiments.")
    parser.add_argument("--seed", default=2021, help="Seed folder name (e.g., 0 or 1)")
    args = parser.parse_args()

    plot_rankme(args.seed)
    plot_decoder_loss(str(args.seed)) 