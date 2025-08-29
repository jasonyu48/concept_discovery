#!/usr/bin/env python3
"""
Plot encoding metrics for experiments under a specific seed directory.

Usage:
    python plot_encoding_metrics.py --seed 0

The script expects the following directory structure (default results_path):
/home/tdmpc2/tdmpc2/logs/cheetah-run/{seed}/{exp_name}/
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
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
# csv no longer needed (removed legacy DecoderLoss.txt support)

# Optional heavy deps (torch & tdmpc2); imported lazily when --recalculate_decoder_loss is set
import torch
import torch.nn.functional as F
from common.world_model import WorldModel
from common.layers import api_model_conversion
from envs import make_env
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import yaml


# --------------------------------------------------------------------------------------
# Helper: recursively convert a (nested) dict to a SimpleNamespace for dot access
# --------------------------------------------------------------------------------------


def _dict_to_ns(d: Dict[str, Any]) -> SimpleNamespace:  # type: ignore
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_ns(v))
        else:
            setattr(ns, k, v)
    return ns


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


# (legacy DecoderLoss.txt reader removed)


# --------------------------------------------------------------------------------------
# Re-calculate decoder loss by loading model & observations (expensive, CPU only)
# --------------------------------------------------------------------------------------


def _recalculate_decoder_loss(exp_path: str, cfg_path: str, task: str) -> Optional[float]:
    """Load the trained model and compute decoder reconstruction loss on stored observations.

    Config is re-loaded with OmegaConf (same as num_lipschitz_sample) to retain correct
    datatypes such as tuples, lists, etc.
    """

    if torch is None or WorldModel is None:
        print("[WARN] PyTorch / tdmpc2 not available – cannot recalculate decoder loss.")
        return None

    # Resolve model checkpoint – allow both final.pt & latest_checkpoint.pt
    ckpt_candidates = [
        os.path.join(exp_path, "models", "final.pt"),
        os.path.join(exp_path, "models", "latest_checkpoint.pt"),
    ]
    ckpt_path = next((p for p in ckpt_candidates if os.path.isfile(p)), None)
    if ckpt_path is None:
        print(f"[WARN] No checkpoint found in {exp_path}/models/")
        return None

    # Reload cfg with OmegaConf to keep complex types intact
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception as e:
        print(f"[WARN] OmegaConf failed to load {cfg_path}: {e}")
        return None

    # Minimal patches needed for evaluation
    cfg.enable_decoder = True
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.multitask = False
    cfg.task_dim = 0
    env = make_env(cfg)

    device = cfg.device

    try:
        model = WorldModel(cfg).to(device).eval()
        state = torch.load(ckpt_path, map_location=device)
        state = state["model"] if "model" in state else state
        state = api_model_conversion(model.state_dict(), state)
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"[WARN] Failed to load model from {ckpt_path}: {e}")
        return None

    # Observation tensor path – override with standard location based on task
    eval_obs_path = f"/scratch//obs_data/{task}/obs/observations.pt"
    if not os.path.isfile(eval_obs_path):
        print(f"[WARN] observation file not found: {eval_obs_path}")
        return None

    try:
        obs_tensor = torch.load(eval_obs_path, map_location="cpu").float()
    except Exception as e:
        print(f"[WARN] Failed to load observations from {eval_obs_path}: {e}")
        return None

    batch_size = 4096
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for start in range(0, obs_tensor.shape[0], batch_size):
            batch = obs_tensor[start:start + batch_size].to(device, non_blocking=True)
            z = model.encode(batch, task=None)
            recon = model._decoder(z)
            recon = recon.reshape_as(batch)
            target = (batch / 255.0 - 0.5) * 2.0
            batch_loss = F.mse_loss(recon, target, reduction="mean").item()
            total_loss += batch_loss * batch.size(0)
            total_samples += batch.size(0)

    if total_samples == 0:
        return None

    return float(total_loss / total_samples)


def _load_experiment(exp_path: str, task: str, recalc: bool = False) -> Optional[Dict[str, Any]]:
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
    # Decoder loss handling
    if recalc:
        dec_loss = _recalculate_decoder_loss(exp_path, cfg_path, task)
        # Persist into json for future quick access
        if dec_loss is not None:
            monitor["decoder_loss_recalc"] = dec_loss
            try:
                with open(monitor_path, "w") as fw:
                    json.dump(monitor, fw, indent=2)
            except Exception as e:
                print(f"[WARN] Could not update {monitor_path}: {e}")
    else:
        # Prefer value stored by monitor JSON
        dec_loss = _last_not_nan(monitor.get("decoder_loss", []))
        if dec_loss is None:
            dec_loss = monitor.get("decoder_loss_recalc")

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
        "decoder_loss": dec_loss,
    }

# ------------------------------ NEW: steps / cluster_acc helpers ------------------------------

def _load_steps_and_cluster_acc(exp_path: str) -> Optional[Dict[str, List[float]]]:
    """Load the full *steps* and *cluster_acc* lists from monitoring_data.json.

    Returns None if the file or the required fields are missing / malformed.
    """
    monitor_path = os.path.join(exp_path, "encoding_monitor", "monitoring_data.json")
    if not os.path.isfile(monitor_path):
        return None

    try:
        with open(monitor_path, "r") as f:
            monitor = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {monitor_path}: {e}")
        return None

    steps = monitor.get("steps")
    cluster_acc = monitor.get("cluster_acc")

    # Basic validation: both should be lists
    if not isinstance(steps, list) or not isinstance(cluster_acc, list):
        return None
    
    # Handle case where cluster_acc has one fewer value than steps
    # (older versions: cluster_acc starts from step 3000, not 0)
    # (newer versions: cluster_acc includes step 0)
    if len(cluster_acc) == len(steps) - 1:
        # Use steps[1:] to match cluster_acc length (older data)
        aligned_steps = steps[1:]
    elif len(cluster_acc) == len(steps):
        # Equal length case (newer data with step 0 included)
        aligned_steps = steps
    else:
        # Unexpected length mismatch
        print(f"[WARN] Length mismatch in {monitor_path}: steps={len(steps)}, cluster_acc={len(cluster_acc)}")
        return None

    return {"steps": aligned_steps, "cluster_acc": cluster_acc}


def plot_cluster_acc(seed: str, task: str) -> None:
    """Plot *steps* vs *cluster_acc* for every experiment in the given seed directory."""
    seed_dir = os.path.join(RESULTS_PATH, str(seed))
    if not os.path.isdir(seed_dir):
        raise FileNotFoundError(f"Seed directory not found: {seed_dir}")

    plt.figure(figsize=(12, 8))
    any_data = False

    for exp_name in sorted(os.listdir(seed_dir)):
        exp_path = os.path.join(seed_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        data = _load_steps_and_cluster_acc(exp_path)
        if data is None:
            continue

        steps = data["steps"]
        acc = data["cluster_acc"]

        # Plot all points for this experiment
        plt.plot(steps, acc, linestyle="-", label=exp_name, linewidth=1)
        any_data = True

    if not any_data:
        print(f"No steps/cluster_acc data found in {seed_dir}.")
        return

    plt.xlabel("Steps")
    plt.ylabel("Cluster Accuracy")
    plt.title(f"Cluster Accuracy vs Steps for seed {seed}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_file = os.path.join(seed_dir, f"steps_vs_cluster_acc_seed_{seed}.png")
    plt.savefig(out_file, dpi=150)
    print(f"Cluster accuracy plot saved to {out_file}")

# -----------------------------------------------------------------------------
# Decoder loss plots
# -----------------------------------------------------------------------------


def plot_decoder_loss(seed: str, task: str, recalc: bool = False) -> None:
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

        data = _load_experiment(exp_path, task, recalc=recalc)
        if data is None:
            continue

        loss_val = data.get("decoder_loss")
        if loss_val is None:
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
        ys.append(loss_val)
        labels.append(exp_name)

    if not xs:
        print(f"No valid decoder loss data found in {seed_dir}.")
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, c="tab:red")

    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8)

    plt.xlabel(x_label)
    plt.ylabel("Decoder loss (MSE)")
    plt.title(f"Decoder loss vs {x_label} for seed {seed}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    fname = ("numbins" if seed == "2021" else "ratio_bufsize")
    out_file = os.path.join(seed_dir, f"decoder_loss_vs_{fname}_seed_{seed}.png")
    plt.savefig(out_file, dpi=150)
    print(f"Decoder loss plot saved to {out_file}")


# ------------------------------ restore plot_rankme ------------------------------

def plot_rankme(seed: str, task: str, recalc: bool = False) -> None:
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

        data = _load_experiment(exp_path, task, recalc=recalc)
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
        out_file = os.path.join(seed_dir, f"est_dim_avg_vs_rankme_seed_{seed}.png")
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

# ------------------------------ end restore plot_rankme ------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot encoding, decoder, and clustering metrics for a seed's experiments.")
    parser.add_argument("--seed", default=2022, help="Seed folder name")
    parser.add_argument("--task", default="counting5", help="Task name (determines results and observation paths)")
    parser.add_argument("--results_root", default="/scratch//concept_discovery/tdmpc2/logs", help="Base directory that contains task subfolders")
    parser.add_argument("--recalculate_decoder_loss", default='False', help="Recompute decoder loss using saved model instead of reading from monitoring data.")
    args = parser.parse_args()

    # Set global RESULTS_PATH based on task and optional root override
    global RESULTS_PATH  # type: ignore
    RESULTS_PATH = os.path.join(args.results_root, args.task)

    # plot_rankme(str(args.seed), task=args.task, recalc=args.recalculate_decoder_loss == 'True')
    plot_decoder_loss(str(args.seed), task=args.task, recalc=args.recalculate_decoder_loss == 'True')
    plot_cluster_acc(str(args.seed), task=args.task) 