#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot how the estimated Lipschitz constant of the Q-network stabilises
as we increase the number of observation samples.

Prerequisites
-------------
• dm_control, gymnasium, mujoco, torch ≥ 2.0
• The tdmpc2 codebase in PYTHONPATH (we import WorldModel etc.)
• GPU is recommended – the Jacobian/SVD maths is heavy.

Paths
-----
Change these three paths if your files live elsewhere.
"""
# ---------------------------------------------------------------------
CHECKPOINT_PATH = "/home/tdmpc2/tdmpc2/logs/cheetah-run/2020/51_bin/models/final.pt"
CFG_PATH        = "/home/tdmpc2/tdmpc2/logs/cheetah-run/2020/51_bin/.hydra/config.yaml"
OBS_PATH        = "/scratch//obs_data/cheetah-run/obs/observations.pt"
# ---------------------------------------------------------------------
import os, math, time, random
import numpy as np
import torch
from torch.func import jacrev, vmap
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# tdmpc2 imports -------------------------------------------------------
from envs import make_env
from common.world_model import WorldModel
from common.layers import api_model_conversion            # needed to load state-dict
# ---------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")   # speed-up on A100 / RTX30

# 1) ------------------------------------------------------------------ #
print("Loading config and environment …")
cfg = OmegaConf.load(CFG_PATH)
cfg.device = device
cfg.task = "cheetah-run"
cfg.task_dim = 0
env = make_env(cfg)                # fills cfg.obs_shape, cfg.action_dim, etc.

# 2) ------------------------------------------------------------------ #
print("Building world model …")
model = WorldModel(cfg).to(device).eval()    # .eval() disables dropout
state = torch.load(CHECKPOINT_PATH, map_location=device)
state = state["model"] if "model" in state else state      # agent.save() format
state = api_model_conversion(model.state_dict(), state)    # handles old checkpoints
model.load_state_dict(state, strict=False)
print("✓ model loaded")

# 3) ------------------------------------------------------------------ #
print(f"Loading observations from {OBS_PATH} …")
obs_all = torch.load(OBS_PATH, map_location='cpu').float()   # (N, C, H, W)
N_total = obs_all.shape[0]
print(f"  tensor shape : {tuple(obs_all.shape)}  ({N_total:,} frames)")

# encode once, in manageable batches
print("Encoding observations …")
batch = 1024
z_list = []
with torch.no_grad():
    for i in range(0, N_total, batch):
        z = model.encode(obs_all[i:i+batch].to(device), task=None)
        z_list.append(z.detach())
z_all = torch.cat(z_list, 0)          # (N, latent_dim)
latent_dim = z_all.shape[1]
print(f"  latent dim   : {latent_dim}")
print("✓ encodings ready\n")

# 4) ------------------------------------------------------------------ #
print("Sampling 64 random actions …")
actions = []
for _ in range(64):
    if hasattr(env, "rand_act"):          # wrapper provides torch tensor already
        a = env.rand_act()
    else:
        a = torch.tensor(env.action_space.sample(), dtype=torch.float32)
    actions.append(a)
probe_actions = torch.stack(actions).to(device)    # (M=64, A)
M = probe_actions.size(0)
print("✓ actions sampled\n")

# 5) ------------------------------------------------------------------ #
def estimate_lipschitz(z_subset: torch.Tensor) -> float:
    """
    Deterministic Lipschitz estimate on a *given* subset of latent states.
    Matches SimpleEncodingSpaceMonitor._estimate_lipschitz, but
    uses user-provided probe_actions (64 random env actions).
    """
    def q_stack(z_single: torch.Tensor):
        z_rep = z_single.repeat(M, 1)
        a_rep = probe_actions
        q_out = model.Q(z_rep, a_rep, task=None, return_type='all', detach=False)
        q_out = q_out.mean(0)    # average over ensemble → (M, num_bins)
        q_out = q_out.mean(0)    # average over actions  → (num_bins)
        return q_out

    jac_fn = jacrev(q_stack)

    max_sv = -float("inf")
    bs = 16
    for i in range(0, z_subset.size(0), bs):
        svals = vmap(lambda z: torch.linalg.svdvals(jac_fn(z)).max())(z_subset[i:i+bs])
        max_sv = max(max_sv, svals.max().item())
    return float(max_sv)

# 6) ------------------------------------------------------------------ #
# Sizes to try (must be <= 16 384)
sample_sizes   = [  32,  64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 ]
results_K      = []  # one value per sample size

rng = torch.Generator(device="cpu")     # CPU generator for reproducibility
rng.manual_seed(0)

# Pick a fixed pool of max 16 384 latent codes --------------------------------
max_pool = min(16384, N_total)
idx_pool = torch.randperm(N_total, generator=rng)[:max_pool]
z_pool   = z_all[idx_pool.to(device)]  # (max_pool, latent_dim)

print("\nEstimating Lipschitz constants …\n")

# Baseline: full pool (up to 16 384 samples)
K_full = estimate_lipschitz(z_pool)
print(f"Baseline (n={max_pool}): K={K_full:.2f}\n")

# Sub-samples -----------------------------------------------------------
for n in sample_sizes:
    if n > max_pool:
        break
    # sample WITHOUT replacement from the fixed pool so all subsets come
    # from the same latent-code population
    idx_sub = torch.randperm(max_pool, generator=rng)[:n]
    K_n = estimate_lipschitz(z_pool[idx_sub.to(device)])
    results_K.append(K_n)
    print(f"  n={n:<6}  K_est={K_n:8.2f}")

print("\n✓ finished\n")

# 7) ------------------------------------------------------------------ #
plt.figure(figsize=(7,4.5))
plt.plot(sample_sizes[:len(results_K)], results_K, 'o-', lw=2, color='steelblue', label='subset estimate')
plt.axhline(K_full, color='red', ls='--', lw=1.5, label=f'full {max_pool} samples')
plt.xscale('log')
plt.xlabel('number of observation samples (log scale)')
plt.ylabel('estimated Lipschitz constant $K_{\\mathrm{est}}$')
plt.title('Lipschitz estimate vs. sample size (single draw)')
plt.legend()
plt.grid(True, ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig("lipschitz_vs_samples.png", dpi=300)