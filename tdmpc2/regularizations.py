# implement two regularizations:
# 1. (orthogonality) the rowspaces of the Jacobians of the encoder with respect to its parameters 
# at different observations should be orthogonal to each other
# 2. (full row rank) the Jacobians should have full row rank
# mathematically, R_orth = ||J_iJ_j^T||^2 where J_i is the Jacobian at observation i
# J_j is the Jacobian at observation j != i
# R_full = -log det(J_iJ_i^T + epsilon I) It should not be active when the determinant is already large

from typing import Tuple, List

import torch
from torch import vmap
from torch.func import functional_call

# Re-use the Jacobian ops implemented in exist_check
from exist_check import _get_functional_encoder, _make_J_ops

import time




__all__ = [
    "orthogonality_regularization",
    "full_rank_regularization",
]

# ----------------------------------------------------------------------
#  Orthogonality regulariser
# ----------------------------------------------------------------------
@torch.no_grad()
def _choose_pairs(batch_size: int, num_pairs: int) -> List[Tuple[int, int]]:
    """Return ≤ num_pairs unordered indices (i, j) with i < j."""
    all_pairs = [(i, j) for i in range(batch_size) for j in range(i + 1, batch_size)]
    if len(all_pairs) <= num_pairs:
        return all_pairs
    # deterministic but you can replace by random.sample
    return all_pairs[:num_pairs]


def orthogonality_regularization(
    encoder: torch.nn.Module,
    observations: torch.Tensor,
    *,
    num_pairs: int = 8,
    hutchinson_samples: int = 1,
    device: str = "cuda",
    latent_dim: int = 512,
) -> torch.Tensor:
    r"""
    Estimate  R_orth = E_{i≠j} || J(x_i) J(x_j)ᵀ ||_F²
    with Hutchinson probes, fully differentiable w.r.t. encoder params.
    """
    observations = observations.to(device)
    B = observations.shape[0]
    if B < 2:                                           # need at least one pair
        return torch.zeros((), device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------
    param_names, enc_params = [], []
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            enc_params.append(p)
    enc_params = tuple(enc_params)                      # freeze structure

    def _enc_forward(params: Tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        """Run encoder(x) with *params* as its weights."""
        param_dict = {k: v for k, v in zip(param_names, params)}
        return functional_call(encoder, param_dict, (x,))  # shape (1, d)

    # ------------------------------------------------------------------
    #  Main computation
    # ------------------------------------------------------------------
    reg_val = torch.zeros((), device=device, dtype=torch.float32)
    pairs = _choose_pairs(B, num_pairs)

    for i, j in pairs:
        x_i = observations[i : i + 1]                   # keep batch dim
        x_j = observations[j : j + 1]

        pair_accum = 0.0
        for _ in range(hutchinson_samples):
            # ---- 1) probe z  ~ Rademacher(±1) with correct output shape ----
            z = torch.randint(0, 2, (1, latent_dim), device=device, dtype=torch.float32)
            z = z.mul_(2).sub_(1)                       # ±1

            # ---- 2) t = J(x_j)ᵀ z  via reverse-mode (vjp) -----------------
            y_j, pullback = torch.autograd.functional.vjp(
                lambda *ps: _enc_forward(ps, x_j),      # func(*params)
                enc_params,
                v=z,
                create_graph=True,
            )
            t_tuple = pullback                          # same structure as params

            # ---- 3) s = J(x_i) t  via forward-mode (jvp) ------------------
            _, s = torch.autograd.functional.jvp(
                lambda *ps: _enc_forward(ps, x_i),
                enc_params,         # primals
                t_tuple,            # tangents
                create_graph=True,
            )                                          # s shape (1, d)

            pair_accum += s.pow(2).sum()

        reg_val += pair_accum / hutchinson_samples

    # average over sampled pairs
    return reg_val / len(pairs)



# ----------------------------------------------------------------------------
# 2. Full-row-rank regularization
# ----------------------------------------------------------------------------

def full_rank_regularization(
    encoder: torch.nn.Module,
    observations: torch.Tensor,
    *,
    epsilon: float = 1e-4,
    activation_margin: float = 5.0,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    R_rank  =  mean_s ReLU(margin − log det(J_s J_sᵀ + ε I))

    • No functorch transforms that re-enter autograd (no vmap/grad).  
    • Uses autograd.functional.{vjp,jvp} with create_graph=True.  
    • Gradient flows to encoder.parameters().
    """
    device       = torch.device(device)
    observations = observations.to(device)
    B            = observations.size(0)

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------
    names, params = zip(*[(n, p) for n, p in encoder.named_parameters()
                          if p.requires_grad])
    params = tuple(params)                              # single tuple

    latent_dim = encoder(observations[:1]).size(1)
    eye_d      = torch.eye(latent_dim, device=device)

    # helper: run encoder(x) with explicit parameter tuple -------------------
    def enc_with(ps: Tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        return functional_call(
            encoder,
            {k: v for k, v in zip(names, ps)},
            (x,),
        )                                               # (1, d)

    logdets: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Loop over the batch (no vmap needed)
    # ------------------------------------------------------------------
    for idx in range(B):
        x = observations[idx : idx + 1]                 # keep batch dim

        cols = []
        for k in range(latent_dim):
            e_k = eye_d[k : k + 1]                      # (1, d) Rademacher basis

            # ----- reverse-mode:  t = Jᵀ e_k ---------------------------
            _, pullback = torch.autograd.functional.vjp(
                lambda *ps: enc_with(ps, x),            # func(*ps)
                params,                                 # ONE positional input
                v=e_k,
                create_graph=True,
            )
            t_tuple = pullback                          # tuple, same structure

            # ----- forward-mode:  col_k = J t --------------------------
            _, col = torch.autograd.functional.jvp(
                lambda *ps: enc_with(ps, x),
                params,                                 # primals
                t_tuple,                                # tangents
                create_graph=True,
            )                       # col shape (1, d)

            cols.append(col.squeeze(0))                 # (d,)

        # Gram matrix G = J Jᵀ and its stabilised version
        G     = torch.stack(cols, dim=1)                # (d, d)
        G_eps = G + epsilon * eye_d
        _, ld = torch.linalg.slogdet(G_eps)             # scalar log-det
        logdets.append(ld)

    logdets = torch.stack(logdets)                      # (B,)

    # Regularisation loss
    loss = torch.relu(activation_margin - logdets).mean()

    # For monitoring: mean |det| (clamped)
    abs_det = torch.exp(torch.clamp(logdets, max=20)).mean()
    return loss, abs_det
