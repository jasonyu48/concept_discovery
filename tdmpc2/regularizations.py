# implement two regularizations:
# 1. (orthogonality) the rowspaces of the Jacobians of the encoder with respect to its parameters 
# at different observations should be orthogonal to each other
# 2. (full row rank) the Jacobians should have full row rank
# mathematically, R_orth = ||J_iJ_j^T||^2 where J_i is the Jacobian at observation i
# J_j is the Jacobian at observation j != i
# R_full = -log det(J_iJ_i^T + epsilon I) It should not be active when the determinant is already large

import math
from typing import Tuple, List, Union

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

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------

def _prepare_encoder(encoder: torch.nn.Module, device: torch.device):
    """Return functional encoder + flattened params + shapes, moved to device."""
    encoder = encoder.to(device)
    fmodel, flat_params, shapes = _get_functional_encoder(encoder)
    return fmodel, flat_params, shapes


def _construct_J_ops(
    fmodel,
    flat_params: torch.Tensor,
    shapes: List[torch.Size],
    obs: torch.Tensor,
):
    """Convenience wrapper around `_make_J_ops`."""
    return _make_J_ops(fmodel, flat_params, shapes, obs.unsqueeze(0))


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
        return observations.new_zeros(())

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
    reg_val = observations.new_zeros(())
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
    device: Union[str, torch.device] = "cuda",
    latent_dim: int = 512,
) -> torch.Tensor:
    """Compute the log-det regularisation term.
    The determinant is computed exactly by constructing the d×d Gram matrix via
    Jacobian-vector products; d is assumed to be 512.

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder network.
    observations : torch.Tensor
        A batch of observations (B, *).
    epsilon : float, default 1e-4
        Positive constant added to the diagonal for numerical stability.
    activation_margin : float, default 0
        Optional margin: when −logdet < margin the loss is clamped to 0. Setting
        a positive margin makes the regularisation active only when logdet is
        sufficiently small.
    device : str or torch.device, default "cuda"
        Device for computation.

    Returns
    -------
    torch.Tensor
        Scalar loss (requires grad).
    """
    # start_time = time.time()
    if observations.ndim == 0:
        raise ValueError("`observations` must be a batch, not a single tensor.")

    device = torch.device(device)
    observations = observations.to(device)

    fmodel, flat_params, shapes = _prepare_encoder(encoder, device)

    B = observations.size(0)
    I_eye = torch.eye(latent_dim, device=device)

    # Compute Jacobians for all observations at once using vmap
    def single_jacobian_ops(obs):
        """Compute J @ I for a single observation"""
        Jv, JT = _construct_J_ops(fmodel, flat_params, shapes, obs)
        # Compute JT(I) -> (latent_dim, P)
        JT_I = JT(I_eye)
        # Compute J(JT(I)) -> (latent_dim, latent_dim) which is J J^T
        JJt = Jv(JT_I)
        return JJt

    # Use vmap to compute all Jacobian Gram matrices at once
    # Shape: (B, latent_dim, latent_dim)
    JJt_batch = vmap(single_jacobian_ops)(observations)

    # Add epsilon regularization to all matrices at once
    epsilon_eye = epsilon * I_eye.unsqueeze(0).expand(B, -1, -1)  # (B, latent_dim, latent_dim)
    JJt_stable = JJt_batch + epsilon_eye

    # Compute log determinants for all matrices at once
    signs, logdets = torch.linalg.slogdet(JJt_stable)  # Both shape (B,)

    # dets = torch.linalg.det(JJt_stable)
    # abs_dets = torch.abs(dets)

    # Apply ReLU activation and compute mean loss
    losses = torch.relu(activation_margin - logdets)  # (B,)
    total_loss = losses.mean()

    # end_time = time.time()
    # print(f"Full-row-rank regularization time: {end_time - start_time} seconds")
    # compute the absolute value of the determinant of the Jacobian at each observation by e^logdet
    abs_det = torch.exp(torch.clamp(logdets, max=20)).mean()
    return total_loss, abs_det
