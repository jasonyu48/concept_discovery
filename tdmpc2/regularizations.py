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
from torch.func import vmap

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


# ----------------------------------------------------------------------------
# 1. Orthogonality regularization
# ----------------------------------------------------------------------------

def orthogonality_regularization(
    encoder: torch.nn.Module,
    observations: torch.Tensor,
    *,
    num_pairs: int = 8,
    hutchinson_samples: int = 1,
    device: Union[str, torch.device] = "cuda",
    latent_dim: int = 512,
) -> torch.Tensor:
    """Compute the orthogonality regularization.

    R_orth = E_{(i,j)} || J_i J_j^T ||_F^2  where (i,j) are pairs of distinct
    observations.  The Frobenius norm is estimated with a Hutchinson trace
    estimator so that we never explicitly materialise the huge Jacobians.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder network mapping an observation to a 512-D latent vector.
    observations : torch.Tensor
        A batch of observations   (B, *).
    num_pairs : int, default 8
        Number of distinct (i,j) pairs sampled from the batch to estimate the
        expectation. If the batch is smaller, we fall back to all valid pairs.
    hutchinson_samples : int, default 1
        Number of Hutchinson probes per pair.
    device : str or torch.device, default "cuda"
        The device on which computations are performed.

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

    B = observations.size(0)
    if B < 2:
        # Cannot form a pair -> no orthogonality information
        return observations.new_zeros(())

    # Prepare functional model once
    fmodel, flat_params, shapes = _prepare_encoder(encoder, device)

    # Pre-compute Jv / JT builders for all observations in the batch because we
    # may reuse them multiple times.
    J_ops: List[Tuple] = [
        _construct_J_ops(fmodel, flat_params, shapes, observations[i]) for i in range(B)
    ]  # list of (Jv, JT)

    latent_dim = latent_dim  # assumption stated by the user
    reg_val = observations.new_zeros(())

    # Helper to draw random pairs (i,j)
    pairs: List[Tuple[int, int]] = []
    if B * (B - 1) // 2 <= num_pairs:
        # Use all pairs if they are few
        for i in range(B):
            for j in range(i + 1, B):
                pairs.append((i, j))
    else:
        # Sample without replacement
        rng = torch.randperm(B * (B - 1) // 2, device=device)[:num_pairs]
        # Map flat index -> (i,j)
        # Flattened ordering: (0,1), (0,2), …, (0,B-1), (1,2), …
        cum = 0
        for idx in rng.tolist():
            # Recover pair indices from triangular number theorem
            # Find smallest i such that idx < (i+1)B - (i+1)(i+2)/2
            i = 0
            k = idx
            while k >= B - i - 1:
                k -= B - i - 1
                i += 1
            j = i + 1 + k
            pairs.append((i, j))

    # Hutchinson estimator for each pair
    for i, j in pairs:
        Jv_i, _ = J_ops[i]
        _, JT_j = J_ops[j]

        pair_estimate = 0.0
        for _ in range(hutchinson_samples):
            z = torch.randint(0, 2, (latent_dim,), device=device, dtype=torch.float32) * 2 - 1  # Rademacher ±1
            z = z.unsqueeze(0)  # (1,d)
            # t = J_j^T z
            t = JT_j(z)              # (1,P)
            # s = J_i t
            s = Jv_i(t)              # (1,d)
            pair_estimate += (s * s).sum()  # ||M z||^2
        reg_val = reg_val + pair_estimate / hutchinson_samples

    if len(pairs) > 0:
        reg_val = reg_val / len(pairs)
    # end_time = time.time()
    # print(f"Orthogonality regularization time: {end_time - start_time} seconds")
    return reg_val


# ----------------------------------------------------------------------------
# 2. Full-row-rank regularization
# ----------------------------------------------------------------------------

def full_rank_regularization(
    encoder: torch.nn.Module,
    observations: torch.Tensor,
    *,
    epsilon: float = 1e-4,
    activation_margin: float = 10.0,
    device: Union[str, torch.device] = "cuda",
    latent_dim: int = 512,
) -> torch.Tensor:
    """Compute the negative log-det regularisation term.

    R_full = max(0,  −log det(J_i J_i^T + ε I) − margin).
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
    #signs, logdets = torch.linalg.slogdet(JJt_stable)  # Both shape (B,)

    dets = torch.linalg.det(JJt_stable)
    abs_dets = torch.abs(dets)

    # Apply ReLU activation and compute mean loss
    losses = torch.relu(activation_margin - abs_dets)  # (B,)
    total_loss = losses.mean()

    # end_time = time.time()
    # print(f"Full-row-rank regularization time: {end_time - start_time} seconds")
    # compute the absolute value of the determinant of the Jacobian at each observation by e^logdet
    # abs_det = torch.exp(logdets)
    return total_loss, abs_dets
