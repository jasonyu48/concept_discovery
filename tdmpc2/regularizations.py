# implement two regularizations:
# 1. (orthogonality) the rowspaces of the Jacobians of the encoder with respect to its parameters 
# at different observations should be orthogonal to each other
# 2. (full row rank) the Jacobians should have full row rank
# mathematically, R_orth = ||J_iJ_j^T||^2 where J_i is the Jacobian at observation i
# J_j is the Jacobian at observation j != i
# R_full = -log det(J_iJ_i^T + epsilon I) It should not be active when the determinant is already large

from typing import Tuple, List

import torch
from torch.func import functional_call

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
    observations = observations.to(device).float()  # Convert uint8 to float for gradient computation
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

def _lanczos(Av_fn, dim: int, k: int, v0: torch.Tensor):
    """Classic Lanczos tridiagonalisation.

    Args:
        Av_fn: callable that returns A @ v (shape (dim,)). Should create graph for autograd.
        dim:   dimension of A (latent dimension).
        k:     number of Lanczos iterations (<= dim).
        v0:    initial vector (shape (dim,)). Must be non-zero.

    Returns:
        Q  – (dim, m) orthonormal basis (m ≤ k).
        T  – (m, m) symmetric tridiagonal matrix s.t.  A ≈ Q T Qᵀ  in Krylov space.
    """
    k = min(k, dim)
    q = v0 / (v0.norm() + 1e-12)
    Q_cols = []
    alphas, betas = [], []
    beta_prev = torch.tensor(0.0, dtype=q.dtype, device=q.device)
    q_prev = torch.zeros_like(q)

    for j in range(k):
        Q_cols.append(q)

        # A @ q
        z = Av_fn(q)

        alpha = torch.dot(q, z)
        alphas.append(alpha)

        # Orthogonalise against previous basis vector
        z = z - alpha * q - beta_prev * q_prev

        beta = z.norm()
        if j == k - 1 or beta.item() == 0.0:
            break

        betas.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    m = len(alphas)
    T = torch.zeros((m, m), dtype=v0.dtype, device=v0.device)
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    Q = torch.stack(Q_cols, dim=1)  # (dim, m)
    return Q, T

def full_rank_regularization(
    encoder: torch.nn.Module,
    observations: torch.Tensor,
    *,
    latent_dim: int,
    num_samples: int = 16,
    epsilon: float = 1e-4,
    activation_margin: float = 5.0,
    # Implementation switch ------------------------------------------------
    row_orthogonal: bool = True,           # NEW: very cheap surrogate using row-orthogonality
    approximate: bool = True,              # used only if row_orthogonal is False
    # ---------------------------------------------------------------------
    num_probe: int = 1,
    num_lanczos: int = 10,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    R_rank  =  mean_s ReLU(margin − log det(J_s J_sᵀ + ε I))

    • No functorch transforms that re-enter autograd (no vmap/grad).  
    • Uses autograd.functional.{vjp,jvp} with create_graph=True.  
    • Gradient flows to encoder.parameters().
    """
    device       = torch.device(device)
    observations = observations.to(device).float()  # Convert uint8 to float for gradient computation
    B            = observations.size(0)

    # --------------------------------------------------------------
    # Optionally subsample observations to reduce compute cost
    # --------------------------------------------------------------
    if num_samples is None or num_samples >= B:
        sample_ids = torch.arange(B, device=device)
    else:
        sample_ids = torch.randperm(B, device=device)[:num_samples]

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------
    names, params = zip(*[(n, p) for n, p in encoder.named_parameters()
                          if p.requires_grad])
    params = tuple(params)                              # single tuple

    eye_d = torch.eye(latent_dim, device=device)

    # helper: run encoder(x) with explicit parameter tuple -------------------
    def enc_with(ps: Tuple[torch.Tensor, ...], x: torch.Tensor) -> torch.Tensor:
        return functional_call(
            encoder,
            {k: v for k, v in zip(names, ps)},
            (x,),
        )                                               # (1, d)

    logdets: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Option 1: fast row-orthonormality surrogate  ----------------------
    # ------------------------------------------------------------------
    if row_orthogonal:
        hutchinson_samples = num_probe  # reuse same arg semantics
        loss_accum = 0.0

        for idx in sample_ids.tolist():
            x = observations[idx : idx + 1]

            obs_accum = 0.0
            for _ in range(hutchinson_samples):
                # Rademacher probe with encoder output shape (1, d)
                z = torch.randint(0, 2, (1, latent_dim), device=device, dtype=torch.float32)
                z = z.mul_(2).sub_(1)

                # Compute G z = J Jᵀ z   via one VJP + one JVP
                _, t_tuple = torch.autograd.functional.vjp(
                    lambda *ps: enc_with(ps, x),
                    params,
                    v=z,
                    create_graph=True,
                )

                _, s = torch.autograd.functional.jvp(
                    lambda *ps: enc_with(ps, x),
                    params,
                    t_tuple,
                    create_graph=True,
                )  # s shape (1, d)

                # Residual (G - am*I) z
                delta = s - activation_margin * z

                obs_accum = obs_accum + delta.pow(2).sum()

            loss_accum = loss_accum + obs_accum / hutchinson_samples

        # Average over sampled observations
        loss = loss_accum / sample_ids.numel()

        # In this surrogate we don't compute |det|; reuse loss for monitoring
        abs_det = loss.detach()
        return loss, abs_det

    # ------------------------------------------------------------------
    # Option 2: log-det based methods (exact or Hutchinson-Lanczos) ------
    # ------------------------------------------------------------------

    for idx in sample_ids.tolist():
        x = observations[idx : idx + 1]  # keep batch dim

        if approximate:
            # ---------------- Hutchinson-Lanczos estimate ----------------

            def Av(v: torch.Tensor) -> torch.Tensor:
                """Return (J Jᵀ + εI) v without materialising J."""
                # Reverse-mode: t = Jᵀ v
                _, t_tuple = torch.autograd.functional.vjp(
                    lambda *ps: enc_with(ps, x),
                    params,
                    v=v.unsqueeze(0),  # add batch dim
                    create_graph=True,
                )
                # Forward-mode: J t
                _, s = torch.autograd.functional.jvp(
                    lambda *ps: enc_with(ps, x),
                    params,
                    t_tuple,
                    create_graph=True,
                )
                s = s.squeeze(0)  # (d,)
                return s + epsilon * v

            ld_est = 0.0
            for _ in range(num_probe):
                z = torch.randint(0, 2, (latent_dim,), device=device, dtype=torch.float32)
                z = z.mul(2).sub(1)  # ±1 Rademacher

                # Lanczos tridiagonalisation on the fly
                _, T = _lanczos(Av, latent_dim, num_lanczos, z)

                # Eigenvalues of small tri-diagonal matrix
                evals = torch.linalg.eigvalsh(T)

                # Prevent log(0)
                evals = evals.clamp(min=1e-12)

                ld_est = ld_est + (z @ z) * torch.log(evals).mean()

            ld = ld_est / num_probe

        else:
            # ---------------- Exact logdet via explicit Gram -------------
            cols = []
            for k in range(latent_dim):
                e_k = eye_d[k : k + 1]  # (1, d) basis vector

                # Reverse-mode: t = Jᵀ e_k
                _, pullback = torch.autograd.functional.vjp(
                    lambda *ps: enc_with(ps, x),
                    params,
                    v=e_k,
                    create_graph=True,
                )
                t_tuple = pullback

                # Forward-mode: J t
                _, col = torch.autograd.functional.jvp(
                    lambda *ps: enc_with(ps, x),
                    params,
                    t_tuple,
                    create_graph=True,
                )
                cols.append(col.squeeze(0))

            G = torch.stack(cols, dim=1)  # (d, d)
            G_eps = G + epsilon * eye_d
            _, ld = torch.linalg.slogdet(G_eps)

        logdets.append(ld)

    logdets = torch.stack(logdets)  # (|sample_ids|,)

    # Regularisation loss
    loss = torch.relu(activation_margin - logdets).mean()
    # loss = logdets.mean()

    # For monitoring: mean |det| (clamped)
    abs_det = torch.exp(torch.clamp(logdets, max=20)).mean()
    return loss, abs_det
