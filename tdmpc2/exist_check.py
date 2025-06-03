import math, torch
from functorch import make_functional, vmap
from torch.func import jvp, vjp
# ================================================================
# 1.  functional-encoder cache
# ------------------------------------------------
_FCACHE = {}


def _get_functional_encoder(enc: torch.nn.Module):
    key = id(enc)
    if key not in _FCACHE:
        fmod, params = make_functional(enc)
        flat = torch.nn.utils.parameters_to_vector(
            [p.detach().requires_grad_(True) for p in params]
        )
        shapes = [p.shape for p in params]
        _FCACHE[key] = (fmod, flat, shapes)
    return _FCACHE[key]


def _unflatten_like(vec, shapes):
    outs, idx = [], 0
    for shape in shapes:
        numel = math.prod(shape)
        outs.append(vec[idx : idx + numel].view(shape))
        idx += numel
    return tuple(outs)


# ================================================================
# 2.  Batched Jv / JT builders   (vmap-compatible)
# ------------------------------------------------
def _make_J_ops(fmodel, flat_params, shapes, obs_u):
    """Return vectorised Jacobian-vector (Jv) and vector-Jacobian (JT)."""

    # ---------- batched Jv --------------------------------------
    def _jvp_single(v_flat):
        v_list = _unflatten_like(v_flat, shapes)
        primals = _unflatten_like(flat_params, shapes)
        _, jvp_val = jvp(                       # torch.func.jvp
            lambda *p: fmodel(p, obs_u).view(-1),
            primals,
            v_list,
            strict=False,
        )
        return jvp_val

    def Jv(mat_V):                              # (B,P) → (B,d)
        return vmap(_jvp_single, randomness="same")(mat_V)

    # ---------- batched JT --------------------------------------
    def _single_vjp(w):
        # Use torch.func.vjp instead of manual gradient computation
        primals = _unflatten_like(flat_params, shapes)
        def func(*p):
            return fmodel(p, obs_u).view(-1)
        
        output, vjp_fn = vjp(func, *primals)
        grads = vjp_fn(w)
        return torch.nn.utils.parameters_to_vector(grads)

    def JT(mat_W):                              # (B,d) → (B,P)
        return vmap(_single_vjp, randomness="same")(mat_W)

    return Jv, JT


# ================================================================
# 3.  Public API
# ------------------------------------------------
def exist_condition_holds(
    encoder: torch.nn.Module,
    s_a: torch.Tensor,
    other_obs: torch.Tensor,
    *,
    tol: float = 1e-3,
    oversample: int = 20,
    cg_iter: int = 15,
    device: str = "cuda",
):
    """
    Probabilistic check of Condition (Exist).

    Returns
    -------
    ok : bool
        True if σ_min ≥ tol.
    sigma_min : float
        Estimated smallest singular value.
    """
    torch.manual_seed(0)        # deterministic sketch

    encoder = encoder.to(device)
    s_a = s_a.to(device)
    other_obs = other_obs.to(device)

    fmodel, flat_params, shapes = _get_functional_encoder(encoder)
    latent_dim = fmodel(
        _unflatten_like(flat_params, shapes),
        s_a.unsqueeze(0),
    ).numel()
    k = latent_dim + oversample
    P = flat_params.numel()

    # build J / JT
    J_sa, _ = _make_J_ops(fmodel, flat_params, shapes, s_a.unsqueeze(0))
    J_others, JT_others = zip(*[
        _make_J_ops(fmodel, flat_params, shapes, o.unsqueeze(0))
        for o in other_obs
    ])

    # ---------- CG projection: batch-mode -----------------------
    with torch.no_grad():
        V = torch.randn(k, P, device=device)

        def Av(mat):
            return torch.cat([J(mat) for J in J_others], dim=1)

        def ATu(mat):
            chunks = torch.split(mat, latent_dim, dim=1)
            return sum(JT(chunk) for JT, chunk in zip(JT_others, chunks))

        AV = Av(V)
        X = torch.zeros_like(AV)
        R = AV.clone()
        Pdir = R.clone()
        Rs_old = (R * R).sum(dim=1, keepdim=True)

        for _ in range(cg_iter):
            AP = Av(ATu(Pdir))
            alpha = Rs_old / (Pdir * AP).sum(dim=1, keepdim=True)
            X = X + alpha * Pdir
            R = R - alpha * AP
            Rs_new = (R * R).sum(dim=1, keepdim=True)
            if Rs_new.max().sqrt() < 1e-8:
                break
            Pdir = R + (Rs_new / Rs_old) * Pdir
            Rs_old = Rs_new

        V = torch.nn.functional.normalize(V - ATu(X), dim=1)  # k×P

    # ---------- build sketch  B = J_sa Vᵀ -----------------------
    B = J_sa(V).T                                             # d×k
    sigma_min = torch.linalg.svdvals(B).min().item()
    return sigma_min > tol, sigma_min
