import torch, math
from functorch import make_functional
from torch.autograd.functional import jvp

# ----------------------------------------------------------------
# flatten / unflatten helpers
# ----------------------------------------------------------------
def _make_functional(module):
    fmod, params = make_functional(module)
    flat = torch.nn.utils.parameters_to_vector(
        [p.detach().requires_grad_(True) for p in params])
    shapes = [p.shape for p in params]
    return fmod, flat, shapes


def _unflatten_like(vec, shapes):
    outs, idx = [], 0
    for shape in shapes:
        numel = math.prod(shape)
        outs.append(vec[idx: idx + numel].view(shape))
        idx += numel
    return tuple(outs)


# ----------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------
# @torch._dynamo.disable()
def exist_condition_holds(
        encoder,
        s_a,                       # Tensor (obs shape)
        other_obs,                 # Tensor (..., obs shape)
        *,
        tol: float = 1e-3,
        oversample: int = 20,
        cg_iter: int = 15,
        device: str = "cuda"
):
    """
    Returns ( ok_flag : bool ,  sigma_min_estimate : float )

    ok_flag == True   ⇒   Condition (Exist) holds up to `tol`.
    """
    torch.manual_seed(0)          # reproducible random sketch

    # -- functional model -----------------------------------------
    encoder = encoder.to(device)
    s_a = s_a.to(device)
    other_obs = other_obs.to(device)

    fmodel, flat_params, shapes = _make_functional(encoder)
    P = flat_params.numel()

    latent_dim = fmodel(
        _unflatten_like(flat_params, shapes),
        s_a.unsqueeze(0)).numel()

    # -- per-observation Jv / JT closures -------------------------
    def make_J_JT(obs):
        """
        Returns two callables:
            Jv(v_flat)  ->  (d,)   (Jacobian‐vector product)
            JT(w)       ->  (P,)   (Jacobian^T · w)
        """
        obs_u = obs.unsqueeze(0)

        def Jv_fn(v_flat):
            v_list = _unflatten_like(v_flat, shapes)
            primals = _unflatten_like(flat_params, shapes)
            _, jvp_val = jvp(
                lambda *p: fmodel(p, obs_u).view(-1),
                primals, v_list, create_graph=False)
            return jvp_val          # (d,)

        def JT_fn(w):
            primals = _unflatten_like(flat_params, shapes)
            y = fmodel(primals, obs_u).view(-1)
            dot = torch.dot(y, w)
            grads = torch.autograd.grad(
                dot, primals, retain_graph=False, create_graph=False)
            return torch.nn.utils.parameters_to_vector(grads)

        return Jv_fn, JT_fn

    J_sa, JT_sa = make_J_JT(s_a)
    J_others, JT_others = zip(*[make_J_JT(o) for o in other_obs])

    # -- projection onto ker A  (conjugate gradient) --------------
    def project_to_kerA(v_flat):
        """
        Returns Pv where P projects onto ker A
        ( A is the vertical stack of J_others ).
        """

        def Av(vec):
            return torch.cat([J(vec) for J in J_others])   # (m-1)*d

        def ATu(u):
            start, outs = 0, []
            for JT in JT_others:
                outs.append(JT(u[start:start + latent_dim]))
                start += latent_dim
            return sum(outs)

        # CG to solve  (A A^T) x = A v   for x
        Av_vec = Av(v_flat)
        if Av_vec.norm() < 1e-8:
            return v_flat          # already in ker A

        x = torch.zeros_like(Av_vec)
        r = Av_vec.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        for _ in range(cg_iter):
            Ap = Av(ATu(p))
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < 1e-8:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return v_flat - ATu(x)     # v - A^T x  (i.e. Pv)

    # -- build random null-space sketch ---------------------------
    k = latent_dim + oversample
    W = torch.empty(P, k, device=device)
    for i in range(k):
        v = torch.randn(P, device=device)
        v = project_to_kerA(v)
        if v.norm() < 1e-6:                       # extremely rare
            v = torch.rand_like(v) * 1e-3
        W[:, i] = v / v.norm()

    # -- form d × k matrix  B = J_sa W  ---------------------------
    B_cols = [J_sa(W[:, i]) for i in range(k)]
    B = torch.stack(B_cols, dim=1)                # (d, k)

    # -- singular spectrum & decision -----------------------------
    svals = torch.linalg.svdvals(B)
    sigma_min = svals.min().item()
    return sigma_min > tol, sigma_min
