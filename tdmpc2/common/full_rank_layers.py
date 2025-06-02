import torch
import torch.nn as nn
import torch.nn.functional as F

class LULinear(nn.Module):
    def __init__(self, in_features, out_features,
                 delta=1e-4, negative_slope=1e-2, bias=True, activate=True):
        super().__init__()
        self.in_features   = in_features
        self.out_features  = out_features
        self.delta         = delta
        self.activate      = activate
        self.negative_slope = negative_slope

        k = min(in_features, out_features)  # size of the square core

        # Initialize L_params (Gaussian) with appropriate shape
        if out_features >= in_features:
            self.L_params = nn.Parameter(torch.randn(out_features, k) / (k**0.5))
        else:
            self.L_params = nn.Parameter(torch.randn(out_features, out_features) / (out_features**0.5))

        # U_params stores both the square‐core columns and any "extra" columns:
        self.U_params   = nn.Parameter(torch.randn(k, in_features) / (in_features**0.5))
        self.U_diag_raw = nn.Parameter(torch.randn(k))  # unconstrained

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _build_L(self):
        m, n = self.out_features, self.in_features
        k = min(m, n)
        L = torch.tril(self.L_params, diagonal=-1)  # strictly lower‐triangular part
        if m >= n:
            # Tall or square: insert I_k in top‐left of L
            L[:k, :k] = L[:k, :k] + torch.eye(k, device=L.device, dtype=L.dtype)
        else:
            # Wide: insert I_m throughout
            L = L + torch.eye(m, device=L.device, dtype=L.dtype)
        return L  # shape (m, k) if tall, or (m, m) if wide

    def _build_U(self):
        m, n = self.out_features, self.in_features
        k = min(m, n)

        # 1) Extract the k×k core from U_params
        core_U = self.U_params[:, :k]  # shape (k, k)

        # 2) Zero out diag & below, keeping only strictly‐upper part
        U_core_strict_upper = torch.triu(core_U, diagonal=1)  # (k, k)

        # 3) Safe‐diagonal trick:
        sign = torch.where(self.U_diag_raw >= 0, 1.0, -1.0)        # shape (k,)
        diag = self.U_diag_raw + sign * self.delta               # |diag[i]| ≥ delta
        U_core = U_core_strict_upper + torch.diag(diag)          # shape (k, k)

        # 4) Append extra columns if n > k
        if n > k:
            extra_cols = self.U_params[:, k:]          # shape (k, n−k)
            U = torch.cat([U_core, extra_cols], dim=1) # shape (k, n)
        else:
            U = U_core  # shape (k, k)

        return U

    def forward(self, x):
        L = self._build_L()    # (m, k)  or (m, m)
        U = self._build_U()    # (k, n)  or (k, n)
        W = L @ U              # (m, n) = (out_features, in_features)

        out = F.linear(x, W, self.bias)
        return F.leaky_relu(out, self.negative_slope) if self.activate else out

    def __repr__(self):
        activation_str = f"LeakyReLU(negative_slope={self.negative_slope})" if self.activate else "None"
        return f"LULinear(in_features={self.in_features}, "\
            f"out_features={self.out_features}, "\
            f"bias={self.bias is not None}, "\
            f"delta={self.delta}, "\
            f"activate={activation_str})"



def full_rank_mlp(in_dim, mlp_dims, out_dim,
                  delta=1e-4, negative_slope=1e-2):
    """
    Build an MLP with LU-parameterised full-rank layers and Leaky-ReLU
    activations on all hidden blocks.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]

    layers = []
    for i in range(len(dims) - 1):
        activate = i < len(dims) - 2                  # last layer stays linear
        layers.append(
            LULinear(dims[i], dims[i + 1],
                     delta=delta,
                     negative_slope=negative_slope,
                     activate=activate)
        )
    return nn.Sequential(*layers)
