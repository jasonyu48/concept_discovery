#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


def _load_snapshot(snapshot_path: Path):
    payload = torch.load(snapshot_path, map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot does not contain a dict payload: {type(payload)}")

    enc = payload.get('encodings', None)
    if enc is None:
        raise KeyError("'encodings' not found in snapshot payload")
    if isinstance(enc, torch.Tensor):
        enc = enc.detach().cpu()
    enc = enc.float()
    if enc.ndim > 2:
        enc = enc.view(enc.shape[0], -1)
    enc_np = enc.numpy()

    labels = payload.get('labels', None)
    if labels is None:
        labels_np = np.zeros(enc_np.shape[0], dtype=int)
    else:
        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = np.array(labels)
        if labels_np.ndim > 1:
            labels_np = labels_np.reshape(-1)
        if labels_np.shape[0] != enc_np.shape[0]:
            raise ValueError(
                f"labels length ({labels_np.shape[0]}) does not match encodings ({enc_np.shape[0]})"
            )

    step = payload.get('step', None)
    return enc_np, labels_np, step


def _compute_tsne(X: np.ndarray, perplexity: float, random_state: int = 42) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("scikit-learn is required. Install with: pip install scikit-learn", file=sys.stderr)
        raise

    n = X.shape[0]
    # Per t-SNE constraints, perplexity must be < (n_samples - 1) / 3
    max_perp = max(5.0, (n - 1) / 3.0 - 1e-6)
    adj_perp = float(min(perplexity, max_perp))
    if adj_perp < 5.0:
        adj_perp = max(2.0, adj_perp)
    if abs(adj_perp - perplexity) > 1e-9:
        print(f"[info] Adjusted perplexity from {perplexity} to {adj_perp:.2f} for {n} samples")

    tsne = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=adj_perp,
        n_iter=1000,
        random_state=random_state,
    )
    Y = tsne.fit_transform(X)
    return Y, adj_perp


def _plot(Y: np.ndarray, labels: np.ndarray, out_path: Path, title: str = "t-SNE Latent Space"):
    import matplotlib.pyplot as plt

    labels_unique = np.unique(labels)
    num_classes = len(labels_unique)

    if num_classes <= 10:
        base_cmap = plt.get_cmap("tab10")
    elif num_classes <= 20:
        base_cmap = plt.get_cmap("tab20")
    else:
        base_cmap = plt.get_cmap("tab20")

    colors = [base_cmap(i % base_cmap.N) for i in range(num_classes)]

    plt.figure(figsize=(6, 5))
    for cls, col in zip(labels_unique, colors):
        idx = labels == cls
        plt.scatter(Y[idx, 0], Y[idx, 1], s=10, alpha=0.85, label=str(cls), color=col)

    if num_classes > 1:
        plt.legend(title="Label", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def _compute_pca(X: np.ndarray, random_state: int = 42):
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("scikit-learn is required. Install with: pip install scikit-learn", file=sys.stderr)
        raise

    pca = PCA(n_components=2, random_state=random_state)
    Y = pca.fit_transform(X)
    var_ratio = getattr(pca, 'explained_variance_ratio_', None)
    return Y, var_ratio


def _plot_label_sorted_distance_heatmap(X: np.ndarray, labels: np.ndarray, out_path: Path, title: str = "Label-sorted pairwise distances"):
    import matplotlib.pyplot as plt

    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    # Compute pairwise distances in latent space
    with torch.no_grad():
        Xt = torch.from_numpy(X).float()
        D = torch.cdist(Xt, Xt, p=2).cpu().numpy()

    # Sort by labels for block-diagonal structure
    order = np.argsort(labels)
    D_sorted = D[order][:, order]
    labels_sorted = labels[order]

    # Determine boundaries where label changes
    change_idxs = np.where(np.diff(labels_sorted) != 0)[0]

    # Robust color scaling to reduce effect of outliers
    vmax = float(np.percentile(D_sorted, 99.0)) if np.isfinite(D_sorted).all() else None

    plt.figure(figsize=(6, 5))
    im = plt.imshow(D_sorted, cmap='viridis', origin='lower', interpolation='nearest', vmax=vmax)
    plt.title(title)
    plt.xlabel('Samples (sorted by label)')
    plt.ylabel('Samples (sorted by label)')

    # Draw grid lines at class boundaries
    for idx in change_idxs:
        pos = idx + 0.5
        plt.axhline(pos, color='white', linewidth=0.5, alpha=0.7)
        plt.axvline(pos, color='white', linewidth=0.5, alpha=0.7)

    cbar = plt.colorbar(im)
    cbar.set_label('L2 distance')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    pjepa_path = '/home/jyu197/onlyreward/concept_discovery/tdmpc2/tdmpc2/logs/counting4/2022/final/concept_discovery_P_JEPA4/encoding_monitor/baseline_encodings/baseline_encodings_best.pt'
    random_path = '/home/jyu197/onlyreward/concept_discovery/tdmpc2/tdmpc2/logs/counting4/2022/final/concept_discovery_random/encoding_monitor/baseline_encodings/baseline_encodings_best.pt'
    onlyreward_path = '/home/jyu197/onlyreward/concept_discovery/tdmpc2/tdmpc2/logs/counting4/2022/final/OR10/encoding_monitor/baseline_encodings/baseline_encodings_best.pt'
    parser = argparse.ArgumentParser(description="t-SNE of baseline encodings from snapshot .pt")
    parser.add_argument("snapshot_path", nargs='?', type=str, default=onlyreward_path, help="Path to baseline_encodings_best.pt (optional; default used if omitted)")
    parser.add_argument("perplexity", nargs='?', type=float, default=30.0, help="t-SNE perplexity (optional; default 30.0)")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for the PNG plot")
    args = parser.parse_args()

    snap_path = Path(args.snapshot_path)
    if not snap_path.exists():
        print(f"Snapshot not found: {snap_path}", file=sys.stderr)
        sys.exit(1)

    enc_np, labels_np, step = _load_snapshot(snap_path)
    Y, adj_perp = _compute_tsne(enc_np, args.perplexity)

    if args.output is not None:
        out_path = Path(args.output)
    else:
        suffix = f"_step{step}" if step is not None else ""
        out_name = f"tsne_baseline_encodings_perp{int(round(adj_perp))}{suffix}.png"
        out_path = snap_path.parent / out_name

    title = (
        f"t-SNE Latent Space (perplexity={adj_perp:.2f}{', step='+str(step) if step is not None else ''})"
    )
    _plot(Y, labels_np, out_path, title=title)
    print(f"Saved t-SNE plot to {out_path}")

    # Also generate a 2D PCA scatter for comparison
    Y_pca, var_ratio = _compute_pca(enc_np)
    if args.output is not None:
        out_path_pca = Path(args.output).with_name(Path(args.output).stem + "_pca.png")
    else:
        suffix = f"_step{step}" if step is not None else ""
        out_name_pca = f"pca_baseline_encodings{suffix}.png"
        out_path_pca = snap_path.parent / out_name_pca

    if isinstance(var_ratio, np.ndarray) and var_ratio.size >= 2:
        total_var = float(var_ratio[:2].sum())
        title_pca = f"PCA 2D Latent Space (explained var={total_var:.2f})"
    else:
        title_pca = "PCA 2D Latent Space"
    _plot(Y_pca, labels_np, out_path_pca, title=title_pca)
    print(f"Saved PCA plot to {out_path_pca}")

    # Label-sorted pairwise distance heatmap in latent space
    if args.output is not None:
        out_path_heat = Path(args.output).with_name(Path(args.output).stem + "_dist_heatmap.png")
    else:
        suffix = f"_step{step}" if step is not None else ""
        out_name_heat = f"dist_heatmap_sorted{suffix}.png"
        out_path_heat = snap_path.parent / out_name_heat
    _plot_label_sorted_distance_heatmap(enc_np, labels_np, out_path_heat, title="Label-sorted pairwise distances (latent)")
    print(f"Saved label-sorted distance heatmap to {out_path_heat}")


if __name__ == "__main__":
    main()


