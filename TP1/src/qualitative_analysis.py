"""Utilities for latent-space qualitative analysis.

This module provides:
- lerp: linear interpolation.
- slerp: spherical interpolation.
- t-SNE helpers to project and visualize latent vectors.

Example:
    python src/qualitative_analysis.py \
        --latents src/results/latents.npy \
        --labels src/results/labels.npy \
        --output src/results/tsne_latents.png
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch


TensorLike = Union[float, torch.Tensor]
ArrayLike = Union[np.ndarray, torch.Tensor]


def _prepare_t(reference: torch.Tensor, t: TensorLike) -> torch.Tensor:
    """Convert t to a tensor and make it broadcastable with reference."""
    if torch.is_tensor(t):
        t_tensor = t.to(device=reference.device, dtype=reference.dtype)
    else:
        t_tensor = torch.tensor(t, device=reference.device, dtype=reference.dtype)

    while t_tensor.ndim < reference.ndim:
        t_tensor = t_tensor.unsqueeze(-1)
    return t_tensor


def lerp(start: torch.Tensor, end: torch.Tensor, t: TensorLike) -> torch.Tensor:
    """Linearly interpolate between start and end using parameter t in [0, 1]."""
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape")

    t_tensor = _prepare_t(start, t)
    return start + (end - start) * t_tensor


def slerp(start: torch.Tensor, end: torch.Tensor, t: TensorLike, eps: float = 1e-7) -> torch.Tensor:
    """Spherically interpolate between vectors along the last dimension.

    Falls back to lerp for near-colinear vectors for numerical stability.
    """
    if start.shape != end.shape:
        raise ValueError("start and end must have the same shape")
    if start.ndim == 0:
        raise ValueError("slerp expects tensors with at least one dimension")

    t_tensor = _prepare_t(start, t)

    start_norm = torch.linalg.vector_norm(start, dim=-1, keepdim=True).clamp_min(eps)
    end_norm = torch.linalg.vector_norm(end, dim=-1, keepdim=True).clamp_min(eps)

    start_dir = start / start_norm
    end_dir = end / end_norm

    dot = torch.sum(start_dir * end_dir, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    lerp_result = lerp(start, end, t_tensor)

    coeff_start = torch.sin((1.0 - t_tensor) * omega) / (sin_omega + eps)
    coeff_end = torch.sin(t_tensor * omega) / (sin_omega + eps)

    interp_dir = coeff_start * start_dir + coeff_end * end_dir
    interp_norm = lerp(start_norm, end_norm, t_tensor)
    slerp_result = interp_dir * interp_norm

    near_colinear = sin_omega.abs() < eps
    return torch.where(near_colinear, lerp_result, slerp_result)


def _to_numpy_array(values: ArrayLike) -> np.ndarray:
    """Convert torch or numpy array-like values to numpy."""
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _to_2d_latents(latents: ArrayLike) -> np.ndarray:
    """Ensure latent vectors are in shape [N, D] for sklearn TSNE."""
    latents_np = _to_numpy_array(latents)
    if latents_np.ndim == 1:
        latents_np = latents_np[:, None]
    elif latents_np.ndim > 2:
        latents_np = latents_np.reshape(latents_np.shape[0], -1)
    if latents_np.shape[0] < 2:
        raise ValueError("Need at least 2 latent samples for t-SNE")
    return latents_np.astype(np.float32, copy=False)


def _load_array(path: Union[str, Path], key: Optional[str] = None) -> np.ndarray:
    """Load arrays from .npy/.npz/.pt/.pth files."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".npy":
        return np.load(p)

    if suffix == ".npz":
        loaded = np.load(p)
        if key is not None:
            if key not in loaded:
                raise KeyError(f"Key '{key}' not found in {p}")
            return loaded[key]
        if not loaded.files:
            raise ValueError(f"No arrays found inside {p}")
        return loaded[loaded.files[0]]

    if suffix in {".pt", ".pth"}:
        loaded = torch.load(p, map_location="cpu")
        if torch.is_tensor(loaded):
            return loaded.detach().cpu().numpy()
        if isinstance(loaded, dict):
            if key is not None:
                if key not in loaded:
                    raise KeyError(f"Key '{key}' not found in {p}")
                return _to_numpy_array(loaded[key])
            for candidate in ("latents", "embeddings", "z", "features", "labels"):
                if candidate in loaded:
                    return _to_numpy_array(loaded[candidate])
            raise ValueError(
                f"Could not infer array key in {p}. "
                "Pass --latents-key or --labels-key."
            )
        raise TypeError(f"Unsupported .pt/.pth content type: {type(loaded)}")

    raise ValueError(f"Unsupported file extension for {p}")


def _parse_learning_rate(value: str) -> Union[str, float]:
    """Allow 'auto' or a numeric learning rate from CLI."""
    if value.strip().lower() == "auto":
        return "auto"
    return float(value)


def run_tsne(
    latents: ArrayLike,
    perplexity: float = 30.0,
    learning_rate: Union[str, float] = "auto",
    n_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Project latent vectors to 2D with t-SNE."""
    try:
        tsne_class = importlib.import_module("sklearn.manifold").TSNE
    except ModuleNotFoundError as exc:
        raise ImportError("scikit-learn is required for t-SNE. Install it with: pip install scikit-learn")
    except AttributeError as exc:
        raise ImportError("Could not access sklearn.manifold.TSNE. Check your scikit-learn install.") from exc

    latents_2d = _to_2d_latents(latents)
    n_samples = latents_2d.shape[0]

    max_perplexity = max(1.0, (n_samples - 1) / 3.0)
    used_perplexity = min(float(perplexity), max_perplexity)

    tsne_kwargs = {
        "n_components": 2,
        "perplexity": used_perplexity,
        "learning_rate": learning_rate,
        "random_state": seed,
        "init": "pca",
    }

    # Keep compatibility with sklearn versions where n_iter was renamed to max_iter.
    signature = inspect.signature(tsne_class.__init__)
    if "max_iter" in signature.parameters:
        tsne_kwargs["max_iter"] = n_iter
    else:
        tsne_kwargs["n_iter"] = n_iter

    model = tsne_class(**tsne_kwargs)
    embedding = model.fit_transform(latents_2d)
    return embedding.astype(np.float32, copy=False)


def plot_tsne(
    embedding: np.ndarray,
    labels: Optional[ArrayLike] = None,
    class_names: Optional[Sequence[str]] = None,
    title: str = "t-SNE of latent vectors",
    point_size: float = 10.0,
    alpha: float = 0.8,
    show_legend: bool = True,
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot a 2D t-SNE embedding and optionally save it."""
    emb = _to_numpy_array(embedding)
    if emb.ndim != 2 or emb.shape[1] != 2:
        raise ValueError("embedding must have shape [N, 2]")

    if labels is None:
        labels_np = np.zeros(emb.shape[0], dtype=np.int64)
    else:
        labels_np = _to_numpy_array(labels).reshape(-1)
        if labels_np.shape[0] != emb.shape[0]:
            raise ValueError("labels and embedding must have the same number of rows")

    unique_labels = np.unique(labels_np)

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels_np == label
        legend_name = str(label)
        if class_names is not None:
            label_int = int(label)
            if 0 <= label_int < len(class_names):
                legend_name = class_names[label_int]

        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=point_size,
            alpha=alpha,
            color=cmap(idx),
            label=legend_name,
        )

    plt.title(title)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid(alpha=0.2)

    if show_legend and len(unique_labels) <= 20:
        plt.legend(loc="best", fontsize=8, frameon=True)

    plt.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=180)
        print(f"Saved t-SNE plot to: {output}")

    plt.close()


def _maybe_subsample(
    latents: np.ndarray,
    labels: Optional[np.ndarray],
    max_samples: Optional[int],
    seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Subsample latent vectors to keep t-SNE runtime manageable."""
    if max_samples is None or latents.shape[0] <= max_samples:
        return latents, labels

    rng = np.random.default_rng(seed)
    idx = rng.choice(latents.shape[0], size=max_samples, replace=False)
    idx = np.sort(idx)

    latents_sub = latents[idx]
    labels_sub = labels[idx] if labels is not None else None
    return latents_sub, labels_sub


def build_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for the t-SNE utility CLI."""
    parser = argparse.ArgumentParser(description="Run t-SNE on latent vectors and save a plot.")
    parser.add_argument("--latents", type=Path, required=True, help="Path to .npy/.npz/.pt with latent vectors")
    parser.add_argument("--labels", type=Path, default=None, help="Optional path to labels (.npy/.npz/.pt)")
    parser.add_argument("--latents-key", type=str, default=None, help="Key for .npz/.pt latent arrays")
    parser.add_argument("--labels-key", type=str, default=None, help="Key for .npz/.pt label arrays")
    parser.add_argument("--output", type=Path, default=Path("src/results/tsne_latents.png"), help="Output plot path")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--learning-rate", type=str, default="auto", help="'auto' or numeric value")
    parser.add_argument("--n-iter", type=int, default=1000, help="Number of t-SNE optimization iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=5000, help="Optional subsample cap for speed")
    parser.add_argument("--title", type=str, default="t-SNE of latent vectors", help="Plot title")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size")
    parser.add_argument("--alpha", type=float, default=0.8, help="Scatter alpha")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend")
    return parser


def main() -> None:
    """CLI entry point for t-SNE latent-space visualization."""
    args = build_cli_parser().parse_args()

    learning_rate = _parse_learning_rate(args.learning_rate)
    latents = _load_array(args.latents, key=args.latents_key)
    labels = _load_array(args.labels, key=args.labels_key) if args.labels is not None else None

    latents_2d = _to_2d_latents(latents)
    labels_1d = _to_numpy_array(labels).reshape(-1) if labels is not None else None

    if labels_1d is not None and labels_1d.shape[0] != latents_2d.shape[0]:
        raise ValueError("labels and latents must have the same number of samples")

    latents_2d, labels_1d = _maybe_subsample(latents_2d, labels_1d, args.max_samples, args.seed)

    embedding = run_tsne(
        latents_2d,
        perplexity=args.perplexity,
        learning_rate=learning_rate,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    plot_tsne(
        embedding=embedding,
        labels=labels_1d,
        title=args.title,
        point_size=args.point_size,
        alpha=args.alpha,
        show_legend=not args.no_legend,
        output_path=args.output,
    )


__all__ = ["lerp", "slerp", "run_tsne", "plot_tsne"]


if __name__ == "__main__":
    main()