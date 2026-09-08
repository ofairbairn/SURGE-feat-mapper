"""Procrustes alignment for Mapper's Latent Atlas across encoder retrains.

When an encoder is retrained, its latent space can come out arbitrarily
rotated/reflected/rescaled even if the underlying structure is unchanged.
This module finds the rigid + uniform-scale transform that maps a new run's
latent space onto a reference run's latent space, using a shared "anchor"
subsample re-encoded by both, so cluster identities and Atlas positions stay
comparable across retrains (Schonemann 1966; Gower 1975).

Convention follows ``scipy.spatial.procrustes`` (translate each cloud to its
own centroid, scale to unit Frobenius norm, then solve an orthogonal
rotation/reflection). Unlike ``scipy.spatial.procrustes`` -- which only
returns the two standardized anchor matrices -- this module keeps the
per-cloud mean/scale/rotation components so the same transform can be
reapplied to samples outside the anchor set (the whole point: align the
*entire* new run's latent space, not just its anchor points).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes as _scipy_procrustes


def select_anchor_positions(
    n_samples: int,
    *,
    n_anchor: int,
    random_state: int,
) -> np.ndarray:
    """Choose deterministic row positions to serve as a reference anchor set."""
    if n_samples < 2:
        raise ValueError("Anchor selection requires at least two samples")
    n_anchor = max(2, min(int(n_anchor), n_samples))
    rng = np.random.default_rng(random_state)
    return np.sort(
        rng.choice(n_samples, size=n_anchor, replace=False)
    ).astype(np.int64)


def fit_procrustes_alignment(
    anchor_reference: np.ndarray,
    anchor_new: np.ndarray,
) -> Dict[str, Any]:
    """Solve for the transform mapping ``anchor_new`` onto ``anchor_reference``.

    Both inputs must be the *same* physical samples encoded by the reference
    and the new encoder respectively, row-for-row. Returns the transform
    components (apply with :func:`apply_procrustes_alignment`) plus QC
    figures, including the disparity from an independent
    ``scipy.spatial.procrustes`` call on the same anchor pair.
    """
    reference = np.asarray(anchor_reference, dtype=np.float64)
    new = np.asarray(anchor_new, dtype=np.float64)
    if reference.shape != new.shape:
        raise ValueError(
            "Anchor reference/new shapes must match, got "
            f"{reference.shape} vs {new.shape}"
        )
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("Procrustes alignment requires at least 2 anchor points")

    mean_reference = reference.mean(axis=0)
    mean_new = new.mean(axis=0)
    centered_reference = reference - mean_reference
    centered_new = new - mean_new

    norm_reference = float(np.linalg.norm(centered_reference))
    norm_new = float(np.linalg.norm(centered_new))
    if norm_reference == 0.0 or norm_new == 0.0:
        raise ValueError(
            "Degenerate anchor set: reference or new anchors have zero "
            "variance after centering"
        )

    normalized_reference = centered_reference / norm_reference
    normalized_new = centered_new / norm_new

    # Solve for the orthogonal (rotation/reflection) matrix R minimizing
    # ||normalized_new @ R - normalized_reference||_F.
    rotation, _ = orthogonal_procrustes(normalized_new, normalized_reference)
    scale = norm_reference / norm_new

    # Independent goodness-of-fit cross-check on the same anchor pair, using
    # scipy.spatial.procrustes directly (it re-derives its own rotation
    # internally; only its disparity score is used here for QC/reporting).
    _, _, disparity = _scipy_procrustes(reference, new)

    aligned_anchor_new = (
        normalized_new @ rotation
    ) * norm_reference + mean_reference
    residual = aligned_anchor_new - reference
    anchor_rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))

    return {
        "mean_reference": mean_reference,
        "mean_new": mean_new,
        "norm_reference": norm_reference,
        "norm_new": norm_new,
        "rotation": rotation,
        "scale": float(scale),
        "n_anchor": int(reference.shape[0]),
        "latent_dim": int(reference.shape[1]),
        "reflection": bool(np.linalg.det(rotation) < 0),
        "disparity": float(disparity),
        "anchor_rmse": anchor_rmse,
    }


def apply_procrustes_alignment(
    Z: np.ndarray,
    alignment: Dict[str, Any],
) -> np.ndarray:
    """Apply a fitted alignment transform to any latent matrix (not just anchors)."""
    Z = np.asarray(Z, dtype=np.float64)
    centered = Z - alignment["mean_new"]
    normalized = centered / alignment["norm_new"]
    rotated = normalized @ alignment["rotation"]
    return rotated * alignment["norm_reference"] + alignment["mean_reference"]


def save_procrustes_alignment(path: Path, alignment: Dict[str, Any]) -> Path:
    np.savez_compressed(
        path,
        mean_reference=alignment["mean_reference"],
        mean_new=alignment["mean_new"],
        norm_reference=np.float64(alignment["norm_reference"]),
        norm_new=np.float64(alignment["norm_new"]),
        rotation=alignment["rotation"],
        scale=np.float64(alignment["scale"]),
    )
    return path


def load_procrustes_alignment(path: Path) -> Dict[str, Any]:
    with np.load(path) as data:
        return {
            "mean_reference": data["mean_reference"],
            "mean_new": data["mean_new"],
            "norm_reference": float(data["norm_reference"]),
            "norm_new": float(data["norm_new"]),
            "rotation": data["rotation"],
            "scale": float(data["scale"]),
        }


def save_reference_anchors(
    path: Path,
    *,
    sample_index: np.ndarray,
    X_raw_anchor: np.ndarray,
    Z_anchor: np.ndarray,
    selected_rung: str,
    input_columns: Sequence[str],
) -> Path:
    """Persist a run's anchor subsample so a future retrain can align to it.

    ``X_raw_anchor`` stores the anchors' *unscaled* input features (in
    ``input_columns`` order) rather than scaled/model-ready values, since a
    future run fits its own scaler on its own train split -- re-scaling the
    anchors with that run's own scaler keeps preprocessing self-consistent.
    """
    np.savez_compressed(
        path,
        sample_index=np.asarray(sample_index),
        X_raw_anchor=np.asarray(X_raw_anchor),
        Z_anchor=np.asarray(Z_anchor),
        selected_rung=np.array(str(selected_rung)),
        input_columns=np.array([str(c) for c in input_columns], dtype=object),
    )
    return path


def load_reference_anchors(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "sample_index": data["sample_index"],
            "X_raw_anchor": data["X_raw_anchor"],
            "Z_anchor": data["Z_anchor"],
            "selected_rung": str(data["selected_rung"]),
            "input_columns": [str(c) for c in data["input_columns"]],
        }


__all__ = [
    "select_anchor_positions",
    "fit_procrustes_alignment",
    "apply_procrustes_alignment",
    "save_procrustes_alignment",
    "load_procrustes_alignment",
    "save_reference_anchors",
    "load_reference_anchors",
]
