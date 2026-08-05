"""
clustering tendency of the latent embedding
"is there any structure in the data?"
hopkins statistic, VAT/iVAT
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


def _compute_hopkins_statistic(
    latent: np.ndarray,
    *,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Optional[float]:
    """Estimate clustering tendency with the Hopkins statistic.

    Values near 0.5 indicate a random cloud; larger values indicate stronger
    clustering tendency.
    """
    from sklearn.neighbors import NearestNeighbors

    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim != 2 or len(latent) < 3:
        return None

    n_samples, n_features = latent.shape
    mins = np.min(latent, axis=0)
    maxs = np.max(latent, axis=0)
    if np.any(~np.isfinite(mins)) or np.any(~np.isfinite(maxs)):
        return None
    if np.allclose(mins, maxs):
        return None

    m = int(sample_size or min(max(10, n_samples // 10), 256))
    m = max(1, min(m, n_samples - 1))

    rng = np.random.default_rng(random_state)
    real_idx = rng.choice(n_samples, size=m, replace=False)
    synthetic = rng.uniform(mins, maxs, size=(m, n_features))

    nn = NearestNeighbors(n_neighbors=2).fit(latent)
    real_distances = nn.kneighbors(latent[real_idx], return_distance=True)[0][:, 1]
    synthetic_distances = nn.kneighbors(synthetic, return_distance=True)[0][:, 0]

    real_sum = float(np.sum(real_distances))
    synth_sum = float(np.sum(synthetic_distances))
    denom = real_sum + synth_sum
    if denom <= 0.0:
        return None
    return synth_sum / denom


def _vat_reordering(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return VAT-reordered distance matrix, order, and MST parent links."""
    from scipy.spatial.distance import pdist, squareform

    latent = np.asarray(latent, dtype=np.float64)
    n_samples = len(latent)
    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=int), np.empty(0, dtype=int)
    if n_samples == 1:
        return np.zeros((1, 1), dtype=np.float64), np.array([0], dtype=int), np.array([-1], dtype=int)

    dist_matrix = squareform(pdist(latent, metric="euclidean"))
    seed_i, _seed_j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    order: List[int] = [int(seed_i)]
    parents: List[int] = [-1]
    remaining = set(range(n_samples))
    remaining.remove(int(seed_i))

    while remaining:
        current = np.array(order, dtype=int)
        candidates = np.array(sorted(remaining), dtype=int)
        candidate_distances = dist_matrix[np.ix_(current, candidates)]
        min_to_tree = np.min(candidate_distances, axis=0)
        candidate_choice = int(np.argmin(min_to_tree))
        next_idx = int(candidates[candidate_choice])
        parent_in_current = int(np.argmin(candidate_distances[:, candidate_choice]))
        parent_idx = parent_in_current
        order.append(next_idx)
        parents.append(parent_idx)
        remaining.remove(next_idx)

    order_arr = np.asarray(order, dtype=int)
    reordered = dist_matrix[np.ix_(order_arr, order_arr)]
    return reordered, order_arr, np.asarray(parents, dtype=int)


def _ivat_from_vat(
    vat_matrix: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Compute an iVAT-style path-max distance image from a VAT MST."""
    n_samples = len(vat_matrix)
    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float64)
    if n_samples == 1:
        return np.zeros((1, 1), dtype=np.float64)

    adjacency: List[List[tuple[int, float]]] = [[] for _ in range(n_samples)]
    for child in range(1, n_samples):
        parent = int(parents[child])
        if parent < 0:
            continue
        weight = float(vat_matrix[child, parent])
        adjacency[child].append((parent, weight))
        adjacency[parent].append((child, weight))

    ivat = np.zeros((n_samples, n_samples), dtype=np.float64)
    for start in range(n_samples):
        visited = {start}
        stack: List[tuple[int, float]] = [(start, 0.0)]
        while stack:
            node, current_max = stack.pop()
            ivat[start, node] = current_max
            for neighbor, weight in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append((neighbor, max(current_max, weight)))

    return ivat


def _summarize_cluster_tendency(
    latent: np.ndarray,
    *,
    hopkins_threshold: float = 0.55,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute tendency diagnostics before clustering latent vectors.

    The automated gate is based on Hopkins. VAT/iVAT are saved as supporting
    diagnostics because they are primarily visual assessments.
    """
    latent = np.asarray(latent, dtype=np.float64)
    hopkins = _compute_hopkins_statistic(
        latent,
        sample_size=sample_size,
        random_state=random_state,
    )
    vat_matrix, vat_order, vat_parents = _vat_reordering(latent)
    ivat_matrix = _ivat_from_vat(vat_matrix, vat_parents)

    gate_passed = True if hopkins is None else bool(hopkins >= hopkins_threshold)
    return {
        "n_samples": int(len(latent)),
        "latent_dim": int(latent.shape[1]) if latent.ndim == 2 else None,
        "hopkins": hopkins,
        "hopkins_threshold": float(hopkins_threshold),
        "gate_passed": gate_passed,
        "gate_reason": (
            "hopkins_below_threshold" if hopkins is not None and hopkins < hopkins_threshold else "pass"
        ),
        "vat_order": vat_order.tolist(),
        "vat_parents": vat_parents.tolist(),
        "vat_matrix": vat_matrix,
        "ivat_matrix": ivat_matrix,
    }


def save_tendency_heatmap(
    matrix: np.ndarray,
    output_path: Union[str, Path],
    *,
    title: str,
) -> Path:
    """Save a VAT or iVAT matrix as an inspectable heatmap."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError("Tendency heatmap requires a non-empty square matrix")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6.0, 5.0))
    image = axis.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("VAT-reordered sample index")
    axis.set_ylabel("VAT-reordered sample index")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = [
    "_compute_hopkins_statistic",
    "_ivat_from_vat",
    "_summarize_cluster_tendency",
    "_vat_reordering",
    "save_tendency_heatmap",
]
