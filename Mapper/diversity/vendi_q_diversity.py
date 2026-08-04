"""Global Vendi Score and order-q diversity profiles for Mapper embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import pairwise_distances

QValue = Union[float, str]

DEFAULT_Q_VALUES: tuple[QValue, ...] = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    "inf",
)


@dataclass(frozen=True)
class VendiDiversityResult:
    """Computed Vendi report plus reusable matrix-level artifacts."""

    report: dict[str, Any]
    similarity_matrix: np.ndarray
    sample_positions: np.ndarray


def build_rbf_similarity_matrix(
    embeddings: Any,
    *,
    bandwidth: Optional[float] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a positive-semidefinite RBF kernel on latent embeddings.

    When ``bandwidth`` is omitted, the median nonzero pairwise Euclidean
    distance is used. The resulting kernel has unit diagonal, as required by
    the Vendi Score.
    """
    Z = _as_embedding_matrix(embeddings)
    squared_distances = pairwise_distances(Z, metric="sqeuclidean")
    positive_squared_distances = squared_distances[squared_distances > 0.0]

    selection = "provided"
    if bandwidth is None:
        selection = "median_nonzero_pairwise_distance"
        if positive_squared_distances.size:
            bandwidth = float(np.sqrt(np.median(positive_squared_distances)))
        else:
            bandwidth = 1.0
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("RBF bandwidth must be a finite positive number")

    gamma = 1.0 / (2.0 * bandwidth**2)
    similarity = np.exp(-gamma * squared_distances)
    similarity = (similarity + similarity.T) / 2.0
    np.fill_diagonal(similarity, 1.0)
    return similarity, {
        "name": "rbf",
        "bandwidth": bandwidth,
        "gamma": gamma,
        "bandwidth_selection": selection,
        "diagonal": 1.0,
    }


def compute_vendi_diversity(
    embeddings: Any,
    *,
    q_values: Optional[Sequence[QValue]] = None,
    rbf_bandwidth: Optional[float] = None,
    max_samples: Optional[int] = 2_000,
    random_state: int = 42,
) -> VendiDiversityResult:
    """Compute Vendi Score V1 on a selected latent representation.

    The original Vendi Score is the order-q score at ``q=1``. The q-profile
    accepts every nonnegative finite q plus ``"inf"``; it is not restricted
    to the interval [0, 1]. For scalability, a reproducible subset is used
    when the embedding contains more than ``max_samples`` rows.
    """
    Z = _as_embedding_matrix(embeddings)
    sample_positions = _sample_positions(
        len(Z), max_samples=max_samples, random_state=random_state
    )
    Z_used = Z[sample_positions]
    similarity, kernel_report = build_rbf_similarity_matrix(
        Z_used,
        bandwidth=rbf_bandwidth,
    )

    try:
        from vendi_score import vendi
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError(
            "Mapper diversity requires vendi-score. Install it with "
            "`pip install vendi-score`."
        ) from exc

    vendi_score = float(vendi.score_K(similarity, q=1))
    if not np.isfinite(vendi_score):
        raise ValueError("vendi-score returned a non-finite global score")

    eigenvalues, tolerance = _normalized_kernel_eigenvalues(similarity)
    orders = _normalize_q_values(
        DEFAULT_Q_VALUES if q_values is None else q_values
    )
    q_profile = []
    for q in orders:
        score = (
            vendi_score
            if q == 1.0
            else _effective_diversity_from_eigenvalues(eigenvalues, q)
        )
        q_profile.append({"q": _serialize_q(q), "score": float(score)})

    report = {
        "method": "vendi_score",
        "vendi_score": vendi_score,
        "q_order": 1.0,
        "effective_modes": vendi_score,
        "interpretation": "effective number of distinct latent modes",
        "embedding": {
            "space": "selected_model_latent_z",
            "n_samples_total": int(len(Z)),
            "n_samples_used": int(len(Z_used)),
            "latent_dimensions": int(Z.shape[1]),
            "subsampled": bool(len(Z_used) < len(Z)),
            "max_samples": None if max_samples is None else int(max_samples),
            "random_state": int(random_state),
        },
        "kernel": kernel_report,
        "spectrum": {
            "effective_rank_tolerance": tolerance,
            "positive_eigenvalues": int(len(eigenvalues)),
            "largest_eigenvalue": float(np.max(eigenvalues)),
            "eigenvalue_sum": float(np.sum(eigenvalues)),
        },
        "q_profile": q_profile,
        "q_guidance": {
            "domain": "q >= 0, including the q -> infinity limit",
            "below_1": "more sensitive to rare latent modes",
            "at_1": "original Shannon-entropy Vendi Score",
            "above_1": "more sensitive to common or dominant latent modes",
        },
    }
    return VendiDiversityResult(
        report=report,
        similarity_matrix=similarity,
        sample_positions=sample_positions,
    )


def plot_vendi_q_profile(
    result: Union[VendiDiversityResult, dict[str, Any]],
    output_path: Union[str, Path],
    *,
    title: str = "Vendi diversity profile of selected latent embedding",
) -> Path:
    """Save a q-order Vendi diversity profile as a PNG image."""
    import matplotlib.pyplot as plt

    report = result.report if isinstance(result, VendiDiversityResult) else result
    profile = report.get("q_profile", [])
    if not profile:
        raise ValueError("Vendi report does not contain a q_profile")

    finite_x: list[float] = []
    finite_y: list[float] = []
    infinity_score: Optional[float] = None
    for point in profile:
        if str(point["q"]).lower() == "inf":
            infinity_score = float(point["score"])
        else:
            finite_x.append(float(point["q"]))
            finite_y.append(float(point["score"]))

    fig, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.plot(finite_x, finite_y, marker="o", linewidth=2, label="VS(q)")
    tick_positions = list(finite_x)
    tick_labels = [f"{q:g}" for q in finite_x]
    if infinity_score is not None:
        max_q = max(finite_x, default=1.0)
        infinity_x = max_q + max(0.5, max_q * 0.12)
        axis.plot(
            [infinity_x],
            [infinity_score],
            marker="D",
            linestyle="none",
            label="q → ∞",
        )
        tick_positions.append(infinity_x)
        tick_labels.append("∞")

    q_one = next(
        (float(point["score"]) for point in profile if point["q"] == 1.0),
        None,
    )
    if q_one is not None:
        axis.scatter([1.0], [q_one], color="black", zorder=3, label="Global VS (q=1)")
    axis.set_xticks(tick_positions, tick_labels)
    axis.set_xlabel("Vendi order q")
    axis.set_ylabel("Effective diversity VS(q)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _as_embedding_matrix(embeddings: Any) -> np.ndarray:
    Z = np.asarray(embeddings, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.ndim != 2 or Z.shape[0] < 1 or Z.shape[1] < 1:
        raise ValueError(f"Expected a non-empty 2D latent embedding, got {Z.shape}")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Latent embeddings contain NaN or infinite values")
    return Z


def _sample_positions(
    n_samples: int,
    *,
    max_samples: Optional[int],
    random_state: int,
) -> np.ndarray:
    if max_samples is None:
        return np.arange(n_samples, dtype=np.int64)
    max_samples = int(max_samples)
    if max_samples < 2 and n_samples > 1:
        raise ValueError("max_samples must be at least 2, or None")
    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=max_samples, replace=False))


def _normalized_kernel_eigenvalues(
    similarity: np.ndarray,
) -> tuple[np.ndarray, float]:
    eigenvalues = np.linalg.eigvalsh(similarity / similarity.shape[0])
    largest_magnitude = float(max(1.0, np.max(np.abs(eigenvalues))))
    tolerance = float(
        max(1e-12, similarity.shape[0] * np.finfo(np.float64).eps * largest_magnitude)
    )
    positive = eigenvalues[eigenvalues > tolerance]
    if not positive.size:
        raise ValueError("Similarity matrix has no positive eigenvalues")
    positive = positive / np.sum(positive)
    return positive, tolerance


def _normalize_q_values(q_values: Iterable[QValue]) -> list[Union[float, str]]:
    normalized: set[Union[float, str]] = set()
    for raw_q in q_values:
        if isinstance(raw_q, str):
            if raw_q.strip().lower() not in {"inf", "infinity"}:
                raise ValueError(f"Invalid Vendi order q={raw_q!r}")
            normalized.add("inf")
            continue
        q = float(raw_q)
        if not np.isfinite(q) or q < 0.0:
            raise ValueError("Finite Vendi orders q must satisfy q >= 0")
        normalized.add(q)
    if not normalized:
        raise ValueError("At least one q value is required")
    finite = sorted(float(q) for q in normalized if q != "inf")
    return [*finite, *(["inf"] if "inf" in normalized else [])]


def _effective_diversity_from_eigenvalues(
    eigenvalues: np.ndarray,
    q: Union[float, str],
) -> float:
    if q == "inf":
        return float(1.0 / np.max(eigenvalues))
    q = float(q)
    if np.isclose(q, 0.0):
        return float(len(eigenvalues))
    if np.isclose(q, 1.0):
        return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))
    return float(np.sum(eigenvalues**q) ** (1.0 / (1.0 - q)))


def _serialize_q(q: Union[float, str]) -> Union[float, str]:
    return "inf" if q == "inf" else float(q)


__all__ = [
    "DEFAULT_Q_VALUES",
    "VendiDiversityResult",
    "build_rbf_similarity_matrix",
    "compute_vendi_diversity",
    "plot_vendi_q_profile",
]
