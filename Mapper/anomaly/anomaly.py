"""
anomaly detection module for Mapper.
Uses Reconstruction error, latent density, Global local outlier score
from hierarchies (GLOSH), IsoForest, Local outlier factor LOF, marginal
vendi contribution, each contributing to an anomaly score that is reported
in output of mapper run. Each metric is reported, and a combined score ranking
the 100 most anomalous points is reported alongside all the metrics scores for
that data point e.g. sample_x = 0.89 : recon_error=0.9, latent_density=0.8...
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

_SIGNAL_REASON_TEXT = {
    "reconstruction_error": "high reconstruction error",
    "hdbscan_glosh": "HDBSCAN flags it as noise (high GLOSH outlier score)",
    "gmm_mahalanobis": "far from every latent cluster (high Mahalanobis distance)",
    "isolation_forest": "flagged by Isolation Forest",
    "local_outlier_factor": "flagged by Local Outlier Factor",
    "marginal_vendi": "represents a distinct new latent mode (high marginal Vendi contribution)",
}

_DENSITY_SIGNALS = (
    "hdbscan_glosh",
    "gmm_mahalanobis",
    "isolation_forest",
    "local_outlier_factor",
)

_SIGNAL_WEIGHTS = {
    "reconstruction_error": 1.0,
    "hdbscan_glosh": 0.5,
    "gmm_mahalanobis": 0.5,
    "isolation_forest": 1.0,
    "local_outlier_factor": 1.0,
    "marginal_vendi": 1.0,
}


def _as_2d_array(latent: Any) -> np.ndarray:
    arr = np.asarray(latent, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D latent array, got shape={arr.shape}")
    return arr


def _subsample_positions(
    n_samples: int,
    *,
    max_samples: Optional[int],
    random_state: int,
) -> np.ndarray:
    if max_samples is None or int(max_samples) >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(
        rng.choice(n_samples, size=max(2, int(max_samples)), replace=False)
    ).astype(np.int64)


def _percentile_rank(values: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Rank finite scores onto [0, 1]; higher input == higher (more anomalous) rank."""
    if values is None:
        return None
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)
    ranks = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return ranks
    if n_valid == 1:
        ranks[valid] = 1.0
        return ranks
    ranks[valid] = (rankdata(values[valid], method="average") - 1.0) / (n_valid - 1.0)
    return ranks


def _compute_hdbscan_glosh(
    latent: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: Optional[int],
) -> Optional[np.ndarray]:
    """HDBSCAN GLOSH outlier score in [0, 1]; higher = more anomalous (Signal 2)."""
    try:
        import hdbscan
    except ImportError:
        return None

    n_samples = len(latent)
    effective_min_cluster_size = max(2, min(int(min_cluster_size), max(2, n_samples - 1)))
    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective_min_cluster_size,
            min_samples=min_samples,
        )
        clusterer.fit(latent)
        return np.asarray(clusterer.outlier_scores_, dtype=np.float64)
    except Exception:
        return None


def _gmm_component_precisions(gmm: Any) -> List[np.ndarray]:
    """Per-component precision matrices across all sklearn GMM covariance types."""
    n_components, n_features = gmm.means_.shape
    cov_type = gmm.covariance_type
    if cov_type == "full":
        return [np.linalg.pinv(gmm.covariances_[k]) for k in range(n_components)]
    if cov_type == "tied":
        precision = np.linalg.pinv(gmm.covariances_)
        return [precision for _ in range(n_components)]
    if cov_type == "diag":
        return [
            np.diag(1.0 / np.clip(gmm.covariances_[k], 1e-12, None))
            for k in range(n_components)
        ]
    if cov_type == "spherical":
        return [
            np.eye(n_features) / max(float(gmm.covariances_[k]), 1e-12)
            for k in range(n_components)
        ]
    raise ValueError(f"Unsupported GMM covariance_type={cov_type!r}")


def _compute_gmm_mahalanobis(
    latent: np.ndarray,
    *,
    n_components: int,
    covariance_type: str,
    random_state: int,
) -> Optional[np.ndarray]:
    """Minimum Mahalanobis distance to any GMM component mean (Signal 2)."""
    from sklearn.mixture import GaussianMixture

    n_samples = len(latent)
    n_components_eff = max(1, min(int(n_components), n_samples))
    try:
        gmm = GaussianMixture(
            n_components=n_components_eff,
            covariance_type=str(covariance_type),
            random_state=random_state,
            reg_covar=1e-6,
        ).fit(latent)
    except Exception:
        return None

    try:
        precisions = _gmm_component_precisions(gmm)
    except Exception:
        return None

    distances = np.full((n_samples, n_components_eff), np.inf, dtype=np.float64)
    for k in range(n_components_eff):
        diff = latent - gmm.means_[k]
        quad = np.einsum("ij,jk,ik->i", diff, precisions[k], diff)
        distances[:, k] = np.sqrt(np.clip(quad, 0.0, None))
    return distances.min(axis=1)


def _compute_isolation_forest(
    latent: np.ndarray,
    *,
    n_estimators: int,
    contamination: Union[str, float],
    random_state: int,
) -> Optional[np.ndarray]:
    """Isolation Forest anomaly score; higher = more anomalous (Signal 3)."""
    from sklearn.ensemble import IsolationForest

    try:
        model = IsolationForest(
            n_estimators=int(n_estimators),
            contamination=contamination,
            random_state=random_state,
        )
        model.fit(latent)
        return -np.asarray(model.score_samples(latent), dtype=np.float64)
    except Exception:
        return None


def _compute_local_outlier_factor(
    latent: np.ndarray,
    *,
    n_neighbors: int,
    contamination: Union[str, float],
) -> Optional[np.ndarray]:
    """Local Outlier Factor score; higher = more anomalous (Signal 3)."""
    from sklearn.neighbors import LocalOutlierFactor

    n_samples = len(latent)
    k = min(int(n_neighbors), max(1, n_samples - 1))
    if k < 1:
        return None
    try:
        model = LocalOutlierFactor(n_neighbors=k, novelty=False, contamination=contamination)
        model.fit_predict(latent)
        return -np.asarray(model.negative_outlier_factor_, dtype=np.float64)
    except Exception:
        return None


def _marginal_vendi_contribution(
    latent: np.ndarray,
    *,
    max_samples: Optional[int],
    random_state: int,
    rbf_bandwidth: Optional[float],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Leave-one-out Vendi Score contribution per (subsampled) point (Signal 5)."""
    n_samples = len(latent)
    positions = _subsample_positions(n_samples, max_samples=max_samples, random_state=random_state)
    info: Dict[str, Any] = {
        "n_samples_used": int(len(positions)),
        "max_samples": None if max_samples is None else int(max_samples),
        "subsampled": bool(len(positions) < n_samples),
    }
    if len(positions) < 3:
        info["available"] = False
        info["reason"] = "fewer than 3 samples available for leave-one-out Vendi"
        return positions, None, info

    try:
        from vendi_score import vendi
    except ImportError:
        info["available"] = False
        info["reason"] = "vendi-score package is not installed"
        return positions, None, info

    from Mapper.diversity import build_rbf_similarity_matrix

    Z_used = latent[positions]
    try:
        similarity, kernel_report = build_rbf_similarity_matrix(Z_used, bandwidth=rbf_bandwidth)
        base_vs = float(vendi.score_K(similarity, q=1))
    except Exception as exc:
        info["available"] = False
        info["reason"] = str(exc)
        return positions, None, info

    contributions = np.full(len(positions), np.nan, dtype=np.float64)
    for local_i in range(len(positions)):
        reduced = np.delete(np.delete(similarity, local_i, axis=0), local_i, axis=1)
        if reduced.shape[0] < 2:
            continue
        try:
            contributions[local_i] = base_vs - float(vendi.score_K(reduced, q=1))
        except Exception:
            continue

    info["available"] = True
    info["kernel"] = kernel_report
    info["base_vendi_score"] = base_vs
    return positions, contributions, info


def _combine_percentile_ranks(
    percentile_ranks: Dict[str, Optional[np.ndarray]],
    n_samples: int,
) -> np.ndarray:
    """Weighted mean of available ranks; NaN where no signal is available."""
    combined = np.full(n_samples, np.nan, dtype=np.float64)
    weighted_sum = np.zeros(n_samples, dtype=np.float64)
    available_weight = np.zeros(n_samples, dtype=np.float64)

    for name, weight in _SIGNAL_WEIGHTS.items():
        ranks = percentile_ranks.get(name)
        if ranks is None:
            continue
        valid = np.isfinite(ranks)
        weighted_sum[valid] += weight * ranks[valid]
        available_weight[valid] += weight

    has_any = available_weight > 0.0
    combined[has_any] = weighted_sum[has_any] / available_weight[has_any]
    return combined


def _build_reasons(
    signal_percentiles: Dict[str, Optional[float]],
    quantile: float,
) -> List[str]:
    """Plain-language reasons for one triage list entry, never an anomaly verdict."""
    reasons = [
        _SIGNAL_REASON_TEXT[name]
        for name, rank in signal_percentiles.items()
        if name in _SIGNAL_REASON_TEXT and rank is not None and rank >= quantile
    ]
    marginal_rank = signal_percentiles.get("marginal_vendi")
    density_high = any(
        (signal_percentiles.get(name) or 0.0) >= quantile for name in _DENSITY_SIGNALS
    )
    if marginal_rank is not None and marginal_rank < 0.5 and density_high:
        reasons.append(
            "low marginal Vendi contribution: consistent with a repeated/duplicated "
            "pattern rather than a distinct new regime"
        )
    if not reasons:
        reasons = ["elevated combined anomaly score across the available signals"]
    return reasons


def run_anomaly_detection(
    latent: Any,
    *,
    recon_error: Optional[Any] = None,
    n_clusters_hint: Optional[int] = None,
    top_n: int = 100,
    reason_quantile: float = 0.90,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    gmm_covariance_type: str = "full",
    isolation_forest_n_estimators: int = 200,
    isolation_forest_contamination: Union[str, float] = "auto",
    lof_n_neighbors: int = 20,
    lof_contamination: Union[str, float] = "auto",
    vendi_max_samples: Optional[int] = 200,
    rbf_bandwidth: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Score latent samples with complementary signals and rank a triage list.

    Combines reconstruction error, HDBSCAN GLOSH + GMM Mahalanobis latent
    density, Isolation Forest, Local Outlier Factor, and marginal Vendi
    contribution (guide Section 8 Signals 1-3 and 5) into one ranked triage
    list. TheMapper never issues an automatic verdict here -- it flags and
    ranks samples with reasons for a scientist to review.
    """
    latent_array = _as_2d_array(latent)
    n_samples = int(len(latent_array))
    if n_samples < 3:
        return ({"status": "insufficient_data", "n_samples": n_samples}, {})

    recon_error_arr = (
        np.asarray(recon_error, dtype=np.float64).reshape(-1)
        if recon_error is not None
        else None
    )
    if recon_error_arr is not None and recon_error_arr.shape[0] != n_samples:
        raise ValueError("recon_error length must match the number of latent rows")

    n_components_hint = (
        int(n_clusters_hint)
        if n_clusters_hint is not None and int(n_clusters_hint) >= 1
        else max(1, min(8, int(round(np.sqrt(n_samples)))))
    )

    hdbscan_glosh = _compute_hdbscan_glosh(
        latent_array,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )
    gmm_mahalanobis = _compute_gmm_mahalanobis(
        latent_array,
        n_components=n_components_hint,
        covariance_type=gmm_covariance_type,
        random_state=random_state,
    )
    isolation_forest_scores = _compute_isolation_forest(
        latent_array,
        n_estimators=isolation_forest_n_estimators,
        contamination=isolation_forest_contamination,
        random_state=random_state,
    )
    lof_scores = _compute_local_outlier_factor(
        latent_array,
        n_neighbors=lof_n_neighbors,
        contamination=lof_contamination,
    )
    vendi_positions, vendi_contribution_sub, vendi_info = _marginal_vendi_contribution(
        latent_array,
        max_samples=vendi_max_samples,
        random_state=random_state,
        rbf_bandwidth=rbf_bandwidth,
    )
    marginal_vendi_full: Optional[np.ndarray] = None
    if vendi_contribution_sub is not None:
        marginal_vendi_full = np.full(n_samples, np.nan, dtype=np.float64)
        marginal_vendi_full[vendi_positions] = vendi_contribution_sub

    raw_signals: Dict[str, Optional[np.ndarray]] = {
        "reconstruction_error": recon_error_arr,
        "hdbscan_glosh": hdbscan_glosh,
        "gmm_mahalanobis": gmm_mahalanobis,
        "isolation_forest": isolation_forest_scores,
        "local_outlier_factor": lof_scores,
        "marginal_vendi": marginal_vendi_full,
    }
    percentile_ranks = {name: _percentile_rank(values) for name, values in raw_signals.items()}
    combined_score = _combine_percentile_ranks(percentile_ranks, n_samples)

    n_ranked = int(np.sum(np.isfinite(combined_score)))
    order = np.argsort(np.where(np.isfinite(combined_score), -combined_score, np.inf))
    n_top = min(int(top_n), n_ranked)

    top_anomalies: List[Dict[str, Any]] = []
    for rank, position in enumerate(order[:n_top], start=1):
        position = int(position)
        signal_scores = {
            name: (
                None if values is None or not np.isfinite(values[position]) else float(values[position])
            )
            for name, values in raw_signals.items()
        }
        signal_percentiles = {
            name: (
                None if ranks is None or not np.isfinite(ranks[position]) else float(ranks[position])
            )
            for name, ranks in percentile_ranks.items()
        }
        top_anomalies.append(
            {
                "rank": rank,
                "sample_position": position,
                "combined_score": float(combined_score[position]),
                "signal_scores": signal_scores,
                "signal_percentiles": signal_percentiles,
                "reasons": _build_reasons(signal_percentiles, reason_quantile),
            }
        )

    report: Dict[str, Any] = {
        "status": "complete",
        "n_samples": n_samples,
        "n_ranked": n_ranked,
        "top_n_requested": int(top_n),
        "top_n_reported": len(top_anomalies),
        "reason_quantile": float(reason_quantile),
        "signals": {
            "reconstruction_error": {"available": recon_error_arr is not None},
            "hdbscan_glosh": {
                "available": hdbscan_glosh is not None,
                "min_cluster_size": int(hdbscan_min_cluster_size),
                "min_samples": None if hdbscan_min_samples is None else int(hdbscan_min_samples),
            },
            "gmm_mahalanobis": {
                "available": gmm_mahalanobis is not None,
                "n_components": int(n_components_hint),
                "covariance_type": str(gmm_covariance_type),
            },
            "isolation_forest": {
                "available": isolation_forest_scores is not None,
                "n_estimators": int(isolation_forest_n_estimators),
                "contamination": isolation_forest_contamination,
            },
            "local_outlier_factor": {
                "available": lof_scores is not None,
                "n_neighbors": int(lof_n_neighbors),
                "contamination": lof_contamination,
            },
            "marginal_vendi": vendi_info,
        },
        "combined": {
            "method": "weighted_mean_of_available_percentile_ranks",
            "weights": dict(_SIGNAL_WEIGHTS),
        },
        "top_anomalies": top_anomalies,
        "warning": (
            "TheMapper ranks and flags these samples for human review. It does not "
            "automatically label any sample as corrupted, anomalous, or new physics "
            "-- a scientist must adjudicate."
        ),
    }

    def _full_or_nan(values: Optional[np.ndarray]) -> np.ndarray:
        return values if values is not None else np.full(n_samples, np.nan, dtype=np.float64)

    arrays: Dict[str, np.ndarray] = {
        "combined_score": combined_score,
        "hdbscan_glosh": _full_or_nan(hdbscan_glosh),
        "gmm_mahalanobis": _full_or_nan(gmm_mahalanobis),
        "isolation_forest": _full_or_nan(isolation_forest_scores),
        "local_outlier_factor": _full_or_nan(lof_scores),
        "marginal_vendi": _full_or_nan(marginal_vendi_full),
    }
    if recon_error_arr is not None:
        arrays["reconstruction_error"] = recon_error_arr

    return report, arrays


__all__ = ["run_anomaly_detection"]
