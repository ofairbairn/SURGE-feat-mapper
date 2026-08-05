"""
Cluster module for Mapper.
By Default runs HDBSCAN clustering algorithm.
After, it verifies using k-means clustering and
gaussian mixed model clustering. Choosing k val
done using silhouette score, gap statistic,
calinski harabasz, davies bouldin, bayesian information
criterion, and vendi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _nearest_neighbor_preservation(
    X_ref: np.ndarray,
    X_embed: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> Optional[float]:
    from sklearn.neighbors import NearestNeighbors

    n = min(len(X_ref), len(X_embed))
    if n < 3:
        return None
    k = max(2, min(n_neighbors + 1, n))

    nn_ref = NearestNeighbors(n_neighbors=k).fit(X_ref)
    nn_emb = NearestNeighbors(n_neighbors=k).fit(X_embed)
    ref_neighbors = nn_ref.kneighbors(return_distance=False)[:, 1:]
    emb_neighbors = nn_emb.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for ref_row, emb_row in zip(ref_neighbors, emb_neighbors):
        inter = len(set(ref_row.tolist()).intersection(set(emb_row.tolist())))
        overlaps.append(inter / max(1, len(ref_row)))
    return float(np.mean(overlaps))


def _compute_gap_statistic(
    X: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 10,
    n_references: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Tibshirani-style gap statistic for choosing the number of clusters.

    Compares log within-cluster dispersion against a uniform reference
    distribution; the recommended ``optimal_k`` is the smallest k within one
    standard error of the following gap (Tibshirani, Walther & Hastie 2001).
    """
    from sklearn.cluster import KMeans

    empty = {
        "optimal_k": None,
        "k_min": int(k_min),
        "k_max": int(k_max),
        "gap_values": [],
        "gap_sds": [],
    }

    X = np.asarray(X, dtype=np.float64)
    n_samples = len(X)
    if n_samples < 4:
        return empty
    k_max = max(k_min, min(int(k_max), n_samples - 1))
    if k_max < k_min:
        return empty

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs))):
        return empty

    def _log_within_sum_of_squares(data: np.ndarray, k: int) -> float:
        clusterer = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = clusterer.fit_predict(data)
        within = 0.0
        for cluster_id in range(k):
            points = data[labels == cluster_id]
            if len(points) == 0:
                continue
            center = points.mean(axis=0)
            within += float(np.sum((points - center) ** 2))
        return float(np.log(within)) if within > 0.0 else 0.0

    rng = np.random.default_rng(random_state)
    gaps: List[float] = []
    gap_sds: List[float] = []
    for k in range(k_min, k_max + 1):
        observed = _log_within_sum_of_squares(X, k)
        reference_logs = []
        for _ in range(n_references):
            reference = rng.uniform(mins, maxs, size=X.shape)
            reference_logs.append(_log_within_sum_of_squares(reference, k))
        gap = float(np.mean(reference_logs)) - observed
        sd = float(np.std(reference_logs)) * np.sqrt(1.0 + 1.0 / n_references)
        gaps.append(gap)
        gap_sds.append(sd)

    optimal_k = k_min
    for index in range(len(gaps) - 1):
        if gaps[index] >= gaps[index + 1] - gap_sds[index + 1]:
            optimal_k = k_min + index
            break
    else:
        optimal_k = k_max

    return {
        "optimal_k": int(optimal_k),
        "k_min": int(k_min),
        "k_max": int(k_max),
        "gap_values": [float(value) for value in gaps],
        "gap_sds": [float(value) for value in gap_sds],
    }


def _gaussian_partition_bic(
    X: np.ndarray,
    labels: np.ndarray,
) -> Optional[float]:
    """BIC of a Gaussian mixture implied by a hard partition.

    Each cluster is treated as a Gaussian with its own mean and (regularized)
    full covariance; the penalty is the number of free parameters times
    log(n_samples)/2 (Pelleg & Moore style). Noise points (label -1) are
    excluded from the mixture.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features == 0:
        return None

    unique_labels = np.unique(labels)
    cluster_labels = [int(label) for label in unique_labels if int(label) != -1]
    n_clusters = len(cluster_labels)
    if n_clusters < 2:
        return None

    log_likelihood = 0.0
    for cluster_id in cluster_labels:
        points = X[labels == cluster_id]
        n_c = len(points)
        if n_c <= n_features:
            return None
        mean = points.mean(axis=0)
        centered = points - mean
        covariance = np.cov(points, rowvar=False) + np.eye(n_features) * 1e-6
        try:
            sign, log_det = np.linalg.slogdet(covariance)
        except np.linalg.LinAlgError:
            return None
        if sign <= 0:
            return None
        inverse = np.linalg.pinv(covariance)
        quadratic = np.sum((centered @ inverse) * centered, axis=1)
        log_likelihood += -0.5 * n_c * (
            n_features * np.log(2.0 * np.pi) + log_det
        ) - 0.5 * float(np.sum(quadratic))

    n_parameters = (
        n_clusters * n_features
        + n_clusters * n_features * (n_features + 1) // 2
    )
    return float(log_likelihood - 0.5 * n_parameters * np.log(n_samples))


def _compute_vendi_per_cluster(
    latent: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples: Optional[int] = 200,
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:
    """Vendi Score V1 within each non-noise cluster (effective latent modes).

    Uses the same median-bandwidth RBF kernel as the global diversity module.
    Returns ``None`` when the vendi-score package is unavailable or no cluster
    is large enough to score.
    """
    try:
        from vendi_score import vendi
    except ImportError:
        return None
    from Mapper.diversity.vendi_q_diversity import build_rbf_similarity_matrix

    latent = np.asarray(latent, dtype=np.float64)
    labels = np.asarray(labels)
    scores: Dict[str, float] = {}
    for cluster_id in np.unique(labels):
        if int(cluster_id) == -1:
            continue
        points = latent[labels == cluster_id]
        if len(points) < 2:
            continue
        if max_samples is not None and len(points) > max_samples:
            rng = np.random.default_rng(random_state)
            positions = rng.choice(len(points), size=max_samples, replace=False)
            points = points[positions]
        try:
            similarity, _kernel = build_rbf_similarity_matrix(points)
            scores[str(int(cluster_id))] = float(vendi.score_K(similarity, q=1))
        except Exception:
            continue

    if not scores:
        return None
    return {
        "per_cluster": scores,
        "mean_vendi": float(np.mean(list(scores.values()))),
        "max_samples": None if max_samples is None else int(max_samples),
    }


def _compute_latent_quality_metrics(
    latent: np.ndarray,
    embedding: np.ndarray,
    clusters: Optional[np.ndarray],
    *,
    n_neighbors: int = 10,
    gap_k_max: int = 8,
    gap_n_references: int = 5,
    vendi_max_samples: Optional[int] = 200,
    random_state: int = 42,
) -> Dict[str, Any]:
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    metrics: Dict[str, Any] = {}
    n = min(len(latent), len(embedding))
    if n < 3:
        metrics["trustworthiness"] = None
        metrics["nearest_neighbor_preservation"] = None
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
        metrics["gap_statistic"] = {
            "optimal_k": None,
            "k_min": 2,
            "k_max": int(gap_k_max),
            "gap_values": [],
            "gap_sds": [],
        }
        metrics["bic"] = None
        metrics["vendi_per_cluster"] = None
        metrics["n_clusters"] = 0
        metrics["noise_fraction"] = None
        return metrics

    k = min(max(2, n_neighbors), n - 1)
    try:
        metrics["trustworthiness"] = float(trustworthiness(latent, embedding, n_neighbors=k))
    except Exception:
        metrics["trustworthiness"] = None

    try:
        metrics["nearest_neighbor_preservation"] = _nearest_neighbor_preservation(
            latent,
            embedding,
            n_neighbors=k,
        )
    except Exception:
        metrics["nearest_neighbor_preservation"] = None

    if clusters is None:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
        metrics["gap_statistic"] = {
            "optimal_k": None,
            "k_min": 2,
            "k_max": int(gap_k_max),
            "gap_values": [],
            "gap_sds": [],
        }
        metrics["bic"] = None
        metrics["vendi_per_cluster"] = None
        metrics["n_clusters"] = 0
        metrics["noise_fraction"] = None
        return metrics

    labels = np.asarray(clusters)
    non_noise_mask = labels != -1
    noise_fraction = 1.0 - float(np.mean(non_noise_mask))
    valid_labels = labels[non_noise_mask]
    valid_latent = np.asarray(latent)[non_noise_mask]
    valid_embedding = embedding[non_noise_mask]
    unique_labels = np.unique(valid_labels)
    metrics["n_clusters"] = int(len(unique_labels))
    metrics["noise_fraction"] = noise_fraction
    if len(unique_labels) < 2 or len(valid_embedding) <= len(unique_labels):
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["calinski_harabasz"] = None
        metrics["gap_statistic"] = {
            "optimal_k": None,
            "k_min": 2,
            "k_max": int(gap_k_max),
            "gap_values": [],
            "gap_sds": [],
        }
        metrics["bic"] = None
        metrics["vendi_per_cluster"] = None
        return metrics

    try:
        metrics["silhouette"] = float(silhouette_score(valid_embedding, valid_labels))
    except Exception:
        metrics["silhouette"] = None
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(valid_embedding, valid_labels))
    except Exception:
        metrics["davies_bouldin"] = None
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(valid_embedding, valid_labels))
    except Exception:
        metrics["calinski_harabasz"] = None

    metrics["gap_statistic"] = _compute_gap_statistic(
        valid_latent,
        k_max=gap_k_max,
        n_references=gap_n_references,
        random_state=random_state,
    )
    metrics["bic"] = _gaussian_partition_bic(valid_latent, valid_labels)
    metrics["vendi_per_cluster"] = _compute_vendi_per_cluster(
        valid_latent,
        valid_labels,
        max_samples=vendi_max_samples,
        random_state=random_state,
    )
    return metrics


def _cluster_latent_embeddings(
    latent: np.ndarray,
    *,
    method: str = "hdbscan",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    agglomerative_linkage: str = "ward",
    gmm_covariance_type: str = "full",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    latent = np.asarray(latent, dtype=np.float64)
    n_samples = latent.shape[0]
    if n_samples < 2 or method.lower() in {"", "none", "off", "no"}:
        return np.full(n_samples, -1, dtype=int), None

    method_normalized = method.lower().strip()
    if method_normalized == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters_eff = int(n_clusters or min(8, max(2, int(np.sqrt(n_samples)))))
        n_clusters_eff = max(2, min(n_clusters_eff, n_samples))
        clusterer = KMeans(n_clusters=n_clusters_eff, random_state=random_state, n_init="auto")
        return clusterer.fit_predict(latent), None

    if method_normalized == "dbscan":
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        return clusterer.fit_predict(latent), None

    if method_normalized == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("hdbscan is required for cluster_method='hdbscan'") from exc

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )
        return clusterer.fit_predict(latent), None

    if method_normalized in {"gmm", "gaussian_mixture", "gm"}:
        from sklearn.mixture import GaussianMixture

        n_components_eff = int(n_clusters or min(8, max(2, int(np.sqrt(n_samples)))))
        n_components_eff = max(2, min(n_components_eff, n_samples))
        clusterer = GaussianMixture(
            n_components=n_components_eff,
            covariance_type=gmm_covariance_type,
            random_state=random_state,
        )
        return clusterer.fit_predict(latent), None

    if method_normalized == "agglomerative":
        from scipy.cluster.hierarchy import linkage
        from sklearn.cluster import AgglomerativeClustering

        n_clusters_eff = int(n_clusters or min(8, max(2, int(np.sqrt(n_samples)))))
        n_clusters_eff = max(2, min(n_clusters_eff, n_samples))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters_eff, linkage=agglomerative_linkage)
        labels = clusterer.fit_predict(latent)
        linkage_matrix = linkage(latent, method=agglomerative_linkage)
        return labels, linkage_matrix

    raise ValueError(
        f"Unsupported cluster_method={method!r}. "
        "Expected one of: none, kmeans, dbscan, hdbscan, gmm, agglomerative."
    )
