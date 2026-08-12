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
    k_min: int = 1,
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
    compute_gap: bool = True,
    compute_vendi: bool = True,
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
        metrics["gap_statistic"] = _empty_gap_statistic(gap_k_max) if compute_gap else None
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
        metrics["gap_statistic"] = _empty_gap_statistic(gap_k_max) if compute_gap else None
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
        metrics["gap_statistic"] = _empty_gap_statistic(gap_k_max) if compute_gap else None
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

    metrics["gap_statistic"] = (
        _compute_gap_statistic(
            valid_latent,
            k_max=gap_k_max,
            n_references=gap_n_references,
            random_state=random_state,
        )
        if compute_gap
        else None
    )
    metrics["bic"] = _gaussian_partition_bic(valid_latent, valid_labels)
    metrics["vendi_per_cluster"] = (
        _compute_vendi_per_cluster(
            valid_latent,
            valid_labels,
            max_samples=vendi_max_samples,
            random_state=random_state,
        )
        if compute_vendi
        else None
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
        n_clusters_eff = max(1, min(n_clusters_eff, n_samples))
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
        n_components_eff = max(1, min(n_components_eff, n_samples))
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
        n_clusters_eff = max(1, min(n_clusters_eff, n_samples))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters_eff, linkage=agglomerative_linkage)
        labels = clusterer.fit_predict(latent)
        linkage_matrix = linkage(latent, method=agglomerative_linkage)
        return labels, linkage_matrix

    raise ValueError(
        f"Unsupported cluster_method={method!r}. "
        "Expected one of: none, kmeans, dbscan, hdbscan, gmm, agglomerative."
    )


def _empty_gap_statistic(k_max: int = 8) -> Dict[str, Any]:
    """Empty gap-statistic payload used when the gap statistic is not computed."""
    return {
        "optimal_k": None,
        "k_min": 1,
        "k_max": int(k_max),
        "gap_values": [],
        "gap_sds": [],
    }


def _partition_scores(
    latent: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compact partition quality scores (silhouette, CH, DB, BIC)."""
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    latent = np.asarray(latent, dtype=np.float64)
    labels = np.asarray(labels)
    mask = labels != -1
    valid_labels = labels[mask]
    valid_latent = latent[mask]
    unique_labels = np.unique(valid_labels)
    scores: Dict[str, Optional[float]] = {
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
        "bic": None,
    }
    if len(unique_labels) < 2 or len(valid_latent) <= len(unique_labels):
        return scores
    for name, func in (
        ("silhouette", silhouette_score),
        ("calinski_harabasz", calinski_harabasz_score),
        ("davies_bouldin", davies_bouldin_score),
    ):
        try:
            scores[name] = float(func(valid_latent, valid_labels))
        except Exception:
            scores[name] = None
    scores["bic"] = _gaussian_partition_bic(valid_latent, valid_labels) #bic voting computed partition labels
    return scores


def _gmm_bic(
    latent: np.ndarray,
    n_components: int,
    *,
    covariance_type: str = "full",
    random_state: int = 42,
) -> Optional[float]:
    """True Gaussian-mixture BIC (sklearn soft-assignment likelihood).

    Lower is better. Complements ``_gaussian_partition_bic`` (a hard-assignment
    approximation); ``None`` when the fit fails or is degenerate.
    """
    from sklearn.mixture import GaussianMixture

    latent = np.asarray(latent, dtype=np.float64)
    n_samples = len(latent)
    if n_samples < 2 or n_components < 1 or n_components > n_samples:
        return None
    try:
        mixture = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        ).fit(latent)
        return float(mixture.bic(latent))
    except Exception:
        return None


def _label_agreement(
    a: np.ndarray,
    b: np.ndarray,
) -> Optional[Dict[str, Optional[float]]]:
    """Adjusted Rand Index and mutual information between two labelings."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape or len(a) == 0:
        return None
    try:
        ari = float(adjusted_rand_score(a, b))
    except Exception:
        ari = None
    try:
        ami = float(adjusted_mutual_info_score(a, b))
    except Exception:
        ami = None
    return {"ari": ari, "ami": ami}


def _compute_global_vendi(
    latent: np.ndarray,
    *,
    max_samples: Optional[int] = 200,
    random_state: int = 42,
) -> Optional[float]:
    """Vendi Score V1 on the whole latent (reference for cluster ratios)."""
    try:
        from vendi_score import vendi
    except ImportError:
        return None
    from Mapper.diversity.vendi_q_diversity import build_rbf_similarity_matrix

    latent = np.asarray(latent, dtype=np.float64)
    if len(latent) < 2:
        return None
    if max_samples is not None and len(latent) > max_samples:
        rng = np.random.default_rng(random_state)
        positions = rng.choice(len(latent), size=max_samples, replace=False)
        latent = latent[positions]
    try:
        similarity, _kernel = build_rbf_similarity_matrix(latent)
        return float(vendi.score_K(similarity, q=1))
    except Exception:
        return None


def _compute_vendi_between_clusters(
    latent: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
) -> Optional[Dict[str, Any]]:
    """Vendi Score V1 on cluster centroids (how distinct the clusters are)."""
    try:
        from vendi_score import vendi
    except ImportError:
        return None
    from Mapper.diversity.vendi_q_diversity import build_rbf_similarity_matrix

    latent = np.asarray(latent, dtype=np.float64)
    labels = np.asarray(labels)
    centroids: List[np.ndarray] = []
    cluster_ids: List[int] = []
    for cluster_id in np.unique(labels):
        if int(cluster_id) == -1:
            continue
        points = latent[labels == cluster_id]
        if len(points) == 0:
            continue
        centroids.append(points.mean(axis=0))
        cluster_ids.append(int(cluster_id))
    if len(centroids) < 2:
        return None
    try:
        similarity, _kernel = build_rbf_similarity_matrix(
            np.asarray(centroids, dtype=np.float64)
        )
        vendi_score = float(vendi.score_K(similarity, q=1))
    except Exception:
        return None
    return {
        "vendi_score": vendi_score,
        "n_clusters": len(centroids),
        "cluster_ids": [str(cluster_id) for cluster_id in cluster_ids],
    }


def run_cluster_analysis(
    latent: np.ndarray,
    *,
    embedding: Optional[np.ndarray] = None,
    k_anchor: Optional[int] = None,
    k_max: int = 10,
    k_window: int = 1,
    gap_k_max: int = 8,
    gap_n_references: int = 5,
    vendi_max_samples: Optional[int] = 200,
    random_state: int = 42,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    gmm_covariance_type: str = "full",
) -> Dict[str, Any]:
    """Cluster a latent embedding with an HDBSCAN-anchored consensus workflow.

    1. HDBSCAN always runs first (density-based, no k required).
    2. The HDBSCAN cluster count anchors k for k-means and GMM; both methods
       are swept across nearby candidate k values, partition metrics are scored
       for each method, true GMM BIC adds a GMM-specific vote, and the gap
       statistic provides an extra method-agnostic k hint.
    3. k-means and GMM are fit at the chosen k and cross-checked with ARI/AMI
       against HDBSCAN.
    4. Within/between-cluster Vendi validate compactness and separation
       relative to the global Vendi score.
    """
    latent = np.asarray(latent, dtype=np.float64)
    n_samples = len(latent)
    if latent.ndim != 2 or n_samples < 2 or latent.shape[1] == 0:
        return {"status": "insufficient_data", "n_samples": int(n_samples)}

    metric_space = (
        np.asarray(embedding, dtype=np.float64)
        if embedding is not None and len(embedding) == n_samples
        else latent
    )

    report: Dict[str, Any] = {
        "status": "complete",
        "method": "hdbscan_anchored_consensus",
        "n_samples": int(n_samples),
        "latent_dim": int(latent.shape[1]),
    }

    hdbscan_labels, _ = _cluster_latent_embeddings(
        latent,
        method="hdbscan",
        random_state=random_state,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )
    hdbscan_quality = _compute_latent_quality_metrics(
        latent,
        metric_space,
        hdbscan_labels,
        gap_k_max=gap_k_max,
        gap_n_references=gap_n_references,
        vendi_max_samples=vendi_max_samples,
        random_state=random_state,
    )
    k_hdbscan = int(len(np.unique(hdbscan_labels[hdbscan_labels != -1])))
    hdbscan_degenerate_all_noise = bool(k_hdbscan == 0)
    report["hdbscan"] = {
        "n_clusters": k_hdbscan,
        "n_noise": int(np.sum(hdbscan_labels == -1)),
        "noise_fraction": float(np.mean(hdbscan_labels == -1)),
        "status": "degenerate_all_noise" if hdbscan_degenerate_all_noise else "complete",
        "quality": hdbscan_quality,
    }

    k_upper = max(1, min(int(k_max), n_samples - 1))
    anchor = (
        int(k_anchor)
        if k_anchor is not None
        else (k_hdbscan if k_hdbscan >= 1 else 1)
    )
    anchor = max(1, min(anchor, k_upper))
    window = max(0, int(k_window))
    candidates = sorted(
        {
            max(1, anchor - window),
            anchor,
            min(k_upper, anchor + window),
        }
    )

    metrics_by_k: Dict[str, Any] = {}
    for candidate in candidates:
        kmeans_labels_candidate, _ = _cluster_latent_embeddings(
            latent,
            method="kmeans",
            n_clusters=candidate,
            random_state=random_state,
        )
        gmm_labels_candidate, _ = _cluster_latent_embeddings(
            latent,
            method="gmm",
            n_clusters=candidate,
            random_state=random_state,
            gmm_covariance_type=gmm_covariance_type,
        )
        metrics_by_k[str(candidate)] = {
            "kmeans": _partition_scores(latent, kmeans_labels_candidate),
            "gmm": {
                **_partition_scores(latent, gmm_labels_candidate),
                "gmm_bic": _gmm_bic(
                    latent,
                    candidate,
                    covariance_type=gmm_covariance_type,
                    random_state=random_state,
                ),
            },
        }

    vote_metrics = {
        ("kmeans", "silhouette"): "max",
        ("kmeans", "calinski_harabasz"): "max",
        ("kmeans", "davies_bouldin"): "min",
        ("kmeans", "bic"): "max",
        ("gmm", "silhouette"): "max",
        ("gmm", "calinski_harabasz"): "max",
        ("gmm", "davies_bouldin"): "min",
        ("gmm", "bic"): "max",
        ("gmm", "gmm_bic"): "min",
    }
    votes: Dict[int, int] = {}
    for (method_name, metric), sense in vote_metrics.items():
        scored = {
            candidate: metrics_by_k[str(candidate)][method_name][metric]
            for candidate in candidates
            if metrics_by_k[str(candidate)][method_name].get(metric) is not None
        }
        if not scored:
            continue
        best = (
            max(scored, key=scored.get)
            if sense == "max"
            else min(scored, key=scored.get)
        )
        votes[best] = votes.get(best, 0) + 1

    gap_report = _compute_gap_statistic(
        latent,
        k_min=1,
        k_max=k_upper,
        n_references=gap_n_references,
        random_state=random_state,
    )
    gap_optimal = gap_report["optimal_k"]
    if gap_optimal is not None and int(gap_optimal) in candidates:
        votes[int(gap_optimal)] = votes.get(int(gap_optimal), 0) + 1

    selected_k = anchor
    selection_reason = "hdbscan_anchor"
    if votes:
        selected_k = max(votes, key=lambda k: (votes[k], k == anchor))
        if selected_k != anchor:
            selection_reason = "metric_majority"

    report["k_selection"] = {
        "anchor": int(anchor),
        "anchor_source": (
            "provided"
            if k_anchor is not None
            else "hdbscan_degenerate_all_noise" if hdbscan_degenerate_all_noise else "hdbscan"
        ),
        "anchor_reliable": bool(k_hdbscan >= 2),
        "k_hdbscan": k_hdbscan,
        "candidates": [int(candidate) for candidate in candidates],
        "metrics_by_k": metrics_by_k,
        "gap_statistic": gap_report,
        "votes": {str(k): v for k, v in votes.items()},
        "selected_k": int(selected_k),
        "selection_reason": selection_reason,
    }

    kmeans_labels, _ = _cluster_latent_embeddings(
        latent,
        method="kmeans",
        n_clusters=selected_k,
        random_state=random_state,
    )
    gmm_labels, _ = _cluster_latent_embeddings(
        latent,
        method="gmm",
        n_clusters=selected_k,
        random_state=random_state,
        gmm_covariance_type=gmm_covariance_type,
    )
    report["kmeans"] = {
        "n_clusters": int(selected_k),
        "quality": _compute_latent_quality_metrics(
            latent,
            metric_space,
            kmeans_labels,
            gap_k_max=gap_k_max,
            gap_n_references=gap_n_references,
            vendi_max_samples=vendi_max_samples,
            random_state=random_state,
        ),
    }
    report["gmm"] = {
        "n_clusters": int(selected_k),
        "quality": _compute_latent_quality_metrics(
            latent,
            metric_space,
            gmm_labels,
            gap_k_max=gap_k_max,
            gap_n_references=gap_n_references,
            vendi_max_samples=vendi_max_samples,
            random_state=random_state,
        ),
    }

    report["agreement"] = {
        "hdbscan_vs_kmeans": _label_agreement(hdbscan_labels, kmeans_labels),
        "hdbscan_vs_gmm": _label_agreement(hdbscan_labels, gmm_labels),
        "kmeans_vs_gmm": _label_agreement(kmeans_labels, gmm_labels),
    }

    global_vendi = _compute_global_vendi(
        latent,
        max_samples=vendi_max_samples,
        random_state=random_state,
    )
    within: Dict[str, Optional[Dict[str, Any]]] = {
        name: _compute_vendi_per_cluster(
            latent,
            labels,
            max_samples=vendi_max_samples,
            random_state=random_state,
        )
        for name, labels in (
            ("hdbscan", hdbscan_labels),
            ("kmeans", kmeans_labels),
            ("gmm", gmm_labels),
        )
    }
    between: Dict[str, Optional[Dict[str, Any]]] = {
        name: _compute_vendi_between_clusters(
            latent,
            labels,
            random_state=random_state,
        )
        for name, labels in (
            ("hdbscan", hdbscan_labels),
            ("kmeans", kmeans_labels),
            ("gmm", gmm_labels),
        )
    }
    ratios: Dict[str, Any] = {}
    for name in ("hdbscan", "kmeans", "gmm"):
        per_cluster = (within.get(name) or {}).get("per_cluster") or {}
        mean_within = (
            float(np.mean(list(per_cluster.values()))) if per_cluster else None
        )
        between_score = (between.get(name) or {}).get("vendi_score")
        ratios[name] = {
            "mean_within_vendi": mean_within,
            "between_vendi": between_score,
            "within_over_global": (
                mean_within / global_vendi
                if mean_within is not None and global_vendi
                else None
            ),
            "between_over_global": (
                between_score / global_vendi
                if between_score is not None and global_vendi
                else None
            ),
        }
    report["vendi"] = {
        "global_vendi": global_vendi,
        "within_per_cluster": within,
        "between_clusters": between,
        "ratios": ratios,
    }

    # Persist method label vectors so downstream stability can evaluate
    # reproducibility on the primary clustering output (HDBSCAN) as well as
    # k-means and GMM confirmation partitions.
    report["labels"] = {
        "hdbscan": np.asarray(hdbscan_labels, dtype=int).tolist(),
        "kmeans": np.asarray(kmeans_labels, dtype=int).tolist(),
        "gmm": np.asarray(gmm_labels, dtype=int).tolist(),
    }

    return report
