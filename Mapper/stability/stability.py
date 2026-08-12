"""Cluster stability analysis for Mapper.

This module evaluates the cluster structure produced by the Mapper clustering
stage using two complementary checks:

1. Bootstrap Jaccard similarity against the reference partition.
2. Consensus clustering from repeated bootstrap partitions.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


_STABILITY_JACCARD_STRONG_THRESHOLD = 0.85
_STABILITY_JACCARD_MODERATE_THRESHOLD = 0.60


def _as_2d_array(latent: Any) -> np.ndarray:
	arr = np.asarray(latent, dtype=np.float64)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	if arr.ndim != 2:
		raise ValueError(f"Expected a 2D latent array, got shape={arr.shape}")
	return arr


def _cluster_sizes(labels: np.ndarray) -> Dict[str, int]:
	labels = np.asarray(labels)
	return {
		str(int(label)): int(np.sum(labels == label))
		for label in np.unique(labels)
	}


def _pairwise_jaccard(reference_labels: np.ndarray, trial_labels: np.ndarray) -> Optional[float]:
	"""Jaccard similarity between two co-membership relations."""
	reference = np.asarray(reference_labels)
	trial = np.asarray(trial_labels)
	if reference.shape != trial.shape or reference.ndim != 1 or len(reference) < 2:
		return None

	same_reference = np.equal.outer(reference, reference)
	same_trial = np.equal.outer(trial, trial)
	np.fill_diagonal(same_reference, False)
	np.fill_diagonal(same_trial, False)

	intersection = int(np.logical_and(same_reference, same_trial).sum())
	union = int(np.logical_or(same_reference, same_trial).sum())
	if union == 0:
		return 1.0
	return float(intersection / union)


def _cluster_best_jaccard(
	reference_labels: np.ndarray,
	trial_labels: np.ndarray,
) -> Dict[str, Any]:
	"""Best-matching cluster Jaccard for each reference cluster."""
	reference = np.asarray(reference_labels)
	trial = np.asarray(trial_labels)
	scores: Dict[str, float] = {}
	for cluster_id in np.unique(reference):
		reference_mask = reference == cluster_id
		if int(np.sum(reference_mask)) == 0:
			continue
		best_score = 0.0
		for trial_cluster in np.unique(trial):
			trial_mask = trial == trial_cluster
			union = int(np.logical_or(reference_mask, trial_mask).sum())
			if union == 0:
				continue
			intersection = int(np.logical_and(reference_mask, trial_mask).sum())
			score = float(intersection / union)
			if score > best_score:
				best_score = score
		scores[str(int(cluster_id))] = best_score

	return {
		"per_cluster": scores,
		"mean_best_jaccard": float(np.mean(list(scores.values()))) if scores else None,
		"min_best_jaccard": float(np.min(list(scores.values()))) if scores else None,
	}


def _fit_kmeans_labels(
	latent: np.ndarray,
	*,
	n_clusters: int,
	random_state: int,
) -> np.ndarray:
	from sklearn.cluster import KMeans

	model = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
	return model.fit_predict(latent)


def _fit_kmeans_model(
	latent: np.ndarray,
	*,
	n_clusters: int,
	random_state: int,
) -> Any:
	from sklearn.cluster import KMeans

	model = KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init="auto")
	model.fit(latent)
	return model


def _fit_gmm_labels(
	latent: np.ndarray,
	*,
	n_clusters: int,
	random_state: int,
	covariance_type: str,
) -> np.ndarray:
	from sklearn.mixture import GaussianMixture

	model = GaussianMixture(
		n_components=int(n_clusters),
		covariance_type=str(covariance_type),
		random_state=random_state,
		reg_covar=1e-6,
	)
	return model.fit(latent).predict(latent)


def _fit_gmm_model(
	latent: np.ndarray,
	*,
	n_clusters: int,
	random_state: int,
	covariance_type: str,
) -> Any:
	from sklearn.mixture import GaussianMixture

	model = GaussianMixture(
		n_components=int(n_clusters),
		covariance_type=str(covariance_type),
		random_state=random_state,
		reg_covar=1e-6,
	)
	model.fit(latent)
	return model


def _fit_hdbscan_model(
	latent: np.ndarray,
	*,
	min_cluster_size: int,
	min_samples: Optional[int],
) -> Any:
	import hdbscan

	model = hdbscan.HDBSCAN(
		min_cluster_size=int(max(2, min_cluster_size)),
		min_samples=None if min_samples is None else int(min_samples),
		prediction_data=True,
	)
	model.fit(latent)
	return model


def _predict_hdbscan_labels(model: Any, latent: np.ndarray) -> np.ndarray:
	import hdbscan

	labels, _strengths = hdbscan.approximate_predict(model, latent)
	return np.asarray(labels, dtype=np.int64)


def _hdbscan_is_all_noise(labels: np.ndarray) -> bool:
	"""True when an HDBSCAN fit assigned every point to noise (no defined clusters)."""
	labels = np.asarray(labels)
	return bool(labels.size > 0 and np.all(labels == -1))


def _bootstrap_positions(
	n_samples: int,
	*,
	bootstrap_fraction: float,
	random_state: int,
) -> np.ndarray:
	if n_samples < 2:
		raise ValueError("Stability analysis requires at least two latent rows")
	fraction = float(bootstrap_fraction)
	if not np.isfinite(fraction) or fraction <= 0.0:
		raise ValueError("bootstrap_fraction must be positive")
	sample_size = max(2, int(round(n_samples * min(fraction, 1.0))))
	sample_size = min(sample_size, n_samples)
	rng = np.random.default_rng(random_state)
	return rng.choice(n_samples, size=sample_size, replace=True).astype(np.int64)


def _coassociation_matrix(label_sets: list[np.ndarray]) -> np.ndarray:
	if not label_sets:
		raise ValueError("At least one label set is required for consensus clustering")

	n_samples = len(label_sets[0])
	matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
	for labels in label_sets:
		same = np.equal.outer(labels, labels).astype(np.float64)
		np.fill_diagonal(same, 0.0)
		matrix += same
	matrix /= float(len(label_sets))
	return matrix


def _consensus_labels(
	coassociation: np.ndarray,
	*,
	n_clusters: int,
) -> np.ndarray:
	from scipy.cluster.hierarchy import fcluster, linkage
	from scipy.spatial.distance import squareform

	coassociation = np.asarray(coassociation, dtype=np.float64)
	if coassociation.ndim != 2 or coassociation.shape[0] != coassociation.shape[1]:
		raise ValueError("Consensus clustering requires a square co-association matrix")

	n_samples = coassociation.shape[0]
	if n_samples == 0:
		return np.empty(0, dtype=np.int64)
	if n_samples == 1 or int(n_clusters) <= 1:
		return np.zeros(n_samples, dtype=np.int64)

	distance = 1.0 - np.clip(coassociation, 0.0, 1.0)
	np.fill_diagonal(distance, 0.0)
	condensed = squareform(distance, checks=False)
	linkage_matrix = linkage(condensed, method="average")
	labels = fcluster(linkage_matrix, t=int(n_clusters), criterion="maxclust")
	return labels.astype(np.int64) - 1


def _summary_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
	if not values:
		return {"mean": None, "std": None, "min": None, "max": None}
	arr = np.asarray(values, dtype=np.float64)
	return {
		"mean": float(np.mean(arr)),
		"std": float(np.std(arr)),
		"min": float(np.min(arr)),
		"max": float(np.max(arr)),
	}


def _stability_jaccard_message(score: Optional[float]) -> Tuple[str, str]:
	"""Map a bootstrap Jaccard score onto a qualitative stability label."""
	if score is None or not np.isfinite(score):
		return "unavailable", "index indicates cluster stability is unavailable"
	if score > _STABILITY_JACCARD_STRONG_THRESHOLD:
		return "strong", "index indicates strong cluster stability"
	if score >= _STABILITY_JACCARD_MODERATE_THRESHOLD:
		return "moderate", "index indicates moderate cluster stability"
	return "poor", "index indicates poor cluster stability"


def run_cluster_stability(
	latent: Any,
	*,
	selected_k: int,
	baseline_labels: Optional[Dict[str, Any]] = None,
	n_bootstraps: int = 25,
	bootstrap_fraction: float = 0.8,
	random_state: int = 42,
	max_samples: Optional[int] = 1000,
	gmm_covariance_type: str = "full",
	hdbscan_min_cluster_size: int = 20,
	hdbscan_min_samples: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
	"""Assess the stability of the selected Mapper clustering solution."""
	latent_array = _as_2d_array(latent)
	n_samples = int(len(latent_array))
	if n_samples < 2:
		return (
			{"status": "insufficient_data", "n_samples": n_samples},
			{},
		)

	stability_max_samples = None if max_samples is None else max(2, int(max_samples))
	if stability_max_samples is None or stability_max_samples >= n_samples:
		sample_positions = np.arange(n_samples, dtype=np.int64)
	else:
		rng = np.random.default_rng(random_state)
		sample_positions = np.sort(
			rng.choice(n_samples, size=stability_max_samples, replace=False)
		).astype(np.int64)

	latent_used = latent_array[sample_positions]
	n_used = int(len(latent_used))
	if n_used < 2:
		return (
			{"status": "insufficient_data", "n_samples": n_samples, "n_samples_used": n_used},
			{"sample_positions": sample_positions},
		)

	effective_k = int(max(1, min(int(selected_k), n_used)))
	bootstrap_count = max(1, int(n_bootstraps))
	bootstrap_fraction = float(bootstrap_fraction)

	baseline_kmeans = _fit_kmeans_labels(
		latent_used,
		n_clusters=effective_k,
		random_state=random_state,
	)
	baseline_gmm = _fit_gmm_labels(
		latent_used,
		n_clusters=effective_k,
		random_state=random_state,
		covariance_type=gmm_covariance_type,
	)

	baseline_hdbscan: Optional[np.ndarray] = None
	hdbscan_baseline_status = "unavailable"
	if baseline_labels is not None and "hdbscan" in baseline_labels:
		provided = np.asarray(baseline_labels["hdbscan"], dtype=np.int64)
		if provided.shape[0] == n_samples:
			baseline_hdbscan = provided[sample_positions]
		elif provided.shape[0] == n_used:
			baseline_hdbscan = provided
	if baseline_hdbscan is None:
		try:
			hdbscan_model = _fit_hdbscan_model(
				latent_used,
				min_cluster_size=hdbscan_min_cluster_size,
				min_samples=hdbscan_min_samples,
			)
			baseline_hdbscan = np.asarray(hdbscan_model.labels_, dtype=np.int64)
		except Exception:
			baseline_hdbscan = None
	if baseline_hdbscan is not None and _hdbscan_is_all_noise(baseline_hdbscan):
		hdbscan_baseline_status = "degenerate_all_noise"
		baseline_hdbscan = None
	elif baseline_hdbscan is not None:
		hdbscan_baseline_status = "complete"

	bootstrap_jaccard_scores: Dict[str, list[float]] = {"kmeans": [], "gmm": [], "hdbscan": []}
	bootstrap_cluster_support: Dict[str, Dict[str, list[float]]] = {
		"kmeans": {},
		"gmm": {},
		"hdbscan": {},
	}
	consensus_inputs: list[np.ndarray] = []
	bootstrap_methods_used = {"kmeans": 0, "gmm": 0, "hdbscan": 0}
	reduced_bootstraps = 0
	hdbscan_degenerate_bootstraps = 0

	rng = np.random.default_rng(random_state)
	for _ in range(bootstrap_count):
		success = False
		for _attempt in range(5):
			positions = _bootstrap_positions(
				n_used,
				bootstrap_fraction=bootstrap_fraction,
				random_state=int(rng.integers(0, 2**31 - 1)),
			)
			unique_positions = np.unique(positions)
			boot_k = max(1, min(effective_k, int(len(unique_positions))))
			if boot_k < effective_k:
				reduced_bootstraps += 1

			bootstrap_latent = latent_used[positions]
			try:
				boot_kmeans_model = _fit_kmeans_model(
					bootstrap_latent,
					n_clusters=boot_k,
					random_state=int(rng.integers(0, 2**31 - 1)),
				)
				boot_kmeans_full = boot_kmeans_model.predict(latent_used)
			except Exception:
				boot_kmeans_model = None
				boot_kmeans_full = None

			try:
				boot_gmm_model = _fit_gmm_model(
					bootstrap_latent,
					n_clusters=boot_k,
					random_state=int(rng.integers(0, 2**31 - 1)),
					covariance_type=gmm_covariance_type,
				)
				boot_gmm_full = boot_gmm_model.predict(latent_used)
			except Exception:
				boot_gmm_model = None
				boot_gmm_full = None

			if baseline_hdbscan is not None:
				try:
					boot_hdbscan_model = _fit_hdbscan_model(
						bootstrap_latent,
						min_cluster_size=hdbscan_min_cluster_size,
						min_samples=hdbscan_min_samples,
					)
					if _hdbscan_is_all_noise(boot_hdbscan_model.labels_):
						# no defined clusters to project onto; skip approximate_predict entirely
						hdbscan_degenerate_bootstraps += 1
						boot_hdbscan_full = None
					else:
						boot_hdbscan_full = _predict_hdbscan_labels(boot_hdbscan_model, latent_used)
				except Exception:
					boot_hdbscan_model = None
					boot_hdbscan_full = None
			else:
				boot_hdbscan_full = None

			if boot_kmeans_full is None and boot_gmm_full is None and boot_hdbscan_full is None:
				continue

			success = True
			if boot_kmeans_full is not None:
				bootstrap_methods_used["kmeans"] += 1
				score = _pairwise_jaccard(baseline_kmeans, boot_kmeans_full)
				if score is not None:
					bootstrap_jaccard_scores["kmeans"].append(score)
				support = _cluster_best_jaccard(baseline_kmeans, boot_kmeans_full)
				for cluster_id, cluster_score in support["per_cluster"].items():
					bootstrap_cluster_support["kmeans"].setdefault(cluster_id, []).append(cluster_score)
				consensus_inputs.append(boot_kmeans_full)

			if boot_gmm_full is not None:
				bootstrap_methods_used["gmm"] += 1
				score = _pairwise_jaccard(baseline_gmm, boot_gmm_full)
				if score is not None:
					bootstrap_jaccard_scores["gmm"].append(score)
				support = _cluster_best_jaccard(baseline_gmm, boot_gmm_full)
				for cluster_id, cluster_score in support["per_cluster"].items():
					bootstrap_cluster_support["gmm"].setdefault(cluster_id, []).append(cluster_score)
				consensus_inputs.append(boot_gmm_full)

			if boot_hdbscan_full is not None and baseline_hdbscan is not None:
				bootstrap_methods_used["hdbscan"] += 1
				score = _pairwise_jaccard(baseline_hdbscan, boot_hdbscan_full)
				if score is not None:
					bootstrap_jaccard_scores["hdbscan"].append(score)
				support = _cluster_best_jaccard(baseline_hdbscan, boot_hdbscan_full)
				for cluster_id, cluster_score in support["per_cluster"].items():
					bootstrap_cluster_support["hdbscan"].setdefault(cluster_id, []).append(cluster_score)
				consensus_inputs.append(boot_hdbscan_full)

			break
		if not success:
			continue

	if not consensus_inputs:
		raise RuntimeError("Bootstrap stability analysis could not produce any partitions")

	coassociation = _coassociation_matrix(consensus_inputs)
	consensus_labels = _consensus_labels(coassociation, n_clusters=effective_k)

	reference_agreement = {
		"kmeans_vs_gmm": {
			"ari": None,
			"ami": None,
			"jaccard": _pairwise_jaccard(baseline_kmeans, baseline_gmm),
		}
	}
	if baseline_hdbscan is not None:
		reference_agreement["hdbscan_vs_kmeans"] = {
			"ari": None,
			"ami": None,
			"jaccard": _pairwise_jaccard(baseline_hdbscan, baseline_kmeans),
		}
		reference_agreement["hdbscan_vs_gmm"] = {
			"ari": None,
			"ami": None,
			"jaccard": _pairwise_jaccard(baseline_hdbscan, baseline_gmm),
		}
	try:
		from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

		reference_agreement["kmeans_vs_gmm"]["ari"] = float(
			adjusted_rand_score(baseline_kmeans, baseline_gmm)
		)
		reference_agreement["kmeans_vs_gmm"]["ami"] = float(
			adjusted_mutual_info_score(baseline_kmeans, baseline_gmm)
		)
		if baseline_hdbscan is not None:
			reference_agreement["hdbscan_vs_kmeans"]["ari"] = float(
				adjusted_rand_score(baseline_hdbscan, baseline_kmeans)
			)
			reference_agreement["hdbscan_vs_kmeans"]["ami"] = float(
				adjusted_mutual_info_score(baseline_hdbscan, baseline_kmeans)
			)
			reference_agreement["hdbscan_vs_gmm"]["ari"] = float(
				adjusted_rand_score(baseline_hdbscan, baseline_gmm)
			)
			reference_agreement["hdbscan_vs_gmm"]["ami"] = float(
				adjusted_mutual_info_score(baseline_hdbscan, baseline_gmm)
			)
	except Exception:
		pass

	kmeans_bootstrap_stats = _summary_stats(bootstrap_jaccard_scores["kmeans"])
	gmm_bootstrap_stats = _summary_stats(bootstrap_jaccard_scores["gmm"])
	hdbscan_bootstrap_stats = _summary_stats(bootstrap_jaccard_scores["hdbscan"])
	bootstrap_index_candidates = [
		value
		for value in (
			kmeans_bootstrap_stats["mean"],
			gmm_bootstrap_stats["mean"],
			hdbscan_bootstrap_stats["mean"],
		)
		if value is not None
	]
	bootstrap_index = (
		float(np.mean(bootstrap_index_candidates))
		if bootstrap_index_candidates
		else None
	)
	bootstrap_index_level, bootstrap_index_message = _stability_jaccard_message(bootstrap_index)

	consensus_agreement = {}
	try:
		from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

		consensus_inputs_by_method = [("kmeans", baseline_kmeans), ("gmm", baseline_gmm)]
		if baseline_hdbscan is not None:
			consensus_inputs_by_method.append(("hdbscan", baseline_hdbscan))
		for name, baseline in consensus_inputs_by_method:
			consensus_agreement[name] = {
				"ari": float(adjusted_rand_score(baseline, consensus_labels)),
				"ami": float(adjusted_mutual_info_score(baseline, consensus_labels)),
				"jaccard": _pairwise_jaccard(baseline, consensus_labels),
			}
	except Exception:
		consensus_agreement = {
			"kmeans": {"ari": None, "ami": None, "jaccard": _pairwise_jaccard(baseline_kmeans, consensus_labels)},
			"gmm": {"ari": None, "ami": None, "jaccard": _pairwise_jaccard(baseline_gmm, consensus_labels)},
		}
		if baseline_hdbscan is not None:
			consensus_agreement["hdbscan"] = {
				"ari": None,
				"ami": None,
				"jaccard": _pairwise_jaccard(baseline_hdbscan, consensus_labels),
			}

	report: Dict[str, Any] = {
		"status": "complete",
		"method": "bootstrap_jaccard_consensus",
		"n_samples": n_samples,
		"n_samples_used": n_used,
		"subsampled": bool(n_used < n_samples),
		"max_samples": None if stability_max_samples is None else int(stability_max_samples),
		"selected_k": int(selected_k),
		"effective_k": int(effective_k),
		"bootstrap_fraction": float(bootstrap_fraction),
		"n_bootstraps": int(bootstrap_count),
		"bootstrap_methods_used": bootstrap_methods_used,
		"reduced_bootstraps": int(reduced_bootstraps),
		"hdbscan_degenerate_bootstraps": int(hdbscan_degenerate_bootstraps),
		"reference_models": {
			"kmeans": {"cluster_sizes": _cluster_sizes(baseline_kmeans)},
			"gmm": {"cluster_sizes": _cluster_sizes(baseline_gmm)},
			"hdbscan": (
				{"cluster_sizes": _cluster_sizes(baseline_hdbscan), "status": hdbscan_baseline_status}
				if baseline_hdbscan is not None
				else {"status": hdbscan_baseline_status}
			),
			"reference_agreement": reference_agreement,
		},
		"bootstrap_jaccard": {
			"kmeans": {
				**_summary_stats(bootstrap_jaccard_scores["kmeans"]),
				"cluster_support": {
					cluster_id: _summary_stats(scores)
					for cluster_id, scores in bootstrap_cluster_support["kmeans"].items()
				},
			},
			"gmm": {
				**_summary_stats(bootstrap_jaccard_scores["gmm"]),
				"cluster_support": {
					cluster_id: _summary_stats(scores)
					for cluster_id, scores in bootstrap_cluster_support["gmm"].items()
				},
			},
			"hdbscan": {
				**_summary_stats(bootstrap_jaccard_scores["hdbscan"]),
				"cluster_support": {
					cluster_id: _summary_stats(scores)
					for cluster_id, scores in bootstrap_cluster_support["hdbscan"].items()
				},
			},
			"bootstrap_jaccard_index": bootstrap_index,
			"bootstrap_jaccard_index_level": bootstrap_index_level,
			"bootstrap_jaccard_index_message": bootstrap_index_message,
		},
		"consensus_clustering": {
			"cluster_sizes": _cluster_sizes(consensus_labels),
			"agreement": consensus_agreement,
			"coassociation_mean": float(np.mean(coassociation)),
			"coassociation_std": float(np.std(coassociation)),
		},
	}

	arrays = {
		"sample_positions": sample_positions.astype(np.int64),
		"baseline_kmeans_labels": baseline_kmeans.astype(np.int64),
		"baseline_gmm_labels": baseline_gmm.astype(np.int64),
		"consensus_labels": consensus_labels.astype(np.int64),
		"coassociation_matrix": coassociation.astype(np.float64),
	}
	if baseline_hdbscan is not None:
		arrays["baseline_hdbscan_labels"] = baseline_hdbscan.astype(np.int64)
	return report, arrays


__all__ = ["run_cluster_stability"]

