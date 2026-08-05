"""Tests for the Mapper consensus cluster analysis module."""

import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

from Mapper.cluster import run_cluster_analysis


def _structured_latent() -> np.ndarray:
    latent, _ = make_blobs(
        n_samples=240,
        centers=3,
        n_features=6,
        cluster_std=1.1,
        random_state=3,
    )
    return latent.astype(np.float64)


def test_run_cluster_analysis_runs_hdbscan_then_kmeans_and_gmm() -> None:
    latent = _structured_latent()
    report = run_cluster_analysis(
        latent,
        gap_k_max=5,
        gap_n_references=2,
        hdbscan_min_cluster_size=15,
    )

    assert report["status"] == "complete"
    assert report["method"] == "hdbscan_anchored_consensus"
    assert report["hdbscan"]["n_clusters"] >= 2
    assert "quality" in report["hdbscan"]
    selection = report["k_selection"]
    assert selection["anchor_source"] == "hdbscan"
    assert selection["selected_k"] >= 2
    assert report["kmeans"]["n_clusters"] == selection["selected_k"]
    assert report["gmm"]["n_clusters"] == selection["selected_k"]
    assert set(report["agreement"]) == {
        "hdbscan_vs_kmeans",
        "hdbscan_vs_gmm",
        "kmeans_vs_gmm",
    }
    assert set(report["vendi"]["ratios"]) == {"hdbscan", "kmeans", "gmm"}
    json.dumps(report)


def test_run_cluster_analysis_accepts_explicit_k_anchor() -> None:
    latent = _structured_latent()
    report = run_cluster_analysis(
        latent,
        k_anchor=3,
        gap_k_max=4,
        gap_n_references=1,
        hdbscan_min_cluster_size=15,
    )

    assert report["k_selection"]["anchor"] == 3
    assert report["k_selection"]["anchor_source"] == "provided"
    assert report["k_selection"]["selected_k"] >= 2
    assert report["kmeans"]["n_clusters"] == report["k_selection"]["selected_k"]


def test_run_cluster_analysis_with_embedding_space() -> None:
    latent = _structured_latent()
    embedding = PCA(n_components=2, random_state=0).fit_transform(latent)
    report = run_cluster_analysis(
        latent,
        embedding=embedding,
        gap_k_max=4,
        gap_n_references=1,
        hdbscan_min_cluster_size=15,
    )

    assert report["status"] == "complete"
    assert report["kmeans"]["quality"]["silhouette"] is not None


def test_run_cluster_analysis_insufficient_data() -> None:
    report = run_cluster_analysis(np.ones((1, 4), dtype=np.float64))
    assert report["status"] == "insufficient_data"

    report = run_cluster_analysis(np.ones((3, 4), dtype=np.float64))
    assert report["status"] == "complete"


def test_run_cluster_analysis_degenerate_constant_latent() -> None:
    latent = np.full((30, 4), 2.0, dtype=np.float64)
    report = run_cluster_analysis(
        latent, gap_k_max=3, gap_n_references=1, hdbscan_min_cluster_size=5
    )

    assert report["status"] == "complete"
    assert report["k_selection"]["selected_k"] >= 2
