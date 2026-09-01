"""Tests for Mapper UMAP color variants."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Mapper.viz import map_viz


def test_save_embedding_renders_four_umap_color_variants(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    plot_calls: list[dict[str, Any]] = []

    def record_plot(_frame: pd.DataFrame, **kwargs: Any) -> None:
        plot_calls.append(kwargs)

    monkeypatch.setattr(map_viz, "_plot_latent_matplotlib", record_plot)
    monkeypatch.setattr(
        map_viz,
        "_plot_latent_interactive",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        map_viz,
        "_compute_latent_quality_metrics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda *_args, **_kwargs: None)

    anomaly = np.array([0.1, 0.4, 0.7, 0.9])
    marginal_vendi = np.array([0.02, 0.01, 0.08, 0.04])
    saved = map_viz._save_embedding(
        np.arange(8, dtype=float).reshape(4, 2),
        np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [1.5, 1.5]]),
        sample_indices=np.arange(4),
        label_values=None,
        cluster_labels=np.array([0, 0, 1, -1]),
        recon_error=None,
        anomaly_score=anomaly,
        data_split=np.array(["train", "validation", "test", "train"]),
        marginal_vendi=marginal_vendi,
        model_name="mapper_pca",
        split="train_val_test",
        embedding_type="umap",
        title="Mapper PCA latent UMAP",
        output_dir=tmp_path,
        color_by="cluster",
        random_state=42,
        anomaly_quantile=0.95,
        interactive_threshold=50_000,
        interactive_hover_sample_frac=0.02,
        cluster_method="hdbscan",
        tendency_summary={"gate_passed": True},
        latent_quality_n_neighbors=2,
    )

    png_names = [Path(path).name for path in saved if path.endswith(".png")]
    assert png_names == [
        "latent_mapper_pca_train_val_test_umap_anomaly.png",
        "latent_mapper_pca_train_val_test_umap_cluster.png",
        "latent_mapper_pca_train_val_test_umap_split.png",
        "latent_mapper_pca_train_val_test_umap_marginal_vendi.png",
    ]
    assert len(plot_calls) == 4
    np.testing.assert_array_equal(plot_calls[0]["color_scores"], anomaly)
    assert plot_calls[1]["color_by"] == "cluster"
    assert plot_calls[2]["category_colors"] == {
        "train": "#1f77b4",
        "validation": "#d62728",
        "test": "#ffd700",
    }
    np.testing.assert_array_equal(plot_calls[3]["color_scores"], marginal_vendi)
