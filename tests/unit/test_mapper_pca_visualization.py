"""Tests for Mapper's dynamic-component PCA visualization."""

from pathlib import Path

import numpy as np

from Mapper.viz import plot_mapper_pca


def test_mapper_pca_plot_uses_only_first_two_scores(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(30, 5))

    result = plot_mapper_pca(
        scores,
        output_dir=tmp_path,
        explained_variance_ratio=np.array([0.5, 0.25, 0.15, 0.06, 0.04]),
    )

    assert result["status"] == "complete"
    assert result["n_components"] == 5
    assert result["plotted_components"] == 2
    assert Path(result["saved_paths"][0]).is_file()
