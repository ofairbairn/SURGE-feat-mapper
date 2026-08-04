"""Tests for Vendi V1 diversity on Mapper latent embeddings."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("vendi_score")

from Mapper.diversity import (  # noqa: E402
    build_rbf_similarity_matrix,
    compute_vendi_diversity,
    plot_vendi_q_profile,
)


def test_identical_embeddings_have_one_effective_mode() -> None:
    embeddings = np.zeros((12, 3))

    result = compute_vendi_diversity(embeddings, max_samples=None)

    assert result.report["vendi_score"] == pytest.approx(1.0)
    assert result.report["effective_modes"] == pytest.approx(1.0)
    assert all(
        point["score"] == pytest.approx(1.0)
        for point in result.report["q_profile"]
    )
    np.testing.assert_allclose(result.similarity_matrix, np.ones((12, 12)))


def test_nearly_identity_kernel_has_one_mode_per_embedding() -> None:
    embeddings = np.eye(5)

    result = compute_vendi_diversity(
        embeddings,
        rbf_bandwidth=1e-3,
        q_values=[0, 0.5, 1, 2, "inf"],
        max_samples=None,
    )

    assert result.report["vendi_score"] == pytest.approx(5.0, rel=1e-6)
    assert [point["q"] for point in result.report["q_profile"]] == [
        0.0,
        0.5,
        1.0,
        2.0,
        "inf",
    ]
    assert all(
        point["score"] == pytest.approx(5.0, rel=1e-6)
        for point in result.report["q_profile"]
    )


def test_q_profile_supports_orders_above_one_and_reproducible_sampling(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(50, 4))

    first = compute_vendi_diversity(
        embeddings,
        q_values=[0.1, 0.5, 1, 2, 5, "inf"],
        max_samples=20,
        random_state=9,
    )
    second = compute_vendi_diversity(
        embeddings,
        q_values=[0.1, 0.5, 1, 2, 5, "inf"],
        max_samples=20,
        random_state=9,
    )

    np.testing.assert_array_equal(first.sample_positions, second.sample_positions)
    scores = [point["score"] for point in first.report["q_profile"]]
    assert all(left >= right for left, right in zip(scores, scores[1:]))
    assert first.report["embedding"]["subsampled"] is True
    plot_path = plot_vendi_q_profile(first, tmp_path / "q_profile.png")
    assert plot_path.is_file() and plot_path.stat().st_size > 0


def test_rbf_kernel_and_q_inputs_are_validated() -> None:
    embeddings = np.arange(12, dtype=float).reshape(6, 2)

    kernel, metadata = build_rbf_similarity_matrix(embeddings)

    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(np.diag(kernel), np.ones(6))
    assert metadata["bandwidth_selection"] == "median_nonzero_pairwise_distance"
    with pytest.raises(ValueError, match="q >= 0"):
        compute_vendi_diversity(embeddings, q_values=[-0.1])
