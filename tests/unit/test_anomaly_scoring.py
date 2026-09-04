"""Tests for weighted Mapper anomaly scoring."""

import numpy as np

from Mapper.anomaly.anomaly import _combine_percentile_ranks


def test_combined_score_uses_requested_signal_weights() -> None:
    ranks = {
        "reconstruction_error": np.array([1.0]),
        "hdbscan_glosh": np.array([0.2]),
        "gmm_mahalanobis": np.array([0.8]),
        "isolation_forest": np.array([0.4]),
        "local_outlier_factor": np.array([0.6]),
        "marginal_vendi": np.array([0.0]),
    }

    combined = _combine_percentile_ranks(ranks, n_samples=1)

    expected = (1.0 + 0.5 * 0.2 + 0.5 * 0.8 + 0.4 + 0.6 + 0.0) / 5.0
    np.testing.assert_allclose(combined, [expected])


def test_combined_score_renormalizes_over_available_signals_per_sample() -> None:
    ranks = {
        "reconstruction_error": np.array([1.0, np.nan]),
        "hdbscan_glosh": None,
        "gmm_mahalanobis": np.array([0.0, 0.2]),
        "isolation_forest": np.array([np.nan, 0.8]),
        "local_outlier_factor": None,
        "marginal_vendi": None,
    }

    combined = _combine_percentile_ranks(ranks, n_samples=2)

    np.testing.assert_allclose(
        combined,
        [(1.0 * 1.0 + 0.5 * 0.0) / 1.5, (0.5 * 0.2 + 1.0 * 0.8) / 1.5],
    )
