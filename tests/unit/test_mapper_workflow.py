"""Tests for Mapper workflow selection and preprocessing orchestration."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from Mapper.preprocess import DataScaler, ImageDataScaler
from surge.workflow.run import run_workflow
from surge.workflow.spec import SurrogateWorkflowSpec


def test_workflow_type_defaults_to_surrogate_and_validates() -> None:
    default_spec = SurrogateWorkflowSpec(dataset_path="unused.csv")
    mapper_spec = SurrogateWorkflowSpec(
        dataset_path="unused.csv", workflow_type=" MAPPER "
    )

    assert default_spec.workflow_type == "surrogate"
    assert mapper_spec.workflow_type == "mapper"
    with pytest.raises(ValueError, match="workflow_type"):
        SurrogateWorkflowSpec(dataset_path="unused.csv", workflow_type="unknown")
    with pytest.raises(ValueError, match="data_type"):
        SurrogateWorkflowSpec(dataset_path="unused.csv", data_type="audio")


def test_mapper_workflow_robustly_scales_from_training_split(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        {
            "input_a": np.r_[rng.normal(size=39), 1_000.0],
            "input_b": np.r_[rng.normal(size=39), -2_000.0],
        }
    )
    dataset_path = tmp_path / "mapper.csv"
    frame.to_csv(dataset_path, index=False)
    spec = SurrogateWorkflowSpec(
        dataset_path=dataset_path,
        workflow_type="mapper",
        task_type="unsupervised",
        test_fraction=0.2,
        val_fraction=0.2,
        seed=17,
        output_dir=tmp_path,
        run_tag="mapper_unit",
    )

    summary = run_workflow(spec)
    root = Path(summary["artifacts"]["root"])

    assert summary["workflow_type"] == "mapper"
    assert "vat_parents" not in summary["cluster_tendency"]
    gate_passed = summary["cluster_tendency"]["gate_passed"]
    if gate_passed:
        assert summary["status"] == "clustering_complete"
        assert summary["clustering"]["status"] == "complete"
        assert "labels" not in summary["clustering"]
        assert (root / "clustering" / "cluster_analysis.json").is_file()
    else:
        assert summary["status"] == "stopped_no_cluster_tendency"
        assert summary["clustering"] is None
    assert summary["clustering_decision"]["proceed_to_clustering"] is gate_passed
    assert summary["next_stage"] is None
    assert summary["representation_ladder"]["selected_rung"] == "pca"
    assert summary["representation_ladder"]["quality_sufficient"] is True
    assert len(summary["representation_ladder"]["rungs_run"]) == 1
    assert summary["diversity"]["selected_representation"] == "pca"
    assert summary["diversity"]["vendi_score"] >= 1.0
    scaler = joblib.load(summary["artifacts"]["scaler"])
    assert isinstance(scaler, DataScaler)
    with np.load(summary["artifacts"]["scaled_splits"]) as splits:
        train_index = splits["train_index"]
        raw_train = frame.loc[train_index, summary["input_columns"]].to_numpy()
        np.testing.assert_allclose(scaler.center_, np.median(raw_train, axis=0))
        np.testing.assert_allclose(
            scaler.inverse_transform(splits["X_train"]), raw_train
        )
        np.testing.assert_allclose(
            np.median(splits["X_train"], axis=0), np.zeros(2), atol=1e-12
        )

    assert (root / "spec.yaml").is_file()
    assert (root / "metrics.json").is_file()
    assert (root / "mapper_latent_splits.npz").is_file()
    assert (root / "diversity" / "vendi_diversity.json").is_file()
    assert (root / "diversity" / "vendi_q_profile.png").is_file()
    assert (root / "diversity" / "vendi_similarity_matrix.npz").is_file()
    assert (root / "tendency" / "cluster_tendency.json").is_file()
    assert (root / "tendency" / "vat_ivat_matrices.npz").is_file()
    assert (root / "tendency" / "vat_heatmap.png").is_file()
    assert (root / "tendency" / "ivat_heatmap.png").is_file()
    assert (root / "workflow_summary.json").is_file()
    saved_summary = json.loads(
        (root / "workflow_summary.json").read_text(encoding="utf-8")
    )
    assert "vat_parents" not in saved_summary["cluster_tendency"]
    if saved_summary["clustering"] is not None:
        assert "labels" not in saved_summary["clustering"]


@pytest.mark.parametrize("gate_passed", [True, False])
def test_cluster_tendency_gate_controls_pipeline_decision(
    gate_passed: bool,
) -> None:
    from Mapper.pipeline import (
        _mapper_status_after_tendency,
        decide_clustering_action,
    )

    decision = decide_clustering_action(
        {
            "gate_passed": gate_passed,
            "gate_reason": "pass" if gate_passed else "hopkins_below_threshold",
        }
    )

    assert decision["gate_passed"] is gate_passed
    assert decision["proceed_to_clustering"] is gate_passed
    assert decision["action"] == (
        "proceed_to_clustering"
        if gate_passed
        else "stop_no_cluster_tendency"
    )
    assert decision["next_stage"] == ("clustering" if gate_passed else None)
    assert decision["stop_reason"] == (
        None if gate_passed else "hopkins_below_threshold"
    )
    assert _mapper_status_after_tendency(
        decision, diversity_complete=True
    ) == ("clustering_ready" if gate_passed else "stopped_no_cluster_tendency")


def test_cluster_tendency_gate_halts_when_hopkins_undefined() -> None:
    from Mapper.pipeline import (
        _mapper_status_after_tendency,
        decide_clustering_action,
    )

    decision = decide_clustering_action(
        {
            "gate_passed": False,
            "gate_reason": "hopkins_undefined",
        }
    )

    assert decision["gate_passed"] is False
    assert decision["proceed_to_clustering"] is False
    assert decision["action"] == "stop_hopkins_undefined"
    assert decision["next_stage"] is None
    assert decision["stop_reason"] == "hopkins_undefined"
    assert decision["message"] == (
        "hopkins statistic undefined: check for zero distances "
        "or identical coordinates"
    )
    assert _mapper_status_after_tendency(
        decision, diversity_complete=True
    ) == "stopped_hopkins_undefined"


def test_cluster_tendency_undefined_on_degenerate_latent() -> None:
    from Mapper.tendency import _summarize_cluster_tendency

    degenerate = np.full((12, 3), 5.0, dtype=np.float64)
    summary = _summarize_cluster_tendency(degenerate)

    assert summary["hopkins"] is None
    assert summary["gate_passed"] is False
    assert summary["gate_reason"] == "hopkins_undefined"


class _FakeLadderAdapter:
    backend = "test"

    def __init__(self, rung: str, rmse: float) -> None:
        self.rung = rung
        self.rmse = rmse
        self.training_history: list[dict[str, float]] = []

    def prepare_for_fit(self, **kwargs: Any) -> None:
        return None

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> None:
        return None

    def mark_fitted(self) -> None:
        return None

    def reconstruction_metrics(self, X: Any) -> dict[str, float]:
        return {
            "mse": self.rmse**2,
            "rmse": self.rmse,
            "mae": self.rmse * 0.8,
            "latent_var_mean": 1.0,
        }

    def encode(self, X: Any) -> np.ndarray:
        return np.asarray(X)[:, :1]

    def save(self, path: Any) -> None:
        Path(path).write_text(self.rung, encoding="utf-8")


def test_mapper_image_layout_routes_ae_and_vae_to_convolutional_adapters() -> None:
    from Mapper.pipeline import _resolve_ladder_model

    spec = SurrogateWorkflowSpec(
        dataset_path="unused.csv",
        workflow_type="mapper",
        data_type="image",
        input_shape=(1, 28, 28),
    )
    layout = {"data_type": "image", "input_shape": [1, 28, 28]}

    ae_key, _, ae_params = _resolve_ladder_model(
        spec, "ae", input_layout=layout
    )
    vae_key, _, vae_params = _resolve_ladder_model(
        spec, "vae", input_layout=layout
    )

    assert ae_key == "pytorch.conv_autoencoder"
    assert vae_key == "pytorch.conv_unsupervised_vae"
    assert ae_params["input_shape"] == (1, 28, 28)
    assert vae_params["input_shape"] == (1, 28, 28)


def test_mapper_auto_detects_contiguous_square_pixel_columns() -> None:
    from Mapper.pipeline import _resolve_mapper_input_layout
    from surge.dataset import SurrogateDataset

    columns = [f"pixel_{index}" for index in range(16)]
    frame = pd.DataFrame(np.zeros((2, 16)), columns=columns)
    dataset = SurrogateDataset.from_dataframe(
        frame,
        input_columns=columns,
        output_columns=columns,
    )
    # Reproduce the lexical ordering produced by the general tabular analyzer.
    dataset.input_columns = sorted(dataset.input_columns)
    spec = SurrogateWorkflowSpec(
        dataset_path="unused.csv",
        workflow_type="mapper",
    )

    layout = _resolve_mapper_input_layout(dataset, spec)

    assert layout["data_type"] == "image"
    assert layout["input_shape"] == [1, 4, 4]
    assert layout["source"] == "contiguous_pixel_columns"
    assert layout["ordered_input_columns"] == columns


def test_mapper_reconstruction_payload_flattens_cnn_output() -> None:
    from Mapper.pipeline import _compute_reconstruction_payload

    class ImageAdapter:
        def reconstruct(self, X: Any) -> np.ndarray:
            return np.asarray(X).reshape(-1, 1, 2, 2)

    X_train = np.arange(8, dtype=np.float32).reshape(2, 4)
    X_val = np.arange(4, dtype=np.float32).reshape(1, 4)
    payload = _compute_reconstruction_payload(
        ImageAdapter(),
        X_train,
        X_val,
        None,
        np.array([1, 0]),
        np.array([2]),
        None,
    )

    assert payload is not None
    assert payload["y_pred"].shape == (3, 4)
    np.testing.assert_allclose(payload["recon_error"], np.zeros(3))


def test_mapper_image_workflow_uses_image_scaling_and_numeric_pixel_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Mapper import pipeline

    rng = np.random.default_rng(29)
    pixels = rng.integers(0, 256, size=(30, 16), dtype=np.int32)
    frame = pd.DataFrame(pixels, columns=[f"pixel_{index}" for index in range(16)])
    frame["label"] = rng.integers(0, 4, size=len(frame))
    dataset_path = tmp_path / "image_mapper.csv"
    frame.to_csv(dataset_path, index=False)
    errors = {"pca": 0.20, "ae": 0.05}
    rung_by_key = {
        "sklearn.pca": "pca",
        "pytorch.conv_autoencoder": "ae",
        "pytorch.conv_unsupervised_vae": "vae",
    }
    calls: list[tuple[str, dict[str, Any]]] = []

    def create_adapter(model_key: str, **kwargs: Any) -> _FakeLadderAdapter:
        rung = rung_by_key[model_key]
        calls.append((model_key, kwargs))
        return _FakeLadderAdapter(rung, errors.get(rung, 0.05))

    monkeypatch.setattr(pipeline.MODEL_REGISTRY, "create", create_adapter)
    spec = SurrogateWorkflowSpec(
        dataset_path=dataset_path,
        workflow_type="mapper",
        data_type="image",
        input_shape=(1, 4, 4),
        test_fraction=0.2,
        val_fraction=0.2,
        output_dir=tmp_path,
        run_tag="image_mapper",
        unsupervised_ladder_thresholds={"max_recon_rmse": 0.10},
        mapper_preprocess={"enabled": False},
        mapper_diversity={"enabled": False},
        mapper_tendency={"enabled": False},
        mapper_anomaly={"enabled": False},
        mapper_visualization={"enabled": False},
    )

    summary = pipeline.run_mapper_workflow(spec)

    assert [key for key, _ in calls] == [
        "sklearn.pca",
        "pytorch.conv_autoencoder",
    ]
    assert calls[1][1]["input_shape"] == (1, 4, 4)
    assert summary["input_layout"]["data_type"] == "image"
    assert summary["input_layout"]["input_shape"] == [1, 4, 4]
    assert summary["input_columns"] == [f"pixel_{index}" for index in range(16)]
    assert "label" not in summary["input_columns"]
    assert summary["scaling"]["method"] == "divide_255"
    scaler = joblib.load(summary["artifacts"]["scaler"])
    assert isinstance(scaler, ImageDataScaler)


@pytest.mark.parametrize(
    ("errors", "expected_rungs", "selected", "sufficient", "exhausted"),
    [
        ({"pca": 0.05}, ["pca"], "pca", True, False),
        ({"pca": 0.20, "ae": 0.05}, ["pca", "ae"], "ae", True, False),
        (
            {"pca": 0.20, "ae": 0.15, "vae": 0.12},
            ["pca", "ae", "vae"],
            "vae",
            False,
            True,
        ),
    ],
)
def test_mapper_ladder_stops_or_climbs_at_each_quality_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    errors: dict[str, float],
    expected_rungs: list[str],
    selected: str,
    sufficient: bool,
    exhausted: bool,
) -> None:
    from Mapper import pipeline

    rng = np.random.default_rng(23)
    frame = pd.DataFrame(
        rng.normal(size=(30, 4)),
        columns=[f"input_{index}" for index in range(4)],
    )
    dataset_path = tmp_path / f"ladder_{selected}_{exhausted}.csv"
    frame.to_csv(dataset_path, index=False)
    rung_by_key = {
        "sklearn.pca": "pca",
        "pytorch.autoencoder": "ae",
        "pytorch.unsupervised_vae": "vae",
    }
    calls: list[str] = []

    def create_adapter(model_key: str, **kwargs: Any) -> _FakeLadderAdapter:
        rung = rung_by_key[model_key]
        calls.append(rung)
        return _FakeLadderAdapter(rung, errors[rung])

    monkeypatch.setattr(pipeline.MODEL_REGISTRY, "create", create_adapter)
    spec = SurrogateWorkflowSpec(
        dataset_path=dataset_path,
        workflow_type="mapper",
        test_fraction=0.2,
        val_fraction=0.2,
        output_dir=tmp_path,
        run_tag=f"ladder_{selected}_{exhausted}",
        unsupervised_ladder_thresholds={"max_recon_rmse": 0.10},
        mapper_diversity={"enabled": False},
        mapper_tendency={"enabled": False},
    )

    summary = pipeline.run_mapper_workflow(spec)
    ladder = summary["representation_ladder"]

    assert calls == expected_rungs
    assert [result["rung"] for result in ladder["rungs_run"]] == expected_rungs
    assert ladder["selected_rung"] == selected
    assert ladder["quality_sufficient"] is sufficient
    assert ladder["exhausted"] is exhausted
    assert Path(summary["artifacts"]["latent_splits"]).is_file()


def test_mapper_ladder_runs_real_pca_ae_and_vae_backends(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        rng.normal(size=(36, 10)),
        columns=[f"input_{index}" for index in range(10)],
    )
    dataset_path = tmp_path / "real_ladder.csv"
    frame.to_csv(dataset_path, index=False)
    spec = SurrogateWorkflowSpec(
        dataset_path=dataset_path,
        workflow_type="mapper",
        test_fraction=0.2,
        val_fraction=0.2,
        output_dir=tmp_path,
        run_tag="real_ladder",
        unsupervised_ladder_thresholds={"max_recon_rmse": 0.0},
        mapper_diversity={"enabled": False},
        mapper_tendency={"enabled": False},
        models=[
            {"key": "sklearn.pca", "params": {"n_components": 2}},
            {
                "key": "pytorch.autoencoder",
                "params": {
                    "latent_dim": 2,
                    "hidden_dims": [8],
                    "n_epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                },
            },
            {
                "key": "pytorch.unsupervised_vae",
                "params": {
                    "latent_dim": 2,
                    "hidden_dims": [8],
                    "n_epochs": 1,
                    "batch_size": 8,
                    "device": "cpu",
                },
            },
        ],
    )

    summary = run_workflow(spec)
    ladder = summary["representation_ladder"]

    assert [rung["rung"] for rung in ladder["rungs_run"]] == ["pca", "ae", "vae"]
    # Ladder is exhausted (none pass the impossible 0.0 threshold); the rung
    # with the smallest validation RMSE is selected, not just the last rung run.
    best_rung = min(ladder["rungs_run"], key=lambda rung: rung["quality_gate"]["value"])
    assert ladder["selected_rung"] == best_rung["rung"]
    assert ladder["quality_sufficient"] is False
    assert ladder["exhausted"] is True
    for rung in ladder["rungs_run"]:
        assert set(rung["reconstruction_metrics"]) == {"train", "val", "test"}
        assert rung["reconstruction_metrics"]["val"]["rmse"] >= 0.0
        assert Path(rung["artifacts"]["model"]).is_file()
