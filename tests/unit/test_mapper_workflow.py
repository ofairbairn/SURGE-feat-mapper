"""Tests for Mapper workflow selection and preprocessing orchestration."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from Mapper.data_preprocess import DataScaler
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

    assert summary["workflow_type"] == "mapper"
    assert summary["status"] == "diversity_complete"
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

    root = Path(summary["artifacts"]["root"])
    assert (root / "spec.yaml").is_file()
    assert (root / "metrics.json").is_file()
    assert (root / "mapper_latent_splits.npz").is_file()
    assert (root / "diversity" / "vendi_diversity.json").is_file()
    assert (root / "diversity" / "vendi_q_profile.png").is_file()
    assert (root / "diversity" / "vendi_similarity_matrix.npz").is_file()
    assert (root / "workflow_summary.json").is_file()


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
    assert ladder["selected_rung"] == "vae"
    assert ladder["quality_sufficient"] is False
    assert ladder["exhausted"] is True
    for rung in ladder["rungs_run"]:
        assert set(rung["reconstruction_metrics"]) == {"train", "val", "test"}
        assert rung["reconstruction_metrics"]["val"]["rmse"] >= 0.0
        assert Path(rung["artifacts"]["model"]).is_file()
