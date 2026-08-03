"""Tests for Mapper workflow selection and preprocessing orchestration."""

from pathlib import Path

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
    assert summary["status"] == "preprocessing_complete"
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
    assert (root / "workflow_summary.json").is_file()
