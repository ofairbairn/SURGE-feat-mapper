"""Orchestration entry point for SURGE Mapper workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from surge.dataset import SurrogateDataset
from surge.engine import EngineRunConfig, SurrogateEngine
from surge.io import (
    init_artifact_paths,
    save_environment_snapshot,
    save_git_revision,
    save_scaler,
    save_spec,
    save_workflow_summary,
)
from surge.io.artifacts import copy_invoked_config_source, save_run_invocation
from surge.utils import posix_str
from surge.workflow.spec import SurrogateWorkflowSpec

from .data_preprocess import DataScaler


def run_mapper_workflow(
    spec: SurrogateWorkflowSpec,
    *,
    invocation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load, split, and robustly scale data for the Mapper pipeline.

    This first Mapper stage intentionally stops after preprocessing. The
    scaler is fitted only on the training partition; validation and test data
    are transformed with the already-fitted training statistics. Scaled
    splits and the fitted scaler are persisted for subsequent representation,
    clustering, anomaly, and visualization stages.
    """
    if spec.workflow_type != "mapper":
        raise ValueError(
            "run_mapper_workflow requires workflow_type='mapper'; "
            f"got {spec.workflow_type!r}"
        )
    if spec.dataset_source is not None:
        raise NotImplementedError(
            "Mapper currently supports datasets loaded through "
            "SurrogateDataset; dataset_source loaders are not wired yet."
        )

    dataset = SurrogateDataset.from_path(
        spec.dataset_path,
        format=spec.dataset_format,
        metadata_path=spec.metadata_path,
        sample=spec.sample_rows,
        sample_random_state=spec.seed,
        analyzer_kwargs={
            "hints": {**spec.metadata_overrides, "task_type": "unsupervised"},
            **spec.analyzer,
        },
    )
    if dataset.df is None or not dataset.input_columns:
        raise ValueError("Mapper requires a non-empty dataset with input features.")

    # Reuse SURGE's deterministic split logic, but deliberately disable its
    # StandardScaler: Mapper owns preprocessing and uses DataScaler below.
    engine = SurrogateEngine(
        run_config=EngineRunConfig(
            test_fraction=spec.test_fraction,
            val_fraction=spec.val_fraction,
            standardize_inputs=False,
            standardize_outputs=False,
            random_state=spec.seed,
            task_type="unsupervised",
            resources=spec.resources,
        )
    )
    engine.configure_dataframe(
        dataset.df,
        dataset.input_columns,
        dataset.input_columns,
    )
    engine.prepare()
    raw = engine.get_raw_splits()

    scaler = DataScaler()
    X_train = scaler.fit_transform(raw.X_train)
    X_val = scaler.transform(raw.X_val)
    X_test = scaler.transform(raw.X_test) if raw.X_test is not None else None

    run_tag = spec.run_tag or _default_mapper_run_tag(dataset.file_path)
    paths = init_artifact_paths(
        spec.output_dir,
        run_tag,
        exist_ok=spec.overwrite_existing_run,
    )
    scaler_path = save_scaler(scaler, "mapper_inputs", paths)
    splits_path = paths.root / "mapper_scaled_splits.npz"
    split_payload = {
        "X_train": X_train,
        "X_val": X_val,
        "train_index": raw.train_index,
        "val_index": raw.val_index,
    }
    if X_test is not None and raw.test_index is not None:
        split_payload["X_test"] = X_test
        split_payload["test_index"] = raw.test_index
    np.savez_compressed(splits_path, **split_payload)

    save_spec(spec.to_dict(), paths)
    save_environment_snapshot(paths)
    save_git_revision(paths, repo_dir=".")
    if invocation:
        save_run_invocation(paths, invocation)
        spec_source = invocation.get("spec_path")
        if spec_source:
            copy_invoked_config_source(paths, Path(spec_source))

    split_sizes = {
        "train": int(X_train.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(X_test.shape[0]) if X_test is not None else 0,
    }
    summary: Dict[str, Any] = {
        "workflow_type": "mapper",
        "status": "preprocessing_complete",
        "dataset": dataset.summary(),
        "input_columns": list(dataset.input_columns),
        "split_sizes": split_sizes,
        "scaling": {
            "method": "robust",
            "fit_split": "train",
            "with_centering": scaler.with_centering,
            "with_scaling": scaler.with_scaling,
            "quantile_range": list(scaler.quantile_range),
        },
        "artifacts": {
            "root": posix_str(paths.root),
            "scaler": posix_str(scaler_path),
            "scaled_splits": posix_str(splits_path),
            "spec": posix_str(paths.spec_file),
            "summary": posix_str(paths.summary_file),
        },
    }
    save_workflow_summary(summary, paths)
    return summary


def _default_mapper_run_tag(file_path: Optional[Path]) -> str:
    prefix = file_path.stem if file_path else "dataset"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_mapper_{timestamp}"


__all__ = ["run_mapper_workflow"]
