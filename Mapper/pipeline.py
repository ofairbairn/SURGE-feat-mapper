"""Orchestration entry point for SURGE Mapper workflows."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from surge.dataset import SurrogateDataset
from surge.engine import EngineRunConfig, SurrogateEngine
from surge.io import (
    init_artifact_paths,
    save_environment_snapshot,
    save_git_revision,
    save_metrics,
    save_model,
    save_scaler,
    save_spec,
    save_workflow_summary,
)
from surge.io.artifacts import copy_invoked_config_source, save_run_invocation
from surge.model.registry import MODEL_REGISTRY
from surge.utils import posix_str
from surge.workflow.spec import ModelConfig, SurrogateWorkflowSpec

from .data_preprocess import DataScaler

_LADDER_RUNGS = ("pca", "ae", "vae")
_LADDER_MODEL_KEYS = {
    "pca": "sklearn.pca",
    "ae": "pytorch.autoencoder",
    "vae": "pytorch.owen_vae",
}
_LADDER_ALIASES = {
    "pca": {"pca", "sklearn.pca"},
    "ae": {"ae", "autoencoder", "pytorch.autoencoder"},
    "vae": {"vae", "owen_vae", "pytorch.owen_vae"},
}
_DEFAULT_MAX_RECON_RMSE = 0.10


def run_mapper_workflow(
    spec: SurrogateWorkflowSpec,
    *,
    invocation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Scale data and select a Mapper representation through a model ladder.

    The scaler and every representation model are fitted only on the training
    partition. Each rung is evaluated on validation reconstruction RMSE. The
    first rung meeting the configured threshold is selected; otherwise the
    runner climbs PCA -> AE -> VAE and retains the final VAE representation.
    Clustering-tendency analysis intentionally remains a later pipeline stage.
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

    ladder_results = []
    selected_adapter = None
    selected_rung = None
    selected_quality_sufficient = False
    for rung in _LADDER_RUNGS:
        model_key, model_name, model_params = _resolve_ladder_model(spec, rung)
        adapter = MODEL_REGISTRY.create(model_key, **model_params)
        if hasattr(adapter, "prepare_for_fit"):
            adapter.prepare_for_fit(
                resources=spec.resources,
                X_shape=X_train.shape,
                y_shape=X_train.shape,
            )

        fit_start = time.perf_counter()
        if rung == "pca":
            adapter.fit(X_train, X_train)
        else:
            adapter.fit(
                X_train,
                X_train,
                X_val=X_val,
                y_val=X_val,
            )
        adapter.mark_fitted()
        fit_seconds = float(time.perf_counter() - fit_start)

        split_metrics = {
            "train": _serializable_metrics(adapter.reconstruction_metrics(X_train)),
            "val": _serializable_metrics(adapter.reconstruction_metrics(X_val)),
            "test": (
                _serializable_metrics(adapter.reconstruction_metrics(X_test))
                if X_test is not None
                else None
            ),
        }
        gate = _reconstruction_quality_gate(
            rung,
            split_metrics["val"],
            spec.unsupervised_ladder_thresholds,
        )
        model_path = save_model(adapter, f"mapper_{rung}", paths)
        rung_artifacts: Dict[str, str] = {"model": posix_str(model_path)}

        history = getattr(adapter, "training_history", None)
        if history:
            history_path = paths.root / f"training_history_mapper_{rung}.json"
            with history_path.open("w", encoding="utf-8") as handle:
                json.dump(list(history), handle, indent=2)
            rung_artifacts["training_history"] = posix_str(history_path)

        rung_result = {
            "rung": rung,
            "model_key": model_key,
            "model_name": model_name,
            "params": model_params,
            "fit_seconds": fit_seconds,
            "reconstruction_metrics": split_metrics,
            "quality_gate": gate,
            "artifacts": rung_artifacts,
        }
        ladder_results.append(rung_result)
        gate_outcome = (
            "PASS"
            if gate["passed"]
            else "EXHAUSTED" if rung == _LADDER_RUNGS[-1] else "CLIMB"
        )
        print(
            f"[Mapper ladder] {rung.upper()} validation "
            f"{gate['metric']}={gate['value']:.6f} "
            f"(required <= {gate['threshold']:.6f}): {gate_outcome}",
            flush=True,
        )

        selected_adapter = adapter
        selected_rung = rung
        selected_quality_sufficient = bool(gate["passed"])
        if selected_quality_sufficient:
            break

    if selected_adapter is None or selected_rung is None:  # pragma: no cover
        raise RuntimeError("Mapper representation ladder did not run any models.")

    Z_train = np.asarray(selected_adapter.encode(X_train))
    Z_val = np.asarray(selected_adapter.encode(X_val))
    Z_test = (
        np.asarray(selected_adapter.encode(X_test)) if X_test is not None else None
    )
    latent_path = paths.root / "mapper_latent_splits.npz"
    latent_payload = {
        "Z_train": Z_train,
        "Z_val": Z_val,
        "train_index": raw.train_index,
        "val_index": raw.val_index,
    }
    if Z_test is not None and raw.test_index is not None:
        latent_payload["Z_test"] = Z_test
        latent_payload["test_index"] = raw.test_index
    np.savez_compressed(latent_path, **latent_payload)

    save_metrics({"representation_ladder": ladder_results}, paths)

    split_sizes = {
        "train": int(X_train.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(X_test.shape[0]) if X_test is not None else 0,
    }
    summary: Dict[str, Any] = {
        "workflow_type": "mapper",
        "status": "representation_complete",
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
        "representation_ladder": {
            "order": list(_LADDER_RUNGS),
            "gate_split": "val",
            "gate_metric": ladder_results[0]["quality_gate"]["metric"],
            "default_max_recon_rmse": _DEFAULT_MAX_RECON_RMSE,
            "rungs_run": ladder_results,
            "selected_rung": selected_rung,
            "quality_sufficient": selected_quality_sufficient,
            "exhausted": (
                selected_rung == _LADDER_RUNGS[-1]
                and not selected_quality_sufficient
            ),
        },
        "next_stage": "clustering_tendency",
        "artifacts": {
            "root": posix_str(paths.root),
            "scaler": posix_str(scaler_path),
            "scaled_splits": posix_str(splits_path),
            "latent_splits": posix_str(latent_path),
            "metrics": posix_str(paths.metrics_file),
            "spec": posix_str(paths.spec_file),
            "summary": posix_str(paths.summary_file),
        },
    }
    save_workflow_summary(summary, paths)
    return summary


def _resolve_ladder_model(
    spec: SurrogateWorkflowSpec,
    rung: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve a rung's registry key and optional YAML model parameters."""
    configured: Optional[ModelConfig] = None
    for model in spec.models:
        if model.key.strip().lower() in _LADDER_ALIASES[rung]:
            configured = model
            break

    model_key = _LADDER_MODEL_KEYS[rung]
    model_name = rung.upper()
    params: Dict[str, Any] = {}
    if configured is not None:
        if configured.key in MODEL_REGISTRY:
            model_key = configured.key
        model_name = configured.name or model_name
        params.update(configured.params)
    params.setdefault("random_state", spec.seed)
    return model_key, model_name, params


def _reconstruction_quality_gate(
    rung: str,
    validation_metrics: Mapping[str, Optional[float]],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate one ladder rung using a configurable validation error limit."""
    metric = str(thresholds.get("reconstruction_metric", "rmse")).strip().lower()
    if metric.startswith("recon_"):
        metric = metric.removeprefix("recon_")
    if metric not in {"mse", "rmse", "mae"}:
        raise ValueError(
            "unsupervised_ladder_thresholds.reconstruction_metric must be "
            "one of: mse, rmse, mae"
        )

    threshold_key = f"{rung}_max_recon_{metric}"
    global_key = f"max_recon_{metric}"
    default_threshold = _DEFAULT_MAX_RECON_RMSE
    if metric == "mse":
        default_threshold = _DEFAULT_MAX_RECON_RMSE**2
    threshold = float(
        thresholds.get(
            threshold_key,
            thresholds.get(global_key, default_threshold),
        )
    )
    if threshold < 0:
        raise ValueError(f"{threshold_key} must be non-negative")

    raw_value = validation_metrics.get(metric)
    value = float(raw_value) if raw_value is not None else float("inf")
    passed = bool(np.isfinite(value) and value <= threshold)
    return {
        "split": "val",
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "threshold_key": (
            threshold_key
            if threshold_key in thresholds
            else global_key if global_key in thresholds else "default"
        ),
        "passed": passed,
    }


def _serializable_metrics(
    metrics: Mapping[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    return {
        str(key): None if value is None else float(value)
        for key, value in metrics.items()
    }


def _default_mapper_run_tag(file_path: Optional[Path]) -> str:
    prefix = file_path.stem if file_path else "dataset"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_mapper_{timestamp}"


__all__ = ["run_mapper_workflow"]
