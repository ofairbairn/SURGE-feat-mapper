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

from .cluster import run_cluster_analysis
from .data_preprocess import DataScaler
from .diversity import compute_vendi_diversity, plot_vendi_q_profile
from .tendency import _summarize_cluster_tendency, save_tendency_heatmap

_LADDER_RUNGS = ("pca", "ae", "vae") #ladder order
_LADDER_MODEL_KEYS = {
    "pca": "sklearn.pca",
    "ae": "pytorch.autoencoder",
    "vae": "pytorch.unsupervised_vae",
}
_LADDER_ALIASES = {
    "pca": {"pca", "sklearn.pca"},
    "ae": {"ae", "autoencoder", "pytorch.autoencoder"},
    "vae": {
        "vae",
        "unsupervised_vae",
        "pytorch.unsupervised_vae",
        "owen_vae",
        "pytorch.owen_vae",
    },
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
    The selected latent representation is then evaluated with Hopkins and
    VAT/iVAT before the workflow is allowed to advance to clustering.
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
        gate = _reconstruction_quality_gate( #per rung gate evaluation
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
        if selected_quality_sufficient: #stop climbing on first check that passes threshold
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
    Z_all, all_indices = _combine_latent_splits(
        Z_train,
        Z_val,
        Z_test,
        raw.train_index,
        raw.val_index,
        raw.test_index,
    )

    diversity_report: Optional[Dict[str, Any]] = None
    diversity_artifacts: Dict[str, str] = {}
    diversity_config = dict(spec.mapper_diversity)
    if bool(diversity_config.get("enabled", True)):
        kernel_name = str(diversity_config.get("kernel", "rbf")).strip().lower()
        if kernel_name != "rbf":
            raise ValueError("Mapper Vendi diversity currently supports kernel='rbf'")
        raw_bandwidth = diversity_config.get("rbf_bandwidth")
        rbf_bandwidth = (
            None
            if raw_bandwidth is None
            or str(raw_bandwidth).strip().lower() in {"auto", "median"}
            else float(raw_bandwidth)
        )
        max_samples_value = diversity_config.get("max_samples", 2_000)
        max_samples = (
            None if max_samples_value is None else int(max_samples_value)
        )
        diversity_result = compute_vendi_diversity(
            Z_all,
            q_values=diversity_config.get("q_values"),
            rbf_bandwidth=rbf_bandwidth,
            max_samples=max_samples,
            random_state=int(diversity_config.get("random_state", spec.seed)),
        )
        diversity_report = diversity_result.report
        diversity_report["selected_representation"] = selected_rung
        diversity_report["embedding"]["scope"] = "train_val_test"

        diversity_dir = paths.root / "diversity"
        diversity_dir.mkdir(parents=True, exist_ok=True)
        diversity_report_path = diversity_dir / "vendi_diversity.json"
        with diversity_report_path.open("w", encoding="utf-8") as handle:
            json.dump(diversity_report, handle, indent=2)
        diversity_plot_path = plot_vendi_q_profile(
            diversity_result,
            diversity_dir / "vendi_q_profile.png",
        )
        diversity_artifacts = {
            "report": posix_str(diversity_report_path),
            "q_profile_plot": posix_str(diversity_plot_path),
        }
        if bool(diversity_config.get("save_similarity_matrix", True)):
            similarity_path = diversity_dir / "vendi_similarity_matrix.npz"
            np.savez_compressed(
                similarity_path,
                K=diversity_result.similarity_matrix,
                sample_index=all_indices[diversity_result.sample_positions],
            )
            diversity_artifacts["similarity_matrix"] = posix_str(similarity_path)
        print(
            "[Mapper diversity] "
            f"VS={diversity_report['vendi_score']:.6f} effective modes "
            f"using {diversity_report['kernel']['name'].upper()} kernel",
            flush=True,
        )

    tendency_report: Optional[Dict[str, Any]] = None
    tendency_decision: Optional[Dict[str, Any]] = None
    tendency_artifacts: Dict[str, str] = {}
    tendency_config = dict(spec.mapper_tendency)
    if bool(tendency_config.get("enabled", True)):
        tendency_max_samples_value = tendency_config.get("max_samples", 1_000)
        tendency_max_samples = (
            None
            if tendency_max_samples_value is None
            else int(tendency_max_samples_value)
        )
        tendency_random_state = int(
            tendency_config.get("random_state", spec.seed)
        )
        tendency_positions = _subsample_positions(
            len(Z_all),
            max_samples=tendency_max_samples,
            random_state=tendency_random_state,
        )
        Z_tendency = Z_all[tendency_positions]
        tendency_summary = _summarize_cluster_tendency(
            Z_tendency,
            hopkins_threshold=float(
                tendency_config.get("hopkins_threshold", 0.55)
            ),
            sample_size=tendency_config.get("hopkins_sample_size"),
            random_state=tendency_random_state,
        )
        tendency_decision = decide_clustering_action(tendency_summary)
        tendency_report = {
            key: value
            for key, value in tendency_summary.items()
            if key not in {"vat_matrix", "ivat_matrix"}
        }
        tendency_report["embedding"] = {
            "space": "selected_model_latent_z",
            "selected_representation": selected_rung,
            "scope": "train_val_test",
            "n_samples_total": int(len(Z_all)),
            "n_samples_used": int(len(Z_tendency)),
            "subsampled": bool(len(Z_tendency) < len(Z_all)),
            "max_samples": tendency_max_samples,
            "random_state": tendency_random_state,
        }
        tendency_report["decision"] = tendency_decision

        tendency_dir = paths.root / "tendency"
        tendency_dir.mkdir(parents=True, exist_ok=True)
        tendency_report_path = tendency_dir / "cluster_tendency.json"
        with tendency_report_path.open("w", encoding="utf-8") as handle:
            json.dump(tendency_report, handle, indent=2)
        tendency_matrices_path = tendency_dir / "vat_ivat_matrices.npz"
        np.savez_compressed(
            tendency_matrices_path,
            vat_matrix=np.asarray(tendency_summary["vat_matrix"]),
            ivat_matrix=np.asarray(tendency_summary["ivat_matrix"]),
            sample_index=all_indices[tendency_positions],
            vat_order=np.asarray(tendency_summary["vat_order"], dtype=np.int64),
            vat_parents=np.asarray(
                tendency_summary["vat_parents"], dtype=np.int64
            ),
        )
        vat_plot_path = save_tendency_heatmap(
            tendency_summary["vat_matrix"],
            tendency_dir / "vat_heatmap.png",
            title=f"{selected_rung.upper()} latent VAT",
        )
        ivat_plot_path = save_tendency_heatmap(
            tendency_summary["ivat_matrix"],
            tendency_dir / "ivat_heatmap.png",
            title=f"{selected_rung.upper()} latent iVAT",
        )
        tendency_artifacts = {
            "report": posix_str(tendency_report_path),
            "matrices": posix_str(tendency_matrices_path),
            "vat_heatmap": posix_str(vat_plot_path),
            "ivat_heatmap": posix_str(ivat_plot_path),
        }
        hopkins_text = (
            "undefined"
            if tendency_report["hopkins"] is None
            else f"{tendency_report['hopkins']:.6f}"
        )
        print(
            "[Mapper tendency] "
            f"Hopkins={hopkins_text}, "
            f"threshold={tendency_report['hopkins_threshold']:.6f}: "
            f"{tendency_decision['action']}",
            flush=True,
        )
        if tendency_decision.get("message"):
            print(
                f"[Mapper tendency] {tendency_decision['message']}",
                flush=True,
            )

    clustering_report: Optional[Dict[str, Any]] = None
    clustering_artifacts: Dict[str, str] = {}
    clustering_config = dict(spec.mapper_clustering)
    if (
        tendency_decision is not None
        and bool(tendency_decision["proceed_to_clustering"])
        and bool(clustering_config.get("enabled", True))
    ):
        try:
            clustering_report = run_cluster_analysis(
                Z_all,
                k_anchor=(
                    int(clustering_config["k_anchor"])
                    if clustering_config.get("k_anchor") is not None
                    else None
                ),
                k_max=int(clustering_config.get("k_max", 10)),
                k_window=int(clustering_config.get("k_window", 1)),
                gap_k_max=int(clustering_config.get("gap_k_max", 8)),
                gap_n_references=int(
                    clustering_config.get("gap_n_references", 5)
                ),
                vendi_max_samples=(
                    None
                    if clustering_config.get("vendi_max_samples") is None
                    else int(clustering_config["vendi_max_samples"])
                ),
                random_state=int(
                    clustering_config.get("random_state", spec.seed)
                ),
                hdbscan_min_cluster_size=int(
                    clustering_config.get("hdbscan_min_cluster_size", 20)
                ),
                hdbscan_min_samples=(
                    None
                    if clustering_config.get("hdbscan_min_samples") is None
                    else int(clustering_config["hdbscan_min_samples"])
                ),
                gmm_covariance_type=str(
                    clustering_config.get("gmm_covariance_type", "full")
                ),
            )
            clustering_report["selected_representation"] = selected_rung
            clustering_report["embedding"] = {
                "space": "selected_model_latent_z",
                "scope": "train_val_test",
                "n_samples": int(len(Z_all)),
            }
            clustering_dir = paths.root / "clustering"
            clustering_dir.mkdir(parents=True, exist_ok=True)
            clustering_report_path = clustering_dir / "cluster_analysis.json"
            with clustering_report_path.open("w", encoding="utf-8") as handle:
                json.dump(clustering_report, handle, indent=2)
            clustering_artifacts = {
                "report": posix_str(clustering_report_path),
            }
            k_selection = clustering_report.get("k_selection", {})
            print(
                "[Mapper clustering] "
                f"status={clustering_report.get('status')}, "
                f"k_hdbscan={k_selection.get('k_hdbscan')}, "
                f"selected_k={k_selection.get('selected_k')}, "
                f"global_vendi="
                f"{clustering_report.get('vendi', {}).get('global_vendi')}",
                flush=True,
            )
        except Exception as exc:  # clustering must not fail the workflow
            clustering_report = {"status": "failed", "error": str(exc)}
            print(f"[Mapper clustering] failed: {exc}", flush=True)

    metrics_payload: Dict[str, Any] = {
        "representation_ladder": ladder_results,
    }
    if diversity_report is not None:
        metrics_payload["diversity"] = diversity_report
    if tendency_report is not None:
        metrics_payload["cluster_tendency"] = tendency_report
    if clustering_report is not None:
        metrics_payload["clustering"] = clustering_report
    save_metrics(metrics_payload, paths)

    split_sizes = {
        "train": int(X_train.shape[0]),
        "val": int(X_val.shape[0]),
        "test": int(X_test.shape[0]) if X_test is not None else 0,
    }
    status = _mapper_status_after_tendency(
        tendency_decision,
        diversity_complete=diversity_report is not None,
    )
    if clustering_report is not None:
        status = _mapper_status_after_clustering(clustering_report)
    summary: Dict[str, Any] = {
        "workflow_type": "mapper",
        "status": status,
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
        "diversity": diversity_report,
        "cluster_tendency": tendency_report,
        "clustering_decision": tendency_decision,
        "clustering": clustering_report,
        "next_stage": (
            None
            if clustering_report is not None
            else (
                tendency_decision["next_stage"]
                if tendency_decision is not None
                else "clustering_tendency"
            )
        ),
        "artifacts": {
            "root": posix_str(paths.root),
            "scaler": posix_str(scaler_path),
            "scaled_splits": posix_str(splits_path),
            "latent_splits": posix_str(latent_path),
            "metrics": posix_str(paths.metrics_file),
            "diversity": diversity_artifacts,
            "tendency": tendency_artifacts,
            "clustering": clustering_artifacts,
            "spec": posix_str(paths.spec_file),
            "summary": posix_str(paths.summary_file),
        },
    }
    save_workflow_summary(summary, paths)
    return summary


def _combine_latent_splits(
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    Z_test: Optional[np.ndarray],
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine selected-model embeddings in original dataset row order."""
    embeddings = [Z_train, Z_val]
    indices = [np.asarray(train_index), np.asarray(val_index)]
    if Z_test is not None and test_index is not None:
        embeddings.append(Z_test)
        indices.append(np.asarray(test_index))
    Z_all = np.vstack(embeddings)
    all_indices = np.concatenate(indices)
    try:
        order = np.argsort(all_indices)
    except TypeError:
        order = np.argsort(all_indices.astype(str))
    return Z_all[order], all_indices[order]


def _subsample_positions(
    n_samples: int,
    *,
    max_samples: Optional[int],
    random_state: int,
) -> np.ndarray:
    """Choose deterministic row positions for quadratic VAT/iVAT diagnostics."""
    if n_samples < 1:
        raise ValueError("Cluster tendency requires at least one latent vector")
    if max_samples is None or max_samples >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if max_samples < 1:
        raise ValueError("mapper_tendency.max_samples must be at least 1")
    rng = np.random.default_rng(random_state)
    return np.sort(
        rng.choice(n_samples, size=max_samples, replace=False)
    ).astype(np.int64)


def decide_clustering_action(
    tendency_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Convert the Hopkins gate result into an explicit pipeline decision."""
    if "gate_passed" not in tendency_summary:
        raise ValueError("Cluster tendency summary is missing 'gate_passed'")

    gate_passed = bool(tendency_summary["gate_passed"])
    gate_reason = str(
        tendency_summary.get("gate_reason", "cluster_tendency_gate_failed")
    )
    if gate_reason == "hopkins_undefined":
        return {
            "gate_passed": False,
            "proceed_to_clustering": False,
            "action": "stop_hopkins_undefined",
            "next_stage": None,
            "stop_reason": gate_reason,
            "message": (
                "hopkins statistic undefined: check for zero distances "
                "or identical coordinates"
            ),
        }
    return {
        "gate_passed": gate_passed,
        "proceed_to_clustering": gate_passed,
        "action": (
            "proceed_to_clustering"
            if gate_passed
            else "stop_no_cluster_tendency"
        ),
        "next_stage": "clustering" if gate_passed else None,
        "stop_reason": None if gate_passed else gate_reason,
    }


def _mapper_status_after_tendency(
    tendency_decision: Optional[Mapping[str, Any]],
    *,
    diversity_complete: bool,
) -> str:
    """Report the Mapper stage reached after applying the tendency gate."""
    if tendency_decision is None:
        return (
            "diversity_complete"
            if diversity_complete
            else "representation_complete"
        )
    if tendency_decision.get("action") == "stop_hopkins_undefined":
        return "stopped_hopkins_undefined"
    if bool(tendency_decision["proceed_to_clustering"]):
        return "clustering_ready"
    return "stopped_no_cluster_tendency"


def _mapper_status_after_clustering(
    clustering_report: Mapping[str, Any],
) -> str:
    """Report the final Mapper stage after consensus clustering has run."""
    stage = str(clustering_report.get("status", "complete"))
    if stage == "failed":
        return "clustering_failed"
    if stage == "insufficient_data":
        return "clustering_insufficient_data"
    return "clustering_complete"


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
    if metric not in {"mse", "rmse", "mae"}: #metrics that are used in the gate
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


__all__ = ["decide_clustering_action", "run_mapper_workflow"]
