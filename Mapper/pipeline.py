"""Orchestration entry point for SURGE Mapper workflows."""

from __future__ import annotations

import json
import re
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
from .data_preprocess import DataScaler, ImageDataScaler, analyze_missingness
from .diversity import compute_vendi_diversity, plot_vendi_q_profile
from .anomaly.anomaly import run_anomaly_detection
from .stability.stability import run_cluster_stability
from .tendency import _summarize_cluster_tendency, save_tendency_heatmap
from .viz import plot_mapper_latent, plot_mapper_pca, plot_mapper_reconstruction

_LADDER_RUNGS = ("pca", "ae", "vae") #ladder order
_LADDER_MODEL_KEYS = {
    "pca": "sklearn.pca",
    "ae": "pytorch.autoencoder",
    "vae": "pytorch.unsupervised_vae",
}
_CONV_LADDER_MODEL_KEYS = {
    "ae": "pytorch.conv_autoencoder",
    "vae": "pytorch.conv_unsupervised_vae",
}
_LADDER_ALIASES = {
    "pca": {"pca", "sklearn.pca"},
    "ae": {
        "ae",
        "autoencoder",
        "pytorch.autoencoder",
        "conv_autoencoder",
        "cae",
        "pytorch.conv_autoencoder",
    },
    "vae": {
        "vae",
        "unsupervised_vae",
        "pytorch.unsupervised_vae",
        "owen_vae",
        "pytorch.owen_vae",
        "conv_unsupervised_vae",
        "conv_vae",
        "pytorch.conv_unsupervised_vae",
    },
}
_GENERIC_RUNG_ALIASES = {
    "pca": {"pca"},
    "ae": {"ae", "autoencoder"},
    "vae": {"vae", "unsupervised_vae"},
}
_CONV_MODEL_ALIASES = {
    "conv_autoencoder",
    "cae",
    "pytorch.conv_autoencoder",
    "conv_unsupervised_vae",
    "conv_vae",
    "pytorch.conv_unsupervised_vae",
}
_PIXEL_COLUMN_PATTERN = re.compile(r"^pixel[_-]?(\d+)$", re.IGNORECASE)
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
    input_layout = _resolve_mapper_input_layout(dataset, spec)
    ordered_columns = input_layout.pop("ordered_input_columns", None)
    if ordered_columns is not None:
        dataset.input_columns = list(ordered_columns)
    print(
        "[Mapper input] "
        f"data_type={input_layout['data_type']}, "
        f"input_shape={input_layout.get('input_shape')}, "
        f"source={input_layout['source']}",
        flush=True,
    )

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
    ###DATA LOADER/SCALER MODULE###
    scaler = (
        ImageDataScaler()
        if input_layout["data_type"] == "image"
        else DataScaler()
    )
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

    ###PREPROCESSING/MISSING VALUES MODULE###
    preprocessing_report: Optional[Dict[str, Any]] = None
    preprocessing_artifacts: Dict[str, Any] = {}
    preprocessing_config = dict(spec.mapper_preprocess)
    if bool(preprocessing_config.get("enabled", True)):
        try:  # missing-value reporting must not fail the workflow
            preprocess_dir = paths.root / "preprocessing"
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            preprocessing_report = analyze_missingness(
                dataset.df,
                preprocess_dir,
                random_state=spec.seed,
                save_plots=bool(preprocessing_config.get("save_plots", True)),
                run_mcar_test=bool(preprocessing_config.get("run_mcar_test", True)),
            )
            preprocessing_report_path = (
                preprocess_dir / "data_preprocessing.json"
            )
            with preprocessing_report_path.open("w", encoding="utf-8") as handle:
                json.dump(preprocessing_report, handle, indent=2)
            preprocessing_artifacts = {
                "report": posix_str(preprocessing_report_path),
            }
            for kind, name in (preprocessing_report.get("plots") or {}).items():
                preprocessing_artifacts[f"plot_{kind}"] = posix_str(
                    preprocess_dir / name
                )
            mcar_result = preprocessing_report.get("mcar_test") or {}
            little_p = mcar_result.get("little_pvalue")
            print(
                "[Mapper Preprocess] "
                f"status=complete, scaling={_scaling_summary(scaler)['method']}, "
                f"missing_values_detected="
                f"{preprocessing_report.get('missing_values_detected')}, "
                f"completeness="
                f"{preprocessing_report.get('completeness_percent')}%, "
                f"missing_cells={preprocessing_report.get('n_missing_cells')}"
                + (
                    f", little_mcar_p={little_p:.4f}"
                    if isinstance(little_p, (int, float))
                    else ""
                ),
                flush=True,
            )
        except Exception as exc:
            preprocessing_report = {"status": "failed", "error": str(exc)}
            print(f"[Mapper Preprocess] failed: {exc}", flush=True)

    save_spec(spec.to_dict(), paths)
    save_environment_snapshot(paths)
    save_git_revision(paths, repo_dir=".")
    if invocation:
        save_run_invocation(paths, invocation)
        spec_source = invocation.get("spec_path")
        if spec_source:
            copy_invoked_config_source(paths, Path(spec_source))
        ###REPRESENTATION STEPS###
    ladder_results = []
    rung_adapters: Dict[str, Any] = {}
    selected_adapter = None
    selected_rung = None
    selected_quality_sufficient = False
    for rung in _LADDER_RUNGS:
        model_key, model_name, model_params = _resolve_ladder_model(
            spec,
            rung,
            input_layout=input_layout,
        )
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
        rung_adapters[rung] = adapter

        split_metrics = {
            "train": _add_rmse_mae_ratio(_serializable_metrics(adapter.reconstruction_metrics(X_train))),
            "val": _add_rmse_mae_ratio(_serializable_metrics(adapter.reconstruction_metrics(X_val))),
            "test": (
                _add_rmse_mae_ratio(_serializable_metrics(adapter.reconstruction_metrics(X_test)))
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

    if not selected_quality_sufficient:
        # ladder exhausted without passing the gate: use the smallest validation
        # error across all rungs run instead of defaulting to the last (VAE) rung
        best_rung_result = min(ladder_results, key=lambda result: result["quality_gate"]["value"])
        selected_rung = best_rung_result["rung"]
        selected_adapter = rung_adapters[selected_rung]
        print(
            "[Mapper ladder] Exhausted without passing the gate; selecting rung with "
            f"smallest validation {best_rung_result['quality_gate']['metric']}: "
            f"{selected_rung.upper()}",
            flush=True,
        )

    ladder_exhausted = (
        bool(ladder_results)
        and ladder_results[-1]["rung"] == _LADDER_RUNGS[-1]
        and not selected_quality_sufficient
    )
    if selected_adapter is None or selected_rung is None:  # pragma: no cover
        raise RuntimeError("Mapper representation ladder did not run any models.")
    #added 254-257 to check for nans
    _check_split_finiteness(
        {"X_train": X_train, "X_val": X_val, "X_test": X_test},
        stage="scaled_inputs",
    )

    Z_train = np.asarray(selected_adapter.encode(X_train))
    Z_val = np.asarray(selected_adapter.encode(X_val))
    Z_test = (
        np.asarray(selected_adapter.encode(X_test)) if X_test is not None else None
    )
    #added 265-268 to check for nans
    _check_split_finiteness(
        {"Z_train": Z_train, "Z_val": Z_val, "Z_test": Z_test},
        stage="latent_embeddings",
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
    recon_payload = _compute_reconstruction_payload(
        selected_adapter,
        X_train,
        X_val,
        X_test,
        raw.train_index,
        raw.val_index,
        raw.test_index,
    )
        ###DIVERSITY MODULE STEP###
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
                tendency_config.get("hopkins_threshold", 0.65)
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

    stability_report: Optional[Dict[str, Any]] = None
    stability_artifacts: Dict[str, str] = {}
    stability_config = dict(spec.mapper_stability)
    if clustering_report is not None and bool(stability_config.get("enabled", True)):
        if str(clustering_report.get("status", "")).lower() == "complete":
            try:
                selected_k = int(clustering_report.get("k_selection", {}).get("selected_k", 1))
                stability_report, stability_arrays = run_cluster_stability(
                    Z_all,
                    selected_k=selected_k,
                    baseline_labels=(
                        clustering_report.get("labels")
                        if isinstance(clustering_report.get("labels"), dict)
                        else None
                    ),
                    n_bootstraps=int(stability_config.get("n_bootstraps", 25)),
                    bootstrap_fraction=float(stability_config.get("bootstrap_fraction", 0.8)),
                    random_state=int(stability_config.get("random_state", spec.seed)),
                    max_samples=(
                        None
                        if stability_config.get("max_samples") is None
                        else int(stability_config["max_samples"])
                    ),
                    gmm_covariance_type=str(
                        clustering_config.get("gmm_covariance_type", "full")
                    ),
                    hdbscan_min_cluster_size=int(
                        clustering_config.get("hdbscan_min_cluster_size", 20)
                    ),
                    hdbscan_min_samples=(
                        None
                        if clustering_config.get("hdbscan_min_samples") is None
                        else int(clustering_config["hdbscan_min_samples"])
                    ),
                )
                stability_report["selected_representation"] = selected_rung
                stability_report["embedding"] = {
                    "space": "selected_model_latent_z",
                    "scope": "train_val_test",
                    "n_samples_total": int(len(Z_all)),
                    "n_samples_used": int(len(stability_arrays["sample_positions"])),
                    "subsampled": bool(len(stability_arrays["sample_positions"]) < len(Z_all)),
                    "max_samples": (
                        None
                        if stability_config.get("max_samples") is None
                        else int(stability_config["max_samples"])
                    ),
                    "random_state": int(stability_config.get("random_state", spec.seed)),
                }

                stability_dir = paths.root / "stability"
                stability_dir.mkdir(parents=True, exist_ok=True)
                stability_report_path = stability_dir / "cluster_stability.json"
                with stability_report_path.open("w", encoding="utf-8") as handle:
                    json.dump(stability_report, handle, indent=2)
                stability_arrays_path = stability_dir / "cluster_stability_arrays.npz"
                np.savez_compressed(
                    stability_arrays_path,
                    sample_index=all_indices[stability_arrays["sample_positions"]],
                    baseline_kmeans_labels=stability_arrays["baseline_kmeans_labels"],
                    baseline_gmm_labels=stability_arrays["baseline_gmm_labels"],
                    baseline_hdbscan_labels=stability_arrays.get(
                        "baseline_hdbscan_labels",
                        np.full(
                            len(stability_arrays["baseline_kmeans_labels"]),
                            -1,
                            dtype=np.int64,
                        ),
                    ),
                    consensus_labels=stability_arrays["consensus_labels"],
                    coassociation_matrix=stability_arrays["coassociation_matrix"],
                )
                stability_artifacts = {
                    "report": posix_str(stability_report_path),
                    "arrays": posix_str(stability_arrays_path),
                }
                consensus_agreement = stability_report.get("consensus_clustering", {}).get("agreement", {})
                print(
                    "[Mapper stability] "
                    f"status={stability_report.get('status')}, "
                    f"selected_k={stability_report.get('selected_k')}, "
                    f"hdbscan_jaccard={stability_report.get('bootstrap_jaccard', {}).get('hdbscan', {}).get('mean')}, "
                    f"kmeans_jaccard={stability_report.get('bootstrap_jaccard', {}).get('kmeans', {}).get('mean')}, "
                    f"consensus_ari={consensus_agreement.get('hdbscan', {}).get('ari')}",
                    flush=True,
                )
            except Exception as exc:  # stability must not fail the workflow
                stability_report = {"status": "failed", "error": str(exc)}
                print(f"[Mapper stability] failed: {exc}", flush=True)

    anomaly_report: Optional[Dict[str, Any]] = None
    anomaly_artifacts: Dict[str, str] = {}
    anomaly_arrays: Optional[Dict[str, np.ndarray]] = None
    anomaly_config = dict(spec.mapper_anomaly)
    if bool(anomaly_config.get("enabled", True)):
        try:
            selected_k_hint = (
                int(clustering_report["k_selection"]["selected_k"])
                if clustering_report is not None
                and str(clustering_report.get("status", "")).lower() == "complete"
                and clustering_report.get("k_selection", {}).get("selected_k") is not None
                else None
            )
            anomaly_report, anomaly_arrays = run_anomaly_detection(
                Z_all,
                recon_error=(
                    recon_payload["recon_error"] if recon_payload is not None else None
                ),
                n_clusters_hint=selected_k_hint,
                top_n=int(anomaly_config.get("top_n", 100)),
                reason_quantile=float(anomaly_config.get("reason_quantile", 0.90)),
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
                isolation_forest_n_estimators=int(
                    anomaly_config.get("isolation_forest_n_estimators", 200)
                ),
                isolation_forest_contamination=anomaly_config.get(
                    "isolation_forest_contamination", "auto"
                ),
                lof_n_neighbors=int(anomaly_config.get("lof_n_neighbors", 20)),
                lof_contamination=anomaly_config.get("lof_contamination", "auto"),
                vendi_max_samples=(
                    None
                    if anomaly_config.get("vendi_max_samples", 200) is None
                    else int(anomaly_config.get("vendi_max_samples", 200))
                ),
                rbf_bandwidth=anomaly_config.get("rbf_bandwidth"),
                random_state=int(anomaly_config.get("random_state", spec.seed)),
            )
            if anomaly_report.get("status") == "complete":
                for entry in anomaly_report["top_anomalies"]:
                    entry["dataset_sample_index"] = int(all_indices[entry["sample_position"]])
                anomaly_report["selected_representation"] = selected_rung
                anomaly_report["embedding"] = {
                    "space": "selected_model_latent_z",
                    "scope": "train_val_test",
                    "n_samples": int(len(Z_all)),
                }

                anomaly_dir = paths.root / "anomaly"
                anomaly_dir.mkdir(parents=True, exist_ok=True)
                anomaly_report_path = anomaly_dir / "anomaly_list.json"
                with anomaly_report_path.open("w", encoding="utf-8") as handle:
                    json.dump(anomaly_report, handle, indent=2)
                anomaly_arrays_path = anomaly_dir / "anomaly_scores.npz"
                np.savez_compressed(
                    anomaly_arrays_path,
                    sample_index=all_indices,
                    **anomaly_arrays,
                )
                anomaly_artifacts = {
                    "report": posix_str(anomaly_report_path),
                    "arrays": posix_str(anomaly_arrays_path),
                }
                top_entry = anomaly_report["top_anomalies"][0] if anomaly_report["top_anomalies"] else None
                print(
                    "[Mapper anomaly] "
                    f"status={anomaly_report.get('status')}, "
                    f"n_ranked={anomaly_report.get('n_ranked')}, "
                    f"top_score={top_entry['combined_score'] if top_entry else None}",
                    flush=True,
                )
            else:
                print(f"[Mapper anomaly] status={anomaly_report.get('status')}", flush=True)
        except Exception as exc:  # anomaly detection must not fail the workflow
            anomaly_report = {"status": "failed", "error": str(exc)}
            print(f"[Mapper anomaly] failed: {exc}", flush=True)

    viz_artifacts: Dict[str, Any] = {}
    visualization_config = dict(spec.mapper_visualization)
    if bool(visualization_config.get("enabled", True)):
        try:  # visualization must not fail the workflow
            viz_dir = paths.root / "visualization"
            viz_dir.mkdir(parents=True, exist_ok=True)
            latent_result = plot_mapper_latent(
                Z_all,
                sample_indices=all_indices,
                output_dir=viz_dir,
                model_name=f"mapper_{selected_rung}",
                cluster_labels=None,
                recon_error=(
                    recon_payload["recon_error"] if recon_payload is not None else None
                ),
                anomaly_score=(
                    anomaly_arrays.get("combined_score")
                    if anomaly_arrays is not None
                    else None
                ),
                hdbscan_min_cluster_size=int(
                    clustering_config.get("hdbscan_min_cluster_size", 20)
                ),
                hdbscan_min_samples=(
                    None
                    if clustering_config.get("hdbscan_min_samples") is None
                    else int(clustering_config["hdbscan_min_samples"])
                ),
                random_state=int(
                    visualization_config.get("random_state", spec.seed)
                ),
                include_tsne=bool(
                    tendency_decision is not None
                    and bool(tendency_decision.get("proceed_to_clustering", False))
                ),
                tendency_summary=tendency_report,
                hopkins_threshold=float(
                    tendency_config.get("hopkins_threshold", 0.65)
                ),
                save_tendency_artifacts=False,
            )
            viz_artifacts["latent"] = latent_result.get("saved_paths", [])
            if selected_rung == "pca":
                pca_result = plot_mapper_pca(
                    Z_all,
                    output_dir=viz_dir,
                    model_name=f"mapper_{selected_rung}",
                    split="train_val_test",
                    explained_variance_ratio=_extract_pca_explained_variance_ratio(
                        selected_adapter
                    ),
                )
                viz_artifacts["pca"] = pca_result.get("saved_paths", [])
            if recon_payload is not None:
                recon_result = plot_mapper_reconstruction(
                    recon_payload["y_true"],
                    recon_payload["y_pred"],
                    sample_indices=all_indices,
                    output_dir=viz_dir,
                    model_name=f"mapper_{selected_rung}",
                )
                viz_artifacts["reconstruction"] = recon_result.get(
                    "saved_paths", []
                )
            n_artifacts = sum(len(v) for v in viz_artifacts.values())
            print(
                "[Mapper visualization] "
                f"status=complete, n_artifacts={n_artifacts}",
                flush=True,
            )
        except Exception as exc:
            print(f"[Mapper visualization] failed: {exc}", flush=True)

    metrics_payload: Dict[str, Any] = {
        "representation_ladder": ladder_results,
        "input_layout": input_layout,
    }
    if preprocessing_report is not None:
        metrics_payload["preprocessing"] = preprocessing_report
    if diversity_report is not None:
        metrics_payload["diversity"] = diversity_report
    if tendency_report is not None:
        metrics_payload["cluster_tendency"] = tendency_report
    if clustering_report is not None:
        metrics_payload["clustering"] = clustering_report
    if stability_report is not None:
        metrics_payload["stability"] = stability_report
    if anomaly_report is not None:
        metrics_payload["anomaly"] = anomaly_report
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
        "input_layout": input_layout,
        "input_columns": list(dataset.input_columns),
        "split_sizes": split_sizes,
        "scaling": _scaling_summary(scaler),
        "preprocessing": preprocessing_report,
        "representation_ladder": {
            "order": list(_LADDER_RUNGS),
            "gate_split": "val",
            "gate_metric": ladder_results[0]["quality_gate"]["metric"],
            "default_max_recon_rmse": _DEFAULT_MAX_RECON_RMSE,
            "rungs_run": ladder_results,
            "selected_rung": selected_rung,
            "quality_sufficient": selected_quality_sufficient,
            "exhausted": ladder_exhausted,
        },
        "diversity": diversity_report,
        "cluster_tendency": tendency_report,
        "clustering_decision": tendency_decision,
        "clustering": clustering_report,
        "stability": stability_report,
        "anomaly": anomaly_report,
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
            "preprocessing": preprocessing_artifacts,
            "diversity": diversity_artifacts,
            "tendency": tendency_artifacts,
            "clustering": clustering_artifacts,
            "stability": stability_artifacts,
            "anomaly": anomaly_artifacts,
            "visualization": viz_artifacts,
            "spec": posix_str(paths.spec_file),
            "summary": posix_str(paths.summary_file),
        },
    }
    save_workflow_summary(summary, paths)
    return summary

#844-890 added to check for nans in scaled inputs and latent embeddings.
def _check_split_finiteness(
    arrays: Mapping[str, np.ndarray],
    *,
    stage: str,
) -> None:
    """Raise a diagnostic error when any split array contains NaN or inf.

    ``stage`` names where the non-finite values were produced so the message
    points at the root cause: ``scaled_inputs`` (case 1) is a RobustScaler
    divide-by-zero from a constant column in the train split; ``latent_embeddings``
    (case 2) is a diverged AE/VAE encoder.
    """
    offenders: list[str] = []
    for name, arr in arrays.items():
        if arr is None:
            continue
        finite = np.isfinite(arr)
        if bool(finite.all()):
            continue
        bad = ~finite
        n_bad = int(np.count_nonzero(bad))
        detail = f"{name}: {n_bad} non-finite of {int(arr.size)} values"
        if arr.ndim == 2:
            bad_rows = np.unique(np.where(bad)[0])[:5]
            bad_cols = np.unique(np.where(bad)[1])[:5]
            detail += (
                f" (sample rows {list(int(r) for r in bad_rows)}, "
                f"feature cols {list(int(c) for c in bad_cols)})"
            )
        offenders.append(detail)
    if not offenders:
        return
    hint = (
        "This usually means a constant column (zero IQR) in the training "
        "split made RobustScaler divide by zero."
        if stage == "scaled_inputs"
        else "This usually means the selected AE/VAE diverged during "
        "training and produced non-finite weights."
    )
    raise ValueError(
        "[Mapper] non-finite values in "
        f"{stage} after the representation ladder:\n  "
        + "\n  ".join(offenders)
        + f"\n{hint}"
    )


def _compute_reconstruction_payload(
    adapter: Any,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: Optional[np.ndarray],
    train_index: np.ndarray,
    val_index: np.ndarray,
    test_index: Optional[np.ndarray],
) -> Optional[Dict[str, np.ndarray]]:
    """Per-sample reconstruction predictions/errors in original row order."""
    if not callable(getattr(adapter, "reconstruct", None)):
        return None
    y_true_parts: list = []
    y_pred_parts: list = []
    index_parts: list = []
    for X, idx in (
        (X_train, train_index),
        (X_val, val_index),
        (X_test, test_index),
    ):
        if X is None or idx is None:
            continue
        try:
            y_true = np.asarray(X, dtype=np.float64)
            y_pred = np.asarray(adapter.reconstruct(X))
        except Exception:
            return None
        if y_pred.shape != y_true.shape:
            same_samples = y_pred.ndim >= 1 and y_pred.shape[0] == y_true.shape[0]
            same_sample_size = (
                same_samples
                and int(np.prod(y_pred.shape[1:])) == int(np.prod(y_true.shape[1:]))
            )
            if not same_sample_size:
                return None
            y_pred = y_pred.reshape(y_true.shape)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
        index_parts.append(np.asarray(idx))
    if not y_true_parts:
        return None
    y_true_all = np.vstack(y_true_parts)
    y_pred_all = np.vstack(y_pred_parts)
    all_indices = np.concatenate(index_parts)
    try:
        order = np.argsort(all_indices)
    except TypeError:
        order = np.argsort(all_indices.astype(str))
    y_true_all = y_true_all[order]
    y_pred_all = y_pred_all[order]
    recon_error = np.sqrt(np.mean(np.square(y_pred_all - y_true_all), axis=1))
    return {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "recon_error": recon_error,
        "n_samples": int(len(all_indices)),
    }


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


def _pixel_column_order(columns: Any) -> Optional[list[str]]:
    """Return contiguous pixel columns in numeric rather than lexical order."""
    indexed: list[tuple[int, str]] = []
    for column in columns:
        match = _PIXEL_COLUMN_PATTERN.fullmatch(str(column))
        if match is None:
            return None
        indexed.append((int(match.group(1)), str(column)))
    if not indexed:
        return None
    indexed.sort(key=lambda item: item[0])
    if [index for index, _ in indexed] != list(range(len(indexed))):
        return None
    return [column for _, column in indexed]


def _normalize_image_shape(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("input_shape must be a list or tuple")
    shape = tuple(int(dimension) for dimension in value)
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise ValueError("Image input_shape must be positive (H, W) or (C, H, W)")
    return shape


def _configured_conv_shape(spec: SurrogateWorkflowSpec) -> Optional[Tuple[int, int, int]]:
    for model in spec.models:
        if model.key.strip().lower() not in _CONV_MODEL_ALIASES:
            continue
        shape = _normalize_image_shape(model.params.get("input_shape"))
        if shape is not None:
            return shape
    return None


def _resolve_mapper_input_layout(
    dataset: SurrogateDataset,
    spec: SurrogateWorkflowSpec,
) -> Dict[str, Any]:
    """Resolve Mapper's semantic sample layout before scaling/model selection."""
    metadata = dict(dataset.metadata or {})
    overrides = dict(spec.metadata_overrides or {})
    raw_data_type = spec.data_type
    source = "workflow.data_type"
    if raw_data_type == "auto":
        raw_data_type = str(
            overrides.get(
                "data_type",
                overrides.get(
                    "modality",
                    metadata.get("data_type", metadata.get("modality", "auto")),
                ),
            )
        ).strip().lower()
        source = (
            "metadata_overrides"
            if "data_type" in overrides or "modality" in overrides
            else "metadata" if "data_type" in metadata or "modality" in metadata
            else "auto"
        )
    aliases = {"images": "image", "vision": "image", "table": "tabular"}
    data_type = aliases.get(raw_data_type, raw_data_type)
    if data_type not in {"auto", "image", "tabular"}:
        raise ValueError(
            "Mapper data_type/modality must be one of: auto, image, tabular"
        )

    raw_shape = spec.input_shape
    shape_source = "workflow.input_shape"
    if raw_shape is None:
        for container_name, container in (
            ("metadata_overrides", overrides),
            ("metadata", metadata),
        ):
            raw_shape = container.get("input_shape", container.get("sample_shape"))
            if raw_shape is not None:
                shape_source = container_name
                break
    configured_shape = _configured_conv_shape(spec)
    if raw_shape is None and configured_shape is not None:
        raw_shape = configured_shape
        shape_source = "configured_conv_adapter"
    input_shape = _normalize_image_shape(raw_shape)

    pixel_columns = _pixel_column_order(dataset.input_columns)
    n_features = len(dataset.input_columns)
    if data_type == "auto" and (input_shape is not None or configured_shape is not None):
        data_type = "image"
        source = shape_source
    if data_type == "auto" and pixel_columns is not None:
        side = int(np.sqrt(n_features))
        if side * side == n_features:
            data_type = "image"
            input_shape = (1, side, side)
            source = "contiguous_pixel_columns"
            shape_source = source
    if data_type == "auto":
        data_type = "tabular"
        source = "default_tabular"

    configured_conv = any(
        model.key.strip().lower() in _CONV_MODEL_ALIASES for model in spec.models
    )
    if data_type == "tabular" and configured_conv:
        raise ValueError(
            "A convolutional Mapper adapter conflicts with data_type='tabular'"
        )
    if data_type == "image" and input_shape is None:
        side = int(np.sqrt(n_features))
        if side * side == n_features:
            input_shape = (1, side, side)
            shape_source = "square_grayscale_feature_count"
        else:
            raise ValueError(
                "Image data requires input_shape=(C, H, W); it cannot be inferred "
                f"from {n_features} input features"
            )
    if data_type == "image":
        assert input_shape is not None
        expected_features = int(np.prod(input_shape))
        if expected_features != n_features:
            raise ValueError(
                f"input_shape={input_shape} contains {expected_features} values, "
                f"but Mapper selected {n_features} input columns"
            )

    return {
        "data_type": data_type,
        "input_shape": list(input_shape) if input_shape is not None else None,
        "flattened_order": "CHW" if data_type == "image" else None,
        "source": source,
        "shape_source": shape_source if input_shape is not None else None,
        "ordered_input_columns": pixel_columns if data_type == "image" else None,
    }


def _scaling_summary(scaler: Any) -> Dict[str, Any]:
    if isinstance(scaler, ImageDataScaler):
        return {
            "method": scaler.method_,
            "fit_split": "train",
            "data_min": scaler.data_min_,
            "data_max": scaler.data_max_,
            "clip": scaler.clip,
            "output_range": [0.0, 1.0],
        }
    return {
        "method": "robust",
        "fit_split": "train",
        "with_centering": scaler.with_centering,
        "with_scaling": scaler.with_scaling,
        "quantile_range": list(scaler.quantile_range),
    }


def _resolve_ladder_model(
    spec: SurrogateWorkflowSpec,
    rung: str,
    *,
    input_layout: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Resolve a rung's registry key and optional YAML model parameters."""
    configured: Optional[ModelConfig] = None
    for model in spec.models:
        if model.key.strip().lower() in _LADDER_ALIASES[rung]:
            configured = model
            break

    image_input = bool(input_layout and input_layout.get("data_type") == "image")
    model_key = (
        _CONV_LADDER_MODEL_KEYS[rung]
        if image_input and rung in _CONV_LADDER_MODEL_KEYS
        else _LADDER_MODEL_KEYS[rung]
    )
    model_name = rung.upper()
    params: Dict[str, Any] = {}
    if configured is not None:
        configured_key = configured.key.strip().lower()
        if (
            configured_key not in _GENERIC_RUNG_ALIASES[rung]
            and configured.key in MODEL_REGISTRY
        ):
            model_key = configured.key
        model_name = configured.name or model_name
        params.update(configured.params)
    if model_key in _CONV_MODEL_ALIASES:
        detected_shape = input_layout.get("input_shape") if input_layout else None
        params.setdefault("input_shape", detected_shape)
        if params["input_shape"] is None:
            raise ValueError(
                f"Mapper model {model_key!r} requires input_shape=(C, H, W)"
            )
        params["input_shape"] = _normalize_image_shape(params["input_shape"])
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


def _add_rmse_mae_ratio(
    metrics: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    """Add rmse/mae to a split's metrics; reporting-only, unused by the gate."""
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")
    metrics["rmse/mae"] = (
        float(rmse) / float(mae) if rmse is not None and mae else None
    )
    return metrics


def _extract_pca_explained_variance_ratio(
    adapter: Any,
) -> Optional[np.ndarray]:
    """Try to extract explained variance ratio from a fitted PCA adapter."""
    if adapter is None:
        return None

    model = getattr(adapter, "_model", None)
    if model is None:
        return None

    backend_model = getattr(model, "model", None)
    if backend_model is None:
        return None

    evr = getattr(backend_model, "explained_variance_ratio_", None)
    if evr is None:
        return None

    evr_arr = np.asarray(evr, dtype=np.float64).reshape(-1)
    if evr_arr.size == 0:
        return None
    return evr_arr


def _default_mapper_run_tag(file_path: Optional[Path]) -> str:
    prefix = file_path.stem if file_path else "dataset"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_mapper_{timestamp}"


__all__ = ["decide_clustering_action", "run_mapper_workflow"]
