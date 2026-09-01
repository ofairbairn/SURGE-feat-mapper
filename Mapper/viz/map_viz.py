"""
Mapper visualization module.

Owns the unsupervised-latent and reconstruction plot families (migrated from
``surge.viz.run_viz``) plus Mapper-native entry points that render from
in-memory arrays (``plot_mapper_latent``, ``plot_mapper_reconstruction``) so
the Mapper pipeline can emit plots straight into the run directory without
reloading the dataset, scalers, or models.
Outputs graphs and plots from the mapper workflow into the runs folder.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False

from surge.io.load_compat import load_model_compat

from Mapper.cluster.cluster import (
    _cluster_latent_embeddings,
    _compute_latent_quality_metrics,
)
from Mapper.tendency.tendency import _summarize_cluster_tendency

LOG = logging.getLogger(__name__)

# Default model display names (short labels for plots)
DEFAULT_MODEL_DISPLAY = {
    "random_forest_profiles": "Random Forest",
    "torch_mlp_mc_dropout": "MLP",
    "gpflow_gpr_profiles": "GPR",
    "xgc_mlp_aparallel": "MLP",
    "xgc_rf_aparallel": "Random Forest",
}

def _model_short_name(name: str) -> str:
    """Convert model filename to short display name."""
    return DEFAULT_MODEL_DISPLAY.get(name, name.replace("_", " ").title())

def _infer_label_values(
    dataset: Any,
    indices: np.ndarray,
    *,
    label_column: Optional[str] = None,
) -> Optional[np.ndarray]:
    df = getattr(dataset, "df", None)
    if df is None:
        return None

    candidate_columns: List[str] = []
    if label_column and label_column in df.columns:
        candidate_columns.append(label_column)
    candidate_columns.extend([c for c in getattr(dataset, "output_columns", []) if c in df.columns])
    for fallback in ("label", "labels", "class", "target", "target_label"):
        if fallback in df.columns and fallback not in candidate_columns:
            candidate_columns.append(fallback)

    if not candidate_columns:
        return None

    frame = df.iloc[indices]
    if len(candidate_columns) == 1:
        return frame[candidate_columns[0]].to_numpy()

    return frame[candidate_columns].astype(str).agg("|".join, axis=1).to_numpy()

def _compute_reconstruction_error(pred_df: pd.DataFrame) -> Optional[np.ndarray]:
    if "recon_error" in pred_df.columns:
        return pd.to_numeric(pred_df["recon_error"], errors="coerce").to_numpy(dtype=np.float64)

    y_true_cols = sorted([c for c in pred_df.columns if c.startswith("y_true_")])
    y_pred_cols = sorted([c for c in pred_df.columns if c.startswith("y_pred_")])
    if not y_true_cols or not y_pred_cols:
        if {"y_true", "y_pred"}.issubset(pred_df.columns):
            true_arr = pd.to_numeric(pred_df["y_true"], errors="coerce").to_numpy(dtype=np.float64)
            pred_arr = pd.to_numeric(pred_df["y_pred"], errors="coerce").to_numpy(dtype=np.float64)
            return np.abs(true_arr - pred_arr)
        return None

    n_pairs = min(len(y_true_cols), len(y_pred_cols))
    diffs: List[np.ndarray] = []
    for true_col, pred_col in zip(y_true_cols[:n_pairs], y_pred_cols[:n_pairs]):
        true_arr = pd.to_numeric(pred_df[true_col], errors="coerce").to_numpy(dtype=np.float64)
        pred_arr = pd.to_numeric(pred_df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
        diffs.append(np.square(true_arr - pred_arr))

    if not diffs:
        return None
    if len(diffs) == 1:
        return np.sqrt(diffs[0])
    return np.sqrt(np.mean(np.column_stack(diffs), axis=1))

def _save_tendency_heatmap(matrix: np.ndarray, *, out_png: Path, title: str) -> Optional[Path]:
    if matrix.size == 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Reordered sample index")
    ax.set_ylabel("Reordered sample index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png

def _build_latent_dataframe(
    *,
    sample_id: np.ndarray,
    embedding: np.ndarray,
    embedding_type: str,
    label: Optional[np.ndarray] = None,
    cluster: Optional[np.ndarray] = None,
    recon_error: Optional[np.ndarray] = None,
    anomaly_score: Optional[np.ndarray] = None,
    data_split: Optional[np.ndarray] = None,
    marginal_vendi: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "sample_id": np.asarray(sample_id, dtype=np.int64),
            "x": np.asarray(embedding[:, 0], dtype=np.float64),
            "y": np.asarray(embedding[:, 1], dtype=np.float64),
            "embedding_type": embedding_type,
        }
    )

    if label is not None:
        df["label"] = pd.Series(label, index=df.index)
    else:
        df["label"] = pd.NA

    if cluster is not None:
        df["cluster"] = pd.Series(cluster, index=df.index)
    else:
        df["cluster"] = pd.NA

    if recon_error is not None:
        recon_arr = np.asarray(recon_error, dtype=np.float64)
        df["recon_error"] = recon_arr
        valid = np.isfinite(recon_arr)
        recon_norm = np.full(recon_arr.shape, np.nan, dtype=np.float64)
        if np.any(valid):
            lo = float(np.nanmin(recon_arr[valid]))
            hi = float(np.nanmax(recon_arr[valid]))
            if hi > lo:
                recon_norm[valid] = (recon_arr[valid] - lo) / (hi - lo)
            else:
                recon_norm[valid] = 0.0
        df["recon_error_norm"] = recon_norm
    else:
        df["recon_error"] = pd.NA
        df["recon_error_norm"] = pd.NA

    if anomaly_score is not None:
        score_arr = np.asarray(anomaly_score, dtype=np.float64).reshape(-1)
        df["combined_anomaly_score"] = score_arr
    else:
        df["combined_anomaly_score"] = np.nan

    if data_split is not None:
        df["data_split"] = pd.Series(data_split, index=df.index, dtype="string")
    else:
        df["data_split"] = pd.NA

    if marginal_vendi is not None:
        df["marginal_vendi"] = np.asarray(
            marginal_vendi, dtype=np.float64
        ).reshape(-1)
    else:
        df["marginal_vendi"] = np.nan

    return df

def _compute_axis_limits(
    frame: pd.DataFrame,
    *,
    pad_fraction: float = 0.02,
) -> Optional[Tuple[float, float, float, float]]:
    """Axis ranges covering every point in ``frame`` so nothing is clipped."""
    x = pd.to_numeric(frame["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=np.float64)
    x_valid = x[np.isfinite(x)]
    y_valid = y[np.isfinite(y)]
    if x_valid.size == 0 or y_valid.size == 0:
        return None
    x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
    y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
    x_pad = max(x_max - x_min, 1e-12) * pad_fraction
    y_pad = max(y_max - y_min, 1e-12) * pad_fraction
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)

def _plot_latent_matplotlib(
    latent_df: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
    color_by: str = "label",
    color_scores: Optional[np.ndarray] = None,
    colorbar_label: str = "Combined anomaly score",
    category_colors: Optional[Dict[str, str]] = None,
    axis_limits: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    plot_df = latent_df.copy()

    score_values: Optional[np.ndarray] = None
    if color_scores is not None:
        candidate = np.asarray(color_scores, dtype=np.float64).reshape(-1)
        if candidate.shape[0] == len(plot_df):
            score_values = candidate

    fig, ax = plt.subplots(figsize=(8, 6))

    if score_values is not None:
        finite_scores = np.isfinite(score_values)
        if np.any(~finite_scores):
            ax.scatter(
                plot_df.loc[~finite_scores, "x"],
                plot_df.loc[~finite_scores, "y"],
                s=18,
                c="#bdbdbd",
                alpha=0.55,
                label="not available",
                edgecolors="none",
            )
        if np.any(finite_scores):
            im = ax.scatter(
                plot_df.loc[finite_scores, "x"],
                plot_df.loc[finite_scores, "y"],
                s=18,
                c=score_values[finite_scores],
                cmap="viridis",
                alpha=0.75,
                edgecolors="none",
            )
            colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            colorbar.set_label(colorbar_label)
        if np.any(~finite_scores):
            ax.legend(loc="best")
    else:
        if color_by not in plot_df.columns or plot_df[color_by].isna().all():
            color_by = "cluster" if "cluster" in plot_df.columns and not plot_df["cluster"].isna().all() else "label"

        color_values = plot_df[color_by].astype("string").fillna("NA")
        unique_values = list(pd.unique(color_values))
        if category_colors is not None:
            color_map = {
                value: category_colors.get(str(value), "#808080")
                for value in unique_values
            }
        else:
            cmap_name = (
                "tab10"
                if len(unique_values) <= 10
                else "tab20" if len(unique_values) <= 20 else "gist_rainbow"
            )
            try:
                cmap_obj = matplotlib.colormaps.get_cmap(cmap_name).resampled(
                    max(len(unique_values), 1)
                )
            except AttributeError:
                cmap_obj = plt.get_cmap(cmap_name, max(len(unique_values), 1))
            color_map = {
                value: (
                    "#808080"
                    if color_by == "cluster" and str(value) == "-1"
                    else cmap_obj(i)
                )
                for i, value in enumerate(unique_values)
            }

        for value in unique_values:
            mask = color_values == value
            ax.scatter(
                plot_df.loc[mask, "x"],
                plot_df.loc[mask, "y"],
                s=18,
                c=[color_map[value]],
                alpha=0.75,
                label=str(value),
                edgecolors="none",
            )
        ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    if axis_limits is None and "cluster" in plot_df.columns:
        centroid_rows = plot_df[
            plot_df["cluster"].notna() & (plot_df["cluster"].astype("string") != "-1")
        ]
        if not centroid_rows.empty:
            centroids = centroid_rows.groupby(centroid_rows["cluster"].astype("string"))[ ["x", "y"] ].mean()
            for cluster_id, row in centroids.iterrows():
                ax.text(
                    row["x"],
                    row["y"],
                    str(cluster_id),
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
                )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_latent_dendrogram(
    latent: np.ndarray,
    *,
    title: str,
    out_png: Path,
    random_state: int = 42,
    max_points: int = 1000,
    linkage_method: str = "ward",
) -> Optional[Path]:
    if latent.shape[0] < 2:
        return None

    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
    except ImportError:
        return None

    latent_plot = latent
    if latent.shape[0] > max_points:
        rng = np.random.default_rng(random_state)
        sample_idx = np.sort(rng.choice(latent.shape[0], size=max_points, replace=False))
        latent_plot = latent[sample_idx]

    linkage_matrix = linkage(latent_plot, method=linkage_method)
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, ax=ax, no_labels=True, color_threshold=None)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _plot_latent_interactive(
    latent_df: pd.DataFrame,
    *,
    title: str,
    out_html: Path,
    color_by: str = "label",
    random_state: int = 42,
    hover_sample_frac: float = 0.02,
    datashader_threshold: int = 50_000,
) -> Optional[Path]:
    try:
        import holoviews as hv
        import holoviews.operation.datashader as hd
        import datashader as ds
    except ImportError:
        LOG.info("Skipping interactive latent plot because holoviews/datashader are unavailable.")
        return None

    hv.extension("bokeh")

    plot_df = latent_df.copy()
    if color_by not in plot_df.columns or plot_df[color_by].isna().all():
        color_by = "cluster" if "cluster" in plot_df.columns and not plot_df["cluster"].isna().all() else "label"

    if color_by in plot_df.columns:
        plot_df[color_by] = plot_df[color_by].astype("string").fillna("NA").astype("category")
    if "cluster" in plot_df.columns:
        plot_df["cluster"] = plot_df["cluster"].astype("string").fillna("NA")
    if "label" in plot_df.columns:
        plot_df["label"] = plot_df["label"].astype("string").fillna("NA")

    hover_cols = [
        "sample_id",
        "label",
        "cluster",
        "data_split",
        "combined_anomaly_score",
        "marginal_vendi",
        "recon_error",
        "recon_error_norm",
        "embedding_type",
    ]
    hover_cols = [col for col in hover_cols if col in plot_df.columns]
    hover_tooltips = []
    for col in hover_cols:
        if col in {
            "combined_anomaly_score",
            "marginal_vendi",
            "recon_error",
            "recon_error_norm",
        }:
            hover_tooltips.append((col, f"@{{{col}}}{{0.000}}"))
        else:
            hover_tooltips.append((col, f"@{{{col}}}"))
    n_categories = int(plot_df[color_by].nunique())
    if n_categories <= 10:
        cmap_name: Any = "Category10"
    elif n_categories <= 20:
        cmap_name = "Category20"
    else:
        from matplotlib import colormaps
        from matplotlib.colors import to_hex

        dynamic_cmap = colormaps.get_cmap("gist_rainbow").resampled(n_categories)
        cmap_name = [to_hex(dynamic_cmap(i)) for i in range(n_categories)]

    points = hv.Points(plot_df, kdims=["x", "y"], vdims=hover_cols)
    if len(plot_df) > datashader_threshold:
        background = hd.datashade(points, aggregator=ds.count_cat(color_by), cmap=cmap_name, how="eq_hist")
        hover_n = min(len(plot_df), max(1000, int(len(plot_df) * hover_sample_frac)))
        sampled = plot_df.sample(n=hover_n, random_state=random_state) if hover_n < len(plot_df) else plot_df
        hover_points = hv.Points(sampled, kdims=["x", "y"], vdims=hover_cols).opts(
            size=6,
            color=color_by,
            cmap=cmap_name,
            tools=["hover"],
            hover_tooltips=hover_tooltips,
            alpha=0.85,
            line_color="white",
            line_alpha=0.2,
        )
        overlay = background * hover_points
    else:
        overlay = points.opts(
            size=6,
            color=color_by,
            cmap=cmap_name,
            tools=["hover"],
            hover_tooltips=hover_tooltips,
            alpha=0.85,
            line_color="white",
            line_alpha=0.2,
        )

    centroid_rows = plot_df[
        plot_df["cluster"].notna() & (plot_df["cluster"] != "-1")
    ]
    if not centroid_rows.empty:
        centroid_df = centroid_rows.groupby("cluster", as_index=False)[["x", "y"]].mean()
        labels = hv.Labels(centroid_df, kdims=["x", "y"], vdims=["cluster"]).opts(
            text_font_size="10pt",
            text_color="black",
        )
        overlay = overlay * labels

    overlay = overlay.opts(
        width=900,
        height=650,
        title=title,
        toolbar="above",
        show_grid=True,
        legend_position="right",
    )

    hv.save(overlay, str(out_html), backend="bokeh")
    return out_html


def _fit_2d_embeddings(
    Z: np.ndarray,
    *,
    random_state: int = 42,
    include_tsne: bool = True,
) -> List[Tuple[str, np.ndarray]]:
    """Compute UMAP and t-SNE 2-D projections of a latent matrix.

    When ``include_tsne`` is False the t-SNE projection is skipped entirely
    (used when the Hopkins tendency gate finds no clusterable structure).
    """
    Z = np.asarray(Z, dtype=np.float64)
    embeddings: List[Tuple[str, np.ndarray]] = []

    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=random_state,
            n_jobs=1,
        )
        embeddings.append(("umap", np.asarray(reducer.fit_transform(Z), dtype=np.float64)))
    except ImportError:
        LOG.warning("umap-learn is not installed; skipping UMAP.")
    except Exception as exc:
        LOG.warning("UMAP failed: %s", exc)

    if include_tsne:
        try:
            from sklearn.manifold import TSNE

            perplexity = float(min(30, max(5, Z.shape[0] // 10)))
            tsne_kwargs = {
                "n_components": 2,
                "perplexity": perplexity,
                "learning_rate": 200,
                "random_state": random_state,
                "init": "pca",
            }
            try:
                tsne = TSNE(max_iter=1000, **tsne_kwargs)
            except TypeError:
                tsne = TSNE(n_iter=1000, **tsne_kwargs)
            embeddings.append(("tsne", np.asarray(tsne.fit_transform(Z), dtype=np.float64)))
        except Exception as exc:
            LOG.warning("t-SNE failed: %s", exc)

    return embeddings


def _save_embedding(
    Z: np.ndarray,
    embedding: np.ndarray,
    *,
    sample_indices: np.ndarray,
    label_values: Optional[np.ndarray],
    cluster_labels: Optional[np.ndarray],
    recon_error: Optional[np.ndarray],
    anomaly_score: Optional[np.ndarray],
    data_split: Optional[np.ndarray],
    marginal_vendi: Optional[np.ndarray],
    model_name: str,
    split: str,
    embedding_type: str,
    title: str,
    output_dir: Path,
    color_by: str,
    random_state: int,
    anomaly_quantile: float,
    interactive_threshold: int,
    interactive_hover_sample_frac: float,
    cluster_method: str,
    tendency_summary: Optional[Dict[str, Any]],
    latent_quality_n_neighbors: int,
) -> List[str]:
    """Build, serialize, and plot one 2-D embedding of a latent space.

    UMAP renders the whole dataset in one plot. t-SNE renders one zoomed-in
    plot per cluster found by the clustering step (axes cover every point in
    that cluster); only when t-SNE was requested but clustering produced no
    usable clusters does it fall back to a single whole-dataset plot. Points
    are a fixed size and colored by the combined anomaly score from
    ``Mapper.anomaly`` when available.
    """
    saved: List[str] = []
    embedding_df = _build_latent_dataframe(
        sample_id=sample_indices,
        embedding=embedding,
        embedding_type=embedding_type,
        label=label_values,
        cluster=cluster_labels,
        recon_error=recon_error,
        anomaly_score=anomaly_score,
        data_split=data_split,
        marginal_vendi=marginal_vendi,
    )
    embedding_df["model_name"] = model_name
    embedding_df["split"] = split

    combined_scores: Optional[np.ndarray] = None
    if "combined_anomaly_score" in embedding_df.columns:
        candidate = pd.to_numeric(
            embedding_df["combined_anomaly_score"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if np.any(np.isfinite(candidate)):
            combined_scores = candidate

    score_source: Optional[np.ndarray] = None
    if combined_scores is not None:
        score_source = combined_scores
    elif recon_error is not None:
        score_source = embedding_df["recon_error_norm"].to_numpy(dtype=np.float64)

    if score_source is not None:
        valid_score = np.isfinite(score_source)
        threshold = (
            float(np.nanquantile(score_source[valid_score], anomaly_quantile))
            if np.any(valid_score)
            else np.nan
        )
        embedding_df["anomaly_score"] = score_source
        embedding_df["anomaly_flag"] = (
            score_source >= threshold
        ) if np.isfinite(threshold) else False
        embedding_df["anomaly_threshold"] = threshold

    quality_payload = {
        "model_name": model_name,
        "split": split,
        "embedding_type": embedding_type,
        "cluster_method": cluster_method,
        "cluster_tendency": {
            key: value
            for key, value in (tendency_summary or {}).items()
            if key not in {"vat_matrix", "ivat_matrix"}
        },
        "quality": _compute_latent_quality_metrics(
            Z,
            embedding,
            cluster_labels,
            n_neighbors=latent_quality_n_neighbors,
        ),
    }

    safe_model_name = model_name.replace(" ", "_")
    parquet_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}.parquet"
    embedding_df.to_parquet(parquet_path, index=False)
    saved.append(str(parquet_path))

    quality_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}_quality.json"
    with quality_path.open("w", encoding="utf-8") as handle:
        json.dump(quality_payload, handle, indent=2)
    saved.append(str(quality_path))

    cluster_ids: List[int] = []
    member_masks: Dict[int, np.ndarray] = {}
    if cluster_labels is not None:
        cluster_arr = np.asarray(cluster_labels).reshape(-1)
        if cluster_arr.shape[0] == len(embedding_df):
            for value in np.unique(cluster_arr):
                try:
                    cluster_id = int(value)
                except (TypeError, ValueError):
                    continue
                if cluster_id == -1:
                    continue
                cluster_ids.append(cluster_id)
                member_masks[cluster_id] = cluster_arr == cluster_id
            cluster_ids.sort()

    if embedding_type == "tsne" and cluster_ids:
        for cluster_number, cluster_id in enumerate(cluster_ids, start=1):
            member_mask = member_masks[cluster_id]
            cluster_df = embedding_df.loc[member_mask]
            cluster_scores = (
                combined_scores[member_mask] if combined_scores is not None else None
            )
            cluster_title = (
                f"{_model_short_name(model_name)} cluster {cluster_number} t-SNE"
            )
            cluster_png = output_dir / (
                f"latent_{safe_model_name}_{split}_{embedding_type}_cluster_{cluster_number}.png"
            )
            _plot_latent_matplotlib(
                cluster_df,
                title=cluster_title,
                out_png=cluster_png,
                color_scores=cluster_scores,
                axis_limits=_compute_axis_limits(cluster_df),
            )
            saved.append(str(cluster_png))

    elif embedding_type == "umap":
        umap_variants = (
            (
                "anomaly",
                "Combined anomaly score",
                None,
                embedding_df["combined_anomaly_score"].to_numpy(dtype=np.float64),
                "Combined anomaly score",
                None,
            ),
            (
                "cluster",
                "Detected cluster",
                "cluster",
                None,
                "",
                None,
            ),
            (
                "split",
                "Dataset split",
                "data_split",
                None,
                "",
                {"train": "#1f77b4", "validation": "#d62728", "test": "#ffd700"},
            ),
            (
                "marginal_vendi",
                "Marginal Vendi contribution",
                None,
                embedding_df["marginal_vendi"].to_numpy(dtype=np.float64),
                "Marginal Vendi contribution",
                None,
            ),
        )
        for (
            suffix,
            title_detail,
            variant_color_by,
            scores,
            score_label,
            palette,
        ) in umap_variants:
            png_path = output_dir / (
                f"latent_{safe_model_name}_{split}_{embedding_type}_{suffix}.png"
            )
            _plot_latent_matplotlib(
                embedding_df,
                title=f"{title} - {title_detail}",
                out_png=png_path,
                color_by=variant_color_by or color_by,
                color_scores=scores,
                colorbar_label=score_label,
                category_colors=palette,
            )
            saved.append(str(png_path))

        # Keep one interactive Bokeh artifact for UMAP only, colored by the
        # dynamically sized cluster palette. t-SNE never emits HTML.
        if not embedding_df["cluster"].isna().all():
            html_path = output_dir / (
                f"latent_{safe_model_name}_{split}_{embedding_type}_cluster.html"
            )
            html_saved = _plot_latent_interactive(
                embedding_df,
                title=f"{title} - Detected cluster",
                out_html=html_path,
                color_by="cluster",
                random_state=random_state,
                hover_sample_frac=interactive_hover_sample_frac,
                datashader_threshold=interactive_threshold,
            )
            if html_saved is not None:
                saved.append(str(html_saved))
    else:
        png_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}.png"
        _plot_latent_matplotlib(
            embedding_df,
            title=title,
            out_png=png_path,
            color_by=color_by,
            color_scores=combined_scores,
        )
        saved.append(str(png_path))
    return saved


def viz_unsupervised_latent(
    run_dir: Path,
    output_dir: Path,
    *,
    split_preference: Tuple[str, ...] = ("val", "train", "test"),
    random_state: int = 42,
    color_by: str = "label",
    cluster_method: str = "hdbscan",
    n_clusters: Optional[int] = None,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    agglomerative_linkage: str = "ward",
    apply_tendency_gate: bool = True,
    hopkins_threshold: float = 0.55,
    tendency_sample_size: Optional[int] = None,
    save_tendency_artifacts: bool = True,
    interactive_threshold: int = 50_000,
    interactive_hover_sample_frac: float = 0.02,
    anomaly_quantile: float = 0.95,
    label_column: Optional[str] = None,
    latent_quality_n_neighbors: int = 10,
) -> List[str]:
    """Generate latent-space UMAP and t-SNE plots for unsupervised runs.

    Uses adapter.encode(X) when available and writes one UMAP and one t-SNE
    plot per model using the first available split from split_preference.
    Clustering is computed on latent vectors (Z) before UMAP/t-SNE projection.
    """
    from surge.dataset import SurrogateDataset
    from surge.model.registry import MODEL_REGISTRY
    from surge.workflow.spec import SurrogateWorkflowSpec

    if not YAML_AVAILABLE:
        LOG.warning("PyYAML required for unsupervised latent visualization.")
        return []

    spec_file = run_dir / "spec.yaml"
    summary_file = run_dir / "workflow_summary.json"
    if not spec_file.exists() or not summary_file.exists():
        LOG.warning("Missing spec.yaml or workflow_summary.json in %s", run_dir)
        return []

    try:
        with spec_file.open("r", encoding="utf-8") as f:
            spec_dict = yaml.safe_load(f) or {}
        spec = SurrogateWorkflowSpec.from_dict(spec_dict)
    except Exception as exc:
        LOG.warning("Could not load spec for latent visualization: %s", exc)
        return []

    try:
        with summary_file.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as exc:
        LOG.warning("Could not load workflow summary for latent visualization: %s", exc)
        return []

    dataset_path = Path(summary.get("dataset", {}).get("file_path", spec.dataset_path))
    if not dataset_path.exists():
        LOG.warning("Dataset path %s not found. Skip latent visualization.", dataset_path)
        return []

    try:
        dataset = SurrogateDataset.from_path(
            dataset_path,
            format=spec.dataset_format,
            metadata_path=spec.metadata_path,
            sample=spec.sample_rows,
            analyzer_kwargs={"hints": spec.metadata_overrides, **spec.analyzer},
        )
    except Exception as exc:
        LOG.warning("Could not load dataset for latent visualization: %s", exc)
        return []

    if dataset.df is None or not dataset.input_columns:
        LOG.warning("Dataset has no usable input columns. Skip latent visualization.")
        return []

    input_scaler = None
    scalers_dir = run_dir / "scalers"
    if spec.standardize_inputs and (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib is required for latent visualization plots.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    def _resolve_encoder(model_obj: Any) -> Optional[Any]:
        """Return a callable encoder if available on adapter or wrapped model."""
        if hasattr(model_obj, "encode") and callable(getattr(model_obj, "encode")):
            return getattr(model_obj, "encode")
        inner = getattr(model_obj, "_model", None)
        if inner is not None and hasattr(inner, "encode") and callable(getattr(inner, "encode")):
            return getattr(inner, "encode")
        return None

    model_entries = summary.get("models", [])
    for model_entry in model_entries:
        model_name = model_entry.get("name", "model")
        model_key = model_entry.get("key")
        if not model_key:
            model_key = (
                ((model_entry.get("resources_used") or {}).get("banner") or {}).get("model")
            )
        artifacts = model_entry.get("artifacts", {})
        prediction_files = artifacts.get("predictions", {})

        split = None
        pred_path = None
        for candidate in split_preference:
            candidate_path = prediction_files.get(candidate)
            if candidate_path:
                split = candidate
                pred_path = Path(candidate_path)
                break
        if split is None or pred_path is None:
            LOG.warning("No prediction split found for %s; skipping latent viz.", model_name)
            continue

        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        if not pred_path.exists():
            for ext in (".parquet", ".csv"):
                alt = run_dir / "predictions" / f"{model_name}_{split}{ext}"
                if alt.exists():
                    pred_path = alt
                    break
        if not pred_path.exists():
            LOG.warning("Prediction file not found for %s %s.", model_name, split)
            continue

        if pred_path.suffix == ".parquet":
            pred_df = pd.read_parquet(pred_path)
        else:
            pred_df = pd.read_csv(pred_path)

        if "index" in pred_df.columns:
            indices = pred_df["index"].to_numpy(dtype=np.int64)
        else:
            LOG.warning(
                "Predictions for %s %s have no index column; using row numbers as sample_id.",
                model_name,
                split,
            )
            indices = np.arange(len(pred_df), dtype=np.int64)
        if indices.size == 0:
            LOG.warning("Predictions for %s %s are empty; skipping.", model_name, split)
            continue

        if dataset.df is None:
            LOG.warning("Dataset frame unavailable for %s; skipping.", model_name)
            continue

        valid_mask = (indices >= 0) & (indices < len(dataset.df))
        valid_indices = indices[valid_mask]
        if valid_indices.size == 0:
            LOG.warning("No valid indices found for %s %s; skipping.", model_name, split)
            continue

        valid_pred_df = pred_df.iloc[np.flatnonzero(valid_mask)]
        X_raw = dataset.df[dataset.input_columns].iloc[valid_indices].values.astype(np.float64)
        X = input_scaler.transform(X_raw) if input_scaler is not None else X_raw
        label_values = _infer_label_values(dataset, valid_indices, label_column=label_column)
        recon_error = _compute_reconstruction_error(valid_pred_df)
        if recon_error is not None and len(recon_error) != len(valid_indices):
            recon_error = recon_error[: len(valid_indices)]

        model_path = Path(artifacts.get("model", ""))
        if not model_path.is_absolute():
            model_path = run_dir / model_path
        if not model_path.exists():
            model_path = run_dir / "models" / Path(str(model_path)).name
        if not model_path.exists():
            LOG.warning("Model artifact not found for %s.", model_name)
            continue

        adapter = None
        encoder = None
        # Prefer direct joblib load first: for many SURGE adapters this keeps
        # richer methods like encode(), while compatibility wrappers can be
        # intentionally minimal (predict-only).
        try:
            raw_obj = joblib.load(model_path)
            encoder = _resolve_encoder(raw_obj)
            if encoder is not None:
                adapter = raw_obj
        except Exception:
            adapter = None

        if adapter is None:
            try:
                adapter = load_model_compat(model_path, model_entry)
                encoder = _resolve_encoder(adapter)
            except Exception as exc:
                LOG.warning("Could not load model %s for latent viz: %s", model_name, exc)
                continue

        if encoder is None and model_key and model_key in MODEL_REGISTRY:
            # Last-resort path: rebuild adapter by registry key, then call load().
            try:
                reg_adapter = MODEL_REGISTRY.create(model_key)
                reg_adapter.load(model_path)
                encoder = _resolve_encoder(reg_adapter)
                if encoder is not None:
                    adapter = reg_adapter
            except Exception as exc:
                LOG.warning(
                    "Registry-based reload failed for %s (%s): %s",
                    model_name,
                    model_key,
                    exc,
                )

        if encoder is None:
            LOG.warning("Model %s has no encode() method; skipping latent viz.", model_name)
            continue

        try:
            Z = np.asarray(encoder(X), dtype=np.float64)
        except Exception as exc:
            LOG.warning("Encoding failed for %s: %s", model_name, exc)
            continue
        if Z.ndim != 2 or Z.shape[0] == 0:
            LOG.warning("Encoded latent matrix invalid for %s: shape=%s", model_name, Z.shape)
            continue

        tendency_summary = _summarize_cluster_tendency(
            Z,
            hopkins_threshold=hopkins_threshold,
            sample_size=tendency_sample_size,
            random_state=random_state,
        )

        safe_model_name = model_name.replace(" ", "_")
        emit_learned = True

        cluster_labels = None
        linkage_matrix = None
        if emit_learned:
            if save_tendency_artifacts:
                tendency_base = output_dir / f"latent_{model_name.replace(' ', '_')}_{split}_tendency"
                tendency_payload = {
                    key: value
                    for key, value in tendency_summary.items()
                    if key not in {"vat_matrix", "ivat_matrix"}
                }
                tendency_json = tendency_base.with_suffix(".json")
                tendency_json.write_text(json.dumps(tendency_payload, indent=2), encoding="utf-8")
                saved_paths.append(str(tendency_json))

                vat_png = _save_tendency_heatmap(
                    np.asarray(tendency_summary["vat_matrix"], dtype=np.float64),
                    out_png=output_dir / f"latent_{model_name.replace(' ', '_')}_{split}_vat.png",
                    title=f"{_model_short_name(model_name)} {split} VAT",
                )
                if vat_png is not None:
                    saved_paths.append(str(vat_png))

                ivat_png = _save_tendency_heatmap(
                    np.asarray(tendency_summary["ivat_matrix"], dtype=np.float64),
                    out_png=output_dir / f"latent_{model_name.replace(' ', '_')}_{split}_ivat.png",
                    title=f"{_model_short_name(model_name)} {split} iVAT",
                )
                if ivat_png is not None:
                    saved_paths.append(str(ivat_png))

            if apply_tendency_gate and not tendency_summary.get("gate_passed", True):
                cluster_labels = np.full(len(valid_indices), -1, dtype=int)
                linkage_matrix = None
                LOG.info(
                    "Skipping clustering for %s %s because Hopkins=%.4f is below threshold %.4f.",
                    model_name,
                    split,
                    float(tendency_summary.get("hopkins") or float("nan")),
                    float(tendency_summary.get("hopkins_threshold") or hopkins_threshold),
                )
            else:
                try:
                    cluster_labels, linkage_matrix = _cluster_latent_embeddings(
                        Z,
                        method=cluster_method,
                        n_clusters=n_clusters,
                        random_state=random_state,
                        dbscan_eps=dbscan_eps,
                        dbscan_min_samples=dbscan_min_samples,
                        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                        hdbscan_min_samples=hdbscan_min_samples,
                        agglomerative_linkage=agglomerative_linkage,
                    )
                except Exception as exc:
                    LOG.warning("Clustering failed for %s %s: %s", model_name, split, exc)
                    cluster_labels = np.full(len(valid_indices), -1, dtype=int)
                    linkage_matrix = None

            if cluster_method.lower().strip() == "agglomerative" and linkage_matrix is not None:
                dendrogram_path = output_dir / f"latent_{safe_model_name}_{split}_dendrogram.png"
                dendrogram_saved = _plot_latent_dendrogram(
                    Z,
                    title=f"{_model_short_name(model_name)} {split} latent dendrogram",
                    out_png=dendrogram_path,
                    random_state=random_state,
                    linkage_method=agglomerative_linkage,
                )
                if dendrogram_saved is not None:
                    saved_paths.append(str(dendrogram_saved))

        if emit_learned:
            tsne_allowed = (
                True
                if not apply_tendency_gate
                else bool(tendency_summary.get("gate_passed", True))
            )
            for embedding_type, embedding in _fit_2d_embeddings(
                Z,
                random_state=random_state,
                include_tsne=tsne_allowed,
            ):
                saved_paths.extend(
                    _save_embedding(
                        Z,
                        embedding,
                        sample_indices=valid_indices,
                        label_values=label_values,
                        cluster_labels=cluster_labels,
                        recon_error=recon_error,
                        anomaly_score=None,
                        data_split=np.full(len(valid_indices), split, dtype=object),
                        marginal_vendi=None,
                        model_name=model_name,
                        split=split,
                        embedding_type=embedding_type,
                        title=f"{_model_short_name(model_name)} latent {embedding_type.upper()}",
                        output_dir=output_dir,
                        color_by=color_by,
                        random_state=random_state,
                        anomaly_quantile=anomaly_quantile,
                        interactive_threshold=interactive_threshold,
                        interactive_hover_sample_frac=interactive_hover_sample_frac,
                        cluster_method=cluster_method,
                        tendency_summary=tendency_summary,
                        latent_quality_n_neighbors=latent_quality_n_neighbors,
                    )
                )

    return saved_paths


def _reconstruction_diagnostics(
    df: pd.DataFrame,
    out_prefix: Path,
    *,
    top_n_features: int = 6,
    worst_n_samples: int = 25,
    anomaly_quantile: float = 0.95,
    include_image_panels: bool = True,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> List[str]:
    """Emit reconstruction diagnostics for one prediction table."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib is required for unsupervised reconstruction visualization.")
        return []

    y_true_cols = sorted([c for c in df.columns if c.startswith("y_true_")])
    y_pred_cols = sorted([c for c in df.columns if c.startswith("y_pred_")])
    if not y_true_cols or not y_pred_cols:
        return []

    n_cols = min(len(y_true_cols), len(y_pred_cols))
    y_true = df[y_true_cols[:n_cols]].to_numpy(dtype=np.float64)
    y_pred = df[y_pred_cols[:n_cols]].to_numpy(dtype=np.float64)
    residual = y_pred - y_true
    abs_residual = np.abs(residual)
    sample_error = np.sqrt(np.mean(np.square(residual), axis=1))

    if len(sample_error) == 0:
        return []

    saved_paths: List[str] = []

    feature_mse = np.mean(np.square(residual), axis=0)
    feature_mae = np.mean(np.abs(residual), axis=0)
    feature_rmse = np.sqrt(feature_mse)
    feature_corr = []
    feature_r2 = []
    for i in range(n_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if np.std(yt) > 0 and np.std(yp) > 0:
            feature_corr.append(float(np.corrcoef(yt, yp)[0, 1]))
        else:
            feature_corr.append(None)
        try:
            from sklearn.metrics import r2_score

            feature_r2.append(float(r2_score(yt, yp)))
        except Exception:
            feature_r2.append(None)

    summary_df = pd.DataFrame(
        {
            "feature": [f"f_{i:02d}" for i in range(n_cols)],
            "mse": feature_mse,
            "mae": feature_mae,
            "rmse": feature_rmse,
            "corr": feature_corr,
            "r2": feature_r2,
        }
    )
    summary_df = summary_df.sort_values("rmse", ascending=False)

    summary_parquet = out_prefix.with_name(out_prefix.name + "_feature_summary.parquet")
    summary_json = out_prefix.with_name(out_prefix.name + "_feature_summary.json")
    summary_df.to_parquet(summary_parquet, index=False)
    summary_json.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    saved_paths.extend([str(summary_parquet), str(summary_json)])

    anomaly_threshold = float(np.quantile(sample_error, anomaly_quantile))
    anomaly_df = pd.DataFrame(
        {
            "sample_id": df["index"].to_numpy(dtype=np.int64)
            if "index" in df.columns
            else np.arange(len(df), dtype=np.int64),
            "sample_recon_error": sample_error,
            "anomaly_score": (sample_error - sample_error.min())
            / max(1e-12, sample_error.max() - sample_error.min()),
            "anomaly_flag": sample_error >= anomaly_threshold,
        }
    )
    anomaly_parquet = out_prefix.with_name(out_prefix.name + "_anomaly_scores.parquet")
    anomaly_json = out_prefix.with_name(out_prefix.name + "_anomaly_scores.json")
    anomaly_df.to_parquet(anomaly_parquet, index=False)
    anomaly_json.write_text(anomaly_df.to_json(orient="records", indent=2), encoding="utf-8")
    saved_paths.extend([str(anomaly_parquet), str(anomaly_json)])

    top_idx = np.argsort(feature_rmse)[::-1][: max(1, min(top_n_features, n_cols))]

    # Feature parity plots.
    n_plot = len(top_idx)
    fig, axes = plt.subplots(n_plot, 1, figsize=(7, 3 * n_plot), squeeze=False)
    for ax, idx in zip(axes.ravel(), top_idx):
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=10, alpha=0.6)
        lo = min(float(np.min(y_true[:, idx])), float(np.min(y_pred[:, idx])))
        hi = max(float(np.max(y_true[:, idx])), float(np.max(y_pred[:, idx])))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title(f"Feature f_{idx:02d} parity")
        ax.set_xlabel("Original")
        ax.set_ylabel("Reconstructed")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    parity_path = out_prefix.with_name(out_prefix.name + "_feature_parity.png")
    fig.savefig(parity_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(parity_path))

    # Residual histograms.
    fig, axes = plt.subplots(n_plot, 1, figsize=(7, 3 * n_plot), squeeze=False)
    for ax, idx in zip(axes.ravel(), top_idx):
        ax.hist(residual[:, idx], bins=40, alpha=0.8)
        ax.axvline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"Feature f_{idx:02d} residual histogram")
        ax.set_xlabel("x_hat - x")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    hist_path = out_prefix.with_name(out_prefix.name + "_residual_histograms.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(hist_path))

    # Sample-level reconstruction error distribution.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sample_error, bins=50, alpha=0.85)
    ax.axvline(anomaly_threshold, linestyle="--", linewidth=1.5, color="red")
    ax.set_title("Sample-level reconstruction error distribution")
    ax.set_xlabel("RMSE per sample")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    err_dist_path = out_prefix.with_name(out_prefix.name + "_sample_error_distribution.png")
    fig.tight_layout()
    fig.savefig(err_dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(err_dist_path))

    # Worst-sample residual heatmap.
    worst_n = max(1, min(worst_n_samples, len(sample_error)))
    worst_idx = np.argsort(sample_error)[-worst_n:]
    heat = abs_residual[worst_idx]
    fig, ax = plt.subplots(figsize=(max(7, n_cols * 0.4), max(4, worst_n * 0.2)))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    ax.set_title("Absolute residual heatmap (worst reconstructed samples)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Worst samples")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    heatmap_path = out_prefix.with_name(out_prefix.name + "_worst_samples_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(heatmap_path))

    if include_image_panels:
        shape = image_shape
        if shape is None:
            side = int(np.sqrt(n_cols))
            if side * side == n_cols:
                shape = (side, side)
        if shape is not None and int(np.prod(shape)) == n_cols:
            panel_n = min(6, worst_n)
            panel_idx = worst_idx[-panel_n:]
            fig, axes = plt.subplots(panel_n, 3, figsize=(9, 2.5 * panel_n), squeeze=False)
            for row_i, sample_i in enumerate(panel_idx):
                orig = y_true[sample_i].reshape(shape)
                recon = y_pred[sample_i].reshape(shape)
                diff = np.abs(recon - orig)
                for col_i, arr in enumerate((orig, recon, diff)):
                    ax = axes[row_i, col_i]
                    im = ax.imshow(arr, cmap="viridis", aspect="auto")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row_i == 0:
                        ax.set_title(("Original", "Reconstructed", "Abs Residual")[col_i])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.tight_layout()
            panel_path = out_prefix.with_name(out_prefix.name + "_image_panels.png")
            fig.savefig(panel_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(str(panel_path))

    return saved_paths


def viz_unsupervised_reconstruction(
    run_dir: Path,
    output_dir: Path,
    *,
    split_preference: Tuple[str, ...] = ("val", "train", "test"),
    top_n_features: int = 6,
    worst_n_samples: int = 25,
    anomaly_quantile: float = 0.95,
    include_image_panels: bool = True,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> List[str]:
    """Visual diagnostics comparing original and reconstructed unsupervised outputs."""
    summary_file = run_dir / "workflow_summary.json"
    if not summary_file.exists():
        return []

    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    saved_paths: List[str] = []
    for model_entry in summary.get("models", []):
        model_name = model_entry.get("name", "model")
        artifacts = model_entry.get("artifacts", {})
        pred_map = artifacts.get("predictions", {})

        split = None
        pred_path = None
        for candidate in split_preference:
            candidate_path = pred_map.get(candidate)
            if candidate_path:
                split = candidate
                pred_path = Path(candidate_path)
                break
        if split is None or pred_path is None:
            continue

        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        if not pred_path.exists():
            for ext in (".parquet", ".csv"):
                alt = run_dir / "predictions" / f"{model_name}_{split}{ext}"
                if alt.exists():
                    pred_path = alt
                    break
        if not pred_path.exists():
            continue

        if pred_path.suffix == ".parquet":
            df = pd.read_parquet(pred_path)
        else:
            df = pd.read_csv(pred_path)

        model_key_safe = model_name.replace(" ", "_")
        out_prefix = output_dir / f"reconstruction_{model_key_safe}_{split}"
        saved_paths.extend(
            _reconstruction_diagnostics(
                df,
                out_prefix,
                top_n_features=top_n_features,
                worst_n_samples=worst_n_samples,
                anomaly_quantile=anomaly_quantile,
                include_image_panels=include_image_panels,
                image_shape=image_shape,
            )
        )

    return saved_paths


def plot_mapper_latent(
    Z: np.ndarray,
    *,
    sample_indices: np.ndarray,
    output_dir: Path,
    model_name: str = "mapper",
    split: str = "train_val_test",
    cluster_labels: Optional[np.ndarray] = None,
    recon_error: Optional[np.ndarray] = None,
    anomaly_score: Optional[np.ndarray] = None,
    data_split: Optional[np.ndarray] = None,
    marginal_vendi: Optional[np.ndarray] = None,
    cluster_method: str = "hdbscan",
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    random_state: int = 42,
    color_by: str = "cluster",
    apply_tendency_gate: bool = True,
    include_tsne: bool = True,
    tendency_summary: Optional[Dict[str, Any]] = None,
    hopkins_threshold: float = 0.55,
    tendency_sample_size: Optional[int] = None,
    save_tendency_artifacts: bool = True,
    interactive_threshold: int = 50_000,
    interactive_hover_sample_frac: float = 0.02,
    anomaly_quantile: float = 0.95,
    latent_quality_n_neighbors: int = 10,
) -> Dict[str, Any]:
    """Plot a Mapper latent space directly from in-memory vectors.

    Unlike ``viz_unsupervised_latent`` (which reloads dataset/scalers/models
    from a run directory), this renders from the arrays the Mapper pipeline
    already holds, so the plots land in the run's output folder at run end.
    ``anomaly_score`` is the per-sample combined anomaly score from
    ``Mapper.anomaly.run_anomaly_detection``; points are colored by it with a
    colorbar. UMAP is rendered from one shared projection with anomaly,
    cluster, dataset-split, and marginal-Vendi colorings. t-SNE plots (one per
    cluster) are emitted only when
    ``include_tsne`` is True and the Hopkins tendency gate passes; UMAP is
    always rendered.
    """
    import matplotlib

    matplotlib.use("Agg")

    Z = np.asarray(Z, dtype=np.float64)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anomaly_scores: Optional[np.ndarray] = None
    if anomaly_score is not None:
        candidate = np.asarray(anomaly_score, dtype=np.float64).reshape(-1)
        if candidate.shape[0] == Z.shape[0]:
            anomaly_scores = candidate
        else:
            LOG.warning(
                "Anomaly score length %s does not match latent rows %s; ignoring.",
                int(candidate.shape[0]),
                int(Z.shape[0]),
            )

    split_values: Optional[np.ndarray] = np.full(len(Z), split, dtype=object)
    if data_split is not None:
        candidate = np.asarray(data_split, dtype=object).reshape(-1)
        if candidate.shape[0] == Z.shape[0]:
            split_values = candidate
        else:
            LOG.warning(
                "Data-split label count %s does not match latent rows %s; ignoring.",
                int(candidate.shape[0]),
                int(Z.shape[0]),
            )

    marginal_vendi_values: Optional[np.ndarray] = None
    if marginal_vendi is not None:
        candidate = np.asarray(marginal_vendi, dtype=np.float64).reshape(-1)
        if candidate.shape[0] == Z.shape[0]:
            marginal_vendi_values = candidate
        else:
            LOG.warning(
                "Marginal Vendi count %s does not match latent rows %s; ignoring.",
                int(candidate.shape[0]),
                int(Z.shape[0]),
            )

    tendency = tendency_summary or _summarize_cluster_tendency(
        Z,
        hopkins_threshold=hopkins_threshold,
        sample_size=tendency_sample_size,
        random_state=random_state,
    )

    labels = cluster_labels
    clustering_blocked_by_gate = False
    if labels is None:
        if apply_tendency_gate and not tendency.get("gate_passed", True):
            labels = np.full(len(Z), -1, dtype=int)
            clustering_blocked_by_gate = True
        else:
            try:
                labels, _ = _cluster_latent_embeddings(
                    Z,
                    method=cluster_method,
                    n_clusters=None,
                    random_state=random_state,
                    hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                    hdbscan_min_samples=hdbscan_min_samples,
                )
            except Exception as exc:
                LOG.warning("Clustering for latent plot failed: %s", exc)
                labels = np.full(len(Z), -1, dtype=int)
    include_tsne_effective = bool(include_tsne) and not clustering_blocked_by_gate

    saved_paths: List[str] = []
    safe_model_name = model_name.replace(" ", "_")

    if save_tendency_artifacts:
        tendency_base = output_dir / f"latent_{safe_model_name}_{split}_tendency"
        tendency_payload = {
            key: value
            for key, value in tendency.items()
            if key not in {"vat_matrix", "ivat_matrix"}
        }
        tendency_json = tendency_base.with_suffix(".json")
        tendency_json.write_text(json.dumps(tendency_payload, indent=2), encoding="utf-8")
        saved_paths.append(str(tendency_json))

        vat_png = _save_tendency_heatmap(
            np.asarray(tendency.get("vat_matrix", np.zeros((0, 0))), dtype=np.float64),
            out_png=output_dir / f"latent_{safe_model_name}_{split}_vat.png",
            title=f"{_model_short_name(model_name)} {split} VAT",
        )
        if vat_png is not None:
            saved_paths.append(str(vat_png))

        ivat_png = _save_tendency_heatmap(
            np.asarray(tendency.get("ivat_matrix", np.zeros((0, 0))), dtype=np.float64),
            out_png=output_dir / f"latent_{safe_model_name}_{split}_ivat.png",
            title=f"{_model_short_name(model_name)} {split} iVAT",
        )
        if ivat_png is not None:
            saved_paths.append(str(ivat_png))

    for embedding_type, embedding in _fit_2d_embeddings(
        Z,
        random_state=random_state,
        include_tsne=include_tsne_effective,
    ):
        saved_paths.extend(
            _save_embedding(
                Z,
                embedding,
                sample_indices=sample_indices,
                label_values=None,
                cluster_labels=labels,
                recon_error=recon_error,
                anomaly_score=anomaly_scores,
                data_split=split_values,
                marginal_vendi=marginal_vendi_values,
                model_name=model_name,
                split=split,
                embedding_type=embedding_type,
                title=f"{_model_short_name(model_name)} latent {embedding_type.upper()}",
                output_dir=output_dir,
                color_by=color_by,
                random_state=random_state,
                anomaly_quantile=anomaly_quantile,
                interactive_threshold=interactive_threshold,
                interactive_hover_sample_frac=interactive_hover_sample_frac,
                cluster_method=cluster_method,
                tendency_summary=tendency,
                latent_quality_n_neighbors=latent_quality_n_neighbors,
            )
        )

    return {
        "status": "complete",
        "model_name": model_name,
        "split": split,
        "n_samples": int(len(Z)),
        "cluster_method": cluster_method,
        "saved_paths": saved_paths,
    }


def plot_mapper_reconstruction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    sample_indices: np.ndarray,
    output_dir: Path,
    model_name: str = "mapper",
    split: str = "train_val_test",
    top_n_features: int = 6,
    worst_n_samples: int = 25,
    anomaly_quantile: float = 0.95,
    include_image_panels: bool = True,
    image_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Plot reconstruction diagnostics directly from in-memory arrays."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_cols = y_true.shape[1]
    df = pd.DataFrame({"index": np.asarray(sample_indices, dtype=np.int64)})
    for i in range(n_cols):
        df[f"y_true_{i:02d}"] = y_true[:, i]
        df[f"y_pred_{i:02d}"] = y_pred[:, i]

    out_prefix = output_dir / f"reconstruction_{model_name.replace(' ', '_')}_{split}"
    saved = _reconstruction_diagnostics(
        df,
        out_prefix,
        top_n_features=top_n_features,
        worst_n_samples=worst_n_samples,
        anomaly_quantile=anomaly_quantile,
        include_image_panels=include_image_panels,
        image_shape=image_shape,
    )

    return {
        "status": "complete",
        "model_name": model_name,
        "split": split,
        "n_samples": int(len(df)),
        "saved_paths": saved,
    }


def plot_mapper_pca(
    Z: np.ndarray,
    *,
    output_dir: Path,
    model_name: str = "mapper_pca",
    split: str = "train_val_test",
    explained_variance_ratio: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Render PCA scores using PC1/PC2 plus a full variance summary.

    ``Z`` may contain any dynamically selected number of components.  The
    score panel deliberately uses only its first two columns; all retained
    components remain available to Mapper and are included in the variance
    panel.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib is required for Mapper PCA visualization.")
        return {
            "status": "skipped",
            "reason": "matplotlib_unavailable",
            "saved_paths": [],
        }

    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.ndim != 2 or Z.shape[0] == 0:
        return {
            "status": "skipped",
            "reason": "empty_or_invalid_latent",
            "saved_paths": [],
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = model_name.replace(" ", "_")
    out_png = output_dir / f"pca_{safe_model_name}_{split}.png"

    evr = None
    if explained_variance_ratio is not None:
        evr_arr = np.asarray(explained_variance_ratio, dtype=np.float64).reshape(-1)
        finite = np.isfinite(evr_arr)
        if np.any(finite):
            evr = evr_arr[finite]

    if evr is not None and evr.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax_scatter = axes[0]
        ax_var = axes[1]
    else:
        fig, ax_scatter = plt.subplots(figsize=(7, 6))
        ax_var = None

    plotted_scores = Z[:, :2]
    if plotted_scores.shape[1] >= 2:
        x_vals = plotted_scores[:, 0]
        y_vals = plotted_scores[:, 1]
        ax_scatter.set_xlabel("PC1")
        ax_scatter.set_ylabel("PC2")
        ax_scatter.set_title(f"{_model_short_name(model_name)} PCA scores")
    else:
        x_vals = np.arange(Z.shape[0], dtype=np.int64)
        y_vals = plotted_scores[:, 0]
        ax_scatter.set_xlabel("Sample index")
        ax_scatter.set_ylabel("PC1")
        ax_scatter.set_title(
            f"{_model_short_name(model_name)} PCA scores (single component)"
        )
    ax_scatter.scatter(x_vals, y_vals, s=18, alpha=0.75, edgecolors="none")
    if plotted_scores.shape[1] >= 2:
        x_min = float(np.nanmin(x_vals))
        x_max = float(np.nanmax(x_vals))
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        x_span = max(1e-12, x_max - x_min)
        y_span = max(1e-12, y_max - y_min)

        # In PC-score space, PC1 and PC2 directions coincide with x/y axes.
        ax_scatter.plot(
            [x_min - 0.05 * x_span, x_max + 0.05 * x_span],
            [0.0, 0.0],
            linestyle="--",
            color="red",
            linewidth=1.6,
            alpha=0.9,
        )
        ax_scatter.plot(
            [0.0, 0.0],
            [y_min - 0.05 * y_span, y_max + 0.05 * y_span],
            linestyle="--",
            color="red",
            linewidth=1.6,
            alpha=0.9,
        )
        ax_scatter.text(
            x_max + 0.04 * x_span,
            0.0,
            "PC1",
            color="red",
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
        )
        ax_scatter.text(
            0.0,
            y_max + 0.04 * y_span,
            "PC2",
            color="red",
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
        )
    ax_scatter.grid(alpha=0.25)

    if ax_var is not None and evr is not None:
        n_plot = min(20, evr.size)
        pcs = np.arange(1, n_plot + 1, dtype=np.int64)
        cumulative = np.cumsum(evr)
        ax_var.bar(pcs, evr[:n_plot], alpha=0.75, label="Explained variance ratio")
        for pc, ratio in zip(pcs, evr[:n_plot]):
            ax_var.annotate(
                f"{ratio * 100:.1f}%",
                (pc, ratio),
                va="bottom",
                ha="center",
                fontsize=8,
            )
        ax_var.plot(
            pcs,
            cumulative[:n_plot],
            color="red",
            marker="o",
            linewidth=1.5,
            markersize=3,
            label="Cumulative explained variance",
        )
        for pc, cum in zip(pcs[1:], cumulative[1:n_plot]):
            ax_var.annotate(
                f"{cum * 100:.1f}%",
                (pc, cum),
                va="bottom",
                ha="center",
                fontsize=8,
            )
        ax_var.set_ylim(0.0, 1.05)
        ax_var.set_xlabel("Principal component")
        ax_var.set_ylabel("Variance ratio")
        ax_var.set_title("PCA explained variance")
        ax_var.grid(alpha=0.25)
        ax_var.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "complete",
        "model_name": model_name,
        "split": split,
        "n_samples": int(Z.shape[0]),
        "n_components": int(Z.shape[1]),
        "plotted_components": int(plotted_scores.shape[1]),
        "saved_paths": [str(out_png)],
    }


__all__ = [
    "DEFAULT_MODEL_DISPLAY",
    "_model_short_name",
    "plot_mapper_latent",
    "plot_mapper_pca",
    "plot_mapper_reconstruction",
    "viz_unsupervised_latent",
    "viz_unsupervised_reconstruction",
]
