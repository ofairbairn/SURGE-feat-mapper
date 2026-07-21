"""Generic run visualization: inference comparison, HPO, and SHAP from any SURGE run directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from ..io.load_compat import load_model_compat
import pandas as pd

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .comparison import plot_inference_comparison_grid
from .classification import (
    plot_calibration_curve,
    plot_classification_dashboard,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from .hpo import plot_hpo_convergence

LOG = logging.getLogger(__name__)

# Default model display names (short labels for plots)
DEFAULT_MODEL_DISPLAY = {
    "random_forest_profiles": "Random Forest",
    "torch_mlp_mc_dropout": "MLP",
    "gpflow_gpr_profiles": "GPR",
    "xgc_mlp_aparallel": "MLP",
    "xgc_rf_aparallel": "Random Forest",
}


def load_predictions(
    run_dir: Path,
    datasets: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Dict[str, Dict[int, Tuple[Any, Any]]]]:
    """Load predictions from a SURGE workflow run. Supports multi-output.

    Accepts both ``*.parquet`` (current workflow default) and ``*.csv``
    (legacy runs). When both exist for the same model/split, parquet
    wins because it round-trips dtypes losslessly.
    """
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return {}

    # Parquet first so it overwrites csv entries for the same model/split.
    pred_files = sorted(
        list(predictions_dir.glob("*.parquet")) + list(predictions_dir.glob("*.csv"))
    )

    results: Dict[str, Dict[str, Dict[int, Tuple[Any, Any]]]] = {}
    for pred_file in pred_files:
        if "_uq" in pred_file.stem:
            continue
        stem = pred_file.stem
        dataset_type = (
            "train"
            if "_train" in pred_file.name
            else ("val" if "_val" in pred_file.name else "test")
        )
        if dataset_type not in datasets:
            continue
        model_name = stem.replace("_train", "").replace("_val", "").replace("_test", "")

        if pred_file.suffix == ".parquet":
            df = pd.read_parquet(pred_file)
        else:
            df = pd.read_csv(pred_file)
        y_true_cols = sorted([c for c in df.columns if c.startswith("y_true_")])
        y_pred_cols = sorted([c for c in df.columns if c.startswith("y_pred_")])
        if not y_true_cols or not y_pred_cols:
            continue

        if model_name not in results:
            results[model_name] = {}
        if dataset_type not in results[model_name]:
            results[model_name][dataset_type] = {}

        for i, (tc, pc) in enumerate(zip(y_true_cols, y_pred_cols)):
            results[model_name][dataset_type][i] = (df[tc].values, df[pc].values)

    return results


def _model_short_name(name: str) -> str:
    """Convert model filename to short display name."""
    return DEFAULT_MODEL_DISPLAY.get(name, name.replace("_", " ").title())


def _normalize_marker_sizes(recon_error_norm: np.ndarray, *, base: float = 24.0, spread: float = 96.0) -> np.ndarray:
    values = np.asarray(recon_error_norm, dtype=np.float64)
    sizes = np.full(values.shape, base, dtype=np.float64)
    valid = np.isfinite(values)
    if np.any(valid):
        sizes[valid] = base + np.clip(values[valid], 0.0, 1.0) * spread
    return sizes


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


def _nearest_neighbor_preservation(
    X_ref: np.ndarray,
    X_embed: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> Optional[float]:
    from sklearn.neighbors import NearestNeighbors

    n = min(len(X_ref), len(X_embed))
    if n < 3:
        return None
    k = max(2, min(n_neighbors + 1, n))

    nn_ref = NearestNeighbors(n_neighbors=k).fit(X_ref)
    nn_emb = NearestNeighbors(n_neighbors=k).fit(X_embed)
    ref_neighbors = nn_ref.kneighbors(return_distance=False)[:, 1:]
    emb_neighbors = nn_emb.kneighbors(return_distance=False)[:, 1:]

    overlaps = []
    for ref_row, emb_row in zip(ref_neighbors, emb_neighbors):
        inter = len(set(ref_row.tolist()).intersection(set(emb_row.tolist())))
        overlaps.append(inter / max(1, len(ref_row)))
    return float(np.mean(overlaps))


def _compute_latent_quality_metrics(
    latent: np.ndarray,
    embedding: np.ndarray,
    clusters: Optional[np.ndarray],
    *,
    n_neighbors: int = 10,
) -> Dict[str, Any]:
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    metrics: Dict[str, Any] = {}
    n = min(len(latent), len(embedding))
    if n < 3:
        metrics["trustworthiness"] = None
        metrics["nearest_neighbor_preservation"] = None
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["n_clusters"] = 0
        metrics["noise_fraction"] = None
        return metrics

    k = min(max(2, n_neighbors), n - 1)
    try:
        metrics["trustworthiness"] = float(trustworthiness(latent, embedding, n_neighbors=k))
    except Exception:
        metrics["trustworthiness"] = None

    try:
        metrics["nearest_neighbor_preservation"] = _nearest_neighbor_preservation(
            latent,
            embedding,
            n_neighbors=k,
        )
    except Exception:
        metrics["nearest_neighbor_preservation"] = None

    if clusters is None:
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        metrics["n_clusters"] = 0
        metrics["noise_fraction"] = None
        return metrics

    labels = np.asarray(clusters)
    non_noise_mask = labels != -1
    noise_fraction = 1.0 - float(np.mean(non_noise_mask))
    valid_labels = labels[non_noise_mask]
    valid_embedding = embedding[non_noise_mask]
    unique_labels = np.unique(valid_labels)
    metrics["n_clusters"] = int(len(unique_labels))
    metrics["noise_fraction"] = noise_fraction
    if len(unique_labels) < 2 or len(valid_embedding) <= len(unique_labels):
        metrics["silhouette"] = None
        metrics["davies_bouldin"] = None
        return metrics

    try:
        metrics["silhouette"] = float(silhouette_score(valid_embedding, valid_labels))
    except Exception:
        metrics["silhouette"] = None
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(valid_embedding, valid_labels))
    except Exception:
        metrics["davies_bouldin"] = None
    return metrics

 
def _compute_hopkins_statistic(
    latent: np.ndarray,
    *,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Optional[float]:
    """Estimate clustering tendency with the Hopkins statistic.

    Values near 0.5 indicate a random cloud; larger values indicate stronger
    clustering tendency.
    """
    from sklearn.neighbors import NearestNeighbors

    latent = np.asarray(latent, dtype=np.float64)
    if latent.ndim != 2 or len(latent) < 3:
        return None

    n_samples, n_features = latent.shape
    mins = np.min(latent, axis=0)
    maxs = np.max(latent, axis=0)
    if np.any(~np.isfinite(mins)) or np.any(~np.isfinite(maxs)):
        return None
    if np.allclose(mins, maxs):
        return None

    m = int(sample_size or min(max(10, n_samples // 10), 256))
    m = max(1, min(m, n_samples - 1))

    rng = np.random.default_rng(random_state)
    real_idx = rng.choice(n_samples, size=m, replace=False)
    synthetic = rng.uniform(mins, maxs, size=(m, n_features))

    nn = NearestNeighbors(n_neighbors=2).fit(latent)
    real_distances = nn.kneighbors(latent[real_idx], return_distance=True)[0][:, 1]
    synthetic_distances = nn.kneighbors(synthetic, return_distance=True)[0][:, 0]

    real_sum = float(np.sum(real_distances))
    synth_sum = float(np.sum(synthetic_distances))
    denom = real_sum + synth_sum
    if denom <= 0.0:
        return None
    return synth_sum / denom


def _vat_reordering(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return VAT-reordered distance matrix, order, and MST parent links."""
    from scipy.spatial.distance import pdist, squareform

    latent = np.asarray(latent, dtype=np.float64)
    n_samples = len(latent)
    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=int), np.empty(0, dtype=int)
    if n_samples == 1:
        return np.zeros((1, 1), dtype=np.float64), np.array([0], dtype=int), np.array([-1], dtype=int)

    dist_matrix = squareform(pdist(latent, metric="euclidean"))
    seed_i, _seed_j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    order: List[int] = [int(seed_i)]
    parents: List[int] = [-1]
    remaining = set(range(n_samples))
    remaining.remove(int(seed_i))

    while remaining:
        current = np.array(order, dtype=int)
        candidates = np.array(sorted(remaining), dtype=int)
        candidate_distances = dist_matrix[np.ix_(current, candidates)]
        min_to_tree = np.min(candidate_distances, axis=0)
        candidate_choice = int(np.argmin(min_to_tree))
        next_idx = int(candidates[candidate_choice])
        parent_in_current = int(np.argmin(candidate_distances[:, candidate_choice]))
        parent_idx = parent_in_current
        order.append(next_idx)
        parents.append(parent_idx)
        remaining.remove(next_idx)

    order_arr = np.asarray(order, dtype=int)
    reordered = dist_matrix[np.ix_(order_arr, order_arr)]
    return reordered, order_arr, np.asarray(parents, dtype=int)


def _ivat_from_vat(
    vat_matrix: np.ndarray,
    parents: np.ndarray,
) -> np.ndarray:
    """Compute an iVAT-style path-max distance image from a VAT MST."""
    n_samples = len(vat_matrix)
    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float64)
    if n_samples == 1:
        return np.zeros((1, 1), dtype=np.float64)

    adjacency: List[List[tuple[int, float]]] = [[] for _ in range(n_samples)]
    for child in range(1, n_samples):
        parent = int(parents[child])
        if parent < 0:
            continue
        weight = float(vat_matrix[child, parent])
        adjacency[child].append((parent, weight))
        adjacency[parent].append((child, weight))

    ivat = np.zeros((n_samples, n_samples), dtype=np.float64)
    for start in range(n_samples):
        visited = {start}
        stack: List[tuple[int, float]] = [(start, 0.0)]
        while stack:
            node, current_max = stack.pop()
            ivat[start, node] = current_max
            for neighbor, weight in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append((neighbor, max(current_max, weight)))

    return ivat


def _summarize_cluster_tendency(
    latent: np.ndarray,
    *,
    hopkins_threshold: float = 0.55,
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute tendency diagnostics before clustering latent vectors.

    The automated gate is based on Hopkins. VAT/iVAT are saved as supporting
    diagnostics because they are primarily visual assessments.
    """
    latent = np.asarray(latent, dtype=np.float64)
    hopkins = _compute_hopkins_statistic(
        latent,
        sample_size=sample_size,
        random_state=random_state,
    )
    vat_matrix, vat_order, vat_parents = _vat_reordering(latent)
    ivat_matrix = _ivat_from_vat(vat_matrix, vat_parents)

    gate_passed = True if hopkins is None else bool(hopkins >= hopkins_threshold)
    return {
        "n_samples": int(len(latent)),
        "latent_dim": int(latent.shape[1]) if latent.ndim == 2 else None,
        "hopkins": hopkins,
        "hopkins_threshold": float(hopkins_threshold),
        "gate_passed": gate_passed,
        "gate_reason": (
            "hopkins_below_threshold" if hopkins is not None and hopkins < hopkins_threshold else "pass"
        ),
        "vat_order": vat_order.tolist(),
        "vat_parents": vat_parents.tolist(),
        "vat_matrix": vat_matrix,
        "ivat_matrix": ivat_matrix,
    }


def _cluster_latent_embeddings(
    latent: np.ndarray,
    *,
    method: str = "hdbscan",
    n_clusters: Optional[int] = None,
    random_state: int = 42,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: Optional[int] = None,
    agglomerative_linkage: str = "ward",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    latent = np.asarray(latent, dtype=np.float64)
    n_samples = latent.shape[0]
    if n_samples < 2 or method.lower() in {"", "none", "off", "no"}:
        return np.full(n_samples, -1, dtype=int), None

    method_normalized = method.lower().strip()
    if method_normalized == "kmeans":
        from sklearn.cluster import KMeans

        n_clusters_eff = int(n_clusters or min(8, max(2, int(np.sqrt(n_samples)))))
        n_clusters_eff = max(2, min(n_clusters_eff, n_samples))
        clusterer = KMeans(n_clusters=n_clusters_eff, random_state=random_state, n_init="auto")
        return clusterer.fit_predict(latent), None

    if method_normalized == "dbscan":
        from sklearn.cluster import DBSCAN

        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        return clusterer.fit_predict(latent), None

    if method_normalized == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("hdbscan is required for cluster_method='hdbscan'") from exc

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )
        return clusterer.fit_predict(latent), None

    if method_normalized == "agglomerative":
        from scipy.cluster.hierarchy import linkage
        from sklearn.cluster import AgglomerativeClustering

        n_clusters_eff = int(n_clusters or min(8, max(2, int(np.sqrt(n_samples)))))
        n_clusters_eff = max(2, min(n_clusters_eff, n_samples))
        clusterer = AgglomerativeClustering(n_clusters=n_clusters_eff, linkage=agglomerative_linkage)
        labels = clusterer.fit_predict(latent)
        linkage_matrix = linkage(latent, method=agglomerative_linkage)
        return labels, linkage_matrix

    raise ValueError(
        f"Unsupported cluster_method={method!r}. Expected one of: none, kmeans, dbscan, hdbscan, agglomerative."
    )


def _save_tendency_heatmap(matrix: np.ndarray, *, out_png: Path, title: str) -> Optional[Path]:
    if matrix.size == 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", interpolation="nearest")
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

    return df


def _plot_latent_matplotlib(
    latent_df: pd.DataFrame,
    *,
    title: str,
    out_png: Path,
    color_by: str = "label",
) -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    if color_by not in latent_df.columns or latent_df[color_by].isna().all():
        color_by = "cluster" if "cluster" in latent_df.columns and not latent_df["cluster"].isna().all() else "label"

    plot_df = latent_df.copy()
    color_values = plot_df[color_by].astype("string").fillna("NA")
    unique_values = list(pd.unique(color_values))
    cmap_name = "tab10" if len(unique_values) <= 10 else "tab20"
    try:
        cmap_obj = matplotlib.colormaps.get_cmap(cmap_name).resampled(max(len(unique_values), 1))
    except AttributeError:
        cmap_obj = plt.get_cmap(cmap_name, max(len(unique_values), 1))
    color_map = {value: cmap_obj(i) for i, value in enumerate(unique_values)}

    sizes = _normalize_marker_sizes(plot_df["recon_error_norm"].to_numpy(dtype=np.float64))
    fig, ax = plt.subplots(figsize=(8, 6))
    for value in unique_values:
        mask = color_values == value
        ax.scatter(
            plot_df.loc[mask, "x"],
            plot_df.loc[mask, "y"],
            s=sizes[mask],
            c=[color_map[value]],
            alpha=0.75,
            label=str(value),
            edgecolors="none",
        )

    if "cluster" in plot_df.columns:
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
    ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
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
        "recon_error",
        "recon_error_norm",
        "embedding_type",
    ]
    hover_cols = [col for col in hover_cols if col in plot_df.columns]
    hover_tooltips = []
    for col in hover_cols:
        if col in {"recon_error", "recon_error_norm"}:
            hover_tooltips.append((col, f"@{{{col}}}{{0.000}}"))
        else:
            hover_tooltips.append((col, f"@{{{col}}}"))
    cmap_name = "Category10" if plot_df[color_by].nunique() <= 10 else "Category20"

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


def _is_classification_run(run_dir: Path) -> bool:
    """Best-effort task-type detection from workflow summary metrics."""
    summary_path = run_dir / "workflow_summary.json"
    if not summary_path.exists():
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        for model in summary.get("models", []):
            metrics = model.get("metrics", {})
            for split in ("train", "val", "test"):
                split_metrics = metrics.get(split) or {}
                if "accuracy" in split_metrics or "f1_macro" in split_metrics:
                    return True
    except Exception:
        return False
    return False


def _resolve_task_type(run_dir: Path) -> str:
    """Resolve workflow task type with spec.yaml first, metrics as fallback."""
    spec_path = run_dir / "spec.yaml"
    if spec_path.exists() and YAML_AVAILABLE:
        try:
            with spec_path.open("r", encoding="utf-8") as handle:
                spec_payload = yaml.safe_load(handle) or {}
            task_type = str(spec_payload.get("task_type", "")).strip().lower()
            if task_type in {"regression", "classification", "unsupervised"}:
                return task_type
        except Exception:
            pass

    return "classification" if _is_classification_run(run_dir) else "regression"


def _load_prediction_frames(
    run_dir: Path,
    datasets: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load prediction tables keyed by model and split."""
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.exists():
        return {}

    pred_files = sorted(
        list(predictions_dir.glob("*.parquet")) + list(predictions_dir.glob("*.csv"))
    )

    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for pred_file in pred_files:
        if "_uq" in pred_file.stem:
            continue
        dataset_type = (
            "train"
            if "_train" in pred_file.name
            else ("val" if "_val" in pred_file.name else "test")
        )
        if dataset_type not in datasets:
            continue
        stem = pred_file.stem
        model_name = stem.replace("_train", "").replace("_val", "").replace("_test", "")

        # Keep parquet when both parquet and csv exist for the same model/split.
        if model_name in frames and dataset_type in frames[model_name]:
            continue

        if pred_file.suffix == ".parquet":
            df = pd.read_parquet(pred_file)
        else:
            df = pd.read_csv(pred_file)
        frames.setdefault(model_name, {})[dataset_type] = df
    return frames


def viz_classification(
    run_dir: Path,
    output_dir: Path,
    *,
    datasets: Tuple[str, ...] = ("train", "val", "test"),
) -> List[str]:
    """Emit classification diagnostics for each model/split with available predictions."""
    frames = _load_prediction_frames(run_dir, datasets=datasets)
    if not frames:
        return []

    saved_paths: List[str] = []
    for model_name, model_splits in frames.items():
        model_dir = output_dir / f"classification_{model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)

        for split, df in model_splits.items():
            y_true_cols = sorted([c for c in df.columns if c.startswith("y_true_")])
            y_pred_cols = sorted([c for c in df.columns if c.startswith("y_pred_")])
            if not y_true_cols or not y_pred_cols:
                continue

            y_true = np.asarray(df[y_true_cols[0]].values)
            y_pred = np.asarray(df[y_pred_cols[0]].values)
            classes = np.unique(np.concatenate([y_true, y_pred], axis=0))
            labels = [str(c) for c in classes]

            cm_path = model_dir / f"{split}_confusion_matrix.png"
            try:
                plot_confusion_matrix(
                    y_true,
                    y_pred,
                    labels=labels,
                    title=f"{_model_short_name(model_name)} {split} confusion",
                    save_path=cm_path,
                )
                saved_paths.append(str(cm_path))
            except Exception as exc:
                LOG.warning("Confusion matrix failed for %s %s: %s", model_name, split, exc)

            y_prob_cols = sorted([c for c in df.columns if c.startswith("y_prob_")])
            if "y_prob" in df.columns:
                y_prob = np.asarray(df["y_prob"].values)
            elif y_prob_cols:
                y_prob = np.asarray(df[y_prob_cols].values)
            else:
                y_prob = None

            if y_prob is None:
                LOG.info(
                    "Skipping ROC/PR/calibration dashboard for %s %s: no y_prob columns in predictions.",
                    model_name,
                    split,
                )
                continue

            roc_path = model_dir / f"{split}_roc_curve.png"
            pr_path = model_dir / f"{split}_precision_recall_curve.png"
            cal_path = model_dir / f"{split}_calibration_curve.png"
            dash_path = model_dir / f"{split}_classification_dashboard.png"

            try:
                plot_roc_curve(
                    y_true,
                    y_prob,
                    labels=labels,
                    title=f"{_model_short_name(model_name)} {split} ROC",
                    save_path=roc_path,
                )
                saved_paths.append(str(roc_path))
            except Exception as exc:
                LOG.warning("ROC plot failed for %s %s: %s", model_name, split, exc)

            try:
                plot_precision_recall_curve(
                    y_true,
                    y_prob,
                    labels=labels,
                    title=f"{_model_short_name(model_name)} {split} PR",
                    save_path=pr_path,
                )
                saved_paths.append(str(pr_path))
            except Exception as exc:
                LOG.warning("PR plot failed for %s %s: %s", model_name, split, exc)

            try:
                plot_calibration_curve(
                    y_true,
                    y_prob,
                    title=f"{_model_short_name(model_name)} {split} calibration",
                    save_path=cal_path,
                )
                saved_paths.append(str(cal_path))
            except Exception as exc:
                LOG.warning("Calibration plot failed for %s %s: %s", model_name, split, exc)

            try:
                plot_classification_dashboard(
                    y_true,
                    y_pred,
                    y_prob,
                    labels=labels,
                    model_name=f"{_model_short_name(model_name)} ({split})",
                    save_path=dash_path,
                )
                saved_paths.append(str(dash_path))
            except Exception as exc:
                LOG.warning("Classification dashboard failed for %s %s: %s", model_name, split, exc)

    return saved_paths


def viz_hpo(
    run_dir: Path,
    output_dir: Path,
    dpi: int = 150,
    plot_metric: Optional[str] = None,
) -> List[str]:
    """
    Generate HPO convergence plots from run_dir/hpo/*.json.
    Returns list of saved plot paths.
    """
    hpo_dir = run_dir / "hpo"
    if not hpo_dir.exists():
        return []

    hpo_files = sorted(hpo_dir.glob("*_hpo.json"))
    if not hpo_files:
        return []

    # Build method names from filenames (xgc_mlp_aparallel_hpo -> MLP)
    hpo_dict = {}
    for f in hpo_files:
        model_key = f.stem.replace("_hpo", "")
        hpo_dict[_model_short_name(model_key)] = str(f)

    # Auto-detect: prefer R² when val_r2 is in trials (or use plot_metric override)
    save_path = output_dir / "hpo_convergence.png"
    try:
        plot_hpo_convergence(
            hpo_files=list(hpo_dict.values()),
            metric=plot_metric,  # None = auto-detect: R² if available, else RMSE
            method_names=list(hpo_dict.keys()),
            save_path=save_path,
            dpi=dpi,
        )
        return [str(save_path)]
    except Exception:
        return []


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
    include_pca_baseline: bool = True,
    pca_n_components: Optional[int] = None,
    pca_max_components: int = 32,
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
    from ..dataset import SurrogateDataset
    from ..model.registry import MODEL_REGISTRY
    from ..workflow.spec import SurrogateWorkflowSpec

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

        pca_latent: Optional[np.ndarray] = None
        pca_recon_error: Optional[np.ndarray] = None
        pca_components_used: Optional[int] = None
        if include_pca_baseline:
            try:
                from sklearn.decomposition import PCA

                n_samples, n_features = X.shape
                max_components = max(1, int(pca_max_components))
                upper = min(n_samples - 1, n_features, max_components)
                if upper >= 1:
                    if pca_n_components is not None:
                        n_components = int(max(1, min(int(pca_n_components), upper)))
                    else:
                        # Keep the baseline roughly comparable to the learned latent width.
                        n_components = int(max(1, min(upper, Z.shape[1])))

                    pca_model = PCA(n_components=n_components, random_state=random_state)
                    pca_latent = np.asarray(pca_model.fit_transform(X), dtype=np.float64)
                    X_recon_pca = np.asarray(pca_model.inverse_transform(pca_latent), dtype=np.float64)
                    pca_recon_error = np.sqrt(np.mean(np.square(X - X_recon_pca), axis=1))
                    pca_components_used = n_components
                else:
                    LOG.warning(
                        "Skipping PCA baseline for %s %s: not enough samples/features (shape=%s).",
                        model_name,
                        split,
                        X.shape,
                    )
            except Exception as exc:
                LOG.warning("PCA baseline failed for %s %s: %s", model_name, split, exc)

        tendency_summary = _summarize_cluster_tendency(
            Z,
            hopkins_threshold=hopkins_threshold,
            sample_size=tendency_sample_size,
            random_state=random_state,
        )

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

        cluster_labels = None
        linkage_matrix = None
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

        safe_model_name = model_name.replace(" ", "_")

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

        def _save_embedding(
            embedding: np.ndarray,
            *,
            embedding_type: str,
            title: str,
        ) -> None:
            embedding_df = _build_latent_dataframe(
                sample_id=valid_indices,
                embedding=embedding,
                embedding_type=embedding_type,
                label=label_values,
                cluster=cluster_labels,
                recon_error=recon_error,
            )
            embedding_df["model_name"] = model_name
            embedding_df["split"] = split

            if recon_error is not None:
                recon_arr = np.asarray(recon_error, dtype=np.float64)
                valid_recon = np.isfinite(recon_arr)
                threshold = float(np.nanquantile(recon_arr[valid_recon], anomaly_quantile)) if np.any(valid_recon) else np.nan
                embedding_df["anomaly_score"] = embedding_df["recon_error_norm"]
                embedding_df["anomaly_flag"] = (
                    embedding_df["recon_error"].astype("float64") >= threshold
                ) if np.isfinite(threshold) else False
                embedding_df["anomaly_threshold"] = threshold

            quality_payload = {
                "model_name": model_name,
                "split": split,
                "embedding_type": embedding_type,
                "cluster_method": cluster_method,
                "cluster_tendency": {
                    key: value
                    for key, value in tendency_summary.items()
                    if key not in {"vat_matrix", "ivat_matrix"}
                },
                "quality": _compute_latent_quality_metrics(
                    Z,
                    embedding,
                    cluster_labels,
                    n_neighbors=latent_quality_n_neighbors,
                ),
            }

            parquet_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}.parquet"
            embedding_df.to_parquet(parquet_path, index=False)
            saved_paths.append(str(parquet_path))

            quality_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}_quality.json"
            with quality_path.open("w", encoding="utf-8") as f:
                json.dump(quality_payload, f, indent=2)
            saved_paths.append(str(quality_path))

            png_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}.png"
            _plot_latent_matplotlib(
                embedding_df,
                title=title,
                out_png=png_path,
                color_by=color_by,
            )
            saved_paths.append(str(png_path))

            html_path = output_dir / f"latent_{safe_model_name}_{split}_{embedding_type}.html"
            html_saved = _plot_latent_interactive(
                embedding_df,
                title=title,
                out_html=html_path,
                color_by=color_by,
                random_state=random_state,
                hover_sample_frac=interactive_hover_sample_frac,
                datashader_threshold=interactive_threshold,
            )
            if html_saved is not None:
                saved_paths.append(str(html_saved))

        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=random_state,
            )
            Z_umap = reducer.fit_transform(Z)
            _save_embedding(
                Z_umap,
                embedding_type="umap",
                title=f"{_model_short_name(model_name)} {split} latent UMAP",
            )
        except ImportError:
            LOG.warning("umap-learn is not installed; skipping UMAP for %s.", model_name)
        except Exception as exc:
            LOG.warning("UMAP failed for %s: %s", model_name, exc)

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
            Z_tsne = tsne.fit_transform(Z)
            _save_embedding(
                Z_tsne,
                embedding_type="tsne",
                title=f"{_model_short_name(model_name)} {split} latent t-SNE",
            )
        except Exception as exc:
            LOG.warning("t-SNE failed for %s: %s", model_name, exc)

        # PCA ladder baseline for unsupervised runs: same split, same downstream
        # gate + clustering + visualization as learned latent embeddings.
        if pca_latent is not None and pca_latent.ndim == 2 and pca_latent.shape[0] > 0:
            pca_tag = f"{safe_model_name}_pca_baseline"

            pca_tendency_summary = _summarize_cluster_tendency(
                pca_latent,
                hopkins_threshold=hopkins_threshold,
                sample_size=tendency_sample_size,
                random_state=random_state,
            )

            if save_tendency_artifacts:
                pca_tendency_base = output_dir / f"latent_{pca_tag}_{split}_tendency"
                pca_tendency_payload = {
                    key: value
                    for key, value in pca_tendency_summary.items()
                    if key not in {"vat_matrix", "ivat_matrix"}
                }
                pca_tendency_json = pca_tendency_base.with_suffix(".json")
                pca_tendency_json.write_text(
                    json.dumps(pca_tendency_payload, indent=2),
                    encoding="utf-8",
                )
                saved_paths.append(str(pca_tendency_json))

                pca_vat_png = _save_tendency_heatmap(
                    np.asarray(pca_tendency_summary["vat_matrix"], dtype=np.float64),
                    out_png=output_dir / f"latent_{pca_tag}_{split}_vat.png",
                    title=f"{_model_short_name(model_name)} {split} PCA VAT",
                )
                if pca_vat_png is not None:
                    saved_paths.append(str(pca_vat_png))

                pca_ivat_png = _save_tendency_heatmap(
                    np.asarray(pca_tendency_summary["ivat_matrix"], dtype=np.float64),
                    out_png=output_dir / f"latent_{pca_tag}_{split}_ivat.png",
                    title=f"{_model_short_name(model_name)} {split} PCA iVAT",
                )
                if pca_ivat_png is not None:
                    saved_paths.append(str(pca_ivat_png))

            pca_cluster_labels = None
            pca_linkage_matrix = None
            if apply_tendency_gate and not pca_tendency_summary.get("gate_passed", True):
                pca_cluster_labels = np.full(len(valid_indices), -1, dtype=int)
                pca_linkage_matrix = None
                LOG.info(
                    "Skipping PCA clustering for %s %s because Hopkins=%.4f is below threshold %.4f.",
                    model_name,
                    split,
                    float(pca_tendency_summary.get("hopkins") or float("nan")),
                    float(pca_tendency_summary.get("hopkins_threshold") or hopkins_threshold),
                )
            else:
                try:
                    pca_cluster_labels, pca_linkage_matrix = _cluster_latent_embeddings(
                        pca_latent,
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
                    LOG.warning("PCA clustering failed for %s %s: %s", model_name, split, exc)
                    pca_cluster_labels = np.full(len(valid_indices), -1, dtype=int)
                    pca_linkage_matrix = None

            if cluster_method.lower().strip() == "agglomerative" and pca_linkage_matrix is not None:
                pca_dendrogram_path = output_dir / f"latent_{pca_tag}_{split}_dendrogram.png"
                pca_dendrogram_saved = _plot_latent_dendrogram(
                    pca_latent,
                    title=f"{_model_short_name(model_name)} {split} PCA dendrogram",
                    out_png=pca_dendrogram_path,
                    random_state=random_state,
                    linkage_method=agglomerative_linkage,
                )
                if pca_dendrogram_saved is not None:
                    saved_paths.append(str(pca_dendrogram_saved))

            def _save_pca_embedding(
                embedding: np.ndarray,
                *,
                embedding_type: str,
                title: str,
            ) -> None:
                pca_df = _build_latent_dataframe(
                    sample_id=valid_indices,
                    embedding=embedding,
                    embedding_type=embedding_type,
                    label=label_values,
                    cluster=pca_cluster_labels,
                    recon_error=pca_recon_error,
                )
                pca_df["model_name"] = model_name
                pca_df["split"] = split
                pca_df["representation"] = "pca_baseline"

                if pca_recon_error is not None:
                    pca_recon_arr = np.asarray(pca_recon_error, dtype=np.float64)
                    pca_valid_recon = np.isfinite(pca_recon_arr)
                    pca_threshold = (
                        float(np.nanquantile(pca_recon_arr[pca_valid_recon], anomaly_quantile))
                        if np.any(pca_valid_recon)
                        else np.nan
                    )
                    pca_df["anomaly_score"] = pca_df["recon_error_norm"]
                    pca_df["anomaly_flag"] = (
                        pca_df["recon_error"].astype("float64") >= pca_threshold
                    ) if np.isfinite(pca_threshold) else False
                    pca_df["anomaly_threshold"] = pca_threshold

                pca_quality_payload = {
                    "model_name": model_name,
                    "split": split,
                    "representation": "pca_baseline",
                    "pca": {"n_components": pca_components_used},
                    "embedding_type": embedding_type,
                    "cluster_method": cluster_method,
                    "cluster_tendency": {
                        key: value
                        for key, value in pca_tendency_summary.items()
                        if key not in {"vat_matrix", "ivat_matrix"}
                    },
                    "quality": _compute_latent_quality_metrics(
                        pca_latent,
                        embedding,
                        pca_cluster_labels,
                        n_neighbors=latent_quality_n_neighbors,
                    ),
                }

                pca_parquet = output_dir / f"latent_{pca_tag}_{split}_{embedding_type}.parquet"
                pca_df.to_parquet(pca_parquet, index=False)
                saved_paths.append(str(pca_parquet))

                pca_quality_path = output_dir / f"latent_{pca_tag}_{split}_{embedding_type}_quality.json"
                with pca_quality_path.open("w", encoding="utf-8") as f:
                    json.dump(pca_quality_payload, f, indent=2)
                saved_paths.append(str(pca_quality_path))

                pca_png = output_dir / f"latent_{pca_tag}_{split}_{embedding_type}.png"
                _plot_latent_matplotlib(
                    pca_df,
                    title=title,
                    out_png=pca_png,
                    color_by=color_by,
                )
                saved_paths.append(str(pca_png))

                pca_html = output_dir / f"latent_{pca_tag}_{split}_{embedding_type}.html"
                pca_html_saved = _plot_latent_interactive(
                    pca_df,
                    title=title,
                    out_html=pca_html,
                    color_by=color_by,
                    random_state=random_state,
                    hover_sample_frac=interactive_hover_sample_frac,
                    datashader_threshold=interactive_threshold,
                )
                if pca_html_saved is not None:
                    saved_paths.append(str(pca_html_saved))

            try:
                import umap

                pca_umap = umap.UMAP(
                    n_neighbors=15,
                    min_dist=0.1,
                    n_components=2,
                    random_state=random_state,
                ).fit_transform(pca_latent)
                _save_pca_embedding(
                    pca_umap,
                    embedding_type="umap",
                    title=f"{_model_short_name(model_name)} {split} PCA baseline UMAP",
                )
            except ImportError:
                LOG.warning("umap-learn is not installed; skipping PCA UMAP for %s.", model_name)
            except Exception as exc:
                LOG.warning("PCA UMAP failed for %s: %s", model_name, exc)

            try:
                from sklearn.manifold import TSNE

                pca_perplexity = float(min(30, max(5, pca_latent.shape[0] // 10)))
                pca_tsne_kwargs = {
                    "n_components": 2,
                    "perplexity": pca_perplexity,
                    "learning_rate": 200,
                    "random_state": random_state,
                    "init": "pca",
                }
                try:
                    pca_tsne = TSNE(max_iter=1000, **pca_tsne_kwargs)
                except TypeError:
                    pca_tsne = TSNE(n_iter=1000, **pca_tsne_kwargs)
                pca_tsne_embedding = pca_tsne.fit_transform(pca_latent)
                _save_pca_embedding(
                    pca_tsne_embedding,
                    embedding_type="tsne",
                    title=f"{_model_short_name(model_name)} {split} PCA baseline t-SNE",
                )
            except Exception as exc:
                LOG.warning("PCA t-SNE failed for %s: %s", model_name, exc)

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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning("matplotlib is required for unsupervised reconstruction visualization.")
        return []

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

        y_true_cols = sorted([c for c in df.columns if c.startswith("y_true_")])
        y_pred_cols = sorted([c for c in df.columns if c.startswith("y_pred_")])
        if not y_true_cols or not y_pred_cols:
            continue

        n_cols = min(len(y_true_cols), len(y_pred_cols))
        y_true = df[y_true_cols[:n_cols]].to_numpy(dtype=np.float64)
        y_pred = df[y_pred_cols[:n_cols]].to_numpy(dtype=np.float64)
        residual = y_pred - y_true
        abs_residual = np.abs(residual)
        sample_error = np.sqrt(np.mean(np.square(residual), axis=1))

        if len(sample_error) == 0:
            continue

        model_key_safe = model_name.replace(" ", "_")
        out_prefix = output_dir / f"reconstruction_{model_key_safe}_{split}"

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


def viz_run(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    output_indices: Optional[List[int]] = None,
    model_display: Optional[Dict[str, str]] = None,
    output_display: Optional[Dict[str, str]] = None,
    xlabel_prefix: str = "Ground Truth",
    ylabel_prefix: str = "Prediction",
    axis_lim: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
    include_hpo: bool = True,
    hpo_plot_metric: Optional[str] = None,
    include_shap: bool = False,
    shap_split: str = "val",
    shap_max_samples: int = 500,
    shap_output_index: Optional[int] = None,
    include_datastreamset_eval: bool = False,
    datastreamset_size: int = 50000,
    datastreamset_max: int = 10,
    datastreamset_eval_set: Optional[str] = None,
    unsupervised_latent: Optional[Dict[str, Any]] = None,
    unsupervised_reconstruction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate inference comparison plots for a SURGE run directory.

    Works for any run (M3DC1, XGC, etc.) by auto-detecting outputs from
    prediction CSV columns (y_true_00, y_pred_00, ...).

    Parameters
    ----------
    include_shap : bool, default=False
        If True, run SHAP analysis for tree models and save to plots/shap_{model}/.
    shap_split : str, default="val"
        Split to use for SHAP (train, val, or test).
    shap_max_samples : int, default=500
        Max samples for SHAP (smaller = faster).
    include_datastreamset_eval : bool, default=False
        If True, evaluate models on held-out dataset datastreamsets (XGC only).
    datastreamset_size : int, default=50000
        Rows per datastreamset for datastreamset evaluation.
    datastreamset_max : int, default=10
        Max datastreamsets to evaluate.
    datastreamset_eval_set : str, optional
        Override set_name for datastreamset eval (e.g. set2_beta0p5 for cross-set eval).

    Returns
    -------
    dict
        R² results and paths to saved plots.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(run_dir, datasets=("train", "val", "test"))
    if not predictions:
        raise FileNotFoundError(
            f"No predictions found in {run_dir / 'predictions'}. "
            "Run a SURGE workflow first."
        )

    models = list(predictions.keys())
    n_outputs = 0
    for m in models:
        for ds in ("train", "val", "test"):
            if ds in predictions[m]:
                n_outputs = max(n_outputs, len(predictions[m][ds]))
                break

    if output_indices is None:
        output_indices = list(range(n_outputs))

    model_display = model_display or {}
    for k, v in DEFAULT_MODEL_DISPLAY.items():
        if k not in model_display:
            model_display[k] = v

    output_names = [f"output_{i}" for i in output_indices]
    output_display = output_display or {f"output_{i}": f"output_{i}" for i in output_indices}

    all_r2: Dict[str, Any] = {}
    saved_paths: List[Path] = []
    task_type = _resolve_task_type(run_dir)

    if task_type == "unsupervised":
        latent_paths = viz_unsupervised_latent(
            run_dir,
            output_dir,
            **(unsupervised_latent or {}),
        )
        reconstruction_paths = viz_unsupervised_reconstruction(
            run_dir,
            output_dir,
            **(unsupervised_reconstruction or {}),
        )
        return {"r2": {}, "saved_paths": [*latent_paths, *reconstruction_paths]}

    is_classification = task_type == "classification"

    if not is_classification:
        for out_idx in output_indices:
            out_name = f"output_{out_idx}"
            results_dict = {out_name: {}}
            for model_name in models:
                results_dict[out_name][model_name] = {}
                for ds in ("train", "val", "test"):
                    if ds in predictions[model_name] and out_idx in predictions[model_name][ds]:
                        results_dict[out_name][model_name][ds] = predictions[model_name][ds][
                            out_idx
                        ]

            if not results_dict[out_name]:
                continue

            save_path = output_dir / f"inference_comparison_{out_name}.png"
            fig, axes, r2_results = plot_inference_comparison_grid(
                results_dict,
                units={out_name: "a.u."},
                save_path=save_path,
                dpi=dpi,
                xlabel_prefix=xlabel_prefix,
                ylabel_prefix=ylabel_prefix,
                ylabel_include_model=False,
                output_display_names=output_display,
                model_display_names=model_display,
                title_include_model_dataset=True,
                layout="models_rows",
                axis_lim=axis_lim,
            )
            all_r2[out_name] = r2_results
            saved_paths.append(save_path)

        # Combined grid if multiple outputs
        if len(output_indices) > 1:
            combined_dict = {}
            for out_idx in output_indices:
                out_name = f"output_{out_idx}"
                combined_dict[out_name] = {}
                for model_name in models:
                    if model_name in predictions:
                        combined_dict[out_name][model_name] = {}
                        for ds in ("train", "val", "test"):
                            if (
                                ds in predictions[model_name]
                                and out_idx in predictions[model_name][ds]
                            ):
                                combined_dict[out_name][model_name][ds] = predictions[
                                    model_name
                                ][ds][out_idx]

            if len(combined_dict) > 1:
                save_path = output_dir / "inference_comparison_grid.png"
                fig, axes, r2_results = plot_inference_comparison_grid(
                    combined_dict,
                    units={k: "a.u." for k in combined_dict},
                    save_path=save_path,
                    dpi=dpi,
                    xlabel_prefix=xlabel_prefix,
                    ylabel_prefix=ylabel_prefix,
                    ylabel_include_model=False,
                    output_display_names=output_display,
                    model_display_names=model_display,
                    title_include_model_dataset=True,
                    layout="outputs_rows",
                    axis_lim=axis_lim,
                )
                all_r2["combined"] = r2_results
                saved_paths.append(save_path)
        elif len(saved_paths) == 1:
            # Single output: also save as inference_comparison_grid.png
            import shutil

            grid_path = output_dir / "inference_comparison_grid.png"
            shutil.copy(saved_paths[0], grid_path)
            saved_paths.append(grid_path)

    if is_classification:
        cls_paths = viz_classification(run_dir, output_dir)
        saved_paths.extend(Path(p) for p in cls_paths)

    # HPO convergence plots
    if include_hpo:
        hpo_paths = viz_hpo(run_dir, output_dir, dpi=dpi, plot_metric=hpo_plot_metric)
        saved_paths.extend(hpo_paths)

    # SHAP analysis
    if include_shap:
        shap_paths = viz_shap(
            run_dir,
            output_dir,
            split=shap_split,
            max_samples=shap_max_samples,
            output_index=shap_output_index,
            dpi=dpi,
        )
        saved_paths.extend(shap_paths)

    # Segment-based evaluation (XGC only)
    datastreamset_result: Optional[Dict[str, Any]] = None
    if include_datastreamset_eval:
        try:
            datastreamset_result = viz_datastreamset_evaluation(
                run_dir,
                output_dir,
                datastreamset_size=datastreamset_size,
                max_datastreamsets=datastreamset_max,
                eval_set_name=datastreamset_eval_set,
            )
        except Exception as e:
            LOG.warning("Segment evaluation failed: %s", e)

    out: Dict[str, Any] = {"r2": all_r2, "saved_paths": [str(p) for p in saved_paths]}
    if datastreamset_result:
        out["datastreamset_eval"] = datastreamset_result
    return out


def viz_shap(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    models: Optional[List[str]] = None,
    split: str = "val",
    max_samples: int = 500,
    max_background: int = 100,
    plot_type: str = "both",
    kernel_nsamples: int = 50,
    save_dependence: bool = True,
    save_waterfall: bool = True,
    save_force: bool = False,
    dpi: int = 150,
    tree_only: bool = False,
    output_index: Optional[int] = None,
) -> List[str]:
    """
    Run SHAP analysis for models in a SURGE run directory.

    Loads dataset, scalers, and models from the run; uses prediction indices
    to get the correct X for the chosen split. Saves SHAP plots to
    output_dir/shap_{model_name}/.

    Parameters
    ----------
    run_dir : Path
        SURGE workflow run directory (contains spec.yaml, models/, scalers/, etc.).
    output_dir : Path, optional
        Where to save plots. Default: run_dir/plots.
    models : list of str, optional
        Model names to run SHAP for. If None, uses all models in the run.
    split : str, default="val"
        Which split to explain: "train", "val", or "test".
    max_samples : int, default=500
        Max samples to explain (for speed).
    max_background : int, default=100
        Background dataset size for explainers.
    plot_type : str, default="both"
        Summary: "bar", "dot", or "both".
    kernel_nsamples : int, default=50
        nsamples for KernelExplainer (when tree_only=False).
    save_dependence : bool, default=True
        Save dependence plots for top features.
    save_waterfall : bool, default=True
        Save waterfall plot.
    save_force : bool, default=False
        Save force plot.
    dpi : int, default=150
        Plot resolution.
    tree_only : bool, default=False
        If True, only run SHAP for tree models (skip KernelExplainer for non-tree).
    output_index : int, optional
        For multi-output models, which output to explain (0-based). For XGC A_parallel,
        output_1 = A_parallel. If None and dataset_format is xgc with 2 outputs, uses 1.

    Returns
    -------
    list of str
        Paths to saved SHAP plots.
    """
    from ..dataset import SurrogateDataset
    from ..workflow.spec import SurrogateWorkflowSpec
    from .importance import SHAP_AVAILABLE, _is_tree_model, run_shap_analysis

    if not SHAP_AVAILABLE:
        LOG.warning("SHAP not installed. Skip viz_shap. Install with: pip install shap")
        return []

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_file = run_dir / "spec.yaml"
    summary_file = run_dir / "workflow_summary.json"
    if not spec_file.exists():
        LOG.warning("No spec.yaml in %s. Skip viz_shap.", run_dir)
        return []
    if not summary_file.exists():
        LOG.warning("No workflow_summary.json in %s. Skip viz_shap.", run_dir)
        return []

    if not YAML_AVAILABLE:
        LOG.warning("PyYAML required for viz_shap.")
        return []

    with spec_file.open("r", encoding="utf-8") as f:
        spec_dict = yaml.safe_load(f)
    spec = SurrogateWorkflowSpec.from_dict(spec_dict)

    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    # Infer output_index for XGC A_parallel: output_1 is A_parallel
    shap_output_index = output_index
    if shap_output_index is None:
        fmt = (spec.dataset_format or "").lower()
        n_out = summary.get("dataset", {}).get("n_outputs", 0)
        if fmt == "xgc" and n_out >= 2:
            shap_output_index = 1

    dataset_path = Path(summary.get("dataset", {}).get("file_path", spec.dataset_path))
    if not dataset_path.exists():
        LOG.warning("Dataset path %s not found. Skip viz_shap.", dataset_path)
        return []

    # Load dataset
    dataset = SurrogateDataset.from_path(
        dataset_path,
        format=spec.dataset_format,
        metadata_path=spec.metadata_path,
        sample=spec.sample_rows,
        analyzer_kwargs={"hints": spec.metadata_overrides, **spec.analyzer},
    )
    if not dataset.input_columns or dataset.df is None:
        LOG.warning("Dataset has no input columns. Skip viz_shap.")
        return []

    # Load input scaler
    scalers_dir = run_dir / "scalers"
    input_scaler = None
    if spec.standardize_inputs and (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")

    model_entries = summary.get("models", [])
    if not model_entries:
        LOG.warning("No models in workflow summary. Skip viz_shap.")
        return []

    model_names = models or [m["name"] for m in model_entries]
    saved_paths: List[str] = []

    for model_entry in model_entries:
        name = model_entry.get("name")
        if name not in model_names:
            continue
        model_path = Path(model_entry.get("artifacts", {}).get("model", ""))
        if not model_path or not model_path.exists():
            if not model_path.is_absolute():
                model_path = run_dir / "models" / Path(str(model_path)).name
        if not model_path.exists():
            LOG.warning("Model %s not found at %s. Skip.", name, model_path)
            continue

        pred_files = model_entry.get("artifacts", {}).get("predictions", {})
        pred_path = pred_files.get(split)
        if not pred_path:
            LOG.warning("No %s predictions for %s. Skip SHAP.", split, name)
            continue
        pred_path = Path(pred_path)
        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        if not pred_path.exists():
            for ext in (".csv", ".parquet"):
                alt = run_dir / "predictions" / f"{name}_{split}{ext}"
                if alt.exists():
                    pred_path = alt
                    break
        if not pred_path.exists():
            LOG.warning("Predictions for %s %s not found. Skip SHAP.", name, split)
            continue

        # Load indices from predictions
        if pred_path.suffix == ".parquet":
            pred_df = pd.read_parquet(pred_path)
        else:
            pred_df = pd.read_csv(pred_path)
        if "index" not in pred_df.columns:
            LOG.warning("Predictions %s has no index column. Skip SHAP.", pred_path)
            continue
        indices = pred_df["index"].values.astype(int)
        indices = indices[:max_samples]

        # Get X from dataset (indices are row positions from the original df)
        X_raw = dataset.df[dataset.input_columns].iloc[indices].values.astype(np.float64)
        if input_scaler is not None:
            X = input_scaler.transform(X_raw)
        else:
            X = X_raw

        X_background = X[: min(max_background, len(X))]
        X = X[:max_samples]

        # Load model (with compat for sklearn 1.3.x and PyTorch pickle protocol)
        try:
            adapter = load_model_compat(model_path, model_entry)
        except Exception as e:
            LOG.warning("Failed to load model %s: %s. Skip SHAP.", name, e)
            continue

        shap_dir = output_dir / f"shap_{name}"
        shap_dir.mkdir(parents=True, exist_ok=True)

        use_tree = _is_tree_model(adapter)
        use_kernel = not tree_only or use_tree

        try:
            results = run_shap_analysis(
                adapter,
                X,
                X_background=X_background,
                output_index=shap_output_index,
                feature_names=dataset.input_columns,
                data_source=dataset,
                use_tree=use_tree,
                use_kernel=use_kernel,
                kernel_nsamples=kernel_nsamples,
                save_dir=shap_dir,
                dpi=dpi,
                plot_type=plot_type,
                save_dependence=save_dependence,
                save_waterfall=save_waterfall,
                save_force=save_force,
            )
            saved_paths.extend(results.get("saved_paths", []))
            if "tree_error" in results:
                LOG.warning("SHAP tree for %s: %s", name, results["tree_error"])
            if "kernel_error" in results:
                LOG.warning("SHAP kernel for %s: %s", name, results["kernel_error"])
        except Exception as e:
            LOG.warning("SHAP failed for %s: %s", name, e)

    return saved_paths


def _load_or_build_train_ranges(
    run_dir: Path,
    spec: Any,
    dataset_path: Path,
    summary: Dict[str, Any],
    input_scaler: Optional[Any],
    output_scaler: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Load train_data_ranges.json or build from training predictions if missing."""
    from ..dataset import SurrogateDataset

    ranges_file = run_dir / "train_data_ranges.json"
    if ranges_file.exists():
        with ranges_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: build from training predictions (for runs before this feature)
    pred_files = list((run_dir / "predictions").glob("*_train.csv"))
    if not pred_files:
        LOG.warning("No train_data_ranges.json and no train predictions. Skip in-range check.")
        return None

    pred_df = pd.read_csv(pred_files[0])
    if "index" not in pred_df.columns:
        return None
    indices = pred_df["index"].values.astype(int)

    try:
        # Use same sample as workflow so indices match
        sample = getattr(spec, "sample_rows", None)
        hints = dict(spec.metadata_overrides or {})
        hints["random_state"] = getattr(spec, "seed", 42)
        dataset = SurrogateDataset.from_path(
            dataset_path,
            format=spec.dataset_format,
            metadata_path=spec.metadata_path,
            sample=sample,
            sample_random_state=getattr(spec, "seed", 42),
            analyzer_kwargs={"hints": hints, **(spec.analyzer or {})},
        )
    except Exception as e:
        LOG.warning("Could not load dataset for train ranges: %s", e)
        return None

    if dataset.df is None or len(dataset.df) == 0:
        return None

    # Use spec's input subset (n_inputs or input_columns) to match model/scaler
    spec_input_cols = getattr(spec, "input_columns", None)
    spec_n_inputs = getattr(spec, "n_inputs", None)
    if spec_input_cols and len(spec_input_cols) > 0:
        eff_input_cols = [c for c in spec_input_cols if c in dataset.df.columns]
    elif spec_n_inputs and dataset.input_columns:
        eff_input_cols = dataset.input_columns[: spec_n_inputs]
    else:
        eff_input_cols = dataset.input_columns

    # Get rows at train indices (indices are row positions in the sampled df)
    n_rows = len(dataset.df)
    valid = (indices >= 0) & (indices < n_rows)
    if not np.any(valid):
        return None
    idx = indices[valid]
    X = dataset.df[eff_input_cols].iloc[idx].values.astype(np.float64)
    y = dataset.df[dataset.output_columns].iloc[idx].values.astype(np.float64)
    if input_scaler is not None:
        X = input_scaler.transform(X)
    if output_scaler is not None:
        y = output_scaler.transform(y)

    ranges = {
        "inputs": {
            "columns": eff_input_cols,
            "min": X.min(axis=0).tolist(),
            "max": X.max(axis=0).tolist(),
        },
        "outputs": {
            "columns": dataset.output_columns,
            "min": y.min(axis=0).tolist(),
            "max": y.max(axis=0).tolist(),
        },
    }
    # Save for future use
    with (run_dir / "train_data_ranges.json").open("w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=2)
    LOG.info("Built and saved train_data_ranges.json from training predictions")
    return ranges


# Drift detection policy (thresholds below; document in workflow outputs)
DRIFT_R2_DROP_THRESHOLD = 0.1
DRIFT_RMSE_RATIO_THRESHOLD = 1.5


def _apply_drift_detection(
    results: Dict[str, Any],
    run_dir: Path,
    output_key: str = "output_1",
    r2_drop_threshold: float = DRIFT_R2_DROP_THRESHOLD,
    rmse_ratio_threshold: float = DRIFT_RMSE_RATIO_THRESHOLD,
) -> None:
    """
    Apply drift detection policy: set drift_detected and ood_triggers per datastreamset,
    and overall drift_warning. Drives continual learning recommendations.
    """
    ref_r2 = None
    ref_rmse = None
    if (run_dir / "workflow_summary.json").exists():
        try:
            with (run_dir / "workflow_summary.json").open() as f:
                wf = json.load(f)
            for m in wf.get("models", []):
                train = m.get("metrics", {}).get("train", {})
                ref_r2 = train.get("r2")
                ref_rmse = train.get("rmse")
                if ref_r2 is not None or ref_rmse is not None:
                    break
        except Exception:
            pass

    datastreamsets_with_drift = []
    for datastreamset_key, datastreamset in results.get("datastreamsets", {}).items():
        triggers = []
        in_range = datastreamset.get("in_range", {})

        # OOD input trigger
        if not in_range.get("inputs_in_range", True):
            triggers.append("input_ood")

        # OOD output trigger
        if not in_range.get("outputs_in_range", True):
            triggers.append("output_ood")

        # Accuracy drop trigger (use first model with metrics)
        for model_name, mdata in datastreamset.get("models", {}).items():
            out_data = mdata.get(output_key, mdata.get("output_0", {}))
            seg_r2 = out_data.get("r2")
            seg_rmse = out_data.get("rmse")
            if seg_r2 is not None and ref_r2 is not None:
                if seg_r2 < ref_r2 - r2_drop_threshold:
                    triggers.append("accuracy_drop")
                    break
            if seg_rmse is not None and ref_rmse is not None and ref_rmse > 0:
                if seg_rmse > ref_rmse * rmse_ratio_threshold:
                    triggers.append("accuracy_drop")
                    break

        drift_detected = len(triggers) > 0
        datastreamset["drift_detected"] = drift_detected
        datastreamset["ood_triggers"] = triggers
        if drift_detected:
            datastreamsets_with_drift.append(datastreamset_key)

    results["drift_detection"] = {
        "policy": "surge.viz.run_viz drift thresholds (DRIFT_R2_DROP_THRESHOLD, DRIFT_RMSE_RATIO_THRESHOLD)",
        "drift_warning": len(datastreamsets_with_drift) > 0,
        "datastreamsets_with_drift": datastreamsets_with_drift,
        "n_datastreamsets_evaluated": len(results.get("datastreamsets", {})),
        "continual_learning_recommendation": (
            "DRIFT DETECTED: Evaluation data is outside training regime or prediction "
            "accuracy has dropped. Recommend ingesting new data from these regions and "
            "retraining the surrogate."
            if datastreamsets_with_drift
            else None
        ),
    }


def _check_datastreamset_in_range(
    X: np.ndarray,
    y: np.ndarray,
    train_ranges: Dict[str, Any],
    input_columns: List[str],
    output_columns: List[str],
) -> Dict[str, Any]:
    """Check if datastreamset min/max fall within training data ranges."""
    in_min = np.array(train_ranges["inputs"]["min"], dtype=np.float64)
    in_max = np.array(train_ranges["inputs"]["max"], dtype=np.float64)
    out_min = np.array(train_ranges["outputs"]["min"], dtype=np.float64)
    out_max = np.array(train_ranges["outputs"]["max"], dtype=np.float64)

    X_min, X_max = X.min(axis=0), X.max(axis=0)
    y_min, y_max = y.min(axis=0), y.max(axis=0)

    # Out of range: datastreamset extends beyond training bounds
    inputs_low = X_min < in_min
    inputs_high = X_max > in_max
    outputs_low = y_min < out_min
    outputs_high = y_max > out_max

    n_in = int(np.sum(inputs_low | inputs_high))
    n_out = int(np.sum(outputs_low | outputs_high))
    inputs_in_range = n_in == 0
    outputs_in_range = n_out == 0

    out_cols = []
    for i in range(len(output_columns)):
        if outputs_low[i] or outputs_high[i]:
            out_cols.append(output_columns[i])
    in_cols = []
    for i in range(len(input_columns)):
        if inputs_low[i] or inputs_high[i]:
            in_cols.append(input_columns[i])

    # Per-OOD-input min/max for plotting range evolution across datastreamsets
    input_min_max_ood = {}
    for col in in_cols:
        i = input_columns.index(col)
        input_min_max_ood[col] = {
            "min": float(X_min[i]),
            "max": float(X_max[i]),
            "train_min": float(in_min[i]),
            "train_max": float(in_max[i]),
        }

    return {
        "inputs_in_range": inputs_in_range,
        "outputs_in_range": outputs_in_range,
        "n_inputs_out_of_range": n_in,
        "n_outputs_out_of_range": n_out,
        "inputs_out_of_range": in_cols[:20],  # cap for readability
        "outputs_out_of_range": out_cols,
        "input_min_max_ood": input_min_max_ood,
    }


def viz_datastreamset_evaluation(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    *,
    datastreamset_size: int = 50000,
    max_datastreamsets: int = 10,
    datastreamset_offset: int = 0,
    eval_set_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained models on held-out datastreamsets of the dataset.

    Loads contiguous row ranges (datastreamsets) from the full dataset and computes
    R², RMSE, MAE per datastreamset. Use to test generalization to unseen data.

    Only supported for XGC format (uses row_range). The workflow's train/val/test
    come from a random sample; these datastreamsets are different contiguous regions.

    Parameters
    ----------
    run_dir : Path
        SURGE workflow run directory.
    output_dir : Path, optional
        Where to save datastreamset_stats.json. Default: run_dir/plots.
    datastreamset_size : int, default=50000
        Rows per datastreamset.
    max_datastreamsets : int, default=10
        Max number of datastreamsets to evaluate (datastreamset 0 = rows 0:datastreamset_size, etc.).
    datastreamset_offset : int, default=0
        Start from this datastreamset index (datastreamset_offset * datastreamset_size).
    eval_set_name : str, optional
        Override set_name when loading datastreamsets (e.g. set2_beta0p5 for cross-set eval).
        If None, uses spec.metadata_overrides.set_name (train set).

    Returns
    -------
    dict
        Per-model, per-datastreamset R², RMSE, MAE; and path to saved JSON.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from ..dataset import SurrogateDataset

    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_file = run_dir / "spec.yaml"
    summary_file = run_dir / "workflow_summary.json"
    if not spec_file.exists() or not summary_file.exists():
        raise FileNotFoundError(
            f"Run dir must have spec.yaml and workflow_summary.json: {run_dir}"
        )

    if not YAML_AVAILABLE:
        raise ImportError("PyYAML required for viz_datastreamset_evaluation.")

    with spec_file.open("r", encoding="utf-8") as f:
        spec_dict = yaml.safe_load(f)
    from ..workflow.spec import SurrogateWorkflowSpec

    spec = SurrogateWorkflowSpec.from_dict(spec_dict)
    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    dataset_path = Path(summary.get("dataset", {}).get("file_path", spec.dataset_path))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    fmt = (spec.dataset_format or "").lower()
    if fmt != "xgc":
        LOG.warning("viz_datastreamset_evaluation only supports xgc format. Skip.")
        return {}

    scalers_dir = run_dir / "scalers"
    input_scaler = None
    output_scaler = None
    if spec.standardize_inputs and (scalers_dir / "inputs.joblib").exists():
        input_scaler = joblib.load(scalers_dir / "inputs.joblib")
    if spec.standardize_outputs and (scalers_dir / "outputs.joblib").exists():
        output_scaler = joblib.load(scalers_dir / "outputs.joblib")

    model_entries = summary.get("models", [])
    if not model_entries:
        raise ValueError("No models in workflow summary.")

    # Load training data ranges for in-distribution checks
    train_ranges = _load_or_build_train_ranges(
        run_dir, spec, dataset_path, summary, input_scaler, output_scaler
    )
    train_set_name = (spec.metadata_overrides or {}).get("set_name", "set1")
    effective_eval_set = eval_set_name if eval_set_name is not None else train_set_name
    if train_ranges:
        results: Dict[str, Any] = {
            "datastreamset_size": datastreamset_size,
            "train_ranges_path": str(run_dir / "train_data_ranges.json"),
            "train_set_name": train_set_name,
            "eval_set_name": effective_eval_set,
            "datastreamsets": {},
        }
    else:
        results = {
            "datastreamset_size": datastreamset_size,
            "train_set_name": train_set_name,
            "eval_set_name": effective_eval_set,
            "datastreamsets": {},
        }

    for datastreamset_idx in range(datastreamset_offset, datastreamset_offset + max_datastreamsets):
        start = datastreamset_idx * datastreamset_size
        end = start + datastreamset_size
        datastreamset_key = f"datastreamset_{datastreamset_idx}_rows_{start}_{end}"

        try:
            hints = dict(spec.metadata_overrides or {})
            hints["set_name"] = effective_eval_set
            hints["row_range"] = (start, end)
            dataset = SurrogateDataset.from_path(
                dataset_path,
                format=spec.dataset_format,
                metadata_path=spec.metadata_path,
                sample=None,
                analyzer_kwargs={"hints": hints, **(spec.analyzer or {})},
            )
        except Exception as e:
            LOG.warning("Failed to load %s: %s", datastreamset_key, e)
            continue

        if dataset.df is None or len(dataset.df) == 0:
            LOG.warning("%s empty. Skip.", datastreamset_key)
            continue

        # Use spec's input subset (n_inputs or input_columns) to match model/scaler
        spec_input_cols = getattr(spec, "input_columns", None)
        spec_n_inputs = getattr(spec, "n_inputs", None)
        if spec_input_cols and len(spec_input_cols) > 0:
            eff_input_cols = [c for c in spec_input_cols if c in dataset.df.columns]
        elif spec_n_inputs and dataset.input_columns:
            eff_input_cols = dataset.input_columns[: spec_n_inputs]
        else:
            eff_input_cols = dataset.input_columns
        X = dataset.df[eff_input_cols].values.astype(np.float64)
        y_true = dataset.df[dataset.output_columns].values.astype(np.float64)
        if input_scaler is not None:
            X = input_scaler.transform(X)
        if output_scaler is not None:
            y_true = output_scaler.transform(y_true)

        datastreamset_entry: Dict[str, Any] = {"n_samples": len(dataset.df), "models": {}}

        # In-distribution check: are datastreamset min/max within training ranges?
        if train_ranges:
            in_range = _check_datastreamset_in_range(
                X, y_true, train_ranges, eff_input_cols, dataset.output_columns
            )
            datastreamset_entry["in_range"] = in_range

        results["datastreamsets"][datastreamset_key] = datastreamset_entry

        for model_entry in model_entries:
            name = model_entry.get("name")
            model_path = Path(model_entry.get("artifacts", {}).get("model", ""))
            if not model_path or not model_path.exists():
                model_path = run_dir / "models" / Path(str(model_path)).name
            if not model_path.exists():
                continue

            try:
                adapter = load_model_compat(model_path, model_entry)
                y_pred = adapter.predict(X)
                if hasattr(y_pred, "numpy"):
                    y_pred = y_pred.numpy()
                y_pred = np.asarray(y_pred)
            except Exception as e:
                LOG.warning("Model %s failed on %s: %s", name, datastreamset_key, e)
                continue

            n_out = y_true.shape[1]
            metrics = {}
            for i in range(n_out):
                r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
                rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
                mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
                resid = y_pred[:, i] - y_true[:, i]
                mean_err = float(np.mean(resid))
                std_err = float(np.std(resid))
                metrics[f"output_{i}"] = {
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "mean_error": mean_err,
                    "std_error": std_err,
                }
            results["datastreamsets"][datastreamset_key]["models"][name] = metrics

    # Apply drift detection policy (OOD + accuracy triggers)
    _apply_drift_detection(results, run_dir, output_key="output_1")

    out_path = output_dir / "datastreamset_evaluation_stats.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    results["saved_path"] = str(out_path)

    # Plot R², RMSE, and prediction error vs datastreamset index
    _plot_datastreamset_evaluation(results, output_dir, datastreamset_size, run_dir=run_dir)
    return results


def _plot_datastreamset_evaluation(
    results: Dict[str, Any],
    output_dir: Path,
    datastreamset_size: int,
    dpi: int = 150,
    run_dir: Optional[Path] = None,
) -> None:
    """Plot R², RMSE, prediction error vs datastreamset index; shade out-of-range datastreamsets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    datastreamsets_data = results.get("datastreamsets", {})
    if not datastreamsets_data:
        return

    # Parse datastreamset indices and row ranges (datastreamset_0_rows_0_50000, datastreamset_1_rows_50000_100000, ...)
    def _datastreamset_sort_key(k: str) -> int:
        parts = k.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 0

    sorted_keys = sorted(datastreamsets_data.keys(), key=_datastreamset_sort_key)
    datastreamset_indices = []
    for key in sorted_keys:
        parts = key.split("_")
        try:
            datastreamset_indices.append(int(parts[1]))
        except (IndexError, ValueError):
            datastreamset_indices.append(len(datastreamset_indices))

    models = set()
    for s in datastreamsets_data.values():
        models.update(s.get("models", {}).keys())
    models = sorted(models)
    model_display = {m: DEFAULT_MODEL_DISPLAY.get(m, m.replace("_", " ").title()) for m in models}

    # Collect R² and RMSE per model, per datastreamset, for output_1 (A_parallel)
    output_key = "output_1"  # A_parallel
    for out_k in ("output_1", "output_0"):
        if any(
            out_k in datastreamsets_data[s].get("models", {}).get(m, {})
            for s in datastreamsets_data
            for m in models
        ):
            output_key = out_k
            break

    # Check if we have in_range data and input_min_max_ood for range evolution
    has_in_range = any("in_range" in s for s in datastreamsets_data.values())
    has_input_range_evolution = has_in_range and any(
        datastreamsets_data[s].get("in_range", {}).get("input_min_max_ood")
        for s in datastreamsets_data
    )
    n_rows = 4 if has_input_range_evolution else (3 if has_in_range else 2)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=True)
    if n_rows == 2:
        ax_r2, ax_rmse = axes[0], axes[1]
        ax_inrange = None
        ax_input_range = None
    elif n_rows == 3:
        ax_r2, ax_rmse, ax_inrange = axes[0], axes[1], axes[2]
        ax_input_range = None
    else:
        ax_r2, ax_rmse, ax_inrange, ax_input_range = axes[0], axes[1], axes[2], axes[3]

    # Reference R² from train/val (workflow summary)
    ref_r2_train = None
    ref_r2_val = None
    if run_dir and (run_dir / "workflow_summary.json").exists():
        try:
            with (run_dir / "workflow_summary.json").open() as f:
                wf = json.load(f)
            for m in wf.get("models", []):
                train_r2 = m.get("metrics", {}).get("train", {}).get("r2")
                val_r2 = m.get("metrics", {}).get("val", {}).get("r2")
                if train_r2 is not None:
                    ref_r2_train = float(train_r2) if ref_r2_train is None else ref_r2_train
                if val_r2 is not None:
                    ref_r2_val = float(val_r2) if ref_r2_val is None else ref_r2_val
                break
        except Exception:
            pass

    # Out-of-range mask for shading
    oor_mask = []
    if has_in_range:
        for key in sorted_keys:
            ir = datastreamsets_data[key].get("in_range", {})
            oor_mask.append(not ir.get("inputs_in_range", True) or not ir.get("outputs_in_range", True))

    has_model_metrics = bool(models)
    for model in models:
        r2_vals = []
        rmse_vals = []
        mae_vals = []
        for key in sorted_keys:
            mdata = datastreamsets_data[key].get("models", {}).get(model, {})
            out_data = mdata.get(output_key, mdata.get("output_0", {}))
            r2_vals.append(out_data.get("r2", np.nan))
            rmse_vals.append(out_data.get("rmse", np.nan))
            mae_vals.append(out_data.get("mae", np.nan))
        if not r2_vals:
            continue
        label = model_display.get(model, model)
        ax_r2.plot(datastreamset_indices, r2_vals, "o-", label=label, linewidth=2, markersize=8)
        ax_rmse.plot(datastreamset_indices, rmse_vals, "o-", label=label, linewidth=2, markersize=8)

    # Shade out-of-range datastreamsets on R² and RMSE
    if has_in_range and oor_mask:
        for i, is_oor in enumerate(oor_mask):
            if is_oor:
                ax_r2.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")
                ax_rmse.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")

    ax_r2.set_ylabel("R² score")
    ax_r2.set_title("Generalization: R² vs held-out datastreamset (A_parallel, red = out-of-range)")
    if ref_r2_train is not None:
        ax_r2.axhline(ref_r2_train, color="gray", linestyle="--", alpha=0.7, label=f"Train R²={ref_r2_train:.3f}")
    if ref_r2_val is not None:
        ax_r2.axhline(ref_r2_val, color="gray", linestyle=":", alpha=0.7, label=f"Val R²={ref_r2_val:.3f}")
    if has_model_metrics:
        ax_r2.legend(loc="best", fontsize=9)
    else:
        ax_r2.text(0.5, 0.5, "No model metrics (retrain in current env)", ha="center", va="center", transform=ax_r2.transAxes)
    ax_r2.grid(True, alpha=0.3)
    # Zoom in on R² around train/val range for better visibility
    if has_model_metrics:
        r2_all = []
        for key in sorted_keys:
            for m in models:
                mdata = datastreamsets_data[key].get("models", {}).get(m, {})
                out_data = mdata.get(output_key, mdata.get("output_0", {}))
                v = out_data.get("r2")
                if v is not None:
                    r2_all.append(v)
        refs = [r for r in (ref_r2_train, ref_r2_val) if r is not None]
        if r2_all or refs:
            y_min = min(r2_all + refs) - 0.02 if (r2_all or refs) else 0
            y_max = max(r2_all + refs) + 0.01 if (r2_all or refs) else 1
            ax_r2.set_ylim(max(0, y_min), min(1.02, y_max))
    else:
        ax_r2.set_ylim(bottom=0)

    ax_rmse.set_xlabel("Segment index" if not has_in_range and ax_input_range is None else "")
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_title("Prediction error (RMSE) vs held-out datastreamset (A_parallel)")
    if has_model_metrics:
        ax_rmse.legend(loc="best", fontsize=9)
    else:
        ax_rmse.text(0.5, 0.5, "No model metrics (retrain in current env)", ha="center", va="center", transform=ax_rmse.transAxes)
    ax_rmse.grid(True, alpha=0.3)

    if has_in_range and ax_inrange is not None:
        in_range_inputs = []
        in_range_outputs = []
        for key in sorted_keys:
            ir = datastreamsets_data[key].get("in_range", {})
            in_range_inputs.append(ir.get("inputs_in_range", True))
            in_range_outputs.append(ir.get("outputs_in_range", True))
        x_pos = np.arange(len(datastreamset_indices))
        width = 0.35
        ax_inrange.bar(x_pos - width / 2, [1 if x else 0 for x in in_range_inputs], width, label="Inputs in range", color="steelblue", alpha=0.8)
        ax_inrange.bar(x_pos + width / 2, [1 if x else 0 for x in in_range_outputs], width, label="Outputs in range", color="coral", alpha=0.8)
        ax_inrange.set_ylabel("In range (1=yes)")
        ax_inrange.set_title("In-distribution: within training min/max?")
        ax_inrange.set_yticks([0, 1])
        ax_inrange.set_yticklabels(["No", "Yes"])
        ax_inrange.set_xticks(x_pos)
        ax_inrange.set_xticklabels([str(i) for i in datastreamset_indices])
        ax_inrange.legend(loc="best", fontsize=9)
        ax_inrange.grid(True, alpha=0.3, axis="y")
        if ax_input_range is None:
            ax_inrange.set_xlabel("Segment index")

    # Input range evolution: how min/max of OOD inputs change across datastreamsets
    if ax_input_range is not None:
        from collections import Counter
        all_ood = []
        for key in sorted_keys:
            ir = datastreamsets_data[key].get("in_range", {})
            mm = ir.get("input_min_max_ood", {})
            all_ood.extend(mm.keys())
        if all_ood:
            top_ood = [c[0] for c in Counter(all_ood).most_common(6)]
            colors = plt.cm.tab10(np.linspace(0, 1, len(top_ood)))
            ax_input_range.axhspan(0, 1, alpha=0.15, color="green", label="Training range (norm)")
            for idx, col in enumerate(top_ood):
                mins_norm, maxs_norm = [], []
                for key in sorted_keys:
                    ir = datastreamsets_data[key].get("in_range", {})
                    mm = ir.get("input_min_max_ood", {}).get(col)
                    if mm:
                        tmin, tmax = mm["train_min"], mm["train_max"]
                        span = tmax - tmin if tmax != tmin else 1.0
                        mins_norm.append((mm["min"] - tmin) / span)
                        maxs_norm.append((mm["max"] - tmin) / span)
                    else:
                        mins_norm.append(np.nan)
                        maxs_norm.append(np.nan)
                c = colors[idx]
                ax_input_range.plot(datastreamset_indices, mins_norm, "o--", color=c, markersize=5, alpha=0.9)
                ax_input_range.plot(datastreamset_indices, maxs_norm, "s-", color=c, markersize=5, alpha=0.9, label=col)
            ax_input_range.set_ylabel("Normalized value (0–1 = train range)")
            ax_input_range.set_xlabel("Datastreamset index")
            ax_input_range.set_title("Input range evolution: OOD inputs min/max vs datastreamset (top 6)")
            ax_input_range.set_ylim(-0.1, 1.5)
            ax_input_range.legend(loc="upper right", fontsize=7, ncol=2)
            ax_input_range.grid(True, alpha=0.3)
        else:
            ax_input_range.set_visible(False)

    plt.tight_layout()
    save_path = output_dir / "datastreamset_evaluation_r2_rmse.png"
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plot_paths = [str(save_path)]

    # Second plot: R² degradation (train R² - datastreamset R²) and MAE vs datastreamset
    if has_model_metrics and ref_r2_train is not None:
        fig2, (ax_degrad, ax_mae) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for model in models:
            r2_vals = []
            mae_vals = []
            for key in sorted_keys:
                mdata = datastreamsets_data[key].get("models", {}).get(model, {})
                out_data = mdata.get(output_key, mdata.get("output_0", {}))
                r2_vals.append(out_data.get("r2", np.nan))
                mae_vals.append(out_data.get("mae", np.nan))
            if not r2_vals:
                continue
            degrad = np.array([ref_r2_train - r for r in r2_vals])
            label = model_display.get(model, model)
            ax_degrad.plot(datastreamset_indices, degrad, "o-", label=label, linewidth=2, markersize=8)
            ax_mae.plot(datastreamset_indices, mae_vals, "o-", label=label, linewidth=2, markersize=8)
        if has_in_range and oor_mask:
            for i, is_oor in enumerate(oor_mask):
                if is_oor:
                    ax_degrad.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")
                    ax_mae.axvspan(datastreamset_indices[i] - 0.4, datastreamset_indices[i] + 0.4, alpha=0.2, color="red")
        ax_degrad.set_ylabel("R² degradation (train − datastreamset)")
        ax_degrad.set_title("R² drop vs held-out datastreamset (higher = worse generalization)")
        ax_degrad.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax_degrad.legend(loc="best", fontsize=9)
        ax_degrad.grid(True, alpha=0.3)
        ax_mae.set_ylabel("MAE (|pred − true|)")
        ax_mae.set_xlabel("Segment index")
        ax_mae.set_title("Mean absolute error vs held-out datastreamset")
        ax_mae.legend(loc="best", fontsize=9)
        ax_mae.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path2 = output_dir / "datastreamset_evaluation_degradation.png"
        fig2.savefig(save_path2, dpi=dpi, bbox_inches="tight")
        plot_paths.append(str(save_path2))
    results["saved_plot_path"] = plot_paths[0]
    results["saved_plot_paths"] = plot_paths
