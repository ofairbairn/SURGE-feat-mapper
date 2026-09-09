"""Feature attribution for Mapper: trace raw dataset columns through the
selected representation (PCA/AE/VAE) back to latent axes and cluster labels.

PCA is linear -- `pca.components_` is the exact recipe from raw features to
each latent axis, so it is read directly, no SHAP involved.

AE/VAE are non-linear, so a static weight matrix does not exist. Instead each
latent axis is treated as a scalar-output function of the raw input and
explained with `shap.GradientExplainer` on the fitted torch encoder (one
explainer per axis). Per-axis SHAP values are then aggregated into a
per-cluster raw-feature ranking, weighted by how much each axis separates a
given cluster from the global mean (the axes doing the separating dominate
that cluster's explanation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Mapper.progress import mapper_progress, timed_operation

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

_DEFAULT_BACKGROUND_SIZE = 100
_DEFAULT_EXPLAIN_SAMPLES = 200
_DEFAULT_MAX_AXES = 8
_DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# PCA: exact loadings, no SHAP
# ---------------------------------------------------------------------------


def _get_pca_loadings(adapter: Any) -> Optional[np.ndarray]:
    """Return the fitted (n_components, n_features) PCA loadings matrix, or None."""
    backend = getattr(adapter, "_model", None)
    sk_model = getattr(backend, "model", None)
    if sk_model is None:
        return None
    components = getattr(sk_model, "components_", None)
    return None if components is None else np.asarray(components, dtype=np.float64)


def pca_feature_report(
    adapter: Any,
    feature_names: List[str],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> Optional[Dict[str, Any]]:
    """Rank raw features per PCA axis directly from `pca.components_`."""
    loadings = _get_pca_loadings(adapter)
    if loadings is None:
        return None
    n_components, n_features = loadings.shape
    if n_features != len(feature_names):
        raise ValueError(
            f"PCA loadings have {n_features} columns but "
            f"{len(feature_names)} feature_names were given"
        )
    axes = []
    for axis in range(n_components):
        weights = loadings[axis]
        order = np.argsort(np.abs(weights))[::-1][:top_k]
        axes.append(
            {
                "axis": int(axis),
                "top_features": [
                    {"feature": feature_names[i], "loading": float(weights[i])}
                    for i in order
                ],
            }
        )
    return {
        "method": "pca_loadings",
        "n_components": int(n_components),
        "n_features": int(n_features),
        "axes": axes,
    }


# ---------------------------------------------------------------------------
# AE/VAE: per-latent-axis GradientExplainer SHAP
# ---------------------------------------------------------------------------


def _get_torch_encoder(adapter: Any) -> Optional[Tuple[Any, Any]]:
    """Return (encoder_module, device) for a fitted AE/VAE adapter, else None."""
    if not TORCH_AVAILABLE:
        return None
    backend = getattr(adapter, "_model", None)
    net = getattr(backend, "model", None)
    device = getattr(backend, "device", None)
    if net is None or not hasattr(net, "encode"):
        return None
    return net, device


if TORCH_AVAILABLE:

    class _LatentAxisEncoder(torch.nn.Module):
        """Wraps an AE/VAE encoder so forward(x) returns one latent axis as a column."""

        def __init__(self, net: Any, axis: int) -> None:
            super().__init__()
            self._net = net
            self._axis = axis

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self._net.encode(x)
            z = out[0] if isinstance(out, tuple) else out
            return z[:, self._axis : self._axis + 1]

else:  # pragma: no cover - torch not installed
    _LatentAxisEncoder = None  # type: ignore[assignment]


def encoder_axis_attributions(
    adapter: Any,
    X_scaled: np.ndarray,
    feature_names: List[str],
    *,
    max_axes: Optional[int] = _DEFAULT_MAX_AXES,
    n_background: int = _DEFAULT_BACKGROUND_SIZE,
    n_explain: int = _DEFAULT_EXPLAIN_SAMPLES,
    random_state: int = 0,
) -> Optional[Dict[str, Any]]:
    """Per-latent-axis GradientExplainer SHAP attribution of raw features.

    Returns SHAP values with shape (n_axes, n_explain, n_features) plus the
    row positions (into ``X_scaled`` / the combined latent matrix) that were
    explained, so callers can join back to cluster labels or dataset indices.
    """
    if not SHAP_AVAILABLE or not TORCH_AVAILABLE:
        return None
    encoder_info = _get_torch_encoder(adapter)
    if encoder_info is None:
        return None
    net, device = encoder_info
    n_samples, n_features = X_scaled.shape
    if len(feature_names) != n_features:
        raise ValueError(
            f"X_scaled has {n_features} columns but "
            f"{len(feature_names)} feature_names were given"
        )

    rng = np.random.default_rng(random_state)
    bg_positions = rng.choice(
        n_samples, size=min(n_background, n_samples), replace=False
    )
    explain_positions = np.sort(
        rng.choice(n_samples, size=min(n_explain, n_samples), replace=False)
    )
    background = torch.tensor(
        X_scaled[bg_positions], dtype=torch.float32, device=device
    )
    explain = torch.tensor(
        X_scaled[explain_positions], dtype=torch.float32, device=device
    )

    net.eval()
    with torch.no_grad():
        sample_out = net.encode(explain[:1])
    sample_z = sample_out[0] if isinstance(sample_out, tuple) else sample_out
    latent_dim = int(sample_z.shape[1])
    n_axes = latent_dim if max_axes is None else min(int(max_axes), latent_dim)
    axes_to_run = list(range(n_axes))

    axis_shap = []
    with timed_operation("attribution", f"GradientExplainer over {n_axes} latent axes"):
        for axis in mapper_progress(
            axes_to_run,
            stage="attribution",
            operation="latent axis SHAP",
            total=n_axes,
            unit="axis",
        ):
            wrapped = _LatentAxisEncoder(net, axis).to(device)
            explainer = shap.GradientExplainer(wrapped, background)
            sv = explainer.shap_values(explain)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
            if sv.ndim == 3:  # (n_explain, n_features, 1)
                sv = sv[:, :, 0]
            axis_shap.append(sv)

    return {
        "method": "gradient_shap",
        "latent_dim": latent_dim,
        "axes_explained": axes_to_run,
        "explain_positions": explain_positions,
        "shap_values": np.stack(axis_shap, axis=0),
        "feature_names": list(feature_names),
    }


def axis_feature_ranking(
    axis_attribution: Dict[str, Any],
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Global per-latent-axis raw-feature ranking (mean |SHAP| over explained points)."""
    shap_values = axis_attribution["shap_values"]
    feature_names = axis_attribution["feature_names"]
    ranking = []
    for i, axis in enumerate(axis_attribution["axes_explained"]):
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        ranking.append(
            {
                "axis": int(axis),
                "top_features": [
                    {"feature": feature_names[j], "mean_abs_shap": float(mean_abs[j])}
                    for j in order
                ],
            }
        )
    return ranking


def cluster_feature_importance(
    axis_attribution: Dict[str, Any],
    Z_all: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> Dict[str, Any]:
    """Aggregate per-axis SHAP into a per-cluster raw-feature ranking.

    Each explained point's per-axis SHAP row is weighted by how far that
    point sits from the global mean on that axis (normalized across axes),
    so axes that most separate a cluster from the rest dominate its
    explanation. Cluster label -1 (HDBSCAN noise) is reported like any other
    label.
    """
    positions = axis_attribution["explain_positions"]
    shap_values = axis_attribution["shap_values"]  # (n_axes, n_explain, n_features)
    axes = axis_attribution["axes_explained"]
    feature_names = axis_attribution["feature_names"]

    Z_axes = Z_all[:, axes]
    axis_mean = Z_axes.mean(axis=0)
    axis_weight = np.abs(Z_axes[positions] - axis_mean)  # (n_explain, n_axes)
    weight_sum = axis_weight.sum(axis=1, keepdims=True)
    axis_weight = np.divide(
        axis_weight, weight_sum, out=np.full_like(axis_weight, 1.0 / max(len(axes), 1)), where=weight_sum > 0
    )
    weighted = np.einsum("pa,apf->pf", axis_weight, shap_values)  # (n_explain, n_features)

    labels_explained = np.asarray(cluster_labels)[positions]
    clusters: Dict[str, Any] = {}
    for label in sorted(set(int(l) for l in labels_explained)):
        mask = labels_explained == label
        mean_abs = np.abs(weighted[mask]).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        clusters[str(label)] = {
            "n_points_explained": int(np.count_nonzero(mask)),
            "top_features": [
                {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
                for i in order
            ],
        }
    return {"method": "gradient_shap_cluster_aggregate", "clusters": clusters}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_feature_attribution(
    adapter: Any,
    selected_rung: str,
    X_scaled: np.ndarray,
    Z_all: np.ndarray,
    feature_names: List[str],
    *,
    cluster_labels: Optional[np.ndarray] = None,
    max_axes: Optional[int] = _DEFAULT_MAX_AXES,
    n_background: int = _DEFAULT_BACKGROUND_SIZE,
    n_explain: int = _DEFAULT_EXPLAIN_SAMPLES,
    top_k: int = _DEFAULT_TOP_K,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Trace raw feature contributions through the selected representation.

    PCA: exact loadings from ``pca.components_``. AE/VAE: per-latent-axis
    GradientExplainer SHAP, aggregated into per-cluster raw-feature rankings
    when ``cluster_labels`` is provided.
    """
    if selected_rung == "pca":
        report = pca_feature_report(adapter, feature_names, top_k=top_k)
        if report is None:
            return {"status": "skipped", "reason": "pca_components_unavailable"}
        report["status"] = "complete"
        report["selected_representation"] = selected_rung
        return report

    if not SHAP_AVAILABLE:
        return {"status": "skipped", "reason": "shap_not_installed"}
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "torch_not_installed"}

    axis_attribution = encoder_axis_attributions(
        adapter,
        X_scaled,
        feature_names,
        max_axes=max_axes,
        n_background=n_background,
        n_explain=n_explain,
        random_state=random_state,
    )
    if axis_attribution is None:
        return {"status": "skipped", "reason": "encoder_unavailable"}

    report: Dict[str, Any] = {
        "status": "complete",
        "selected_representation": selected_rung,
        "method": axis_attribution["method"],
        "latent_dim": axis_attribution["latent_dim"],
        "n_axes_explained": len(axis_attribution["axes_explained"]),
        "n_background": int(min(n_background, len(X_scaled))),
        "n_explained": int(len(axis_attribution["explain_positions"])),
        "axis_ranking": axis_feature_ranking(axis_attribution, top_k=top_k),
    }
    if cluster_labels is not None:
        cluster_result = cluster_feature_importance(
            axis_attribution, Z_all, np.asarray(cluster_labels), top_k=top_k
        )
        report["cluster_ranking"] = cluster_result["clusters"]
    return report


__all__ = [
    "pca_feature_report",
    "encoder_axis_attributions",
    "axis_feature_ranking",
    "cluster_feature_importance",
    "run_feature_attribution",
]
