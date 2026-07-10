"""Classification diagnostic plots (ROC, PR, confusion, calibration).

Follows SURGE_BENCHMARKS_VIZ_PLAN §3.1. Uses matplotlib + scikit-learn only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    MPL_AVAILABLE = False
    plt = None
    Figure = Any


def _ensure_mpl() -> None:
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for classification plots")


def _save_figure(fig: Any, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    stem = save_path.with_suffix("")
    if save_path.suffix.lower() == ".png":
        pdf_path = stem.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _resolve_classes(y_true: np.ndarray, y_prob: np.ndarray | None = None) -> np.ndarray:
    classes = np.unique(y_true)
    if y_prob is not None and y_prob.ndim == 2:
        n_classes = y_prob.shape[1]
        if classes.size != n_classes and np.issubdtype(classes.dtype, np.integer):
            min_class = int(classes.min())
            max_class = int(classes.max())
            if max_class - min_class + 1 == n_classes:
                classes = np.arange(min_class, min_class + n_classes)
    return classes


def plot_roc_curve(
    y_true,
    y_prob,
    *,
    labels: list[str] | None = None,
    title: str = "ROC curve",
    save_path: Path | str | None = None,
    ax=None,
) -> Any:
    """Binary: ``y_prob`` (n,). Multiclass: ``y_prob`` (n, n_classes) OvR micro/macro."""
    _ensure_mpl()
    from sklearn.metrics import RocCurveDisplay
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    classes = _resolve_classes(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.figure

    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_score = y_prob[:, 1]
        RocCurveDisplay.from_predictions(
            y_true,
            y_score,
            ax=ax,
            pos_label=classes[-1],
            name=labels[1] if labels and len(labels) > 1 else str(classes[-1]),
        )
    elif y_prob.ndim == 1:
        RocCurveDisplay.from_predictions(
            y_true,
            y_prob,
            ax=ax,
            pos_label=classes[-1],
            name=labels[1] if labels and len(labels) > 1 else str(classes[-1]),
        )
    else:
        yb = label_binarize(y_true, classes=classes)
        for c, class_label in enumerate(classes):
            if yb[:, c].sum() == 0:
                continue
            RocCurveDisplay.from_predictions(
                yb[:, c],
                y_prob[:, c],
                ax=ax,
                name=labels[c] if labels and c < len(labels) else str(class_label),
            )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    _save_figure(fig, Path(save_path) if save_path else None)
    return fig


def plot_precision_recall_curve(
    y_true,
    y_prob,
    *,
    labels: list[str] | None = None,
    title: str = "Precision–recall",
    save_path: Path | str | None = None,
    ax=None,
) -> Any:
    _ensure_mpl()
    from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    classes = _resolve_classes(y_true, y_prob)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5.5))
    else:
        fig = ax.figure

    if y_prob.ndim == 1:
        ap = average_precision_score(y_true, y_prob, pos_label=classes[-1])
        PrecisionRecallDisplay.from_predictions(
            y_true, y_prob, ax=ax, pos_label=classes[-1], name=f"AP={ap:.3f}"
        )
    else:
        if y_prob.shape[1] == 2:
            PrecisionRecallDisplay.from_predictions(
                y_true, y_prob[:, 1], ax=ax, pos_label=classes[-1]
            )
        else:
            yb = label_binarize(y_true, classes=classes)
            for c, class_label in enumerate(classes):
                if yb[:, c].sum() == 0:
                    continue
                ap = average_precision_score(yb[:, c], y_prob[:, c])
                PrecisionRecallDisplay.from_predictions(
                    yb[:, c],
                    y_prob[:, c],
                    ax=ax,
                    name=(labels[c] if labels and c < len(labels) else str(class_label)) + f" AP={ap:.2f}",
                )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    _save_figure(fig, Path(save_path) if save_path else None)
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    *,
    labels: Sequence[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
    save_path: Path | str | None = None,
    ax=None,
) -> Any:
    _ensure_mpl()
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.figure
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, Path(save_path) if save_path else None)
    return fig


def plot_calibration_curve(
    y_true,
    y_prob,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Calibration",
    save_path: Path | str | None = None,
    ax=None,
) -> Any:
    _ensure_mpl()
    from sklearn.calibration import calibration_curve
    from ..metrics import expected_calibration_error

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
    else:
        fig = ax.figure

    if y_prob.ndim == 1:
        prob_true, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)
        ax.plot(mean_pred, prob_true, "s-", label="model")
    else:
        y_hat = y_prob.argmax(axis=1)
        conf = y_prob.max(axis=1)
        correct = (y_hat == y_true).astype(float)
        prob_true, mean_pred = calibration_curve(
            correct, conf, n_bins=n_bins, strategy=strategy
        )
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)
        ax.plot(mean_pred, prob_true, "s-", label="model (max prob)")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.set_xlabel("mean predicted value")
    ax.set_ylabel("fraction of positives")
    ax.set_title(f"{title} (ECE ≈ {ece:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, Path(save_path) if save_path else None)
    return fig


def plot_classification_dashboard(
    y_true,
    y_pred,
    y_prob,
    *,
    labels: list[str] | None = None,
    model_name: str = "",
    save_path: Path | str | None = None,
) -> Any:
    """2×2: ROC, PR, confusion, calibration."""
    _ensure_mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    suptitle = "Classification dashboard"
    if model_name:
        suptitle = f"{suptitle} — {model_name}"
    fig.suptitle(suptitle)

    plot_roc_curve(y_true, y_prob, labels=labels, title="ROC", ax=axes[0, 0])
    plot_precision_recall_curve(y_true, y_prob, labels=labels, title="PR", ax=axes[0, 1])
    plot_confusion_matrix(
        y_true, y_pred, labels=labels, title="Confusion", ax=axes[1, 0]
    )
    plot_calibration_curve(y_true, y_prob, title="Calibration", ax=axes[1, 1])

    fig.tight_layout()
    _save_figure(fig, Path(save_path) if save_path else None)
    return fig


__all__ = [
    "plot_calibration_curve",
    "plot_classification_dashboard",
    "plot_confusion_matrix",
    "plot_precision_recall_curve",
    "plot_roc_curve",
]
