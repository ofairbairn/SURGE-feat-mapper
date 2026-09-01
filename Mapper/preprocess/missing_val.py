"""Missing-value detection and reporting for the SURGE Mapper pipeline.

This module mirrors the missingness detection used by
:func:`surge.preprocessing.analyze_dataset_structure` (per-column
``isnull().sum()`` counts and cell-level completeness), then extends it with
the diagnostics the rest of ``surge`` does not provide:

* per-column missing-value indices for the report,
* missingno visualisations (binary matrix, nullity-correlation heatmap,
  completeness bar) saved as PNGs,
* a Little's MCAR chi-square test and pairwise t-test matrix via
  ``pyampute.exploration.mcar_statistical_tests.MCARTest``.

The analysis is intended to run on the raw loaded dataframe (before
:meth:`surge.engine.SurrogateEngine.configure_dataframe` drops NaN rows), so
missingness is reported even when the engine later discards those rows.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _row_labels(index: pd.Index) -> list:
    """Serialize dataframe index labels into JSON-friendly scalars."""
    labels = []
    for label in index:
        if isinstance(label, (int, np.integer)):
            labels.append(int(label))
        elif isinstance(label, (float, np.floating)):
            labels.append(float(label))
        else:
            labels.append(str(label))
    return labels


def _save_missingno_plot(
    kind: str,
    df: pd.DataFrame,
    output_dir: Path,
    *,
    min_missing_columns: int = 1,
) -> Optional[Path]:
    """Render one missingno plot and save it under ``output_dir``."""
    import matplotlib
    import missingno as msno

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_missing_columns = int((df.isnull().sum() > 0).sum())
    if n_missing_columns < min_missing_columns or len(df) == 0:
        return None
    missing_columns = df.columns[df.isnull().any()]
    missing_df = df.loc[:, missing_columns]

    if kind == "matrix":
        path = output_dir / "missingno_matrix.png"
        fig, ax = plt.subplots(figsize=(14.0, 6.0))
        msno.matrix(missing_df, ax=ax, sparkline=False, fontsize=8)
        ax.set_title("nullity matrix")
    elif kind == "heatmap":
        path = output_dir / "missingno_heatmap.png"
        fig, ax = plt.subplots(figsize=(10.0, 8.0))
        msno.heatmap(missing_df, ax=ax, fontsize=8)
        ax.set_title("Missingness Correlation Heatmap")
    elif kind == "bar":
        path = output_dir / "missingno_bar.png"
        fig, ax = plt.subplots(figsize=(14.0, 6.0))
        msno.bar(missing_df, ax=ax, fontsize=8)
        ax.set_title("bar chart of completeness")
    else:
        raise ValueError(f"Unsupported missingno plot kind: {kind!r}")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_little_mcar_test(numeric_df: pd.DataFrame) -> Dict[str, Any]:
    """Little's chi-square MCAR test; None (plus reason) when not applicable."""
    result: Dict[str, Any] = {"performed": False, "little_pvalue": None}
    if numeric_df.shape[0] < 2 or numeric_df.shape[1] < 2:
        result["note"] = (
            "requires at least two rows and two numeric columns with missingness"
        )
        return result
    if int((numeric_df.isnull().sum() > 0).sum()) < 1:
        result["note"] = "no numeric column contains missing values"
        return result
    try:
        from pyampute.exploration.mcar_statistical_tests import MCARTest

        test = MCARTest(method="little")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pvalue = test.little_mcar_test(numeric_df)
        result["performed"] = True
        result["little_pvalue"] = None if pvalue is None else float(pvalue)
    except Exception as exc:  # singular covariance, single pattern, etc.
        result["little_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _run_pairwise_ttest(numeric_df: pd.DataFrame) -> Dict[str, Any]:
    """Pairwise t-test p-value matrix; None (plus reason) when not applicable."""
    result: Dict[str, Any] = {"performed": False, "pairwise_ttest": None}
    if numeric_df.shape[0] < 2 or numeric_df.shape[1] < 2:
        result["note"] = (
            "requires at least two rows and two numeric columns with missingness"
        )
        return result
    if int((numeric_df.isnull().sum() > 0).sum()) < 1:
        result["note"] = "no numeric column contains missing values"
        return result
    try:
        from pyampute.exploration.mcar_statistical_tests import MCARTest

        test = MCARTest(method="ttest")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matrix = test.mcar_t_tests(numeric_df)
        result["performed"] = True
        result["pairwise_ttest"] = {
            str(col): {
                str(other): (None if pd.isna(value) else float(value))
                for other, value in row.items()
            }
            for col, row in matrix.iterrows()
        }
    except Exception as exc:  # empty groups (all-NaN column), etc.
        result["ttest_error"] = f"{type(exc).__name__}: {exc}"
    return result


def analyze_missingness(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
    *,
    random_state: int = 42,
    save_plots: bool = True,
    run_mcar_test: bool = True,
) -> Dict[str, Any]:
    """Detect missing values in ``df`` and build a JSON-serializable report.

    Parameters
    ----------
    df:
        Raw dataframe to analyze. Pass the dataframe *before* the engine has
        dropped NaN rows so missingness is not hidden.
    output_dir:
        Directory for the missingno PNGs (skipped when ``None``).
    random_state:
        Reserved for deterministic future diagnostics (e.g. subsampling).
    save_plots:
        Render and save missingno plots when missing values are detected.
    run_mcar_test:
        Run Little's MCAR chi-square test and the pairwise t-test matrix on
        the numeric columns that contain missing values.

    Returns
    -------
    dict
        Report with per-column counts, missing indices, completeness, plot
        paths and MCAR test results. Always JSON-serializable.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"analyze_missingness expects a DataFrame, got {type(df)}")

    missing_counts = df.isnull().sum()
    detected_columns = {
        str(col): int(count)
        for col, count in missing_counts.items()
        if count > 0
    }
    missing_values_detected = bool(detected_columns)

    n_rows = int(df.shape[0])
    n_columns = int(df.shape[1])
    n_cells_total = n_rows * n_columns
    n_missing_cells = int(missing_counts.sum())
    n_missing_rows = int(df.isnull().any(axis=1).sum())
    completeness_percent = (
        (n_cells_total - n_missing_cells) / n_cells_total * 100
        if n_cells_total
        else 0.0
    )
    row_completeness_percent = (
        (n_rows - n_missing_rows) / n_rows * 100 if n_rows else 0.0
    )

    report: Dict[str, Any] = {
        "status": "complete",
        "scope": "raw_dataset_before_engine_dropna",
        "missing_values_detected": missing_values_detected,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_cells_total": n_cells_total,
        "n_missing_cells": n_missing_cells,
        "n_missing_rows": n_missing_rows,
        "completeness_percent": round(completeness_percent, 2),
        "row_completeness_percent": round(row_completeness_percent, 2),
        "missing_columns": {
            column: {
                "count": count,
                "percent": round(count / n_rows * 100, 2) if n_rows else 0.0,
            }
            for column, count in detected_columns.items()
        },
    }

    if missing_values_detected:
        report["missing_indices"] = {
            column: _row_labels(df.index[df[column].isnull()])
            for column in detected_columns
        }

    report["plots"] = {}
    report["plot_errors"] = {}
    if save_plots and output_dir is not None:
        plot_dir = Path(output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("matrix", "heatmap", "bar"):
            min_missing_columns = 2 if kind == "heatmap" else 1
            try:
                path = _save_missingno_plot(
                    kind,
                    df,
                    plot_dir,
                    min_missing_columns=min_missing_columns,
                )
                if path is not None:
                    report["plots"][kind] = path.name
            except Exception as exc:  # individual plots must not fail the report
                report["plot_errors"][kind] = f"{type(exc).__name__}: {exc}"

    if run_mcar_test and missing_values_detected:
        numeric_df = df.select_dtypes(include=[np.number])
        report["mcar_test"] = {
            **_run_little_mcar_test(numeric_df),
            **_run_pairwise_ttest(numeric_df),
        }
        report["mcar_test"]["random_state"] = random_state
        mcar_note = report["mcar_test"].get("note")
        if mcar_note:
            report["mcar_test"]["note"] = (
                f"{mcar_note}; non-numeric columns are excluded from the test"
            )
    else:
        report["mcar_test"] = {
            "performed": False,
            "note": (
                "no missing values detected"
                if not missing_values_detected
                else "MCAR testing disabled"
            ),
        }

    return report


__all__ = ["analyze_missingness"]
