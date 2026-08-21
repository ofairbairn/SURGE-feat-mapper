"""Clustering tendency diagnostics (Hopkins statistic, VAT/iVAT)."""

from .tendency import (
    _compute_hopkins_statistic,
    _ivat_from_vat,
    _summarize_cluster_tendency,
    _vat_reordering,
    save_tendency_heatmap,
)

__all__ = [
    "_compute_hopkins_statistic",
    "_ivat_from_vat",
    "_summarize_cluster_tendency",
    "_vat_reordering",
    "save_tendency_heatmap",
]
