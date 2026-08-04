"""Clustering tendency diagnostics (Hopkins statistic, VAT/iVAT)."""

from .hopkins_vat import (
    _compute_hopkins_statistic,
    _ivat_from_vat,
    _summarize_cluster_tendency,
    _vat_reordering,
)

__all__ = [
    "_compute_hopkins_statistic",
    "_ivat_from_vat",
    "_summarize_cluster_tendency",
    "_vat_reordering",
]
