"""Tests for Mapper clustering-tendency diagnostics."""

from Mapper.tendency import get_hopkins_m


def test_hopkins_sample_size_uses_ten_percent() -> None:
    assert get_hopkins_m(200) == 20


def test_hopkins_sample_size_obeys_default_bounds() -> None:
    assert get_hopkins_m(50) == 10
    assert get_hopkins_m(10_000) == 50


def test_hopkins_sample_size_stays_below_dataset_size() -> None:
    assert get_hopkins_m(2) == 1
    assert get_hopkins_m(7) == 6


def test_hopkins_sample_size_accepts_custom_bounds() -> None:
    assert get_hopkins_m(1_000, max_samples=25, min_samples=5) == 25
