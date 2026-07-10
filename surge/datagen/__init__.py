"""
SURGE Data Generation Module

This module provides utilities for generating parameter samples and managing
dataset creation workflows.
"""

from .generator import DataGenerator, save_sampling_plots
from .converters import (
    convert_chinese_mnist_jpg_index_to_csv,
    convert_mnist_npy_to_csv,
    convert_mnist_npy_to_parquet,
    convert_npy_to_csv,
)

__all__ = [
    "DataGenerator",
    "save_sampling_plots",
    "convert_chinese_mnist_jpg_index_to_csv",
    "convert_mnist_npy_to_csv",
    "convert_mnist_npy_to_parquet",
    "convert_npy_to_csv",
]











