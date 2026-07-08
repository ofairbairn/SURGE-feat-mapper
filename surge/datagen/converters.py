"""
Data format converters for SURGE.

This module provides utilities to convert between different data formats
(e.g., .npy to .csv/.parquet) for compatibility with SURGE workflows.

Example:
    Convert MNIST .npy files to CSV format::

        from surge.datagen.converters import convert_mnist_npy_to_csv

        # Convert training data
        convert_mnist_npy_to_csv(
            features_path="mnist_train_features.npy",
            targets_path="mnist_train_targets.npy",
            output_path="mnist_train.csv"
        )

        # Convert test data
        convert_mnist_npy_to_csv(
            features_path="mnist_test_features.npy",
            targets_path="mnist_test_targets.npy",
            output_path="mnist_test.csv"
        )
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def convert_mnist_npy_to_csv(
    features_path: Union[str, Path],
    targets_path: Union[str, Path],
    output_path: Union[str, Path],
    feature_names: Optional[list] = None,
    target_name: str = "digit",
    normalize_features: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Convert MNIST .npy files (features + targets) to CSV format.

    This utility combines separate feature and target .npy files into a single
    CSV where the target is the last column. The result is compatible with
    SURGE's dataset loaders (the loader will auto-detect digits/pixels as outputs).

    Parameters
    ----------
    features_path : str or Path
        Path to the MNIST features .npy file (shape: N × 784 for flattened,
        or N × 28 × 28 for image-shaped).
    targets_path : str or Path
        Path to the MNIST targets .npy file (shape: N,).
    output_path : str or Path
        Path where the output CSV will be written.
    feature_names : list, optional
        List of feature column names. If None, generates names like
        'pixel_0', 'pixel_1', ..., 'pixel_783' for 784-D data.
    target_name : str, default="digit"
        Name for the target column.
    normalize_features : bool, default=False
        If True, scale features to [0, 1] range (255 → 1.0).
    verbose : bool, default=True
        Print conversion progress.

    Returns
    -------
    Path
        Path to the generated CSV file.

    Example
    -------
    >>> from surge.datagen.converters import convert_mnist_npy_to_csv
    >>> csv_path = convert_mnist_npy_to_csv(
    ...     features_path="mnist_train_features.npy",
    ...     targets_path="mnist_train_targets.npy",
    ...     output_path="data/mnist_train.csv"
    ... )
    >>> print(f"✅ Saved to {csv_path}")
    """
    features_path = Path(features_path)
    targets_path = Path(targets_path)
    output_path = Path(output_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📂 Loading features from {features_path}")
    X = np.load(features_path)

    if verbose:
        print(f"📂 Loading targets from {targets_path}")
    y = np.load(targets_path)

    # Flatten if needed (e.g., 28×28 images → 784)
    if X.ndim > 2:
        if verbose:
            print(f"   Flattening from shape {X.shape} → ({X.shape[0]}, {np.prod(X.shape[1:])})")
        X = X.reshape(X.shape[0], -1)

    if X.ndim != 2:
        raise ValueError(f"Features must be 2D after flattening; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Targets must be 1D; got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch: features has {X.shape[0]} rows, targets has {y.shape[0]} rows"
        )

    if verbose:
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Targets shape: {y.shape}")

    # Normalize if requested
    if normalize_features:
        if verbose:
            print(f"   Normalizing features to [0, 1]")
        X = X.astype(np.float32) / 255.0

    # Generate feature names if not provided
    if feature_names is None:
        n_features = X.shape[1]
        feature_names = [f"pixel_{i}" for i in range(n_features)]

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Mismatch: {len(feature_names)} feature names but data has {X.shape[1]} columns"
        )

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

    if verbose:
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns[:5])} ... {target_name}")
        print(f"   Data types:\n{df.dtypes.value_counts()}")
        print(f"   Target distribution:\n{df[target_name].value_counts().sort_index()}")

    # Save to CSV
    if verbose:
        print(f"💾 Saving to {output_path}")
    df.to_csv(output_path, index=False)

    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✅ Saved {df.shape[0]} rows × {df.shape[1]} columns ({file_size_mb:.1f} MB)")

    return output_path


def convert_mnist_npy_to_parquet(
    features_path: Union[str, Path],
    targets_path: Union[str, Path],
    output_path: Union[str, Path],
    feature_names: Optional[list] = None,
    target_name: str = "digit",
    normalize_features: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Convert MNIST .npy files (features + targets) to Parquet format.

    Parquet is more efficient than CSV (smaller file size, faster reads).
    This utility combines separate feature and target .npy files into a
    single Parquet where the target is the last column.

    Parameters
    ----------
    features_path : str or Path
        Path to the MNIST features .npy file.
    targets_path : str or Path
        Path to the MNIST targets .npy file.
    output_path : str or Path
        Path where the output Parquet will be written.
    feature_names : list, optional
        List of feature column names.
    target_name : str, default="digit"
        Name for the target column.
    normalize_features : bool, default=False
        If True, scale features to [0, 1] range.
    verbose : bool, default=True
        Print conversion progress.

    Returns
    -------
    Path
        Path to the generated Parquet file.

    Example
    -------
    >>> from surge.datagen.converters import convert_mnist_npy_to_parquet
    >>> parquet_path = convert_mnist_npy_to_parquet(
    ...     features_path="mnist_train_features.npy",
    ...     targets_path="mnist_train_targets.npy",
    ...     output_path="data/mnist_train.parquet"
    ... )
    """
    features_path = Path(features_path)
    targets_path = Path(targets_path)
    output_path = Path(output_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📂 Loading features from {features_path}")
    X = np.load(features_path)

    if verbose:
        print(f"📂 Loading targets from {targets_path}")
    y = np.load(targets_path)

    # Flatten if needed
    if X.ndim > 2:
        if verbose:
            print(f"   Flattening from shape {X.shape} → ({X.shape[0]}, {np.prod(X.shape[1:])})")
        X = X.reshape(X.shape[0], -1)

    if X.ndim != 2:
        raise ValueError(f"Features must be 2D after flattening; got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Targets must be 1D; got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Mismatch: features has {X.shape[0]} rows, targets has {y.shape[0]} rows"
        )

    if verbose:
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Targets shape: {y.shape}")

    # Normalize if requested
    if normalize_features:
        if verbose:
            print(f"   Normalizing features to [0, 1]")
        X = X.astype(np.float32) / 255.0

    # Generate feature names if not provided
    if feature_names is None:
        n_features = X.shape[1]
        feature_names = [f"pixel_{i}" for i in range(n_features)]

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"Mismatch: {len(feature_names)} feature names but data has {X.shape[1]} columns"
        )

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

    if verbose:
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"   Columns: {list(df.columns[:5])} ... {target_name}")
        print(f"   Target distribution:\n{df[target_name].value_counts().sort_index()}")

    # Save to Parquet
    if verbose:
        print(f"💾 Saving to {output_path}")
    df.to_parquet(output_path, index=False, compression="snappy")

    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✅ Saved {df.shape[0]} rows × {df.shape[1]} columns ({file_size_mb:.1f} MB)")

    return output_path


def convert_npy_to_csv(
    features_path: Union[str, Path],
    targets_path: Union[str, Path],
    output_path: Union[str, Path],
    feature_names: Optional[list] = None,
    target_name: str = "target",
    verbose: bool = True,
) -> Path:
    """
    Generic converter for any .npy features + targets to CSV.

    Parameters
    ----------
    features_path : str or Path
        Path to features .npy file.
    targets_path : str or Path
        Path to targets .npy file.
    output_path : str or Path
        Path where CSV will be written.
    feature_names : list, optional
        Column names for features. If None, generates "feature_0", "feature_1", etc.
    target_name : str, default="target"
        Column name for targets.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    Path
        Path to generated CSV.
    """
    features_path = Path(features_path)
    targets_path = Path(targets_path)
    output_path = Path(output_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Targets file not found: {targets_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📂 Loading features from {features_path}")
    X = np.load(features_path)

    if verbose:
        print(f"📂 Loading targets from {targets_path}")
    y = np.load(targets_path)

    # Ensure 2D features
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)

    if X.ndim != 2:
        raise ValueError(f"Features must be 2D; got shape {X.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: {X.shape[0]} vs {y.shape[0]}")

    # Generate feature names if not provided
    if feature_names is None:
        n_features = X.shape[1]
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Create and save DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

    if verbose:
        print(f"📊 DataFrame: {df.shape}")

    df.to_csv(output_path, index=False)

    if verbose:
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✅ Saved to {output_path} ({file_size_mb:.1f} MB)")

    return output_path


__all__ = [
    "convert_mnist_npy_to_csv",
    "convert_mnist_npy_to_parquet",
    "convert_npy_to_csv",
]
