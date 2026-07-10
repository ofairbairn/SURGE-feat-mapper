"""
Data format converters for SURGE.

This module provides utilities to convert between different data formats
(e.g., .npy to .csv/.parquet) for compatibility with SURGE workflows.

Example:
    Convert MNIST .npy files to CSV format:

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

Example:
    Convert .csv delimiter from semicolon ; to comma ,
    from surge.datagen.converters import convert_csv_delimiter
convert_csv_delimiter(

    r"C:\\Users\\Bipo1\\Downloads\\UCI Wine Data\\winequality-red.csv",
    r"C:\\Users\\Bipo1\\Downloads\\UCI Wine Data\\winequality-red-comma.csv",
)

convert_csv_delimiter(
    r"C:\\Users\\Bipo1\\Downloads\\UCI Wine Data\\winequality-white.csv",
    r"C:\\Users\\Bipo1\\Downloads\\UCI Wine Data\\winequality-white-comma.csv",
)
"""

from pathlib import Path
from typing import Optional, Union, Sequence
import json
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

"""
Single function that covers CSV, Parquet, JSON, and HDF5 and automatically handles image vs tabular shapes
Later, look into refining this whole converters.py module to avoid redundancy
N.B. requires h5py for HDF5 support, and pyarrow for Parquet support
THIS NEEDS TO BE WORKED ON AND IS NOT COMPLETE OR FULLY TESTED YET
"""
def convert_npy_to_format(
    features_path: Union[str, Path],
    targets_path: Optional[Union[str, Path]],
    output_path: Union[str, Path],
    *,
    format: str = "csv",                 # "csv", "parquet", "json", "hdf5"
    feature_names: Optional[Sequence[str]] = None,
    target_name: str = "target",
    normalize: bool = False,
    flatten_images_for_tabular: bool = True,
    image_shape_meta: bool = True,
    parquet_engine: str = "pyarrow",
    verbose: bool = True,
) -> Path:
    features_path = Path(features_path)
    output_path = Path(output_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if targets_path is not None:
        targets_path = Path(targets_path)
        if not targets_path.exists():
            raise FileNotFoundError(f"Targets file not found: {targets_path}")

    X = np.load(features_path)
    y = None if targets_path is None else np.load(targets_path)

    # Optionally normalize numeric range
    if normalize:
        X = X.astype(np.float32)
        X = (X - X.min()) / max(1e-12, (X.max() - X.min()))

    # If image-shaped and user wants tabular output, flatten
    is_image = X.ndim > 2
    if is_image and flatten_images_for_tabular and format in ("csv", "parquet", "json"):
        n_samples = X.shape[0]
        flat_dim = int(np.prod(X.shape[1:]))
        X_flat = X.reshape(n_samples, flat_dim)
        X_out = X_flat
        if feature_names is None:
            feature_names = [f"pixel_{i}" for i in range(flat_dim)]
        if image_shape_meta:
            meta = {"original_shape": X.shape[1:]}
    else:
        # Ensure 2D for tabular formats; for HDF5 we can keep original shape
        if X.ndim == 1:
            X_out = X.reshape(-1, 1)
        elif X.ndim == 2:
            X_out = X
        else:
            X_out = X  # keep multidim for HDF5
        if feature_names is None and X_out.ndim == 2:
            feature_names = [f"feature_{i}" for i in range(X_out.shape[1])]

    # Save according to format
    if format == "csv":
        if X_out.ndim != 2:
            raise ValueError("CSV requires 2D tabular data; set flatten_images_for_tabular=True")
        df = pd.DataFrame(X_out, columns=list(feature_names))
        if y is not None:
            df[target_name] = y.reshape(-1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    elif format == "parquet":
        if X_out.ndim != 2:
            raise ValueError("Parquet requires 2D tabular data; set flatten_images_for_tabular=True")
        df = pd.DataFrame(X_out, columns=list(feature_names))
        if y is not None:
            df[target_name] = y.reshape(-1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine=parquet_engine)
    elif format == "json":
        if X_out.ndim != 2:
            raise ValueError("JSON tabular export requires 2D data; consider HDF5 for arrays")
        df = pd.DataFrame(X_out, columns=list(feature_names))
        if y is not None:
            df[target_name] = y.reshape(-1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient="records", lines=False)
    elif format == "hdf5":
        import h5py
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=X, compression="gzip")
            if y is not None:
                f.create_dataset("targets", data=y, compression="gzip")
            if image_shape_meta and is_image:
                f.attrs["original_feature_shape"] = json.dumps(X.shape[1:])
    else:
        raise ValueError(f"Unsupported format: {format}")

    if verbose:
        print(f"Saved {output_path} (format={format})")
    return output_path


def convert_chinese_mnist_jpg_index_to_csv(
    index_csv_path: Union[str, Path],
    images_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    label_column: str = "value",
    filename_column: Optional[str] = None,
    filename_template: str = "input_{suite_id}_{sample_id}_{code}.jpg",
    normalize_features: bool = False,
    target_size: Optional[tuple[int, int]] = None,
    feature_prefix: str = "pixel_",
    verbose: bool = True,
) -> Path:
    """Convert Kaggle Chinese-MNIST (index CSV + JPG folder) into a flat CSV.

    The resulting file is a tabular dataset with pixel features and one label
    column, ready for SURGE workflows.

    Parameters
    ----------
    index_csv_path : str or Path
        Path to ``chinese_mnist.csv``.
    images_dir : str or Path
        Directory containing JPG images.
    output_path : str or Path
        Destination CSV path.
    label_column : str, default="value"
        Label column in the index CSV.
    filename_column : str, optional
        If provided and present in CSV, this column is treated as image filename.
    filename_template : str, default="input_{suite_id}_{sample_id}_{code}.jpg"
        Template used when ``filename_column`` is not provided.
    normalize_features : bool, default=False
        Whether to scale pixel values from [0,255] to [0,1].
    target_size : tuple[int, int], optional
        Optional image resize target as ``(width, height)``.
    feature_prefix : str, default="pixel_"
        Prefix for flattened pixel columns.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    Path
        Path to generated CSV.
    """
    index_csv_path = Path(index_csv_path)
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    if not index_csv_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Pillow is required to read JPG images. Install with: pip install pillow"
        ) from exc

    if verbose:
        print(f"Loading index CSV: {index_csv_path}")
    index_df = pd.read_csv(index_csv_path)

    if label_column not in index_df.columns:
        raise ValueError(
            f"label_column '{label_column}' not found in index CSV columns: {list(index_df.columns)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    filenames: list[str] = []
    if filename_column and filename_column in index_df.columns:
        filenames = [str(v) for v in index_df[filename_column].tolist()]
    else:
        for _, row in index_df.iterrows():
            try:
                filenames.append(filename_template.format(**row.to_dict()))
            except KeyError as exc:
                raise ValueError(
                    "filename_template references a missing CSV column: "
                    f"{exc}. Available columns: {list(index_df.columns)}"
                ) from exc

    flat_rows: list[np.ndarray] = []
    missing: list[str] = []
    expected_dim: Optional[int] = None

    for name in filenames:
        img_path = images_dir / name
        if not img_path.exists():
            missing.append(str(img_path))
            continue

        with Image.open(img_path) as im:
            gray = im.convert("L")
            if target_size is not None:
                gray = gray.resize(target_size, Image.BILINEAR)
            arr = np.asarray(gray, dtype=np.float32)

        flat = arr.reshape(-1)
        if expected_dim is None:
            expected_dim = int(flat.shape[0])
        elif expected_dim != int(flat.shape[0]):
            raise ValueError(
                f"Inconsistent image size detected. Expected flattened length {expected_dim}, "
                f"got {flat.shape[0]} for {img_path}."
            )

        if normalize_features:
            flat = flat / 255.0

        flat_rows.append(flat)

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} image files referenced by index CSV. "
            f"First missing entries:\n{preview}"
        )

    if not flat_rows:
        raise ValueError("No images were loaded; check images_dir and index CSV mapping.")

    X = np.stack(flat_rows, axis=0)
    feature_names = [f"{feature_prefix}{i}" for i in range(X.shape[1])]
    out_df = pd.DataFrame(X, columns=feature_names)
    out_df[label_column] = index_df[label_column].to_numpy()

    if verbose:
        print(f"Writing CSV: {output_path}")
        print(f"Rows: {out_df.shape[0]}, Features: {X.shape[1]}, Label: {label_column}")

    out_df.to_csv(output_path, index=False)
    return output_path
###### EXAMPLE USE OF convert_npy_to_format #######
# # CSV (flatten images)
# convert_npy_to_format(
#     "mnist_train_features.npy",
#     "mnist_train_targets.npy",
#     "mnist_train.csv",
#     format="csv",
#     normalize=True
# )

# # Parquet (tabular)
# convert_npy_to_format("features.npy", "targets.npy", "data.parquet", format="parquet")

# # HDF5 (preserve image shape)
# convert_npy_to_format("features.npy", "targets.npy", "data.h5", format="hdf5", flatten_images_for_tabular=False)
###################################################

__all__ = [
    "convert_mnist_npy_to_csv",
    "convert_mnist_npy_to_parquet",
    "convert_npy_to_csv",
    "convert_chinese_mnist_jpg_index_to_csv",
]
