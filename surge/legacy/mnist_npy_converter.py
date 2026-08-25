"""
Convert MNIST .npy files to CSV/Parquet format for SURGE workflows.

This script demonstrates how to convert MNIST training and test data from
the raw .npy format to formats compatible with SURGE's dataset loaders.

Usage:
    python examples/mnist_data_converter_example.py

The script will:
  1. Read mnist_train_features.npy and mnist_train_targets.npy
  2. Read mnist_test_features.npy and mnist_test_targets.npy
  3. Convert to CSV and Parquet formats in data/datasets/mnist_converted/
  4. Print summary statistics
"""

from pathlib import Path

from surge.datagen.converters import (
    convert_mnist_npy_to_csv,
    convert_mnist_npy_to_parquet,
)

# ============================================================================
# Configuration: Update these paths to match your data location
# ============================================================================

# Path to your MNIST .npy files
MNIST_DATA_DIR = Path("C:\\Users\\Bipo1\\Downloads\\MNIST Data")

# Output directory for converted files
OUTPUT_DIR = Path("data/datasets/mnist_converted")

# Filenames
TRAIN_FEATURES = "mnist_train_features.npy"
TRAIN_TARGETS = "mnist_train_targets.npy"
TEST_FEATURES = "mnist_test_features.npy"
TEST_TARGETS = "mnist_test_targets.npy"


def main():
    """Convert MNIST .npy files to CSV and Parquet formats."""
    print("=" * 70)
    print("MNIST .npy to CSV/Parquet Converter")
    print("=" * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR.resolve()}")
    print()

    # ========================================================================
    # Convert training data
    # ========================================================================
    print("🔄 Converting training data...")
    print("-" * 70)

    train_features_path = MNIST_DATA_DIR / TRAIN_FEATURES
    train_targets_path = MNIST_DATA_DIR / TRAIN_TARGETS

    # To CSV
    train_csv_path = OUTPUT_DIR / "mnist_train.csv"
    print(f"\n📝 Converting to CSV: {train_csv_path.name}")
    convert_mnist_npy_to_csv(
        features_path=train_features_path,
        targets_path=train_targets_path,
        output_path=train_csv_path,
        target_name="digit",
        verbose=True,
    )

    # To Parquet (more efficient)
    train_parquet_path = OUTPUT_DIR / "mnist_train.parquet"
    print(f"\n📝 Converting to Parquet: {train_parquet_path.name}")
    convert_mnist_npy_to_parquet(
        features_path=train_features_path,
        targets_path=train_targets_path,
        output_path=train_parquet_path,
        target_name="digit",
        verbose=True,
    )

    # ========================================================================
    # Convert test data
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔄 Converting test data...")
    print("-" * 70)

    test_features_path = MNIST_DATA_DIR / TEST_FEATURES
    test_targets_path = MNIST_DATA_DIR / TEST_TARGETS

    # To CSV
    test_csv_path = OUTPUT_DIR / "mnist_test.csv"
    print(f"\n📝 Converting to CSV: {test_csv_path.name}")
    convert_mnist_npy_to_csv(
        features_path=test_features_path,
        targets_path=test_targets_path,
        output_path=test_csv_path,
        target_name="digit",
        verbose=True,
    )

    # To Parquet
    test_parquet_path = OUTPUT_DIR / "mnist_test.parquet"
    print(f"\n📝 Converting to Parquet: {test_parquet_path.name}")
    convert_mnist_npy_to_parquet(
        features_path=test_features_path,
        targets_path=test_targets_path,
        output_path=test_parquet_path,
        target_name="digit",
        verbose=True,
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ Conversion Complete!")
    print("=" * 70)
    print()
    print("📊 Generated Files:")
    print(f"  • {train_csv_path.name} ({train_csv_path.stat().st_size / (1024**2):.1f} MB)")
    print(
        f"  • {train_parquet_path.name} ({train_parquet_path.stat().st_size / (1024**2):.1f} MB)"
    )
    print(f"  • {test_csv_path.name} ({test_csv_path.stat().st_size / (1024**2):.1f} MB)")
    print(f"  • {test_parquet_path.name} ({test_parquet_path.stat().st_size / (1024**2):.1f} MB)")
    print()
    print("📂 Location:")
    print(f"  {OUTPUT_DIR.resolve()}")
    print()
    print("🚀 Next Steps:")
    print("  1. Create a YAML config that uses these CSV/Parquet files")
    print("  2. Run: surge run your_config.yaml")
    print()
    print("💡 Tip: Parquet files are smaller and faster to load.")
    print()


if __name__ == "__main__":
    main()
