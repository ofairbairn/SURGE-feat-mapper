"""Toy SURGE workflow example for the MNIST CNN adapter.

This script:
1. Loads MNIST .npy files from disk.
2. Flattens each image to 784 pixels.
3. Writes a CSV file with one row per image and a label column.
4. Builds a SurrogateWorkflowSpec for the registered model key ``pytorch.mnist_cnn``.
5. Runs the SURGE workflow end to end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

DATA_DIR = Path(r"C:\Users\Bipo1\Downloads\MNIST Data")
CSV_OUT = Path("runs/mnist/mnist.csv")


def build_mnist_csv(
    *,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    output_path: Path = CSV_OUT,
) -> Path:
    """Create a CSV dataset from the real MNIST .npy files for SURGE."""
    x_train = np.load(DATA_DIR / "mnist_train_features.npy")
    y_train = np.load(DATA_DIR / "mnist_train_targets.npy")

    x_test = np.load(DATA_DIR / "mnist_test_features.npy")
    y_test = np.load(DATA_DIR / "mnist_test_targets.npy")

    # Flatten each 28x28 image into 784 pixel columns.
    x_train = x_train.reshape(-1, 28 * 28).astype(np.float32)
    y_train = y_train.astype(np.int64)

    x_test = x_test.reshape(-1, 28 * 28).astype(np.float32)
    y_test = y_test.astype(np.int64)

    if n_train is not None:
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
    if n_test is not None:
        x_test = x_test[:n_test]
        y_test = y_test[:n_test]

    pixel_cols = [f"pixel_{i}" for i in range(x_train.shape[1])]

    train_df = pd.DataFrame(x_train, columns=pixel_cols)
    train_df["label"] = y_train

    test_df = pd.DataFrame(x_test, columns=pixel_cols)
    test_df["label"] = y_test

    combined = pd.concat([train_df, test_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote MNIST CSV to {output_path}")
    return output_path


def main() -> None:
    csv_path = build_mnist_csv()

    # SURGE needs to know which columns are inputs and which are outputs.
    # The pixel columns are the features, and the label column is the target.
    pixel_cols = [f"pixel_{i}" for i in range(28 * 28)]

    spec = SurrogateWorkflowSpec(
        dataset_path=str(csv_path),
        dataset_format="csv",
        metadata_overrides={"inputs": pixel_cols, "outputs": ["label"]},
        models=[
            {
                "key": "pytorch.mnist_cnn",
                "params": {
                    "epochs": 10,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "device": "cpu",
                },
            }
        ],
        output_dir="runs/mnist",
        run_tag="mnist",
        overwrite_existing_run=True,
        test_fraction=0.2,
        val_fraction=0.1,
        task_type="classification",
    )

    summary = run_surrogate_workflow(spec)
    print("SURGE workflow completed.")
    print(summary["models"][0]["metrics"])


if __name__ == "__main__":
    main()
