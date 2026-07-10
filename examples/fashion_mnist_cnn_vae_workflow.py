"""Prepare Fashion-MNIST and run SURGE workflow: CNN + VAE classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import surge  # noqa: F401 - trigger model registration
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

TRAIN_FEATURES = Path(r"C:\Users\Bipo1\Downloads\Fashion MNIST Data\fashion_mnist_train_features.npy")
TRAIN_TARGETS = Path(r"C:\Users\Bipo1\Downloads\Fashion MNIST Data\fashion_mnist_train_targets.npy")
TEST_FEATURES = Path(r"C:\Users\Bipo1\Downloads\Fashion MNIST Data\fashion_mnist_test_features.npy")
TEST_TARGETS = Path(r"C:\Users\Bipo1\Downloads\Fashion MNIST Data\fashion_mnist_test_targets.npy")

CSV_OUT = Path("data/datasets/fashion_mnist/fashion_mnist_all.csv")
SPEC_PATH = Path("examples/configs/fashion_mnist_cnn_vae.yaml")


def _flatten_features(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.astype(np.float32)
    return x.reshape(x.shape[0], -1).astype(np.float32)


def prepare_csv(output_path: Path = CSV_OUT) -> Path:
    x_train = np.load(TRAIN_FEATURES)
    y_train = np.load(TRAIN_TARGETS)
    x_test = np.load(TEST_FEATURES)
    y_test = np.load(TEST_TARGETS)

    x_train = _flatten_features(x_train)
    x_test = _flatten_features(x_test)

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    x_all = np.vstack([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])

    feature_cols = [f"pixel_{i}" for i in range(x_all.shape[1])]
    df = pd.DataFrame(x_all, columns=feature_cols)
    df["label"] = y_all.astype(np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote dataset: {output_path} shape={df.shape}")
    return output_path


def run_spec(spec_path: Path = SPEC_PATH) -> dict:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec = SurrogateWorkflowSpec.from_dict(payload)
    return run_surrogate_workflow(spec, invocation={"script": str(Path(__file__).resolve())})


def main() -> None:
    prepare_csv()
    summary = run_spec()
    print("Workflow complete.")
    for model in summary.get("models", []):
        print(f"{model.get('name', model.get('key'))}: {model.get('metrics', {})}")

    run_root = summary.get("artifacts", {}).get("root")
    if run_root:
        viz_result = viz_run(Path(run_root))
        print("Visualization complete.")
        for path in viz_result.get("saved_paths", []):
            print(path)


if __name__ == "__main__":
    main()
