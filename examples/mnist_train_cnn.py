"""Run the MNIST CNN adapter from a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from surge.model import MNISTCNNAdapter

DEFAULT_CONFIG = Path("examples/configs/mnist_train_cnn.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SURGE MNIST CNN adapter.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to the YAML config.")
    return parser.parse_args()


def _load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data = dict(payload["data"])
    model_params = dict(payload.get("model", {}))
    fit_params = dict(payload.get("fit", {}))
    output = dict(payload.get("output", {}))

    x_train = _load_array(data["train_features"])
    y_train = _load_array(data["train_targets"]).reshape(-1)
    x_test = _load_array(data["test_features"])
    y_test = _load_array(data["test_targets"]).reshape(-1)

    val_fraction = float(fit_params.pop("validation_fraction", 0.1))
    X_fit, X_val, y_fit, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_fraction,
        random_state=int(model_params.get("random_state", 42)),
        stratify=y_train,
    )

    adapter = MNISTCNNAdapter(**model_params)
    adapter.fit(X_fit, y_fit, X_val=X_val, y_val=y_val)

    preds = np.asarray(adapter.predict(x_test)).reshape(-1)
    accuracy = float(np.mean(preds == y_test))

    model_path = Path(output.get("model_path", "runs/mnist/mnist_cnn.joblib"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.save(model_path)

    print(f"Loaded config: {config_path}")
    print(f"Saved model: {model_path}")
    print(f"Train history entries: {len(adapter.training_history)}")
    print(f"First 10 predictions: {preds[:10]}")
    print(f"Test accuracy: {accuracy:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())