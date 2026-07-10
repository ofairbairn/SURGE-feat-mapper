"""Run the MNIST SURGE workflow from a YAML config.

The config keeps the data locations and workflow parameters in one place, while
this runner materializes the CSV and fills in the feature/target metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import surge  # noqa: F401 - trigger model registration
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

DEFAULT_CONFIG = Path("examples/configs/mnist_surge_workflow.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SURGE MNIST workflow.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to the YAML config.")
    return parser.parse_args()


def _load_array(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


def build_mnist_csv(*, data_cfg: dict[str, str], output_path: Path) -> tuple[Path, list[str]]:
    x_train = _load_array(data_cfg["train_features"]).reshape(-1, 28 * 28).astype(np.float32)
    y_train = _load_array(data_cfg["train_targets"]).reshape(-1).astype(np.int64)
    x_test = _load_array(data_cfg["test_features"]).reshape(-1, 28 * 28).astype(np.float32)
    y_test = _load_array(data_cfg["test_targets"]).reshape(-1).astype(np.int64)

    pixel_cols = [f"pixel_{i}" for i in range(x_train.shape[1])]
    train_df = pd.DataFrame(x_train, columns=pixel_cols)
    train_df["label"] = y_train
    test_df = pd.DataFrame(x_test, columns=pixel_cols)
    test_df["label"] = y_test

    combined = pd.concat([train_df, test_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote MNIST CSV to {output_path}")
    return output_path, pixel_cols


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    data_cfg = dict(payload["data"])
    workflow_cfg = dict(payload["workflow"])
    csv_out = Path(data_cfg.get("csv_out", "runs/mnist/mnist.csv"))
    csv_path, pixel_cols = build_mnist_csv(data_cfg=data_cfg, output_path=csv_out)

    workflow_cfg["dataset_path"] = str(csv_path)
    workflow_cfg["dataset_format"] = "csv"
    workflow_cfg["metadata_overrides"] = {"inputs": pixel_cols, "outputs": ["label"]}

    spec = SurrogateWorkflowSpec.from_dict(workflow_cfg)
    summary = run_surrogate_workflow(spec, invocation={"script": str(Path(__file__).resolve())})

    print("Workflow complete.")
    for model in summary.get("models", []):
        print(f"{model.get('name', model.get('key'))}: {model.get('metrics', {})}")

    run_root = summary.get("artifacts", {}).get("root")
    if run_root:
        viz_result = viz_run(Path(run_root))
        print("Visualization complete.")
        for path in viz_result.get("saved_paths", []):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
