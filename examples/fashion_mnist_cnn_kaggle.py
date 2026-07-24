"""Run SURGE workflow for Fashion-MNIST (Kaggle) classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

import surge  # noqa: F401 - ensure adapters are registered
from surge.datagen.converters import convert_mnist_npy_to_csv
from surge.model.backends.mnist_cnn import MNISTCNNModel
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

DEFAULT_CONFIG = Path("examples/configs/fashion_mnist_cnn_kaggle.yml")
FASHION_CLASSES = [
	"T-shirt/top",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot",
]


def print_test_classification_analysis(summary: dict) -> None:
	"""Print Fashion-MNIST diagnostics from persisted workflow predictions."""
	for model in summary.get("models", []):
		model_name = model.get("name", model.get("key", "model"))
		prediction_path = model.get("artifacts", {}).get("predictions", {}).get("test")
		if not prediction_path:
			print(f"Classification analysis skipped for {model_name}: no test predictions.")
			continue

		path = Path(prediction_path)
		frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
		true_columns = sorted(column for column in frame if column.startswith("y_true_"))
		pred_columns = sorted(column for column in frame if column.startswith("y_pred_"))
		if not true_columns or not pred_columns:
			print(f"Classification analysis skipped for {model_name}: invalid prediction columns.")
			continue

		print(f"\nFashion-MNIST test classification analysis for {model_name}:")
		MNISTCNNModel.analyze_predictions(
			frame[true_columns[0]].to_numpy(),
			frame[pred_columns[0]].to_numpy(),
			labels=range(len(FASHION_CLASSES)),
			class_names=FASHION_CLASSES,
			top_k=3,
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run Fashion-MNIST classification using SURGE MNIST CNN adapter."
	)
	parser.add_argument(
		"--config",
		type=Path,
		default=DEFAULT_CONFIG,
		help="Path to YAML config.",
	)
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	config_path = args.config.resolve()
	payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

	conversion_cfg = dict(payload.get("conversion", {}))
	workflow_cfg = dict(payload.get("workflow", {}))

	if conversion_cfg:
		output_csv = convert_mnist_npy_to_csv(
			features_path=conversion_cfg["features_path"],
			targets_path=conversion_cfg["targets_path"],
			output_path=conversion_cfg["output_csv_path"],
			target_name=conversion_cfg.get("target_name", "label"),
			normalize_features=bool(conversion_cfg.get("normalize_features", False)),
			verbose=bool(conversion_cfg.get("verbose", True)),
		)
		workflow_cfg["dataset_path"] = str(output_csv)
	elif not workflow_cfg.get("dataset_path"):
		raise ValueError(
			"No conversion block found and workflow.dataset_path is missing. "
			"Set workflow.dataset_path to a pre-converted CSV/Parquet path or re-enable conversion."
		)

	workflow_cfg.setdefault("dataset_format", "auto")
	spec = SurrogateWorkflowSpec.from_dict(workflow_cfg)
	summary = run_surrogate_workflow(spec, invocation={"script": str(Path(__file__).resolve())})

	print("Workflow complete.")
	for model in summary.get("models", []):
		print(f"{model.get('name', model.get('key'))}: {model.get('metrics', {})}")
	print_test_classification_analysis(summary)

	run_root = summary.get("artifacts", {}).get("root")
	if run_root:
		viz_result = viz_run(Path(run_root))
		print("Visualization complete.")
		for path in viz_result.get("saved_paths", []):
			print(path)

	return 0

if __name__ == "__main__":
	raise SystemExit(main())
