"""Preprocess Kaggle Chinese-MNIST and run a SURGE CNN workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import surge  # noqa: F401 - ensure adapters are registered
from surge.datagen.converters import convert_chinese_mnist_jpg_index_to_csv
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

DEFAULT_CONFIG = Path("examples/configs/chinese_mnist_cnn_kaggle.yaml")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert Chinese-MNIST JPG+index to CSV and run SURGE workflow."
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
	payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

	conversion_cfg = dict(payload.get("conversion", {}))
	workflow_cfg = dict(payload.get("workflow", {}))

	output_csv = convert_chinese_mnist_jpg_index_to_csv(
		index_csv_path=conversion_cfg["index_csv_path"],
		images_dir=conversion_cfg["images_dir"],
		output_path=conversion_cfg["output_csv_path"],
		label_column=conversion_cfg.get("label_column", "value"),
		filename_column=conversion_cfg.get("filename_column"),
		filename_template=conversion_cfg.get(
			"filename_template", "input_{suite_id}_{sample_id}_{code}.jpg"
		),
		normalize_features=bool(conversion_cfg.get("normalize_features", False)),
		target_size=tuple(conversion_cfg["target_size"]) if conversion_cfg.get("target_size") else None,
		feature_prefix=conversion_cfg.get("feature_prefix", "pixel_"),
		verbose=bool(conversion_cfg.get("verbose", True)),
	)

	workflow_cfg["dataset_path"] = str(output_csv)
	workflow_cfg.setdefault("dataset_format", "csv")

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