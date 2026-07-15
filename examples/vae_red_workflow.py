"""Run SURGE workflow for AE on UCI Red data."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import surge  # noqa: F401 - ensure adapters are registered
from surge.datagen.converters import convert_mnist_npy_to_csv
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

# replaced this line
DEFAULT_CONFIG = Path("examples/configs/ae_red.yml")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run SURGE workflow for Alvaro's VAE."
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
	viz_cfg = dict(payload.get("viz", {}))

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

	run_root = summary.get("artifacts", {}).get("root")
	if run_root:
		viz_result = viz_run(Path(run_root), **viz_cfg)
		print("Visualization complete.")
		for path in viz_result.get("saved_paths", []):
			print(path)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

