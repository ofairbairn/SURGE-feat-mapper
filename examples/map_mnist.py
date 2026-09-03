"""Run a SURGE Mapper workflow on MNIST Data."""

from __future__ import annotations

import sys
import argparse
from importlib.resources import path
from pathlib import Path

import yaml

import surge  # noqa: F401 - ensure adapters are registered
#from surge.viz import viz_run
from surge.workflow.run import run_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

DEFAULT_CONFIG = Path("examples/configs/map_mnist.yml")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
	sys.path.insert(0, str(_REPO))

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run a SURGE Mapper workflow (representation ladder + diversity)."
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

	workflow_cfg = dict(payload.get("workflow", {}))
	viz_cfg = dict(payload.get("viz", {}))

	if not workflow_cfg.get("dataset_path"):
		raise ValueError(
			"workflow.dataset_path is missing. "
			"Point it at a pre-converted CSV/Parquet path."
		)

	workflow_cfg.setdefault("dataset_format", "auto")
	spec = SurrogateWorkflowSpec.from_dict(workflow_cfg)
	summary = run_workflow(spec, invocation={"script": str(Path(__file__).resolve())})

	if spec.workflow_type == "mapper":
		print("Mapper workflow complete.")
		ladder = summary.get("representation_ladder", {})
		print(f"Selected representation: {ladder.get('selected_rung')}")
		for rung in ladder.get("rungs_run", []):
			print(f"  {rung.get('rung')}: {rung.get('quality_gate')}")
		if summary.get("diversity"):
			print(f"Vendi score: {summary['diversity'].get('vendi_score')}")
	else:
		print("Workflow complete.")
		for model in summary.get("models", []):
			print(f"{model.get('name', model.get('key'))}: {model.get('metrics', {})}")

	# Mapper runs do not write a predictions/ directory (they produce their own
	# diversity report and q-profile plot), so only invoke viz_run on surrogate runs.
	run_root = summary.get("artifacts", {}).get("root")
	if spec.workflow_type != "mapper" and run_root:
		from surge.viz import viz_run
		viz_result = viz_run(Path(run_root), **viz_cfg)
		print("Visualization complete.")
		for path in viz_result.get("saved_paths", []):
			print(path)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())