"""Run the SURGE Mapper ladder on a generated swiss roll incomplete dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

import surge  # noqa: F401 - register model adapters

from surge.workflow.run import run_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

_REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _REPO / "examples" / "configs" / "map_incomplete.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test structure reporting with synth data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the Mapper YAML configuration.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = args.config.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    workflow_cfg = dict(payload.get("workflow", payload))

    spec = SurrogateWorkflowSpec.from_dict(workflow_cfg)
    summary = run_workflow(
        spec,
        invocation={"script": str(Path(__file__).resolve()), "spec_path": str(config_path)},
    )

    print("swiss roll incomplete Mapper workflow complete.")
    ladder = summary["representation_ladder"]
    print(f"Selected representation: {ladder['selected_rung']}")
    print("Validation reconstruction RMSE:")
    for rung in ladder["rungs_run"]:
        quality_gate = rung["quality_gate"]
        metrics = rung["reconstruction_metrics"]["val"]
        print(
            f"  {rung['rung'].upper():>3}: "
            f"RMSE={metrics.get('rmse'):.6f}, "
            f"gate={quality_gate['passed']}"
        )

    print(f"Run artifacts: {summary['artifacts']['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
