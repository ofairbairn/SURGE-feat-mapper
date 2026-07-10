"""Prepare UCI wine data and run SURGE workflow: RF classifier + VAE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

import surge  # noqa: F401 - trigger model registration
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

RED_WINE = Path(r"C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-red-comma.csv")
WHITE_WINE = Path(r"C:\Users\Bipo1\Downloads\UCI Wine Data\winequality-white-comma.csv")

CSV_OUT = Path("data/datasets/uci_wine/wine_quality_combined.csv")
SPEC_PATH = Path("examples/configs/wine_rf_vae.yaml")


def prepare_csv(output_path: Path = CSV_OUT) -> Path:
    red = pd.read_csv(RED_WINE)
    white = pd.read_csv(WHITE_WINE)

    red = red.copy()
    white = white.copy()
    red["wine_type"] = 0
    white["wine_type"] = 1

    df = pd.concat([red, white], ignore_index=True)
    df = df.rename(columns={"quality": "quality_label"})
    df["quality_label"] = df["quality_label"].astype(int)

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
