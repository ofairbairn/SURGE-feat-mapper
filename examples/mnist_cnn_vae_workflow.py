"""Run SURGE MNIST workflow: CNN + VAE classification."""

from __future__ import annotations

from pathlib import Path

import yaml

import surge  # noqa: F401 - trigger model registration
from surge.viz import viz_run
from surge.workflow.run import run_surrogate_workflow
from surge.workflow.spec import SurrogateWorkflowSpec

SPEC_PATH = Path("examples/configs/mnist_cnn_vae.yaml")


def run_spec(spec_path: Path = SPEC_PATH) -> dict:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec = SurrogateWorkflowSpec.from_dict(payload)
    return run_surrogate_workflow(spec, invocation={"script": str(Path(__file__).resolve())})


def main() -> None:
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
