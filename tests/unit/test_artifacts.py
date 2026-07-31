"""Tests for the workflow artifact directory contract."""

from pathlib import Path

from surge.io.artifacts import init_artifact_paths


def test_output_dir_is_parent_of_canonical_runs_directory(tmp_path: Path) -> None:
    paths = init_artifact_paths(tmp_path, "example_run")

    assert paths.root == tmp_path / "runs" / "example_run"
    assert paths.models_dir == paths.root / "models"
    assert paths.predictions_dir == paths.root / "predictions"
