"""Tests for Mapper terminal progress helpers."""

import pytest

from Mapper import progress


def test_timed_operation_reports_start_and_elapsed_finish(monkeypatch, capsys) -> None:
    times = iter((10.0, 12.345))
    monkeypatch.setattr(progress.time, "perf_counter", lambda: next(times))

    with progress.timed_operation("test", "opaque work"):
        pass

    output = capsys.readouterr().out
    assert "[Mapper test] opaque work - started" in output
    assert "[Mapper test] opaque work - finished in 2.35s" in output


def test_timed_operation_reports_failure_elapsed_time(monkeypatch, capsys) -> None:
    times = iter((4.0, 5.0))
    monkeypatch.setattr(progress.time, "perf_counter", lambda: next(times))

    with pytest.raises(RuntimeError, match="stopped"):
        with progress.timed_operation("test", "failing work"):
            raise RuntimeError("stopped")

    output = capsys.readouterr().out
    assert "[Mapper test] failing work - failed after 1.00s" in output


def test_mapper_progress_uses_mapper_stage_label(capsys) -> None:
    list(
        progress.mapper_progress(
            range(2),
            stage="test",
            operation="items",
            total=2,
            unit="item",
        )
    )

    assert "[Mapper test] items" in capsys.readouterr().out
