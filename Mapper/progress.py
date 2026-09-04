"""Consistent terminal progress and elapsed-time reporting for Mapper."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from tqdm.auto import tqdm


def mapper_progress(
    iterable: Optional[Any] = None,
    *,
    stage: str,
    operation: str,
    total: Optional[int] = None,
    unit: str = "step",
    leave: bool = True,
) -> Any:
    """Create a consistently labelled Mapper tqdm progress bar."""
    return tqdm(
        iterable,
        total=total,
        desc=f"[Mapper {stage}] {operation}",
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        file=sys.stdout,
    )


def mapper_status(message: str) -> None:
    """Write a line without corrupting an active tqdm display."""
    tqdm.write(message, file=sys.stdout)


@contextmanager
def timed_operation(stage: str, operation: str) -> Iterator[None]:
    """Print start and finish/failure lines around an opaque operation."""
    label = f"[Mapper {stage}] {operation}"
    mapper_status(f"{label} - started")
    started = time.perf_counter()
    try:
        yield
    except BaseException:
        elapsed = time.perf_counter() - started
        mapper_status(f"{label} - failed after {elapsed:.2f}s")
        raise
    else:
        elapsed = time.perf_counter() - started
        mapper_status(f"{label} - finished in {elapsed:.2f}s")


__all__ = ["mapper_progress", "mapper_status", "timed_operation"]
