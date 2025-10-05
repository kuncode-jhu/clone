"""Utilities for preparing output and checkpoint directories for training runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def prepare_run_directory(path_like: Path | str) -> Tuple[Path, Optional[str]]:
    """Ensure a directory is ready for a new training run.

    If the requested directory already exists and contains files, a new directory
    with a timestamp suffix is created alongside the requested directory to avoid
    overwriting previous results.

    Args:
        path_like: Directory path where outputs or checkpoints should be written.

    Returns:
        A tuple consisting of the directory Path that should be used for the
        current run and an optional informational message describing any
        adjustments that were made (e.g., choosing a suffixed directory).
    """

    target = Path(path_like)
    target.mkdir(parents=True, exist_ok=True)

    # If the directory already contains files, create a new suffixed directory so
    # we do not trample existing checkpoints/logs.
    if any(target.iterdir()):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = target.parent / f"{target.name}_{timestamp}"
        suffix = 1

        while candidate.exists():
            suffix += 1
            candidate = target.parent / f"{target.name}_{timestamp}_{suffix}"

        candidate.mkdir(parents=True, exist_ok=False)
        message = (
            f"Existing directory {target} is not empty; using {candidate} for this run instead."
        )
        return candidate, message

    return target, None


def is_subpath(child: Path | str, parent: Path | str) -> bool:
    """Return True if *child* is located within *parent* (accounting for relative paths)."""

    child_path = Path(child)
    parent_path = Path(parent)

    try:
        child_path.relative_to(parent_path)
    except ValueError:
        return False
    return True

