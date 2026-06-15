"""Compatibility helpers for legacy script entry points."""

from __future__ import annotations

import runpy
from pathlib import Path


def load_script(wrapper_file: str, relative_target: str, caller_name: str) -> dict[str, object]:
    target = Path(wrapper_file).resolve().parent / relative_target
    run_name = "__main__" if caller_name == "__main__" else caller_name
    return runpy.run_path(str(target), run_name=run_name)
