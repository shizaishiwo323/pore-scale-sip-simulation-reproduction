#!/usr/bin/env python3
"""Compatibility wrapper for the digital rock visualization script."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _compat_loader import load_script

globals().update(load_script(__file__, "digital_rock_visualization/visualize_digital_rock_3d.py", __name__))
