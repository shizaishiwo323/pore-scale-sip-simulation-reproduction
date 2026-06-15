#!/usr/bin/env python3
"""Compatibility wrapper for the pore-network script."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _compat_loader import load_script

globals().update(load_script(__file__, "pore_network/parse_pnextract_network.py", __name__))
