#!/usr/bin/env python3
"""Document the retired membrane-mismatch shortcut.

Earlier revisions scanned post-extraction throat-length and resistance factors
as a fast surrogate. That path is now intentionally disabled: membrane geometry
must be corrected by re-running pnextract from recorded input parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PNEXTRACT_EXE = ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe"


def retired_membrane_mismatch_report() -> dict[str, object]:
    return {
        "status": "retired",
        "post_extraction_geometry_scaling_allowed": False,
        "default_pnextract_executable": str(DEFAULT_PNEXTRACT_EXE),
        "reason": (
            "Applying factors to already extracted pore radii, throat lengths, or geometry-derived Zdc "
            "would create non-extracted geometry. Re-run the original pnextract algorithm with the "
            "recorded input volume and explicit run parameters instead."
        ),
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Retired Niu 2020 Membrane Mismatch Diagnostic",
        "",
        f"- Status: `{report['status']}`",
        f"- Post-extraction geometry scaling allowed: `{report['post_extraction_geometry_scaling_allowed']}`",
        f"- Required pnextract executable: `{report['default_pnextract_executable']}`",
        "",
        str(report["reason"]),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-out", default=str(ROOT / "results" / "niu2020" / "niu2020_membrane_mismatch_diagnosis_retired.json"))
    parser.add_argument("--summary-md", default=str(ROOT / "results" / "niu2020" / "niu2020_membrane_mismatch_diagnosis_retired.md"))
    args = parser.parse_args()

    report = retired_membrane_mismatch_report()
    metadata_path = Path(args.metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(Path(args.summary_md), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
