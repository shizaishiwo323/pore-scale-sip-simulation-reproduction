#!/usr/bin/env python3
"""Create a pnextract copy with throat-length diagnostic CSV output.

The vendored pnextract tree is treated as read-only. This helper copies it,
optionally applies the Dong-Blunt throat-length patch, and injects diagnostic
output into the copied ``blockNet_write_cnm.cpp`` only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "code" / "vendor" / "pnextract"
DEFAULT_OUT_ROOT = Path(tempfile.gettempdir()) / "sip_pnextract_throat_diagnostics"

sys.path.insert(0, str(SCRIPT_DIR))
from build_patched_pnextract_dong_blunt import (  # noqa: E402
    OLD_FLOOR_RE,
    OLD_LENGTH_BLOCK_RE,
    build_compile_command,
    replacement_floor_block,
    replacement_length_block,
    run_build,
)


DIAGNOSTIC_VECTOR_MARKER = "vector<double> t_ltrot(nTrots,0.); // throat portion of t_lengthP1toP2s"
RAW_LENGTH_MARKER = "lthroat = lengthP1toP2-lp1-lp2;"
SUMMARY_MARKER = 'cout<<  " P1-to-P2 length < 3    for " <<lengthP1toP2Warnings<<" throats"<<endl;'


def diagnostic_vector_block(indent: str = "\t") -> str:
    lines = [
        DIAGNOSTIC_VECTOR_MARKER,
        "vector<double> diag_lpt1(nTrots,0.);",
        "vector<double> diag_lpt2(nTrots,0.);",
        "vector<double> diag_rp1(nTrots,0.);",
        "vector<double> diag_rp2(nTrots,0.);",
        "vector<double> diag_rr(nTrots,0.);",
        "vector<double> diag_raw_lthroat_before_floor(nTrots,0.);",
        "vector<double> diag_center_minus_radii(nTrots,0.);",
        "vector<int> diag_boundary(nTrots,0);",
        "vector<int> diag_overlap(nTrots,0);",
        "int diagRawLt1 = 0, diagRawLt5 = 0, diagRawLt10 = 0, diagOverlapCount = 0, diagBoundaryCount = 0;",
    ]
    return "\n".join(indent + line for line in lines)


def diagnostic_record_block(indent: str = "\t\t") -> str:
    lines = [
        RAW_LENGTH_MARKER,
        "diag_lpt1[ti] = lpt1;",
        "diag_lpt2[ti] = lpt2;",
        "diag_rp1[ti] = rp1;",
        "diag_rp2[ti] = rp2;",
        "diag_rr[ti] = rr;",
        "diag_raw_lthroat_before_floor[ti] = lthroat;",
        "diag_center_minus_radii[ti] = lengthP1toP2-rp1-rp2;",
        "diag_boundary[ti] = (tr.e1 < 2 || tr.e2 < 2);",
        "diag_overlap[ti] = (lengthP1toP2 < rp1 + rp2);",
        "if (lthroat < 1.) ++diagRawLt1;",
        "if (lthroat < 5.) ++diagRawLt5;",
        "if (lthroat < 10.) ++diagRawLt10;",
        "if (diag_overlap[ti]) ++diagOverlapCount;",
        "if (diag_boundary[ti]) ++diagBoundaryCount;",
    ]
    return "\n".join(indent + line for line in lines)


def diagnostic_csv_block(indent: str = "\t") -> str:
    lines = [
        "{",
        '\tFILE* diag = fopen((cg.name() + "_throat_length_diagnostics.csv").c_str(), "w");',
        '\tfprintf(diag, "throat_id,e1,e2,lpt1_voxels,lpt2_voxels,rp1_voxels,rp2_voxels,rr_voxels,lengthP1toP2_voxels,raw_lthroat_voxels,floored_lthroat_voxels,center_minus_radii_voxels,is_boundary,is_overlap,throat_radius_voxels,throat_volume_voxels3\\n");',
        "\tfor (int ti = 0; ti < nTrots; ++ti) {",
        "\t\tconst throatNE& tr = *throatIs[ti];",
        '\t\tfprintf(diag, "%d,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%.12g,%.12g\\n",',
        "\t\t\tti+1, int(tr.e1-1), int(tr.e2-1),",
        "\t\t\tdiag_lpt1[ti], diag_lpt2[ti], diag_rp1[ti], diag_rp2[ti], diag_rr[ti],",
        "\t\t\tt_lengthP1toP2s[ti], diag_raw_lthroat_before_floor[ti], t_ltrot[ti], diag_center_minus_radii[ti],",
        "\t\t\tdiag_boundary[ti], diag_overlap[ti], t_radiuss[ti], tr.volumn);",
        "\t}",
        "\tfclose(diag);",
        "}",
        'cout<<" throat diagnostics: rawL<1/5/10 voxels = "<<diagRawLt1<<"/"<<diagRawLt5<<"/"<<diagRawLt10',
        '\t<<", overlap="<<diagOverlapCount<<", boundary="<<diagBoundaryCount<<endl;',
    ]
    return "\n".join(indent + line if line else "" for line in lines)


def apply_dong_blunt_if_requested(
    text: str,
    *,
    beta: float | None,
    min_throat_length_voxels: float,
    target: Path,
) -> text:
    if beta is None:
        return text
    if beta <= 0:
        raise ValueError("beta must be positive")
    if min_throat_length_voxels <= 0:
        raise ValueError("min_throat_length_voxels must be positive")
    match = OLD_LENGTH_BLOCK_RE.search(text)
    if match is None:
        raise RuntimeError(f"Could not find fixed 0.67 throat-length block in {target}")
    patched = OLD_LENGTH_BLOCK_RE.sub(replacement_length_block(beta, indent=match.group("indent")), text, count=1)
    floor_match = OLD_FLOOR_RE.search(patched)
    if floor_match is None:
        raise RuntimeError(f"Could not find throat-length floor block in {target}")
    return OLD_FLOOR_RE.sub(
        replacement_floor_block(min_throat_length_voxels, indent=floor_match.group("indent")),
        patched,
        count=1,
    )


def inject_diagnostics(text: str, target: Path) -> str:
    if "diag_raw_lthroat_before_floor" in text:
        raise RuntimeError(f"Diagnostics already appear to be injected in {target}")
    if DIAGNOSTIC_VECTOR_MARKER not in text:
        raise RuntimeError(f"Could not find diagnostic vector insertion marker in {target}")
    patched = text.replace(DIAGNOSTIC_VECTOR_MARKER, diagnostic_vector_block(indent="\t"), 1)
    if RAW_LENGTH_MARKER not in patched:
        raise RuntimeError(f"Could not find raw throat-length marker in {target}")
    patched = patched.replace(RAW_LENGTH_MARKER, diagnostic_record_block(indent="\t\t"), 1)
    if SUMMARY_MARKER not in patched:
        raise RuntimeError(f"Could not find calcThroats summary marker in {target}")
    patched = patched.replace(SUMMARY_MARKER, diagnostic_csv_block(indent="\t") + "\n" + "\t" + SUMMARY_MARKER, 1)
    return patched


def copy_and_patch_pnextract_source(
    source_root: Path,
    patched_root: Path,
    *,
    beta: float | None = None,
    min_throat_length_voxels: float = 1.0,
) -> dict:
    if patched_root.exists():
        raise FileExistsError(f"patched_root already exists; choose a new output directory: {patched_root}")
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    shutil.copytree(source_root, patched_root)
    target = patched_root / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp"
    text = target.read_text(encoding="utf-8")
    text = apply_dong_blunt_if_requested(
        text,
        beta=beta,
        min_throat_length_voxels=min_throat_length_voxels,
        target=target,
    )
    patched = inject_diagnostics(text, target)
    target.write_text(patched, encoding="utf-8")

    summary = {
        "source_root": str(source_root),
        "patched_root": str(patched_root),
        "patched_file": str(target),
        "diagnostics": "throat_length_export",
        "diagnostic_csv": "<pnextract input name>_throat_length_diagnostics.csv",
        "beta": None if beta is None else float(beta),
        "min_throat_length_voxels": float(min_throat_length_voxels),
        "notes": [
            "Diagnostics are injected only in the copied source tree.",
            "The CSV records throat-level raw length before floor, floored length, pore/throat radii, boundary flags, and overlap flags.",
        ],
    }
    (patched_root / "throat_diagnostics_patch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", default=str(DEFAULT_VENDOR_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--min-throat-length-voxels", type=float, default=1.0)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    tag = "default_formula" if args.beta is None else f"beta_{args.beta:.6g}".replace(".", "p").replace("-", "m")
    out_root = Path(args.out_root)
    patched_root = out_root / tag / "source"
    build_dir = out_root / tag / "build"
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""
    exe_path = build_dir / f"pnextract_throat_diagnostics_{tag}{exe_suffix}"

    summary = copy_and_patch_pnextract_source(
        Path(args.vendor_root),
        patched_root,
        beta=args.beta,
        min_throat_length_voxels=args.min_throat_length_voxels,
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    command = build_compile_command(patched_root, build_dir, exe_path, cxx=args.cxx)
    summary.update(
        {
            "build_dir": str(build_dir),
            "exe_path": str(exe_path),
            "compile_command": command,
            "built": False,
        }
    )
    if not args.skip_build:
        result = run_build(command, cwd=PROJECT_ROOT)
        summary.update(
            {
                "built": True,
                "compile_stdout_tail": result.stdout[-4000:],
                "compile_stderr_tail": result.stderr[-4000:],
            }
        )

    summary_path = out_root / tag / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
