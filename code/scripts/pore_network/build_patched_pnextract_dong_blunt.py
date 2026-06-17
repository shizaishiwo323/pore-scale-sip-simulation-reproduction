#!/usr/bin/env python3
"""Create and optionally build a pnextract copy with Dong-Blunt throat lengths.

The project keeps ``code/vendor/pnextract`` read-only. This helper copies that
tree into a result directory, patches only ``blockNet_write_cnm.cpp``, and can
compile a local ``pnextract`` executable from the copied source.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "code" / "vendor" / "pnextract"
DEFAULT_OUT_ROOT = Path(tempfile.gettempdir()) / "sip_pnextract_dong_blunt"


OLD_LENGTH_BLOCK_RE = re.compile(
    r"(?P<indent>[ \t]*)"
    r"(?:///[^\n]*\n[ \t]*)?"
    r"lp1\s*=\s*lpt1\s*\*\s*0\.67\s*;\s*\n"
    r"[ \t]*lp2\s*=\s*lpt2\s*\*\s*0\.67\s*;\s*\n"
    r"[ \t]*if\s*\(\s*tr\.e1\s*<\s*2\s*\)\s*lp1\s*=\s*1\s*;\s*\n"
    r"[ \t]*if\s*\(\s*tr\.e2\s*<\s*2\s*\)\s*lp2\s*=\s*1\s*;\s*\n"
    r"[ \t]*lthroat\s*=\s*lengthP1toP2\s*-\s*lp1\s*-\s*lp2\s*;",
)

OLD_FLOOR_RE = re.compile(
    r"(?P<indent>[ \t]*)if\s*\(\s*lthroat\s*<\s*0\.0000001\s*\)\s*lthroat\s*=\s*1\s*;"
)


def replacement_length_block(beta: float, *, indent: str = "\t\t") -> str:
    lines = [
        "/// - Dong & Blunt (2009) pore-throat segmentation:",
        "///   li = lit * (1 - beta * rt / ri), lt = lij - li - lj.",
        f"const double poreThroatSegmentationBeta = {beta:.12g};",
        "lp1 = lpt1 * (1.0 - poreThroatSegmentationBeta * rr / rp1);",
        "lp2 = lpt2 * (1.0 - poreThroatSegmentationBeta * rr / rp2);",
        "lp1 = std::min(std::max(lp1, 0.0), lpt1);",
        "lp2 = std::min(std::max(lp2, 0.0), lpt2);",
        "if (tr.e1 < 2 )\tlp1 = 1;",
        "if (tr.e2 < 2 )\tlp2 = 1;",
        "lthroat = lengthP1toP2-lp1-lp2;",
    ]
    return "\n".join(
        f"{indent}{line}" if line else "" for line in lines
    )


def replacement_floor_block(min_throat_length_voxels: float, *, indent: str = "\t\t") -> str:
    return "\n".join(
        [
            f'{indent}const double minThroatLengthVoxels = cg.getOr("minThroatLengthVoxels", {min_throat_length_voxels:.12g});',
            f"{indent}lthroat = std::max(lthroat, minThroatLengthVoxels);",
        ]
    )


def copy_and_patch_pnextract_source(
    source_root: Path,
    patched_root: Path,
    *,
    beta: float,
    min_throat_length_voxels: float = 1.0,
) -> dict:
    if beta <= 0:
        raise ValueError("beta must be positive")
    if min_throat_length_voxels <= 0:
        raise ValueError("min_throat_length_voxels must be positive")
    if patched_root.exists():
        raise FileExistsError(f"patched_root already exists; choose a new output directory: {patched_root}")
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    shutil.copytree(source_root, patched_root)
    target = patched_root / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp"
    text = target.read_text(encoding="utf-8")
    match = OLD_LENGTH_BLOCK_RE.search(text)
    if match is None:
        raise RuntimeError(f"Could not find the fixed 0.67 throat-length block in {target}")
    patched = OLD_LENGTH_BLOCK_RE.sub(
        replacement_length_block(beta, indent=match.group("indent")),
        text,
        count=1,
    )
    floor_match = OLD_FLOOR_RE.search(patched)
    if floor_match is None:
        raise RuntimeError(f"Could not find the one-voxel throat-length floor in {target}")
    patched = OLD_FLOOR_RE.sub(
        replacement_floor_block(min_throat_length_voxels, indent=floor_match.group("indent")),
        patched,
        count=1,
    )
    target.write_text(patched, encoding="utf-8")

    summary = {
        "source_root": str(source_root),
        "patched_root": str(patched_root),
        "patched_file": str(target),
        "beta": float(beta),
        "min_throat_length_voxels": float(min_throat_length_voxels),
        "patch": "Dong-Blunt radius-ratio throat segmentation replaces fixed 0.67 pore-length fraction.",
        "formula": "lp = lpt * (1 - beta * rt / rp); lthroat = max(lengthP1toP2 - lp1 - lp2, minThroatLengthVoxels)",
    }
    (patched_root / "dong_blunt_patch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_compile_command(patched_root: Path, build_dir: Path, exe_path: Path, *, cxx: str = "g++") -> list[str]:
    src = patched_root / "src"
    return [
        cxx,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-pedantic",
        "-DRELEASE_DATE=\"dong-blunt-beta-patched\"",
        "-D_FILE_OFFSET_BITS=64",
        f"-I{src / 'include'}",
        f"-I{src / 'libvoxel'}",
        f"-I{src / 'pnm' / 'pnextract'}",
        str(src / "pnm" / "pnextract" / "blockNet.cpp"),
        str(src / "pnm" / "pnextract" / "nextract.cpp"),
        str(src / "pnm" / "pnextract" / "medialSurf.cpp"),
        str(src / "pnm" / "pnextract" / "writers_vtk.cpp"),
        str(src / "pnm" / "pnextract" / "writers_vxl.cpp"),
        str(src / "libvoxel" / "voxelImage.cpp"),
        "-o",
        str(exe_path),
    ]


def run_build(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", default=str(DEFAULT_VENDOR_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--min-throat-length-voxels", type=float, default=1.0)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    beta_tag = f"beta_{args.beta:.6g}".replace(".", "p").replace("-", "m")
    out_root = Path(args.out_root)
    patched_root = out_root / beta_tag / "source"
    build_dir = out_root / beta_tag / "build"
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""
    exe_path = build_dir / f"pnextract_dong_blunt_{beta_tag}{exe_suffix}"

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

    summary_path = out_root / beta_tag / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
