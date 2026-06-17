#!/usr/bin/env python3
"""Scan Dong-Blunt throat-segmentation beta for Niu 2020 Berea pnextract."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts" / "pore_network"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "sip_simulation"))

from build_patched_pnextract_dong_blunt import (  # noqa: E402
    DEFAULT_VENDOR_ROOT,
    build_compile_command,
    copy_and_patch_pnextract_source,
    run_build,
)
from run_segmented_core_pnextract_ballstick import (  # noqa: E402
    expected_pnextract_prefix,
    prepare_pnextract_input,
    run_command,
    verify_pnextract_outputs,
)
from scan_pnextract_for_niu2020_membrane import (  # noqa: E402
    append_pnextract_lines,
    build_parse_command,
    evaluate_network,
    plot_scan_summary,
    read_figure5,
    write_summary_md,
)


def beta_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    values: list[float] = []
    current = start
    while current <= stop + 0.5 * step:
        values.append(round(current, 10))
        current += step
    return values


def beta_tag(beta: float) -> str:
    return f"beta_{beta:.6g}".replace(".", "p").replace("-", "m")


def build_or_reuse_patched_pnextract(
    *,
    beta: float,
    vendor_root: Path,
    build_root: Path,
    cxx: str,
    force_rebuild: bool = False,
) -> dict:
    tag = beta_tag(beta)
    root = build_root / tag
    patched_root = root / "source"
    build_dir = root / "build"
    exe_path = build_dir / f"pnextract_dong_blunt_{tag}.exe"
    summary_path = root / "build_summary.json"

    if exe_path.exists() and summary_path.exists() and not force_rebuild:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["reused"] = True
        return summary
    if patched_root.exists() and force_rebuild:
        raise FileExistsError(
            f"Refusing to delete existing patched source for rebuild. Choose a new --build-root: {patched_root}"
        )

    summary = copy_and_patch_pnextract_source(
        vendor_root,
        patched_root,
        beta=beta,
        min_throat_length_voxels=1.0,
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    command = build_compile_command(patched_root, build_dir, exe_path, cxx=cxx)
    result = run_build(command, cwd=PROJECT_ROOT)
    summary.update(
        {
            "build_dir": str(build_dir),
            "exe_path": str(exe_path),
            "compile_command": command,
            "built": True,
            "reused": False,
            "compile_stdout_tail": result.stdout[-4000:],
            "compile_stderr_tail": result.stderr[-4000:],
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def scan_one_beta(
    *,
    beta: float,
    min_throat_length_voxels: float,
    pnextract_exe: Path,
    out_dir: Path,
    segmented_volume: Path,
    figure5: Path,
    voxel_size_um: float,
    pnextract_lines: list[str],
    python_exe: Path,
) -> dict:
    paper_pore, paper_throat = read_figure5(figure5)
    floor_tag = f"floor_{min_throat_length_voxels:.6g}".replace(".", "p").replace("-", "m")
    tag = f"{beta_tag(beta)}_{floor_tag}"
    candidate_dir = out_dir / "candidates" / tag
    prepare_dir = candidate_dir / "pnextract_inputs"
    network_dir = candidate_dir / "network_parsed"
    title = f"niu2020_dong_blunt_{tag}"
    start = time.perf_counter()
    row: dict = {
        "candidate": tag,
        "beta": float(beta),
        "min_throat_length_voxels": float(min_throat_length_voxels),
        "candidate_dir": str(candidate_dir),
        "network_dir": str(network_dir),
        "pnextract_exe": str(pnextract_exe),
        "pnextract_lines": [*pnextract_lines, f"minThroatLengthVoxels {min_throat_length_voxels:.12g}"],
    }
    try:
        prepare_summary = prepare_pnextract_input(
            segmented_volume,
            prepare_dir,
            title=title,
            pore_values=[0],
            voxel_size_um=voxel_size_um,
            downsample=1,
        )
        mhd_path = Path(prepare_summary["mhd_path"])
        append_pnextract_lines(mhd_path, row["pnextract_lines"])
        prefix = expected_pnextract_prefix(prepare_dir, title)
        pn_result = run_command([str(pnextract_exe), str(mhd_path.name)], cwd=prepare_dir)
        verify_pnextract_outputs(prefix)
        parse_result = run_command(build_parse_command(python_exe, prefix, network_dir), cwd=PROJECT_ROOT)
        pores = pd.read_csv(network_dir / "pores.csv")
        throats = pd.read_csv(network_dir / "throats.csv")
        metrics, comparison = evaluate_network(paper_pore, paper_throat, pores, throats)
        comparison_csv = candidate_dir / "figure5_distribution_comparison.csv"
        comparison.to_csv(comparison_csv, index=False)
        row.update(
            {
                **metrics,
                "status": "ok",
                "prepare_summary": json.dumps(prepare_summary, ensure_ascii=False),
                "pnextract_stdout_tail": pn_result.stdout[-4000:],
                "pnextract_stderr": pn_result.stderr,
                "parse_stdout": parse_result.stdout,
                "parse_stderr": parse_result.stderr,
                "comparison_csv": str(comparison_csv),
            }
        )
    except Exception as exc:  # pragma: no cover - exercised by command-line runs
        row.update({"status": "error", "error": repr(exc)})
    row["elapsed_s"] = time.perf_counter() - start
    row["pnextract_lines_json"] = json.dumps(row["pnextract_lines"], ensure_ascii=False)
    return row


def write_readme(path: Path, summary: pd.DataFrame, artifacts: dict[str, object]) -> None:
    write_summary_md(path, summary, artifacts)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "# Niu 2020 pnextract Membrane Geometry Scan",
        "# Niu 2020 Dong-Blunt Beta pnextract Geometry Scan",
    )
    text = text.replace(
        "This run scans pnextract parameters against the Niu Figure5 pore-node and pore-throat distributions before any AC3D rerun.",
        "This run scans the Dong-Blunt pore-throat segmentation coefficient beta after patching pnextract's throat-length formula in a copied source tree.",
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "niu2020_berea_dong_blunt_beta_scan_v1"))
    parser.add_argument(
        "--segmented-volume",
        default=str(PROJECT_ROOT / "results" / "niu2020_berea_reproduction" / "segmented_core" / "niu2020_berea_solid255_pore0.tiff"),
    )
    parser.add_argument("--figure5", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"))
    parser.add_argument("--voxel-size-um", type=float, default=2.8)
    parser.add_argument("--vendor-root", default=str(DEFAULT_VENDOR_ROOT))
    parser.add_argument("--build-root", default=str(Path(tempfile.gettempdir()) / "sip_pnextract_dong_blunt"))
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--beta-start", type=float, default=0.50)
    parser.add_argument("--beta-stop", type=float, default=0.75)
    parser.add_argument("--beta-step", type=float, default=0.025)
    parser.add_argument(
        "--min-throat-length-voxels",
        default="1.0",
        help="Comma-separated voxel-unit floors, e.g. 0.1,0.25,0.5,1.0",
    )
    parser.add_argument("--pnextract-line", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    values = beta_values(args.beta_start, args.beta_stop, args.beta_step)
    floor_values = [float(part.strip()) for part in args.min_throat_length_voxels.split(",") if part.strip()]
    if not floor_values:
        raise ValueError("--min-throat-length-voxels must contain at least one value")
    (out_dir / "candidate_plan.csv").write_text(
        "candidate,beta,min_throat_length_voxels\n"
        + "".join(
            f"{beta_tag(v)}_floor_{floor:.6g}".replace(".", "p").replace("-", "m") + f",{v:.10g},{floor:.10g}\n"
            for v in values
            for floor in floor_values
        ),
        encoding="utf-8",
    )

    rows = []
    build_summaries = []
    for value in values:
        for floor in floor_values:
            build_summary = build_or_reuse_patched_pnextract(
                beta=value,
                vendor_root=Path(args.vendor_root),
                build_root=Path(args.build_root),
                cxx=args.cxx,
            )
            build_summaries.append(build_summary)
            row = scan_one_beta(
                beta=value,
                min_throat_length_voxels=floor,
                pnextract_exe=Path(build_summary["exe_path"]),
                out_dir=out_dir,
                segmented_volume=Path(args.segmented_volume),
                figure5=Path(args.figure5),
                voxel_size_um=args.voxel_size_um,
                pnextract_lines=list(args.pnextract_line),
                python_exe=Path(sys.executable),
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "scan_metrics_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "scan_metrics.csv", index=False)

    paper_pore, paper_throat = read_figure5(Path(args.figure5))
    artifacts = plot_scan_summary(summary, paper_pore, paper_throat, out_dir)
    ok = summary[summary["status"].eq("ok")].copy()
    if not ok.empty:
        best = ok.sort_values("objective_score").iloc[0]
        best_network = Path(best["network_dir"])
        best_dir = out_dir / "best_candidate"
        if not best_dir.exists():
            shutil.copytree(best_network, best_dir)
        artifacts["best_candidate_dir"] = str(best_dir)
    build_json = out_dir / "build_summaries.json"
    build_json.write_text(json.dumps(build_summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts["build_summaries_json"] = str(build_json)
    artifacts["scan_metrics_csv"] = str(out_dir / "scan_metrics.csv")
    artifacts["candidate_plan_csv"] = str(out_dir / "candidate_plan.csv")
    (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_readme(out_dir / "README.md", summary, artifacts)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
