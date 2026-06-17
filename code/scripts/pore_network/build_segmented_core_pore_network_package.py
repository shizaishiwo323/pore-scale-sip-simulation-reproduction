#!/usr/bin/env python3
"""Build a digital-rock visualization and pore-network result package from one segmented TIFF."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_TIFF = Path(
    r"C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\data_inventory\ct_backed_samples_raw_copy_20260605\sample_16_Grainstone\CT_slices\1-CTseg\9-16small340.tif"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "pore_network"
DEFAULT_PNEXTRACT_EXE = PROJECT_ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe"
DEFAULT_VOXEL_SIZE_UM = 1.7
DEFAULT_DIGITAL_ROCK_DOWNSAMPLE = 1
DEFAULT_NETWORK_DOWNSAMPLE = 1
DEFAULT_DISTRIBUTION_BINS = 32
AUTO_PORE_VALUES = "auto-minority"


class ResultPaths(NamedTuple):
    result_dir: Path
    config_json: Path
    binary_tiff: Path
    binary_raw: Path
    remap_metadata: Path
    digital_rock_html: Path
    digital_rock_metadata: Path
    pnextract_prepare_dir: Path
    pnextract_network_dir: Path
    pore_network_html: Path
    pore_network_metadata: Path
    distribution_png: Path
    distribution_metadata: Path
    run_summary: Path


def _format_label(value: object) -> str:
    value_float = float(value)
    if np.isfinite(value_float) and value_float.is_integer():
        return str(int(value_float))
    return f"{value_float:g}"


def _label_to_number(value: object) -> int | float:
    value_float = float(value)
    if np.isfinite(value_float) and value_float.is_integer():
        return int(value_float)
    return value_float


def _slug(text: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", text.strip())
    clean = re.sub(r"_+", "_", clean).strip("_.")
    return clean or "segmented_core"


def parse_int_values(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    return values


def parse_pore_values(text: str) -> str | list[int]:
    clean = text.strip().lower()
    if clean in {"auto", "auto-minority", "minority"}:
        return AUTO_PORE_VALUES
    return parse_int_values(text)


def default_run_name(input_tiff: Path) -> str:
    return _slug(input_tiff.stem)


def build_result_paths(*, output_root: Path, run_name: str, sample_id: str) -> ResultPaths:
    result_dir = output_root / run_name
    segmented_dir = result_dir / "segmented_core"
    digital_rock_dir = result_dir / "digital_rock"
    pore_network_dir = result_dir / "pore_network"
    source_data_dir = result_dir / "source_data"
    config_dir = result_dir / "configs"
    sample_slug = _slug(sample_id)
    return ResultPaths(
        result_dir=result_dir,
        config_json=config_dir / f"{sample_slug}_pore_network_package_config_zh.json",
        binary_tiff=segmented_dir / f"{sample_slug}_solid255_pore0.tiff",
        binary_raw=segmented_dir / f"{sample_slug}_solid255_pore0.raw",
        remap_metadata=source_data_dir / f"{sample_slug}_solid255_pore0_remap_metadata.json",
        digital_rock_html=digital_rock_dir / f"{sample_slug}_fiji3d_volume_interactive.html",
        digital_rock_metadata=digital_rock_dir / f"{sample_slug}_fiji3d_volume_interactive_metadata.json",
        pnextract_prepare_dir=pore_network_dir / "pnextract_inputs",
        pnextract_network_dir=pore_network_dir / "pnextract_network",
        pore_network_html=pore_network_dir / f"{sample_slug}_pnextract_ballstick_interactive.html",
        pore_network_metadata=pore_network_dir / f"{sample_slug}_pnextract_ballstick_interactive_metadata.json",
        distribution_png=pore_network_dir / f"{sample_slug}_pore_throat_frequency_distribution.png",
        distribution_metadata=source_data_dir / f"{sample_slug}_pore_throat_frequency_distribution_metadata.json",
        run_summary=result_dir / "run_summary.json",
    )


def load_segmented_tiff(input_tiff: Path) -> np.ndarray:
    volume = tifffile.imread(input_tiff)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D segmented TIFF stack, got shape {volume.shape}")
    return np.asarray(volume)


def infer_auto_minority_pore_values(volume: np.ndarray) -> tuple[list[object], list[object], str]:
    values, counts = np.unique(volume, return_counts=True)
    if len(values) != 2:
        raise ValueError(
            "auto-minority pore inference expects exactly two segmented classes; "
            f"found {len(values)} values: {[_format_label(v) for v in values]}"
        )
    order = np.argsort(counts)
    if int(counts[order[0]]) == int(counts[order[1]]):
        raise ValueError("auto-minority pore inference cannot resolve equal-sized classes")
    return [values[int(order[0])]], [values[int(order[1])]], "auto_minority_pore_majority_solid"


def resolve_pore_and_solid_values(volume: np.ndarray, pore_values: str | list[int]) -> tuple[list[object], list[object], str]:
    values = list(np.unique(volume))
    if pore_values == AUTO_PORE_VALUES:
        return infer_auto_minority_pore_values(volume)
    pore_value_set = {float(value) for value in pore_values}
    resolved_pores = [value for value in values if float(value) in pore_value_set]
    if len(resolved_pores) != len(pore_value_set):
        raise ValueError(
            "explicit pore values are not all present in the input volume; "
            f"requested={sorted(pore_value_set)}, present={[_format_label(v) for v in values]}"
        )
    resolved_solids = [value for value in values if float(value) not in pore_value_set]
    if not resolved_solids:
        raise ValueError("explicit pore values select every component; at least one non-pore component is required")
    return resolved_pores, resolved_solids, "explicit_pore_values"


def write_binary_core(
    *,
    input_tiff: Path,
    output_tiff: Path,
    output_raw: Path,
    metadata_out: Path,
    pore_values: str | list[int],
    solid_value: int = 255,
) -> dict:
    volume = load_segmented_tiff(input_tiff)
    resolved_pores, resolved_solids, inference_mode = resolve_pore_and_solid_values(volume, pore_values)
    pore_mask = np.isin(volume, np.asarray(resolved_pores, dtype=volume.dtype))
    mapped = np.where(pore_mask, 0, int(solid_value)).astype(np.uint8)

    output_tiff.parent.mkdir(parents=True, exist_ok=True)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output_tiff, mapped)
    output_raw.write_bytes(mapped.tobytes(order="C"))

    source_values, source_counts = np.unique(volume, return_counts=True)
    total_voxels = int(volume.size)
    pore_voxels = int(np.count_nonzero(mapped == 0))
    solid_voxels = int(total_voxels - pore_voxels)
    mapping = {**{_format_label(v): 0 for v in resolved_pores}, **{_format_label(v): int(solid_value) for v in resolved_solids}}
    summary = {
        "input_tiff": str(input_tiff),
        "output_tiff": str(output_tiff),
        "output_raw": str(output_raw),
        "metadata_json": str(metadata_out),
        "shape_zyx": [int(v) for v in volume.shape],
        "input_dtype": str(volume.dtype),
        "source_values": {_format_label(v): int(c) for v, c in zip(source_values, source_counts)},
        "inference_mode": inference_mode,
        "pore_values": [_label_to_number(v) for v in resolved_pores],
        "solid_source_values": [_label_to_number(v) for v in resolved_solids],
        "mapping": mapping,
        "output_convention": {"pore": 0, "solid": int(solid_value)},
        "pore_voxels": pore_voxels,
        "solid_voxels": solid_voxels,
        "total_voxels": total_voxels,
        "porosity": float(pore_voxels / total_voxels) if total_voxels else 0.0,
        "notes": [
            "Derived binary volume follows the project convention pore=0 and solid=255.",
            "The full-resolution voxel count is used for porosity before any visualization or pnextract downsampling.",
        ],
    }
    metadata_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_digital_rock_command(
    *,
    python_exe: Path,
    paths: ResultPaths,
    voxel_size_um: float,
    digital_rock_downsample: int,
) -> list[str]:
    return [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "digital_rock_visualization" / "render_segmented_core_fiji3d_html.py"),
        "--input",
        str(paths.binary_tiff),
        "--out",
        str(paths.digital_rock_html),
        "--metadata-out",
        str(paths.digital_rock_metadata),
        "--solid-value",
        "255",
        "--downsample",
        str(int(digital_rock_downsample)),
        "--voxel-size-um",
        f"{float(voxel_size_um):g}",
        "--components",
        "0",
        "255",
        "--initial-visible",
        "pore",
        "--interpolation",
        "linear",
    ]


def build_pore_network_command(
    *,
    python_exe: Path,
    paths: ResultPaths,
    title: str,
    voxel_size_um: float,
    network_downsample: int,
    distribution_bins: int,
    pnextract_exe: Path | None,
) -> list[str]:
    command = [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "run_segmented_core_pnextract_ballstick.py"),
        "--input",
        str(paths.binary_tiff),
        "--title",
        title,
        "--pore-values",
        "0",
        "--solid-value",
        "255",
        "--voxel-size-um",
        f"{float(voxel_size_um):g}",
        "--downsample",
        str(int(network_downsample)),
        "--prepare-dir",
        str(paths.pnextract_prepare_dir),
        "--network-dir",
        str(paths.pnextract_network_dir),
        "--html-out",
        str(paths.pore_network_html),
        "--metadata-out",
        str(paths.pore_network_metadata),
        "--distribution-out",
        str(paths.distribution_png),
        "--distribution-metadata-out",
        str(paths.distribution_metadata),
        "--distribution-bins",
        str(int(distribution_bins)),
    ]
    if pnextract_exe is not None:
        command.extend(["--pnextract-exe", str(pnextract_exe)])
    return command


def run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )


def resolve_pnextract_exe(value: str | None) -> Path | None:
    if value:
        return Path(value)
    return DEFAULT_PNEXTRACT_EXE if DEFAULT_PNEXTRACT_EXE.exists() else None


def write_config(
    *,
    paths: ResultPaths,
    input_tiff: Path,
    run_name: str,
    voxel_size_um: float,
    digital_rock_downsample: int,
    network_downsample: int,
    pore_values: str | list[int],
    distribution_bins: int,
) -> dict:
    config = {
        "_说明": "本配置由 build_segmented_core_pore_network_package.py 自动写出，用于复跑分割三维数字岩心可视化和孔隙网络提取。",
        "input_tiff": str(input_tiff),
        "run_name": run_name,
        "result_dir": str(paths.result_dir),
        "voxel_size_um": float(voxel_size_um),
        "_voxel_size_um_说明": "当前使用各向同性体素边长；若样品有独立 x/y/z 标定，应显式修改并复跑。",
        "pore_values": pore_values if pore_values == AUTO_PORE_VALUES else [int(v) for v in pore_values],
        "_pore_values_说明": "auto-minority 表示自动把两类标签中的少数相作为孔隙，并在 remap metadata 中记录证据。",
        "output_convention": {"pore": 0, "solid": 255},
        "digital_rock_downsample": int(digital_rock_downsample),
        "network_downsample": int(network_downsample),
        "distribution_bins": int(distribution_bins),
        "workflow": [
            "1. 输入分割 TIFF 重映射为 pore=0, solid=255 的派生 TIFF/RAW。",
            "2. 使用 render_segmented_core_fiji3d_html.py 导出 Fiji/VTK 风格三维数字岩心 HTML。",
            "3. 使用 run_segmented_core_pnextract_ballstick.py 生成 pnextract 输入、提取网络、解析 CSV、渲染球棍 HTML。",
            "4. 根据 pores.csv 与 throats.csv 计算 pore node radius 和 throat length 的频率分布直方图。",
        ],
    }
    paths.config_json.parent.mkdir(parents=True, exist_ok=True)
    paths.config_json.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained pore-network result package from one segmented 3-D TIFF. "
            "By default it runs the sample 16 TIFF requested in this project."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT_TIFF), help="Segmented 3-D TIFF input.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for result packages.")
    parser.add_argument("--run-name", default=None, help="Result subdirectory name under --output-root.")
    parser.add_argument("--pore-values", type=parse_pore_values, default=AUTO_PORE_VALUES)
    parser.add_argument("--voxel-size-um", type=float, default=DEFAULT_VOXEL_SIZE_UM)
    parser.add_argument("--digital-rock-downsample", type=int, default=DEFAULT_DIGITAL_ROCK_DOWNSAMPLE)
    parser.add_argument("--network-downsample", type=int, default=DEFAULT_NETWORK_DOWNSAMPLE)
    parser.add_argument("--distribution-bins", type=int, default=DEFAULT_DISTRIBUTION_BINS)
    parser.add_argument("--pnextract-exe")
    parser.add_argument("--skip-digital-rock", action="store_true")
    parser.add_argument("--skip-pore-network", action="store_true")
    args = parser.parse_args()

    input_tiff = Path(args.input)
    run_name = args.run_name or default_run_name(input_tiff)
    sample_id = input_tiff.stem
    paths = build_result_paths(output_root=Path(args.output_root), run_name=run_name, sample_id=sample_id)
    paths.result_dir.mkdir(parents=True, exist_ok=True)

    config = write_config(
        paths=paths,
        input_tiff=input_tiff,
        run_name=run_name,
        voxel_size_um=float(args.voxel_size_um),
        digital_rock_downsample=int(args.digital_rock_downsample),
        network_downsample=int(args.network_downsample),
        pore_values=args.pore_values,
        distribution_bins=int(args.distribution_bins),
    )
    remap_summary = write_binary_core(
        input_tiff=input_tiff,
        output_tiff=paths.binary_tiff,
        output_raw=paths.binary_raw,
        metadata_out=paths.remap_metadata,
        pore_values=args.pore_values,
        solid_value=255,
    )

    python_exe = Path(sys.executable)
    commands: dict[str, list[str] | None] = {"digital_rock": None, "pore_network": None}
    stdout: dict[str, str | None] = {"digital_rock": None, "pore_network": None}
    stderr: dict[str, str | None] = {"digital_rock": None, "pore_network": None}

    if not args.skip_digital_rock:
        digital_command = build_digital_rock_command(
            python_exe=python_exe,
            paths=paths,
            voxel_size_um=float(args.voxel_size_um),
            digital_rock_downsample=int(args.digital_rock_downsample),
        )
        commands["digital_rock"] = digital_command
        result = run_command(digital_command, cwd=PROJECT_ROOT)
        stdout["digital_rock"] = result.stdout[-4000:]
        stderr["digital_rock"] = result.stderr[-4000:]

    if not args.skip_pore_network:
        network_command = build_pore_network_command(
            python_exe=python_exe,
            paths=paths,
            title=f"{_slug(sample_id)}_pnextract",
            voxel_size_um=float(args.voxel_size_um),
            network_downsample=int(args.network_downsample),
            distribution_bins=int(args.distribution_bins),
            pnextract_exe=resolve_pnextract_exe(args.pnextract_exe),
        )
        commands["pore_network"] = network_command
        result = run_command(network_command, cwd=PROJECT_ROOT)
        stdout["pore_network"] = result.stdout[-4000:]
        stderr["pore_network"] = result.stderr[-4000:]

    summary = {
        "script": str(Path(__file__).resolve()),
        "result_dir": str(paths.result_dir),
        "config": config,
        "remap": remap_summary,
        "outputs": {
            "binary_tiff": str(paths.binary_tiff),
            "binary_raw": str(paths.binary_raw),
            "remap_metadata": str(paths.remap_metadata),
            "digital_rock_html": None if args.skip_digital_rock else str(paths.digital_rock_html),
            "digital_rock_metadata": None if args.skip_digital_rock else str(paths.digital_rock_metadata),
            "pnextract_prepare_dir": None if args.skip_pore_network else str(paths.pnextract_prepare_dir),
            "pnextract_network_dir": None if args.skip_pore_network else str(paths.pnextract_network_dir),
            "pore_network_html": None if args.skip_pore_network else str(paths.pore_network_html),
            "pore_network_metadata": None if args.skip_pore_network else str(paths.pore_network_metadata),
            "distribution_png": None if args.skip_pore_network else str(paths.distribution_png),
            "distribution_metadata": None if args.skip_pore_network else str(paths.distribution_metadata),
        },
        "commands": commands,
        "stdout_tail": stdout,
        "stderr_tail": stderr,
    }
    paths.run_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["run_summary"] = str(paths.run_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
