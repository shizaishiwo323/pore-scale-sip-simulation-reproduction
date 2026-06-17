#!/usr/bin/env python3
"""Build a self-contained Niu 2020 Berea reproduction result directory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTER_LEGACY_ROOT = PROJECT_ROOT.parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260617_original_pnextract_defaults"
DEFAULT_FIGURE5 = PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"
DEFAULT_TIFF = PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.tiff"
DEFAULT_RAW = PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.raw"
CONFIG_SOURCE = PROJECT_ROOT / "configs" / "sip_simulation" / "niu2020_berea_ac3d_reproduction_config.py"
DEFAULT_LEGACY_NETWORK_DIR = OUTER_LEGACY_ROOT / "outputs" / "figure5_pnextract_comparison" / "network_parsed"
DEFAULT_PNEXTRACT_EXE = PROJECT_ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe"
DEFAULT_PNEXTRACT_PARAMETER_FILE = (
    PROJECT_ROOT
    / "code"
    / "vendor"
    / "pnextract"
    / "config"
    / "niu2020_contact_split_conserve_pnextract_lines.txt"
)
SAMPLE_ID = "niu2020_berea"
VOXEL_SIZE_UM = 2.8
NETWORK_PORE_COORD_COLUMNS = {"pore_center_x_m", "pore_center_y_m", "pore_center_z_m"}
NETWORK_REQUIRED_PORE_COLUMNS = {"pore_id", "pore_radius_m", *NETWORK_PORE_COORD_COLUMNS}
NETWORK_REQUIRED_THROAT_COLUMNS = {"pore1_id", "pore2_id", "throat_radius_m"}


def load_pnextract_parameter_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def write_chinese_parameter_config(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_SOURCE, out)


def read_figure5_geometry(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_excel(path, sheet_name=0, header=None)
    data = table.iloc[1:].copy()
    pore = data.iloc[:, [0, 1]].copy()
    pore.columns = ["size_m", "relative_volume"]
    throat = data.iloc[:, [2, 3]].copy()
    throat.columns = ["size_m", "relative_volume"]
    for frame in (pore, throat):
        frame["size_m"] = pd.to_numeric(frame["size_m"], errors="coerce")
        frame["relative_volume"] = pd.to_numeric(frame["relative_volume"], errors="coerce")
        frame.dropna(subset=["size_m", "relative_volume"], inplace=True)
        total = frame["relative_volume"].sum()
        if total > 0:
            frame["relative_volume"] = frame["relative_volume"] / total
    return pore.reset_index(drop=True), throat.reset_index(drop=True)


def remap_niu_tiff_to_notebook_binary(
    input_tiff: Path,
    out_dir: Path,
    *,
    sample_id: str = SAMPLE_ID,
    input_volume: np.ndarray | None = None,
    voxel_size_um: float = VOXEL_SIZE_UM,
) -> dict:
    """Follow notebooks/seged_DRP_and_PNM.ipynb: remap segmentation to pore=0, solid=255."""
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(input_volume if input_volume is not None else tifffile.imread(input_tiff))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF stack, got shape {volume.shape}")

    values, counts = np.unique(volume, return_counts=True)
    if len(values) < 2:
        raise ValueError(f"Expected at least two segmented values, got {[int(v) for v in values]}")
    solid_source_values = [int(values[int(np.argmax(counts))])]
    pore_source_values = [int(v) for v in values if int(v) not in solid_source_values]
    binary = np.where(np.isin(volume, solid_source_values), 255, 0).astype(np.uint8)

    binary_tiff = out_dir / f"{sample_id}_solid255_pore0.tiff"
    binary_raw = out_dir / f"{sample_id}_solid255_pore0.raw"
    metadata_json = out_dir / f"{sample_id}_solid255_pore0_remap_metadata.json"
    tifffile.imwrite(binary_tiff, binary, photometric="minisblack")
    binary.tofile(binary_raw)

    pore_voxels = int(np.count_nonzero(binary == 0))
    solid_voxels = int(np.count_nonzero(binary == 255))
    total_voxels = int(binary.size)
    metadata = {
        "input_tiff": str(input_tiff),
        "binary_tiff": str(binary_tiff),
        "binary_raw": str(binary_raw),
        "metadata_json": str(metadata_json),
        "shape_zyx": [int(v) for v in binary.shape],
        "dtype": str(binary.dtype),
        "source_value_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
        "solid_source_values": solid_source_values,
        "pore_source_values": pore_source_values,
        "solid_output_value": 255,
        "pore_output_value": 0,
        "solid_voxels": solid_voxels,
        "pore_voxels": pore_voxels,
        "total_voxels": total_voxels,
        "porosity": float(pore_voxels / total_voxels) if total_voxels else 0.0,
        "voxel_size_um_assumed_xyz": [float(voxel_size_um), float(voxel_size_um), float(voxel_size_um)],
        "notebook_reference": "notebooks/seged_DRP_and_PNM.ipynb remap cell: majority class -> solid 255; other class(es) -> pore 0.",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def _bar_widths_log(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.asarray([x[0] * 0.2], dtype=float)
    mids = np.sqrt(x[:-1] * x[1:])
    first = x[0] ** 2 / mids[0]
    last = x[-1] ** 2 / mids[-1]
    edges = np.concatenate([[first], mids, [last]])
    return np.diff(edges)


def write_figure4_style_distribution(figure5: Path, figure_out: Path, source_data_csv: Path) -> dict:
    pore, throat = read_figure5_geometry(figure5)
    combined = pd.concat(
        [
            pore.assign(kind="pore_node_size"),
            throat.assign(kind="pore_throat_length"),
        ],
        ignore_index=True,
    )
    source_data_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(source_data_csv, index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0})
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    panels = [
        (axes[0], pore, "(a)", "Pore node size (m)"),
        (axes[1], throat, "(b)", "Pore throat length (m)"),
    ]
    for ax, frame, panel, xlabel in panels:
        x = frame["size_m"].to_numpy(dtype=float)
        y = frame["relative_volume"].to_numpy(dtype=float)
        ax.bar(x, y, width=_bar_widths_log(x) * 0.48, color="#6f6f6f", edgecolor="#202020", linewidth=0.8, align="center")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 0.15)
        ax.set_ylabel("Volume fraction")
        ax.set_xlabel(xlabel)
        ax.text(-0.18, 1.03, panel, transform=ax.transAxes, fontsize=18, ha="left", va="bottom")
        ax.tick_params(which="both", direction="in", top=False, right=False)
        ax.minorticks_on()
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_png": str(figure_out),
        "source_data_csv": str(source_data_csv),
        "figure5_xlsx": str(figure5),
        "note": "Figure 4 style distribution redrawn from Niu 2020 Figure5.xlsx geometry data.",
    }


def build_visualization_commands(*, result_dir: Path, python_exe: Path, binary: dict) -> list[list[str]]:
    digital_dir = result_dir / "digital_rock"
    return [
        [
            str(python_exe),
            str(PROJECT_ROOT / "code" / "scripts" / "digital_rock_visualization" / "render_segmented_core_fiji3d_html.py"),
            "--input",
            str(binary["binary_tiff"]),
            "--out",
            str(digital_dir / "berea_fiji3d_volume_interactive.html"),
            "--metadata-out",
            str(digital_dir / "berea_fiji3d_volume_interactive_metadata.json"),
            "--solid-value",
            "255",
            "--components",
            "0",
            "255",
            "--initial-visible",
            "pore",
            "--downsample",
            "1",
            "--voxel-size-um",
            "2.8",
            "--interpolation",
            "linear",
        ],
    ]


def build_sip_plot_command(*, result_dir: Path, python_exe: Path) -> list[str]:
    return [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "plot_niu2020_conductivity_mechanism_comparison.py"),
        "--figure-base",
        str(result_dir / "figures" / "niu2020_conductivity_mechanism_comparison"),
        "--source-data-csv",
        str(result_dir / "source_data" / "niu2020_conductivity_mechanism_comparison_source_data.csv"),
        "--provenance-md",
        str(result_dir / "provenance" / "niu2020_conductivity_mechanism_comparison_provenance.md"),
    ]


def network_has_renderable_coordinates(pores_csv: Path, throats_csv: Path) -> bool:
    """Return True when the pnextract tables include physical pore centers."""
    try:
        pore_columns = set(pd.read_csv(pores_csv, nrows=0).columns)
        throat_columns = set(pd.read_csv(throats_csv, nrows=0).columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False
    return NETWORK_REQUIRED_PORE_COLUMNS.issubset(pore_columns) and NETWORK_REQUIRED_THROAT_COLUMNS.issubset(throat_columns)


def build_pnextract_ballstick_command(*, result_dir: Path, python_exe: Path, binary: dict) -> list[str]:
    command = [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "run_segmented_core_pnextract_ballstick.py"),
        "--input",
        str(binary["binary_tiff"]),
        "--title",
        f"{SAMPLE_ID}_pnextract",
        "--pore-values",
        "0",
        "--solid-value",
        "255",
        "--voxel-size-um",
        f"{VOXEL_SIZE_UM:g}",
        "--downsample",
        "1",
        "--prepare-dir",
        str(result_dir / "pore_network" / "pnextract_inputs"),
        "--network-dir",
        str(result_dir / "pore_network" / "pnextract"),
        "--html-out",
        str(result_dir / "pore_network" / "berea_pnextract_ballstick_interactive.html"),
        "--metadata-out",
        str(result_dir / "pore_network" / "berea_pnextract_ballstick_interactive_metadata.json"),
        "--distribution-out",
        str(result_dir / "pore_network" / "berea_pnextract_pore_throat_frequency_distribution.png"),
        "--distribution-metadata-out",
        str(result_dir / "pore_network" / "berea_pnextract_pore_throat_frequency_distribution_metadata.json"),
        "--pnextract-exe",
        str(DEFAULT_PNEXTRACT_EXE),
    ]
    for line in load_pnextract_parameter_lines(DEFAULT_PNEXTRACT_PARAMETER_FILE):
        command.extend(["--pnextract-line", line])
    return command


def build_network_html_command(*, result_dir: Path, python_exe: Path, pores_csv: Path, throats_csv: Path) -> list[str]:
    return [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "render_berea_pore_network_html.py"),
        "--pores",
        str(pores_csv),
        "--throats",
        str(throats_csv),
        "--out",
        str(result_dir / "pore_network" / "berea_pore_network_ballstick_interactive.html"),
        "--metadata-out",
        str(result_dir / "pore_network" / "berea_pore_network_ballstick_interactive_metadata.json"),
        "--voxel-size-m",
        "2.8e-6",
    ]


def copy_legacy_network(network_dir: Path, result_dir: Path) -> tuple[Path, Path]:
    out_dir = result_dir / "pore_network" / "network_parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    pores_out = out_dir / "pores.csv"
    throats_out = out_dir / "throats.csv"
    shutil.copy2(network_dir / "pores.csv", pores_out)
    shutil.copy2(network_dir / "throats.csv", throats_out)
    summary = network_dir / "network_summary.json"
    if summary.exists():
        shutil.copy2(summary, out_dir / "network_summary.json")
    return pores_out, throats_out


def run_command(command: list[str], *, cwd: Path) -> dict:
    result = subprocess.run(command, cwd=str(cwd), text=True, encoding="utf-8", errors="replace", capture_output=True)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def write_manifest(result_dir: Path, records: dict) -> None:
    manifest = result_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    md = result_dir / "README.md"
    md.write_text(
        "\n".join(
            [
                "# Niu 2020 Berea Reproduction Result Package",
                "",
                "本目录集中保存本次复现的参数配置、SIP 对比图、Fiji/VTK 风格三维数字岩心可视化、孔隙网络可视化、Figure 4 风格孔径/孔喉分布图和 provenance。",
                "",
                "- `configs/`: 中文注释参数配置。",
                "- `figures/`: 论文图风格 PNG/SVG/PDF。",
                "- `segmented_core/`: 按 notebook 链路重映射的 `pore=0, solid=255` 二值 TIFF/RAW。",
                "- `digital_rock/`: 基于二值体的 Fiji 3D Viewer 风格交互 HTML；后续数字岩心可视化只保留这种风格。",
                "- `pore_network/`: 由 `run_segmented_core_pnextract_ballstick.py` 生成的 pnextract 输入、网络 CSV、交互 HTML、孔径/孔喉分布图。",
                "- `source_data/`: 图件源数据。",
                "- `provenance/`: 脚本运行记录。",
                "",
                "Policy: extracted pore/throat geometry and geometry-derived Zdc are not post-scaled. The pore network is generated with the original pnextract algorithm and its built-in default medial-surface parameters unless a run config explicitly records otherwise.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--legacy-network-dir", default=str(DEFAULT_LEGACY_NETWORK_DIR))
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--skip-network-html", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    python_exe = Path(args.python_exe)
    if not DEFAULT_PNEXTRACT_EXE.exists():
        raise FileNotFoundError(f"default pnextract executable is missing: {DEFAULT_PNEXTRACT_EXE}")
    if not DEFAULT_PNEXTRACT_PARAMETER_FILE.exists():
        raise FileNotFoundError(f"default pnextract parameter file is missing: {DEFAULT_PNEXTRACT_PARAMETER_FILE}")

    config_out = result_dir / "configs" / "niu2020_berea_parameters_zh.py"
    write_chinese_parameter_config(config_out)
    binary_summary = remap_niu_tiff_to_notebook_binary(
        DEFAULT_TIFF,
        result_dir / "segmented_core",
        sample_id=SAMPLE_ID,
        voxel_size_um=VOXEL_SIZE_UM,
    )

    figure4_summary = write_figure4_style_distribution(
        DEFAULT_FIGURE5,
        result_dir / "pore_network" / "niu2020_figure4_pore_node_throat_distribution.png",
        result_dir / "source_data" / "niu2020_figure4_pore_node_throat_distribution_source_data.csv",
    )

    command_records = []
    command_records.append(run_command(build_sip_plot_command(result_dir=result_dir, python_exe=python_exe), cwd=PROJECT_ROOT))
    if not args.skip_visualizations:
        for command in build_visualization_commands(result_dir=result_dir, python_exe=python_exe, binary=binary_summary):
            command_records.append(run_command(command, cwd=PROJECT_ROOT))
    if not args.skip_network_html:
        command_records.append(run_command(build_pnextract_ballstick_command(result_dir=result_dir, python_exe=python_exe, binary=binary_summary), cwd=PROJECT_ROOT))
        network_html_status = "generated by run_segmented_core_pnextract_ballstick.py"
    elif args.skip_network_html:
        network_html_status = "skipped by --skip-network-html"

    records = {
        "result_dir": str(result_dir),
        "chinese_parameter_config": str(config_out),
        "binary_remap": binary_summary,
        "figure4_distribution": figure4_summary,
        "notebook_reference": "notebooks/seged_DRP_and_PNM.ipynb: remap TIFF -> binary 0/255, render Fiji/VTK HTML, then run pnextract wrapper.",
        "network_html_status": network_html_status,
        "commands": command_records,
    }
    (result_dir / "provenance").mkdir(parents=True, exist_ok=True)
    (result_dir / "provenance" / "build_niu2020_berea_result_package_commands.json").write_text(
        json.dumps(command_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(result_dir, records)
    print(json.dumps(records, indent=2, ensure_ascii=False))
    failed = [record for record in command_records if record["returncode"] != 0]
    if failed:
        raise SystemExit(f"{len(failed)} command(s) failed; see {result_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
