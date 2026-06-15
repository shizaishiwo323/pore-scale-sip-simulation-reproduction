#!/usr/bin/env python3
"""Plot Niu 2020 Figure 6 experiment against this project's AC3D mechanisms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EPSILON0_F_M = 8.8541878128e-12
MECHANISMS = ["interfacial", "pore", "membrane", "all"]
LABELS = {
    "interfacial": "Interfacial / Maxwell",
    "pore": "Pore polarization",
    "membrane": "Membrane polarization",
    "all": "All polarizations",
}
COLORS = {
    "interfacial": "#1f77b4",
    "pore": "#2ca02c",
    "membrane": "#d62728",
    "all": "#333333",
}
LINESTYLES = {
    "interfacial": (0, (3.0, 2.0)),
    "pore": "--",
    "membrane": "-.",
    "all": "-",
}


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_figure6(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    conductivity = read_numeric_block(
        path,
        [0, 1, 2],
        ["frequency_hz", "real_conductivity_s_m", "imaginary_conductivity_s_m"],
    )
    permittivity = read_numeric_block(
        path,
        [3, 4, 5],
        ["frequency_hz", "real_relative_permittivity", "imaginary_relative_permittivity"],
    )
    return conductivity, permittivity


def add_real_relative_permittivity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    omega = 2.0 * np.pi * out["frequency_hz"].to_numpy(dtype=float)
    out["real_relative_permittivity"] = out["effective_sigma_imag_s_m"].to_numpy(dtype=float) / (omega * EPSILON0_F_M)
    return out


def load_mechanisms(paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        frame = add_real_relative_permittivity(pd.read_csv(paths[mechanism]))
        frame["mechanism"] = mechanism
        frame["mechanism_label"] = LABELS[mechanism]
        frame["source_csv"] = str(paths[mechanism])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def positive(values: pd.Series | np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    return np.maximum(np.abs(np.asarray(values, dtype=float)), floor)


def plot_mechanism(ax: plt.Axes, mechanisms: pd.DataFrame, mechanism: str, value_col: str) -> None:
    frame = mechanisms.loc[mechanisms["mechanism"] == mechanism].sort_values("frequency_hz")
    ax.loglog(
        frame["frequency_hz"],
        positive(frame[value_col]),
        color=COLORS[mechanism],
        linestyle=LINESTYLES[mechanism],
        linewidth=2.2 if mechanism == "all" else 1.35,
        label=LABELS[mechanism],
    )


def collect_source_data(conductivity: pd.DataFrame, permittivity: pd.DataFrame, mechanisms: pd.DataFrame) -> pd.DataFrame:
    exp_cond = pd.DataFrame(
        {
            "dataset": "figure6_experiment_conductivity",
            "frequency_hz": conductivity["frequency_hz"],
            "real_conductivity_s_m": conductivity["real_conductivity_s_m"],
            "imaginary_conductivity_s_m": conductivity["imaginary_conductivity_s_m"],
        }
    )
    exp_perm = pd.DataFrame(
        {
            "dataset": "figure6_experiment_permittivity",
            "frequency_hz": permittivity["frequency_hz"],
            "real_relative_permittivity": permittivity["real_relative_permittivity"],
            "imaginary_relative_permittivity": permittivity["imaginary_relative_permittivity"],
        }
    )
    sim = pd.DataFrame(
        {
            "dataset": "our_ac3d_" + mechanisms["mechanism"].astype(str),
            "frequency_hz": mechanisms["frequency_hz"],
            "real_conductivity_s_m": mechanisms["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": mechanisms["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": mechanisms["effective_sigma_imag_s_m"].abs(),
            "real_relative_permittivity": mechanisms["real_relative_permittivity"],
            "real_relative_permittivity_magnitude": mechanisms["real_relative_permittivity"].abs(),
            "iterations": mechanisms.get("iterations"),
            "relative_residual_norm": mechanisms.get("relative_residual_norm"),
            "info": mechanisms.get("info"),
            "source_csv": mechanisms["source_csv"],
        }
    )
    return pd.concat([exp_cond, exp_perm, sim], ignore_index=True, sort=False)


def make_figure(conductivity: pd.DataFrame, permittivity: pd.DataFrame, mechanisms: pd.DataFrame, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9.5,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.loglog(conductivity["frequency_hz"], conductivity["real_conductivity_s_m"], "o", color="black", markersize=4, label="Figure 6 experiment")
    for mechanism in MECHANISMS:
        plot_mechanism(ax, mechanisms, mechanism, "effective_sigma_real_s_m")
    ax.set_ylabel("$\\sigma'_{eff}$ (S/m)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("(a) Real conductivity")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.loglog(
        conductivity["frequency_hz"],
        conductivity["imaginary_conductivity_s_m"],
        "o",
        color="black",
        markersize=4,
        label="Figure 6 experiment",
    )
    for mechanism in MECHANISMS:
        plot_mechanism(ax, mechanisms, mechanism, "effective_sigma_imag_s_m")
    ax.set_ylabel("$|\\sigma''_{eff}|$ (S/m)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("(b) Imaginary conductivity magnitude")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.loglog(
        permittivity["frequency_hz"],
        permittivity["real_relative_permittivity"],
        "o",
        color="black",
        markersize=4,
        label="Figure 6 experiment",
    )
    for mechanism in MECHANISMS:
        plot_mechanism(ax, mechanisms, mechanism, "real_relative_permittivity")
    ax.set_ylabel("$|\\epsilon'_{eff}/\\epsilon_0|$")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("(c) Real relative permittivity magnitude")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for mechanism in MECHANISMS:
        frame = mechanisms.loc[mechanisms["mechanism"] == mechanism].sort_values("frequency_hz")
        ax.loglog(
            frame["frequency_hz"],
            positive(frame["relative_residual_norm"]),
            color=COLORS[mechanism],
            linestyle=LINESTYLES[mechanism],
            linewidth=2.2 if mechanism == "all" else 1.35,
            label=LABELS[mechanism],
        )
    ax.axhline(1.0e-5, color="0.45", linestyle=":", linewidth=1.0, label="rtol 1e-5")
    ax.set_ylabel("Final relative residual")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title("(d) Solver convergence")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.22)
        ax.set_xlim(1.0e-3, 1.0e9)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {".png": {"dpi": 360}, ".svg": {}, ".pdf": {}}.items():
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def parse_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "all": Path(args.all_csv),
        "pore": Path(args.pore_csv),
        "membrane": Path(args.membrane_csv),
        "interfacial": Path(args.interfacial_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure6-xlsx", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "Figure6.xlsx"))
    parser.add_argument("--all-csv", required=True)
    parser.add_argument("--pore-csv", required=True)
    parser.add_argument("--membrane-csv", required=True)
    parser.add_argument("--interfacial-csv", required=True)
    parser.add_argument(
        "--figure-base",
        default=str(PROJECT_ROOT / "figures" / "niu2020" / "niu2020_figure6_tiff_full350_mechanism_comparison"),
    )
    parser.add_argument(
        "--source-data-csv",
        default=str(PROJECT_ROOT / "results" / "source_data" / "niu2020_figure6_tiff_full350_mechanism_comparison_source_data.csv"),
    )
    parser.add_argument(
        "--metadata-json",
        default=str(PROJECT_ROOT / "results" / "niu2020" / "niu2020_figure6_tiff_full350_mechanism_comparison_metadata.json"),
    )
    parser.add_argument("--tiff-input", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.tiff"))
    parser.add_argument("--prepared-input-metadata", default="")
    args = parser.parse_args()

    figure6_path = Path(args.figure6_xlsx)
    conductivity, permittivity = read_figure6(figure6_path)
    paths = parse_paths(args)
    mechanisms = load_mechanisms(paths)
    source_data = collect_source_data(conductivity, permittivity, mechanisms)
    source_path = Path(args.source_data_csv)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(source_path, index=False)

    output_base = Path(args.figure_base)
    make_figure(conductivity, permittivity, mechanisms, output_base)

    metadata = {
        "figure6_xlsx": str(figure6_path),
        "tiff_input": str(Path(args.tiff_input)),
        "prepared_input_metadata": args.prepared_input_metadata or None,
        "mechanism_csvs": {key: str(value) for key, value in paths.items()},
        "figure_outputs": {
            "png": str(output_base.with_suffix(".png")),
            "svg": str(output_base.with_suffix(".svg")),
            "pdf": str(output_base.with_suffix(".pdf")),
        },
        "source_data_csv": str(source_path),
        "plot_note": "Imaginary conductivity and real relative permittivity are plotted as magnitudes for log-scale display; signed values are preserved in source_data_csv.",
    }
    metadata_path = Path(args.metadata_json)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output_base.with_suffix('.png')}")
    print(f"wrote {output_base.with_suffix('.svg')}")
    print(f"wrote {output_base.with_suffix('.pdf')}")
    print(f"wrote {source_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
