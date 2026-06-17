#!/usr/bin/env python3
"""Plot Niu 2020 Berea sigma' and sigma'' mechanism comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260617_original_pnextract_defaults"
DEFAULT_SWEEP_DIR = DEFAULT_RESULT_DIR / "simulation_sweeps"
MECHANISMS = ["interfacial", "pore", "membrane", "all"]
FREQUENCY_X_LIMITS = (1.0e-4, 1.0e9)
SIGMA_IMAG_Y_LIMITS = (1.0e-7, 1.0e1)
LABELS = {
    "experiment": "Experiment",
    "interfacial": "Dielectric",
    "pore": "EDL",
    "membrane": "Membrane",
    "all": "EDL+Dielectric+Membrane",
}
COLORS = {
    "interfacial": "#f28e2b",
    "pore": "#4e91e4",
    "membrane": "#65c987",
    "all": "#f15b62",
}
LINESTYLES = {
    "interfacial": (0, (5, 5)),
    "pore": (0, (5, 5)),
    "membrane": (0, (5, 5)),
    "all": "-",
}
DEFAULT_SWEEPS = {
    "all": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_all_original_pnextract_fft_x" / "sweep_results.csv",
    "pore": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_pore_fft_x" / "sweep_results.csv",
    "membrane": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_membrane_original_pnextract_fft_x" / "sweep_results.csv",
    "interfacial": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_interfacial_precision_merged" / "sweep_results.csv",
}


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_real_conductivity_experiment(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(data_dir / "Figure7.xlsx", [2, 3], ["frequency_hz", "real_conductivity_s_m"])


def read_imaginary_conductivity_experiment(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(data_dir / "Figure8.xlsx", [0, 1], ["frequency_hz", "imaginary_conductivity_s_m"])


def load_component_results(paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        path = paths[mechanism]
        frame = pd.read_csv(path)
        frame["mechanism"] = mechanism
        frame["source_csv"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def positive(values: pd.Series | np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    return np.maximum(np.abs(np.asarray(values, dtype=float)), floor)


def collect_source_data(
    real_experiment: pd.DataFrame,
    imag_experiment: pd.DataFrame,
    components: pd.DataFrame,
) -> pd.DataFrame:
    real = pd.DataFrame(
        {
            "dataset": "experiment_real",
            "frequency_hz": real_experiment["frequency_hz"],
            "real_conductivity_s_m": real_experiment["real_conductivity_s_m"],
            "source": "data/Niu 2020data/Figure7.xlsx columns 2-3",
            "paper_simulation_columns_used": False,
        }
    )
    imag = pd.DataFrame(
        {
            "dataset": "experiment_imag",
            "frequency_hz": imag_experiment["frequency_hz"],
            "imaginary_conductivity_s_m": imag_experiment["imaginary_conductivity_s_m"],
            "source": "data/Niu 2020data/Figure8.xlsx columns 0-1",
            "paper_simulation_columns_used": False,
        }
    )
    sim = pd.DataFrame(
        {
            "dataset": "simulation_" + components["mechanism"].astype(str),
            "mechanism": components["mechanism"],
            "frequency_hz": components["frequency_hz"],
            "real_conductivity_s_m": components["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": components["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": positive(components["effective_sigma_imag_s_m"]),
            "source_csv": components["source_csv"],
            "paper_simulation_columns_used": False,
        }
    )
    return pd.concat([real, imag, sim], ignore_index=True, sort=False)


def plot_mechanism(ax: plt.Axes, components: pd.DataFrame, mechanism: str, y_col: str) -> None:
    frame = components.loc[components["mechanism"] == mechanism].sort_values("frequency_hz")
    width = 2.7 if mechanism == "all" else 1.6
    ax.loglog(
        frame["frequency_hz"],
        positive(frame[y_col]),
        color=COLORS[mechanism],
        linestyle=LINESTYLES[mechanism],
        linewidth=width,
        solid_capstyle="round",
        label=LABELS[mechanism],
    )


def make_figure(
    real_experiment: pd.DataFrame,
    imag_experiment: pd.DataFrame,
    components: pd.DataFrame,
    output_base: Path,
    title: str = "Berea sandstone",
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 13,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), constrained_layout=True)
    axes[0].set_title(title, fontsize=18, pad=14)

    ax = axes[0]
    ax.loglog(
        real_experiment["frequency_hz"],
        positive(real_experiment["real_conductivity_s_m"]),
        marker="v",
        linestyle="None",
        markersize=4.8,
        color="black",
        label=LABELS["experiment"],
    )
    plot_mechanism(ax, components, "all", "effective_sigma_real_s_m")
    ax.set_ylabel("$\\sigma'$ (S/m)")
    ax.set_xlim(*FREQUENCY_X_LIMITS)
    ax.set_ylim(1.0e-4, 1.0e-1)
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.8)

    ax = axes[1]
    ax.loglog(
        imag_experiment["frequency_hz"],
        positive(imag_experiment["imaginary_conductivity_s_m"]),
        marker="o",
        linestyle="None",
        markersize=3.7,
        color="black",
        label=LABELS["experiment"],
    )
    for mechanism in ["interfacial", "pore", "membrane", "all"]:
        plot_mechanism(ax, components, mechanism, "effective_sigma_imag_s_m")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("$\\sigma''$ (S/m)")
    ax.set_xlim(*FREQUENCY_X_LIMITS)
    ax.set_ylim(*SIGMA_IMAG_Y_LIMITS)
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.8)

    for axis in axes:
        axis.grid(False)
        axis.tick_params(which="major", length=6, width=1.0)
        axis.tick_params(which="minor", length=3, width=0.8)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {".png": {"dpi": 420}, ".svg": {}, ".pdf": {}}.items():
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_provenance(path: Path, data_dir: Path, component_paths: dict[str, Path], source_data_csv: Path) -> None:
    spectra_dir = PROJECT_ROOT / "results" / "spectra" / "niu2020_berea_fullres_original_pnextract_defaults"
    base_spectrum = spectra_dir / "polarization_spectra_from_pnextract.csv"
    component_spectra_dir = spectra_dir / "components_paper_mode"
    lines = [
        "# Niu 2020 Conductivity Mechanism Comparison Provenance",
        "",
        "This figure compares Niu 2020 Berea experimental conductivity data against local AC3D sweep outputs.",
        "",
        "## Allowed Inputs",
        "",
        f"- CT segmentation: `{data_dir / 'microCT_Berea.raw'}` and `{data_dir / 'microCT_Berea.tiff'}`.",
        f"- Real-conductivity experiment: `{data_dir / 'Figure7.xlsx'}` columns 2-3.",
        f"- Imaginary-conductivity experiment: `{data_dir / 'Figure8.xlsx'}` columns 0-1.",
        "- Figure8 paper simulation/component columns are not used as this project's simulation curves.",
        "",
        "## Polarization Input Spectra",
        "",
        f"- Base pnextract spectrum: `{base_spectrum}`.",
        f"- Paper-mode mechanism spectra directory: `{component_spectra_dir}`.",
        "- No post-extraction pore/throat geometry or geometry-derived Zdc scaling is used.",
        "- Pore/throat geometry must come from the original pnextract executable with built-in default medial-surface parameters unless explicit run metadata says otherwise.",
        "",
        "## Plotted Data",
        "",
        f"- Source data CSV: `{source_data_csv}`.",
        "- The plotted sigma'' simulation values use absolute magnitude for log-axis display; signed values are preserved in source data.",
        "",
        "## Simulation CSVs",
        "",
        "| mechanism | source CSV |",
        "|---|---|",
    ]
    for mechanism in MECHANISMS:
        lines.append(f"| {mechanism} | `{component_paths[mechanism]}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_component_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "all": Path(args.all_csv),
        "pore": Path(args.pore_csv),
        "membrane": Path(args.membrane_csv),
        "interfacial": Path(args.interfacial_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "Niu 2020data"))
    parser.add_argument("--all-csv", default=str(DEFAULT_SWEEPS["all"]))
    parser.add_argument("--pore-csv", default=str(DEFAULT_SWEEPS["pore"]))
    parser.add_argument("--membrane-csv", default=str(DEFAULT_SWEEPS["membrane"]))
    parser.add_argument("--interfacial-csv", default=str(DEFAULT_SWEEPS["interfacial"]))
    parser.add_argument(
        "--figure-base",
        default=str(DEFAULT_RESULT_DIR / "figures" / "niu2020_conductivity_mechanism_comparison"),
    )
    parser.add_argument(
        "--source-data-csv",
        default=str(DEFAULT_RESULT_DIR / "source_data" / "niu2020_conductivity_mechanism_comparison_source_data.csv"),
    )
    parser.add_argument(
        "--provenance-md",
        default=str(DEFAULT_RESULT_DIR / "provenance" / "niu2020_conductivity_mechanism_comparison_provenance.md"),
    )
    parser.add_argument("--title", default="Niu 2020 Berea")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    component_paths = parse_component_paths(args)
    real_experiment = read_real_conductivity_experiment(data_dir)
    imag_experiment = read_imaginary_conductivity_experiment(data_dir)
    components = load_component_results(component_paths)
    source = collect_source_data(real_experiment, imag_experiment, components)

    source_path = Path(args.source_data_csv)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_path, index=False)
    make_figure(real_experiment, imag_experiment, components, Path(args.figure_base), title=args.title)
    write_provenance(Path(args.provenance_md), data_dir, component_paths, source_path)

    print(f"wrote {Path(args.figure_base).with_suffix('.png')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.svg')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.pdf')}")
    print(f"wrote {source_path}")
    print(f"wrote {args.provenance_md}")


if __name__ == "__main__":
    main()
