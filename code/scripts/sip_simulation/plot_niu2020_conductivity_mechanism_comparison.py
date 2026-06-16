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
OUTER_LEGACY_ROOT = PROJECT_ROOT.parent
MECHANISMS = ["interfacial", "pore", "membrane", "all"]
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
    "all": OUTER_LEGACY_ROOT
    / "outputs"
    / "ac3d_gpu_full350_complex64_fft_representative_12freq_rtol1e-5"
    / "sweep_results.csv",
    "pore": OUTER_LEGACY_ROOT
    / "outputs"
    / "ac3d_gpu_full350_complex64_fft_paper_component_pore_rtol1e-5"
    / "sweep_results.csv",
    "membrane": OUTER_LEGACY_ROOT
    / "outputs"
    / "ac3d_gpu_full350_complex64_fft_paper_component_membrane_rtol1e-5"
    / "sweep_results.csv",
    "interfacial": OUTER_LEGACY_ROOT
    / "outputs"
    / "ac3d_gpu_full350_complex64_fft_paper_component_interfacial_rtol1e-5"
    / "sweep_results.csv",
}


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_real_conductivity_experiment(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(data_dir / "Figure6.xlsx", [0, 1], ["frequency_hz", "real_conductivity_s_m"])


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
            "source": "data/Niu 2020data/Figure6.xlsx columns 0-1",
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
    fig.suptitle(title, fontsize=18, y=0.995)

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
    ax.set_xlim(1.0e-4, 1.0e5)
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
    ax.set_xlim(1.0e-4, 1.0e5)
    ax.set_ylim(1.0e-7, 1.0e-3)
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
    lines = [
        "# Niu 2020 Conductivity Mechanism Comparison Provenance",
        "",
        "This figure compares Niu 2020 Berea experimental conductivity data against local AC3D sweep outputs.",
        "",
        f"- Real-conductivity experiment: `{data_dir / 'Figure6.xlsx'}` columns 0-1.",
        f"- Imaginary-conductivity experiment: `{data_dir / 'Figure8.xlsx'}` columns 0-1.",
        f"- Source data CSV: `{source_data_csv}`.",
        "- Figure8 paper simulation/component columns are not used as this project's simulation curves.",
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
        default=str(PROJECT_ROOT / "figures" / "niu2020" / "niu2020_conductivity_mechanism_comparison"),
    )
    parser.add_argument(
        "--source-data-csv",
        default=str(PROJECT_ROOT / "results" / "source_data" / "niu2020_conductivity_mechanism_comparison_source_data.csv"),
    )
    parser.add_argument(
        "--provenance-md",
        default=str(PROJECT_ROOT / "results" / "niu2020" / "niu2020_conductivity_mechanism_comparison_provenance.md"),
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
