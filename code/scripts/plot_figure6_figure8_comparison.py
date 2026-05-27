#!/usr/bin/env python3
"""Create stage-6 Figure 6-8 comparison plots and metrics.

This script intentionally treats the current full-grid AC3D result as a
diagnostic single point, because the available full 350^3 solve is one
frequency/direction and stopped at maxiter before reaching rtol=1e-8.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EPSILON0_F_M = 8.8541878128e-12


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_paper_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    figure6 = data_dir / "Figure6.xlsx"
    figure7 = data_dir / "Figure7.xlsx"
    figure8 = data_dir / "Figure8.xlsx"
    return {
        "figure6_conductivity_experiment": read_numeric_block(
            figure6,
            [0, 1, 2],
            ["frequency_hz", "real_conductivity_s_m", "imaginary_conductivity_s_m"],
        ),
        "figure6_permittivity_experiment": read_numeric_block(
            figure6,
            [3, 4, 5],
            ["frequency_hz", "real_relative_permittivity", "imaginary_relative_permittivity"],
        ),
        "figure7_simulation_real": read_numeric_block(
            figure7,
            [0, 1],
            ["frequency_hz", "real_conductivity_s_m"],
        ),
        "figure7_experiment_real": read_numeric_block(
            figure7,
            [2, 3],
            ["frequency_hz", "real_conductivity_s_m"],
        ),
        "figure8_experiment": read_numeric_block(
            figure8,
            [0, 1, 2],
            ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"],
        ),
        "figure8_simulation_all": read_numeric_block(
            figure8,
            [3, 4, 5],
            ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"],
        ),
        "figure8_pore": read_numeric_block(
            figure8,
            [6, 7, 8],
            ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"],
        ),
        "figure8_membrane": read_numeric_block(
            figure8,
            [9, 10, 11],
            ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"],
        ),
        "figure8_interfacial": read_numeric_block(
            figure8,
            [12, 13, 14],
            ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"],
        ),
    }


def nearest_value(table: pd.DataFrame, frequency_hz: float, value_col: str) -> tuple[float, float]:
    frequencies = table["frequency_hz"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(np.log(frequencies) - np.log(frequency_hz))))
    return float(frequencies[idx]), float(table.iloc[idx][value_col])


def relative_error(value: float, reference: float) -> float:
    if not np.isfinite(reference) or reference == 0.0:
        return float("nan")
    return (value - reference) / reference


def load_full350_result(path: Path) -> dict[str, float | int | str]:
    result = json.loads(path.read_text(encoding="utf-8"))
    frequency = float(result["frequency_hz"])
    omega = 2.0 * np.pi * frequency
    effective_imag = float(result["effective_sigma_imag_s_m"])
    result["real_relative_permittivity"] = effective_imag / (omega * EPSILON0_F_M)
    return result


def add_metric(
    rows: list[dict[str, float | int | str]],
    metric: str,
    value: float,
    reference_table: str,
    reference_frequency: float,
    reference_value: float,
) -> None:
    rows.append(
        {
            "metric": metric,
            "full350_value": value,
            "reference_table": reference_table,
            "reference_frequency_hz": reference_frequency,
            "reference_value": reference_value,
            "absolute_error": value - reference_value,
            "relative_error": relative_error(value, reference_value),
        }
    )


def write_metrics(tables: dict[str, pd.DataFrame], result: dict[str, float | int | str], output_csv: Path) -> pd.DataFrame:
    frequency = float(result["frequency_hz"])
    rows: list[dict[str, float | int | str]] = []

    comparisons = [
        ("real_conductivity_vs_figure6_experiment", "effective_sigma_real_s_m", "figure6_conductivity_experiment", "real_conductivity_s_m"),
        ("imaginary_conductivity_vs_figure6_experiment", "effective_sigma_imag_s_m", "figure6_conductivity_experiment", "imaginary_conductivity_s_m"),
        ("real_conductivity_vs_figure7_simulation", "effective_sigma_real_s_m", "figure7_simulation_real", "real_conductivity_s_m"),
        ("real_conductivity_vs_figure7_experiment", "effective_sigma_real_s_m", "figure7_experiment_real", "real_conductivity_s_m"),
        ("imaginary_conductivity_vs_figure8_simulation", "effective_sigma_imag_s_m", "figure8_simulation_all", "imaginary_conductivity_s_m"),
        ("imaginary_conductivity_vs_figure8_experiment", "effective_sigma_imag_s_m", "figure8_experiment", "imaginary_conductivity_s_m"),
        ("real_relative_permittivity_vs_figure8_simulation", "real_relative_permittivity", "figure8_simulation_all", "real_relative_permittivity"),
        ("real_relative_permittivity_vs_figure8_experiment", "real_relative_permittivity", "figure8_experiment", "real_relative_permittivity"),
    ]
    for metric, result_key, table_key, value_col in comparisons:
        reference_frequency, reference_value = nearest_value(tables[table_key], frequency, value_col)
        add_metric(rows, metric, float(result[result_key]), table_key, reference_frequency, reference_value)

    metrics = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    return metrics


def save_stage6_point(result: dict[str, float | int | str], output_csv: Path) -> None:
    row = {
        "frequency_hz": result["frequency_hz"],
        "direction": result["direction"],
        "effective_sigma_real_s_m": result["effective_sigma_real_s_m"],
        "effective_sigma_imag_s_m": result["effective_sigma_imag_s_m"],
        "real_relative_permittivity": result["real_relative_permittivity"],
        "relative_residual_norm": result["relative_residual_norm"],
        "iterations": result["iterations"],
        "info": result["info"],
        "solver": result["solver"],
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(output_csv, index=False)


def plot_marker(ax: plt.Axes, result: dict[str, float | int | str], y_key: str, label: str) -> None:
    ax.scatter(
        [float(result["frequency_hz"])],
        [float(result[y_key])],
        marker="*",
        s=150,
        color="#c43b3b",
        edgecolor="black",
        linewidth=0.5,
        label=label,
        zorder=5,
    )


def plot_figure6(tables: dict[str, pd.DataFrame], result: dict[str, float | int | str], output_png: Path) -> None:
    cond = tables["figure6_conductivity_experiment"]
    perm = tables["figure6_permittivity_experiment"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].loglog(cond["frequency_hz"], cond["real_conductivity_s_m"], "o-", label="Fig. 6 experiment real")
    axes[0].loglog(cond["frequency_hz"], cond["imaginary_conductivity_s_m"], "s-", label="Fig. 6 experiment imaginary")
    plot_marker(axes[0], result, "effective_sigma_real_s_m", "Full 350^3 x real, this run")
    axes[0].scatter([result["frequency_hz"]], [result["effective_sigma_imag_s_m"]], marker="X", s=100, color="#4b73b9", label="Full 350^3 x imag, this run")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Conductivity (S/m)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].semilogx(perm["frequency_hz"], perm["real_relative_permittivity"], "o-", label="Fig. 6 experiment real")
    axes[1].semilogx(perm["frequency_hz"], perm["imaginary_relative_permittivity"], "s-", label="Fig. 6 experiment imaginary")
    plot_marker(axes[1], result, "real_relative_permittivity", "Full 350^3 x real, this run")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative permittivity (-)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle("Stage 6 diagnostic comparison with Figure 6 data")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_figure7(tables: dict[str, pd.DataFrame], result: dict[str, float | int | str], output_png: Path) -> None:
    sim = tables["figure7_simulation_real"]
    exp = tables["figure7_experiment_real"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.loglog(sim["frequency_hz"], sim["real_conductivity_s_m"], "-", label="Fig. 7 simulation")
    ax.loglog(exp["frequency_hz"], exp["real_conductivity_s_m"], "o", label="Fig. 7 experiment")
    plot_marker(ax, result, "effective_sigma_real_s_m", "Full 350^3 x, this run")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Real conductivity (S/m)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Stage 6 diagnostic comparison with Figure 7")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_figure8(tables: dict[str, pd.DataFrame], result: dict[str, float | int | str], output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    styles = [
        ("figure8_experiment", "Fig. 8 experiment", "o"),
        ("figure8_simulation_all", "Fig. 8 simulation all", "-"),
        ("figure8_pore", "Pore polarization", "--"),
        ("figure8_membrane", "Membrane polarization", "-."),
        ("figure8_interfacial", "Interfacial polarization", ":"),
    ]
    for key, label, style in styles:
        table = tables[key]
        axes[0].loglog(table["frequency_hz"], table["imaginary_conductivity_s_m"], style, label=label)
        axes[1].loglog(table["frequency_hz"], table["real_relative_permittivity"], style, label=label)
    plot_marker(axes[0], result, "effective_sigma_imag_s_m", "Full 350^3 x, this run")
    plot_marker(axes[1], result, "real_relative_permittivity", "Full 350^3 x, this run")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Imaginary conductivity (S/m)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Real relative permittivity (-)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Stage 6 diagnostic comparison with Figure 8")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "论文数据"))
    parser.add_argument(
        "--result-json",
        default=str(ROOT / "outputs" / "ac3d_matrix_free_full350_krylov_jacobi_20260520_retry1" / "matrix_free_single_result.json"),
    )
    parser.add_argument("--metrics-csv", default=str(ROOT / "outputs" / "figure6_figure8_metrics.csv"))
    parser.add_argument("--stage6-point-csv", default=str(ROOT / "outputs" / "figure6_figure8_stage6_full350_point.csv"))
    parser.add_argument("--figure6-png", default=str(ROOT / "figures" / "reproduced_figure6.png"))
    parser.add_argument("--figure7-png", default=str(ROOT / "figures" / "reproduced_figure7.png"))
    parser.add_argument("--figure8-png", default=str(ROOT / "figures" / "reproduced_figure8.png"))
    args = parser.parse_args()

    tables = read_paper_tables(Path(args.data_dir))
    result = load_full350_result(Path(args.result_json))
    metrics = write_metrics(tables, result, Path(args.metrics_csv))
    save_stage6_point(result, Path(args.stage6_point_csv))
    plot_figure6(tables, result, Path(args.figure6_png))
    plot_figure7(tables, result, Path(args.figure7_png))
    plot_figure8(tables, result, Path(args.figure8_png))

    print(f"wrote {args.metrics_csv}")
    print(f"wrote {args.stage6_point_csv}")
    print(f"wrote {args.figure6_png}")
    print(f"wrote {args.figure7_png}")
    print(f"wrote {args.figure8_png}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
