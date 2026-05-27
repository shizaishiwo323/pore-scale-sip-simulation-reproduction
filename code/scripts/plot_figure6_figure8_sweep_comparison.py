#!/usr/bin/env python3
"""Plot Figure 6-8 comparisons using a full-grid GPU sweep curve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_figure6_figure8_comparison import read_paper_tables, relative_error  # noqa: E402


EPSILON0_F_M = 8.8541878128e-12


def load_result_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_to_row(result: dict[str, object]) -> dict[str, object]:
    frequency = float(result["frequency_hz"])
    omega = 2.0 * np.pi * frequency
    sigma_imag = float(result["effective_sigma_imag_s_m"])
    return {
        "frequency_hz": frequency,
        "direction": result.get("direction", "x"),
        "effective_sigma_real_s_m": float(result["effective_sigma_real_s_m"]),
        "effective_sigma_imag_s_m": sigma_imag,
        "real_relative_permittivity": sigma_imag / (omega * EPSILON0_F_M),
        "relative_residual_norm": float(result["relative_residual_norm"]),
        "iterations": int(result["iterations"]),
        "info": int(result["info"]),
        "rtol": float(result["rtol"]),
        "dtype": result["dtype"],
        "solver": result["solver"],
        "preconditioner": result.get("preconditioner", ""),
        "used_warm_start": bool(result.get("used_warm_start", False)),
        "elapsed_total_s": float(result.get("elapsed_total_s", result.get("solve_elapsed_s", np.nan))),
        "source": result.get("result_path", ""),
    }


def load_sweep(sweep_csv: Path, replacement_jsons: list[Path]) -> pd.DataFrame:
    sweep = pd.read_csv(sweep_csv)
    rows = []
    for _, row in sweep.iterrows():
        result = row.to_dict()
        result["source"] = result.get("result_path", str(sweep_csv))
        rows.append(result_to_row(result))
    for path in replacement_jsons:
        result = load_result_json(path)
        result["source"] = str(path)
        rows.append(result_to_row(result))
    data = pd.DataFrame(rows)
    data = data.sort_values(["frequency_hz", "relative_residual_norm", "info", "iterations"])
    data = data.drop_duplicates(subset=["frequency_hz"], keep="first")
    return data.sort_values("frequency_hz").reset_index(drop=True)


def nearest_value(table: pd.DataFrame, frequency_hz: float, value_col: str) -> tuple[float, float]:
    frequencies = table["frequency_hz"].to_numpy(dtype=float)
    idx = int(np.argmin(np.abs(np.log(frequencies) - np.log(frequency_hz))))
    return float(frequencies[idx]), float(table.iloc[idx][value_col])


def write_metrics(tables: dict[str, pd.DataFrame], sweep: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
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
    rows: list[dict[str, object]] = []
    for _, row in sweep.iterrows():
        frequency = float(row["frequency_hz"])
        for metric, result_key, table_key, value_col in comparisons:
            reference_frequency, reference_value = nearest_value(tables[table_key], frequency, value_col)
            value = float(row[result_key])
            rows.append(
                {
                    "frequency_hz": frequency,
                    "metric": metric,
                    "full350_value": value,
                    "reference_table": table_key,
                    "reference_frequency_hz": reference_frequency,
                    "reference_value": reference_value,
                    "absolute_error": value - reference_value,
                    "relative_error": relative_error(value, reference_value),
                    "relative_residual_norm": row["relative_residual_norm"],
                    "iterations": row["iterations"],
                    "info": row["info"],
                }
            )
    metrics = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_csv, index=False)
    return metrics


def plot_figure6(tables: dict[str, pd.DataFrame], sweep: pd.DataFrame, output_png: Path) -> None:
    cond = tables["figure6_conductivity_experiment"]
    perm = tables["figure6_permittivity_experiment"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    axes[0].loglog(cond["frequency_hz"], cond["real_conductivity_s_m"], "o-", label="Fig. 6 experiment real")
    axes[0].loglog(cond["frequency_hz"], cond["imaginary_conductivity_s_m"], "s-", label="Fig. 6 experiment imaginary")
    axes[0].loglog(sweep["frequency_hz"], sweep["effective_sigma_real_s_m"], "^-", label="Full 350^3 GPU FFT real")
    axes[0].loglog(sweep["frequency_hz"], sweep["effective_sigma_imag_s_m"], "v-", label="Full 350^3 GPU FFT imaginary")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Conductivity (S/m)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].semilogx(perm["frequency_hz"], perm["real_relative_permittivity"], "o-", label="Fig. 6 experiment real")
    axes[1].semilogx(perm["frequency_hz"], perm["imaginary_relative_permittivity"], "s-", label="Fig. 6 experiment imaginary")
    axes[1].semilogx(sweep["frequency_hz"], sweep["real_relative_permittivity"], "^-", label="Full 350^3 GPU FFT real")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Relative permittivity (-)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Full-grid GPU FFT sweep comparison with Figure 6")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_figure7(tables: dict[str, pd.DataFrame], sweep: pd.DataFrame, output_png: Path) -> None:
    sim = tables["figure7_simulation_real"]
    exp = tables["figure7_experiment_real"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    ax.loglog(sim["frequency_hz"], sim["real_conductivity_s_m"], "-", label="Fig. 7 simulation")
    ax.loglog(exp["frequency_hz"], exp["real_conductivity_s_m"], "o", label="Fig. 7 experiment")
    ax.loglog(sweep["frequency_hz"], sweep["effective_sigma_real_s_m"], "^-", label="Full 350^3 GPU FFT")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Real conductivity (S/m)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Full-grid GPU FFT sweep comparison with Figure 7")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_figure8(tables: dict[str, pd.DataFrame], sweep: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
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
    axes[0].loglog(sweep["frequency_hz"], sweep["effective_sigma_imag_s_m"], "^-", label="Full 350^3 GPU FFT")
    axes[1].loglog(sweep["frequency_hz"], sweep["real_relative_permittivity"], "^-", label="Full 350^3 GPU FFT")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Imaginary conductivity (S/m)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Real relative permittivity (-)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Full-grid GPU FFT sweep comparison with Figure 8")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_residual_summary(sweep: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    axes[0].semilogx(sweep["frequency_hz"], sweep["relative_residual_norm"], "o-")
    axes[0].axhline(1.0e-5, color="0.25", linestyle="--", label="rtol 1e-5")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Final relative residual")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].semilogx(sweep["frequency_hz"], sweep["iterations"], "o-")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Krylov iterations")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle("Full-grid GPU FFT sweep convergence summary")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "论文数据"))
    parser.add_argument("--sweep-csv", required=True)
    parser.add_argument("--replacement-json", nargs="*", default=[])
    parser.add_argument("--out-sweep-csv", default=str(ROOT / "outputs" / "figure6_figure8_stage7_gpu_fft_sweep.csv"))
    parser.add_argument("--metrics-csv", default=str(ROOT / "outputs" / "figure6_figure8_stage7_gpu_fft_metrics.csv"))
    parser.add_argument("--figure6-png", default=str(ROOT / "figures" / "reproduced_figure6_gpu_fft_sweep.png"))
    parser.add_argument("--figure7-png", default=str(ROOT / "figures" / "reproduced_figure7_gpu_fft_sweep.png"))
    parser.add_argument("--figure8-png", default=str(ROOT / "figures" / "reproduced_figure8_gpu_fft_sweep.png"))
    parser.add_argument("--residual-png", default=str(ROOT / "figures" / "ac3d_gpu_fft_stage7_sweep_convergence.png"))
    args = parser.parse_args()

    tables = read_paper_tables(Path(args.data_dir))
    sweep = load_sweep(Path(args.sweep_csv), [Path(path) for path in args.replacement_json])
    out_sweep_csv = Path(args.out_sweep_csv)
    out_sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_sweep_csv, index=False)
    metrics = write_metrics(tables, sweep, Path(args.metrics_csv))
    plot_figure6(tables, sweep, Path(args.figure6_png))
    plot_figure7(tables, sweep, Path(args.figure7_png))
    plot_figure8(tables, sweep, Path(args.figure8_png))
    plot_residual_summary(sweep, Path(args.residual_png))

    print(f"wrote {out_sweep_csv}")
    print(f"wrote {args.metrics_csv}")
    print(f"wrote {args.figure6_png}")
    print(f"wrote {args.figure7_png}")
    print(f"wrote {args.figure8_png}")
    print(f"wrote {args.residual_png}")
    print(sweep[["frequency_hz", "effective_sigma_real_s_m", "effective_sigma_imag_s_m", "relative_residual_norm", "iterations", "info"]].to_string(index=False))
    print(metrics.groupby("metric")["relative_error"].agg(["count", "mean", "median"]).to_string())


if __name__ == "__main__":
    main()
