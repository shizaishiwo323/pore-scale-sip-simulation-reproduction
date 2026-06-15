#!/usr/bin/env python3
"""Compute paper-consistent 2.5 Hz conductivity from SI parameters.

Uses only the supplementary data and the Waxman--Smits style equations
summarized for Rembert et al.:

Qv = ((1 - phi) / phi) * rho * CEC
sigma'  = phi^m * Sw^n     * (sigma_w + beta * Qv / Sw)
sigma'' = phi^m * Sw^(n-1) * lambda * Qv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SI01 = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0002-data set si-s01.csv"
DEFAULT_SI02 = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0003-data set si-s02.csv"
DEFAULT_SI03 = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0004-data set si-s03.csv"
DEFAULT_OUT_CSV = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "paper_consistent_2p5hz_waxman_smits.csv"
DEFAULT_OUT_FIG = PROJECT_ROOT / "figures" / "ac2d_microfluidic" / "paper_consistent_2p5hz_waxman_smits_comparison.png"
FARADAY_C_PER_EQ = 96485.33212


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def interpolate(source: pd.DataFrame, source_time: str, source_value: str, target_time: np.ndarray) -> np.ndarray:
    return np.interp(target_time, source[source_time].to_numpy(dtype=float), source[source_value].to_numpy(dtype=float))


def water_conductivity_at_times(times_h: np.ndarray, si03: pd.DataFrame, *, sigma_hcl_s_m: float, end_time_h: float) -> np.ndarray:
    src_t = si03["Time (h)"].to_numpy(dtype=float)
    src_sigma = si03["Water conductivity (S/m)"].to_numpy(dtype=float)
    out = np.interp(np.minimum(times_h, src_t[-1]), src_t, src_sigma)
    later = times_h > src_t[-1]
    if np.any(later):
        denom = max(end_time_h - src_t[-1], np.finfo(float).eps)
        frac = np.clip((times_h[later] - src_t[-1]) / denom, 0.0, 1.0)
        out[later] = src_sigma[-1] + frac * (sigma_hcl_s_m - src_sigma[-1])
    return out


def compute_model(
    *,
    si01: pd.DataFrame,
    si02: pd.DataFrame,
    si03: pd.DataFrame,
    sigma_hcl_s_m: float,
    m: float,
    n: float,
    rho_calcite_kg_m3: float,
    beta_m2_s_v: float,
    alpha: float,
    faraday_c_eq: float,
) -> pd.DataFrame:
    times = si02["Time (h)"].to_numpy(dtype=float)
    phi = interpolate(si01, "Time (h)", "Porosity", times)
    sw = interpolate(si01, "Time (h)", "Water saturation", times)
    sigma_w = water_conductivity_at_times(times, si03, sigma_hcl_s_m=sigma_hcl_s_m, end_time_h=float(times.max()))
    cec_meq_g = si02["CEC (mEq/g)"].to_numpy(dtype=float)
    # 1 mEq/g = 1 eq/kg. Multiplying by Faraday converts charge equivalents to C.
    cec_eq_kg = cec_meq_g
    qv_c_m3 = ((1.0 - phi) / np.maximum(phi, np.finfo(float).eps)) * rho_calcite_kg_m3 * cec_eq_kg * faraday_c_eq
    lam = beta_m2_s_v / alpha
    sigma_real = phi**m * sw**n * (sigma_w + beta_m2_s_v * qv_c_m3 / np.maximum(sw, np.finfo(float).eps))
    sigma_imag = phi**m * sw ** (n - 1.0) * lam * qv_c_m3
    return pd.DataFrame(
        {
            "time_h": times,
            "phi": phi,
            "sw": sw,
            "sigma_w_s_m": sigma_w,
            "cec_meq_g": cec_meq_g,
            "qv_c_m3": qv_c_m3,
            "paper_real_s_m": si02["Real conductivity (S/m)"].to_numpy(dtype=float),
            "paper_imag_s_m": si02["Imaginary conductivity (S/m)"].to_numpy(dtype=float),
            "model_real_s_m": sigma_real,
            "model_imag_s_m": sigma_imag,
            "real_error_s_m": sigma_real - si02["Real conductivity (S/m)"].to_numpy(dtype=float),
            "imag_error_s_m": sigma_imag - si02["Imaginary conductivity (S/m)"].to_numpy(dtype=float),
        }
    )


def plot_model(table: pd.DataFrame, figure_path: Path) -> None:
    real_mae = float(np.mean(np.abs(table["real_error_s_m"])))
    imag_mae = float(np.mean(np.abs(table["imag_error_s_m"])))
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True, gridspec_kw={"height_ratios": [1, 1, 0.85]})
    fig.suptitle("Paper-consistent 2.5 Hz Waxman--Smits calculation", fontsize=14, weight="bold")

    axes[0].scatter(table["time_h"], table["paper_real_s_m"], color="black", marker="s", s=20, label="Paper SI-S02")
    axes[0].plot(table["time_h"], table["model_real_s_m"], color="#d62728", linewidth=1.7, label=f"Formula result (MAE={real_mae:.4f})")
    axes[0].plot(table["time_h"], table["sigma_w_s_m"], color="#1f77b4", linestyle="--", linewidth=1.1, label="sigma_w(t)")
    axes[0].set_ylabel("Real, sigma' (S/m)")
    axes[0].legend(fontsize=8, loc="best")

    axes[1].scatter(table["time_h"], table["paper_imag_s_m"], color="black", marker="s", s=20, label="Paper SI-S02")
    axes[1].plot(table["time_h"], table["model_imag_s_m"], color="#d62728", linewidth=1.7, label=f"Formula result (MAE={imag_mae:.4f})")
    axes[1].set_ylabel("Imaginary, sigma'' (S/m)")
    axes[1].legend(fontsize=8, loc="best")

    axes[2].plot(table["time_h"], table["cec_meq_g"], color="#1f77b4", linewidth=1.5, label="CEC")
    ax2 = axes[2].twinx()
    ax2.plot(table["time_h"], table["sw"], color="#e6550d", linewidth=1.2, label="Sw")
    axes[2].set_ylabel("CEC (mEq/g)")
    ax2.set_ylabel("Sw")
    axes[2].set_xlabel("Time, t (h)")
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, fontsize=8, loc="best")

    for ax in axes:
        ax.set_xlim(0.0, max(4.5, table["time_h"].max()))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=260, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--si01", default=str(DEFAULT_SI01))
    parser.add_argument("--si02", default=str(DEFAULT_SI02))
    parser.add_argument("--si03", default=str(DEFAULT_SI03))
    parser.add_argument("--sigma-hcl", type=float, default=0.44)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--n", type=float, default=2.0)
    parser.add_argument("--rho-calcite", type=float, default=2710.0)
    parser.add_argument("--beta", type=float, default=19.98e-8)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--faraday", type=float, default=FARADAY_C_PER_EQ)
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-fig", default=str(DEFAULT_OUT_FIG))
    args = parser.parse_args()

    table = compute_model(
        si01=read_semicolon_csv(Path(args.si01)),
        si02=read_semicolon_csv(Path(args.si02)),
        si03=read_semicolon_csv(Path(args.si03)),
        sigma_hcl_s_m=args.sigma_hcl,
        m=args.m,
        n=args.n,
        rho_calcite_kg_m3=args.rho_calcite,
        beta_m2_s_v=args.beta,
        alpha=args.alpha,
        faraday_c_eq=args.faraday,
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    plot_model(table, Path(args.out_fig))
    print(f"real_mae_s_m {np.mean(np.abs(table['real_error_s_m'])):.8g}")
    print(f"imag_mae_s_m {np.mean(np.abs(table['imag_error_s_m'])):.8g}")
    print(f"wrote {out_csv}")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
