#!/usr/bin/env python3
"""Calibrate mechanistic Schwarz parameters for AC2D microfluidic SIP."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAPER = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0003-data set si-s02.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "mechanistic_calibration_v1"
DEFAULT_FIGURE = PROJECT_ROOT / "figures" / "ac2d_microfluidic" / "ac2d_mechanistic_calibration_grid.png"
DEFAULT_SIGMA_S_VALUES = np.logspace(-6.0, -3.0, 13)
DEFAULT_LENGTH_VALUES = np.array([5.0e-6, 8.0e-6, 1.0e-5, 1.3e-5, 2.0e-5, 3.0e-5, 5.0e-5])


def load_paper_2p5(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=";")
    return table.rename(
        columns={
            "Time (h)": "time_h",
            "Real conductivity (S/m)": "paper_sigma_real_s_m",
            "Imaginary conductivity (S/m)": "paper_sigma_imag_s_m",
        }
    )


def score_run(results_path: Path, paper: pd.DataFrame) -> dict[str, float]:
    ours = pd.read_csv(results_path)
    ours = ours[(ours["mechanism"] == "all") & np.isclose(ours["frequency_hz"], 2.5)].sort_values("time_h")
    if ours.empty:
        raise ValueError(f"no 2.5 Hz all-mechanism rows in {results_path}")
    mask = (paper["time_h"] >= ours["time_h"].min()) & (paper["time_h"] <= ours["time_h"].max())
    if not mask.any():
        raise ValueError("paper and AC2D time ranges do not overlap")
    interp_imag = np.interp(paper.loc[mask, "time_h"], ours["time_h"], ours["sigma_imag_s_m"])
    interp_real = np.interp(paper.loc[mask, "time_h"], ours["time_h"], ours["sigma_real_s_m"])
    return {
        "imag_mae_s_m": float(np.mean(np.abs(paper.loc[mask, "paper_sigma_imag_s_m"] - interp_imag))),
        "real_mae_s_m": float(np.mean(np.abs(paper.loc[mask, "paper_sigma_real_s_m"] - interp_real))),
        "imag_peak_s_m": float(np.max(np.abs(ours["sigma_imag_s_m"]))),
        "real_min_s_m": float(np.min(ours["sigma_real_s_m"])),
        "real_max_s_m": float(np.max(ours["sigma_real_s_m"])),
    }


def run_sweep(
    *,
    sigma_s_s: float,
    characteristic_length_m: float,
    out_dir: Path,
    frames: list[int],
    downsample: int,
) -> Path:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "run_ac2d_microfluidic_sweep.py"),
        "--driver-mode",
        "si03",
        "--mechanisms",
        "maxwell",
        "interface",
        "all",
        "--frequencies",
        "2.5",
        "--downsample",
        str(downsample),
        "--schwarz-sigma-s",
        str(sigma_s_s),
        "--schwarz-characteristic-length-m",
        str(characteristic_length_m),
        "--out-dir",
        str(out_dir),
        "--figure-dir",
        str(out_dir / "figures"),
    ]
    if frames:
        command.extend(["--frames", *[str(frame) for frame in frames]])
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return out_dir / "ac2d_sweep_results.csv"


def safe_name(value: float) -> str:
    return f"{value:.3e}".replace("+", "").replace("-", "m").replace(".", "p")


def plot_grid(table: pd.DataFrame, figure_path: Path) -> None:
    pivot = table.pivot(index="characteristic_length_m", columns="sigma_s_s", values="imag_mae_s_m")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([f"{value:.0e}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticklabels([f"{value * 1e6:.1f}" for value in pivot.index])
    ax.set_xlabel("Sigma_s (S)")
    ax.set_ylabel("Characteristic length (um)")
    ax.set_title("AC2D mechanistic calibration, 2.5 Hz imaginary MAE")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("MAE of sigma'' (S/m)")
    best = table.loc[table["imag_mae_s_m"].idxmin()]
    ax.text(
        0.02,
        0.98,
        f"best: Sigma_s={best['sigma_s_s']:.2e} S, r={best['characteristic_length_m'] * 1e6:.1f} um",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
    )
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=260, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_values(values: list[str] | None, default: np.ndarray) -> np.ndarray:
    if not values:
        return default.astype(float)
    return np.asarray([float(value) for value in values], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=str(DEFAULT_PAPER))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--sigma-s-values", nargs="*")
    parser.add_argument("--length-values", nargs="*")
    parser.add_argument("--frames", nargs="+", type=int, default=list(range(1, 29)))
    parser.add_argument("--downsample", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    run_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    paper = load_paper_2p5(Path(args.paper))
    sigma_s_values = parse_values(args.sigma_s_values, DEFAULT_SIGMA_S_VALUES)
    length_values = parse_values(args.length_values, DEFAULT_LENGTH_VALUES)

    rows: list[dict[str, float | str]] = []
    for sigma_s in sigma_s_values:
        for length in length_values:
            case_dir = run_dir / f"sigma_{safe_name(float(sigma_s))}_r_{safe_name(float(length))}"
            results_path = run_sweep(
                sigma_s_s=float(sigma_s),
                characteristic_length_m=float(length),
                out_dir=case_dir,
                frames=args.frames,
                downsample=args.downsample,
            )
            score = score_run(results_path, paper)
            rows.append(
                {
                    "sigma_s_s": float(sigma_s),
                    "characteristic_length_m": float(length),
                    "tau_s": float(length**2 / (2.0 * 1.3e-9)),
                    "results_path": str(results_path),
                    **score,
                }
            )

    table = pd.DataFrame(rows).sort_values(["imag_mae_s_m", "real_mae_s_m"]).reset_index(drop=True)
    out_csv = out_dir / "calibration_grid.csv"
    table.to_csv(out_csv, index=False)
    plot_grid(table, Path(args.figure))
    print(f"best {table.iloc[0].to_dict()}")
    print(f"wrote {out_csv}")
    print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
