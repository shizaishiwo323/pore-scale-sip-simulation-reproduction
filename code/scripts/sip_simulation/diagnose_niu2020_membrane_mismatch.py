#!/usr/bin/env python3
"""Diagnose the Niu 2020 membrane-polarization mismatch.

This is a fast surrogate diagnostic. It does not replace a full AC3D rerun:
it reuses the already converged membrane-only full-grid AC3D result to estimate
the field upscaling factor, then scans explicit throat-length and Zdc scales in
the Titov membrane input spectrum.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_polarization_spectra import compute_spectra, load_pnextract  # noqa: E402
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_paper_membrane_figure8(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(
        data_dir / "Figure8.xlsx",
        [9, 10, 11],
        ["frequency_hz", "paper_membrane_imag_s_m", "paper_membrane_real_relative_permittivity"],
    )


def log_rmse_ratio(model: np.ndarray, target: np.ndarray) -> float:
    model = np.asarray(model, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(model) & np.isfinite(target) & (model > 0) & (target > 0)
    if not np.any(mask):
        return float("inf")
    return float(np.sqrt(np.mean((np.log10(model[mask]) - np.log10(target[mask])) ** 2)))


def grid_search_membrane_scales(
    frequency_hz: np.ndarray,
    target_imag_s_m: np.ndarray,
    model_func: Callable[[float, float], np.ndarray],
    length_scales: np.ndarray,
    zdc_scales: np.ndarray,
) -> dict[str, float]:
    best: dict[str, float] = {
        "best_length_scale": float("nan"),
        "best_zdc_scale": float("nan"),
        "best_score": float("inf"),
    }
    rows = []
    for length_scale in length_scales:
        for zdc_scale in zdc_scales:
            model = model_func(float(length_scale), float(zdc_scale))
            score = log_rmse_ratio(model, target_imag_s_m)
            rows.append((float(length_scale), float(zdc_scale), float(score)))
            if score < best["best_score"]:
                best = {
                    "best_length_scale": float(length_scale),
                    "best_zdc_scale": float(zdc_scale),
                    "best_score": float(score),
                }
    best["n_trials"] = float(len(rows))
    return best


def estimate_field_factor(
    ac3d_membrane_csv: Path,
    base_spectrum_csv: Path,
) -> dict[str, float]:
    ac3d = pd.read_csv(ac3d_membrane_csv)
    base = pd.read_csv(base_spectrum_csv)
    ratios: list[float] = []
    for _, row in ac3d.iterrows():
        freq = float(row["frequency_hz"])
        idx = int(np.argmin(np.abs(np.log(base["frequency_hz"].to_numpy(dtype=float) / freq))))
        delta = float(base.iloc[idx]["delta_sigma_membrane_imag_s_m"])
        eff = float(row["effective_sigma_imag_s_m"])
        if np.isfinite(delta) and delta > 0 and np.isfinite(eff):
            ratios.append(eff / delta)
    ratios_array = np.asarray(ratios, dtype=float)
    return {
        "median": float(np.median(ratios_array)),
        "mean": float(np.mean(ratios_array)),
        "min": float(np.min(ratios_array)),
        "max": float(np.max(ratios_array)),
        "n": float(len(ratios_array)),
    }


def make_model_function(
    frequency_hz: np.ndarray,
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    params: PolarizationParameters,
    field_factor: float,
) -> Callable[[float, float], np.ndarray]:
    def model(length_scale: float, zdc_scale: float) -> np.ndarray:
        spectra, _ = compute_spectra(
            frequency_hz,
            pores,
            throats,
            params,
            figure5_zdc_ohm=None,
            membrane_length_scale=length_scale,
            membrane_zdc_scale=zdc_scale,
        )
        return field_factor * spectra["delta_sigma_membrane_imag_s_m"].to_numpy(dtype=float)

    return model


def write_summary(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Niu 2020 Membrane Polarization Mismatch Diagnosis",
        "",
        "This diagnostic uses the paper Figure 8 membrane component as the target and the current full-resolution pnextract network as the source geometry.",
        "It scans explicit scales applied to Titov-model throat length `L` and dc resistance `Zdc` before any expensive AC3D rerun.",
        "",
        "## Key Result",
        "",
        f"- Median AC3D field factor from the previous membrane-only run: `{summary['field_factor_median']:.6g}`.",
        f"- Best membrane length scale: `{summary['best_length_scale']:.6g}`.",
        f"- Best Zdc scale: `{summary['best_zdc_scale']:.6g}`.",
        f"- Log10 RMSE against paper membrane imaginary conductivity: `{summary['best_score']:.6g}` dex.",
        f"- Unscaled log10 RMSE: `{summary['unscaled_score']:.6g}` dex.",
        "",
        "## Interpretation",
        "",
        "The implemented Titov formula is retained. The fitted scales indicate that the current pnextract `throat_length_m`/geometry-derived `Zdc` are not paper-consistent inputs for the membrane component.",
        "This should be treated as a transparent calibration candidate, not as proof of the original authors' hidden network definitions.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(paper: pd.DataFrame, unscaled: np.ndarray, best_model: np.ndarray, figure_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    ax.loglog(paper["frequency_hz"], paper["paper_membrane_imag_s_m"], "o", label="Paper Figure 8 membrane")
    ax.loglog(paper["frequency_hz"], unscaled, "--", label="Current pnextract membrane")
    ax.loglog(paper["frequency_hz"], best_model, "-", label="Scaled membrane surrogate")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Imaginary conductivity (S/m)")
    ax.grid(True, which="both", linewidth=0.3, color="#dddddd")
    ax.legend(frameon=False, fontsize=7)
    figure_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {".png": {"dpi": 360}, ".svg": {}, ".pdf": {}}.items():
        fig.savefig(figure_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "Niu 2020data"))
    parser.add_argument("--network-dir", default=str(ROOT / "results" / "pnextract" / "niu2020_berea_fullres" / "network_parsed"))
    parser.add_argument(
        "--base-spectrum",
        default=str(ROOT / "results" / "spectra" / "niu2020_berea_fullres_pnextract" / "polarization_spectra_from_pnextract.csv"),
    )
    parser.add_argument(
        "--ac3d-membrane",
        default=str(ROOT / "results" / "niu2020_berea_full350_membrane_fft_x" / "sweep_results.csv"),
    )
    parser.add_argument("--out-csv", default=str(ROOT / "results" / "niu2020" / "niu2020_membrane_mismatch_diagnosis.csv"))
    parser.add_argument("--metadata-out", default=str(ROOT / "results" / "niu2020" / "niu2020_membrane_mismatch_diagnosis.json"))
    parser.add_argument("--summary-md", default=str(ROOT / "results" / "niu2020" / "niu2020_membrane_mismatch_diagnosis.md"))
    parser.add_argument("--figure-base", default=str(ROOT / "figures" / "niu2020" / "niu2020_membrane_mismatch_diagnosis"))
    args = parser.parse_args()

    params = PolarizationParameters()
    paper = read_paper_membrane_figure8(Path(args.data_dir))
    pores, throats = load_pnextract(Path(args.network_dir), params)
    field_factor = estimate_field_factor(Path(args.ac3d_membrane), Path(args.base_spectrum))
    frequencies = paper["frequency_hz"].to_numpy(dtype=float)
    target = paper["paper_membrane_imag_s_m"].to_numpy(dtype=float)
    model_func = make_model_function(frequencies, pores, throats, params, field_factor["median"])

    length_scales = np.logspace(-4, 0, 81)
    zdc_scales = np.logspace(-1, 3, 81)
    best = grid_search_membrane_scales(frequencies, target, model_func, length_scales, zdc_scales)
    unscaled = model_func(1.0, 1.0)
    best_model = model_func(best["best_length_scale"], best["best_zdc_scale"])

    out = paper.copy()
    out["current_pnextract_membrane_imag_s_m"] = unscaled
    out["scaled_membrane_surrogate_imag_s_m"] = best_model
    out["current_to_paper_ratio"] = out["current_pnextract_membrane_imag_s_m"] / out["paper_membrane_imag_s_m"]
    out["scaled_to_paper_ratio"] = out["scaled_membrane_surrogate_imag_s_m"] / out["paper_membrane_imag_s_m"]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    metadata = {
        "field_factor": field_factor,
        "best": best,
        "unscaled_score": log_rmse_ratio(unscaled, target),
        "parameters": params.__dict__,
        "network_dir": str(Path(args.network_dir)),
        "base_spectrum": str(Path(args.base_spectrum)),
        "ac3d_membrane": str(Path(args.ac3d_membrane)),
        "note": "Fast surrogate only; rerun full AC3D with scaled component spectra before using as final reproduction.",
    }
    metadata_path = Path(args.metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "field_factor_median": field_factor["median"],
        "best_length_scale": best["best_length_scale"],
        "best_zdc_scale": best["best_zdc_scale"],
        "best_score": best["best_score"],
        "unscaled_score": metadata["unscaled_score"],
    }
    write_summary(Path(args.summary_md), summary)
    make_plot(paper, unscaled, best_model, Path(args.figure_base))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
