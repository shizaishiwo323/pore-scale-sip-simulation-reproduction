#!/usr/bin/env python3
"""Reverse-constrain the short-throat deficit against Niu 2020 Figure5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "sip_simulation"))

from pore_scale_electrical.polarization import (  # noqa: E402
    PolarizationParameters,
    membrane_polarization_conductance,
    throat_zdc_from_geometry,
    upscale_conductance_to_water_conductivity,
)
from scan_pnextract_for_niu2020_membrane import (  # noqa: E402
    cumulative_fraction,
    log_edges,
    normalize,
    read_figure5,
    weighted_distribution,
)


DEFAULT_NETWORK_DIR = (
    PROJECT_ROOT
    / "results"
    / "niu2020_berea_dong_blunt_beta_floor_scan_v2"
    / "recommended_figure5_candidate_beta_0p625_floor_0p25"
)
DEFAULT_FIGURE5 = PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_short_throat_reverse_constraint_v1"


def distribution_on_centers(values_m: np.ndarray, weights: np.ndarray, centers_m: np.ndarray) -> np.ndarray:
    return weighted_distribution(np.asarray(values_m, dtype=float), np.asarray(weights, dtype=float), np.asarray(centers_m, dtype=float))


def bin_indices(values_m: np.ndarray, centers_m: np.ndarray) -> np.ndarray:
    edges = log_edges(np.asarray(centers_m, dtype=float))
    return np.digitize(np.asarray(values_m, dtype=float), edges) - 1


def bin_deficit_table(centers_m: np.ndarray, paper_y: np.ndarray, candidate_y: np.ndarray) -> pd.DataFrame:
    paper = normalize(np.asarray(paper_y, dtype=float))
    candidate = normalize(np.asarray(candidate_y, dtype=float))
    delta = paper - candidate
    return pd.DataFrame(
        {
            "bin_center_m": centers_m,
            "bin_center_um": centers_m * 1.0e6,
            "paper_relative_volume": paper,
            "candidate_relative_volume": candidate,
            "paper_minus_candidate": delta,
            "positive_deficit_fraction": np.maximum(delta, 0.0),
            "excess_fraction": np.maximum(-delta, 0.0),
        }
    )


def reweight_to_paper_bins(
    lengths_m: np.ndarray,
    weights: np.ndarray,
    centers_m: np.ndarray,
    paper_y: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    lengths = np.asarray(lengths_m, dtype=float)
    original_weights = np.asarray(weights, dtype=float)
    centers = np.asarray(centers_m, dtype=float)
    paper = normalize(np.asarray(paper_y, dtype=float))

    valid = np.isfinite(lengths) & np.isfinite(original_weights) & (lengths > 0.0) & (original_weights > 0.0)
    adjusted = np.zeros_like(original_weights, dtype=float)
    total_original = float(original_weights[valid].sum())
    indices = bin_indices(lengths, centers)
    candidate_mass = np.zeros(len(centers), dtype=float)
    for idx in range(len(centers)):
        mask = valid & (indices == idx)
        candidate_mass[idx] = original_weights[mask].sum()

    reachable = candidate_mass > 0.0
    unreachable_paper_fraction = float(paper[~reachable].sum())
    reachable_paper = paper.copy()
    reachable_paper[~reachable] = 0.0
    if reachable_paper.sum() <= 0.0:
        raise ValueError("no reachable paper bins overlap the candidate throat lengths")
    reachable_paper = reachable_paper / reachable_paper.sum()

    for idx in range(len(centers)):
        mask = valid & (indices == idx)
        if not np.any(mask):
            continue
        adjusted[mask] = original_weights[mask] * (reachable_paper[idx] * total_original / candidate_mass[idx])

    details = {
        "original_total_weight": total_original,
        "adjusted_total_weight": float(adjusted.sum()),
        "unreachable_paper_fraction": unreachable_paper_fraction,
        "reachable_paper_fraction": float(1.0 - unreachable_paper_fraction),
    }
    return adjusted, details


def relaxation_frequency_hz(lengths_m: np.ndarray, diffusion_coefficient_m2_s: float) -> np.ndarray:
    lengths = np.asarray(lengths_m, dtype=float)
    tau = lengths**2 / (4.0 * diffusion_coefficient_m2_s)
    return 1.0 / (2.0 * np.pi * tau)


def zdc_from_throats(throats: pd.DataFrame, lengths_m: np.ndarray, params: PolarizationParameters) -> np.ndarray:
    return throat_zdc_from_geometry(
        length_m=np.asarray(lengths_m, dtype=float),
        radius_m=throats["throat_radius_m"].to_numpy(dtype=float),
        shape_factor=throats["throat_shape_factor"].to_numpy(dtype=float),
        water_conductivity_s_m=params.water_conductivity_s_m,
    )


def weighted_fraction(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    denom = float(weights[valid].sum())
    if denom <= 0:
        return float("nan")
    return float(weights[valid & mask].sum() / denom)


def membrane_proxy_metrics(
    label: str,
    lengths_m: np.ndarray,
    weights: np.ndarray,
    zdc_ohm: np.ndarray,
    params: PolarizationParameters,
) -> dict[str, float | str]:
    lengths = np.asarray(lengths_m, dtype=float)
    weights = np.asarray(weights, dtype=float)
    zdc = np.asarray(zdc_ohm, dtype=float)
    valid = np.isfinite(lengths) & np.isfinite(weights) & np.isfinite(zdc) & (lengths > 0.0) & (weights > 0.0) & (zdc > 0.0)
    normalized_weights = normalize(weights[valid])
    conductance_proxy = normalized_weights / zdc[valid]
    freq = relaxation_frequency_hz(lengths[valid], params.diffusion_coefficient_m2_s)
    out: dict[str, float | str] = {
        "case": label,
        "valid_throats": int(valid.sum()),
        "mean_length_um": float(np.sum(lengths[valid] * normalized_weights) * 1.0e6),
        "median_length_um": float(np.quantile(lengths[valid], 0.5) * 1.0e6),
        "mean_zdc_ohm": float(np.sum(zdc[valid] * normalized_weights)),
        "conductance_proxy_sum_1_ohm": float(conductance_proxy.sum()),
        "relaxation_frequency_weighted_geomean_hz": float(np.exp(np.sum(np.log(freq) * normalized_weights))),
    }
    for threshold_um in (1.0, 5.0, 10.0):
        threshold = threshold_um * 1.0e-6
        short = lengths < threshold
        out[f"volume_fraction_lt_{threshold_um:g}um"] = weighted_fraction(lengths, weights, short)
        out[f"conductance_proxy_fraction_lt_{threshold_um:g}um"] = weighted_fraction(lengths[valid], conductance_proxy, short[valid])
    return out


def compute_membrane_spectrum(
    label: str,
    lengths_m: np.ndarray,
    weights: np.ndarray,
    zdc_ohm: np.ndarray,
    params: PolarizationParameters,
    frequencies_hz: np.ndarray,
) -> pd.DataFrame:
    valid = np.isfinite(lengths_m) & np.isfinite(weights) & np.isfinite(zdc_ohm) & (lengths_m > 0.0) & (weights > 0.0) & (zdc_ohm > 0.0)
    conductance = membrane_polarization_conductance(frequencies_hz, lengths_m[valid], weights[valid], zdc_ohm[valid], params)
    delta_sigma = upscale_conductance_to_water_conductivity(conductance, params)
    return pd.DataFrame(
        {
            "case": label,
            "frequency_hz": frequencies_hz,
            "delta_sigma_membrane_real_s_m": delta_sigma.real,
            "delta_sigma_membrane_imag_s_m": delta_sigma.imag,
        }
    )


def plot_deficit(deficit: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(deficit["bin_center_um"], deficit["paper_relative_volume"], "o-", label="Niu Figure5")
    ax.plot(deficit["bin_center_um"], deficit["candidate_relative_volume"], "s--", label="Current exported L")
    ax.bar(deficit["bin_center_um"], deficit["positive_deficit_fraction"], width=0.12 * deficit["bin_center_um"], alpha=0.35, label="Positive deficit")
    ax.set_xscale("log")
    ax.set_xlabel("Pore throat length (um)")
    ax.set_ylabel("Relative volume")
    ax.set_title("Short-throat deficit against Niu Figure5")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_spectra(spectra: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for case, group in spectra.groupby("case"):
        ax.loglog(group["frequency_hz"], np.abs(group["delta_sigma_membrane_imag_s_m"]), label=case)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Delta sigma'' membrane| (S/m)")
    ax.set_title("Synthetic short-throat membrane sensitivity")
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_readme(out_dir: Path, network_dir: Path, figure5: Path, threshold_table: pd.DataFrame, proxy: pd.DataFrame) -> None:
    rows = threshold_table.set_index("threshold_um")
    current = proxy.set_index("case").loc["current_exported"]
    reweighted = proxy.set_index("case").loc["reweighted_to_figure5_bins"]
    lines = [
        "# Niu 2020 Short-Throat Reverse Constraint",
        "",
        "This diagnostic estimates how much short-throat relative volume is missing from the current pnextract network and how a Figure5-like reweighting changes membrane-polarization proxy quantities.",
        "",
        f"- Network directory: `{network_dir}`",
        f"- Figure5 workbook: `{figure5}`",
        "- The synthetic correction is diagnostic only; it does not claim to reconstruct Niu's hidden pore-network topology.",
        "",
        "## Short-throat deficits",
        "",
        "| Threshold | Niu Figure5 | Current exported L | Missing absolute fraction |",
        "| --- | ---: | ---: | ---: |",
    ]
    for threshold in (1.0, 5.0, 10.0):
        row = rows.loc[threshold]
        lines.append(
            f"| <{threshold:g} um | {row['paper_fraction']*100:.4g}% | {row['candidate_fraction']*100:.4g}% | {row['missing_fraction']*100:.4g}% |"
        )
    lines.extend(
        [
            "",
            "## Membrane proxy shift",
            "",
            f"- Current exported mean length: `{current['mean_length_um']:.4g} um`; reweighted mean length: `{reweighted['mean_length_um']:.4g} um`.",
            f"- Current conductance proxy sum: `{current['conductance_proxy_sum_1_ohm']:.6g}`; reweighted: `{reweighted['conductance_proxy_sum_1_ohm']:.6g}`.",
            f"- Current relaxation-frequency weighted geomean: `{current['relaxation_frequency_weighted_geomean_hz']:.6g} Hz`; reweighted: `{reweighted['relaxation_frequency_weighted_geomean_hz']:.6g} Hz`.",
            "",
            "## Outputs",
            "",
            "- `short_throat_threshold_deficits.csv`: cumulative deficits below 1, 5, and 10 um.",
            "- `figure5_bin_deficit.csv`: bin-by-bin positive deficit and excess volume fraction.",
            "- `membrane_proxy_metrics.csv`: Zdc, conductance, relaxation-frequency proxy metrics.",
            "- `synthetic_membrane_spectra.csv`: current vs reweighted membrane input spectra without AC3D field solve.",
            "- `figures/short_throat_deficit.png` and `figures/synthetic_membrane_sensitivity.png`.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-dir", default=str(DEFAULT_NETWORK_DIR))
    parser.add_argument("--figure5", default=str(DEFAULT_FIGURE5))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    network_dir = Path(args.network_dir)
    figure5 = Path(args.figure5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    params = PolarizationParameters()
    _, paper_throat = read_figure5(figure5)
    centers = paper_throat["center_m"].to_numpy(dtype=float)
    paper_y = normalize(paper_throat["relative_volume"].to_numpy(dtype=float))
    throats = pd.read_csv(network_dir / "throats.csv")
    lengths = throats["throat_length_m"].to_numpy(dtype=float)
    weights = throats["throat_volume_m3"].to_numpy(dtype=float)
    candidate_y = distribution_on_centers(lengths, weights, centers)

    threshold_rows = []
    for threshold_um in (1.0, 5.0, 10.0):
        threshold_m = threshold_um * 1.0e-6
        paper_fraction = cumulative_fraction(centers, paper_y, threshold_m)
        candidate_fraction = cumulative_fraction(lengths, weights, threshold_m)
        threshold_rows.append(
            {
                "threshold_um": threshold_um,
                "paper_fraction": paper_fraction,
                "candidate_fraction": candidate_fraction,
                "missing_fraction": max(paper_fraction - candidate_fraction, 0.0),
                "candidate_to_paper_ratio": candidate_fraction / paper_fraction if paper_fraction > 0 else np.nan,
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)

    deficit = bin_deficit_table(centers, paper_y, candidate_y)
    adjusted_weights, reweight_details = reweight_to_paper_bins(lengths, weights, centers, paper_y)
    zdc = zdc_from_throats(throats, lengths, params)
    proxy = pd.DataFrame(
        [
            membrane_proxy_metrics("current_exported", lengths, weights, zdc, params),
            membrane_proxy_metrics("reweighted_to_figure5_bins", lengths, adjusted_weights, zdc, params),
        ]
    )

    frequencies = np.logspace(-3, 9, 97)
    spectra = pd.concat(
        [
            compute_membrane_spectrum("current_exported", lengths, weights, zdc, params, frequencies),
            compute_membrane_spectrum("reweighted_to_figure5_bins", lengths, adjusted_weights, zdc, params, frequencies),
        ],
        ignore_index=True,
    )

    threshold_table.to_csv(out_dir / "short_throat_threshold_deficits.csv", index=False)
    deficit.to_csv(out_dir / "figure5_bin_deficit.csv", index=False)
    pd.DataFrame([reweight_details]).to_csv(out_dir / "synthetic_reweighting_details.csv", index=False)
    proxy.to_csv(out_dir / "membrane_proxy_metrics.csv", index=False)
    spectra.to_csv(out_dir / "synthetic_membrane_spectra.csv", index=False)
    plot_deficit(deficit, out_dir / "figures" / "short_throat_deficit.png")
    plot_spectra(spectra, out_dir / "figures" / "synthetic_membrane_sensitivity.png")

    summary = {
        "network_dir": str(network_dir),
        "figure5": str(figure5),
        "n_throats": int(len(throats)),
        "reweighting": reweight_details,
        "threshold_deficits": threshold_table.to_dict(orient="records"),
        "proxy_metrics": proxy.to_dict(orient="records"),
    }
    (out_dir / "short_throat_reverse_constraint_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, network_dir, figure5, threshold_table, proxy)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
