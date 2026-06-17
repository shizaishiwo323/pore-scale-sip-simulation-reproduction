#!/usr/bin/env python3
"""Scan pnextract parameters for Niu 2020 membrane-polarization geometry.

The scan is intentionally geometry-first. It scores candidate pnextract
networks against the Niu Figure 4/Figure5 pore-node and pore-throat
distributions before any AC3D rerun.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "pore_network"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "sip_simulation"))

from compute_polarization_spectra import compute_spectra, default_frequencies, load_pnextract  # noqa: E402
from make_polarization_component_spectra import make_component_spectrum  # noqa: E402
from parse_pnextract_network import main as parse_main  # noqa: F401,E402
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402
from run_segmented_core_pnextract_ballstick import (  # noqa: E402
    expected_pnextract_prefix,
    extract_pnextract_bin,
    find_pnextract_exe,
    prepare_pnextract_input,
    run_command,
    verify_pnextract_outputs,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    min_r_pore: float | None = None
    medial_surface_settings: tuple[float, float, float, float, float, float, int, float, float] | None = None

    def pnextract_lines(self) -> list[str]:
        lines: list[str] = []
        if self.min_r_pore is not None:
            lines.append(f"minRPore {self.min_r_pore:.8g}")
        if self.medial_surface_settings is not None:
            parts = []
            for value in self.medial_surface_settings:
                if isinstance(value, int):
                    parts.append(str(value))
                else:
                    parts.append(f"{value:.8g}")
            lines.append("medialSurfaceSettings " + " ".join(parts))
        return lines


def default_candidates() -> list[Candidate]:
    candidates: list[Candidate] = [Candidate("baseline_default")]
    for min_r in (0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        candidates.append(Candidate(f"minRPore_{min_r:g}", min_r_pore=min_r))

    for min_r in (0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0):
        candidates.append(
            Candidate(
                f"fineA_minRPore_{min_r:g}",
                min_r_pore=min_r,
                medial_surface_settings=(0.05, 0.98, 0.65, max(min_r + 0.25, 0.5), 0.8, 1.2, 1, 0.05, min_r),
            )
        )
        candidates.append(
            Candidate(
                f"fineB_minRPore_{min_r:g}",
                min_r_pore=min_r,
                medial_surface_settings=(0.05, 0.98, 0.70, max(min_r + 0.25, 0.5), 1.0, 1.35, 0, 0.02, max(0.1, min_r * 0.5)),
            )
        )
        candidates.append(
            Candidate(
                f"short_split_minRPore_{min_r:g}",
                min_r_pore=min_r,
                medial_surface_settings=(0.05, 0.98, 0.60, max(min_r + 0.10, 0.35), 1.2, 1.5, 0, 0.0, 0.1),
            )
        )
        candidates.append(
            Candidate(
                f"ultra_short_minRPore_{min_r:g}",
                min_r_pore=min_r,
                medial_surface_settings=(0.02, 0.98, 0.50, max(min_r + 0.05, 0.15), 1.6, 1.8, 0, 0.0, 0.05),
            )
        )
    return candidates


def broad_mult_parameter_candidates() -> list[Candidate]:
    """A broader deterministic multi-parameter scan.

    The recipes vary all medial-surface parameters, not just ``minRPore``.
    Values are deliberately centered on small-scale throat preservation while
    keeping a few coarser/smoother alternatives to protect the pore-node fit.
    """

    candidates: list[Candidate] = [Candidate("baseline_default")]
    min_r_values = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
    recipes: list[tuple[str, tuple[float, float, float, float, float, float, int, float, float]]] = [
        ("very_fine_low_mid", (0.00, 0.98, 0.35, 0.05, 1.60, 1.80, 0, 0.00, 0.00)),
        ("very_fine_mid", (0.02, 0.98, 0.50, 0.10, 1.60, 1.80, 0, 0.00, 0.05)),
        ("fine_balanced", (0.02, 0.98, 0.60, 0.15, 1.20, 1.50, 0, 0.00, 0.10)),
        ("fine_high_len", (0.02, 0.98, 0.60, 0.15, 2.00, 2.00, 0, 0.00, 0.05)),
        ("fine_more_noise", (0.02, 0.98, 0.65, 0.30, 1.60, 1.80, 1, 0.02, 0.10)),
        ("low_mid_low_noise", (0.05, 0.98, 0.40, 0.10, 1.20, 1.40, 0, 0.00, 0.05)),
        ("mid_low_noise", (0.05, 0.98, 0.70, 0.10, 1.20, 1.35, 0, 0.02, 0.05)),
        ("mid_smooth_guard", (0.05, 0.98, 0.70, 0.35, 0.80, 1.20, 1, 0.05, 0.25)),
        ("coarse_guard", (0.05, 0.98, 0.80, 0.60, 0.60, 1.10, 2, 0.10, 0.50)),
        ("low_yz_clip", (0.02, 0.85, 0.50, 0.15, 1.60, 1.80, 0, 0.00, 0.05)),
        ("symmetric_boundary", (0.02, 0.02, 0.50, 0.15, 1.60, 1.80, 0, 0.00, 0.05)),
        ("high_mid_split", (0.02, 0.98, 0.85, 0.10, 1.80, 2.20, 0, 0.00, 0.05)),
    ]

    for min_r in min_r_values:
        candidates.append(Candidate(f"minRPore_only_{min_r:g}", min_r_pore=min_r))
        for recipe_name, settings in recipes:
            candidates.append(
                Candidate(
                    f"{recipe_name}_minRPore_{min_r:g}",
                    min_r_pore=min_r,
                    medial_surface_settings=settings,
                )
            )

    # A few no-minRPore controls let pnextract keep its distance-map-derived
    # minRPore while only medial-surface parameters change.
    for recipe_name, settings in recipes[:6]:
        candidates.append(Candidate(f"{recipe_name}_autoMinR", medial_surface_settings=settings))
    return candidates


def candidates_for_set(name: str) -> list[Candidate]:
    if name == "focused":
        return default_candidates()
    if name == "broad":
        return broad_mult_parameter_candidates()
    if name == "combined":
        by_name: dict[str, Candidate] = {}
        for candidate in default_candidates() + broad_mult_parameter_candidates():
            by_name[candidate.name] = candidate
        return list(by_name.values())
    raise ValueError(f"unknown candidate set: {name}")


def read_figure5(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(path, sheet_name=0)
    pore = pd.DataFrame(
        {
            "center_m": pd.to_numeric(raw["pore node size (m)"], errors="coerce"),
            "relative_volume": pd.to_numeric(raw["pore node relaive volume"], errors="coerce"),
        }
    ).dropna()
    throat = pd.DataFrame(
        {
            "center_m": pd.to_numeric(raw["pore throat length (m)"], errors="coerce"),
            "relative_volume": pd.to_numeric(raw["Pore throat relative volume"], errors="coerce"),
        }
    ).dropna()
    pore["relative_volume"] = normalize(pore["relative_volume"].to_numpy(dtype=float))
    throat["relative_volume"] = normalize(throat["relative_volume"].to_numpy(dtype=float))
    return pore, throat


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("relative-volume values must sum to a positive number")
    return values / total


def log_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    centers = centers[np.isfinite(centers) & (centers > 0)]
    centers = np.sort(centers)
    mids = np.sqrt(centers[:-1] * centers[1:])
    return np.concatenate([[centers[0] ** 2 / mids[0]], mids, [centers[-1] ** 2 / mids[-1]]])


def weighted_distribution(values: np.ndarray, weights: np.ndarray, centers: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    hist, _ = np.histogram(values[mask], bins=log_edges(centers), weights=weights[mask])
    return normalize(hist)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    values = values[mask]
    weights = normalize(weights[mask])
    order = np.argsort(values)
    values = values[order]
    cdf = np.cumsum(weights[order])
    return float(values[np.searchsorted(cdf, quantile, side="left")])


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    values = values[mask]
    weights = normalize(weights[mask])
    return {
        "min_m": float(values.min()),
        "mean_m": float(np.sum(values * weights)),
        "q10_m": weighted_quantile(values, weights, 0.10),
        "median_m": weighted_quantile(values, weights, 0.50),
        "q90_m": weighted_quantile(values, weights, 0.90),
        "max_m": float(values.max()),
    }


def cumulative_fraction(values: np.ndarray, weights: np.ndarray, threshold_m: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    values = values[mask]
    weights = normalize(weights[mask])
    return float(weights[values < threshold_m].sum())


def evaluate_network(
    paper_pore: pd.DataFrame,
    paper_throat: pd.DataFrame,
    pores: pd.DataFrame,
    throats: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    paper_pore_y = paper_pore["relative_volume"].to_numpy(dtype=float)
    paper_throat_y = paper_throat["relative_volume"].to_numpy(dtype=float)
    pore_centers = paper_pore["center_m"].to_numpy(dtype=float)
    throat_centers = paper_throat["center_m"].to_numpy(dtype=float)

    pore_y = weighted_distribution(
        pores["pore_radius_m"].to_numpy(dtype=float),
        pores["pore_volume_m3"].to_numpy(dtype=float),
        pore_centers,
    )
    throat_y = weighted_distribution(
        throats["throat_length_m"].to_numpy(dtype=float),
        throats["throat_volume_m3"].to_numpy(dtype=float),
        throat_centers,
    )

    paper_pore_stats = weighted_stats(pore_centers, paper_pore_y)
    paper_throat_stats = weighted_stats(throat_centers, paper_throat_y)
    pore_stats = weighted_stats(pores["pore_radius_m"].to_numpy(dtype=float), pores["pore_volume_m3"].to_numpy(dtype=float))
    throat_stats = weighted_stats(
        throats["throat_length_m"].to_numpy(dtype=float),
        throats["throat_volume_m3"].to_numpy(dtype=float),
    )

    thresholds = [1.0e-6, 5.0e-6, 10.0e-6]
    short_errors = []
    metrics: dict[str, float] = {
        "pore_l1": float(np.abs(pore_y - paper_pore_y).sum()),
        "throat_l1": float(np.abs(throat_y - paper_throat_y).sum()),
        "pore_mean_ratio": pore_stats["mean_m"] / paper_pore_stats["mean_m"],
        "throat_mean_ratio": throat_stats["mean_m"] / paper_throat_stats["mean_m"],
        "pore_median_ratio": pore_stats["median_m"] / paper_pore_stats["median_m"],
        "throat_median_ratio": throat_stats["median_m"] / paper_throat_stats["median_m"],
        "n_pores": float(len(pores)),
        "n_throats": float(len(throats)),
    }
    for threshold in thresholds:
        key = f"throat_volume_fraction_lt_{threshold * 1e6:g}um"
        paper_fraction = cumulative_fraction(throat_centers, paper_throat_y, threshold)
        candidate_fraction = cumulative_fraction(
            throats["throat_length_m"].to_numpy(dtype=float),
            throats["throat_volume_m3"].to_numpy(dtype=float),
            threshold,
        )
        metrics[f"paper_{key}"] = paper_fraction
        metrics[f"candidate_{key}"] = candidate_fraction
        short_errors.append(abs(candidate_fraction - paper_fraction))

    pore_constraint_penalty = max(0.0, abs(metrics["pore_mean_ratio"] - 1.0) - 0.10) * 4.0
    metrics["objective_score"] = (
        2.0 * metrics["throat_l1"]
        + 0.5 * metrics["pore_l1"]
        + 12.0 * short_errors[0]
        + 8.0 * short_errors[1]
        + 5.0 * short_errors[2]
        + pore_constraint_penalty
    )

    comparison = pd.concat(
        [
            pd.DataFrame(
                {
                    "quantity": "pore_node_size",
                    "bin_center_m": pore_centers,
                    "paper_relative_volume": paper_pore_y,
                    "candidate_relative_volume": pore_y,
                }
            ),
            pd.DataFrame(
                {
                    "quantity": "pore_throat_length",
                    "bin_center_m": throat_centers,
                    "paper_relative_volume": paper_throat_y,
                    "candidate_relative_volume": throat_y,
                }
            ),
        ],
        ignore_index=True,
    )
    return metrics, comparison


def append_pnextract_lines(mhd_path: Path, lines: list[str]) -> None:
    if not lines:
        return
    with mhd_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        for line in lines:
            handle.write(line + "\n")


def build_parse_command(python_exe: Path, prefix: Path, outdir: Path) -> list[str]:
    return [
        str(python_exe),
        str(CODE_ROOT / "scripts" / "pore_network" / "parse_pnextract_network.py"),
        "--prefix",
        str(prefix),
        "--outdir",
        str(outdir),
    ]


def run_candidate(
    candidate: Candidate,
    *,
    segmented_volume: Path,
    out_dir: Path,
    voxel_size_um: float,
    pnextract_exe: Path,
    python_exe: Path,
    paper_pore: pd.DataFrame,
    paper_throat: pd.DataFrame,
    resume: bool,
) -> dict[str, object]:
    candidate_dir = out_dir / "candidates" / candidate.name
    prepare_dir = candidate_dir / "pnextract_inputs"
    network_dir = candidate_dir / "network_parsed"
    title = f"niu2020_scan_{candidate.name}"
    metrics_path = candidate_dir / "metrics.json"
    comparison_path = candidate_dir / "figure5_distribution_comparison.csv"

    if resume and metrics_path.exists() and (network_dir / "pores.csv").exists() and (network_dir / "throats.csv").exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["candidate_dir"] = str(candidate_dir)
        metrics["network_dir"] = str(network_dir)
        return metrics

    candidate_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prepare_summary = prepare_pnextract_input(
        segmented_volume,
        prepare_dir,
        title=title,
        pore_values=[0],
        voxel_size_um=voxel_size_um,
        downsample=1,
    )
    mhd_path = Path(prepare_summary["mhd_path"])
    append_pnextract_lines(mhd_path, candidate.pnextract_lines())
    prefix = expected_pnextract_prefix(prepare_dir, title)

    result: dict[str, object] = {
        "candidate": candidate.name,
        "candidate_dir": str(candidate_dir),
        "network_dir": str(network_dir),
        "parameters": asdict(candidate),
        "pnextract_lines": candidate.pnextract_lines(),
        "prepare_summary": prepare_summary,
    }
    try:
        pn_result = run_command([str(pnextract_exe), str(mhd_path.name)], cwd=prepare_dir)
        verify_pnextract_outputs(prefix)
        parse_result = run_command(build_parse_command(python_exe, prefix, network_dir), cwd=PROJECT_ROOT)
        pores = pd.read_csv(network_dir / "pores.csv")
        throats = pd.read_csv(network_dir / "throats.csv")
        metrics, comparison = evaluate_network(paper_pore, paper_throat, pores, throats)
        comparison.to_csv(comparison_path, index=False)
        result.update(metrics)
        result.update(
            {
                "status": "ok",
                "elapsed_s": time.perf_counter() - started,
                "pnextract_stdout_tail": pn_result.stdout[-4000:],
                "pnextract_stderr": pn_result.stderr,
                "parse_stdout": parse_result.stdout,
                "parse_stderr": parse_result.stderr,
                "comparison_csv": str(comparison_path),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        result.update({"status": "failed", "error": repr(exc), "elapsed_s": time.perf_counter() - started})

    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def read_paper_membrane_figure8(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    out = raw.iloc[2:, [9, 10]].copy()
    out.columns = ["frequency_hz", "paper_membrane_imag_s_m"]
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna().reset_index(drop=True)


def nearest_values(source: pd.DataFrame, target_freq: np.ndarray, column: str) -> np.ndarray:
    src_f = source["frequency_hz"].to_numpy(dtype=float)
    src_v = source[column].to_numpy(dtype=float)
    out = []
    for freq in target_freq:
        idx = int(np.argmin(np.abs(np.log(src_f / freq))))
        out.append(src_v[idx])
    return np.asarray(out, dtype=float)


def write_best_spectra(
    *,
    best_network_dir: Path,
    out_dir: Path,
    figure8_xlsx: Path,
) -> dict[str, object]:
    params = PolarizationParameters()
    spectra_dir = out_dir / "best_candidate" / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    pores, throats = load_pnextract(best_network_dir, params)
    base, base_metadata = compute_spectra(
        default_frequencies(),
        pores,
        throats,
        params,
        figure5_zdc_ohm=None,
    )
    base_path = spectra_dir / "polarization_spectra_base.csv"
    base.to_csv(base_path, index=False)
    (spectra_dir / "polarization_spectra_base_metadata.json").write_text(
        json.dumps(base_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    component = make_component_spectrum(base, "membrane", params, "paper")
    component_path = spectra_dir / "polarization_spectra_membrane.csv"
    component.to_csv(component_path, index=False)

    paper = read_paper_membrane_figure8(figure8_xlsx)
    model_delta = nearest_values(base, paper["frequency_hz"].to_numpy(dtype=float), "delta_sigma_membrane_imag_s_m")
    model_water = nearest_values(component, paper["frequency_hz"].to_numpy(dtype=float), "apparent_water_sigma_imag_s_m")
    comparison = paper.copy()
    comparison["nearest_base_delta_sigma_membrane_imag_s_m"] = model_delta
    comparison["nearest_membrane_component_water_imag_s_m"] = model_water
    comparison["delta_over_paper"] = comparison["nearest_base_delta_sigma_membrane_imag_s_m"] / comparison["paper_membrane_imag_s_m"]
    comparison_path = spectra_dir / "figure8_membrane_input_spectrum_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    ax.loglog(comparison["frequency_hz"], comparison["paper_membrane_imag_s_m"], "o", label="Niu Figure8 membrane effective")
    ax.loglog(
        comparison["frequency_hz"],
        np.abs(comparison["nearest_base_delta_sigma_membrane_imag_s_m"]),
        "s--",
        label="Best pnextract Delta sigma membrane input",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Imaginary conductivity (S/m)")
    ax.grid(True, which="both", color="#dddddd", linewidth=0.4)
    ax.legend(frameon=False, fontsize=8)
    figure_path = spectra_dir / "figure8_membrane_input_spectrum_comparison.png"
    fig.savefig(figure_path, dpi=320)
    plt.close(fig)

    return {
        "base_spectra_csv": str(base_path),
        "base_metadata_json": str(spectra_dir / "polarization_spectra_base_metadata.json"),
        "membrane_component_spectra_csv": str(component_path),
        "figure8_input_comparison_csv": str(comparison_path),
        "figure8_input_comparison_png": str(figure_path),
    }


def plot_scan_summary(summary: pd.DataFrame, paper_pore: pd.DataFrame, paper_throat: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    ok = summary[summary["status"].eq("ok")].copy()
    paths: dict[str, str] = {}
    if ok.empty:
        return paths
    ok = ok.sort_values("objective_score")
    top = ok.head(10)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.barh(top["candidate"], top["objective_score"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Objective score (lower is better)")
    ax.set_title("pnextract parameter scan score")
    score_path = out_dir / "figures" / "pnextract_scan_objective_top10.png"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(score_path, dpi=300)
    plt.close(fig)
    paths["score_top10_png"] = str(score_path)

    best = ok.iloc[0]
    cmp = pd.read_csv(Path(best["comparison_csv"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for ax, quantity, xlabel in [
        (axes[0], "pore_node_size", "Pore node size / radius (um)"),
        (axes[1], "pore_throat_length", "Pore throat length (um)"),
    ]:
        sub = cmp[cmp["quantity"].eq(quantity)]
        ax.semilogx(sub["bin_center_m"] * 1e6, sub["paper_relative_volume"] * 100.0, "o-", color="black", label="Niu Figure5")
        ax.semilogx(
            sub["bin_center_m"] * 1e6,
            sub["candidate_relative_volume"] * 100.0,
            "s--",
            color="#0072B2",
            label=f"Best scan: {best['candidate']}",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Relative volume (%)")
        ax.grid(axis="y", color="#dddddd", linewidth=0.4)
        ax.legend(frameon=False, fontsize=8)
    dist_path = out_dir / "figures" / "best_candidate_vs_niu2020_figure5.png"
    fig.savefig(dist_path, dpi=320)
    plt.close(fig)
    paths["best_distribution_png"] = str(dist_path)
    return paths


def write_summary_md(path: Path, summary: pd.DataFrame, artifacts: dict[str, object]) -> None:
    ok = summary[summary["status"].eq("ok")].copy()
    lines = [
        "# Niu 2020 pnextract Membrane Geometry Scan",
        "",
        "This run scans pnextract parameters against the Niu Figure5 pore-node and pore-throat distributions before any AC3D rerun.",
        "The objective score emphasizes pore-throat length L1 mismatch and short-throat volume fractions below 1, 5, and 10 um while penalizing broken pore-node means.",
        "",
        f"- Candidate rows: {len(summary)}",
        f"- Successful candidates: {len(ok)}",
    ]
    if not ok.empty:
        best = ok.sort_values("objective_score").iloc[0]
        lines.extend(
            [
                f"- Best candidate: `{best['candidate']}`",
                f"- Best score: `{best['objective_score']:.6g}`",
                f"- Pore L1: `{best['pore_l1']:.6g}`; throat L1: `{best['throat_l1']:.6g}`",
                f"- Pore mean ratio: `{best['pore_mean_ratio']:.6g}`; throat mean ratio: `{best['throat_mean_ratio']:.6g}`",
                f"- Candidate throat volume <1 um: `{100.0 * best['candidate_throat_volume_fraction_lt_1um']:.4g}%` vs paper `{100.0 * best['paper_throat_volume_fraction_lt_1um']:.4g}%`",
                f"- Candidate throat volume <5 um: `{100.0 * best['candidate_throat_volume_fraction_lt_5um']:.4g}%` vs paper `{100.0 * best['paper_throat_volume_fraction_lt_5um']:.4g}%`",
                f"- Candidate throat volume <10 um: `{100.0 * best['candidate_throat_volume_fraction_lt_10um']:.4g}%` vs paper `{100.0 * best['paper_throat_volume_fraction_lt_10um']:.4g}%`",
                "",
                "## Best Candidate pnextract Lines",
                "",
            ]
        )
        lines.extend([f"- `{line}`" for line in json.loads(best["pnextract_lines_json"])])
    lines.extend(["", "## Artifacts", ""])
    for key, value in artifacts.items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "niu2020_berea_pnextract_membrane_scan_v1"))
    parser.add_argument("--segmented-volume", default=str(PROJECT_ROOT / "results" / "niu2020_berea_reproduction" / "segmented_core" / "niu2020_berea_solid255_pore0.tiff"))
    parser.add_argument("--figure5", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"))
    parser.add_argument("--figure8", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "Figure8.xlsx"))
    parser.add_argument("--voxel-size-um", type=float, default=2.8)
    parser.add_argument("--pnextract-exe")
    parser.add_argument("--pnextract-archive", default=str(PROJECT_ROOT.parent / "pnextract" / "bin.7z"))
    parser.add_argument("--pnextract-bin-dir", default=str(PROJECT_ROOT / "results" / "tools" / "pnextract_bin"))
    parser.add_argument(
        "--candidate-set",
        choices=["focused", "broad", "combined"],
        default="focused",
        help="focused reproduces the original hand-tuned scan; broad varies all medial-surface parameters more systematically.",
    )
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_pore, paper_throat = read_figure5(Path(args.figure5))
    pnextract_exe = Path(args.pnextract_exe) if args.pnextract_exe else None
    if pnextract_exe is None:
        if Path(args.pnextract_archive).exists():
            extract_pnextract_bin(Path(args.pnextract_archive), Path(args.pnextract_bin_dir))
        pnextract_exe = find_pnextract_exe(Path(args.pnextract_bin_dir), PROJECT_ROOT.parent / "pnextract", CODE_ROOT / "vendor" / "pnextract")
    if pnextract_exe is None or not pnextract_exe.exists():
        raise FileNotFoundError("Could not locate pnextract.exe")

    candidates = candidates_for_set(args.candidate_set)
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    pd.DataFrame(
        [
            {
                "candidate": c.name,
                "candidate_set": args.candidate_set,
                "min_r_pore": c.min_r_pore,
                "medial_surface_settings": " ".join(map(str, c.medial_surface_settings)) if c.medial_surface_settings else "",
                "pnextract_lines": json.dumps(c.pnextract_lines(), ensure_ascii=False),
            }
            for c in candidates
        ]
    ).to_csv(out_dir / "candidate_plan.csv", index=False)

    rows = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] running {candidate.name}", flush=True)
        row = run_candidate(
            candidate,
            segmented_volume=Path(args.segmented_volume),
            out_dir=out_dir,
            voxel_size_um=args.voxel_size_um,
            pnextract_exe=pnextract_exe,
            python_exe=Path(sys.executable),
            paper_pore=paper_pore,
            paper_throat=paper_throat,
            resume=args.resume,
        )
        row["pnextract_lines_json"] = json.dumps(row.get("pnextract_lines", []), ensure_ascii=False)
        row["candidate_set"] = args.candidate_set
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "scan_metrics_partial.csv", index=False)

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "scan_metrics.csv"
    summary.to_csv(summary_path, index=False)
    figures = plot_scan_summary(summary, paper_pore, paper_throat, out_dir)

    artifacts: dict[str, object] = {"scan_metrics_csv": str(summary_path), "candidate_plan_csv": str(out_dir / "candidate_plan.csv"), **figures}
    ok = summary[summary["status"].eq("ok")].copy()
    if not ok.empty:
        best = ok.sort_values("objective_score").iloc[0]
        best_dir = out_dir / "best_candidate"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_record = best.to_dict()
        (best_dir / "best_candidate_metrics.json").write_text(json.dumps(best_record, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        best_network_dir = Path(str(best["network_dir"]))
        artifacts.update(write_best_spectra(best_network_dir=best_network_dir, out_dir=out_dir, figure8_xlsx=Path(args.figure8)))
        shutil.copy2(best_network_dir / "pores.csv", best_dir / "pores.csv")
        shutil.copy2(best_network_dir / "throats.csv", best_dir / "throats.csv")
        shutil.copy2(best_network_dir / "network_summary.json", best_dir / "network_summary.json")
        artifacts["best_candidate_dir"] = str(best_dir)

    artifacts_path = out_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_md(out_dir / "README.md", summary, artifacts)
    print(json.dumps(artifacts, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
