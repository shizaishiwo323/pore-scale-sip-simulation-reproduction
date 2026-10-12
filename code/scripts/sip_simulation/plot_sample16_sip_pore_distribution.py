"""Infer a conditional pore-throat distribution from Sample 16 SIP data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/scripts/sip_simulation"))
from run_sample89_3d_sip_comparison import (  # noqa: E402
    add_bulk_conductivity_from_psip_impedance,
    parse_psip_csv,
)


def fit_debye_distribution(
    frequency_hz: np.ndarray,
    sigma_real_s_m: np.ndarray,
    sigma_imag_s_m: np.ndarray,
    relaxation_frequency_hz: np.ndarray,
    regularization: float,
) -> dict[str, np.ndarray | float | bool]:
    """Fit non-negative Debye amplitudes with second-difference smoothing."""
    f, real, imag, f0 = map(
        lambda values: np.asarray(values, dtype=float),
        (frequency_hz, sigma_real_s_m, sigma_imag_s_m, relaxation_frequency_hz),
    )
    if not (f.ndim == real.ndim == imag.ndim == f0.ndim == 1):
        raise ValueError("Inputs must be one-dimensional.")
    if len(f) < 6 or not (len(f) == len(real) == len(imag)):
        raise ValueError("At least six matching observations are required.")
    if regularization < 0 or np.any(f <= 0) or np.any(f0 <= 0) or np.any(imag <= 0):
        raise ValueError("Frequencies and imaginary conductivity must be positive.")

    x = f[:, None] / f0[None, :]
    basis = 1j * x / (1.0 + 1j * x)
    real_scale = max(float(np.ptp(real)), 1e-12)
    imag_scale = max(float(np.max(imag)), 1e-12)
    amplitude_scale = max(real_scale, imag_scale)
    dc_scale = float(np.median(real))
    design = np.block(
        [
            [np.full((len(f), 1), dc_scale / real_scale), basis.real * amplitude_scale / real_scale],
            [np.zeros((len(f), 1)), basis.imag * amplitude_scale / imag_scale],
        ]
    )
    target = np.r_[real / real_scale, imag / imag_scale]
    second_difference = np.diff(np.eye(len(f0)), n=2, axis=0)
    if regularization:
        design = np.vstack(
            [
                design,
                np.c_[np.zeros((len(second_difference), 1)), np.sqrt(regularization) * second_difference],
            ]
        )
        target = np.r_[target, np.zeros(len(second_difference))]

    result = lsq_linear(design, target, bounds=(0.0, np.inf), tol=1e-12, max_iter=5000)
    amplitudes = result.x[1:] * amplitude_scale
    if not result.success or amplitudes.sum() <= 0:
        raise RuntimeError(f"Debye inversion failed: {result.message}")
    prediction = result.x[0] * dc_scale + basis @ amplitudes
    return {
        "success": result.success,
        "sigma_dc_s_m": result.x[0] * dc_scale,
        "amplitudes_s_m": amplitudes,
        "prediction": prediction,
        "real_rmse_s_m": float(np.sqrt(np.mean((prediction.real - real) ** 2))),
        "imag_rmse_s_m": float(np.sqrt(np.mean((prediction.imag - imag) ** 2))),
    }


def log_edges(centers: np.ndarray) -> np.ndarray:
    log_centers = np.log(np.asarray(centers, dtype=float))
    return np.exp(
        np.r_[
            1.5 * log_centers[0] - 0.5 * log_centers[1],
            0.5 * (log_centers[:-1] + log_centers[1:]),
            1.5 * log_centers[-1] - 0.5 * log_centers[-2],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/SIP/LKC-16.csv",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(r"E:\Users\13404\Downloads\NMR-SIP\Enhanced pore space analysis by use of μ-CT, MIP, NMR, and SIP.pdf"),
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results/sample16_sip_pore_distribution_20260915")
    parser.add_argument("--frequency-min-hz", type=float, default=0.1)
    parser.add_argument("--frequency-max-hz", type=float, default=100.0)
    parser.add_argument("--diffusion-coefficient-m2-s", type=float, default=3.8e-12)
    parser.add_argument("--ct-porosity", type=float, default=0.19105887418950482)
    parser.add_argument("--regularization", type=float, default=0.3)
    parser.add_argument("--bins", type=int, default=61)
    args = parser.parse_args()
    if not (0 < args.frequency_min_hz < args.frequency_max_hz and 0 < args.ct_porosity < 1):
        raise ValueError("Frequency limits and CT porosity are invalid.")
    if args.diffusion_coefficient_m2_s <= 0 or args.bins < 10:
        raise ValueError("Positive diffusion and at least 10 bins are required.")

    experiment = add_bulk_conductivity_from_psip_impedance(
        parse_psip_csv(args.input),
        current_resistor_ohm=10000.0,
        sample_length_cm=3.6322,
        sample_diameter_cm=2.52476,
    )
    keep = experiment["frequency_hz"].between(args.frequency_min_hz, args.frequency_max_hz)
    keep &= experiment["experiment_sigma_imag_s_m"] > 0
    selected = experiment.loc[keep].copy()
    f = selected["frequency_hz"].to_numpy()
    real = selected["experiment_sigma_real_s_m"].to_numpy()
    imag = selected["experiment_sigma_imag_s_m"].to_numpy()
    relaxation_frequency = np.geomspace(args.frequency_min_hz / 10, args.frequency_max_hz * 10, args.bins)
    fit = fit_debye_distribution(f, real, imag, relaxation_frequency, args.regularization)
    amplitude = np.asarray(fit["amplitudes_s_m"])
    mass = amplitude / amplitude.sum()
    radius_um = np.sqrt(args.diffusion_coefficient_m2_s / (np.pi * relaxation_frequency)) * 1e6
    order = np.argsort(radius_um)
    radius_um, relaxation_frequency, amplitude, mass = (
        values[order] for values in (radius_um, relaxation_frequency, amplitude, mass)
    )
    cumulative_mass = np.cumsum(mass)
    equivalent_porosity_percent = 100 * args.ct_porosity * mass
    cumulative_porosity_percent = 100 * args.ct_porosity * cumulative_mass
    supported_radius_um = np.sort(
        np.sqrt(args.diffusion_coefficient_m2_s / (np.pi * np.array([f.min(), f.max()]))) * 1e6
    )
    peak_index = int(np.argmax(mass))
    peak_is_supported = bool(supported_radius_um[0] <= radius_um[peak_index] <= supported_radius_um[1])

    out = args.out.resolve()
    for folder in (out, out / "source_data", out / "figures", out / "provenance"):
        folder.mkdir(parents=True, exist_ok=True)
    experiment.to_csv(out / "source_data/experiment_conductivity_all.csv", index=False)
    selected.to_csv(out / "source_data/fit_window_observations.csv", index=False)
    distribution = pd.DataFrame(
        {
            "pore_throat_radius_um": radius_um,
            "pore_throat_diameter_um": 2 * radius_um,
            "relaxation_frequency_hz": relaxation_frequency,
            "relaxation_time_s": 1 / (2 * np.pi * relaxation_frequency),
            "debye_amplitude_s_m": amplitude,
            "normalized_pore_volume_fraction_per_log_bin": mass,
            "conditional_bulk_porosity_contribution_percent": equivalent_porosity_percent,
            "conditional_cumulative_bulk_porosity_percent": cumulative_porosity_percent,
        }
    )
    distribution.to_csv(out / "source_data/sip_equivalent_pore_throat_distribution.csv", index=False)
    prediction = np.asarray(fit["prediction"])
    pd.DataFrame(
        {
            "frequency_hz": f,
            "measured_real_s_m": real,
            "fitted_real_s_m": prediction.real,
            "measured_imag_s_m": imag,
            "fitted_imag_s_m": prediction.imag,
        }
    ).to_csv(out / "source_data/debye_fit.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.15), layout="constrained")
    grid = fig.add_gridspec(2, 5)
    ax_dist = fig.add_subplot(grid[:, :3])
    ax_real = fig.add_subplot(grid[0, 3:])
    ax_imag = fig.add_subplot(grid[1, 3:])
    edges = log_edges(radius_um)
    ax_dist.bar(
        radius_um,
        equivalent_porosity_percent,
        width=np.diff(edges) * 0.82,
        color="#4C78A8",
        edgecolor="#2F5D7E",
        linewidth=0.35,
        alpha=0.72,
        label="Differential contribution",
    )
    ax_dist.axvspan(*supported_radius_um, color="#72B7B2", alpha=0.13, label="SIP-supported band")
    ax_dist.axvline(radius_um[peak_index], color="#B4552D", linewidth=1.0, linestyle="--")
    ax_dist.set(
        xscale="log",
        xlabel="Equivalent pore-throat radius (µm)",
        ylabel="Conditional porosity contribution\n(% of bulk volume per log bin)",
        ylim=(0, max(equivalent_porosity_percent) * 1.18),
    )
    ax_cum = ax_dist.twinx()
    ax_cum.spines["right"].set_visible(True)
    ax_cum.plot(radius_um, cumulative_porosity_percent, color="#E07B39", linewidth=1.7)
    ax_cum.set_ylabel("Conditional cumulative porosity (%)", color="#B4552D")
    ax_cum.tick_params(axis="y", colors="#B4552D")
    ax_cum.set_ylim(0, 100 * args.ct_porosity * 1.06)
    ax_dist.legend(loc="upper left", fontsize=6.5)
    ax_dist.text(0.02, 0.97, "a", transform=ax_dist.transAxes, va="top", fontweight="bold", fontsize=8)

    for axis, measured, fitted, label, panel in (
        (ax_real, real, prediction.real, "Real conductivity (S m$^{-1}$)", "b"),
        (ax_imag, imag, prediction.imag, "Imaginary conductivity (S m$^{-1}$)", "c"),
    ):
        axis.semilogx(f, measured, "o", ms=3.3, mfc="white", mec="#222222", mew=0.7, label="Measured")
        axis.semilogx(f, fitted, color="#007C83", linewidth=1.5, label="Debye fit")
        axis.set(xlabel="Frequency (Hz)", ylabel=label)
        axis.text(0.03, 0.92, panel, transform=axis.transAxes, va="top", fontweight="bold", fontsize=8)
    ax_real.legend(loc="upper left", fontsize=6.5)
    fig.suptitle("Sample 16: SIP-derived equivalent pore-throat distribution", fontsize=9, fontweight="bold")
    figure = out / "figures/sample16_sip_equivalent_pore_throat_porosity_distribution.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    plt.close(fig)

    diameter_um = 2 * radius_um
    diameter_edges = log_edges(diameter_um)
    supported_diameter_um = 2 * supported_radius_um
    fig_hist, ax_hist = plt.subplots(figsize=(6.8, 4.5), layout="constrained")
    ax_hist.bar(
        diameter_um,
        100 * mass,
        width=np.diff(diameter_edges) * 0.92,
        color="#8C8C8C",
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax_hist.axvspan(*supported_diameter_um, color="#72B7B2", alpha=0.12, zorder=0)
    ax_hist.set(
        xscale="log",
        xlim=(diameter_edges[0], diameter_edges[-1]),
        ylim=(0, 1.08 * max(100 * mass)),
        xlabel="Equivalent pore-throat diameter (µm)",
        ylabel="Frequency fraction (%)",
        title="SIP-derived equivalent pore-throat diameter distribution",
    )
    ax_hist.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax_hist.set_axisbelow(True)
    ax_hist.text(
        0.98,
        0.96,
        "Measured-frequency support",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        color="#477F7B",
        fontsize=6.5,
    )
    for spine in ax_hist.spines.values():
        spine.set_visible(True)
    ax_hist.tick_params(which="both", direction="in", top=True, right=True)
    ax_hist.text(-0.10, 1.04, "a", transform=ax_hist.transAxes, fontweight="bold", fontsize=9)
    histogram_figure = out / "figures/sample16_sip_equivalent_pore_throat_diameter_frequency_histogram.png"
    fig_hist.savefig(histogram_figure, dpi=600, bbox_inches="tight")
    plt.close(fig_hist)

    manifest = {
        "status": "conditional_sip_equivalent_pore_throat_distribution",
        "method": "nonnegative Debye decomposition with second-difference Tikhonov smoothing",
        "mapping": "r = sqrt(2*tau*D_plus) = sqrt(D_plus/(pi*f_relaxation))",
        "paper": {
            "title": "Enhanced pore space analysis by use of µ-CT, MIP, NMR, and SIP",
            "doi": "10.5194/se-9-1225-2018",
            "sha256": hashlib.sha256(args.paper.read_bytes()).hexdigest(),
            "diffusion_coefficient_m2_s": args.diffusion_coefficient_m2_s,
        },
        "input": str(args.input.relative_to(ROOT)),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "frequency_window_hz": [args.frequency_min_hz, args.frequency_max_hz],
        "used_frequency_range_hz": [float(f.min()), float(f.max())],
        "used_points": len(f),
        "excluded_points": int(len(experiment) - len(selected)),
        "supported_radius_range_um": supported_radius_um.tolist(),
        "grid_radius_range_um": [float(radius_um.min()), float(radius_um.max())],
        "regularization": args.regularization,
        "ct_porosity": args.ct_porosity,
        "porosity_scaling": "All normalized SIP chargeability mass is conditionally scaled to CT total porosity; not MIP-anchored.",
        "peak_radius_um": float(radius_um[peak_index]),
        "peak_diameter_um": float(2 * radius_um[peak_index]),
        "peak_inside_supported_band": peak_is_supported,
        "real_rmse_s_m": fit["real_rmse_s_m"],
        "imag_rmse_s_m": fit["imag_rmse_s_m"],
        "sigma_dc_s_m": fit["sigma_dc_s_m"],
        "figure": str(figure.relative_to(ROOT)),
        "frequency_histogram_figure": str(histogram_figure.relative_to(ROOT)),
        "warnings": [
            "This is a model-dependent effective pore-throat distribution, not direct CT geometry.",
            "The absolute porosity scale is conditional because Sample 16 has no MIP anchoring curve in the supplied data.",
            "A peak outside the supported band is boundary-limited and must not be reported as a resolved modal pore size."
            if not peak_is_supported
            else "The peak lies inside the frequency-supported radius band.",
        ],
    }
    (out / "provenance/run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "README_zh.md").write_text(
        f"""# 样品16 SIP 等效孔喉半径与孔隙率贡献分布

本结果按 Zhang et al. (2018) 的思路，将实验复电导率在 {f.min():g}-{f.max():g} Hz 内做非负 Debye 分解，再用 `r = sqrt(2*tau*D+)` 映射到等效孔喉半径。采用论文给定的 `D+ = {args.diffusion_coefficient_m2_s:.2e} m2/s`。

- 使用频点：{len(f)}；论文建议的几何解释上限为 100 Hz，样品16实际保留到 {f.max():g} Hz。
- 频率直接支持的半径范围：{supported_radius_um[0]:.3f}-{supported_radius_um[1]:.3f} µm。
- 最大分布分量：半径 {radius_um[peak_index]:.3f} µm（直径 {2*radius_um[peak_index]:.3f} µm）；是否位于直接支持区间：{peak_is_supported}。
- 拟合 RMSE：实部 {fit['real_rmse_s_m']:.3e} S/m，虚部 {fit['imag_rmse_s_m']:.3e} S/m。

图中柱形的绝对纵轴把归一化电荷率分布乘以 CT 孔隙率 19.11%。由于所给样品16目录没有 MIP 曲线，这不是论文中由 MIP 锚定的绝对孔隙率，只是“全部 CT 孔隙均按 SIP 分布参与极化”的条件性换算。可靠的主结果是 CSV 中的归一化孔体积分数及其累计曲线。另导出的频率分布直方图把同一归一化 Debye 权重乘以 100%，不是 CT 孔对象数量频率；淡蓝背景表示实验频带直接支持的等效直径范围。

该分布表征 SIP 敏感的等效孔喉尺度，不是 CT 直接测得的孔体几何，也不能单独恢复三维连通性。论文来源：Zhang et al. (2018), Solid Earth 9, 1225-1238, doi:10.5194/se-9-1225-2018。
""",
        encoding="utf-8",
    )
    assert np.isclose(distribution["normalized_pore_volume_fraction_per_log_bin"].sum(), 1.0)
    assert np.isclose(distribution["conditional_bulk_porosity_contribution_percent"].sum(), 100 * args.ct_porosity)
    assert np.isclose((100 * mass).sum(), 100.0)
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
