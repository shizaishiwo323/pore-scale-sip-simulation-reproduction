"""Sample 16: Bucker-Hordt-informed per-throat eta, seven y-direction GPU solves."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import shutil

import cupy as cp
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import i0e

import run_sample89_3d_sip_comparison as solver
from compute_polarization_spectra import load_pnextract
from make_polarization_component_spectra import make_component_spectrum
from pore_scale_electrical.polarization import (
    PolarizationParameters,
    membrane_polarization_conductance,
    upscale_conductance_to_water_conductivity,
)


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/sample16_gpu_7freq_20260912_2340"
SOURCE = ROOT / "results/sample16_gpu_yz_7freq_20260913/y"
RUN = ROOT / "results/sample16_membrane_mechanistic_y_gpu7_20260914"
FREQUENCIES = np.array([0.1, 0.704961, 6.343811, 44.72136, 315.267933, 2837.038291, 20000.0])
PARAMS = PolarizationParameters(water_conductivity_s_m=0.12, dynamic_pore_size_m=2.45913e-6)

# Bucker & Hordt example electrochemistry; provisional for sample 16 because pH/zeta/f_Q were not measured.
TEMPERATURE_K = 298.15
ZETA_POTENTIAL_V = -0.075
STERN_PARTITION = 0.6
D_CATION_M2_S = 1.334e-9
D_ANION_M2_S = 2.032e-9
K_B = 1.380649e-23
E_CHARGE = 1.602176634e-19
FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
EPSILON_0 = 8.8541878128e-12
C0_MOL_M3 = PARAMS.water_conductivity_s_m * GAS_CONSTANT * TEMPERATURE_K / (
    FARADAY**2 * (D_CATION_M2_S + D_ANION_M2_S)
)
DEBYE_LENGTH_M = np.sqrt(
    PARAMS.water_relative_permittivity * EPSILON_0 * K_B * TEMPERATURE_K
    / (2.0 * C0_MOL_M3 * E_CHARGE * FARADAY)
)


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf8")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def cross_section_concentrations(radius_m: np.ndarray, zeta_v: float = ZETA_POTENTIAL_V) -> tuple[np.ndarray, np.ndarray]:
    """Bucker-Hordt eqs. 15, 18, 19, averaged over each cylindrical section."""
    radius = np.asarray(radius_m, dtype=float)
    nodes, weights = leggauss(64)
    u, weights = (nodes + 1.0) / 2.0, weights / 2.0
    kr = radius[:, None] / DEBYE_LENGTH_M
    potential_ratio = i0e(kr * u) / i0e(kr) * np.exp(kr * (u - 1.0))
    exponent = E_CHARGE * zeta_v * potential_ratio / (K_B * TEMPERATURE_K)
    anion = 2.0 * np.sum(np.exp(exponent) * u * weights, axis=1)
    cation = 2.0 * np.sum(np.exp(-exponent) * u * weights, axis=1)
    return cation, anion


def pair_polarizability(pore_radius_m: np.ndarray, pore_length_m: np.ndarray,
                        throat_radius_m: np.ndarray, throat_length_m: np.ndarray,
                        zeta_v: float = ZETA_POTENTIAL_V) -> dict[str, np.ndarray]:
    """Bucker-Hordt 2013a EDL transfer numbers inserted into 2013b eqs. 17-19."""
    bp1, bn1 = cross_section_concentrations(pore_radius_m, zeta_v)
    bp2, bn2 = cross_section_concentrations(throat_radius_m, zeta_v)
    bp1 = (bp1 - STERN_PARTITION) / (1.0 - STERN_PARTITION)
    bp2 = (bp2 - STERN_PARTITION) / (1.0 - STERN_PARTITION)
    dp1, dn1 = D_CATION_M2_S * bp1, D_ANION_M2_S * bn1
    dp2, dn2 = D_CATION_M2_S * bp2, D_ANION_M2_S * bn2
    tp1, tp2 = dp1 / (dp1 + dn1), dp2 / (dp2 + dn2)
    tn1, tn2 = 1.0 - tp1, 1.0 - tp2
    tau1 = pore_length_m**2 / (8.0 * dp1 * tn1)
    tau2 = throat_length_m**2 / (8.0 * dp2 * tn2)
    passive = pore_length_m / (dp1 + dn1) + throat_length_m / (dp2 + dn2)
    polarization = 8.0 * (tn2 * tp1 - tn1 * tp2) ** 2 / (
        pore_length_m / tau1 + throat_length_m / tau2
    )
    eta = polarization / (passive + polarization)
    return {"eta": eta, "tp_pore": tp1, "tp_throat": tp2, "bp_pore": bp1, "bp_throat": bp2}


def mechanistic_throats(network_dir: Path) -> pd.DataFrame:
    pores = pd.read_csv(network_dir / "pores.csv").set_index("pore_id")
    throats = pd.read_csv(network_dir / "throats.csv")
    throats = throats.loc[(throats.pore1_id >= 1) & (throats.pore2_id >= 1)].copy()
    rt, lt = throats.throat_radius_m.to_numpy(), throats.throat_length_m.to_numpy()
    pair_results = []
    for endpoint in (1, 2):
        lp = throats[f"pore{endpoint}_length_m"].to_numpy()
        rp = throats[f"pore{endpoint}_id"].map(pores.pore_radius_m).to_numpy()
        valid = lp > 0.0
        result = {key: np.full(len(throats), np.nan) for key in ("eta", "tp_pore", "tp_throat", "bp_pore", "bp_throat")}
        calculated = pair_polarizability(rp[valid], lp[valid], rt[valid], lt[valid])
        for key, values in calculated.items():
            result[key][valid] = values
        pair_results.append(result)

    l1, l2 = throats.pore1_length_m.to_numpy(), throats.pore2_length_m.to_numpy()
    weights = np.column_stack([l1, l2])
    eta_pairs = np.column_stack([pair_results[0]["eta"], pair_results[1]["eta"]])
    weights[~np.isfinite(eta_pairs)] = 0.0
    eta = np.divide(np.nansum(eta_pairs * weights, axis=1), weights.sum(axis=1),
                    out=np.zeros(len(throats)), where=weights.sum(axis=1) > 0.0)
    assert np.isfinite(eta).all() and np.all((eta >= 0.0) & (eta < 1.0))
    assert np.max(pair_polarizability(np.array([10e-6]), np.array([10e-6]),
                                      np.array([1e-6]), np.array([10e-6]), zeta_v=0.0)["eta"]) < 1e-24

    out = throats[["throat_id", "pore1_id", "pore2_id", "throat_radius_m", "throat_length_m",
                   "pore1_length_m", "pore2_length_m", "throat_volume_m3"]].copy()
    out["eta_pore1_pair"] = pair_results[0]["eta"]
    out["eta_pore2_pair"] = pair_results[1]["eta"]
    out["membrane_polarizability"] = eta
    out["cation_transfer_pore1"] = pair_results[0]["tp_pore"]
    out["cation_transfer_pore2"] = pair_results[1]["tp_pore"]
    out["cation_transfer_throat"] = np.nanmean(
        np.column_stack([pair_results[0]["tp_throat"], pair_results[1]["tp_throat"]]), axis=1
    )
    return out


def prepare() -> dict:
    for folder in ("configs/source_snapshot", "pore_network", "spectra", "source_data", "provenance", "figures", "simulation_sweeps"):
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    inputs = {}
    copies = [
        (BASE / "configs/field_ds2.mhd", RUN / "configs/field_ds2.mhd"),
        (BASE / "configs/field_ds2.raw", RUN / "configs/field_ds2.raw"),
        (BASE / "pore_network/network_parsed/pores.csv", RUN / "pore_network/pores.csv"),
        (BASE / "pore_network/network_parsed/throats.csv", RUN / "pore_network/throats.csv"),
        (SOURCE / "sample16_membrane_experiment_bulk_conductivity_from_impedance.csv",
         RUN / "source_data/experiment_bulk_conductivity.csv"),
    ]
    for source, target in copies:
        shutil.copy2(source, target)
        assert sha(source) == sha(target)
        inputs[str(target.relative_to(RUN))] = {"source": str(source), "sha256": sha(source)}

    eta_frame = mechanistic_throats(RUN / "pore_network")
    eta_frame.to_csv(RUN / "source_data/throat_mechanistic_polarizability.csv", index=False)
    _, throats = load_pnextract(RUN / "pore_network", PARAMS)
    eta = eta_frame.membrane_polarizability.to_numpy()
    assert len(eta) == len(throats)
    increment = membrane_polarization_conductance(
        FREQUENCIES, throats.length_m.to_numpy(), throats.weight.to_numpy(), throats.zdc_ohm.to_numpy(),
        PARAMS, polarizability=eta,
    )
    delta = upscale_conductance_to_water_conductivity(increment, PARAMS)
    base_spectrum = pd.DataFrame({
        "frequency_hz": FREQUENCIES,
        "omega_rad_s": 2.0 * np.pi * FREQUENCIES,
        "delta_sigma_membrane_real_s_m": delta.real,
        "delta_sigma_membrane_imag_s_m": delta.imag,
    })
    base_spectrum.to_csv(RUN / "spectra/mechanistic_membrane_base.csv", index=False)
    component = make_component_spectrum(base_spectrum, "membrane", PARAMS, "paper")
    component["mechanism"] = "membrane"
    spectra_path = RUN / "spectra/mechanistic_membrane_component.csv"
    component.to_csv(spectra_path, index=False)

    info = solver.parse_mhd(RUN / "configs/field_ds2.mhd")
    options = dict(
        mechanism="membrane", spectra_path=spectra_path, frequencies_hz=FREQUENCIES.tolist(),
        voxel_size_m=info["voxel_size_m"], out_dir=RUN / "simulation_sweeps/membrane",
        pore_label=0, solid_label=1, direction="y", rtol=1e-6, atol=0.0, maxiter=50000,
        dtype_name="complex128", preconditioner="jacobi", fft_reference="mean-face",
        progress_every=500, residual_every=500, iteration_sleep_s=0.0,
        zero_solid_regularization_s_m=0.0, resume=False, solver_backend="gpu-active-pore-csr",
        jacobi_omega=0.65, sor_omega=1.35,
    )
    serialized = {key: str(value) if isinstance(value, Path) else value for key, value in options.items()}
    dump(RUN / "configs/solver.json", serialized)
    eta_summary = {
        "count": len(eta), "min": float(eta.min()), "median": float(np.median(eta)),
        "mean": float(eta.mean()), "max": float(eta.max()),
        "throat_volume_weighted_mean": float(np.average(eta, weights=eta_frame.throat_volume_m3)),
    }
    dump(RUN / "configs/parameters_zh.json", {
        "性质": "Bucker-Hordt机制约束eta_j的样品16独立膜极化敏感性复算；不是参数实测或独立验证。",
        "原模型保留": "保留Niu/Titov膜阻抗核、原始pnextract喉长/半径/体积权重和几何Zdc；只把统一eta=0.01替换为逐内部孔喉eta_j。",
        "eta_j计算": "每条喉分别与两端相邻孔体构成wide-pore/narrow-throat局部对；用EDL截面平均浓度、迁移数及Bucker-Hordt 2013b式17-19计算两组eta，再按pnextract两端孔段长度加权。零长度端不参与；两端均零时eta=0。",
        "电化学假设": "水电导0.12 S/m通过Nernst-Einstein关系换算NaCl等效浓度；zeta=-75 mV、f_Q=0.6采用Bucker-Hordt文献示例，不是样品16实测。",
        "hybrid_model_boundary": "这是mechanistic eta_j + Niu/Titov kernel混合敏感性模型；并非完整extended Marshall-Madden impedance。",
        "相电导率": "水相=sigma_w+Delta_sigma_membrane；固相严格为0；无孔极化、Maxwell介电背景和固相电导。",
        "网格与求解": "400x333x333 ds2场、3.84 um体素、y方向、周期边界、GPU active-pore CSR CG/Jacobi、complex128、rtol=1e-6。不是full-grid验证。",
        "frequencies_hz": FREQUENCIES.tolist(), "polarization_parameters": asdict(PARAMS),
        "electrochemistry": {
            "temperature_k": TEMPERATURE_K, "zeta_potential_v": ZETA_POTENTIAL_V,
            "stern_partition_fq": STERN_PARTITION, "cation_diffusion_m2_s": D_CATION_M2_S,
            "anion_diffusion_m2_s": D_ANION_M2_S, "equivalent_nacl_concentration_mol_m3": C0_MOL_M3,
            "debye_length_m": DEBYE_LENGTH_M,
        },
        "eta_summary": eta_summary, "inputs": inputs,
        "primary_sources": ["https://doi.org/10.1093/gji/ggt136", "https://doi.org/10.1190/GEO2012-0548.1"],
    })
    dump(RUN / "provenance/figure_contract.json", {
        "core_conclusion": "EDL-constrained eta_j strongly reduces the standalone membrane response relative to fixed eta=0.01 for the resolved sample-16 throat sizes.",
        "archetype": "quantitative grid", "backend": "Python/matplotlib", "output": "183 mm wide PNG, 600 dpi",
        "panels": {"a": "real conductivity comparison", "b": "positive imaginary conductivity comparison"},
        "reviewer_risk": "zeta and Stern partition are literature examples; experiment is total response, while simulations are membrane-only and ds2.",
    })
    snapshots = {}
    for path in (Path(__file__), Path(solver.__file__), ROOT / "code/src/pore_scale_electrical/polarization.py",
                 ROOT / "code/src/pore_scale_electrical/ac3d_pore_gpu.py"):
        target = RUN / "configs/source_snapshot" / path.name
        shutil.copy2(path, target)
        snapshots[str(path.relative_to(ROOT))] = sha(path)
    dump(RUN / "provenance/source_hashes.json", snapshots)
    return options


def run(options: dict) -> None:
    info = solver.parse_mhd(RUN / "configs/field_ds2.mhd")
    labels = solver.load_label_volume(info)
    assert labels.shape == (400, 333, 333) and set(np.unique(labels)) == {0, 1}
    with (RUN / "provenance/gpu.log").open("w", encoding="utf8") as log, redirect_stdout(log), redirect_stderr(log):
        sweep = solver.run_component_sweep(labels=labels, **options)
    assert len(sweep) == 7 and sweep.converged_strict.all() and (sweep.relative_residual_norm <= 1e-6).all()
    serialized = json.loads((RUN / "configs/solver.json").read_text(encoding="utf8"))
    for result in (RUN / "simulation_sweeps/membrane").glob("frequency_*/result.json"):
        dump(result.parent / "config.yml", serialized)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new = pd.read_csv(RUN / "simulation_sweeps/membrane/sweep_results.csv")
    old = pd.read_csv(SOURCE / "simulation_sweeps/membrane/sweep_results.csv")
    experiment = pd.read_csv(RUN / "source_data/experiment_bulk_conductivity.csv")
    assert len(new) == len(old) == len(experiment) == 7 and np.array_equal(new.frequency_hz, FREQUENCIES)
    assert new.converged_strict.all() and (new.relative_residual_norm <= 1e-6).all()
    assert (new.solid_sigma_real_s_m == 0).all() and (new.solid_sigma_imag_s_m == 0).all()

    eta_frame = pd.read_csv(RUN / "source_data/throat_mechanistic_polarizability.csv")
    _, throats = load_pnextract(RUN / "pore_network", PARAMS)
    eta = eta_frame.membrane_polarizability.to_numpy()
    dense_frequency = np.geomspace(FREQUENCIES.min(), FREQUENCIES.max(), 601)

    def water(frequency: np.ndarray, polarizability: np.ndarray | float) -> np.ndarray:
        increment = membrane_polarization_conductance(
            frequency, throats.length_m.to_numpy(), throats.weight.to_numpy(), throats.zdc_ohm.to_numpy(),
            PARAMS, polarizability=polarizability,
        )
        return PARAMS.water_conductivity_s_m + upscale_conductance_to_water_conductivity(increment, PARAMS)

    new_effective = new.effective_sigma_real_s_m.to_numpy() + 1j * new.effective_sigma_imag_s_m.to_numpy()
    old_effective = old.effective_sigma_real_s_m.to_numpy() + 1j * old.effective_sigma_imag_s_m.to_numpy()
    new_water = water(FREQUENCIES, eta)
    old_water = water(FREQUENCIES, 0.01)
    new_factor = float(np.median((new_effective / new_water).real))
    old_factor = float(np.median((old_effective / old_water).real))
    assert np.max(np.abs(new_factor * new_water - new_effective) / np.abs(new_effective)) < 1e-5
    assert np.max(np.abs(old_factor * old_water - old_effective) / np.abs(old_effective)) < 1e-5
    dense_new, dense_old = new_factor * water(dense_frequency, eta), old_factor * water(dense_frequency, 0.01)

    comparison = pd.concat([
        new.assign(case="mechanistic_eta_j"), old.assign(case="fixed_eta_0.01")
    ], ignore_index=True)
    comparison.to_csv(RUN / "source_data/membrane_gpu7_before_after.csv", index=False)
    pd.DataFrame({
        "frequency_hz": dense_frequency,
        "mechanistic_real_s_m": dense_new.real, "mechanistic_imag_s_m": dense_new.imag,
        "fixed_eta_real_s_m": dense_old.real, "fixed_eta_imag_s_m": dense_old.imag,
    }).to_csv(RUN / "source_data/membrane_dense_before_after.csv", index=False)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    colors = {"experiment": "#222222", "fixed": "#D18B37", "mechanistic": "#2F7D71"}
    for index, part in enumerate(("real", "imag")):
        ax = axes[index]
        exp_values = experiment[f"experiment_sigma_{part}_s_m"].to_numpy()
        keep = exp_values > 0.0
        ax.plot(FREQUENCIES[keep], exp_values[keep], "o", color=colors["experiment"], ms=3.5,
                label="Experiment (total response)")
        ax.plot(dense_frequency, getattr(dense_old, part), "--", color=colors["fixed"], lw=1.4,
                label=r"Fixed $\eta=0.01$")
        ax.plot(FREQUENCIES, getattr(old_effective, part), "s", color=colors["fixed"], mfc="white", ms=3.2)
        ax.plot(dense_frequency, getattr(dense_new, part), "-", color=colors["mechanistic"], lw=1.6,
                label=r"Mechanistic $\eta_j$")
        ax.plot(FREQUENCIES, getattr(new_effective, part), "D", color=colors["mechanistic"], mfc="white", ms=3.2)
        ax.set(xscale="log", yscale="log", xlim=(0.08, 25000), xlabel="Frequency (Hz)",
               ylabel=[r"$\sigma'$ (S m$^{-1}$)", r"$\sigma''$ (S m$^{-1}$)"][index])
        ax.text(-0.16, 1.04, "ab"[index], transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.grid(which="major", color="#E5E5E5", linewidth=0.5)
    axes[0].set_ylim(1e-4, 1e-2)
    axes[1].set_ylim(max(1e-12, min(dense_new.imag.min(), new_effective.imag.min()) * 0.5), 3e-4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=3)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.82, wspace=0.34)
    fig.text(0.52, 0.07, "Standalone membrane mechanism; y-direction ds2 field; seven GPU points.", ha="center")
    figure_path = RUN / "figures/sample16_membrane_mechanistic_before_after.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    positive = experiment.experiment_sigma_imag_s_m.to_numpy() > 0.0
    old_rmse = float(np.sqrt(np.mean((np.log10(old_effective.imag[positive]) - np.log10(experiment.experiment_sigma_imag_s_m[positive])) ** 2)))
    new_rmse = float(np.sqrt(np.mean((np.log10(new_effective.imag[positive]) - np.log10(experiment.experiment_sigma_imag_s_m[positive])) ** 2)))
    verification = {
        "new_gpu_points": 7, "all_strictly_converged": True,
        "max_relative_residual": float(new.relative_residual_norm.max()),
        "solid_exactly_zero": True, "per_throat_eta_count": len(eta),
        "eta_volume_weighted_mean": float(np.average(eta, weights=eta_frame.throat_volume_m3)),
        "eta_median": float(np.median(eta)), "eta_range": [float(eta.min()), float(eta.max())],
        "new_to_old_imag_ratio": (new_effective.imag / old_effective.imag).tolist(),
        "positive_experiment_imag_log10_rmse": {"fixed_eta_0.01": old_rmse, "mechanistic_eta_j": new_rmse},
        "comparison_boundary": "Experiment is total SIP response; both simulations are standalone membrane-only, so RMSE is descriptive, not a membrane fit score.",
        "model_boundary": "Mechanistic eta_j is inserted into the Niu/Titov kernel; full extended Marshall-Madden impedance is not solved.",
        "figure_png": str(figure_path), "png_dpi": 600,
    }
    dump(RUN / "provenance/verification.json", verification)
    (RUN / "README.md").write_text(
        "# 样品16：机制约束膜极化率复算\n\n"
        f"对10679条内部孔喉计算逐喉eta_j；体积加权均值为{verification['eta_volume_weighted_mean']:.6e}，"
        f"中位数为{verification['eta_median']:.6e}，范围{eta.min():.6e}至{eta.max():.6e}。"
        "固定eta=0.01因此不再使用。\n\n"
        "eta_j由相邻孔体/孔喉半径和长度、Debye长度、截面Boltzmann离子浓度、Stern修正、迁移数与"
        "Bucker-Hordt极化率公式得到，再放入原Niu/Titov膜阻抗核。原始几何、Zdc和体积权重未改。"
        "这属于混合mechanistic improvement，并非完整extended Marshall-Madden求解。\n\n"
        f"等效NaCl浓度为{C0_MOL_M3:.6f} mol/m3，Debye长度为{DEBYE_LENGTH_M*1e9:.4f} nm。"
        "zeta=-75 mV和f_Q=0.6采用文献示例；样品16缺少对应pH、盐组成、zeta和Stern分配实测，"
        "所以本结果是机制敏感性结果，不是唯一物理定值。\n\n"
        f"7个y方向GPU频点全部严格收敛，最大真实相对残差{verification['max_relative_residual']:.3e}。"
        "场网格为400x333x333、3.84 um体素的ds2诊断场，非full-grid。水相只含0.12 S/m直流背景与膜增量，"
        "固相严格绝缘；无孔极化与Maxwell背景。\n\n"
        f"相对原固定eta结果，7频点虚部比值为：{', '.join(f'{value:.3e}' for value in verification['new_to_old_imag_ratio'])}。"
        f"对实验正虚部的描述性log10 RMSE由{old_rmse:.3f}变为{new_rmse:.3f}；"
        "实验是全部机制总响应，不能把这个数当成独立膜机制拟合优度。\n",
        encoding="utf8",
    )
    print(json.dumps(verification, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    options = prepare()
    run(options)
    plot()
