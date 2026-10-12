"""数量等权且固定孔喉截面积0.6777 um2的样品16膜极化敏感性复算。"""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import gc, json, shutil, sys

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/src"))
sys.path.insert(0, str(ROOT / "code/scripts/sip_simulation"))
import run_sample89_3d_sip_comparison as solver
from compute_polarization_spectra import compute_spectra, load_pnextract
from make_polarization_component_spectra import make_component_spectrum
from pore_scale_electrical.polarization import PolarizationParameters

SOURCE = ROOT / "results/sample16_avizo_count_weighted_membrane_gpu7_20260916"
RUN = ROOT / "results/sample16_avizo_count_fixed_area_0p6777_membrane_gpu7_20260916"
FREQUENCIES = np.array([0.1, 0.704961, 6.343811, 44.72136, 315.267933, 2837.038291, 20000.0])
PARAMS = PolarizationParameters(water_conductivity_s_m=0.12, dynamic_pore_size_m=2.45913e-6)
AREA_UM2, AREA_M2 = 0.6777, 0.6777e-12


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def prepare():
    for folder in ("configs/source_snapshot", "spectra", "source_data", "provenance", "figures", "simulation_sweeps/membrane"):
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    for name in ("field_ds2.mhd", "field_ds2.raw"):
        shutil.copy2(SOURCE / "configs" / name, RUN / "configs" / name)
    shutil.copy2(SOURCE / "source_data/pores.csv", RUN / "source_data/pores.csv")
    throats = pd.read_csv(SOURCE / "source_data/throats.csv")
    throats["original_membrane_active_area_m2"] = throats["membrane_active_area_m2"]
    throats["membrane_active_area_m2"] = AREA_M2
    throats.to_csv(RUN / "source_data/throats.csv", index=False)
    shutil.copy2(SOURCE / "source_data/experiment_all.csv", RUN / "source_data/experiment_all.csv")

    pores, loaded = load_pnextract(RUN / "source_data", PARAMS)
    assert len(loaded) == 15113
    np.testing.assert_array_equal(loaded.membrane_active_area_m2.to_numpy(), np.full(len(loaded), AREA_M2))
    base, metadata = compute_spectra(FREQUENCIES, pores, loaded, PARAMS, figure5_zdc_ohm=None, membrane_weight_mode="count")
    assert metadata["membrane_weight_mode"] == "count"
    assert metadata["membrane_zdc_geometry_source"] == "explicit_membrane_active_area_m2"
    base.to_csv(RUN / "spectra/count_fixed_area_base.csv", index=False)
    component = make_component_spectrum(base, "membrane", PARAMS, "paper")
    component["mechanism"] = "membrane"
    component.to_csv(RUN / "spectra/count_fixed_area_membrane_component.csv", index=False)
    dump(RUN / "configs/parameters_zh.json", {
        "性质": "独立膜极化敏感性：数量等权，并将所有孔喉电学截面积固定为0.6777 um2。",
        "固定截面积_m2": AREA_M2, "孔喉数量": len(loaded), "每条喉归一化权重": 1 / len(loaded),
        "保持不变": "实际Avizo ChannelLength、tau=L^2/(4D)、eta=0.01、D=1.3e-9 m2/s、水电导0.12 S/m。",
        "Zdc": "Zdc,j=ChannelLength_j/(0.12*0.6777e-12)。",
        "场求解": "400x333x333 ds2 CT场，3.84 um体素，y方向，GPU active-pore CSR，complex128，rtol=1e-6。",
        "frequencies_hz": FREQUENCIES.tolist(), "spectra_metadata": metadata,
    })
    dump(RUN / "provenance/figure_contract.json", {
        "core_conclusion": "Fixing every throat to the minimum cross-sectional area suppresses the count-weighted standalone membrane response.",
        "evidence_chain": {"a": "real conductivity", "b": "positive imaginary conductivity"},
        "archetype": "quantitative grid", "backend": "Python/matplotlib", "export": "183 mm PNG at 600 dpi",
        "review_risk": "Fixed area is a diagnostic assumption; experiment is the total SIP response while simulations are membrane-only and ds2.",
    })
    for path in (Path(__file__), Path(solver.__file__), ROOT / "code/scripts/sip_simulation/compute_polarization_spectra.py",
                 ROOT / "code/src/pore_scale_electrical/polarization.py", ROOT / "code/src/pore_scale_electrical/ac3d_pore_gpu.py"):
        shutil.copy2(path, RUN / "configs/source_snapshot" / path.name)


def run():
    info = solver.parse_mhd(RUN / "configs/field_ds2.mhd")
    labels = solver.load_label_volume(info)
    options = dict(labels=labels, mechanism="membrane", spectra_path=RUN / "spectra/count_fixed_area_membrane_component.csv",
        frequencies_hz=FREQUENCIES.tolist(), voxel_size_m=info["voxel_size_m"], out_dir=RUN / "simulation_sweeps/membrane",
        pore_label=0, solid_label=1, direction="y", rtol=1e-6, atol=0.0, maxiter=50000, dtype_name="complex128",
        preconditioner="jacobi", fft_reference="mean-face", progress_every=500, residual_every=100,
        iteration_sleep_s=0.0, zero_solid_regularization_s_m=0.0, resume=True,
        solver_backend="gpu-active-pore-csr", jacobi_omega=0.65, sor_omega=1.35)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in options.items() if k != "labels"}
    dump(RUN / "configs/solver.json", config)
    with (RUN / "provenance/gpu.log").open("a", encoding="utf8") as log, redirect_stdout(log), redirect_stderr(log):
        result = solver.run_component_sweep(**options)
    assert len(result) == 7 and result.converged_strict.all() and (result.relative_residual_norm <= 1e-6).all()
    for path in (RUN / "simulation_sweeps/membrane").glob("frequency_*/result.json"):
        dump(path.parent / "config.yml", config)
    gc.collect(); cp.get_default_memory_pool().free_all_blocks()


def plot():
    fixed = pd.read_csv(RUN / "simulation_sweeps/membrane/sweep_results.csv").sort_values("frequency_hz")
    actual = pd.read_csv(SOURCE / "simulation_sweeps/membrane/sweep_results.csv").sort_values("frequency_hz")
    experiment = pd.read_csv(RUN / "source_data/experiment_all.csv")
    comparison = pd.concat([actual.assign(area_case="actual_area"), fixed.assign(area_case="fixed_0.6777_um2")], ignore_index=True)
    comparison.to_csv(RUN / "source_data/fixed_vs_actual_area_count_weighted_gpu7.csv", index=False)
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
        "font.size":7, "axes.spines.right":False, "axes.spines.top":False, "axes.linewidth":0.8, "legend.frameon":False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for i, part in enumerate(("real", "imag")):
        ax = axes[i]; exp = experiment[f"experiment_sigma_{part}_s_m"]; keep = exp > 0
        ax.plot(experiment.frequency_hz[keep], exp[keep], "o", ms=2.5, color="#333333", alpha=.5, label="Experiment (total)")
        ax.plot(actual.frequency_hz, actual[f"effective_sigma_{part}_s_m"], "s--", ms=3.2, lw=1.2,
                color="#4C78A8", mfc="white", label="Actual area; count weighted")
        ax.plot(fixed.frequency_hz, fixed[f"effective_sigma_{part}_s_m"], "o-", ms=3.5, lw=1.6,
                color="#E15759", label=r"Fixed area = 0.6777 $\mu$m$^2$")
        ax.set(xscale="log", yscale="log", xlim=(.08,25000), xlabel="Frequency (Hz)",
               ylabel=[r"$\sigma'$ (S m$^{-1}$)",r"$\sigma''$ (S m$^{-1}$)"][i])
        ax.grid(which="major", color="#E5E5E5", lw=.5)
        ax.text(-.16,1.04,"ab"[i],transform=ax.transAxes,fontweight="bold",fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52,1.01), ncol=3, fontsize=6.5)
    fig.subplots_adjust(left=.105,right=.985,bottom=.22,top=.82,wspace=.34)
    fig.text(.52,.07,"Standalone membrane mechanism; count weighted; Avizo lengths; seven GPU points.",ha="center")
    figure = RUN / "figures/sample16_count_weighted_fixed_area_0p6777_gpu7.png"
    fig.savefig(figure,dpi=600,bbox_inches="tight"); plt.close(fig)
    ratio_real = fixed.effective_sigma_real_s_m.to_numpy()/actual.effective_sigma_real_s_m.to_numpy()
    ratio_imag = fixed.effective_sigma_imag_s_m.to_numpy()/actual.effective_sigma_imag_s_m.to_numpy()
    verification = {"gpu_points":7,"all_strictly_converged":True,"max_true_relative_residual":float(fixed.relative_residual_norm.max()),
        "fixed_area_um2":AREA_UM2,"weighting":"equal count","relaxation":"actual Avizo ChannelLength",
        "fixed_to_actual_area_ratio":{"real":ratio_real.tolist(),"imag":ratio_imag.tolist()},"figure_png":str(figure),"png_dpi":600}
    dump(RUN / "provenance/verification.json", verification)
    (RUN / "README.md").write_text("# 数量加权、固定截面积膜极化\n\n所有15113条孔喉的电学截面积固定为0.6777 um2；数量等权，实际Avizo喉长及其弛豫时间不变。该设置是诊断性敏感性假设，不是CT实际截面积。\n\n"+
        f"7个GPU点全部严格收敛，最大真实相对残差{verification['max_true_relative_residual']:.3e}。实验为全部机制总响应，两条模拟均为独立膜机制。\n",encoding="utf8")
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    prepare(); run(); plot()
