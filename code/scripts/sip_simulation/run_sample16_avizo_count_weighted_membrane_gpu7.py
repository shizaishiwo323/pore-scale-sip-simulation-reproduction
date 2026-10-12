"""样品16 Avizo 网络：孔喉数量等权的7频点膜极化GPU复算。"""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import gc
import json
import shutil
import sys

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

SOURCE = ROOT / "results/sample16_avizo_ct_geometry_gpu7_20260916"
RUN = ROOT / "results/sample16_avizo_count_weighted_membrane_gpu7_20260916"
FREQUENCIES = np.array([0.1, 0.704961, 6.343811, 44.72136, 315.267933, 2837.038291, 20000.0])
PARAMS = PolarizationParameters(water_conductivity_s_m=0.12, dynamic_pore_size_m=2.45913e-6)


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def prepare():
    for folder in ("configs/source_snapshot", "spectra", "source_data", "provenance", "figures", "simulation_sweeps/membrane"):
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    for name in ("field_ds2.mhd", "field_ds2.raw"):
        shutil.copy2(SOURCE / "configs" / name, RUN / "configs" / name)
    for name in ("pores.csv", "throats.csv"):
        shutil.copy2(SOURCE / "pore_network" / name, RUN / "source_data" / name)
    shutil.copy2(SOURCE / "source_data/experiment_all.csv", RUN / "source_data/experiment_all.csv")

    pores, throats = load_pnextract(RUN / "source_data", PARAMS)
    count_base, metadata = compute_spectra(
        FREQUENCIES, pores, throats, PARAMS, figure5_zdc_ohm=None,
        membrane_weight_mode="count",
    )
    count_base.to_csv(RUN / "spectra/count_weighted_base.csv", index=False)
    component = make_component_spectrum(count_base, "membrane", PARAMS, "paper")
    component["mechanism"] = "membrane"
    component.to_csv(RUN / "spectra/count_weighted_membrane_component.csv", index=False)
    assert metadata["membrane_weight_mode"] == "count"
    assert metadata["membrane_relaxation_length_source"] == "throat_length_m"
    assert len(throats) == 15113 and np.allclose(np.ones(len(throats)) / len(throats), 1 / 15113)

    dump(RUN / "provenance/figure_contract.json", {
        "core_conclusion": "Equal-count throat weighting changes the predicted standalone membrane response relative to volume weighting while preserving the CT-derived relaxation-time distribution.",
        "evidence_chain": {"a": "real conductivity", "b": "positive imaginary conductivity"},
        "archetype": "quantitative grid", "backend": "Python/matplotlib",
        "export": "183 mm PNG at 600 dpi with CSV source data",
        "review_risk": "The experiment is a total SIP response; both simulations are standalone membrane mechanisms on a ds2 field.",
    })
    dump(RUN / "configs/parameters_zh.json", {
        "性质": "仅重跑样品16独立膜极化；孔喉按数量等权，不重跑其他机制。",
        "唯一改动": "膜极化权重由throat_volume_m3改为count；15113条孔喉各自归一化权重为1/15113。",
        "保持不变": "Avizo ChannelLength、Area、Zdc、tau=L^2/(4D)、eta=0.01、D=1.3e-9 m2/s、水电导0.12 S/m。",
        "相定义": "水相=sigma_w+Delta_sigma_membrane；固相严格为0；无孔极化和Maxwell背景。",
        "场求解": "400x333x333 ds2 CT场，3.84 um体素，y方向，GPU active-pore CSR，complex128，rtol=1e-6。",
        "frequencies_hz": FREQUENCIES.tolist(), "throat_count": len(throats),
        "normalized_weight_each": 1 / len(throats), "spectra_metadata": metadata,
    })
    for path in (Path(__file__), Path(solver.__file__), ROOT / "code/scripts/sip_simulation/compute_polarization_spectra.py",
                 ROOT / "code/src/pore_scale_electrical/polarization.py", ROOT / "code/src/pore_scale_electrical/ac3d_pore_gpu.py"):
        shutil.copy2(path, RUN / "configs/source_snapshot" / path.name)


def run():
    info = solver.parse_mhd(RUN / "configs/field_ds2.mhd")
    labels = solver.load_label_volume(info)
    options = dict(
        labels=labels, mechanism="membrane", spectra_path=RUN / "spectra/count_weighted_membrane_component.csv",
        frequencies_hz=FREQUENCIES.tolist(), voxel_size_m=info["voxel_size_m"],
        out_dir=RUN / "simulation_sweeps/membrane", pore_label=0, solid_label=1,
        direction="y", rtol=1e-6, atol=0.0, maxiter=50000, dtype_name="complex128",
        preconditioner="jacobi", fft_reference="mean-face", progress_every=500,
        residual_every=100, iteration_sleep_s=0.0, zero_solid_regularization_s_m=0.0,
        resume=True, solver_backend="gpu-active-pore-csr", jacobi_omega=0.65, sor_omega=1.35,
    )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in options.items() if k != "labels"}
    dump(RUN / "configs/solver.json", config)
    with (RUN / "provenance/gpu.log").open("a", encoding="utf8") as log, redirect_stdout(log), redirect_stderr(log):
        result = solver.run_component_sweep(**options)
    assert len(result) == 7 and result.converged_strict.all() and (result.relative_residual_norm <= 1e-6).all()
    for path in (RUN / "simulation_sweeps/membrane").glob("frequency_*/result.json"):
        dump(path.parent / "config.yml", config)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def plot():
    count = pd.read_csv(RUN / "simulation_sweeps/membrane/sweep_results.csv").sort_values("frequency_hz")
    volume = pd.read_csv(SOURCE / "simulation_sweeps/membrane/sweep_results.csv").sort_values("frequency_hz")
    experiment = pd.read_csv(RUN / "source_data/experiment_all.csv")
    assert len(count) == len(volume) == 7 and count.converged_strict.all()
    comparison = pd.concat([count.assign(weighting="count"), volume.assign(weighting="volume")], ignore_index=True)
    comparison.to_csv(RUN / "source_data/count_vs_volume_membrane_gpu7.csv", index=False)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 0.8, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for index, part in enumerate(("real", "imag")):
        ax = axes[index]
        exp = experiment[f"experiment_sigma_{part}_s_m"]
        keep = exp > 0
        ax.plot(experiment.frequency_hz[keep], exp[keep], "o", ms=2.5, color="#333333", alpha=0.50, label="Experiment (total)")
        ax.plot(volume.frequency_hz, volume[f"effective_sigma_{part}_s_m"], "s--", ms=3.2, lw=1.2,
                color="#E28E2C", mfc="white", label="Volume weighted")
        ax.plot(count.frequency_hz, count[f"effective_sigma_{part}_s_m"], "o-", ms=3.5, lw=1.6,
                color="#2F7D71", label="Count weighted")
        ax.set(xscale="log", yscale="log", xlim=(0.08, 25000), xlabel="Frequency (Hz)",
               ylabel=[r"$\sigma'$ (S m$^{-1}$)", r"$\sigma''$ (S m$^{-1}$)"][index])
        ax.grid(which="major", color="#E5E5E5", lw=0.5)
        ax.text(-0.16, 1.04, "ab"[index], transform=ax.transAxes, fontweight="bold", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=3, fontsize=6.7)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.82, wspace=0.34)
    fig.text(0.52, 0.07, "Standalone membrane mechanism; Avizo geometry; y-direction ds2 field; seven GPU points.", ha="center")
    figure = RUN / "figures/sample16_membrane_count_vs_volume_weighting_gpu7.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    plt.close(fig)

    ratio_real = count.effective_sigma_real_s_m.to_numpy() / volume.effective_sigma_real_s_m.to_numpy()
    ratio_imag = count.effective_sigma_imag_s_m.to_numpy() / volume.effective_sigma_imag_s_m.to_numpy()
    verification = {
        "gpu_points": 7, "all_strictly_converged": True,
        "max_true_relative_residual": float(count.relative_residual_norm.max()),
        "weighting": "equal count, normalized weight 1/15113 per throat",
        "unchanged_relaxation": "tau=L^2/(4D) from Avizo ChannelLength",
        "count_to_volume_ratio": {"real": ratio_real.tolist(), "imag": ratio_imag.tolist()},
        "figure_png": str(figure), "png_dpi": 600,
    }
    dump(RUN / "provenance/verification.json", verification)
    (RUN / "README.md").write_text(
        "# 样品16：数量加权膜极化7频点复算\n\n"
        "本轮只把15113条Avizo孔喉的膜权重从体积归一化改为数量等权，每条喉权重1/15113。"
        "孔喉长度、面积、Zdc和由ChannelLength得到的弛豫时间均未改变。\n\n"
        f"7个GPU点全部严格收敛，最大真实相对残差{verification['max_true_relative_residual']:.3e}。"
        "图中实验为全部机制总响应，数量/体积加权曲线均为独立膜机制，不能按拟合优度直接比较。\n",
        encoding="utf8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
    run()
    plot()
