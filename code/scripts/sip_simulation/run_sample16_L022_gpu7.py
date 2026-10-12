"""样品16：L_relax=0.22微米、eta不变，重算膜机制与全部机制。"""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
import csv
import gc
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_sample16_effective_membrane_length_gpu7 as previous


ROOT = previous.ROOT
SOURCE = ROOT / "results/sample16_effective_membrane_length_gpu7_20260913"
L030_RUN = ROOT / "results/正式结果/sample16_membrane_only_L030_gpu7_20260913"
RUN = ROOT / "results/sample16_effective_membrane_L022_gpu7_20260916"
FREQUENCIES = previous.FREQ
L_RELAX = 0.22e-6
ETA = 0.002442810092526288
EPSILON_0 = 8.8541878128e-12
PARAMS = previous.prior.PolarizationParameters(
    **json.loads((SOURCE / "configs/calibration_parameters_zh.json").read_text(encoding="utf-8"))["parameters"]
)
solver, cp = previous.solver, previous.cp


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    assert np.isclose(PARAMS.membrane_polarizability, ETA)
    for folder in ["configs/source_snapshot", "pore_network", "spectra", "source_data",
                   "provenance", "figures", "simulation_sweeps"]:
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    for name in ["field_ds2.mhd", "field_ds2.raw"]:
        shutil.copy2(SOURCE / "configs" / name, RUN / "configs" / name)
    for name in ["experiment_all.csv", "selected_7_frequencies.csv"]:
        shutil.copy2(SOURCE / "source_data" / name, RUN / "source_data" / name)

    source_network = SOURCE / "pore_network/original"
    shutil.copy2(source_network / "pores.csv", RUN / "pore_network/pores.csv")
    with (source_network / "throats.csv").open(newline="", encoding="utf-8") as src, \
            (RUN / "pore_network/throats.csv").open("w", newline="", encoding="utf-8") as dst:
        reader, writer = csv.reader(src), csv.writer(dst)
        header = next(reader)
        length_index = header.index("throat_length_m")
        writer.writerow(header + ["membrane_relaxation_length_m", "membrane_zdc_length_m"])
        for row in reader:
            writer.writerow(row + [repr(L_RELAX), row[length_index]])

    raw = pd.read_csv(source_network / "throats.csv")
    modified = pd.read_csv(RUN / "pore_network/throats.csv")
    pd.testing.assert_frame_equal(modified[raw.columns], raw, check_exact=True)
    _, original = previous.prior.load_pnextract(source_network, PARAMS)
    _, throats = previous.prior.load_pnextract(RUN / "pore_network", PARAMS)
    assert np.all(throats.membrane_relaxation_length_m == L_RELAX)
    assert np.array_equal(throats.zdc_ohm, original.zdc_ohm)
    assert np.array_equal(throats.weight, original.weight)

    _, spectra, _ = solver.make_component_spectra_files(
        output_prefix="L022", frequencies_hz=FREQUENCIES,
        network_dir=RUN / "pore_network", out_dir=RUN / "spectra",
        mode="paper", params=PARAMS, solid_conductivity_s_m=0.0024,
        solid_background_components={"maxwell", "all"},
    )
    reused = {}
    for mechanism in ["pore", "maxwell"]:
        source = SOURCE / "simulation_sweeps" / f"original_{mechanism}"
        target = RUN / "simulation_sweeps" / mechanism
        shutil.copytree(source, target, dirs_exist_ok=True)
        frame = pd.read_csv(target / "sweep_results.csv")
        assert len(frame) == 7 and frame.converged_strict.all()
        reused[mechanism] = {"source": str(source), "sha256": sha(source / "sweep_results.csv")}

    config = {
        "性质": "样品16固定有效弛豫长度敏感性计算；不是CT直接测得的孔喉长度。",
        "effective_relaxation_length_m": L_RELAX,
        "parameters": asdict(PARAMS),
        "长度分工": "tau=L_relax^2/(4D)使用0.22微米；Zdc仍使用pnextract原始几何孔喉长度。",
        "唯一物理参数变化": "相对0.3微米方案仅将L_relax改为0.22微米；eta及其余参数不变。",
        "重新计算": "独立膜机制和全部机制各7个GPU频点。",
        "复用": reused,
        "网格": "400x333x333，体素3.84微米，downsample=2，y方向；诊断网格而非full-grid。",
        "频点_hz": FREQUENCIES.tolist(),
        "spectra": {key: str(path) for key, path in spectra.items()},
    }
    dump(RUN / "configs/parameters_zh.json", config)
    for path in [Path(__file__), Path(solver.__file__), ROOT / "code/src/pore_scale_electrical/polarization.py"]:
        shutil.copy2(path, RUN / "configs/source_snapshot" / path.name)


def run_gpu():
    info = solver.parse_mhd(RUN / "configs/field_ds2.mhd")
    labels = solver.load_label_volume(info)
    assert labels.shape == (400, 333, 333) and set(np.unique(labels)) == {0, 1}
    for mechanism in ["membrane", "all"]:
        is_membrane = mechanism == "membrane"
        options = dict(
            mechanism=mechanism,
            spectra_path=RUN / "spectra" / f"L022_component_spectra_{mechanism}.csv",
            frequencies_hz=FREQUENCIES.tolist(), voxel_size_m=info["voxel_size_m"],
            out_dir=RUN / "simulation_sweeps" / mechanism,
            pore_label=0, solid_label=1, direction="y", rtol=1e-6, atol=0.0,
            maxiter=50000 if is_membrane else 5000, dtype_name="complex128",
            preconditioner="jacobi" if is_membrane else "fft", fft_reference="mean-face",
            progress_every=500 if is_membrane else 100,
            residual_every=500 if is_membrane else 100, iteration_sleep_s=0.0,
            zero_solid_regularization_s_m=0.0, resume=True,
            solver_backend="gpu-active-pore-csr" if is_membrane else "gpu-ac3d",
            jacobi_omega=0.65, sor_omega=1.35,
        )
        serial = {key: str(value) if isinstance(value, Path) else value for key, value in options.items()}
        dump(RUN / "configs" / f"{mechanism}_solver.json", serial)
        print(f"START {mechanism} GPU sweep", flush=True)
        with (RUN / "provenance" / f"{mechanism}_gpu.log").open("a", encoding="utf-8") as log, \
                redirect_stdout(log), redirect_stderr(log):
            sweep = solver.run_component_sweep(labels=labels, **options)
        assert len(sweep) == 7 and sweep.converged_strict.all()
        assert (sweep.relative_residual_norm <= 1e-6).all()
        for result in options["out_dir"].glob("frequency_*/result.json"):
            dump(result.parent / "config.yml", serial)
        gc.collect()
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        print(f"DONE {mechanism} GPU sweep", flush=True)


def plot():
    experiment = pd.read_csv(RUN / "source_data/experiment_all.csv")
    frames = {key: pd.read_csv(RUN / "simulation_sweeps" / key / "sweep_results.csv")
              for key in ["maxwell", "pore", "membrane", "all"]}
    for frame in frames.values():
        assert len(frame) == 7 and np.array_equal(frame.frequency_hz, FREQUENCIES)
        assert frame.converged_strict.all() and (frame.relative_residual_norm <= 1e-6).all()
    assert (frames["membrane"].solid_sigma_real_s_m == 0).all()
    assert (frames["membrane"].solid_sigma_imag_s_m == 0).all()

    combined = pd.concat([frame.assign(component=key) for key, frame in frames.items()], ignore_index=True)
    combined.to_csv(RUN / "source_data/mechanism_gpu7.csv", index=False)
    experiment.to_csv(RUN / "source_data/experiment_all.csv", index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "mathtext.fontset": "stix",
                         "axes.linewidth": 0.8, "axes.spines.right": False,
                         "axes.spines.top": False, "xtick.direction": "in",
                         "ytick.direction": "in", "legend.frameon": False})
    styles = [("maxwell", "Interfacial + dielectric", "#3157D5", "s"),
              ("pore", "Pore polarization", "#148A2A", "D"),
              ("membrane", "Membrane polarization", "#E53935", "o"),
              ("all", "All mechanisms (joint solve)", "#737373", "^")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for index, part in enumerate(["real", "imag"]):
        ax = axes[index]
        values = experiment[f"experiment_sigma_{part}_s_m"]
        keep = values > 0
        ax.plot(experiment.frequency_hz[keep], values[keep], "k.", ms=4,
                label="LKC-16 experiment", zorder=5)
        for key, label, color, marker in styles:
            frame = frames[key]
            ax.plot(frame.frequency_hz, frame[f"effective_sigma_{part}_s_m"],
                    "-" if key == "all" else "--", color=color,
                    marker=marker, mfc="white", ms=4, lw=2 if key == "all" else 1.6,
                    label=label)
        ax.set(xscale="log", yscale="log", xlim=(0.08, 25000),
               xlabel="Frequency (Hz)",
               ylabel=[r"$\sigma'$ (S/m)", r"$\sigma''$ (S/m)"][index])
        ax.text(-0.13, 1.03, ["(a)", "(b)"][index], transform=ax.transAxes, fontsize=15)
    axes[0].set_ylim(1e-4, 1e-2)
    axes[1].set_ylim(5e-11, 5e-4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 0.91), ncol=3)
    fig.suptitle(r"Sample 16 | $L_{\mathrm{relax}}=0.22\ \mu$m | $\eta=0.00244$ | 7 GPU frequencies", y=0.98)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.76, wspace=0.28)
    conductivity = RUN / "figures/SIP_mechanism_comparison_L022.png"
    conductivity_preview = RUN / "figures/SIP_mechanism_comparison_L022_preview.png"
    fig.savefig(conductivity, dpi=600, bbox_inches="tight")
    fig.savefig(conductivity_preview, dpi=150, bbox_inches="tight")
    plt.close(fig)

    combined["relative_permittivity"] = combined.effective_sigma_imag_s_m / (
        2 * np.pi * combined.frequency_hz * EPSILON_0
    )
    experiment["relative_permittivity"] = experiment.experiment_sigma_imag_s_m / (
        2 * np.pi * experiment.frequency_hz * EPSILON_0
    )
    assert np.allclose(combined.relative_permittivity, combined.real_relative_permittivity, rtol=1e-12)
    pd.concat([
        combined[["frequency_hz", "component", "relative_permittivity"]].assign(source="GPU simulation"),
        experiment[["frequency_hz", "relative_permittivity"]].assign(
            component="experiment", source="LKC-16 experiment"),
    ], ignore_index=True).to_csv(RUN / "source_data/effective_permittivity_L022.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    keep = experiment.relative_permittivity > 0
    ax.plot(experiment.frequency_hz[keep], experiment.relative_permittivity[keep],
            "^", color="black", mfc="white", ms=5, ls="none", label="LKC-16 experiment")
    for key, label, color, marker in styles:
        frame = combined[combined.component == key]
        ax.plot(frame.frequency_hz, frame.relative_permittivity,
                "-" if key == "all" else "--", color=color, marker=marker,
                mfc="white", ms=4, lw=2 if key == "all" else 1.6, label=label)
    ax.set(xscale="log", yscale="log", xlim=(0.08, 25000), ylim=(1e-5, 1e7),
           xlabel="Frequency, $f$ (Hz)",
           ylabel=r"Effective relative permittivity, $\varepsilon'_{\mathrm{eff}}/\varepsilon_0$")
    ax.legend(loc="upper right", fontsize=10, handlelength=3.2)
    ax.set_title(r"Sample 16 | $L_{\mathrm{relax}}=0.22\ \mu$m | $\eta=0.00244$ | 7 GPU frequencies", pad=12)
    fig.tight_layout()
    permittivity = RUN / "figures/effective_permittivity_L022.png"
    permittivity_preview = RUN / "figures/effective_permittivity_L022_preview.png"
    fig.savefig(permittivity, dpi=600, bbox_inches="tight")
    fig.savefig(permittivity_preview, dpi=150, bbox_inches="tight")
    plt.close(fig)

    old_all = pd.read_csv(SOURCE / "simulation_sweeps/calibrated_L030_all/sweep_results.csv")
    new_all = frames["all"]
    comparison = pd.DataFrame({
        "frequency_hz": FREQUENCIES,
        "L030_sigma_real_s_m": old_all.effective_sigma_real_s_m,
        "L022_sigma_real_s_m": new_all.effective_sigma_real_s_m,
        "L030_sigma_imag_s_m": old_all.effective_sigma_imag_s_m,
        "L022_sigma_imag_s_m": new_all.effective_sigma_imag_s_m,
    })
    comparison.to_csv(RUN / "source_data/L022_vs_L030_all.csv", index=False)
    selected = pd.read_csv(RUN / "source_data/selected_7_frequencies.csv")
    def log_metrics(frame):
        real = np.sqrt(np.mean(np.log10(
            frame.effective_sigma_real_s_m / selected.experiment_sigma_real_s_m
        ) ** 2))
        positive = selected.experiment_sigma_imag_s_m > 0
        imag = np.sqrt(np.mean(np.log10(
            frame.effective_sigma_imag_s_m[positive] /
            selected.experiment_sigma_imag_s_m[positive]
        ) ** 2))
        return {"real_log10_rmse": float(real), "imag_log10_rmse": float(imag),
                "joint_log10_rmse": float(np.sqrt((real ** 2 + imag ** 2) / 2))}
    verification = {
        "new_gpu_points": 14, "reused_gpu_points": 14,
        "all_new_points_strictly_converged": True,
        "max_new_relative_residual": max(float(frames[key].relative_residual_norm.max())
                                         for key in ["membrane", "all"]),
        "effective_relaxation_length_m": L_RELAX,
        "membrane_polarizability": ETA,
        "nonpositive_experiment_permittivity_omitted": int((~keep).sum()),
        "all_curve": "joint GPU field solution; standalone effective conductivities were not summed",
        "experiment_fit": {"L022": log_metrics(new_all), "L030": log_metrics(old_all)},
        "L022_vs_L030_all_change_percent": {
            "real_by_frequency": (100 * (new_all.effective_sigma_real_s_m /
                                          old_all.effective_sigma_real_s_m - 1)).tolist(),
            "imag_by_frequency": (100 * (new_all.effective_sigma_imag_s_m /
                                          old_all.effective_sigma_imag_s_m - 1)).tolist(),
        },
        "figures": [str(conductivity), str(permittivity)],
    }
    dump(RUN / "provenance/verification.json", verification)
    (RUN / "README.md").write_text(
        "# 样品16：0.22微米有效弛豫长度\n\n"
        "固定L_relax=0.22 μm、eta=0.002442810092526288，重新计算独立膜机制和全部机制各7个GPU频点；"
        "孔极化与界面/介电机制复用同网格已收敛结果。原始几何孔喉长度、面积、体积权重及Zdc保持不变。\n\n"
        "计算采用400×333×333、3.84 μm体素、downsample=2、y方向诊断场，并非full-grid。"
        "总响应来自联合场求解，没有相加独立机制的体有效电导率。\n\n"
        "SIP与介电常数图位于figures，逐频结果位于simulation_sweeps，源数据位于source_data，"
        "收敛及0.30微米对比见provenance/verification.json。\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
    run_gpu()
    plot()
