"""样品16：直接使用Avizo Excel孔喉几何，执行7频点GPU SIP机制求解。"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
import gc
import hashlib
import json
import shutil
import subprocess
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
from pore_scale_electrical.polarization import PolarizationParameters

INPUT = ROOT / "data/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/CT_slices/16-avizo-pore throat.xlsx"
CT = ROOT / "data/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/CT_slices/5-16seged.tiff"
EXPERIMENT = ROOT / "data/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/SIP/LKC-16.csv"
FIELD_SOURCE = ROOT / "results/sample16_gpu_7freq_20260912_2340/configs"
DIGITAL_ROCK_SOURCE = ROOT / "results/正式结果/sample16_gpu_20260912/digital_rock"
RUN = ROOT / "results/sample16_avizo_ct_geometry_gpu7_20260916"
FREQUENCIES = np.array([0.1, 0.704961, 6.343811, 44.72136, 315.267933, 2837.038291, 20000.0])
PARAMS = PolarizationParameters(water_conductivity_s_m=0.12, dynamic_pore_size_m=2.45913e-6)
MECHANISMS = ("maxwell", "pore", "membrane", "all")


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def convert_avizo_excel() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_pores = pd.read_excel(INPUT, sheet_name="Pores")
    raw_throats = pd.read_excel(INPUT, sheet_name="Throats")
    assert not raw_pores["Pore ID"].duplicated().any()
    assert not raw_throats["Throat ID"].duplicated().any()
    assert raw_throats["Pore ID #1"].isin(raw_pores["Pore ID"]).all()
    assert raw_throats["Pore ID #2"].isin(raw_pores["Pore ID"]).all()
    assert (raw_pores[["Volume", "EqRadius"]] > 0).all().all()
    assert (raw_throats[["Area", "EqRadius", "ChannelLength"]] > 0).all().all()
    np.testing.assert_allclose(raw_pores["EqRadius"], (3 * raw_pores["Volume"] / (4 * np.pi)) ** (1 / 3), rtol=3e-7)
    np.testing.assert_allclose(raw_throats["Area"], np.pi * raw_throats["EqRadius"] ** 2, rtol=2e-7)

    # Avizo IDs are zero-based; +1 avoids collision with the solver's boundary-pore convention.
    pores = pd.DataFrame({
        "pore_id": raw_pores["Pore ID"].astype(np.int64) + 1,
        "pore_center_x_m": raw_pores["X Coord"] * 1e-6,
        "pore_center_y_m": raw_pores["Y Coord"] * 1e-6,
        "pore_center_z_m": raw_pores["Z Coord"] * 1e-6,
        "pore_radius_m": raw_pores["EqRadius"] * 1e-6,
        "pore_volume_m3": raw_pores["Volume"] * 1e-18,
        "pore_surface_area_m2": raw_pores["Area"] * 1e-12,
        "coordination_number": raw_pores["Coordination Number"].astype(np.int64),
        "avizo_label_id": raw_pores["LabelID"].astype(np.int64),
    })
    area_m2 = raw_throats["Area"].to_numpy(dtype=float) * 1e-12
    length_m = raw_throats["ChannelLength"].to_numpy(dtype=float) * 1e-6
    throats = pd.DataFrame({
        "throat_id": raw_throats["Throat ID"].astype(np.int64) + 1,
        "pore1_id": raw_throats["Pore ID #1"].astype(np.int64) + 1,
        "pore2_id": raw_throats["Pore ID #2"].astype(np.int64) + 1,
        "throat_radius_m": raw_throats["EqRadius"] * 1e-6,
        "throat_length_m": length_m,
        "throat_volume_m3": area_m2 * length_m,
        "throat_shape_factor": np.full(len(raw_throats), 1 / (4 * np.pi)),
        "membrane_active_area_m2": area_m2,
    })
    assert "membrane_relaxation_length_m" not in throats
    expected_coordination = pd.concat([throats.pore1_id, throats.pore2_id]).value_counts()
    actual_coordination = pores.set_index("pore_id").coordination_number
    pd.testing.assert_series_equal(actual_coordination, expected_coordination.reindex(actual_coordination.index, fill_value=0), check_names=False)
    return pores, throats


def write_distribution(pores: pd.DataFrame, throats: pd.DataFrame) -> None:
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, name, values, weights, xlabel in (
        (axes[0], "pore_radius", pores.pore_radius_m * 1e6, pores.pore_volume_m3, r"Pore radius ($\mu$m)"),
        (axes[1], "throat_length", throats.throat_length_m * 1e6, throats.throat_volume_m3, r"Throat length ($\mu$m)"),
    ):
        edges = np.geomspace(values.min(), values.max(), 33)
        count, _ = np.histogram(values, bins=edges)
        volume, _ = np.histogram(values, bins=edges, weights=weights)
        count_fraction = count / count.sum()
        volume_fraction = volume / volume.sum()
        centers = np.sqrt(edges[:-1] * edges[1:])
        width = np.diff(edges)
        ax.bar(edges[:-1], count_fraction, width=width, align="edge", color="#4C78A8", alpha=0.42, label="Frequency fraction")
        ax.step(centers, volume_fraction, where="mid", color="#E45756", lw=1.5, label="Volume fraction")
        ax.set(xscale="log", xlabel=xlabel, ylabel="Fraction per bin")
        ax.grid(axis="y", color="#E5E5E5", lw=0.5)
        for i in range(len(centers)):
            records.append({"quantity": name, "left_um": edges[i], "right_um": edges[i + 1],
                            "center_um": centers[i], "count": int(count[i]),
                            "frequency_fraction": count_fraction[i], "volume_fraction": volume_fraction[i]})
    axes[0].legend(fontsize=6.5)
    for label, ax in zip("ab", axes):
        ax.text(-0.17, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.95, wspace=0.32)
    fig.savefig(RUN / "pore_network/avizo_pore_throat_distributions.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(records).to_csv(RUN / "pore_network/avizo_pore_throat_distribution_source_data.csv", index=False)


def prepare() -> None:
    for folder in ("configs/source_snapshot", "digital_rock", "pore_network", "spectra", "source_data", "provenance", "figures", "simulation_sweeps"):
        (RUN / folder).mkdir(parents=True, exist_ok=True)
    pores, throats = convert_avizo_excel()
    pores.to_csv(RUN / "pore_network/pores.csv", index=False)
    throats.to_csv(RUN / "pore_network/throats.csv", index=False)
    write_distribution(pores, throats)
    shutil.copy2(INPUT, RUN / "source_data/16-avizo-pore throat.xlsx")
    for name in ("field_ds2.mhd", "field_ds2.raw"):
        shutil.copy2(FIELD_SOURCE / name, RUN / "configs" / name)
    for source_name, target_name in (("sample16_fiji3d.html", "sample16_fiji3d.html"), ("sample16_preview.png", "sample16_preview.png"), ("metadata.json", "metadata.json")):
        shutil.copy2(DIGITAL_ROCK_SOURCE / source_name, RUN / "digital_rock" / target_name)

    experiment = solver.add_bulk_conductivity_from_psip_impedance(
        solver.parse_psip_csv(EXPERIMENT), current_resistor_ohm=10000,
        sample_length_cm=3.6322, sample_diameter_cm=2.52476,
    )
    experiment.to_csv(RUN / "source_data/experiment_all.csv", index=False)
    selected = experiment.set_index("frequency_hz").loc[FREQUENCIES].reset_index()
    selected.to_csv(RUN / "source_data/selected_7_frequencies.csv", index=False)

    _, component_paths, spectra_metadata = solver.make_component_spectra_files(
        output_prefix="sample16_avizo", frequencies_hz=FREQUENCIES,
        network_dir=RUN / "pore_network", out_dir=RUN / "spectra", mode="paper",
        params=PARAMS, solid_conductivity_s_m=0.0024,
        solid_background_components={"maxwell", "all"},
    )
    base_metadata = spectra_metadata["base_metadata"]
    assert base_metadata["membrane_relaxation_length_source"] == "throat_length_m"
    assert base_metadata["membrane_zdc_length_source"] == "throat_length_m"
    dump(RUN / "provenance/figure_contract.json", {
        "core_conclusion": "Avizo/CT-derived pore and throat geometry yields distinct seven-frequency interfacial, pore, membrane and joint SIP responses without a fixed relaxation length.",
        "evidence_chain": {"a": "real conductivity against experiment", "b": "positive imaginary conductivity against experiment", "distribution": "direct input geometry audit"},
        "archetype": "quantitative grid", "backend": "Python/matplotlib",
        "export": "183 mm PNG at 600 dpi with CSV source data",
        "review_risk": "The electrical field is ds2, Avizo ChannelLength is used as the relaxation length, and eta=0.01 remains a model parameter rather than a Sample16 measurement.",
    })
    dump(RUN / "configs/parameters_zh.json", {
        "性质": "样品16 Avizo/CT几何驱动的7频点GPU SIP机制计算；不运行pnextract，不使用固定弛豫长度。",
        "几何输入": str(INPUT), "孔数": len(pores), "孔喉数": len(throats),
        "单位判定": "坐标、EqRadius和ChannelLength为um；Area为um2；Volume为um3。等效半径关系与CT物理尺寸均已核验。",
        "膜弛豫": "每条孔喉使用Avizo ChannelLength；tau_m,j=L_j^2/(4D)。未创建membrane_relaxation_length_m固定列。",
        "膜直流阻抗": "Zdc,j=ChannelLength_j/(sigma_w*Area_j)，使用Excel实际Area；不缩放几何。",
        "电场": "复用同一原始CT生成的400x333x333 ds2标签场，3.84 um体素，y方向；不是800x666x666 full-grid。",
        "机制": "interfacial(maxwell)、pore、membrane、all分别求解；pore/membrane固相为0，maxwell/all固相为0.0024 S/m并含介电背景。",
        "frequencies_hz": FREQUENCIES.tolist(), "polarization_parameters": asdict(PARAMS),
        "solver": "complex128; rtol=1e-6; maxiter=50000; y direction; FFT for maxwell/all; active-pore CSR for pore/membrane",
        "inputs_sha256": {"avizo_excel": sha256(INPUT), "segmented_ct": sha256(CT), "field_ds2_raw": sha256(RUN / "configs/field_ds2.raw")},
        "component_spectra": {k: str(v) for k, v in component_paths.items()},
    })
    for path in (Path(__file__), Path(solver.__file__), ROOT / "code/scripts/sip_simulation/compute_polarization_spectra.py",
                 ROOT / "code/src/pore_scale_electrical/polarization.py", ROOT / "code/src/pore_scale_electrical/ac3d_gpu.py",
                 ROOT / "code/src/pore_scale_electrical/ac3d_pore_gpu.py"):
        shutil.copy2(path, RUN / "configs/source_snapshot" / path.name)

    render_command = [sys.executable, str(ROOT / "code/scripts/pore_network/render_berea_pore_network_html.py"),
        "--pores", str(RUN / "pore_network/pores.csv"), "--throats", str(RUN / "pore_network/throats.csv"),
        "--out", str(RUN / "pore_network/sample16_avizo_network.html"),
        "--metadata-out", str(RUN / "pore_network/sample16_avizo_network_metadata.json"),
        "--voxel-size-m", "1.92e-6", "--segmented-volume", str(CT), "--pore-value", "2", "--solid-value", "1"]
    with (RUN / "provenance/network_render.log").open("w", encoding="utf8") as log:
        subprocess.run(render_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def run() -> None:
    info = solver.parse_mhd(RUN / "configs/field_ds2.mhd")
    labels = solver.load_label_volume(info)
    assert labels.shape == (400, 333, 333) and set(np.unique(labels)) == {0, 1}
    for mechanism in MECHANISMS:
        options = dict(
            labels=labels, mechanism=mechanism,
            spectra_path=RUN / f"spectra/sample16_avizo_component_spectra_{mechanism}.csv",
            frequencies_hz=FREQUENCIES.tolist(), voxel_size_m=info["voxel_size_m"],
            out_dir=RUN / "simulation_sweeps" / mechanism, pore_label=0, solid_label=1,
            direction="y", rtol=1e-6, atol=0.0, maxiter=50000, dtype_name="complex128",
            preconditioner="fft" if mechanism in {"maxwell", "all"} else "jacobi",
            fft_reference="mean-face", progress_every=500, residual_every=100,
            iteration_sleep_s=0.0, zero_solid_regularization_s_m=0.0, resume=True,
            solver_backend="gpu-ac3d" if mechanism in {"maxwell", "all"} else "gpu-active-pore-csr",
            jacobi_omega=0.65, sor_omega=1.35,
        )
        serializable = {k: str(v) if isinstance(v, Path) else v for k, v in options.items() if k != "labels"}
        dump(RUN / f"configs/{mechanism}_solver.json", serializable)
        with (RUN / f"provenance/{mechanism}_gpu.log").open("a", encoding="utf8") as log, redirect_stdout(log), redirect_stderr(log):
            frame = solver.run_component_sweep(**options)
        assert len(frame) == 7 and frame.converged_strict.all() and (frame.relative_residual_norm <= 1e-6).all()
        for result in (RUN / "simulation_sweeps" / mechanism).glob("frequency_*/result.json"):
            dump(result.parent / "config.yml", serializable)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print(f"DONE {mechanism}", flush=True)


def plot() -> None:
    experiment = pd.read_csv(RUN / "source_data/experiment_all.csv")
    frames = []
    for mechanism in MECHANISMS:
        frame = pd.read_csv(RUN / f"simulation_sweeps/{mechanism}/sweep_results.csv").sort_values("frequency_hz")
        assert len(frame) == 7 and np.allclose(frame.frequency_hz, FREQUENCIES, rtol=0, atol=1e-10)
        assert frame.converged_strict.all() and (frame.relative_residual_norm <= 1e-6).all()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(RUN / "source_data/four_mechanisms_7frequencies.csv", index=False)

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
                         "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
                         "axes.linewidth": 0.8, "legend.frameon": False})
    colors = {"maxwell": "#4C78A8", "pore": "#B279A2", "membrane": "#59A14F", "all": "#E15759"}
    labels = {"maxwell": "Interfacial", "pore": "Pore", "membrane": "Membrane", "all": "All"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for index, part in enumerate(("real", "imag")):
        ax = axes[index]
        exp_values = experiment[f"experiment_sigma_{part}_s_m"]
        keep = exp_values > 0
        ax.plot(experiment.frequency_hz[keep], exp_values[keep], "o", ms=2.5, color="#222222", alpha=0.55, label="Experiment")
        for mechanism, frame in zip(MECHANISMS, frames):
            values = frame[f"effective_sigma_{part}_s_m"]
            positive = values > 0
            ax.plot(frame.frequency_hz[positive], values[positive], "o-", ms=3.2, lw=1.35,
                    color=colors[mechanism], label=labels[mechanism])
        ax.set(xscale="log", yscale="log", xlim=(0.08, 25000), xlabel="Frequency (Hz)",
               ylabel=[r"$\sigma'$ (S m$^{-1}$)", r"$\sigma''$ (S m$^{-1}$)"][index])
        ax.grid(which="major", color="#E5E5E5", lw=0.5)
        ax.text(-0.16, 1.04, "ab"[index], transform=ax.transAxes, fontweight="bold", fontsize=9)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.52, 1.01), ncol=5, fontsize=6.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.82, wspace=0.34)
    fig.text(0.52, 0.07, "Avizo/CT geometry; y-direction ds2 field; seven GPU frequencies.", ha="center")
    figure = RUN / "figures/sample16_avizo_four_mechanisms_gpu7.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    plt.close(fig)

    tau = pd.read_csv(RUN / "pore_network/throats.csv").throat_length_m.to_numpy() ** 2 / (4 * PARAMS.diffusion_coefficient_m2_s)
    verification = {
        "gpu_points": len(combined), "all_strictly_converged": True,
        "max_true_relative_residual": float(combined.relative_residual_norm.max()),
        "no_pnextract_invoked": True, "fixed_relaxation_length_used": False,
        "relaxation_formula": "tau_m,j = ChannelLength_j^2 / (4 D)",
        "relaxation_time_s": {"min": float(tau.min()), "median": float(np.median(tau)), "max": float(tau.max())},
        "input_geometry": {"pores": 24117, "throats": 15113},
        "field_grid": {"shape_zyx": [400, 333, 333], "voxel_size_um": 3.84, "direction": "y", "classification": "ds2, not full-grid"},
        "figure_png": str(figure), "png_dpi": 600,
        "experiment_nonpositive_imag_omitted_from_log_plot": int((experiment.experiment_sigma_imag_s_m <= 0).sum()),
    }
    dump(RUN / "provenance/verification.json", verification)
    (RUN / "README.md").write_text(
        "# 样品16：Avizo/CT几何驱动的7频点SIP模拟\n\n"
        "本结果直接读取 `16-avizo-pore throat.xlsx`，未调用pnextract。24,117个孔与15,113条孔喉均通过ID、端点、正值、等效半径和配位数一致性检查。\n\n"
        "膜弛豫时间逐喉按 `tau=L^2/(4D)` 计算，L为Avizo `ChannelLength`；不存在固定0.3 um弛豫长度。Zdc使用同一ChannelLength与Excel Area，几何未缩放。"
        f"本次弛豫时间范围为{tau.min():.6e}至{tau.max():.6e} s，中位数{np.median(tau):.6e} s。\n\n"
        f"interfacial、pore、membrane、all共28个GPU点均严格收敛，最大真实相对残差{verification['max_true_relative_residual']:.3e}。"
        "电场为原始CT派生的400x333x333 ds2网格、3.84 um体素、y方向，不是800x666x666 full-grid。\n\n"
        "实验曲线是总响应；独立机制曲线不能分别当作对实验的拟合。eta=0.01、D=1.3e-9 m2/s等仍是当前模型参数，不是样品16独立实测值。"
        "图中log轴省略实验非正虚部，但CSV保留全部实验记录。\n",
        encoding="utf8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if "--prepare-only" in sys.argv:
        prepare()
    elif "--plot-only" in sys.argv:
        plot()
    else:
        if "--run-only" not in sys.argv:
            prepare()
        run()
        plot()
