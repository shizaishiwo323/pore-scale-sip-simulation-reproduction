"""由样品16已验证的7频点复电导率绘制相对介电常数。"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/正式结果/sample16_membrane_only_L030_gpu7_20260913"
GPU_SOURCE = RUN / "source_data/mechanism_overlay_gpu7.csv"
EXPERIMENT_SOURCE = RUN / "source_data/mechanism_overlay_experiment.csv"
PARAMETERS_SOURCE = RUN / "configs/parameters_zh.json"
EPSILON_0 = 8.8541878128e-12


def relative_permittivity(imaginary_conductivity, frequency):
    return imaginary_conductivity / (2 * np.pi * frequency * EPSILON_0)


def main():
    gpu = pd.read_csv(GPU_SOURCE)
    experiment = pd.read_csv(EXPERIMENT_SOURCE)
    parameters = json.loads(PARAMETERS_SOURCE.read_text(encoding="utf-8"))
    assert parameters["effective_relaxation_length_m"] == 3e-7
    assert np.isclose(parameters["parameters"]["membrane_polarizability"], 0.002442810092526288)
    assert set(gpu.component) == {"maxwell", "pore", "membrane", "all"}
    assert gpu.groupby("component").size().eq(7).all()
    assert gpu.converged_strict.all() and (gpu.relative_residual_norm <= 1e-6).all()

    gpu["relative_permittivity"] = relative_permittivity(
        gpu.effective_sigma_imag_s_m, gpu.frequency_hz
    )
    experiment["relative_permittivity"] = relative_permittivity(
        experiment.experiment_sigma_imag_s_m, experiment.frequency_hz
    )
    assert np.allclose(gpu.relative_permittivity, gpu.real_relative_permittivity, rtol=1e-12)

    source = pd.concat([
        gpu[["frequency_hz", "component", "relative_permittivity"]].assign(source="GPU simulation"),
        experiment[["frequency_hz", "relative_permittivity"]].assign(
            component="experiment", source="LKC-16 experiment"
        ),
    ], ignore_index=True)
    output_csv = RUN / "source_data/effective_permittivity_L030.csv"
    source.to_csv(output_csv, index=False)

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 11, "mathtext.fontset": "stix",
        "axes.linewidth": 0.8, "axes.spines.right": False,
        "axes.spines.top": False, "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    positive = experiment.relative_permittivity > 0
    ax.plot(experiment.frequency_hz[positive], experiment.relative_permittivity[positive],
            "^", color="black", mfc="white", ms=5, mew=0.9, ls="none",
            label="LKC-16 experiment", zorder=5)
    styles = [
        ("pore", "Pore polarization", "#148A2A", "--", "D"),
        ("membrane", "Membrane polarization", "#E53935", "--", "o"),
        ("maxwell", "Interfacial + dielectric", "#3157D5", "--", "s"),
        ("all", "All mechanisms (joint solve)", "#737373", "-", "^"),
    ]
    for key, label, color, linestyle, marker in styles:
        frame = gpu[gpu.component == key].sort_values("frequency_hz")
        ax.plot(frame.frequency_hz, frame.relative_permittivity,
                color=color, ls=linestyle, lw=2.1 if key == "all" else 1.6,
                marker=marker, ms=4, mfc="white", label=label, zorder=4 if key == "all" else 3)

    ax.set(xscale="log", yscale="log", xlim=(0.08, 25000), ylim=(1e-5, 1e7),
           xlabel="Frequency, $f$ (Hz)",
           ylabel=r"Effective relative permittivity, $\varepsilon'_{\mathrm{eff}}/\varepsilon_0$")
    ax.legend(loc="upper right", fontsize=10, handlelength=3.2)
    ax.set_title(r"Sample 16 | $L_{\mathrm{relax}}=0.3\ \mu$m | $\eta=0.00244$ | 7 GPU frequencies", pad=12)
    fig.tight_layout()

    figure = RUN / "figures/effective_permittivity_L030.png"
    preview = RUN / "figures/effective_permittivity_L030_preview.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    fig.savefig(preview, dpi=150, bbox_inches="tight")
    plt.close(fig)

    provenance = {
        "quantity": "effective relative permittivity epsilon_eff_prime/epsilon_0",
        "conversion": "sigma_imag/(2*pi*f*epsilon_0)",
        "epsilon_0_f_m": EPSILON_0,
        "effective_relaxation_length_m": 3e-7,
        "membrane_polarizability": parameters["parameters"]["membrane_polarizability"],
        "simulation_points_per_mechanism": 7,
        "new_gpu_points": 0,
        "all_curve": "joint field solution; standalone effective responses were not summed",
        "diagnostic_grid": "400x333x333, voxel 3.84 um, downsample=2, y direction",
        "omitted_nonpositive_experimental_points_on_log_axis": int((~positive).sum()),
        "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (GPU_SOURCE, EXPERIMENT_SOURCE, PARAMETERS_SOURCE)},
        "outputs": [str(figure.relative_to(ROOT)), str(output_csv.relative_to(ROOT))],
    }
    (RUN / "provenance/effective_permittivity_L030.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copy2(__file__, RUN / "configs/source_snapshot" / Path(__file__).name)
    print(json.dumps({"figure": str(figure), "preview": str(preview),
                      "omitted_experiment_points": int((~positive).sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
