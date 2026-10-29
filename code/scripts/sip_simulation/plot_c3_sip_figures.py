"""Publication figures for the C3 two-pore SIP study (PNG only; legend is the only annotation text).

Reads a result package written by ``run_ideal_two_pore_channel_sip2d.py`` (C3 config) after
``analyze_c3_sip_pore_size_inversion.py`` and writes into ``<results>/figures/``:

    c3_sip_niu_style_spectra.png       eps'_eff/eps0, sigma''_eff, sigma'_eff and its increment by mechanism
    c3_sip_uniform_vs_regional_spectra.png  sigma' and sigma'' of the Niu-uniform vs regional pore-polarization modes
    c3_sip_geometry_dc_field.png       region map and dc field energy of the periodic C3 cell
    c3_sip_relaxation_to_length.png    relaxation-time distributions mapped to length for each scenario
    c3_sip_inversion_summary.png       recovered lengths of every inversion route vs the true sizes

Each figure is checked with the nature-figure alignment gate, the 5 pt text floor and the
collision audit on a temporary QA PDF; the JSON reports are written beside the PNG.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SKILL_SCRIPTS = Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

# Niu et al. (2020, Figure 7) line conventions.
MECH = {
    "pore_niu": dict(color="#2e7d32", ls=(0, (6, 3)), lw=1.4, label="Pore polarization"),
    "membrane": dict(color="#c62828", ls=(0, (1.5, 1.5)), lw=1.6, label="Membrane polarization"),
    "interfacial": dict(color="#1f4fa3", ls=(0, (3.5, 2.5)), lw=1.4, label="Interfacial polarization"),
    "all_niu": dict(color="#8a8a8a", ls="-", lw=2.4, label="Simulation (all polarizations)"),
}
REF = {
    "body": dict(color="#222222", ls=(0, (5, 2.5)), lw=0.9, label="True pore-body radius ($R_B$, $R_S$)"),
    "h": dict(color="#222222", ls=(0, (1, 1.6)), lw=1.0, label="True throat half-width $h$"),
    "L": dict(color="#222222", ls=(0, (6, 2, 1, 2)), lw=0.9, label="True throat length $L$"),
    "lam": dict(color="#7a5195", ls=(0, (3, 1.5)), lw=1.0, label="Dynamic pore size $\\Lambda$ (Eq. 13)"),
}
FREQ_TICKS = [1e-3, 1e0, 1e3, 1e6, 1e9]
EPS_LIM, IMAG_LIM, DELTA_LIM = (1e0, 1e9), (1e-9, 1e2), (1e-8, 1e-3)
IMAG_LIM_MODE = (1e-9, 1e1)


def positive(values: np.ndarray) -> np.ndarray:
    """Mask non-positive values before plotting on a logarithmic axis."""

    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


def in_window(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Keep a log-axis curve inside [low, high]: the first point leaving the window is pinned to the bound,
    later out-of-window points are masked, so no clipped path runs across neighbouring panels."""

    values = np.asarray(values, dtype=float).copy()
    outside = ~((values >= low) & (values <= high))
    pinned = np.clip(values, low, high)
    keep_edge = outside & np.r_[False, ~outside[:-1]] | outside & np.r_[~outside[1:], False]
    values[outside] = np.nan
    values[keep_edge] = pinned[keep_edge]
    return values


def export(fig, png: Path, *, alignment_axes=None) -> dict:
    """Alignment gate, temporary-PDF text/collision audits, then the PNG (the only figure file)."""

    report = {}
    kwargs = {"axes": alignment_axes} if alignment_axes is not None else {}
    require_matplotlib_panel_alignment(
        fig, json_out=png.with_suffix(".alignment.json"), overlay_svg=None, tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, **kwargs
    )
    with tempfile.TemporaryDirectory() as tmp:
        qa_pdf = Path(tmp) / "figure_qa.pdf"
        fig.savefig(qa_pdf, bbox_inches="tight")
        text = subprocess.run([sys.executable, str(SKILL_SCRIPTS / "audit_pdf_text.py"), str(qa_pdf), "--min-pt", "5", "--json"],
                              capture_output=True, text=True)
        png.with_suffix(".text-audit.json").write_text(text.stdout, encoding="utf-8")
        collision = subprocess.run([sys.executable, str(SKILL_SCRIPTS / "audit_figure_collisions.py"), str(qa_pdf),
                                    "--json-out", str(png.with_suffix(".collision-audit.json"))], capture_output=True, text=True)
        report = {"text_audit_exit": text.returncode, "collision_exit": collision.returncode, "collision_summary": collision.stdout.strip()[-600:]}
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return report


def figure_spectra(sweeps: dict, out: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"hspace": 0.12, "wspace": 0.32})
    eps_ax, imag_ax, real_ax, delta_ax = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for name in ("pore_niu", "membrane", "interfacial", "all_niu"):
        frame, style = sweeps[name], MECH[name]
        f = frame["frequency_hz"].to_numpy()
        real = frame["sigma_real_s_m"].to_numpy()
        eps_ax.plot(f, in_window(positive(frame["effective_relative_permittivity"].to_numpy()), *EPS_LIM), **style)
        imag_ax.plot(f, in_window(positive(frame["sigma_imag_s_m"].to_numpy()), *IMAG_LIM), **style)
        real_ax.plot(f, real * 1e3, **style)
        delta_ax.plot(f, in_window(positive(real - real[0]), *DELTA_LIM), **style)
    eps_ax.set(xscale="log", yscale="log", ylim=EPS_LIM, ylabel="Effective permittivity $\\epsilon'_{eff}/\\epsilon_0$")
    imag_ax.set(xscale="log", yscale="log", ylim=IMAG_LIM, ylabel="Imaginary conductivity $\\sigma''_{eff}$ (S m$^{-1}$)")
    real_ax.set(xscale="log", ylabel="Real conductivity $\\sigma'_{eff}$ (mS m$^{-1}$)", xlabel="Frequency $f$ (Hz)")
    delta_ax.set(xscale="log", yscale="log", ylim=DELTA_LIM, xlabel="Frequency $f$ (Hz)",
                 ylabel="$\\sigma'_{eff}(f) - \\sigma'_{eff}(10^{-3}$ Hz) (S m$^{-1}$)")
    for ax in axes.ravel():
        ax.set_xlim(1e-3, 1e10)
        ax.set_xticks(FREQ_TICKS)
        ax.grid(color="#e6e6e6", lw=0.5, which="major")
    handles = [Line2D([], [], **MECH[k]) for k in ("pore_niu", "membrane", "interfacial", "all_niu")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=4, handlelength=3.2, columnspacing=1.6)
    fig.subplots_adjust(top=0.9)
    return export(fig, out / "c3_sip_niu_style_spectra.png")


MODE = {
    "pore_niu": dict(color="#2e7d32", ls=(0, (5, 2.5)), lw=1.4, label="Pore only, Niu uniform (volume-weighted $C^*$, global $\\Lambda$)"),
    "pore_regional": dict(color="#d9822b", ls=(0, (5, 2.5)), lw=1.4, label="Pore only, regional (own $\\tau_k$, $\\Lambda_k$ per region)"),
    "all_niu": dict(color="#2e7d32", ls="-", lw=2.0, label="All mechanisms, Niu uniform"),
    "all_regional": dict(color="#d9822b", ls="-", lw=2.0, label="All mechanisms, regional"),
}


def figure_mode_comparison(sweeps: dict, out: Path, results: Path) -> dict:
    regions = json.loads((results / "provenance" / "run_manifest.json").read_text(encoding="utf-8"))["regions"]
    marks = [
        (regions["big_pore_body"]["stern_peak_frequency_hz"], (0, (1, 1.6)), "Stern frequency $1/2\\pi\\tau$: big pore $B$"),
        (regions["small_pore_body"]["stern_peak_frequency_hz"], (0, (4, 1.5)), "Stern frequency: small pore $S$"),
        (regions["throat_big_to_small"]["stern_peak_frequency_hz"], (0, (6, 2, 1, 2)), "Stern frequency: throat"),
    ]
    fig, (real_ax, imag_ax) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.3})
    for name, style in MODE.items():
        frame = sweeps[name]
        f = frame["frequency_hz"].to_numpy()
        real_ax.plot(f, frame["sigma_real_s_m"].to_numpy() * 1e3, **style)
        imag_ax.plot(f, in_window(positive(frame["sigma_imag_s_m"].to_numpy()), *IMAG_LIM_MODE), **style)
    for freq, ls, _ in marks:
        for ax in (real_ax, imag_ax):
            ax.axvline(freq, color="#555555", ls=ls, lw=0.8, zorder=0)
    real_ax.set(xscale="log", ylabel="Real conductivity $\sigma'_{eff}$ (mS m$^{-1}$)", xlabel="Frequency $f$ (Hz)")
    imag_ax.set(xscale="log", yscale="log", ylim=IMAG_LIM_MODE, ylabel="Imaginary conductivity $\sigma''_{eff}$ (S m$^{-1}$)",
                xlabel="Frequency $f$ (Hz)")
    for ax in (real_ax, imag_ax):
        ax.set_xlim(1e-3, 1e10)
        ax.set_xticks(FREQ_TICKS)
        ax.grid(color="#e6e6e6", lw=0.5, which="major")
    handles = [Line2D([], [], **MODE[k]) for k in MODE] + [Line2D([], [], color="#555555", ls=ls, lw=0.8, label=lab) for _, ls, lab in marks]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, handlelength=3.0, columnspacing=1.6)
    return export(fig, out / "c3_sip_uniform_vs_regional_spectra.png")


def figure_geometry(results: Path, out: Path) -> dict:
    labels = np.load(results / "source_data" / "region_map.npy")
    fields = np.load(results / "source_data" / "dc_pore_only_fields.npz")
    xe, ye = fields["x_edges_m"] * 1e6, fields["y_edges_m"] * 1e6
    ye = ye - 0.5 * ye[-1]
    energy = fields["ex"] ** 2 + fields["ey"] ** 2  # E0 = 1 V/m, so this is |E|^2 / E0^2
    energy = np.where(labels > 0, energy, np.nan)
    fig = plt.figure(figsize=(7.2, 2.15))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.035], wspace=0.18)
    ax_map, ax_field, cax = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    colors = ["#d9d6cf", "#4f7fb8", "#8cbf6d", "#e07b39", "#e07b39"]
    cmap = matplotlib.colors.ListedColormap(colors)
    ax_map.pcolormesh(xe, ye, labels, cmap=cmap, vmin=-0.5, vmax=4.5, shading="flat", rasterized=True)
    log_energy = np.log10(np.clip(energy, 1e-6, None))  # floor keeps log10 finite in stagnant corners
    mesh = ax_field.pcolormesh(xe, ye, log_energy, cmap="magma", vmin=-3, vmax=float(np.nanmax(log_energy)), shading="flat", rasterized=True)
    ax_field.set_facecolor("#d9d6cf")
    for ax in (ax_map, ax_field):
        ax.set(aspect="equal", xlim=(xe[0], xe[-1]), ylim=(ye[0], ye[-1]), xlabel="$x$ ($\\mu$m)")
    ax_map.set_ylabel("$y$ ($\\mu$m)")
    ax_field.set_yticklabels([])
    fig.canvas.draw()  # equal-aspect maps shrink inside their cells; match the colorbar to the map height
    box, cbox = ax_field.get_position(), cax.get_position()
    cax.set_position([cbox.x0, box.y0, cbox.width, box.height])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("$\\log_{10}(|E|^2/E_0^2)$, dc")
    legend = [Patch(color=colors[0], label="Solid (insulating)"), Patch(color=colors[1], label="Big pore $B$ ($R_B$ = 20 $\\mu$m)"),
              Patch(color=colors[2], label="Small pore $S$ ($R_S$ = 5 $\\mu$m)"), Patch(color=colors[3], label="Throat ($w$ = 1 $\\mu$m, $L$ = 15 $\\mu$m)")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, handlelength=1.4, columnspacing=1.4)
    return export(fig, out / "c3_sip_geometry_dc_field.png", alignment_axes=[ax_map, ax_field])


def figure_relaxation(results: Path, out: Path, summary: dict) -> dict:
    rtd = pd.read_csv(results / "source_data" / "inversion_relaxation_distributions.csv")
    truth = summary["truth"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})

    def curve(ax, run, kernel, column, style):
        sub = rtd[(rtd["run"] == run) & (rtd["kernel"] == kernel)]
        amp = sub["amplitude_s_m"].to_numpy()
        ax.plot(sub[column], amp / amp.sum(), **style)

    def refs(ax, keys):
        for key in keys:
            values = {"body": [truth["big_radius_um"], truth["small_radius_um"]], "h": [truth["throat_half_width_um"]],
                      "L": [truth["throat_length_um"]], "lam": [truth["dynamic_pore_size_um"]]}[key]
            for value in values:
                ax.axvline(value, **{k: v for k, v in REF[key].items() if k != "label"}, zorder=0)

    curve(axes[0], "pore_niu", "debye", "radius_um_D1.3e-9", dict(color=MECH["pore_niu"]["color"], lw=1.6, label="Niu framework, pore only: Debye, $r^2=2D\\tau$"))
    curve(axes[0], "all_niu", "debye", "radius_um_D1.3e-9", dict(color=MECH["all_niu"]["color"], lw=1.6, ls=(0, (4, 2)), label="Niu framework, all mechanisms: Debye, $r^2=2D\\tau$"))
    refs(axes[0], ("body", "L"))
    curve(axes[1], "membrane", "debye", "radius_um_D1.3e-9", dict(color=MECH["membrane"]["color"], lw=1.6, label="Membrane only, read as pore: Debye, $r^2=2D\\tau$"))
    curve(axes[1], "membrane", "titov", "length_um_D1.3e-9", dict(color="#6d1b1b", lw=1.6, ls=(0, (4, 2)), label="Membrane only, read as throat: Titov, $L^2=4D\\tau_m$"))
    refs(axes[1], ("L",))
    curve(axes[2], "pore_regional", "debye", "radius_um_D1.3e-9", dict(color="#66bb6a", lw=1.6, label="Region-resolved, pore only: Debye, $r^2=2D\\tau$"))
    curve(axes[2], "all_regional", "debye", "radius_um_D1.3e-9", dict(color="#3a3a3a", lw=1.6, ls=(0, (4, 2)), label="Region-resolved, all mechanisms: Debye, $r^2=2D\\tau$"))
    refs(axes[2], ("body", "h", "lam"))
    for ax in axes:
        ax.set(xscale="log", xlim=(0.05, 300), xlabel="Inferred length ($\\mu$m)")
        ax.grid(color="#e6e6e6", lw=0.5, which="major")
    axes[0].set(ylabel="Normalized amplitude per bin", ylim=(0, None))
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    handles += [Line2D([], [], **REF[k]) for k in ("body", "h", "L", "lam")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, handlelength=3.0, columnspacing=1.6)
    return export(fig, out / "c3_sip_relaxation_to_length.png")


def figure_summary(out: Path, summary: dict, results: Path) -> dict:
    peaks = pd.read_csv(results / "source_data" / "inversion_peaks.csv")
    truth = summary["truth"]
    rows = [
        ("Niu, pore only — Debye, $D=1.3\\times10^{-9}$", "pore_niu", "debye_D1.3e-9", MECH["pore_niu"]["color"], "o"),
        ("Niu, all — Debye, $D=1.3\\times10^{-9}$", "all_niu", "debye_D1.3e-9", MECH["all_niu"]["color"], "o"),
        ("Niu, all — Debye, $D=3.8\\times10^{-12}$", "all_niu", "debye_D3.8e-12", MECH["all_niu"]["color"], "s"),
        ("Niu, all — $\\sigma''$ peak frequency", "all_niu", "peak_frequency_D1.3e-9", MECH["all_niu"]["color"], "^"),
        ("Niu, all — amplitude $\\rightarrow\\Lambda$", "all_niu", "amplitude_Lambda", MECH["all_niu"]["color"], "D"),
        ("Membrane only — Debye (as pore)", "membrane", "debye_D1.3e-9", MECH["membrane"]["color"], "o"),
        ("Membrane only — Titov (as throat length)", "membrane", "titov_length_D1.3e-9", "#6d1b1b", "o"),
        ("Region-resolved, all — Debye, $D=1.3\\times10^{-9}$", "all_regional", "debye_D1.3e-9", "#3a3a3a", "o"),
        ("Region-resolved, all — amplitude $\\rightarrow\\Lambda$", "all_regional", "amplitude_Lambda", "#3a3a3a", "D"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for y, (_, run, route, color, marker) in enumerate(rows):
        sub = peaks[(peaks["run"] == run) & (peaks["route"] == route)]
        frac = sub["amplitude_fraction"].fillna(1.0).to_numpy()
        ax.scatter(sub["peak_um"], np.full(len(sub), y), s=12 + 70 * frac, color=color, marker=marker, edgecolor="white", linewidth=0.5, zorder=3)
    for key, values in (("body", [truth["big_radius_um"], truth["small_radius_um"]]), ("h", [truth["throat_half_width_um"]]),
                        ("L", [truth["throat_length_um"]]), ("lam", [truth["dynamic_pore_size_um"]])):
        for value in values:
            ax.axvline(value, **{k: v for k, v in REF[key].items() if k != "label"}, zorder=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set(xscale="log", xlim=(0.1, 300), ylim=(len(rows) - 0.5, -0.5), xlabel="Recovered length ($\\mu$m)")
    ax.grid(color="#e6e6e6", lw=0.5, axis="x", which="major")
    handles = [Line2D([], [], **REF[k]) for k in ("body", "h", "L", "lam")]
    handles += [Line2D([], [], ls="none", marker="o", color="#777777", markersize=s, label=f"Peak, amplitude fraction {f}") for s, f in ((3.6, "0.1"), (8.8, "1.0"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, -0.06), ncol=3, handlelength=2.6, columnspacing=1.4)
    return export(fig, out / "c3_sip_inversion_summary.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "c3_two_pore_niu2020_sip2d_20261014")
    args = parser.parse_args()
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    runs = ("pore_niu", "membrane", "interfacial", "all_niu", "pore_regional", "all_regional")
    sweeps = {name: pd.read_csv(args.results / "simulation_sweeps" / name / "sweep_results.csv") for name in runs}
    summary = json.loads((args.results / "source_data" / "inversion_summary.json").read_text(encoding="utf-8"))
    qa = {
        "c3_sip_niu_style_spectra": figure_spectra(sweeps, out),
        "c3_sip_uniform_vs_regional_spectra": figure_mode_comparison(sweeps, out, args.results),
        "c3_sip_geometry_dc_field": figure_geometry(args.results, out),
        "c3_sip_relaxation_to_length": figure_relaxation(args.results, out, summary),
        "c3_sip_inversion_summary": figure_summary(out, summary, args.results),
    }
    (out / "figure_qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(json.dumps(qa, indent=1))


if __name__ == "__main__":
    main()
