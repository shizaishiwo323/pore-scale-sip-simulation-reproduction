"""sigma' / sigma'' of the Berea regional 3-D SIP simulation against the Niu et al. (2020) experiment.

Reads ``results/<run>/simulation_sweeps/<mechanism>/sweep_results.csv`` written by
``run_berea_regional_sip3d.py`` and the experiment table it exported, and writes into
``results/<run>/figures/`` (PNG only; the legend is the only annotation text):

    berea_regional_sip_vs_experiment.png   top: sigma'  (experiment, all mechanisms)
                                           bottom: sigma'' (experiment, Dielectric, EDL, Membrane, all)

Layout follows the user-supplied LKC-53 reference (two stacked log-log panels, black experiment
markers, red total, dashed single mechanisms). The 10 simulated frequencies are drawn as
markers joined by straight segments; nothing is interpolated between solves.
Source data: ``results/<run>/source_data/berea_regional_sip_vs_experiment_source_data.csv``.
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
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = PROJECT_ROOT / "results" / "berea_regional_sip_pnextract_10freq_20261014"
SKILL_SCRIPTS = Path.home() / ".claude" / "skills" / "nature-figure" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

# The collision audit needs PyMuPDF, which lives in the t2agent environment (QA only, no plotting).
QA_PYTHON = Path.home() / ".conda" / "envs" / "t2agent" / "python.exe"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

# Colour mapping of the reference figure: black experiment, red total, orange/blue/green mechanisms.
EXPERIMENT_STYLE = dict(color="#111111", ls="none", ms=2.6, mew=0.0, zorder=4)
SIM = {
    "interfacial": dict(color="#F28E2B", ls=(0, (4, 2)), lw=1.0, label="Dielectric"),
    "pore_regional": dict(color="#4C8FD6", ls=(0, (4, 2)), lw=1.0, label="EDL (regional)"),
    "membrane_regional": dict(color="#46B37F", ls=(0, (4, 2)), lw=1.0, label="Membrane (regional)"),
    "all_regional": dict(color="#E5534B", ls="-", lw=1.8, label="EDL+Dielectric+Membrane"),
}
MARKER = dict(marker="o", ms=2.4, mew=0.0)
X_LIM = (1e-3, 1e9)
REAL_LIM = (1e-3, 1e-1)
IMAG_LIM = (1e-8, 1e1)
# A simulated sigma'' is shown only where the mean-current and energy-form estimates agree.
RESOLUTION_TOLERANCE = 0.05


def positive(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


def in_window(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where((values >= low) & (values <= high), values, np.nan)


def load(run: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    experiment = pd.read_csv(run / "source_data" / "niu2020_experiment_sigma_real_imag.csv")
    sweeps = {name: pd.read_csv(run / "simulation_sweeps" / name / "sweep_results.csv").sort_values("frequency_hz") for name in SIM}
    return experiment, sweeps


def resolved_imag(frame: pd.DataFrame) -> np.ndarray:
    ok = frame["imag_relative_difference_mean_current_vs_energy"].to_numpy() <= RESOLUTION_TOLERANCE
    return np.where(ok, frame["sigma_imag_s_m"].to_numpy(), np.nan)


def style_axis(ax, ylim, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=40))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=40))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.tick_params(which="major", length=3, width=0.7)
    ax.tick_params(which="minor", length=1.6, width=0.5)


def figure(experiment: pd.DataFrame, sweeps: dict[str, pd.DataFrame], png: Path) -> tuple[dict, pd.DataFrame]:
    fig, (real_ax, imag_ax) = plt.subplots(2, 1, figsize=(3.5, 5.0), gridspec_kw={"hspace": 0.28})
    f_exp = experiment["frequency_hz"].to_numpy()

    real_ax.plot(f_exp, positive(experiment["sigma_real_s_m"]), marker="v", label="Experiment (Niu et al., 2020)", **EXPERIMENT_STYLE)
    total = sweeps["all_regional"]
    real_ax.plot(total["frequency_hz"], in_window(positive(total["sigma_real_s_m"]), *REAL_LIM), **SIM["all_regional"], **MARKER)
    style_axis(real_ax, REAL_LIM, "σ′ (S/m)")
    real_ax.legend(loc="upper left", handlelength=2.6, borderaxespad=0.6)

    imag_ax.plot(f_exp, positive(experiment["sigma_imag_s_m"]), marker="o", label="Experiment (Niu et al., 2020)", **EXPERIMENT_STYLE)
    rows = []
    for name, style in SIM.items():
        frame = sweeps[name]
        shown = in_window(positive(resolved_imag(frame)), *IMAG_LIM)
        # Thin dashed mechanisms above the thick total so the Dielectric line stays visible where it equals the total.
        imag_ax.plot(frame["frequency_hz"], shown, **style, **MARKER, zorder=2 if name == "all_regional" else 3)
        for row, imag_shown in zip(frame.itertuples(index=False), shown):
            rows.append(
                {
                    "dataset": "simulation_regional_3d",
                    "series": style["label"],
                    "run": name,
                    "frequency_hz": row.frequency_hz,
                    "sigma_real_s_m": row.sigma_real_s_m,
                    "sigma_imag_s_m": row.sigma_imag_s_m,
                    "energy_form_sigma_imag_s_m": row.energy_form_sigma_imag_s_m,
                    "sigma_real_plotted": name == "all_regional" and REAL_LIM[0] <= row.sigma_real_s_m <= REAL_LIM[1],
                    "sigma_imag_plotted": bool(np.isfinite(imag_shown)),
                }
            )
    style_axis(imag_ax, IMAG_LIM, "σ″ (S/m)")
    imag_ax.set_xlabel("Frequency (Hz)")
    # Plot order already matches the reference legend: experiment, Dielectric, EDL, Membrane, total.
    imag_ax.legend(loc="upper left", handlelength=2.6, borderaxespad=0.6)

    for row in experiment.itertuples(index=False):
        rows.append(
            {
                "dataset": "experiment_niu2020_figure7_figure8",
                "series": "Experiment (Niu et al., 2020)",
                "run": "experiment",
                "frequency_hz": row.frequency_hz,
                "sigma_real_s_m": row.sigma_real_s_m,
                "sigma_imag_s_m": row.sigma_imag_s_m,
                "energy_form_sigma_imag_s_m": np.nan,
                "sigma_real_plotted": REAL_LIM[0] <= row.sigma_real_s_m <= REAL_LIM[1],
                "sigma_imag_plotted": IMAG_LIM[0] <= row.sigma_imag_s_m <= IMAG_LIM[1],
            }
        )
    report = export(fig, png)
    return report, pd.DataFrame(rows)


def export(fig, png: Path) -> dict:
    """Alignment gate, temporary-PDF text/collision audits, then the PNG (the only figure file)."""

    require_matplotlib_panel_alignment(
        fig, json_out=png.with_suffix(".alignment.json"), overlay_svg=None, tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        qa_pdf = Path(tmp) / "figure_qa.pdf"
        fig.savefig(qa_pdf, bbox_inches="tight")
        text = subprocess.run(
            [sys.executable, str(SKILL_SCRIPTS / "audit_pdf_text.py"), str(qa_pdf), "--min-pt", "5", "--json"],
            capture_output=True, text=True,
        )
        png.with_suffix(".text-audit.json").write_text(text.stdout, encoding="utf-8")
        qa_python = str(QA_PYTHON if QA_PYTHON.exists() else sys.executable)
        collision = subprocess.run(
            [qa_python, str(SKILL_SCRIPTS / "audit_figure_collisions.py"), str(qa_pdf), "--json-out", str(png.with_suffix(".collision-audit.json"))],
            capture_output=True, text=True,
        )
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "text_audit_exit": text.returncode,
        "collision_exit": collision.returncode,
        "collision_summary": collision.stdout.strip()[-800:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    experiment, sweeps = load(run)
    (run / "figures").mkdir(parents=True, exist_ok=True)
    png = run / "figures" / "berea_regional_sip_vs_experiment.png"
    report, source = figure(experiment, sweeps, png)
    source.to_csv(run / "source_data" / "berea_regional_sip_vs_experiment_source_data.csv", index=False)
    report.update(
        {
            "figure": str(png.relative_to(PROJECT_ROOT)),
            "x_limits_hz": X_LIM,
            "sigma_real_limits_s_m": REAL_LIM,
            "sigma_imag_limits_s_m": IMAG_LIM,
            "imag_resolution_rule": f"simulated sigma'' plotted only if |mean-current - energy-form| / |energy-form| <= {RESOLUTION_TOLERANCE}",
            "experiment_points": int(len(experiment)),
            "experiment_nonpositive_imag_filtered": int((experiment["sigma_imag_s_m"] <= 0).sum()),
        }
    )
    (run / "figures" / "figure_qa_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
