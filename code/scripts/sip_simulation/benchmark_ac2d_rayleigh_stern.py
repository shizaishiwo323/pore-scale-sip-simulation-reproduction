"""Analytical benchmarks for the periodic 2-D solver (AC2D) and the Stern kernel.

Reference solution: Rayleigh multipole method for a square array of circular
cylinders (Rayleigh 1892; Perrins, McKenzie & McPhedran 1979, Eq. 7),
generalized to an order-dependent cylinder contrast sigma_l. This is exact for
a homogeneous cylinder, a coated cylinder and a thin surface-conductance layer.

Benchmarks (all in true 2-D, field along x):

* B0  - disks of relative conductivity sigma in water on N x N uniform grids,
        sigma in {0, 10, 10+5i}; error vs the Rayleigh solution.
* B0' - Keller reciprocity eps(sigma) * eps(1/sigma) = 1 (solver self-check).
* B1  - insulating grains carrying a Stern surface conductance
        Sigma*(w) = Sigma_S i w tau / (1 + i w tau) (Niu Eq. 17), three kernels:
          niu3d      tau = a^2 / (2 D+)       (Schwarz sphere, current code)
          2d_dipole  tau = a^2 / D+           (circle, dipole mode)
          2d_modal   tau_l = a^2 / (l^2 D+)   (circle, every multipole mode;
                                               exact thin-layer Schwarz model)
        Numerical routes: (N1) resolved surface shell of thickness delta with
        conductivity Sigma*/delta, compared with the exact coated-cylinder
        solution; (N2) Niu Eq. 12 water increment 2 C*/Lambda with the
        field-weighted dynamic pore size Lambda (what the production runs use).

B1 uses a = 0.2 um so that Du = Sigma_S / (a sigma_w) ~ 0.15; this is a test
parameter chosen for sensitivity, not a physical claim about any sample.

Example (from the sip模拟 project root):

    python code/scripts/sip_simulation/benchmark_ac2d_rayleigh_stern.py \
        --output-dir results/benchmark_ac2d_rayleigh_stern_20261023
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from pore_scale_electrical_2d import (  # noqa: E402
    Grid2D,
    cell_fields,
    dynamic_pore_size_2d,
    radius_from_frequency,
    solve_ac2d_periodic,
)

# Perrins et al. (1979) Table 3, square array, eps = sigma_eff / sigma_matrix.
PERRINS_TABLE3 = {
    (0.1, 10.0): 1.1782, (0.3, 10.0): 1.6520, (0.5, 10.0): 2.4155, (0.7, 10.0): 4.0617,
    (0.1, np.inf): 1.2222, (0.3, np.inf): 1.8602, (0.5, np.inf): 3.0802, (0.7, np.inf): 7.4327,
}

# B1 physical parameters (Niu 2020 / project values; a is a test choice).
SIGMA_W = 0.043          # S/m, Niu 2020
SIGMA_S = 1.3e-9         # S, Niu 2020
D_PLUS = 3.8e-12         # m2/s, user-specified Stern diffusivity of the project
B1_RADIUS = 0.2e-6       # m, test choice -> Du ~ 0.15
B1_FRACTION = 0.3        # grain area fraction
B1_SHELL_FRACTION = 0.1  # shell thickness delta / a for route N1
KERNELS = ("niu3d", "2d_dipole", "2d_modal")


# ----------------------------------------------------------------------------
# Rayleigh multipole reference
# ----------------------------------------------------------------------------

def square_lattice_sums(l_max: int) -> dict[int, float]:
    """S_l = sum' (m + i n)^-l over the unit square lattice, l = 4, 8, ... <= l_max.

    S_2 = pi (Rayleigh's shape-dependent value for a square array); S_l = 0
    for l = 2 mod 4. S_4 uses a large square truncation (tail ~ M^-2).
    """

    sums = {2: math.pi}
    for l in range(4, l_max + 1, 4):
        m_max = 1500 if l == 4 else 120
        total = 0.0
        n = np.arange(-m_max, m_max + 1)
        for m in range(-m_max, m_max + 1):
            z = m + 1j * n
            if m == 0:
                z = z[n != 0]
            total += float(np.sum(z ** (-l)).real)
        sums[l] = total
    return sums


_LATTICE: dict[int, float] = {}


def rayleigh_square(fraction: float, contrast_of_order, n_orders: int = 20) -> complex:
    """Effective conductivity / matrix conductivity of a square cylinder array.

    ``contrast_of_order(l)`` returns the (complex) relative conductivity
    sigma_l that cylinder multipole order l (odd) sees. Rayleigh system in
    scaled unknowns x_l = B_l / a^l, each row multiplied by (1 - sigma_l) so
    that sigma_l = 1 (no scattering) stays regular.
    """

    global _LATTICE
    orders = np.arange(1, 2 * n_orders, 2)
    l_needed = 2 * int(orders[-1])
    if not _LATTICE or max(_LATTICE) < l_needed:
        _LATTICE = square_lattice_sums(l_needed + 4)
    a = math.sqrt(fraction / math.pi)
    matrix = np.zeros((orders.size, orders.size), dtype=np.complex128)
    rhs = np.zeros(orders.size, dtype=np.complex128)
    for i, l in enumerate(orders):
        s = complex(contrast_of_order(int(l)))
        plus, minus = (1.0 + s, 1.0 - s) if np.isfinite(s) else (1.0, -1.0)
        matrix[i, i] += plus
        for j, m in enumerate(orders):
            lattice = _LATTICE.get(int(l + m), 0.0)
            if lattice:
                matrix[i, j] += minus * math.comb(int(l + m - 1), int(l)) * lattice * a ** (l + m)
        if l == 1:
            rhs[i] = minus * a
    x = np.linalg.solve(matrix, rhs)
    return 1.0 - 2.0 * math.pi * a * x[0]


def check_rayleigh_against_perrins() -> pd.DataFrame:
    rows = []
    for (fraction, sigma), table in PERRINS_TABLE3.items():
        value = rayleigh_square(fraction, lambda l, s=sigma: s, n_orders=30).real
        rows.append({"fraction": fraction, "contrast": sigma, "perrins_table3": table,
                     "rayleigh_this_code": value, "rel_diff": value / table - 1.0})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Grids and maps
# ----------------------------------------------------------------------------

def disk_radius_map(n: int, cell_m: float) -> tuple[Grid2D, np.ndarray]:
    """Uniform n x n grid over one periodic cell; radius of every cell centre."""

    h = cell_m / n
    centres = (np.arange(n) + 0.5) * h - 0.5 * cell_m
    return Grid2D.uniform(n, n, h), np.hypot(centres[None, :], centres[:, None])


def solve_eps(sigma_map: np.ndarray, grid: Grid2D, matrix_sigma: complex):
    result = solve_ac2d_periodic(sigma_map, grid, "x")
    return result, result.effective_conductivity_s_m / matrix_sigma


# ----------------------------------------------------------------------------
# B0 / B0'
# ----------------------------------------------------------------------------

def run_b0(sizes, fractions, contrasts) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, recip = [], []
    for fraction in fractions:
        a = math.sqrt(fraction / math.pi)
        for n in sizes:
            grid, radius = disk_radius_map(n, 1.0)
            disk = radius < a
            f_disc = float(disk.mean())
            for contrast in contrasts:
                exact = rayleigh_square(fraction, lambda l, s=contrast: s)
                exact_disc = rayleigh_square(f_disc, lambda l, s=contrast: s)
                t0 = time.perf_counter()
                result, eps = solve_eps(np.where(disk, contrast, 1.0), grid, 1.0)
                rows.append({
                    "fraction": fraction, "fraction_discrete": f_disc, "n": n, "radius_cells": a * n,
                    "contrast": str(contrast),
                    "eps_num_re": eps.real, "eps_num_im": eps.imag,
                    "eps_exact_re": exact.real, "eps_exact_im": exact.imag,
                    "rel_err": abs(eps - exact) / abs(exact),
                    "rel_err_vs_discrete_fraction": abs(eps - exact_disc) / abs(exact_disc),
                    "residual": result.residual_norm, "seconds": time.perf_counter() - t0,
                })
                if contrast != 0:
                    _, eps_inv = solve_eps(np.where(disk, 1.0 / contrast, 1.0), grid, 1.0)
                    product = eps * eps_inv
                    recip.append({"fraction": fraction, "n": n, "contrast": str(contrast),
                                  "product_re": product.real, "product_im": product.imag,
                                  "abs_product_minus_1": abs(product - 1.0)})
            print(f"B0 f={fraction} n={n} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(recip)


# ----------------------------------------------------------------------------
# B1
# ----------------------------------------------------------------------------

def stern_tau(kernel: str, radius_m: float, order: int = 1) -> float:
    if kernel == "niu3d":
        return radius_m**2 / (2.0 * D_PLUS)
    if kernel == "2d_dipole":
        return radius_m**2 / D_PLUS
    if kernel == "2d_modal":
        return radius_m**2 / (order**2 * D_PLUS)
    raise ValueError(kernel)


def stern_conductance(kernel: str, omega: float, radius_m: float, order: int = 1) -> complex:
    iwt = 1j * omega * stern_tau(kernel, radius_m, order)
    return SIGMA_S * iwt / (1.0 + iwt)


def thin_layer_contrast(kernel, omega, radius_m):
    """sigma_l = l Sigma_l*(w) / (a sigma_w): insulating core + surface conductance."""

    return lambda l: l * stern_conductance(kernel, omega, radius_m, l) / (radius_m * SIGMA_W)


def coated_contrast(kernel, omega, radius_m, delta_m):
    """Insulating core r < a - delta inside a shell of conductivity Sigma*/delta."""

    rho = (radius_m - delta_m) / radius_m
    shell = stern_conductance(kernel, omega, radius_m) / delta_m / SIGMA_W
    return lambda l: shell * (1.0 - rho ** (2 * l)) / (1.0 + rho ** (2 * l))


def exact_dynamic_pore_size(dukhin: float = 1e-6) -> float:
    """Johnson-Koplik-Schwartz Lambda from the exact first-order surface-conductance response.

    sigma_eff(Sigma) = sigma_eff(0) (1 + 2 Sigma / (Lambda sigma_w)) + O(Sigma^2), so
    Lambda = 2 a Du / (relative increment) with a vanishing static Du (all orders l Du).
    """

    e0 = rayleigh_square(B1_FRACTION, lambda l: 0.0).real
    e1 = rayleigh_square(B1_FRACTION, lambda l: l * dukhin).real
    return 2.0 * B1_RADIUS * dukhin / ((e1 - e0) / e0)


def run_b1(sizes, frequencies, numeric_frequencies) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = B1_RADIUS
    cell = a * math.sqrt(math.pi / B1_FRACTION)
    delta = B1_SHELL_FRACTION * a
    rows = []
    omega_all = 2 * np.pi * frequencies
    # Exact references on the full frequency grid.
    for kernel in KERNELS:
        for f_hz, w in zip(frequencies, omega_all):
            eps = rayleigh_square(B1_FRACTION, thin_layer_contrast(kernel, w, a))
            rows.append({"kernel": kernel, "route": "exact_thin_layer", "n": 0, "frequency_hz": f_hz,
                         "sigma_re": SIGMA_W * eps.real, "sigma_im": SIGMA_W * eps.imag})
            if kernel != "2d_modal":
                eps_c = rayleigh_square(B1_FRACTION, coated_contrast(kernel, w, a, delta))
                rows.append({"kernel": kernel, "route": "exact_coated_shell", "n": 0, "frequency_hz": f_hz,
                             "sigma_re": SIGMA_W * eps_c.real, "sigma_im": SIGMA_W * eps_c.imag})
    lambdas = []
    for n in sizes:
        grid, radius = disk_radius_map(n, cell)
        grain = radius < a
        # N2: Niu Eq. 12 increment with the field-weighted dynamic pore size.
        base_map = np.where(grain, 0.0, SIGMA_W)
        base, _ = solve_eps(base_map, grid, SIGMA_W)
        fields = cell_fields(base, base_map, grid)
        dps = dynamic_pore_size_2d(~grain, fields, grid)
        lam = dps.dynamic_pore_size_m
        lambdas.append({"n": n, "dynamic_pore_size_m": lam, "dynamic_pore_size_exact_m": exact_dynamic_pore_size(),
                        "geometric_size_m": dps.geometric_size_m,
                        "geometric_size_exact_m": B1_RADIUS * (1 - B1_FRACTION) / B1_FRACTION,
                        "sigma_dc_s_m": base.effective_conductivity_s_m.real,
                        "sigma_dc_exact_s_m": SIGMA_W * rayleigh_square(B1_FRACTION, lambda l: 0.0).real})
        for kernel in ("niu3d", "2d_dipole"):
            for route, lam_used in (("niu_eq12_increment", lam), ("niu_eq12_exact_lambda", exact_dynamic_pore_size())):
                for f_hz, w in zip(frequencies, omega_all):
                    value = base.effective_conductivity_s_m * (
                        1.0 + 2.0 * stern_conductance(kernel, w, a) / (lam_used * SIGMA_W))
                    rows.append({"kernel": kernel, "route": route, "n": n, "frequency_hz": f_hz,
                                 "sigma_re": value.real, "sigma_im": value.imag})
        # N1: resolved shell a - delta < r < a.
        core = radius < a - delta
        shell = grain & ~core
        for kernel in ("niu3d", "2d_dipole"):
            for f_hz in numeric_frequencies:
                w = 2 * np.pi * f_hz
                shell_sigma = stern_conductance(kernel, w, a) / delta
                sigma_map = np.where(core, 0.0, np.where(shell, shell_sigma, SIGMA_W))
                result, _ = solve_eps(sigma_map, grid, SIGMA_W)
                value = result.effective_conductivity_s_m
                rows.append({"kernel": kernel, "route": "numeric_resolved_shell", "n": n, "frequency_hz": f_hz,
                             "sigma_re": value.real, "sigma_im": value.imag})
        print(f"B1 n={n} done (Lambda = {lam * 1e6:.4f} um)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(lambdas)


def b1_errors(spectra: pd.DataFrame) -> pd.DataFrame:
    """Relative complex error of every numerical route vs its exact reference."""

    out = []
    reference_of = {"numeric_resolved_shell": "exact_coated_shell", "niu_eq12_increment": "exact_thin_layer",
                    "niu_eq12_exact_lambda": "exact_thin_layer"}
    for (kernel, route, n), group in spectra[spectra["n"] > 0].groupby(["kernel", "route", "n"]):
        ref = spectra[(spectra.kernel == kernel) & (spectra.route == reference_of[route])].set_index("frequency_hz")
        thin = spectra[(spectra.kernel == kernel) & (spectra.route == "exact_thin_layer")].set_index("frequency_hz")
        for _, row in group.iterrows():
            num = complex(row.sigma_re, row.sigma_im)
            r = ref.loc[row.frequency_hz]
            t = thin.loc[row.frequency_hz]
            exact = complex(r.sigma_re, r.sigma_im)
            exact_thin = complex(t.sigma_re, t.sigma_im)
            out.append({"kernel": kernel, "route": route, "n": n, "frequency_hz": row.frequency_hz,
                        "reference": reference_of[route],
                        "rel_err_complex": abs(num - exact) / abs(exact),
                        "rel_err_imag": abs(num.imag - exact.imag) / max(abs(exact.imag), 1e-300),
                        "rel_err_vs_thin_layer": abs(num - exact_thin) / abs(exact_thin)})
    return pd.DataFrame(out)


def b1_peaks(spectra: pd.DataFrame) -> pd.DataFrame:
    """Peak frequency of sigma'' per exact kernel and the radius each conversion implies."""

    rows = []
    for kernel in KERNELS:
        s = spectra[(spectra.kernel == kernel) & (spectra.route == "exact_thin_layer")].sort_values("frequency_hz")
        f = s.frequency_hz.to_numpy()
        y = s.sigma_im.to_numpy()
        k = int(np.argmax(y))
        lf = np.log10(f[k - 1:k + 2])
        c = np.polyfit(lf, y[k - 1:k + 2], 2)  # parabolic refinement in log f
        f_peak = 10 ** (-c[1] / (2 * c[0]))
        rows.append({
            "kernel": kernel,
            "tau_dipole_s": stern_tau(kernel, B1_RADIUS),
            "kernel_frequency_hz": 1 / (2 * np.pi * stern_tau(kernel, B1_RADIUS)),
            "peak_frequency_hz": f_peak,
            "peak_sigma_im_s_m": float(np.polyval(c, -c[1] / (2 * c[0]))),
            "radius_sphere_conversion_um": float(radius_from_frequency(f_peak, D_PLUS)) * 1e6,
            "radius_2d_conversion_um": math.sqrt(D_PLUS / (2 * np.pi * f_peak)) * 1e6,
            "true_radius_um": B1_RADIUS * 1e6,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def make_figures(out: Path, b0: pd.DataFrame, recip: pd.DataFrame, spectra: pd.DataFrame,
                 errors: pd.DataFrame, peaks: pd.DataFrame, n_fine: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                         "savefig.dpi": 300, "font.family": "DejaVu Sans"})
    colors = {"0.0": "#1f4e79", "10.0": "#b5651d", "(10+5j)": "#2e7d32"}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    for contrast, g in b0.groupby("contrast"):
        for fraction, mk in zip(sorted(g.fraction.unique()), ("o", "s", "^")):
            h = g[g.fraction == fraction].sort_values("n")
            ax.loglog(h.n, h.rel_err, marker=mk, ms=3, color=colors.get(contrast, "k"), lw=0.8,
                      label=f"σ = {contrast}" if fraction == g.fraction.min() else None)
    n = np.array(sorted(b0.n.unique()), dtype=float)
    ax.loglog(n, 0.5 * (n / n[0]) ** -1 * b0[b0.n == n[0]].rel_err.max(), "k--", lw=0.6, label="slope −1")
    ax.set_xlabel("cells per period N")
    ax.set_ylabel("|ε_num − ε_Rayleigh| / |ε_Rayleigh|")
    ax.set_title("(a) B0: disk arrays (o f=0.1, □ 0.3, △ 0.5)", loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for contrast, g in recip.groupby("contrast"):
        for fraction, mk in zip(sorted(g.fraction.unique()), ("o", "s", "^")):
            h = g[g.fraction == fraction].sort_values("n")
            ax.loglog(h.n, h.abs_product_minus_1, marker=mk, ms=3, color=colors.get(contrast, "k"), lw=0.8,
                      label=f"σ = {contrast}" if fraction == g.fraction.min() else None)
    ax.set_xlabel("cells per period N")
    ax.set_ylabel("|ε(σ)·ε(1/σ) − 1|")
    ax.set_title("(b) B0′: Keller reciprocity", loc="left")
    ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(n, [str(int(v)) for v in n])
    fig.tight_layout()
    fig.savefig(out / "fig_b0_convergence.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, kernel, title in zip(axes[:2], ("niu3d", "2d_dipole"),
                                 ("(a) kernel τ = a²/(2D₊)", "(b) kernel τ = a²/D₊")):
        thin = spectra[(spectra.kernel == kernel) & (spectra.route == "exact_thin_layer")]
        ax.semilogx(thin.frequency_hz, thin.sigma_im * 1e6, "k-", label="exact (Rayleigh)")
        niu = spectra[(spectra.kernel == kernel) & (spectra.route == "niu_eq12_increment") & (spectra.n == n_fine)]
        ax.semilogx(niu.frequency_hz, niu.sigma_im * 1e6, "--", color="#b5651d", label="Niu Eq.12 (2C*/Λ)")
        shell = spectra[(spectra.kernel == kernel) & (spectra.route == "numeric_resolved_shell") & (spectra.n == n_fine)]
        ax.semilogx(shell.frequency_hz, shell.sigma_im * 1e6, "o", ms=3, mfc="none", color="#1f4e79",
                    label="resolved shell (FV)")
        niu_x = spectra[(spectra.kernel == kernel) & (spectra.route == "niu_eq12_exact_lambda") & (spectra.n == n_fine)]
        ax.semilogx(niu_x.frequency_hz, niu_x.sigma_im * 1e6, ":", color="#2e7d32", label="Niu Eq.12, exact Λ")
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("σ″ (µS/m)")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    for ax in axes[:2]:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.45)
    ax = axes[2]
    styles = {"niu3d": ("#b5651d", "τ = a²/(2D₊)"), "2d_dipole": ("#1f4e79", "τ = a²/D₊"),
              "2d_modal": ("#2e7d32", "τ_l = a²/(l²D₊)")}
    for kernel in KERNELS:
        thin = spectra[(spectra.kernel == kernel) & (spectra.route == "exact_thin_layer")]
        color, label = styles[kernel]
        ls = ":" if kernel == "2d_modal" else "-"
        ax.semilogx(thin.frequency_hz, thin.sigma_im * 1e6, ls, color=color, label=label)
        p = peaks[peaks.kernel == kernel].iloc[0]
        ax.axvline(p.peak_frequency_hz, color=color, lw=0.5, ls="--")
    ax.set_xlabel("frequency (Hz)")
    ax.set_title("(c) exact, three kernels", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    fig.tight_layout()
    fig.savefig(out / "fig_b1_stern_spectra.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for (kernel, route), g in errors[errors.n == n_fine].groupby(["kernel", "route"]):
        if kernel != "2d_dipole":
            continue  # kernels give identical errors up to a frequency shift; show one
        color = {"numeric_resolved_shell": "#1f4e79", "niu_eq12_increment": "#b5651d",
                 "niu_eq12_exact_lambda": "#2e7d32"}[route]
        mk = "o" if route == "numeric_resolved_shell" else None
        ls = "none" if mk else "-"
        g = g.sort_values("frequency_hz")
        label = {"numeric_resolved_shell": "resolved shell vs exact shell",
                 "niu_eq12_increment": "Niu Eq.12, code Λ",
                 "niu_eq12_exact_lambda": "Niu Eq.12, exact Λ"}[route]
        ax.loglog(g.frequency_hz, g.rel_err_complex, marker=mk, ms=3, mfc="none", ls=ls, color=color, label=label)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("|σ*_num − σ*_exact| / |σ*_exact|")
    ax.set_title(f"B1 errors, N = {n_fine}, τ = a²/D₊", loc="left")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_b1_errors.png")
    plt.close(fig)


# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--b1-sizes", type=int, nargs="+", default=[256, 512])
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    check = check_rayleigh_against_perrins()
    check.to_csv(out / "rayleigh_vs_perrins_table3.csv", index=False)
    print(check.to_string(index=False), flush=True)

    b0, recip = run_b0(args.sizes, (0.1, 0.3, 0.5), (0.0, 10.0, 10.0 + 5.0j))
    b0.to_csv(out / "b0_disk_arrays.csv", index=False)
    recip.to_csv(out / "b0_keller_reciprocity.csv", index=False)

    frequencies = np.logspace(-1, 4, 81)
    numeric_frequencies = frequencies[::4]
    spectra, lambdas = run_b1(args.b1_sizes, frequencies, numeric_frequencies)
    errors = b1_errors(spectra)
    peaks = b1_peaks(spectra)
    spectra.to_csv(out / "b1_stern_spectra.csv", index=False)
    errors.to_csv(out / "b1_errors.csv", index=False)
    peaks.to_csv(out / "b1_peaks_and_radius_conversion.csv", index=False)
    lambdas.to_csv(out / "b1_dynamic_pore_size.csv", index=False)

    make_figures(out, b0, recip, spectra, errors, peaks, max(args.b1_sizes))
    manifest = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "model_dimension": "2D (translation-invariant extrusion), square periodic cell, field along x",
        "b0": {"sizes": args.sizes, "fractions": [0.1, 0.3, 0.5], "contrasts": ["0", "10", "10+5i"]},
        "b1": {"sigma_w_s_m": SIGMA_W, "sigma_s_s": SIGMA_S, "d_plus_m2_s": D_PLUS,
               "radius_m": B1_RADIUS, "fraction": B1_FRACTION, "shell_delta_over_a": B1_SHELL_FRACTION,
               "dukhin_number": SIGMA_S / (B1_RADIUS * SIGMA_W), "sizes": args.b1_sizes,
               "frequencies_hz": [float(frequencies[0]), float(frequencies[-1]), len(frequencies)],
               "parameter_sources": {"sigma_w, Sigma_S": "Niu et al. 2020", "D_plus": "project value (user-specified)",
                                     "radius, fraction, shell": "benchmark test choices"}},
        "reference": "Rayleigh multipole, square lattice, order-dependent contrast; checked vs Perrins 1979 Table 3",
        "runtime_s": time.perf_counter() - t_start,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(peaks.to_string(index=False))
    print(errors.groupby(["kernel", "route", "n"]).rel_err_complex.max().to_string())
    print(f"done in {manifest['runtime_s']:.0f} s -> {out.name}")


if __name__ == "__main__":
    main()
