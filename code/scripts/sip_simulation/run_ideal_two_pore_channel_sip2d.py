"""SIP response of the ideal 2-D big-pore / slit-throat / small-pore chain.

Uses the periodic AC2D solver and Niu et al. (2020) mechanisms in
``code/src/pore_scale_electrical_2d``. Writes a result package:

    results/<run>/configs/            config copy + resolved parameters
    results/<run>/simulation_sweeps/  <run_name>/sweep_results.csv per mechanism
    results/<run>/source_data/        region map, DC fields, Lambda table, Debye fits
    results/<run>/figures/            mechanism spectra, fields, Debye radius maps
    results/<run>/provenance/         run_manifest.json, README_zh.md

Example (from the sip模拟 project root):

    python code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py \
        --output-dir results/ideal_two_pore_channel_sip2d_20261012
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from pore_scale_electrical_2d import (  # noqa: E402
    Grid2D,
    RegionSpec,
    SIP2DParameters,
    TwoPoreChain2D,
    cell_fields,
    conductivity_map,
    debye_decomposition,
    dynamic_pore_size_2d,
    membrane_increments,
    phase_values,
    pore_increments,
    radius_from_frequency,
    solve_ac2d_periodic,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sip_simulation" / "ideal_two_pore_channel_sip2d_config.py"
REGION_NAMES = {1: "big_pore_body", 2: "small_pore_body", 3: "throat_big_to_small", 4: "throat_small_to_big_periodic"}

_WORKER: dict = {}


def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("ideal_sip2d_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.CONFIG)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def build_parameters(config: dict) -> SIP2DParameters:
    e, p = config["electrical"], config["polarization"]
    return SIP2DParameters(
        water_conductivity_s_m=e["water_conductivity_s_m"],
        solid_conductivity_s_m=e["solid_conductivity_s_m"],
        water_relative_permittivity=e["water_relative_permittivity"],
        solid_relative_permittivity=e["solid_relative_permittivity"],
        epsilon_0_f_m=e["epsilon_0_f_m"],
        surface_conductance_s=p["surface_conductance_s"],
        stern_diffusion_m2_s=p["stern_diffusion_m2_s"],
        membrane_diffusion_m2_s=p["membrane_diffusion_m2_s"],
        membrane_polarizability=p["membrane_polarizability"],
    )


def dc_field_analysis(chain: TwoPoreChain2D, grid: Grid2D, labels: np.ndarray, sigma_w: float):
    """Pore-only dc solve (solid = 0) -> fields, global and regional dynamic pore sizes."""

    pore = labels > 0
    sigma = np.where(pore, sigma_w, 0.0)
    result = solve_ac2d_periodic(sigma, grid)
    fields = cell_fields(result, sigma, grid)
    exact = chain.analytic_properties()
    rows = []
    global_lambda = dynamic_pore_size_2d(pore, fields, grid)
    rows.append({"region": "all_pore_space", "label": 0, **_lambda_row(global_lambda), "analytic_inscribed_radius_m": np.nan})
    regional = {}
    for label, name in REGION_NAMES.items():
        lam = dynamic_pore_size_2d(pore, fields, grid, labels == label)
        regional[label] = lam
        rows.append(
            {
                "region": name,
                "label": label,
                **_lambda_row(lam),
                "analytic_area_m2": exact[label]["area_m2"],
                "analytic_wall_length_m": exact[label]["wall_length_m"],
                "analytic_geometric_size_m": exact[label]["geometric_size_m"],
                "analytic_inscribed_radius_m": exact[label]["inscribed_radius_m"],
            }
        )
    return result, fields, global_lambda, regional, pd.DataFrame(rows)


def _lambda_row(lam) -> dict:
    return {
        "dynamic_pore_size_m": lam.dynamic_pore_size_m,
        "geometric_size_2A_over_P_m": lam.geometric_size_m,
        "pore_area_m2": lam.pore_area_m2,
        "wall_length_m": lam.wall_length_m,
        "area_field_energy_v2": lam.area_integral_v2,
        "wall_tangential_energy_v2_per_m": lam.wall_integral_v2_per_m,
    }


def region_specs(chain: TwoPoreChain2D, regional_lambda: dict, config: dict) -> dict[int, RegionSpec]:
    h = 0.5 * chain.throat_width_m
    throat_stern = bool(config["polarization"]["throat_stern_polarization"])
    return {
        1: RegionSpec(1, REGION_NAMES[1], "pore_body", chain.big_radius_m, regional_lambda[1].dynamic_pore_size_m),
        2: RegionSpec(2, REGION_NAMES[2], "pore_body", chain.small_radius_m, regional_lambda[2].dynamic_pore_size_m),
        3: RegionSpec(3, REGION_NAMES[3], "throat", h, regional_lambda[3].dynamic_pore_size_m, chain.throat_length_m, throat_stern),
        4: RegionSpec(4, REGION_NAMES[4], "throat", h, regional_lambda[4].dynamic_pore_size_m, chain.throat_length_m, throat_stern),
    }


def _init_worker(labels, dx, dy, tables, direction):
    _WORKER.update(labels=labels, grid=Grid2D(dx, dy), tables=tables, direction=direction)


def _solve_one(task):
    run_name, index = task
    solid, water = _WORKER["tables"][run_name]
    sigma = conductivity_map(_WORKER["labels"], solid[index], {k: v[index] for k, v in water.items()})
    start = time.perf_counter()
    result = solve_ac2d_periodic(sigma, _WORKER["grid"], _WORKER["direction"])
    return run_name, index, result.effective_conductivity_s_m, result.transverse_conductivity_s_m, result.residual_norm, result.n_unknowns, time.perf_counter() - start


def run_sweeps(config, params, regions, labels, grid, frequencies, global_lambda, out_dir: Path) -> dict[str, pd.DataFrame]:
    areas = {k: float(grid.cell_area_m2[labels == k].sum()) for k in regions}
    membrane = membrane_increments(frequencies, regions, params)
    pore = {
        mode: pore_increments(
            frequencies,
            regions,
            params,
            pore_mode=mode,
            region_areas_m2=areas,
            global_dynamic_pore_size_m=global_lambda.dynamic_pore_size_m,
        )
        for mode in ("regional", "uniform")
    }
    runs = config["sweep"]["runs"]
    tables = {
        run["name"]: phase_values(run["mechanism"], frequencies, regions, params, pore=pore[run["pore_mode"]], membrane=membrane)
        for run in runs
    }
    tasks = [(run["name"], i) for run in runs for i in range(frequencies.size)]
    # Full-domain (dielectric) runs are the slow ones; schedule them first.
    tasks.sort(key=lambda t: 0 if dict((r["name"], r["mechanism"]) for r in runs)[t[0]] in ("interfacial", "all") else 1)
    rows: dict[str, list] = {run["name"]: [] for run in runs}
    workers = int(config["sweep"]["workers"])
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(labels, grid.dx_m, grid.dy_m, tables, config["sweep"]["direction"]),
    ) as pool:
        for done, (name, index, sigma_eff, transverse, residual, unknowns, seconds) in enumerate(pool.map(_solve_one, tasks, chunksize=1), 1):
            rows[name].append((index, sigma_eff, transverse, residual, unknowns, seconds))
            if done % 50 == 0 or done == len(tasks):
                print(f"  solved {done}/{len(tasks)} ({time.perf_counter() - start:.0f}s)", flush=True)

    omega = 2.0 * np.pi * frequencies
    frames = {}
    for run in runs:
        data = sorted(rows[run["name"]])
        sigma_eff = np.array([r[1] for r in data])
        frame = pd.DataFrame(
            {
                "frequency_hz": frequencies,
                "sigma_real_s_m": sigma_eff.real,
                "sigma_imag_s_m": sigma_eff.imag,
                "phase_mrad": 1e3 * np.arctan2(sigma_eff.imag, sigma_eff.real),
                "transverse_sigma_abs_s_m": np.abs([r[2] for r in data]),
                "relative_residual": [r[3] for r in data],
                "unknowns": [r[4] for r in data],
                "lu_seconds": [r[5] for r in data],
                "mechanism": run["mechanism"],
                "pore_mode": run["pore_mode"] if run["mechanism"] in ("pore", "all") else "n/a",
                "effective_relative_permittivity": sigma_eff.imag / (omega * params.epsilon_0_f_m),
            }
        )
        sweep_dir = out_dir / "simulation_sweeps" / run["name"]
        sweep_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(sweep_dir / "sweep_results.csv", index=False)
        (sweep_dir / "config.json").write_text(
            json.dumps(
                {
                    "run": run,
                    "solver": "periodic AC2D finite volume, scipy SuperLU direct (MMD_AT_PLUS_A), one factorization per frequency",
                    "residual_history": "direct solver: single relative residual per frequency in sweep_results.csv",
                    "grid_shape_ny_nx": list(labels.shape),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        frames[run["name"]] = frame
    return frames


def run_debye(frames, config, out_dir: Path) -> dict[str, dict]:
    dd = config["debye"]
    f0 = np.geomspace(dd["relaxation_frequency_min_hz"], dd["relaxation_frequency_max_hz"], dd["bins"])
    tau = 1.0 / (2.0 * np.pi * f0)
    summaries = {}
    for name, frame in frames.items():
        sigma = frame["sigma_real_s_m"].to_numpy() + 1j * frame["sigma_imag_s_m"].to_numpy()
        fit = debye_decomposition(
            frame["frequency_hz"].to_numpy(), sigma, tau, dd["regularization"], include_permittivity=dd["include_permittivity"]
        )
        radius = fit.radius_m(dd["radius_diffusion_m2_s"])
        amplitude = fit.amplitudes_s_m
        table = pd.DataFrame(
            {
                "relaxation_frequency_hz": f0,
                "relaxation_time_s": tau,
                "radius_um": radius * 1e6,
                "amplitude_s_m": amplitude,
                "normalized_amplitude": amplitude / amplitude.sum() if amplitude.sum() > 0 else 0.0,
            }
        )
        table.to_csv(out_dir / "source_data" / f"debye_decomposition_{name}.csv", index=False)
        pd.DataFrame(
            {
                "frequency_hz": frame["frequency_hz"],
                "sigma_real_s_m": sigma.real,
                "sigma_imag_s_m": sigma.imag,
                "fit_real_s_m": fit.prediction.real,
                "fit_imag_s_m": fit.prediction.imag,
                "radius_at_frequency_um": radius_from_frequency(frame["frequency_hz"], dd["radius_diffusion_m2_s"]) * 1e6,
            }
        ).to_csv(out_dir / "source_data" / f"debye_fit_{name}.csv", index=False)
        peaks = local_peaks(radius * 1e6, amplitude)
        summaries[name] = {
            "sigma_0_s_m": fit.sigma_0_s_m,
            "epsilon_inf_f_m": fit.epsilon_inf_f_m,
            "epsilon_inf_relative": fit.epsilon_inf_f_m / config["electrical"]["epsilon_0_f_m"],
            "total_chargeability": fit.total_chargeability,
            "real_rmse_s_m": fit.real_rmse_s_m,
            "imag_rmse_s_m": fit.imag_rmse_s_m,
            "imag_rmse_over_max_imag": fit.imag_rmse_s_m / max(float(np.max(np.abs(sigma.imag))), 1e-300),
            "success": fit.success,
            "peaks": peaks,
        }
    return summaries


def local_peaks(radius_um: np.ndarray, amplitude: np.ndarray, threshold: float = 0.01) -> list[dict]:
    """Local maxima of a Debye spectrum with the amplitude fraction between valleys."""

    order = np.argsort(radius_um)
    r, a = radius_um[order], amplitude[order]
    if a.max() <= 0:
        return []
    idx = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > threshold * a.max()]
    edges = [0] + [idx[k] + int(np.argmin(a[idx[k] : idx[k + 1]])) for k in range(len(idx) - 1)] + [len(a)]
    total = a.sum()
    return [
        {
            "radius_um": float(r[i]),
            "amplitude_fraction": float(a[edges[k] : edges[k + 1]].sum() / total),
            "weighted_log_mean_radius_um": float(10 ** np.average(np.log10(r[edges[k] : edges[k + 1]]), weights=a[edges[k] : edges[k + 1]] + 1e-300)),
        }
        for k, i in enumerate(idx)
    ]


def grid_convergence(chain, sigma_w, steps) -> pd.DataFrame:
    rows = []
    for step in steps:
        grid = chain.grid(coarse_step_m=step)
        labels = chain.region_map(grid)
        result = solve_ac2d_periodic(np.where(labels > 0, sigma_w, 0.0), grid)
        rows.append(
            {
                "coarse_step_um": step * 1e6,
                "grid_ny": grid.ny,
                "grid_nx": grid.nx,
                "pore_unknowns": result.n_unknowns,
                "dc_sigma_eff_s_m": result.effective_conductivity_s_m.real,
                "formation_factor": sigma_w / result.effective_conductivity_s_m.real,
            }
        )
    return pd.DataFrame(rows)


def make_figures(out_dir, chain, grid, labels, fields, frames, debye, lambda_table, config):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 0.8})
    um = 1e6
    xe, ye = grid.x_edges_m * um, (grid.y_edges_m - 0.5 * chain.height_m) * um

    # Geometry and dc field.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [2.2, 1]}, constrained_layout=True)
    cmap = ListedColormap(["#d9d4c7", "#3b6fb6", "#5a9e6f", "#d28e38", "#c96f3a"])
    energy = np.abs(fields[0]) ** 2 + np.abs(fields[1]) ** 2
    energy = np.where(labels > 0, energy, np.nan)
    for row, (xlim, ylim, title) in enumerate(
        [((0, chain.period_m * um), (-chain.height_m * um / 2, chain.height_m * um / 2), "full periodic cell"),
         ((chain.throat_length_m * 0.5 * um + 2 * chain.big_radius_m * um - 1.0, chain.throat_length_m * 0.5 * um + 2 * chain.big_radius_m * um + 1.5), (-0.5, 0.5), "zoom: big pore -> throat mouth")]
    ):
        ax = axes[row, 0]
        ax.pcolormesh(xe, ye, labels, cmap=cmap, vmin=-0.5, vmax=4.5, shading="flat", rasterized=True)
        ax.set(xlim=xlim, ylim=ylim, aspect="equal" if row == 0 else "auto", xlabel="x (um)", ylabel="y (um)")
        ax.set_title(f"Region labels ({title})", loc="left", fontsize=8)
        ax = axes[row, 1]
        mesh = ax.pcolormesh(xe, ye, energy, norm=LogNorm(vmin=np.nanpercentile(energy, 1), vmax=np.nanmax(energy)), cmap="magma", shading="flat", rasterized=True)
        ax.set(xlim=xlim, ylim=ylim, xlabel="x (um)")
        ax.set_title("|E|^2 (dc, E0 = 1 V/m)", loc="left", fontsize=8)
        fig.colorbar(mesh, ax=ax, shrink=0.8)
    fig.savefig(out_dir / "figures" / "sip2d_geometry_and_dc_field.png", dpi=300)
    plt.close(fig)

    # Mechanism spectra.
    styles = {
        "interfacial": ("#7a7a7a", "-"),
        "pore_regional": ("#3b6fb6", "-"),
        "pore_uniform_niu": ("#3b6fb6", "--"),
        "membrane": ("#d28e38", "-"),
        "all_regional": ("#111111", "-"),
        "all_uniform_niu": ("#111111", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    for name, frame in frames.items():
        color, ls = styles.get(name, ("#999999", ":"))
        f = frame["frequency_hz"]
        axes[0].loglog(f, frame["sigma_real_s_m"], color=color, ls=ls, lw=1.2, label=name)
        imag = frame["sigma_imag_s_m"].where(frame["sigma_imag_s_m"] > 0)
        axes[1].loglog(f, imag, color=color, ls=ls, lw=1.2, label=name)
        axes[2].semilogx(f, frame["phase_mrad"], color=color, ls=ls, lw=1.2, label=name)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="sigma' (S/m)")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="sigma'' (S/m)")
    axes[2].set(xlabel="Frequency (Hz)", ylabel="Phase (mrad)", yscale="symlog")
    for ax, tag in zip(axes, "abc"):
        ax.grid(alpha=0.2, which="both")
        ax.set_title(tag, loc="left", weight="bold")
    axes[1].legend(fontsize=6, frameon=False)
    fig.savefig(out_dir / "figures" / "sip2d_mechanism_spectra.png", dpi=300)
    plt.close(fig)

    # Debye decomposition in radius space.
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for name in ("pore_regional", "membrane", "all_regional", "all_uniform_niu"):
        table = pd.read_csv(out_dir / "source_data" / f"debye_decomposition_{name}.csv")
        color, ls = styles[name]
        amp = table["amplitude_s_m"] / table["amplitude_s_m"].max()
        ax.semilogx(table["radius_um"], amp, color=color, ls=ls, lw=1.2, label=name)
    for label, color in ((1, "#3b6fb6"), (2, "#5a9e6f"), (3, "#d28e38")):
        r = lambda_table.loc[lambda_table["label"] == label, "analytic_inscribed_radius_m"].iloc[0] * um
        ax.axvline(r, color=color, ls=":", lw=0.9)
    ax.set(xlabel="r = sqrt(2 tau D+) (um)", ylabel="Debye amplitude / max", ylim=(0, 1.05))
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=6, frameon=False)
    fig.savefig(out_dir / "figures" / "sip2d_debye_radius_distribution.png", dpi=300)
    plt.close(fig)


def write_readme(out_dir: Path, manifest: dict) -> None:
    text = f"""# 理想二维大孔-喉道-小孔周期链 SIP 结果包

- 状态：{manifest['status']}
- 模型维度：二维（每单位纵深的平移不变挤出体），不是三维。
- 求解器：`code/src/pore_scale_electrical_2d`（AC2D-periodic 有限体积，Niu 2020 机制的二维改写）。
- 运行命令：`{manifest['command']}`

## 包含内容

- `configs/`：模板配置副本与实际解析参数（JSON）。
- `simulation_sweeps/<run>/sweep_results.csv`：每个机制每个频点的有效复电导率、相位、残差、未知数和 LU 时间。
  直接 LU 求解每个频点只有一个相对残差，没有 Krylov 残差历史。
- `source_data/`：区域标签、网格、直流场、动态孔径表、网格收敛表、Debye 分解结果。
- `figures/`：几何与直流场、机制谱、Debye 半径分布。

## 不适用项

AGENTS.md 要求的三维数字岩心 HTML 与 pnextract 孔隙网络 HTML 不适用：本模型是解析定义的二维理想几何，
没有 CT 体数据，也没有提取网络。实际孔体/孔喉尺寸由解析几何给出（`source_data/dynamic_pore_size_table.csv`
中的 analytic_* 列），孔径分布直方图在 NMR 项目的联合结果中绘制。

## 参数口径

- 孔（Stern）极化 D+ = {manifest['parameters']['stern_diffusion_m2_s']:.3g} m2/s：用户指定（与 Debye 半径换算同值）；
  Niu 2020 原文为 1.3e-9 m2/s。
- 膜极化 D = {manifest['parameters']['membrane_diffusion_m2_s']:.3g} m2/s、eta = {manifest['parameters']['membrane_polarizability']}、
  Sigma_S = {manifest['parameters']['surface_conductance_s']:.3g} S、sigma_w = {manifest['parameters']['water_conductivity_s_m']} S/m：Niu 2020。
- 几何尺寸、网格与频带：本次理想几何的设计选择。
- `*_regional`：各区域独立 Eq.12 增量（二维显式几何主结果）；`*_uniform_niu`：Niu 三维原式的全局均匀增量（对照）。
"""
    (out_dir / "provenance" / "README_zh.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "ideal_two_pore_channel_sip2d_20261012")
    parser.add_argument("--geometry-json", type=Path, default=None, help="override geometry (keys as in CONFIG['geometry'])")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="coarse grid, 1 point/decade; output marked as smoke")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.geometry_json is not None:
        config["geometry"].update(json.loads(args.geometry_json.read_text(encoding="utf-8")))
    if args.workers is not None:
        config["sweep"]["workers"] = args.workers
    if args.smoke:
        config["grid"]["coarse_step_m"] = 0.25e-6
        config["sweep"]["points_per_decade"] = 1
    out_dir = args.output_dir.resolve()
    for sub in ("configs", "simulation_sweeps", "source_data", "figures", "provenance"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "configs" / args.config.name)
    (out_dir / "configs" / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    g = config["geometry"]
    chain = TwoPoreChain2D(g["big_radius_m"], g["small_radius_m"], g["throat_width_m"], g["throat_length_m"], g["solid_margin_m"])
    grid = chain.grid(config["grid"]["coarse_step_m"], config["grid"]["throat_cells"], config["grid"]["growth"])
    labels = chain.region_map(grid)
    params = build_parameters(config)
    print(f"grid {grid.shape} ({labels.size} cells); period {chain.period_m*1e6:.1f} um x {chain.height_m*1e6:.1f} um", flush=True)

    t0 = time.perf_counter()
    dc, fields, global_lambda, regional_lambda, lambda_table = dc_field_analysis(chain, grid, labels, params.water_conductivity_s_m)
    lambda_table.to_csv(out_dir / "source_data" / "dynamic_pore_size_table.csv", index=False)
    np.save(out_dir / "source_data" / "region_map.npy", labels)
    np.savez_compressed(
        out_dir / "source_data" / "dc_pore_only_fields.npz",
        x_edges_m=grid.x_edges_m,
        y_edges_m=grid.y_edges_m,
        ex=fields[0].real,
        ey=fields[1].real,
        potential=np.nan_to_num(dc.potential.real),
    )
    convergence = grid_convergence(chain, params.water_conductivity_s_m, [0.25e-6, config["grid"]["coarse_step_m"], 0.1e-6])
    convergence.to_csv(out_dir / "source_data" / "dc_grid_convergence.csv", index=False)
    print(f"dc analysis done ({time.perf_counter()-t0:.0f}s); Lambda_global = {global_lambda.dynamic_pore_size_m*1e6:.4f} um", flush=True)

    regions = region_specs(chain, regional_lambda, config)
    sw = config["sweep"]
    frequencies = np.logspace(
        math.log10(sw["frequency_min_hz"]),
        math.log10(sw["frequency_max_hz"]),
        int(round(sw["points_per_decade"] * math.log10(sw["frequency_max_hz"] / sw["frequency_min_hz"]))) + 1,
    )
    t1 = time.perf_counter()
    frames = run_sweeps(config, params, regions, labels, grid, frequencies, global_lambda, out_dir)
    sweep_seconds = time.perf_counter() - t1
    debye = run_debye(frames, config, out_dir)
    make_figures(out_dir, chain, grid, labels, fields, frames, debye, lambda_table, config)

    script = Path(__file__).resolve()
    manifest = {
        "status": "smoke_test_only" if args.smoke else "complete",
        "model_dimension": "2D (per unit depth; translationally invariant extrusion)",
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "geometry": {**g, "period_m": chain.period_m, "height_m": chain.height_m, "big_center_m": chain.big_center_m, "small_center_m": chain.small_center_m},
        "grid": {**config["grid"], "shape_ny_nx": list(grid.shape), "min_dy_m": float(grid.dy_m.min()), "dx_m": float(grid.dx_m[0])},
        "parameters": {**config["electrical"], **config["polarization"]},
        "regions": {
            spec.name: {
                "label": spec.label,
                "kind": spec.kind,
                "relaxation_radius_m": spec.relaxation_radius_m,
                "dynamic_pore_size_m": spec.dynamic_pore_size_m,
                "membrane_length_m": spec.membrane_length_m,
                "stern_polarization": spec.stern_polarization,
                "stern_relaxation_time_s": spec.relaxation_radius_m**2 / (2 * params.stern_diffusion_m2_s),
                "stern_peak_frequency_hz": params.stern_diffusion_m2_s / (math.pi * spec.relaxation_radius_m**2),
                "membrane_relaxation_time_s": (spec.membrane_length_m**2 / (4 * params.membrane_diffusion_m2_s)) if spec.membrane_length_m else None,
            }
            for spec in regions.values()
        },
        "global_dynamic_pore_size_m": global_lambda.dynamic_pore_size_m,
        "dc_pore_only_sigma_eff_s_m": dc.effective_conductivity_s_m.real,
        "dc_formation_factor": params.water_conductivity_s_m / dc.effective_conductivity_s_m.real,
        "frequencies_hz": [float(frequencies.min()), float(frequencies.max()), int(frequencies.size)],
        "runs": config["sweep"]["runs"],
        "max_relative_residual": {name: float(frame["relative_residual"].max()) for name, frame in frames.items()},
        "debye": {"settings": config["debye"], "summaries": debye},
        "timing_s": {"sweeps": sweep_seconds, "total": time.perf_counter() - t0},
        "not_applicable": ["3D digital-rock HTML", "pnextract pore-network HTML (analytic 2D geometry; no CT volume or extracted network)"],
        "provenance": {
            "python": sys.version,
            "script": str(script),
            "script_sha256": sha256(script),
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "solver_package_sha256": {p.name: sha256(p) for p in sorted((PROJECT_ROOT / "code" / "src" / "pore_scale_electrical_2d").glob("*.py"))},
            "git_commit": git_commit(PROJECT_ROOT),
        },
    }
    (out_dir / "provenance" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    write_readme(out_dir, manifest)
    print(json.dumps({"status": manifest["status"], "output_dir": str(out_dir), "sweep_seconds": round(sweep_seconds)}), flush=True)


if __name__ == "__main__":
    main()
