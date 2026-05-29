# AC2D SI03 Mechanistic SIP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use paper SI-S03 only as the time-dependent pore-water conductivity driver, while computing the imaginary SIP response from the project's own mechanistic AC2D polarization model rather than the paper CEC/Waxman-Smits equivalent-circuit model.

**Architecture:** Split the current "paper-driven" workflow into two explicit paths: a diagnostic CEC/Waxman-Smits path kept for comparison, and a new SI03 mechanistic path used as the main AC2D simulation. The new path maps frame time to `sigma_w(t)`, then uses Maxwell-Wagner phase contrast plus Schwarz/Debye surface polarization with calibrated `Sigma_s`, `D`, and characteristic length `r`.

**Tech Stack:** Python, NumPy, pandas, SciPy, pytest, matplotlib, existing `pore_scale_electrical` package.

---

## Current Analysis

The user intends to use:

- Paper SI-S03: `docs/validation/2024gl111271-sup-0004-data set si-s03.csv`
- Our frame times: `data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/global_evolution_log.csv`
- Our interface images: `data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/interface_images/timestep_*.png`

SI-S03 contains `Time (h)`, `Water conductivity (S/m)`, `Porosity`, and `pH`. It should drive only the background water/electrolyte conductivity.

Observed time alignment:

| source | time range | conductivity range |
| --- | ---: | ---: |
| SI-S03 | `0.000000-0.218667 h` | `0.0096907-0.2542 S/m` |
| our 28 frames | `0.000278-4.534444 h` | requires interpolation/extrapolation |

Recommended conductivity mapping:

1. Interpolate SI-S03 within `0-0.218667 h`.
2. For later frames, linearly extrapolate from the last SI-S03 value to `sigma_HCl = 0.44 S/m` at the final simulated frame time.
3. Store the mapped `sigma_w(t)` in every output row so it is always auditable.

Sample mapped frame values using the above rule:

| frame | time (h) | `sigma_w(t)` (S/m) |
| ---: | ---: | ---: |
| 1 | 0.000278 | 0.019809 |
| 5 | 0.004444 | 0.111398 |
| 10 | 0.142222 | 0.241064 |
| 11 | 0.284444 | 0.257032 |
| 16 | 1.534444 | 0.310846 |
| 21 | 2.784444 | 0.364660 |
| 26 | 4.034444 | 0.418474 |
| 28 | 4.534444 | 0.440000 |

Current code issue to correct: `run_ac2d_microfluidic_sweep.py` currently has a `paper` driver path that imports SI01/SI02/SI03 and uses `waxman_smits_water_increment(...)`, which is a CEC/Waxman-Smits equivalent-circuit increment. That should not be the default for the mechanistic AC2D model.

Important physical issue: the current Schwarz grain term uses the whole calcite equivalent radius as `r`. In frame 1, `r_eq ~= 6.19e-4 m`, so with `D = 1.3e-9 m2/s`, `tau = r^2/(2D) ~= 147 s`. At `2.5 Hz`, `omega tau ~= 2300`, so the Debye imaginary factor is only about `1/(omega tau) ~= 4.3e-4`. This pushes `sigma''` too low unless `Sigma_s` is made unrealistically large.

For a 2.5 Hz-centered Schwarz response with `D = 1.3e-9 m2/s`, the characteristic length should be:

```text
tau_peak = 1 / (2*pi*2.5) = 0.0637 s
r_peak = sqrt(2*D*tau_peak) = 1.29e-5 m = 12.9 um
```

Therefore the mechanistic model needs a configurable characteristic length. The default should not be the whole calcite body radius; it should be a local interface/roughness/throat scale, initially exposed as a constant parameter around `10-20 um`, then later derived from image geometry if needed.

---

## File Structure

- Modify `code/src/pore_scale_electrical/microfluidic_2d.py`
  - Add SI03-only solution conductivity driver.
  - Add mechanistic Schwarz interface/grain polarization helper that does not use CEC.
  - Keep Waxman-Smits helper, but mark it diagnostic and do not use it in the default AC2D path.

- Modify `code/scripts/run_ac2d_microfluidic_sweep.py`
  - Add `--driver-mode si03` as the new default.
  - Rename current CEC path to `--driver-mode paper-cec` or keep `paper` only as explicit diagnostic mode.
  - Add `--schwarz-characteristic-length-m`, `--schwarz-sigma-s`, and `--schwarz-diffusion-coefficient`.
  - Output to `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`.

- Modify `code/scripts/plot_ac2d_2p5hz_paper_comparison.py`
  - Point default result path to the mechanistic run.
  - Label the model as `AC2D mechanistic: SI03 sigma_w + MW + Schwarz`.

- Modify `code/scripts/plot_ac2d_microfluidic_paper_style.py`
  - Point default result and geometry paths to the mechanistic run.

- Create `code/scripts/calibrate_ac2d_mechanistic_params.py`
  - Run a small parameter sweep over `Sigma_s` and `r`.
  - Use SI-S03 for `sigma_w(t)`.
  - Compare to paper SI-S02 `sigma''(2.5 Hz)` only as an observation target, not via CEC.
  - Save calibration table and plot.

- Modify `code/tests/test_microfluidic_2d.py`
  - Add tests for SI03-only conductivity mapping.
  - Add tests for Schwarz characteristic-length behavior.
  - Add a regression test that the mechanistic path does not require SI02/CEC.

---

## Task 1: Add SI03-Only Conductivity Driver

**Files:**
- Modify: `code/src/pore_scale_electrical/microfluidic_2d.py`
- Test: `code/tests/test_microfluidic_2d.py`

- [ ] **Step 1: Write the failing test**

Add this test to `code/tests/test_microfluidic_2d.py`:

```python
def test_si03_solution_conductivity_driver_interpolates_and_extends():
    si03 = {
        "Time (h)": [0.0, 0.5],
        "Water conductivity (S/m)": [0.01, 0.2],
    }

    drivers = si03_solution_conductivity_at_times(
        np.array([0.25, 1.0]),
        si03=si03,
        sigma_hcl_s_m=0.44,
        end_time_h=1.0,
    )

    assert np.allclose(drivers, [0.105, 0.44])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_si03_solution_conductivity_driver_interpolates_and_extends
```

Expected: FAIL with import/name error for `si03_solution_conductivity_at_times`.

- [ ] **Step 3: Implement the SI03-only helper**

In `code/src/pore_scale_electrical/microfluidic_2d.py`, add:

```python
def si03_solution_conductivity_at_times(
    time_h: np.ndarray | float,
    *,
    si03: pd.DataFrame | dict[str, object],
    sigma_hcl_s_m: float = 0.44,
    end_time_h: float | None = None,
) -> np.ndarray:
    """Interpolate SI-S03 water conductivity and extend to HCl conductivity."""

    target = np.atleast_1d(np.asarray(time_h, dtype=float))
    frame = pd.DataFrame(si03)
    src_t = frame["Time (h)"].to_numpy(dtype=float)
    src_sigma = frame["Water conductivity (S/m)"].to_numpy(dtype=float)
    out = np.interp(np.minimum(target, src_t[-1]), src_t, src_sigma)
    later = target > src_t[-1]
    if np.any(later):
        final_time = float(np.max(target)) if end_time_h is None else float(end_time_h)
        denom = max(final_time - src_t[-1], np.finfo(float).eps)
        frac = np.clip((target[later] - src_t[-1]) / denom, 0.0, 1.0)
        out[later] = src_sigma[-1] + frac * (sigma_hcl_s_m - src_sigma[-1])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_si03_solution_conductivity_driver_interpolates_and_extends
```

Expected: PASS.

---

## Task 2: Add Mechanistic Schwarz Interface Polarization

**Files:**
- Modify: `code/src/pore_scale_electrical/microfluidic_2d.py`
- Test: `code/tests/test_microfluidic_2d.py`

- [ ] **Step 1: Write the failing tests**

Add:

```python
def test_schwarz_interface_delta_peaks_when_omega_tau_is_one():
    freq = np.array([2.5])
    diffusion = 1.3e-9
    radius = np.sqrt(2.0 * diffusion / (2.0 * np.pi * freq[0]))
    delta = schwarz_interface_conductivity_delta(
        freq,
        interface_density_1_m=np.array([1000.0]),
        sigma_s_s=1.0e-5,
        characteristic_length_m=radius,
        diffusion_coefficient_m2_s=diffusion,
    )

    assert np.isclose(delta[0].real, 5.0e-3, rtol=1e-6)
    assert np.isclose(delta[0].imag, 5.0e-3, rtol=1e-6)


def test_schwarz_interface_delta_uses_local_length_not_calcite_body_radius():
    freq = np.array([2.5])
    diffusion = 1.3e-9
    density = np.array([878.0])
    sigma_s = 1.0e-5

    local = schwarz_interface_conductivity_delta(
        freq,
        interface_density_1_m=density,
        sigma_s_s=sigma_s,
        characteristic_length_m=1.3e-5,
        diffusion_coefficient_m2_s=diffusion,
    )
    body = schwarz_interface_conductivity_delta(
        freq,
        interface_density_1_m=density,
        sigma_s_s=sigma_s,
        characteristic_length_m=6.2e-4,
        diffusion_coefficient_m2_s=diffusion,
    )

    assert abs(local[0].imag) > 100.0 * abs(body[0].imag)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_schwarz_interface_delta_peaks_when_omega_tau_is_one tests\test_microfluidic_2d.py::test_schwarz_interface_delta_uses_local_length_not_calcite_body_radius
```

Expected: FAIL with import/name error for `schwarz_interface_conductivity_delta`.

- [ ] **Step 3: Implement mechanistic Schwarz helper**

Add to `code/src/pore_scale_electrical/microfluidic_2d.py`:

```python
def schwarz_interface_conductivity_delta(
    frequency_hz: np.ndarray,
    *,
    interface_density_1_m: np.ndarray | float,
    sigma_s_s: float,
    characteristic_length_m: np.ndarray | float,
    diffusion_coefficient_m2_s: float,
) -> np.ndarray:
    """Upscale Schwarz/Debye surface conductance to volumetric conductivity.

    C_p* = i omega tau / (1 + i omega tau) * Sigma_s
    Delta sigma* = C_p* * interface_density
    tau = r^2 / (2D)
    """

    freq = np.asarray(frequency_hz, dtype=float)
    omega = 2.0 * np.pi * freq
    density = np.asarray(interface_density_1_m, dtype=float)
    radius = np.asarray(characteristic_length_m, dtype=float)
    tau = radius**2 / (2.0 * diffusion_coefficient_m2_s)
    iwt = 1j * omega * tau
    return sigma_s_s * density * iwt / (1.0 + iwt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_schwarz_interface_delta_peaks_when_omega_tau_is_one tests\test_microfluidic_2d.py::test_schwarz_interface_delta_uses_local_length_not_calcite_body_radius
```

Expected: PASS.

---

## Task 3: Add `si03` Driver Mode to the Sweep Script

**Files:**
- Modify: `code/scripts/run_ac2d_microfluidic_sweep.py`
- Test: `code/tests/test_microfluidic_2d.py`

- [ ] **Step 1: Update CLI defaults**

In `code/scripts/run_ac2d_microfluidic_sweep.py`:

```python
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1"
DEFAULT_MECHANISMS = ["maxwell", "grain", "all"]
```

Change:

```python
parser.add_argument("--driver-mode", choices=["paper", "constant"], default="paper")
```

to:

```python
parser.add_argument("--driver-mode", choices=["si03", "paper-cec", "constant"], default="si03")
parser.add_argument("--schwarz-sigma-s", type=float, default=7.0e-5)
parser.add_argument("--schwarz-characteristic-length-m", type=float, default=1.3e-5)
parser.add_argument("--schwarz-diffusion-coefficient", type=float, default=1.3e-9)
```

Use `paper-cec` only for the previous CEC/Waxman-Smits diagnostic behavior.

- [ ] **Step 2: Route `si03` mode**

In the frame loop, replace the default paper branch with:

```python
if args.driver_mode == "si03":
    water_conductivity = si03_solution_conductivity_at_times(
        np.array([frame_times[frame]["time_h"]]),
        si03=read_semicolon_csv(Path(args.paper_si03)),
        sigma_hcl_s_m=args.sigma_hcl,
        end_time_h=max(frame_times[f]["time_h"] for f in args.frames),
    )[0]
    interface_density = metrics.interface_length_m / max(metrics.active_area_m2, np.finfo(float).eps)
    schwarz_delta = schwarz_interface_conductivity_delta(
        frequencies,
        interface_density_1_m=interface_density,
        sigma_s_s=args.schwarz_sigma_s,
        characteristic_length_m=args.schwarz_characteristic_length_m,
        diffusion_coefficient_m2_s=args.schwarz_diffusion_coefficient,
    )
elif args.driver_mode == "paper-cec":
    ...
elif args.driver_mode == "constant":
    water_conductivity = args.water_conductivity
    schwarz_delta = np.zeros_like(frequencies, dtype=np.complex128)
```

Pass `schwarz_delta` into `mechanism_phase_conductivities(...)` through `interface_delta_conductivity_s_m=schwarz_delta`, or rename this parameter to `mechanistic_interface_delta_conductivity_s_m` if making the semantic boundary clearer.

- [ ] **Step 3: Keep metadata auditable**

Add to `metadata`:

```python
"driver_mode": args.driver_mode,
"si03_driver": {
    "path": str(Path(args.paper_si03)),
    "sigma_hcl_s_m": args.sigma_hcl,
    "extension": "linear from last SI-S03 value to sigma_hcl at final simulated frame",
},
"schwarz_parameters": {
    "sigma_s_s": args.schwarz_sigma_s,
    "characteristic_length_m": args.schwarz_characteristic_length_m,
    "diffusion_coefficient_m2_s": args.schwarz_diffusion_coefficient,
    "tau_s": args.schwarz_characteristic_length_m**2 / (2.0 * args.schwarz_diffusion_coefficient),
},
```

- [ ] **Step 4: Run a smoke test**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\run_ac2d_microfluidic_sweep.py --frames 1 10 20 28 --frequencies 2.5 --downsample 16
```

Expected:

- Writes `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/ac2d_sweep_results.csv`.
- CSV contains `paper_sigma_w_s_m` or renamed `solution_sigma_w_s_m`.
- CSV does not require `paper_cec_meq_g` for `driver_mode=si03`.
- `all` includes Maxwell-Wagner plus mechanistic Schwarz interface/grain terms.

---

## Task 4: Add Parameter Calibration Script

**Files:**
- Create: `code/scripts/calibrate_ac2d_mechanistic_params.py`

- [ ] **Step 1: Create the script**

Create a script with this structure:

```python
#!/usr/bin/env python3
"""Calibrate mechanistic Schwarz parameters for AC2D microfluidic SIP."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_paper_2p5(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=";")
    return table.rename(
        columns={
            "Time (h)": "time_h",
            "Imaginary conductivity (S/m)": "paper_sigma_imag_s_m",
            "Real conductivity (S/m)": "paper_sigma_real_s_m",
        }
    )


def score_run(results_path: Path, paper: pd.DataFrame) -> dict[str, float]:
    ours = pd.read_csv(results_path)
    ours = ours[(ours["mechanism"] == "all") & np.isclose(ours["frequency_hz"], 2.5)].sort_values("time_h")
    mask = (paper["time_h"] >= ours["time_h"].min()) & (paper["time_h"] <= ours["time_h"].max())
    interp_imag = np.interp(paper.loc[mask, "time_h"], ours["time_h"], ours["sigma_imag_s_m"])
    interp_real = np.interp(paper.loc[mask, "time_h"], ours["time_h"], ours["sigma_real_s_m"])
    return {
        "imag_mae_s_m": float(np.mean(np.abs(paper.loc[mask, "paper_sigma_imag_s_m"] - interp_imag))),
        "real_mae_s_m": float(np.mean(np.abs(paper.loc[mask, "paper_sigma_real_s_m"] - interp_real))),
    }
```

The script should run a small grid:

```python
sigma_s_values = np.logspace(-6, -3, 13)
length_values = np.array([5e-6, 8e-6, 1.0e-5, 1.3e-5, 2.0e-5, 3.0e-5, 5.0e-5])
```

For each pair, call `run_ac2d_microfluidic_sweep.py` with:

```powershell
--driver-mode si03 --mechanisms maxwell grain all --frequencies 2.5 --downsample 16 --schwarz-sigma-s <value> --schwarz-characteristic-length-m <value>
```

Save:

- `results/ac2d_microfluidic/mechanistic_calibration_v1/calibration_grid.csv`
- `figures/ac2d_microfluidic/ac2d_mechanistic_calibration_grid.png`

- [ ] **Step 2: Run the calibration script**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\calibrate_ac2d_mechanistic_params.py
```

Expected:

- Calibration CSV lists `sigma_s_s`, `characteristic_length_m`, `tau_s`, `imag_mae_s_m`, and `real_mae_s_m`.
- Best rows should prefer a local characteristic length around the `10 um` scale if the 2.5 Hz imaginary response is the target.

---

## Task 5: Update Figures and Comparison Labels

**Files:**
- Modify: `code/scripts/plot_ac2d_2p5hz_paper_comparison.py`
- Modify: `code/scripts/plot_ac2d_microfluidic_paper_style.py`

- [ ] **Step 1: Point plotting defaults to mechanistic output**

Set:

```python
DEFAULT_OURS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_sweep_results.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_sweep_results.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "geometry_metrics.csv"
```

Use output names:

```python
ac2d_2p5hz_si03_mechanistic_vs_paper.png
ac2d_microfluidic_paper_style_si03_mechanistic.png
```

- [ ] **Step 2: Update plot labels**

Use labels:

```text
Paper SI-S02 measurement
AC2D mechanistic: SI-S03 sigma_w(t) + MW + Schwarz
```

Do not label the new curve as `Waxman-Smits`, `CEC`, or `paper-consistent`.

- [ ] **Step 3: Generate figures**

Run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_2p5hz_paper_comparison.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_microfluidic_paper_style.py
```

Expected output:

- `figures/ac2d_microfluidic/ac2d_2p5hz_si03_mechanistic_vs_paper.png`
- `figures/ac2d_microfluidic/ac2d_microfluidic_paper_style_si03_mechanistic.png`

---

## Task 6: Documentation Cleanup

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md` if it already documents AC2D results

- [ ] **Step 1: Clarify model variants**

Add this distinction:

```text
AC2D SI03 mechanistic model:
- Uses SI-S03 only for sigma_w(t).
- Does not use CEC or Waxman-Smits as a source term.
- Computes imaginary response from Maxwell-Wagner phase contrast and Schwarz/Debye interface polarization.

Paper CEC diagnostic:
- Recomputes the paper's empirical/equivalent-circuit model.
- Useful only as a reference check.
- Not the project's mechanistic AC2D result.
```

- [ ] **Step 2: Record parameter defaults**

Record:

```text
D = 1.3e-9 m2/s
r_char default = 1.3e-5 m
tau = r_char^2 / (2D) ~= 0.065 s
2.5 Hz omega*tau ~= 1
Sigma_s initial scan range = 1e-6 to 1e-3 S
```

---

## Verification Commands

After implementation, run:

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py tests\test_ac2d_solver.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\run_ac2d_microfluidic_sweep.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_2p5hz_paper_comparison.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_microfluidic_paper_style.py
```

Expected:

- Tests pass.
- New results are under `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`.
- New figures are under `figures/ac2d_microfluidic/`.
- The output metadata says `driver_mode = si03`.
- No output row requires CEC for the main mechanistic run.

---

## Self-Review

Spec coverage:

- Uses SI-S03 for dissolution/water conductivity: covered by Tasks 1 and 3.
- Aligns our frame times to SI-S03 time data: covered by Task 1 and Task 3.
- Does not use CEC/Waxman-Smits for virtual/imaginary response: covered by Tasks 2 and 3.
- Keeps our mechanistic model: covered by Schwarz helper and AC2D sweep updates.
- Calibrates `Sigma_s` and characteristic length: covered by Task 4.
- Keeps paper CEC route only as diagnostic: covered by Tasks 3 and 6.

No placeholders remain. The old CEC path is explicitly retained only as `paper-cec` diagnostic mode so previous results remain reproducible without being confused with the new mechanistic model.
