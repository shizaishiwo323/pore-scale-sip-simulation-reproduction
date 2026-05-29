# AC2D 使用 SI03 的机理型 SIP 修改计划

> **给后续执行代理的要求：** 实施本计划时应使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项执行。步骤使用 checkbox（`- [ ]`）格式便于追踪。

**目标：** 只把论文 SI-S03 作为孔隙水/酸液电导率 `sigma_w(t)` 的时间驱动；虚部 SIP 响应不使用论文的 CEC/Waxman-Smits 等效电路模型，而改用本项目自己的 AC2D 机理型极化模型。

**总体架构：** 将当前 “paper-driven” 工作流拆成两条明确路径：一条是保留给论文 CEC/Waxman-Smits 公式复核的诊断路径，另一条是新的 AC2D 主路径。新的主路径先把我们的帧时间映射到 SI-S03 的 `sigma_w(t)`，再用 Maxwell-Wagner 相对比和 Schwarz/Debye 表面极化计算复电导率。

**技术栈：** Python、NumPy、pandas、SciPy、pytest、matplotlib、现有 `pore_scale_electrical` 包。

---

## 当前分析

本轮修改应使用以下数据：

- 论文 SI-S03：`docs/validation/2024gl111271-sup-0004-data set si-s03.csv`
- 我们的帧时间：`data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/global_evolution_log.csv`
- 我们的二维界面图像：`data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/interface_images/timestep_*.png`

SI-S03 中包含 `Time (h)`、`Water conductivity (S/m)`、`Porosity` 和 `pH`。在新的 AC2D 机理模型里，SI-S03 只负责提供背景水相/酸液电导率 `sigma_w(t)`，不提供 CEC 极化项。

时间范围检查结果：

| 数据源 | 时间范围 | 电导率范围 |
| --- | ---: | ---: |
| SI-S03 | `0.000000-0.218667 h` | `0.0096907-0.2542 S/m` |
| 我们的 28 帧 | `0.000278-4.534444 h` | 需要插值和外推 |

推荐的 `sigma_w(t)` 映射规则：

1. 在 `0-0.218667 h` 内对 SI-S03 做线性插值。
2. 超过 SI-S03 最后一行后，从最后一个 SI-S03 电导率值线性外推到最终酸液电导率 `sigma_HCl = 0.44 S/m`。
3. 每一帧输出都保存映射后的 `sigma_w(t)`，保证后续图表和误差来源可追踪。

按上述规则得到的典型帧值：

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

当前代码中需要修正的关键点：`run_ac2d_microfluidic_sweep.py` 现在的 `paper` driver 会读取 SI01/SI02/SI03，并调用 `waxman_smits_water_increment(...)` 把 CEC/Waxman-Smits 等效电路项加入 AC2D。这不应作为新的机理型 AC2D 主路径。

另一个重要物理问题：当前 Schwarz 颗粒极化项使用整个方解石投影的等效半径作为 `r`。例如 frame 1 的 `r_eq ~= 6.19e-4 m`，若 `D = 1.3e-9 m2/s`，则：

```text
tau = r^2 / (2D) ~= 147 s
omega*tau at 2.5 Hz ~= 2300
```

这会让 2.5 Hz 远离 Debye 弛豫峰，虚部因子大约只有 `1/(omega*tau) ~= 4.3e-4`，导致 `sigma''` 很小。若希望 2.5 Hz 附近有显著 Schwarz 极化响应，特征长度不应取整个方解石颗粒半径，而应取局部界面粗糙度、扩散长度或喉道尺度。

以 `2.5 Hz` 为中心、`D = 1.3e-9 m2/s` 估算：

```text
tau_peak = 1 / (2*pi*2.5) = 0.0637 s
r_peak = sqrt(2*D*tau_peak) = 1.29e-5 m = 12.9 um
```

因此，新模型应加入一个可配置的 Schwarz 特征长度，默认先用 `10-20 um` 量级，而不是整个方解石颗粒半径。

---

## 文件结构

- 修改 `code/src/pore_scale_electrical/microfluidic_2d.py`
  - 增加 SI03-only 的溶液电导率驱动函数。
  - 增加不依赖 CEC 的 Schwarz/Debye 界面极化函数。
  - 保留 Waxman-Smits 函数，但标注为论文公式诊断，不作为默认 AC2D 路径。

- 修改 `code/scripts/run_ac2d_microfluidic_sweep.py`
  - 增加 `--driver-mode si03` 并设为新默认。
  - 将当前 CEC 路径改名为 `--driver-mode paper-cec` 或仅作为显式诊断模式。
  - 增加 `--schwarz-characteristic-length-m`、`--schwarz-sigma-s`、`--schwarz-diffusion-coefficient`。
  - 新结果输出到 `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`。

- 修改 `code/scripts/plot_ac2d_2p5hz_paper_comparison.py`
  - 默认读取新的机理型 AC2D 结果。
  - 图例标注为 `AC2D mechanistic: SI03 sigma_w + MW + Schwarz`。

- 修改 `code/scripts/plot_ac2d_microfluidic_paper_style.py`
  - 默认读取新的机理型结果和几何统计。

- 新建 `code/scripts/calibrate_ac2d_mechanistic_params.py`
  - 对 `Sigma_s` 和 `r_char` 做小规模参数扫描。
  - 使用 SI-S03 提供的 `sigma_w(t)`。
  - 可以拿 SI-S02 的 `sigma''(2.5 Hz)` 作为观测目标做量级校准，但不使用 CEC 公式。
  - 输出校准表和热力图。

- 修改 `code/tests/test_microfluidic_2d.py`
  - 增加 SI03-only 电导率映射测试。
  - 增加 Schwarz 特征长度行为测试。
  - 增加回归测试，确认新的 `si03` 主路径不依赖 SI02/CEC。

---

## Task 1：增加 SI03-only 电导率驱动

**文件：**
- 修改：`code/src/pore_scale_electrical/microfluidic_2d.py`
- 测试：`code/tests/test_microfluidic_2d.py`

- [ ] **Step 1：先写失败测试**

在 `code/tests/test_microfluidic_2d.py` 中加入：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_si03_solution_conductivity_driver_interpolates_and_extends
```

预期：因为 `si03_solution_conductivity_at_times` 还不存在，测试失败。

- [ ] **Step 3：实现 SI03-only 函数**

在 `code/src/pore_scale_electrical/microfluidic_2d.py` 中加入：

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

- [ ] **Step 4：运行测试，确认通过**

运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_si03_solution_conductivity_driver_interpolates_and_extends
```

预期：PASS。

---

## Task 2：增加机理型 Schwarz 界面极化

**文件：**
- 修改：`code/src/pore_scale_electrical/microfluidic_2d.py`
- 测试：`code/tests/test_microfluidic_2d.py`

- [ ] **Step 1：先写失败测试**

加入：

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

- [ ] **Step 2：运行测试，确认失败**

运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py::test_schwarz_interface_delta_peaks_when_omega_tau_is_one tests\test_microfluidic_2d.py::test_schwarz_interface_delta_uses_local_length_not_calcite_body_radius
```

预期：因为 `schwarz_interface_conductivity_delta` 不存在，测试失败。

- [ ] **Step 3：实现 Schwarz 极化函数**

在 `code/src/pore_scale_electrical/microfluidic_2d.py` 中加入：

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

- [ ] **Step 4：运行测试，确认通过**

运行同 Step 2 的 pytest 命令。

预期：PASS。

---

## Task 3：在 sweep 脚本中加入 `si03` 主路径

**文件：**
- 修改：`code/scripts/run_ac2d_microfluidic_sweep.py`

- [ ] **Step 1：修改 CLI 默认值**

把默认输出目录改为：

```python
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1"
DEFAULT_MECHANISMS = ["maxwell", "grain", "all"]
```

把：

```python
parser.add_argument("--driver-mode", choices=["paper", "constant"], default="paper")
```

改成：

```python
parser.add_argument("--driver-mode", choices=["si03", "paper-cec", "constant"], default="si03")
parser.add_argument("--schwarz-sigma-s", type=float, default=7.0e-5)
parser.add_argument("--schwarz-characteristic-length-m", type=float, default=1.3e-5)
parser.add_argument("--schwarz-diffusion-coefficient", type=float, default=1.3e-9)
```

其中 `paper-cec` 只保留旧的 CEC/Waxman-Smits 诊断行为。

- [ ] **Step 2：实现 `si03` 分支**

在逐帧循环中加入：

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

把 `schwarz_delta` 传入 `mechanism_phase_conductivities(...)` 的 `interface_delta_conductivity_s_m` 参数。后续也可以把这个参数重命名为 `mechanistic_interface_delta_conductivity_s_m`，语义会更清楚。

- [ ] **Step 3：保存可追踪 metadata**

metadata 中加入：

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

- [ ] **Step 4：运行 smoke test**

运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\run_ac2d_microfluidic_sweep.py --frames 1 10 20 28 --frequencies 2.5 --downsample 16
```

预期：

- 写入 `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/ac2d_sweep_results.csv`。
- CSV 中包含 `solution_sigma_w_s_m` 或类似字段。
- `driver_mode=si03` 时不需要 `paper_cec_meq_g`。
- `all` 表示 Maxwell-Wagner + 机理型 Schwarz 界面/颗粒极化。

---

## Task 4：增加机理参数标定脚本

**文件：**
- 新建：`code/scripts/calibrate_ac2d_mechanistic_params.py`

- [ ] **Step 1：创建脚本**

脚本用途：

- 扫描 `Sigma_s` 和 `r_char`。
- 每组参数调用 `run_ac2d_microfluidic_sweep.py`。
- 用 SI-S02 的 `sigma''(2.5 Hz)` 只作为观测目标计算误差，不使用 CEC 公式。
- 保存校准表和热力图。

建议扫描范围：

```python
sigma_s_values = np.logspace(-6, -3, 13)
length_values = np.array([5e-6, 8e-6, 1.0e-5, 1.3e-5, 2.0e-5, 3.0e-5, 5.0e-5])
```

每组参数运行：

```powershell
--driver-mode si03 --mechanisms maxwell grain all --frequencies 2.5 --downsample 16 --schwarz-sigma-s <value> --schwarz-characteristic-length-m <value>
```

输出：

- `results/ac2d_microfluidic/mechanistic_calibration_v1/calibration_grid.csv`
- `figures/ac2d_microfluidic/ac2d_mechanistic_calibration_grid.png`

- [ ] **Step 2：运行校准**

运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\calibrate_ac2d_mechanistic_params.py
```

预期：

- 校准表包含 `sigma_s_s`、`characteristic_length_m`、`tau_s`、`imag_mae_s_m`、`real_mae_s_m`。
- 如果目标是 2.5 Hz 虚部量级，最优特征长度应偏向 `10 um` 量级，而不是 `0.6 mm` 的颗粒整体半径。

---

## Task 5：更新图表默认路径和标签

**文件：**
- 修改：`code/scripts/plot_ac2d_2p5hz_paper_comparison.py`
- 修改：`code/scripts/plot_ac2d_microfluidic_paper_style.py`

- [ ] **Step 1：默认读取新的机理型结果**

设置：

```python
DEFAULT_OURS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_sweep_results.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_sweep_results.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "geometry_metrics.csv"
```

输出文件名：

```python
ac2d_2p5hz_si03_mechanistic_vs_paper.png
ac2d_microfluidic_paper_style_si03_mechanistic.png
```

- [ ] **Step 2：更新图例**

使用：

```text
Paper SI-S02 measurement
AC2D mechanistic: SI-S03 sigma_w(t) + MW + Schwarz
```

不要把新曲线命名为 `Waxman-Smits`、`CEC` 或 `paper-consistent`。

- [ ] **Step 3：生成图表**

运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_2p5hz_paper_comparison.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_microfluidic_paper_style.py
```

预期输出：

- `figures/ac2d_microfluidic/ac2d_2p5hz_si03_mechanistic_vs_paper.png`
- `figures/ac2d_microfluidic/ac2d_microfluidic_paper_style_si03_mechanistic.png`

---

## Task 6：更新项目说明

**文件：**
- 修改：`AGENTS.md`
- 如 `README.md` 已写 AC2D 结果，也同步修改。

- [ ] **Step 1：区分两个模型**

加入如下说明：

```text
AC2D SI03 机理模型：
- 只用 SI-S03 提供 sigma_w(t)。
- 不使用 CEC 或 Waxman-Smits 作为源项。
- 虚部响应来自 Maxwell-Wagner 相对比和 Schwarz/Debye 界面极化。

论文 CEC 诊断模型：
- 用于复算论文经验/等效电路模型。
- 只作为参考检查。
- 不是本项目的机理型 AC2D 结果。
```

- [ ] **Step 2：记录默认参数**

记录：

```text
D = 1.3e-9 m2/s
r_char default = 1.3e-5 m
tau = r_char^2 / (2D) ~= 0.065 s
2.5 Hz omega*tau ~= 1
Sigma_s initial scan range = 1e-6 to 1e-3 S
```

---

## 验证命令

全部修改完成后运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_microfluidic_2d.py tests\test_ac2d_solver.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\run_ac2d_microfluidic_sweep.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_2p5hz_paper_comparison.py
C:\Users\imgw\.conda\envs\ml\python.exe code\scripts\plot_ac2d_microfluidic_paper_style.py
```

预期：

- 测试通过。
- 新结果位于 `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`。
- 新图位于 `figures/ac2d_microfluidic/`。
- 输出 metadata 中显示 `driver_mode = si03`。
- 主机理模拟结果不依赖 CEC。

---

## 自检

本计划覆盖了：

- 使用 SI-S03 作为溶解/水相电导率时间驱动。
- 将我们的帧时间映射到 SI-S03 时间轴。
- 不使用 CEC/Waxman-Smits 计算虚部。
- 保留我们的 Maxwell-Wagner + Schwarz/Debye 机理模型。
- 对 `Sigma_s` 和特征长度进行参数扫描和合理标定。
- 将论文 CEC 路线保留为诊断模型，避免和机理模型混淆。
