# pore_scale_electrical_2d — 二维孔隙尺度 SIP 求解器（AC2D-periodic）

本目录是 Niu et al. (2020) 三维 AC3D 框架（`code/src/pore_scale_electrical/`）的二维改写，
独立成包，不修改原三维代码。控制方程与机制定义保持不变，只把离散与几何量从三维换成二维。

## 控制方程（与 AC3D 相同）

对给定宏观电场 `E0`（沿 x 或 y），求周期修正电势 `u`：

```
div( sigma*(x, omega) (E0 - grad u) ) = 0,      sigma_eff = <J> / E0
```

- 单元中心有限体积；面电导用半单元串联（调和）平均，并推广到非均匀矩形网格
  （`Grid2D(dx_m, dy_m)`），可在细喉道处局部加密。均匀网格时与 AC3D 完全一致
  （`tests/test_pore_scale_electrical_2d.py` 中与 `ac3d_solver.solve_ac3d` 对比，相对误差 < 1e-10）。
- `sigma = 0` 的单元（孔/膜机制下的绝缘固相）不进入未知数；每个连通导电分量锚定一个电势，
  与 AC3D active-domain 求解一致。
- 稀疏直接 LU（SuperLU，`MMD_AT_PLUS_A` 排序），每个频点一次分解，返回相对残差。

## 二维约定

二维结果按“每单位纵深”计，等价于沿 z 平移不变的无限挤出体。电导率 (S/m) 与动态孔径
`Lambda` (m) 都是强度量，因此 `sigma_eff` 与人为厚度无关。报告中应称为 **2D**，不是 2.5D 或 3D。

## 极化机制（`polarization.py`）

| 机制 | 方程（Niu 2020） | 二维实现 |
|---|---|---|
| 孔（Stern/EDL）极化 | Eq.17 `C_p* = Sigma_S iwt/(1+iwt)`, `tau = r^2/(2 D+)`；Eq.12 `Delta sigma = 2 C_p*/Lambda` | `pore_mode="regional"`：每个孔区域独立 `r_k`、`Lambda_k`；`pore_mode="uniform"`：Niu 三维原式（按孔面积加权的 `C*` + 全局 `Lambda`，所有水相同一增量） |
| 膜（孔喉）极化 | Titov Eq.19 `Z* = Z_dc[1 - eta(1 - (1-e^{-2s})/(2s))]`, `s = sqrt(i w tau_m)`, `tau_m = L^2/(4D)` | 喉道区域电导率取 `sigma_w Z_dc / Z*`，使该喉道阻抗严格等于 Titov `Z*` |
| 界面（Maxwell-Wagner）极化 | 水/固相直流电导 + `i w epsilon` | 同三维 |
| all | 各增量加入水相后一次场求解 | 同三维；不能把单机制有效电导率相加 |

机制隔离遵守 Niu Section 5.3：`pore`、`membrane` 固相电导为 0 且不含介电背景；
`interfacial` 只含直流 + 介电项；`all` 包含全部。

**为什么膜极化不用 Eq.12 上标度：** 二维中喉道电导按单位纵深计（S/m），`2(1/Z* - 1/Z_dc)/Lambda`
会得到 S/m²，必须引入任意厚度。改为在喉道区域内赋强度量 `sigma_w Z_dc/Z*`，与 Titov 喉道阻抗等价且无厚度依赖。

**regional 与 uniform 的区别：** regional 下每个区域对 `sigma_eff` 的贡献按该区域的场能
`int |E|^2` 加权（`Lambda_k` 取该区域的场加权动态孔径时，等价于壁面积分 `int C |E_t|^2 dl`）；
uniform 则把按体积加权的 `C*` 均匀加到所有水相。显式几何中电流受限于窄喉道时，两者结果差别很大，
必须在结果中说明采用哪一种。

## 动态孔径 Λ 的计算（2026-10-24 修正）

- `dynamic_pore_size_2d_curve`（推荐）：壁面积分沿解析给出的光滑孔壁计算——在孔壁内侧 2、4 个网格处插值切向电场，
  线性外推到壁面。对方形排列圆盘的 Rayleigh 精确解，a/h ≥ 80 时误差 ≤ 0.2%（a/h = 40 时 1.4%）。
  理想孔链由 `TwoPoreChain2D.wall_curves()` 提供壁面点；`run_ideal_two_pore_channel_sip2d.py` 已改用此函数。
- `dynamic_pore_size_2d`（旧）：在阶梯状网格面上积分，斜壁处有不随加密消失的偏差（圆盘阵列 Λ 偏大 34–39%），
  只保留作对照（结果表中的 `dynamic_pore_size_staircase_m` 列）。
- 已知局限：喉口处孔壁与喉壁成尖角，电场在角点奇异。孔体区域的 Λ_k 主要来自喉口附近的一小段孔壁，在 0.125 µm 网格上
  不收敛；短喉道（L = 2 µm）的全局 Λ 在喉口加密到 0.025 µm 时降低 3–16% 且继续减小，长喉道（L/w ≥ 30）不超过 1.6%。
  喉道反演中 δ、β 与 Λ 在同一网格上标定，结论（20/20 耦合判断）对加密不敏感。
  基准与检验见 `code/scripts/sip_simulation/benchmark_ac2d_rayleigh_stern.py` 与
  `results/benchmark_ac2d_rayleigh_stern_20261023/README_zh.md`。

## 其他模块

- `geometry.py`：二维动态孔径 `dynamic_pore_size_2d_curve`（光滑壁面，推荐）与 `dynamic_pore_size_2d`（阶梯网格面，旧）、
  非均匀网格 `graded_axis`、理想几何 `TwoPoreChain2D`（大孔–狭长喉道–小孔周期链）。
- `debye.py`：非负 Debye 分解（二阶差分 Tikhonov 平滑；可选 `i w eps_inf` 列吸收高频介电尾部），
  与 `code/scripts/sip_simulation/plot_sample16_sip_pore_distribution.py::fit_debye_distribution` 同一形式；
  半径换算 `r = sqrt(2 tau D+)`。`relaxation_decomposition` 可同时拟合多种核（`debye` 孔极化、`titov` 膜极化、
  `warburg` = Cole–Cole c = 0.5），每种核各自平滑、各自换算长度（孔半径 `r = sqrt(2 D tau)`，喉长 `L = sqrt(4 D tau_m)`）。
  **注意**：Titov 膜极化谱可以被一串 Debye 项几乎完美拟合（测试 `test_titov_membrane_shape_is_not_identifiable_from_debye_terms`），
  所以只凭谱形无法判断机制；混合核分解需要谱以外的先验信息。

## 使用

```python
from pore_scale_electrical_2d import Grid2D, solve_ac2d_periodic
grid = Grid2D.uniform(nx, ny, dx_m)
result = solve_ac2d_periodic(sigma_map, grid, direction="x")
result.effective_conductivity_s_m
```

完整示例：`code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py`
（参数：`configs/sip_simulation/ideal_two_pore_channel_sip2d_config.py`）。

## 测试

```powershell
Set-Location 'C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\code'
& 'C:\Users\imgw\.conda\envs\ml\python.exe' -m pytest tests/test_pore_scale_electrical_2d.py -q
```

覆盖：均匀介质、并联/串联层（非均匀网格）、绝缘基质中直通道（解析 `sigma_w w/Ly`、`Lambda = w`）、
孤立包裹体、与 AC3D 交叉验证、核函数与 `pore_scale_electrical.polarization` 逐值相等、
regional/uniform 增量、机制隔离规则、Debye 分解回收、理想几何面积。
