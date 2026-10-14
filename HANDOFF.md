# HANDOFF — SIP 项目（pore-scale-sip-simulation-reproduction）

更新时间：2026-10-14 04:43（本机时间，UTC+01:00）
更新人：Claude Code 会话“NMR考虑耦合的知识分解”，汇总了“NMR/SIP二维几何模拟与反演”会话中 SIP 部分的工作

## 1. 最近的工作：二维孔隙尺度 SIP 求解器

将 Niu et al. (2020) 的三维 AC3D 框架（`code/src/pore_scale_electrical/`）改写为二维，**独立成包**，不修改原三维代码。
控制方程和机制定义不变，只把离散与几何量换成二维。详细说明见 `code/src/pore_scale_electrical_2d/README.md`。

| 文件 | 作用 |
|---|---|
| `code/src/pore_scale_electrical_2d/solver.py` | `Grid2D`（可非均匀矩形网格）、`solve_ac2d_periodic`（单元中心有限体积，调和面电导，active domain，每个连通分量锚定电势，SuperLU） |
| `code/src/pore_scale_electrical_2d/polarization.py` | 孔（Stern）极化 `regional` / `uniform`（Niu 原式）、Titov 膜（喉道）极化、Maxwell–Wagner 界面极化；`SIP2DParameters` |
| `code/src/pore_scale_electrical_2d/geometry.py` | 二维动态孔径 Λ、`graded_axis` 局部加密、理想几何 `TwoPoreChain2D`（大孔–狭长喉道–小孔周期链） |
| `code/src/pore_scale_electrical_2d/debye.py` | 非负 Debye 分解（二阶差分 Tikhonov，可选 ε∞ 列），半径换算 r = √(2τD⁺) |
| `code/tests/test_pore_scale_electrical_2d.py` | 10 个测试，含与 AC3D 均匀网格的对照（相对误差 < 1e-10） |
| `configs/sip_simulation/ideal_two_pore_channel_sip2d_config.py` | 理想几何算例配置（中文注释） |
| `code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py` | 运行脚本：`--output-dir`、`--smoke`、`--geometry-json`、`--workers` |

复现（在本仓库根目录）：

```bash
python code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py --output-dir results/ideal_two_pore_channel_sip2d_20261012
python -m pytest -q code/tests/test_pore_scale_electrical_2d.py
```

2026-10-14 复跑测试：10 passed。

## 2. 关键设定与注意事项

- 几何：B（R = 20 µm）– 喉道（w = 0.1 µm，L = 15 µm）– S（R = 5 µm）的周期链，固相边距 2 µm；
  喉道处 y 向局部加密（喉宽跨 8 个单元）。结果为 **2D（每单位纵深）**。
- 参数：σ_w = 0.043 S/m，Σ_S = 1.3e-9 S，D⁺(Stern) = 3.8e-12 m²/s，膜极化 D = 1.3e-9 m²/s，η = 0.01，ε = 80 / 7。
  D⁺ 取自文献（Schwarz, 1962; Nordsiek & Weller, 2008; Zhang et al., 2018），不是实测值。
- 二维中膜极化不用 Eq.12 的 2(1/Z* − 1/Z_dc)/Λ，否则会引入任意厚度。改为在喉道区域赋强度量 σ_w·Z_dc/Z*，与 Titov 喉道阻抗等价。
- `regional` 与 `uniform` 两种孔极化模式，在喉道限流时结果差别很大。报告中必须写明采用哪一种。

## 3. 结果与 NMR 联合分析

- 运行结果 `results/ideal_two_pore_channel_sip2d_20261012/` 按本仓库约定只保留在本地（`.gitignore` 忽略 `results/`）。
- 图件和汇总数据的快照、NMR–SIP 联合分析（孤立孔解析解、耦合孔反算、通俗版讲解、PPT），都在 NMR 仓库的 **`NMR-SIP/`** 目录
  （https://github.com/shizaishiwo323/NMR-simulation ，见 `NMR-SIP/README.md` 与该仓库的 `HANDOFF.md`）。
- NMR 侧入口 `advanced_tools/ideal_two_pore_channel_nmr_sip.py` 会读取（必要时调用）本仓库的上述运行脚本，两个仓库需要在本机并列放置。

## 4. 本次提交到 main 的内容

- `code/src/pore_scale_electrical_2d/`、`code/tests/test_pore_scale_electrical_2d.py`、
  `configs/sip_simulation/ideal_two_pore_channel_sip2d_config.py`、`code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py`；
- 本文件。

**故意未提交**：`pnextract/Makefile`、`pnextract/README.md`（另一条工作线的未跟踪文件，状态不明）。

## 5. 环境注意事项

- Python 使用 `C:/Users/imgw/.conda/envs/t2agent/python.exe` 并设置 `PYTHONNOUSERSITE=1`，否则用户目录下的 SciPy 与 NumPy 2 不兼容。
- 仓库路径含中文，打印路径时设置 `PYTHONIOENCODING=utf-8`。
