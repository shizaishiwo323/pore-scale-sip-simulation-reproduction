"""NMR 喉道扫参的 20 个算例（w × L）的二维 SIP 模拟参数：电学/电化学参数与 C3 完全相同（Niu et al., 2020）。

运行入口：``code/scripts/sip_simulation/run_throat_sweep_sip2d.py --config <本文件>``。

几何来源：NMR 项目 ``advanced_tools/ideal_two_pore_throat_sweep_nmr.py`` 的 WIDTHS_UM × LENGTHS_UM
（R_B = 20 µm，R_S = 5 µm，喉道宽 w ∈ {0.1, 0.5, 1, 2, 4} µm，壁到壁长 L ∈ {2, 5, 15, 30} µm）。
``geometry`` 中的 throat_width_m / throat_length_m 仅为占位，运行时逐个算例覆盖。

模型维度：二维（每单位纵深）。SIP 用沿喉道方向的周期链 … B – 喉道 – S – 喉道 – B …（见 C3 配置的说明）。

频段：只算实验室频段并各外扩一些（1e-3–1e5 Hz，每十倍频 5 点）；界面极化的高频段不在本扫参的目标内。
网格：粗网格 0.125 µm（与 C3 相同）；喉道宽方向至少 8 个单元，宽喉道（w/8 > 0.125 µm）时喉道内取 0.125 µm。

参数来源：Niu, Q., Zhang, C., & Prasad, M. (2020). JGR Solid Earth, 125, e2020JB020515（Table 1 与 Section 4.2、5.3）。
"""

from __future__ import annotations

CONFIG = {
    "geometry": {
        # 孔体与 C3 相同；throat_width_m / throat_length_m 为占位（C3 值），运行时由 sweep.widths_m × lengths_m 覆盖。
        "big_radius_m": 20.0e-6,
        "small_radius_m": 5.0e-6,
        "throat_width_m": 1.0e-6,
        "throat_length_m": 15.0e-6,
        # 大孔上下方的固相边距：使相邻周期链在 y 方向被绝缘固相隔开（设计选择）。
        # 只影响按单胞面积的归一化与固相电容通路（界面极化）。
        "solid_margin_m": 2.0e-6,
    },
    "grid": {
        # x 方向均匀 0.125 µm（与 C3 相同）；y 方向喉道内至少 throat_cells 个单元、且不粗于 0.125 µm。
        # 网格收敛：脚本的 --grid-check 对极端算例另算 0.0625 µm 粗网格的直流 F 与 Lambda。
        "coarse_step_m": 0.125e-6,
        "throat_cells": 8,
        "growth": 1.2,
    },
    "electrical": {
        # Niu (2020) Section 4.2 / Table 1：孔隙水电导率 0.043 S/m，介电常数 80 eps0。
        "water_conductivity_s_m": 0.043,
        # Niu (2020) Section 4.2：固相不导电（only i w eps_s）。
        "solid_conductivity_s_m": 0.0,
        "water_relative_permittivity": 80.0,
        # Niu (2020) Table 1：固相本征介电常数 7 eps0。
        "solid_relative_permittivity": 7.0,
        # Niu (2020) Table 1 注：eps0 = 8.85e-12 F/m。
        "epsilon_0_f_m": 8.85e-12,
    },
    "polarization": {
        # Niu (2020) Table 1：Sigma_S = 1.3e-9 S（0.025 C/m2 表面电荷密度 × Na+ 迁移率 5.19e-8 m2/(V s)）。
        "surface_conductance_s": 1.3e-9,
        # Niu (2020) Table 1：EDL 中 Na+ 扩散系数 D = 1.3e-9 m2/s（Nernst–Einstein 与上述迁移率一致）。
        # 孔极化 tau_p = r^2/(2D)（Eq. 18）与膜极化 tau_m = L^2/(4D)（Eq. 20）使用同一个 D。
        "stern_diffusion_m2_s": 1.3e-9,
        "membrane_diffusion_m2_s": 1.3e-9,
        # Niu (2020) Table 1：膜极化率 eta0 = 1%。
        "membrane_polarizability": 0.01,
        # 二维显式几何中喉道壁同样有双电层：region-resolved 算例中喉道按 r = h = w/2 计入 Stern 极化（设定，非推导）。
        "throat_stern_polarization": True,
        # Niu 原式（uniform）只用孔节点（pore body）的体积加权孔径分布（Eq. 9，Figure 4a），不含喉道。
        "uniform_psd_kinds": ["pore_body"],
    },
    "sweep": {
        # 实验室频段 1e-3–1e4 Hz 再向高频外扩一个数量级（Debye 分解需要的边缘）。每十倍频 5 点。
        "frequency_min_hz": 1.0e-3,
        "frequency_max_hz": 1.0e5,
        "points_per_decade": 5,
        "direction": "x",
        "workers": 6,
        # Niu (2020) Section 5.3 的单机制定义：
        #   interfacial：只赋直流电导与高频介电常数；
        #   pore / membrane：水相 sigma_w + Delta sigma*，固相为 0，不含介电背景；
        #   all：全部加入水相后一次求解，固相 i w eps_s。
        # *_niu    = Niu 原式：孔极化为全局均匀增量 2 C_p*/Lambda（主结果）；
        # *_regional = 显式几何变体：每个区域用自己的 r 与场加权 Lambda_k（用于反演讨论）。
        # 膜极化在两种情况下都按喉道局部的 Titov 阻抗赋值（二维中 Eq.12 的膜极化上标度会引入任意厚度）。
        # 本扫参只需三组：孔极化（区域分配，最干净的场加权检验）与两种“全部机理”（实测谱的替身）。
        "runs": [
            {"name": "pore_regional", "mechanism": "pore", "pore_mode": "regional"},
            {"name": "all_regional", "mechanism": "all", "pore_mode": "regional"},
            {"name": "all_niu", "mechanism": "all", "pore_mode": "uniform"},
        ],
        # 20 个算例（与 NMR 扫参一致）。
        "widths_m": [0.1e-6, 0.5e-6, 1.0e-6, 2.0e-6, 4.0e-6],
        "lengths_m": [2.0e-6, 5.0e-6, 15.0e-6, 30.0e-6],
        # 实验室频段（Debye 分解与幅值反算只用此频段）。
        "lab_band_hz": [1.0e-3, 1.0e4],
    },
    "debye": {
        # Debye 分解的弛豫频率网格：比数据频带两端各外扩一个数量级，每十倍频 10 点。
        "relaxation_frequency_min_hz": 1.0e-4,
        "relaxation_frequency_max_hz": 1.0e11,
        "bins": 151,
        "regularization": 1.0e-3,
        "include_permittivity": True,
        # 半径换算 r = sqrt(2 tau D) 与正演同用 Niu 的 D = 1.3e-9 m2/s。
        "radius_diffusion_m2_s": 1.3e-9,
    },
}
