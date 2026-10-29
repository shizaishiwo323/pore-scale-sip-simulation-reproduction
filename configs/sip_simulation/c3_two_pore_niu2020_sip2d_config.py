"""算例 C3（大孔–喉道–小孔）的二维 SIP 模拟参数：全部电学/电化学参数取 Niu et al. (2020)。

运行入口：``code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py --config <本文件>``。
运行时本文件会被原样复制到 ``results/<run_name>/configs/``。

几何来源：NMR 项目 ``NMR-SIP/docs/coupled_pore_backcalc/c3_intuitive`` 的算例 C3
（R_B = 20 µm，R_S = 5 µm，喉道宽 w = 1 µm，长 L = 15 µm，孔壁到孔壁）。

模型维度：二维（每单位纵深）。电导率与动态孔径 Lambda 都是强度量，结果不依赖厚度。

SIP 需要贯通的导电路径。C3 是封闭的“大孔–喉道–小孔”，因此沿喉道方向施加外电场，
等价于把 C3 沿喉道方向周期延拓：… B – 喉道 – S – 喉道 – B …（每个周期两段相同喉道）。
这与“过两孔圆心的两个对称面为等势面、只算一个 C3 单元”完全等价。

参数来源：Niu, Q., Zhang, C., & Prasad, M. (2020). A framework for pore-scale simulation of
effective electrical conductivity and permittivity of porous media in the frequency range from
1 mHz to 1 GHz. JGR Solid Earth, 125, e2020JB020515（Table 1 与 Section 4.2、5.3）。
"""

from __future__ import annotations

CONFIG = {
    "geometry": {
        # 算例 C3 的几何（用户提供）。
        "big_radius_m": 20.0e-6,
        "small_radius_m": 5.0e-6,
        "throat_width_m": 1.0e-6,
        "throat_length_m": 15.0e-6,
        # 大孔上下方的固相边距：使相邻周期链在 y 方向被绝缘固相隔开（设计选择）。
        # 只影响按单胞面积的归一化与固相电容通路（界面极化）。
        "solid_margin_m": 2.0e-6,
    },
    "grid": {
        # 均匀 0.125 µm 单元：喉道宽 1 µm 跨 8 个单元，小孔半径跨 40 个单元。
        # 运行时另算 0.25 / 0.125 / 0.1 µm 三套网格的直流有效电导率作为网格收敛检查。
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
        # 二维显式几何中喉道壁同样有双电层：region-resolved 算例中喉道按 r = h = 0.5 µm 计入 Stern 极化。
        "throat_stern_polarization": True,
        # Niu 原式（uniform）只用孔节点（pore body）的体积加权孔径分布（Eq. 9，Figure 4a），不含喉道。
        "uniform_psd_kinds": ["pore_body"],
    },
    "sweep": {
        # Niu (2020) 频带 1e-3–1e9 Hz；为完整显示界面极化高频段，扩展到 1e10 Hz。每十倍频 8 点。
        "frequency_min_hz": 1.0e-3,
        "frequency_max_hz": 1.0e10,
        "points_per_decade": 8,
        "direction": "x",
        "workers": 6,
        # Niu (2020) Section 5.3 的单机制定义：
        #   interfacial：只赋直流电导与高频介电常数；
        #   pore / membrane：水相 sigma_w + Delta sigma*，固相为 0，不含介电背景；
        #   all：全部加入水相后一次求解，固相 i w eps_s。
        # *_niu    = Niu 原式：孔极化为全局均匀增量 2 C_p*/Lambda（主结果）；
        # *_regional = 显式几何变体：每个区域用自己的 r 与场加权 Lambda_k（用于反演讨论）。
        # 膜极化在两种情况下都按喉道局部的 Titov 阻抗赋值（二维中 Eq.12 的膜极化上标度会引入任意厚度）。
        "runs": [
            {"name": "interfacial", "mechanism": "interfacial", "pore_mode": "uniform"},
            {"name": "pore_niu", "mechanism": "pore", "pore_mode": "uniform"},
            {"name": "membrane", "mechanism": "membrane", "pore_mode": "uniform"},
            {"name": "all_niu", "mechanism": "all", "pore_mode": "uniform"},
            {"name": "pore_regional", "mechanism": "pore", "pore_mode": "regional"},
            {"name": "all_regional", "mechanism": "all", "pore_mode": "regional"},
        ],
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
