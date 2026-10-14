"""理想二维“大孔-狭长喉道-小孔”周期链 SIP 模拟参数（AC2D-periodic）。

可执行 Python 配置文件。运行入口：
``code/scripts/sip_simulation/run_ideal_two_pore_channel_sip2d.py``。
运行时本文件会被原样复制到 ``results/<run_name>/configs/``，
并在 ``provenance/run_manifest.json`` 中记录实际采用的数值。

模型维度：二维（每单位纵深；等价于沿 z 无限平移不变的挤出体）。
电导率 (S/m) 与动态孔径 Lambda (m) 都是强度量，二维结果不依赖人为厚度。
不是三维求解，也不是 2.5D。

参数来源分三类，必须区分：
1. Niu et al. (2020) Berea 论文参数（本项目 Niu 复现链路的基线）；
2. 用户指定参数（D+ = 3.8e-12 m2/s；Schwarz 1962; Nordsiek & Weller 2008;
   Zhang et al. 2018）；
3. 本次理想几何的设计选择（几何尺寸、网格、频带），其依据见
   NMR 项目 ``docs/ideal_two_pore_channel_nmr_sip.md`` 中的尺寸选取分析。
"""

from __future__ import annotations

CONFIG = {
    "geometry": {
        # 大孔（pore body B）半径。设计选择：NMR 孤立孔 T2 = 1/(2 rho2/R + 1/T2B) ≈ 1.2 s，
        # 仍低于体弛豫 3 s；SIP Stern 弛豫峰 f = D+/(pi R^2) ≈ 3.0e-3 Hz，位于 1e-4–1e5 Hz 模拟频带内。
        "big_radius_m": 20.0e-6,
        # 小孔（pore body S）半径。设计选择：孤立孔 T2 ≈ 0.43 s，与大孔 T2 相差约 2.8 倍
        # （连通后因喉道壁面汇作用进一步拉开）；SIP 峰 f ≈ 4.8e-2 Hz，与大孔峰相隔 1.2 个数量级。
        "small_radius_m": 5.0e-6,
        # 狭长喉道宽度 w（半宽 h = 0.05 um）。设计选择：NMR 中喉道壁面的扩散-弛豫长度
        # l = sqrt(D0 h / rho2) ≈ 4.5 um，使其对小孔 T2 的“汇”效应保持在可解释范围；
        # SIP 喉道 Stern 峰 f = D+/(pi h^2) ≈ 4.8e2 Hz，低于 Maxwell-Wagner 主导频段。
        "throat_width_m": 0.1e-6,
        # 喉道长度（孔壁到孔壁）。设计选择：长宽比 150（“狭长”）；Titov 膜极化
        # tau_m = L^2/(4D) ≈ 4.3e-2 s，膜极化特征频率与孔/喉 Stern 峰错开约 1 个数量级以上。
        "throat_length_m": 15.0e-6,
        # 大孔上下方的固相边距。设计选择：使相邻周期链在 y 方向被绝缘固相隔开（不额外连通）；
        # 只影响有效电导率按单胞面积的归一化与界面（Maxwell-Wagner）极化的固相电容通路。
        "solid_margin_m": 2.0e-6,
    },
    "grid": {
        # x 方向均匀单元、y 方向在喉道附近局部加密（喉道壁恰为单元面）。
        # 0.125 um 与 0.1 um 两套网格的直流有效电导率相差 0.1%（见 provenance 网格收敛表）。
        "coarse_step_m": 0.125e-6,
        # 喉道宽度方向单元数：w / 8 = 0.0125 um。
        "throat_cells": 8,
        # 加密带向外的几何增长比。
        "growth": 1.2,
    },
    "electrical": {
        # 水相电导率，Niu et al. (2020) Berea Table 1。
        "water_conductivity_s_m": 0.043,
        # 固相直流电导率。按 Niu Section 5.3 孔/膜机制固相取 0；界面机制同样取绝缘颗粒，
        # 使界面极化只来自介电常数差异。若需非零固相电导须另行记录来源。
        "solid_conductivity_s_m": 0.0,
        # 水/固相相对介电常数，Niu et al. (2020)。
        "water_relative_permittivity": 80.0,
        "solid_relative_permittivity": 7.0,
        "epsilon_0_f_m": 8.85e-12,
    },
    "polarization": {
        # 孔（Stern/EDL）极化表面电导 Sigma_S，Niu et al. (2020) Berea：1.3e-9 S
        # （0.025 C/m2 表面电荷 × Na+ 迁移率）。这里只作为极化幅值参数沿用。
        "surface_conductance_s": 1.3e-9,
        # 孔极化弛豫的扩散系数 D+：采用用户指定的 Stern 层反离子扩散系数 3.8e-12 m2/s
        # （Schwarz 1962; Nordsiek & Weller 2008; Zhang et al. 2018），与 Debye 分解后
        # r = sqrt(2 tau D+) 的换算保持同一数值，使正演与反算自洽。
        # 注意：Niu (2020) Berea 原文取 1.3e-9 m2/s；两者相差约 342 倍（半径换算相差约 18.5 倍）。
        "stern_diffusion_m2_s": 3.8e-12,
        # 膜（孔喉）极化 Titov 模型使用体相离子扩散系数，Niu et al. (2020)：1.3e-9 m2/s。
        # 膜极化是喉道内体相浓差极化，不是 Stern 层机制，因此不改用 3.8e-12。
        "membrane_diffusion_m2_s": 1.3e-9,
        # 膜极化率 eta，Niu et al. (2020) Berea：0.01。
        "membrane_polarizability": 0.01,
        # 喉道是否同时计入 Stern 孔极化（r = h，Lambda = 喉道动态孔径）。
        # 设为 True：二维显式几何中喉道壁同样带有双电层；若只想复现 Niu 网络中
        # “孔极化只用孔体半径”的做法，可改为 False 并另存结果。
        "throat_stern_polarization": True,
    },
    "sweep": {
        # 模拟频带与采样：1e-4–1e5 Hz，每十倍频 10 点（样品16 SIP 对比图默认横轴范围）。
        "frequency_min_hz": 1.0e-4,
        "frequency_max_hz": 1.0e5,
        "points_per_decade": 10,
        # 外加宏观电场方向：沿周期链（x）。
        "direction": "x",
        # 并行进程数（每个频点一次稀疏直接 LU）。
        "workers": 6,
        # 运行的机制组合：
        #   pore/all 的 regional = 每个区域独立的 Eq.12 增量（二维显式几何主结果）；
        #   uniform = Niu 三维原式（按孔面积加权的 C* 与全局动态孔径，所有水相同一增量），作对照。
        "runs": [
            {"name": "interfacial", "mechanism": "interfacial", "pore_mode": "regional"},
            {"name": "pore_regional", "mechanism": "pore", "pore_mode": "regional"},
            {"name": "pore_uniform_niu", "mechanism": "pore", "pore_mode": "uniform"},
            {"name": "membrane", "mechanism": "membrane", "pore_mode": "regional"},
            {"name": "all_regional", "mechanism": "all", "pore_mode": "regional"},
            {"name": "all_uniform_niu", "mechanism": "all", "pore_mode": "uniform"},
        ],
    },
    "debye": {
        # Debye 分解的弛豫频率网格 f0 = 1/(2 pi tau)：比数据频带两端各外扩一个数量级。
        "relaxation_frequency_min_hz": 1.0e-5,
        "relaxation_frequency_max_hz": 1.0e6,
        "bins": 111,
        # 二阶差分 Tikhonov 平滑（归一化尺度）。无噪声合成数据取较小值；只影响峰宽，不移动峰位。
        "regularization": 1.0e-3,
        # 包含 i w eps_inf 列，吸收高频介电（Maxwell-Wagner）尾部。
        "include_permittivity": True,
        # 半径换算 r = sqrt(2 tau D+) 所用 D+（用户指定）。
        "radius_diffusion_m2_s": 3.8e-12,
    },
}
