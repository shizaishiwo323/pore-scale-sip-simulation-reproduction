"""Niu 2020 Berea 三维 CT 的“区域分配（regional）”SIP 多机制模拟参数。

运行入口：``code/scripts/sip_simulation/run_berea_regional_sip3d.py --config <本文件>``。
运行时本文件会被原样复制到 ``results/<run_name>/configs/``，实际解析值另存为 ``resolved_config.json``。

方法来源：二维算例 C3 的 regional 模式（``code/src/pore_scale_electrical_2d``，
结果包 ``results/c3_two_pore_niu2020_sip2d_20261014``）。三维改写见
``code/src/pore_scale_electrical/ac3d_regional.py``：

- 孔体区域 = pnextract ``_VElems`` 给出的每个孔体体素集合；
- 喉道区域 = 与 pnextract 喉道界面体素（``write_throats``）距离 ≤ L_t/2、且位于该喉道所连两孔之内的孔隙体素；
- 每个区域 k 有自己的 Stern 弛豫半径 r_k 和场加权动态孔径 Lambda_k，Eq.12 增量 2 C_p*(r_k)/Lambda_k 只加在本区域；
- 喉道区域另按 Titov 阻抗赋强度量电导率 sigma_w Z_dc / Z*（膜极化）。

模型维度：真实三维 CT（350^3 体素、不降采样）上的三维场求解，属于 3D。
但区域划分（尤其喉道段的空间范围）是本项目对 pnextract 结果的三维解释，不是 Niu 原文做法；
Niu 原文是全局均匀增量（uniform）。结论中必须注明“regional（区域分配）模式”。

参数来源：Niu, Q., Zhang, C., & Prasad, M. (2020). JGR Solid Earth, 125, e2020JB020515
（Table 1、Section 4.2、Section 5.3）。孔、孔喉尺寸全部来自本次 pnextract 提取，不使用论文 Figure 5 的孔径/喉长数据。
实验对比数据取论文工作簿 Figure7.xlsx（实部实验列）与 Figure8.xlsx（虚部实验列），只用于对比。
"""

from __future__ import annotations

CONFIG = {
    "input": {
        # 用户指定的输入：Niu 2020 Berea 分割体 TIFF（350 x 350 x 350，uint16）。
        # TIFF 字节序与 RAW 相反：TIFF 值 256 = RAW 标签 1 = 孔隙水，512 = RAW 标签 2 = 固相（已逐体素核对一致）。
        "tiff_path": "data/Niu 2020data/microCT_Berea.tiff",
        "pore_value": 256,
        "solid_value": 512,
        # Niu (2020) Section 4：体素边长 2.8 um；本次不降采样（full grid）。
        "voxel_size_m": 2.8e-6,
        "downsample": 1,
    },
    "pnextract": {
        # 使用项目原版 pnextract（code/vendor/pnextract/bin/pnextract.exe），内置默认提取参数，
        # 不追加 minRPore / medialSurfaceSettings（遵守 AGENTS.md 的 Niu Berea 默认 PNM 规则）。
        # 相对上一版只额外打开两个“输出”开关，不改变提取算法：
        #   write_elements true -> _VElems.raw：每个孔隙体素所属孔体编号（区域分配的孔体区域）；
        #   write_throats  true -> _throats.raw.gz：喉道界面体素（区域分配的喉道“种子”）。
        "exe": "code/vendor/pnextract/bin/pnextract.exe",
        "keywords": ["write_cnm true", "write_vtkNetwork true", "write_elements true", "write_throats true"],
        # pnextract 的 _throats 图像写出时整体偏移一个体素（写入 padded 场的 0..n-1 段），读入时平移回去；
        # 平移后喉道界面体素落在固相上的数目为 0（平移前为 41586），据此确认。
        "throat_image_shift_voxels": 1,
    },
    "regions": {
        # 喉道段长度取 pnextract link2 的 throat_length（喉道自身长度，不含两端孔体长度），
        # 区域为界面两侧各 L_t/2；L_t 同时用于 Titov tau_m = L_t^2/(4D)，使膜极化区长度与 tau_m 一致。
        "throat_half_extent_fraction_of_length": 0.5,
        # 孔体 Stern 弛豫半径 = pnextract node2 孔体内切半径；喉道 = link1 喉道内切半径
        # （与二维 C3 一致：大孔取 R_B，喉道取半宽 h，均为内切半径）。
        "pore_body_relaxation_radius": "node2.pore_radius_m",
        "throat_relaxation_radius": "link1.throat_radius_m",
        # 二维 C3 的 throat_stern_polarization = True：喉道壁同样有双电层，计入 Stern 极化。
        "throat_stern_polarization": True,
        # 与边界相连的 191 条 pnextract 喉道（pore1 = -1/0）同样建区域，只限于其唯一内部孔体之内。
        "include_boundary_throats": True,
        # 场能量过小（区域平均 |E|^2 < 该值 × E0^2，即孤立/死端孔）时 Lambda_k 不可靠，改用几何 2V/S；
        # 这些区域对 sigma_eff 的贡献 ∝ ∫|E|^2，本来就近似为 0。
        "min_mean_field_energy_fraction": 1.0e-8,
    },
    "electrical": {
        # Niu (2020) Section 4.2 / Table 1：孔隙水电导率 0.043 S/m，相对介电常数 80。
        "water_conductivity_s_m": 0.043,
        # Niu (2020) Section 4.2：固相不导电，只有 i w eps_s。
        "solid_conductivity_s_m": 0.0,
        "water_relative_permittivity": 80.0,
        # Niu (2020) Table 1：固相相对介电常数 7。
        "solid_relative_permittivity": 7.0,
        # Niu (2020) Table 1 注：eps0 = 8.85e-12 F/m。
        "epsilon_0_f_m": 8.85e-12,
    },
    "polarization": {
        # Niu (2020) Table 1：Sigma_S = 1.3e-9 S（0.025 C/m2 × Na+ 迁移率 5.19e-8 m2/(V s)）。
        "surface_conductance_s": 1.3e-9,
        # Niu (2020) Table 1：D = 1.3e-9 m2/s；孔极化 tau = r^2/(2D)（Eq.18），膜极化 tau_m = L^2/(4D)（Eq.20）。
        "stern_diffusion_m2_s": 1.3e-9,
        "membrane_diffusion_m2_s": 1.3e-9,
        # Niu (2020) Table 1：膜极化率 eta = 1%。
        "membrane_polarizability": 0.01,
    },
    "frequencies": {
        # 频率覆盖与实验一致：Figure7/8 实验频点 2e-3 Hz 至 8.5973e8 Hz（SIP 段 + 介电段，共 85 点）。
        # 为节省计算只取 10 个频点：先在 log 频率上等分 10 个目标，再取最近（log 距离）的实验频点，
        # 这样每个模拟点都能和同频实验点直接对比。
        "n_points": 10,
        "experiment_real_workbook": "data/Niu 2020data/Figure7.xlsx",
        "experiment_imag_workbook": "data/Niu 2020data/Figure8.xlsx",
    },
    "solver": {
        # 与既有 Berea full-grid 复现相同：周期边界、沿数组第 0 轴（项目约定的 "x"，即 TIFF 切片轴）加宏观电场。
        "drive_axis": 0,
        "direction_label": "x",
        # GPU BiCGSTAB + FFT-Poisson 预条件（参考电导 = 当前频率下水相基础电导）。
        # 区域分配下的极化增量只占水相电导的 ~1e-3，均值电流估计的误差与残差同阶，
        # 因此 rtol 取 1e-8（既有 uniform 复现为 1e-5）；同时输出二次精度的能量形式 sigma_eff 作收敛核对。
        "rtol": 1.0e-8,
        # 含介电项的机制（interfacial / all）在 w eps_w / sigma_w 超过阈值后，sigma'' 与 sigma' 同量级，
        # 1e-6 已给出 ~1e-6 的相对精度；而 1e-8 在 MHz 段 BiCGSTAB 出现停滞（2.2 MHz 点 12 分钟 >2400 次迭代未收敛），
        # 因此在该区段改用 1e-6。低频段（极化信号仅为 sigma' 的 1e-5 量级）仍保持 1e-8。
        "rtol_dielectric_dominated": 1.0e-6,
        "dielectric_ratio_threshold": 0.01,
        "dc_rtol": 1.0e-9,
        "maxiter": 4000,
        "residual_every": 25,
        # 同一机制内用上一频点的电势作初值（warm start），第一个频点用直流孔隙电势。
        "warm_start": True,
    },
    "runs": [
        # Niu (2020) Section 5.3 的单机制定义（与二维 C3 相同）：
        #   interfacial（图中 Dielectric）：水相 sigma_w + i w eps_w，固相 i w eps_s；无孔/膜极化。
        #   pore_regional（图中 EDL）：水相 sigma_w + 区域 Stern 增量，固相 0，无介电背景。
        #   membrane_regional（图中 Membrane）：孔体 sigma_w，喉道段 sigma_w Z_dc/Z*，固相 0，无介电背景。
        #   all_regional：全部增量加入水相后一次求解，固相 i w eps_s。
        {"name": "interfacial", "mechanism": "interfacial"},
        {"name": "pore_regional", "mechanism": "pore"},
        {"name": "membrane_regional", "mechanism": "membrane"},
        {"name": "all_regional", "mechanism": "all"},
    ],
}
