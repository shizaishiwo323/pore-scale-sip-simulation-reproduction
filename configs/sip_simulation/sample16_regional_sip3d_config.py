"""样品 16（LKC 鲕模粒灰岩 / G3）三维 CT 区域分配（regional）SIP 多机制模拟参数。

运行入口：``code/scripts/sip_simulation/run_sample16_regional_sip3d.py --config <本文件>``；
运行时本文件被复制到 ``results/<run_name>/configs/``，解析值另存为 ``resolved_config.json``。

不再沿用 Niu (2020) 的硅质砂岩参数。每个参数的来源标在注释里，分三类：
  [测] 本样品实测（F. Zhang 硕士论文 / 附录、样品属性表、NMR 原始谱、CT）；
  [文] 文献中方解石 / 碳酸盐的值（10 mM NaCl 附近，与本样品盐水最接近）；
  [推] 由上两类数据推出的量（推导方式写在注释里），不对 SIP 实验做拟合。

物理图像（两尺度）：
  - CT 可分辨孔隙（rev336 的 19.1%，以鲕模孔为主）：区域分配 —— pnextract 孔体 / 喉道段各自的 Stern 极化，
    喉道段 Titov 膜极化；
  - CT 中的“固相”= 方解石 + 未分辨的微孔基质（He 孔隙度 26.4% − CT 19.1%）：赋予导电且可极化的等效复电导率。
    鲕模灰岩的大孔多不连通（本样品 CT 孔相在 Colgin/James 的研究中未贯通；rev336 只沿 x 贯通），
    电流必须穿过微孔基质，所以基质不能按 Niu 的绝缘固相处理。
"""

from __future__ import annotations

CONFIG = {
    "input": {
        # [测] rev336 = 5-16seged[z328:664, y200:536, x8:344] 的孔相（项目审计已核对），336³，孔隙度 19.1%，
        # 与整块 CT 孔隙度一致；它是项目一直使用的样品 16 REV。0 = CT 可分辨孔隙，255 = 固相（含未分辨微孔基质）。
        "tiff_path": "data_inventory/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/CT_slices/1-CTseg/rev336.tiff",
        "pore_value": 0,
        "solid_value": 255,
        "title": "sample16_rev336",
        # [测] 10x 物镜，Xradia MicroXCT-400，体素 1.92 µm（Kansas 样品信息表、Colgin 答辩 slide 8、James 论文 Table 1）。
        "voxel_size_m": 1.92e-6,
        "downsample": 1,
    },
    "pnextract": {
        # 项目原版 pnextract，内置默认提取参数；只打开 VElems / throats 两个输出开关（区域分配需要）。
        "exe": "code/vendor/pnextract/bin/pnextract.exe",
        "keywords": ["write_cnm true", "write_vtkNetwork true", "write_elements true", "write_throats true"],
    },
    "regions": {
        # 与 Berea 三维区域分配相同的规则（见 configs/sip_simulation/berea_regional_sip3d_config.py）。
        "throat_half_extent_fraction_of_length": 0.5,
        "throat_stern_polarization": True,
        "include_boundary_throats": True,
        "min_mean_field_energy_fraction": 1.0e-8,
    },
    "experiment": {
        # [测] PSIP 原始文件；阻抗 Z = 10 kΩ × Magnitude × exp(i Phase)，σ* = (L/A)/Z（项目既有换算，
        # run_sample89_3d_sip_comparison.py），L = 3.6322 cm、D = 2.52476 cm（样品属性表）。
        # 注意：该文件未记录盐水电导率；其 σ′(1 Hz) = 5.8e-3 S/m 与 σw = 0.12 S/m、F = 54.35 不自洽（应为 ~2.2e-3），
        # 同日重复测量 LKC-16-1.csv 也一样，因此可能是另一盐度下的测量。σ″ 随盐度变化弱，是更可靠的对比量；
        # 相位与几何因子无关。
        "psip_csv": "data_inventory/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/SIP/LKC-16.csv",
        "current_resistor_ohm": 1.0e4,
        "sample_length_cm": 3.6322,
        "sample_diameter_cm": 2.52476,
        # [测] 硕士论文 Table 4.2（σw = 0.12 S/m NaCl、24 ℃ 的样品 16 特征值，1 Hz）。
        "thesis_point": {"frequency_hz": 1.0, "sigma_real_s_m": 2.57e-3, "sigma_imag_s_m": 3.25e-6},
        # [测] 论文只画 ≤ 1 kHz（避免高频耦合）；更高频的实验点只作参考。
        "reliable_max_frequency_hz": 1.0e3,
    },
    "electrical": {
        # [测] 饱和液 NaCl，σw = 0.12 S/m，24 ℃（硕士论文 p.41；约 10 mM NaCl）。
        "water_conductivity_s_m": 0.12,
        # [文] 24 ℃ 稀 NaCl 水的相对介电常数 ≈ 78.4（纯水 25 ℃ 78.4；10 mM 的盐效应 < 1%）。
        "water_relative_permittivity": 78.4,
        # [文] 方解石单晶静态介电常数各向同性平均 ≈ 8.3（Young & Frederikse 1973, J. Phys. Chem. Ref. Data 2, 313）。
        "calcite_relative_permittivity": 8.3,
        "epsilon_0_f_m": 8.854e-12,
    },
    "porosity": {
        # [测] He 孔隙度 26.40%（商业实验室，论文 Table 4.2）；NMR 孔隙度 23.88%（Table 4.2/4.3）。
        "helium": 0.264,
        "nmr": 0.2388,
    },
    "matrix": {
        # [推] 基质直流电导率 σm：唯一的标定量，标定目标是独立实测的地层因子 F = 54.35
        # （论文用 4 个盐度拟合，不含表面电导），即直流纯体积导电下 σ_eff = σw / F。
        # 标定后检查隐含的微孔胶结指数 m_µ = ln(σw/σm) / ln(1/φ_µ)，φ_µ = (φ_He − φ_CT)/(1 − φ_CT)。
        "formation_factor": 54.35,
        "initial_sigma_m_s_m": 1.3e-3,
        "calibration_rtol": 2.0e-3,
        "max_calibration_solves": 4,
        # [推] 基质介电常数：方解石 + 微孔水按 CRIM 混合（√ε_m = (1−φ_µ)√ε_calcite + φ_µ√ε_w）。
        "permittivity_mixing": "CRIM",
    },
    "nmr": {
        # [测] 样品 16 仪器反演谱（Magritek 2 MHz，CPMG，TE = 0.6 ms，200 个分箱，0.1–12000 ms）。
        "spectrum_csv": "data_inventory/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/NMR_combined/16/T2CPMG/Spectra/spectrum.csv",
        "echo_time_s": 6.0e-4,
        # [推] 微孔 = T2 谱中最短的 (1 − φ_CT/φ_NMR) 份额（CT 看不到的那部分孔隙）；
        # 截止 T2_c 对应 CT 分辨边界半径 r_c = 1 个体素 = 1.92 µm，于是 ρ2 = r_c / (3 T2_c)（球形孔，论文 Eq.3 的 a = 3）。
        # 得到的 ρ2 与论文给出的灰岩 ρ2（平均 2.8、样品表 4.21 µm/s）同量级，作为独立核对。
        "ct_resolution_radius_m": 1.92e-6,
        "shape_factor_a": 3.0,
        # [推] T2 < 3 TE 的分箱由 0.6 ms 回波间隔无法可靠分辨（论文也指出 < 0.02 ms 信号丢失），不计入微孔表面。
        "min_reliable_t2_echo_multiple": 3.0,
    },
    "polarization": {
        # 情景网格（全部取文献 / 实测范围内的值，用线性响应快速计算，再选出情景做完整三维求解）：
        # Σ_S [文] 方解石 Stern 层电导：Li et al. 2016 (JCIS 468, 262) 10 mM NaCl 拟合 1.2–2.2 nS（pH 8 约 1.8 nS）；
        #           Leroy et al. 2017 (GJI 209, 123) 约 3.2 nS。
        "stern_conductance_s": [1.2e-9, 1.8e-9, 2.2e-9, 3.2e-9],
        # D_+ [文] Stern 层反离子扩散系数（Nernst–Einstein 由迁移率换算）：Li 2016 拟合 5.1e-10；
        #           Leroy 2017 假设 β = 0.1 体相迁移率 → 1.46e-10。1.3e-9（硅质 / 体相 Na+）只作对照，不是方解石值。
        "stern_diffusion_m2_s": [1.46e-10, 5.1e-10, 1.3e-9],
        # ρ2 [推/测]：CT–NMR 分辨边界推得的值为中心；论文灰岩平均 2.8 µm/s、样品表 4.21 µm/s 为敏感性。
        "rho2_um_s": ["ct_boundary", 2.8, 4.21],
        # 膜极化 [推]：文献没有碳酸盐 η。按窄孔（Titov / Marshall–Madden）膜极化图像，η 由窄孔的离子选择性决定，
        # 用 Dukhin 数估计：η = Du / (1 + Du)，Du = Σ_d (S/V) / σw，Σ_d = 扩散层电导（Li 2016：0.15–0.4 nS，网格取三值）。
        #   CT 喉道：S/V = 2 / r_t（圆柱），r_t 为 pnextract 喉道内切半径 —— µm 级喉道的 Du ~ 1e-3，膜极化很弱；
        #   微孔基质“桥”（鲕模孔之间的微孔方解石）：S/V = NMR 微孔体积平均 S/V —— Du ~ 0.03，选择性明显。
        # 校核：用 Stern + 扩散层总电导、按 η ∝ Du² 估计，同样得到 η ≈ 0.025，两种估计一致。
        # 文献支持：Zhang, Niu & Zhang (SEG 2018, 同批 LKC 样品) 需要加入膜极化才能拟合鲕模粒灰岩。
        # τ_m = L² / (4 D_w)，D_w = 体相 Na+ 扩散系数 1.33e-9 m²/s（膜极化是孔隙水中的浓差扩散）；
        #   CT 喉道 L = pnextract 喉长；基质桥 L = 基质局部厚度（CT 固相最大内切球直径，逐体素）。
        "diffuse_layer_conductance_s": [1.5e-10, 3.0e-10, 4.0e-10],
        "membrane_water_diffusion_m2_s": 1.33e-9,
        "matrix_membrane": True,
        # 完整三维求解的中心情景：Σ_S 1.8 nS（Li 2016 pH 8）、D 5.1e-10（Li 2016 拟合）、ρ2 = CT–NMR 边界值、Σ_d 0.3 nS。
        "central": {"stern_conductance_s": 1.8e-9, "stern_diffusion_m2_s": 5.1e-10, "rho2_um_s": "ct_boundary",
                    "diffuse_layer_conductance_s": 3.0e-10},
    },
    "variants": {
        # 第二轮（用户要求）：加入直流表面导电，并分别按论文盐水和按实验文件自身的电导率计算。
        # 直流表面导电 [推/文]：Stern 层与扩散层在直流下都导电（Revil 类模型：σ0 含 Stern 贡献，极化项另加在其上），
        #   Σ_DC = Σ_S + Σ_d；与 Stern 增量相同的几何换算：CT 区域 2Σ_DC/Λ_k，基质 (σm/σw) Σ_DC Σ_j w_j (S/V)_j。
        #   它是所有机制共同的直流背景，不是新的极化机制；F = 54.35 是不含表面电导的本征地层因子，σm 标定不变。
        # 盐水：
        #   "thesis"：σw = 0.12 S/m（论文记录）；
        #   "csv"：按 LKC-16.csv 自身电导率反推 σw —— 令模型 σ′(0.1 Hz) 等于该文件 0.1 Hz 的 σ′（全场直流求解 + 线性响应极化实部）。
        #     基质、孔隙水的体积电导率同比例缩放（F 与盐度无关）；Σ_S、Σ_d、D 仍取 10 mM 的文献值（盐度依赖未建模，记为假设）；
        #     膜极化 η 随 Du = Σ_d (S/V) / σw 自动变化。
        "sternDC_brine_thesis": {"stern_dc_conduction": True, "brine": "thesis"},
        "sternDC_brine_csv": {"stern_dc_conduction": True, "brine": "csv"},
    },
    "frequencies": {
        # 实验频带 0.1 Hz–20 kHz；在 0.1 Hz–12.3 kHz（去掉 20 kHz 异常点：幅值跳变、相位变号）上取 10 个实验频点。
        "n_points": 10,
        "min_hz": 0.1,
        "max_hz": 1.3e4,
        "dense_points": 241,
    },
    "solver": {
        # 沿数组第 0 轴（CT 切片方向，假定为岩心轴向，即 SIP 测量方向）加场；周期边界。
        "drive_axis": 0,
        "rtol": 1.0e-8,
        "dc_rtol": 1.0e-9,
        "maxiter": 4000,
        "residual_every": 25,
        "warm_start": True,
    },
}
