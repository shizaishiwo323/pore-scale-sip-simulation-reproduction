# Niu 2020 膜极化引用文献审计：几何、电化学参数与代码改进

日期：2026-06-17

## 本次整理

用户补齐的 PDF 已统一重命名到：

`docs/references/niu2020_membrane_polarization_refs_20260617/`

命名规则为 `年份_作者_DOI.pdf`，便于按年代、作者和 DOI 检索。全部 PDF 也已提取文本到同目录的 `extracted_text/`，用于后续全文检索。

| 文件名 | 作用 |
|---|---|
| `1959_marshall_madden_1959_10.1190_1.1438659.pdf` | 膜极化/扩散耦合源头模型 |
| `1992_delima_sharma_10.1190_1.1443257.pdf` | 页岩砂中 Maxwell-Wagner 与双电层/膜极化的等效介质处理 |
| `1999_chelidze_gueguen_10.1046_j.1365-246x.1999.00799.x.pdf` | 多种岩石电谱模型综述 |
| `2002_titov_komarov_tarasov_levitski_10.1016_S0926-9851(02)00168-4.pdf` | short narrow pores (SNP) 膜极化模型 |
| `2013_bucker_hordt_2013a_10.1093_gji_ggt136.pdf` | 显式引入孔半径和 EDL 的膜极化解析扩展 |
| `2013_bucker_hordt_2013b_10.1190_geo2012-0548.1.pdf` | Marshall-Madden 模型的双长度尺度、SNP/LNP 极限 |
| `2016_bairlein_bucker_hordt_hinze_10.1093_gji_ggw027.pdf` | 温度、离子迁移率、Debye 长度、zeta 电位对 SIP 特征时间的影响 |
| `2019_bucker_flores-orozco_undorf_kemna_10.1029_2019JB017679.pdf` | Stern 层/扩散层耦合与孔喉收缩几何的数值验证 |

## 核心结论

Niu et al. (2020) 使用的是高度压缩后的膜极化等效公式：每个孔喉只需要一个特征长度 `L`、一个直流阻抗 `Zdc`、一个固定极化率 `eta0` 和扩散系数 `D`，再按孔喉分布卷积。这套公式本身与其引用的 Marshall-Madden / Titov / Buecker-Hoerdt 系列是一致的简化分支。

但是，引用文献里真正决定谱峰位置和幅值的物理量比 Niu 文中展开得多得多。特别是：

1. 特征频率不只取决于“孔喉长度分布是否像 Figure 5”，还取决于 `L` 被解释为主动离子选择区长度、被动宽孔长度，还是二者的某个极限模型。
2. `Zdc` 不只是 `throat length / throat area`，在原始理论中它来自主动窄区和被动宽区的串联几何、电导率、截面积和迁移数差异。
3. `eta0=0.01` 在 Niu 中是固定等效极化率；但在原始理论里，极化强度来自主动/被动区迁移数差异、长度比、截面积比、表面电导、EDL 覆盖程度、离子迁移率等。
4. 孔喉半径不仅影响 `Zdc` 截面积，也可能通过 EDL 占截面积比例影响迁移数，从而改变幅值和谱形。Niu 没有把这部分显式写出来。
5. 体积权重卷积在 Niu 中只给出宏观形式，文中没有足够细节说明 `f(L)` 应对应 pnextract 的 throat volume、volume density、count fraction、contact-split 子喉，还是网络导通贡献权重。

因此，我们之前的 `length_scale` 和 `zdc_scale` 确实不应作为正式参数；更合理的方向不是缩放几何，而是把 Niu 隐含压缩掉的“膜极化几何映射层”显式补出来。

## 逐篇阅读要点

### Marshall & Madden 1959

原始图像是两个区域串联：一个普通区域和一个具有阳离子选择性的 membrane zone。关键不是单纯表面有电荷，而是沿电流通道的离子迁移数发生变化，导致浓度梯度和扩散电位。文中还强调：如果选择性区域太多、导电路径几乎全部在选择性区域内，极化反而小；也就是说长度比和区域分布是极化强度的核心参数。

对代码的含义：`membrane_polarizability=0.01` 不应该长期只是常数。至少应允许一个 derived mode，用主动/被动区迁移数差和长度比估算 `eta0`，并保留 Niu 常数模式用于复现。

### Titov et al. 2002

Titov 把砂中电流通道分成 large/passive capillaries 和 narrow/active capillaries。主动窄区是离子选择区，宽区是被动区。模型显式包含：

- 主动/被动区长度 `l2, l1`
- 主动/被动区截面积 `S2, S1`
- 几何系数 `L1=S1/l1, L2=S2/l2`
- 主动区效率系数，与表面电导和孔内平均电导有关
- 大/窄孔之间的离子 transport number 差异
- Debye 长度、孔形状对截面平均浓度的影响

SNP 极限假设是主动窄区远短于被动宽区。这个极限下，特征时间随主动选择区长度平方增长。这一点解释了为什么“Figure 5 孔喉长度分布看起来对”仍可能峰频不对：我们提取的 throat length 未必就是 Titov 意义上的 active ion-selective length。

对代码的含义：当前 `tau = throat_length_m**2/(4D)` 是 Niu 简化式，但应把 `relaxation_length_m` 从 `throat_length_m` 分离出来。contact-split 情况下，视觉/几何 throat length、网络 throat length、active selective length 应该是不同字段。

### Buecker & Hoerdt 2013a

这篇把孔半径和 EDL 显式引入膜极化解析模型。核心是：窄孔半径越小，EDL 占截面积比例越大，阳/阴离子对总电流贡献越不对称，transport number 差异越明显。文中还把 Stern 层用 partition coefficient 表示。

对代码的含义：孔喉半径不应只用于 `A = r^2/(4G)` 计算 `Zdc`。如果要从物理上解释 active aperture，则更完整的方式是新增一个 EDL-derived transport-number 模块，用半径、Debye 长度、zeta/表面电荷、Stern 分配系数、离子迁移率估算每个 throat 的选择性/`eta0`。

### Buecker & Hoerdt 2013b

这篇非常关键。它指出 Marshall-Madden 阻抗实际上有两个区域、两个时间常数；不同极限下，主控特征时间可以由窄区长度控制，也可以由宽区长度控制。文中区分了 short narrow pore (SNP) 和 long narrow pore (LNP) 模型，并指出特征时间与相应区域长度平方成正比。

对我们最重要的启发：如果把 pnextract throat length 直接当成 Niu 的 `L`，其实默认选择了 SNP-like 的单长度分支。但真实网络中 contact-split throat 可能更像“短 active aperture + 相邻 passive pore body/conduit”的组合。只要这个映射错了，孔喉长度分布与论文 Figure 5 相似，也不能保证膜峰频一致。

对代码的含义：应新增诊断性双长度模型：

- `active_length_m`：窄的选择性 contact/aperture 长度，控制 SNP 分支
- `passive_length_m`：相邻 pore-body/conduit 长度，控制 LNP 或过渡分支
- `active_area_m2/passive_area_m2`
- `transport_number_contrast` 或 `mobility_contrast`

正式 Niu 复现仍可默认用 Niu 单长度式，但要能输出这两个长度的审计图和峰频敏感性。

### Bairlein et al. 2016

这篇说明温度通过离子迁移率、扩散系数、动态黏度、Debye 长度、zeta 电位和表面电导影响 SIP 谱。实验证据显示升温会把相位峰推向更高频率。它还指出特征时间与几何尺度平方、扩散系数倒数有关。

对 Niu 复现的直接影响较小，因为 Niu 固定了 `D=1.3e-9 m2/s` 和水电导率。但对代码完善有意义：如果以后扩展盐度/温度/样品条件，不能只改水电导率，还要同步更新 `D(T)`、迁移率、Debye 长度、zeta/表面电导。

### Buecker et al. 2019

这篇用有限元验证 Stern 层和扩散层极化，并把模型推进到 pore-constriction 几何。关键结论是：Stern 层通常比扩散层贡献强，但在孔喉收缩、压实、EDL 逐渐连通时，扩散层贡献会上升；孔喉几何和 EDL 连通性会改变极化贡献。

对代码的含义：Niu 的膜极化模型没有显式分 Stern/diffuse，也没有 EDL 连通性。因此，active aperture 不能被写成“论文明确参数”；它更像一个有效电学孔喉窗口，用来近似 contact 区域中真正发生离子选择/膜导电的部分。若要让它物理化，应通过 EDL/transport-number 模块或有限元/解析子模型来约束。

### de Lima & Sharma 1992

这篇主要是 shaly sands 的 charged clay platelets、Stern/Guoy-Chapman 双电层和 Maxwell-Wagner 等效介质处理。它对干净 Berea 的 pnextract throat 膜极化不是直接公式来源，但提醒我们：微观双电层模型可以通过 total current conductivity function 和混合律进入宏观复电导。

对代码的含义：如果后续处理含黏土样品或 LKC 样品，膜极化不应全部由几何 throat 模型承担，需要单独的 clay/EDL 等效介质分支。

### Chelidze & Gueguen 1999

这是综述，价值在于提醒低频介电/复电导谱可能来自多机制叠加，包括 Maxwell-Wagner、surface polarization、open EDL、percolation 等。对 Niu 的膜项没有提供直接可替代公式，但有助于界定哪些机制不该混入 Niu Section 5.3 的单机制分解。

## Niu 2020 没有说透、但对复现很关键的地方

| 问题 | Niu 2020 处理 | 引用文献中的展开 | 对当前误差的意义 |
|---|---|---|---|
| `L` 是什么长度 | 用 pore throat size/length 分布进入 `tau=L^2/(4D)` | `L` 可代表 active narrow zone、passive wide zone，或双长度模型的极限 | 峰频错位的第一嫌疑 |
| `Zdc` 怎么从网络算 | 说由孔喉几何和水电导率计算，但未给足公式 | 原始模型包含主动/被动区串联、截面积、迁移数/扩散参数 | 幅值错位主因，也会影响谱形权重 |
| `eta0` 来源 | 固定约 1% | 由 transport-number 差异、长度比、表面电导、EDL 覆盖程度决定 | 固定值可复现 Niu，但不是可泛化物理模型 |
| 孔喉半径角色 | 主要表现为几何/分布 | 半径还控制 EDL 占比和 ion selectivity | active aperture 需要物理化 |
| `f(L)` 权重 | 给出卷积思想 | 原始文献没有 pnextract volume/contact-split 对应规则 | 体积权重和导通权重可能不同 |
| contact-split throat | 未说明 | 原始理论关心 narrow active zone，而不是图像分割 throat 的全部几何长度 | contact split 后的 active length/aperture 应显式记录 |

## 当前代码与文献的对应

当前核心代码位于 `code/src/pore_scale_electrical/polarization.py`：

- `membrane_relaxation_time()` 固定为 `L^2/(4D)`。
- `elementary_membrane_impedance()` 实现 Niu Eq. 19 的等效阻抗形式。
- `elementary_membrane_conductance_perturbation()` 使用 `1/Z* - 1/Zdc`。
- `throat_zdc_from_geometry()` 使用 `Zdc = L/(sigma_w A)`，其中 `A = r^2/(4G)`。

当前脚本 `code/scripts/sip_simulation/compute_polarization_spectra.py` 已禁止后处理 `length_scale` 和 `zdc_scale`，这是对的。但它仍然把：

- `throat_length_m` 同时作为膜弛豫长度；
- `throat_radius_m + throat_shape_factor` 只作为 `Zdc` 截面积；
- `eta0` 固定为常数；
- `throat_volume_m3` 默认作为膜卷积权重；

这些都属于 Niu 简化实现，而不是完整的 Titov/Buecker-Hoerdt 物理模型。

## 建议的代码完善路线

### 1. 立即补一个 membrane geometry provenance 层

不要再叫 `length_scale`。建议在 pnextract 解析结果中显式输出：

- `throat_visual_length_m`：可视化/分割几何长度
- `relaxation_length_m`：进入 `tau=L^2/(4D)` 的膜弛豫长度
- `zdc_length_m`：进入 `Zdc=L/(sigma A)` 的欧姆长度
- `hydraulic_radius_m`：几何/可视化半径
- `active_electrical_radius_m`：膜导电有效 aperture 半径
- `active_area_m2`：进入 `Zdc` 的有效电学截面积
- `membrane_weight`：进入 Eq. 10 卷积的权重
- `membrane_geometry_mode`：例如 `niu_single_throat`, `contact_split_active_aperture`, `titov_snp_diagnostic`

这样 active aperture 不再是“缩放因子”，而是一个带 provenance 的几何解释字段。

### 2026-06-17 已落地的第一步代码改进

本轮已经在谱计算入口中加入显式膜几何字段支持：

- `membrane_relaxation_length_m`：若存在，用作 `tau=L^2/(4D)` 的弛豫长度；若不存在，仍用默认 `length_m`，保持 Niu 单长度模式兼容。
- `membrane_zdc_length_m`：若存在，用作 `Zdc=L/(sigma A)` 的欧姆长度；若不存在，仍用 throat length。
- `membrane_active_area_m2`：若存在，直接作为膜导电有效截面积计算 `Zdc`。
- `membrane_active_radius_m`：若存在，则结合 `throat_shape_factor` 换算有效截面积。
- metadata 新增 `membrane_geometry_mode`、`membrane_relaxation_length_source`、`membrane_zdc_length_source`、`membrane_zdc_geometry_source`、`membrane_relaxation_length_m`、`membrane_zdc_length_m`、`membrane_active_area_m2`。

对应代码：

- `code/src/pore_scale_electrical/polarization.py` 新增 `throat_zdc_from_length_area()`。
- `code/scripts/sip_simulation/compute_polarization_spectra.py` 的 `load_pnextract()` 会保留/生成上述膜几何字段。
- `compute_spectra()` 会把弛豫长度和 Zdc 几何分开用于计算与 provenance。

这一步没有把任何全局 `length_scale` 或 `zdc_scale` 恢复为合法参数；它只是允许网络提取阶段或诊断数据显式给出“膜极化真正使用的物理长度/面积”。没有显式列时，真实 v11 网络 smoke metadata 仍标记为 `single_throat_geometry`，并记录：

- `membrane_relaxation_length_source = throat_length_m`
- `membrane_zdc_length_source = throat_length_m`
- `membrane_zdc_geometry_source = throat_radius_m_with_throat_shape_factor`

已运行验证：

- `tests/test_compute_polarization_spectra.py`：13 passed。
- `code/tests/test_compute_polarization_spectra.py`：1 passed。
- `tests/test_run_sample89_3d_sip_comparison.py tests/test_fit_niu2020_membrane_convolution.py`：10 passed。
- v11 网络 smoke：`tmp/v11_geometry_provenance_smoke_spectra.csv` 与 `tmp/v11_geometry_provenance_smoke_spectra.metadata.json`。

### 2026-06-17 追加：局部膜谱与 full-grid 有效电导不能直接混比

本轮重新生成了 Berea v11 contact-split diagnostic 网络：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/`

该目录用同一 v11 pnextract 可执行文件和同一 active-aperture 参数重跑 Berea 网络，恢复了
`niu2020_berea_pnextract_contact_patch_dong_blunt_diagnostics.csv`，并由解析器显式合并出：

- `membrane_relaxation_length_m`
- `membrane_zdc_length_m`
- `membrane_active_radius_m`
- `membrane_active_area_m2`

网络数量与旧正式 v11 包一致：`15439 pores / 19743 throats`，diagnostics 与 throat 逐条匹配。

用显式膜几何计算的局部膜极化谱在 Niu Figure 8 频率点上给出：

- 论文膜峰频：`46415.888 Hz`
- 局部膜谱峰频：`46415.888 Hz`
- 局部膜谱峰值：`7.10558e-4 S/m`
- 论文膜峰值：`4.38724e-5 S/m`

局部膜谱幅值看似高出约 `16.20` 倍，但这不是最终 REV 有效电导。Niu Figure 8 应与 full-grid AC3D
有效复电导比较。使用同一 v11 膜谱作为水相输入的 full-grid x-direction sweep 在论文峰频处为：

- full-grid 膜项：`4.32917e-5 S/m`
- full-grid / 论文峰值比：`0.9868`
- full-grid / 局部膜谱衰减：`0.0609`

新的诊断脚本：

`code/scripts/sip_simulation/diagnose_niu2020_membrane_field_attenuation.py`

输出逐频率表：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/berea_v11_local_to_fullgrid_field_attenuation_vs_niu2020.csv`

以及 summary：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/berea_v11_local_to_fullgrid_field_attenuation_vs_niu2020_summary.json`

这一步的物理含义是：`compute_polarization_spectra.py` 产生的是水相局部极化增量
`Delta sigma_w*`，不是样品尺度有效电导率。只有经过 AC3D full-grid 场求解后，孔隙率、连通性、
电场分布和边界条件才会把局部水相增量折算为可与 Niu Figure 8 对比的 REV 有效复电导。

因此，当前幅值诊断不应把“局部膜谱高 16 倍”直接解释为 `eta0` 或 `Zdc` 错了；至少在 Berea v11
膜项峰值附近，full-grid 场求解已经把局部增量衰减到与论文膜峰幅值一致。真正还没有完全解决的是
全频带谱形，特别是低频段 full-grid 曲线低于 Niu Figure 8 的部分。

### 2026-06-17 追加：低频谱形偏低支持双长度 active/passive 诊断

将 `local -> full-grid` 场衰减写成逐频率诊断后，发现 `full/local` 在主要频带几乎是常数：

- `full/local` 中位数约 `0.0608`
- 论文峰频处 `full/local = 0.0609`

因此，低频 mismatch 主要不是 AC3D 场求解衰减随频率异常变化造成的，而是局部膜谱形状相对 Niu
Figure 8 过窄：短 active contact-split 长度准确控制了峰频，但低频侧缺少更长长度尺度的贡献。

这与 Bücker & Hördt (2013b) 的 SNP/LNP 解释一致：Marshall-Madden 膜极化并不必然只有一个长度；
短窄 active zone 和相邻宽/被动 pore zone 可以带来两个特征时间。为验证这个方向，新增诊断脚本：

`code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py`

它保留当前 v11 active branch：

```text
active length = membrane_relaxation_length_m
active Zdc    = zdc_ohm from membrane_active_area_m2
active weight = volume_over_length
```

然后加入一个 diagnostic passive branch，长度候选来自：

- `center_to_center`
- `conduit = pore1_length + throat_length + pore2_length`
- `pore1_plus_pore2`

并用非负 `alpha` 拟合该 passive branch 的幅度。注意：`alpha` 是诊断系数，用来评估双长度谱形是否解释低频肩部；
它不是正式 Niu 默认参数，也不是 `length_scale/zdc_scale`。

真实 Berea v11 explicit 网络的输出：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/`

最佳候选为：

```text
passive_length_source = pore1_plus_pore2
passive_weight_source = volume_over_passive_length
passive_zdc_source    = center_zdc
alpha                 = 36.337
```

效果：

- active-only peak-normalized RMSE：`0.05852`
- dual-length diagnostic peak-normalized RMSE：`0.02235`
- 论文峰频：`46415.888 Hz`
- dual-length 峰频：`46415.888 Hz`
- 论文峰频处幅值比：`0.9886`
- 全频带 ratio 范围：`0.818 - 1.115`

这说明目前最可信的下一步物理改进不是继续调 active aperture 或 Zdc，而是把 Niu 单长度膜公式扩展为
可审计的 active/passive 双长度诊断模型。正式采用前仍需做两件事：

1. 给 `alpha` 找到可由 active/passive 迁移数差、长度比、面积比或 EDL 选择性推导的物理表达；
2. 用 dual-length water-phase spectrum 重跑 full-grid AC3D，而不是只用常数 field attenuation 近似。

随后已完成一个 7 个代表频率的 full-grid AC3D smoke：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_dual_length_x_7freq_smoke/`

对比表：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_fullgrid_7freq_vs_active_and_niu2020.csv`

summary：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_fullgrid_7freq_vs_active_and_niu2020_summary.json`

代表频率 full-grid 结果：

- 全部 `info=0`
- 最大相对残差：`9.82e-6`
- active-only 7 点 peak-normalized RMSE：`0.05450`
- dual-length 7 点 peak-normalized RMSE：`0.01676`
- active-only 7 点中位比值：`0.835`
- dual-length 7 点中位比值：`0.989`
- dual-length 7 点 ratio 范围：`0.838 - 1.120`

特别是低频端：

```text
0.001 Hz: active-only / paper = 0.040, dual-length / paper = 1.120
0.1   Hz: active-only / paper = 0.060, dual-length / paper = 1.098
10    Hz: active-only / paper = 0.284, dual-length / paper = 0.974
```

这说明双长度 active/passive 分支不只是常数场衰减近似下好看；它在 full-grid AC3D 代表频率中也确实补上了低频肩部，同时保持峰频和峰值。下一步若要提升为正式候选模型，应运行完整 Figure 8 40 频点 full-grid sweep，并把 `alpha` 的来源从拟合系数推进到 transport-number/EDL/几何推导。

已进一步完成完整 Niu Figure 8 40 个频点 full-grid AC3D sweep：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_dual_length_x_figure8freq/`

对比数据：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_fullgrid_40freq_vs_active_and_niu2020.csv`

summary：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_fullgrid_40freq_vs_active_and_niu2020_summary.json`

对比图：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/dual_length_membrane_fullgrid_40freq_vs_niu2020.png`

完整 40 点结果：

- 全部 `info=0`
- 最大相对残差：`9.82e-6`
- active-only peak-normalized RMSE：`0.05856`
- dual-length peak-normalized RMSE：`0.02231`
- active-only 中位比值：`0.811`
- dual-length 中位比值：`0.920`
- active-only ratio 范围：`0.040 - 1.000`
- dual-length ratio 范围：`0.820 - 1.120`
- 论文峰频处 dual-length / paper：`0.9887`

这说明 full-grid 40 点下，dual-length 诊断模型显著改善了低频端，并保持峰频/峰值基本不变。当前最大限制已经不再是“能否对上 Niu Figure 8”，而是 `alpha=36.337` 的物理来源：它目前仍是拟合出来的 passive branch 系数。若要把它从 diagnostic 提升为正式模型，需要从 active/passive transport-number contrast、EDL selectivity、长度比和面积比推导出同量级的系数，或用独立样品/方向数据约束它。

### 2026-06-17 追加：`alpha` 的几何补偿账本

为避免把 `alpha=36.337` 当成黑箱调参，`fit_niu2020_dual_length_membrane.py`
现在同步输出最佳双长度候选的 active/passive branch balance：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_best_branch_balance.csv`

最佳候选仍为：

```text
passive_length_source = pore1_plus_pore2
passive_weight_source = volume_over_passive_length
passive_zdc_source    = center_zdc
alpha                 = 36.337
```

该账本显示：

- active 中位弛豫长度：`7.50 um`
- passive 中位弛豫长度：`28.28 um`
- passive/active 中位长度比：`3.77`
- passive/active 权重和比：`0.0774`
- passive/active 中位 Zdc 比：`4.47`
- 未乘 `alpha` 的 passive 局部峰值仅为 active 局部峰值的 `0.00379`
- 乘 `alpha` 后，passive 局部峰值约为 active 局部峰值的 `0.138`

这解释了为什么双长度模型不会破坏 Niu 峰频和峰值：active contact-split 分支仍主导峰值；passive 分支即使乘上 `alpha`，峰值贡献也只有 active 峰值约 14%，其主要作用是补偿 active-only 曲线在低频侧缺失的长时间尺度肩部。

同时，这也说明 `alpha` 目前还不能被写成正式物理参数。它不是 `length_scale` 或 `zdc_scale`，因为它没有移动 active branch 的几何长度或整体欧姆电阻；但它仍是一个拟合出来的 passive branch 强度系数。Bücker & Hördt (2013b) 的理论支持 active/passive zones、mobility/transport-number contrast 以及 SNP/LNP 双时间尺度；Titov et al. (2002) 也把 active/passive transport-number difference、active/passive length/section 写入膜极化强度。但 Niu 2020 没有给出如何从三维 pnextract 网络直接计算这个 passive branch 系数。因此在当前代码中，dual-length 仍应保持为 `diagnostic_two_length_active_passive_branch_fit`，不能提升为默认正式复现参数。

2026-06-18 继续推进：脚本新增 Titov/Fridrikhsberg-Sidorova 风格的几何 chargeability factor 诊断：

```text
gamma0 ~ 4 * (Delta n)^2 / [(l1/S1 + l2/(a2 S2)) * (S1/l1 + S2/l2)]
```

这里 `l1, S1` 为 passive zone 长度/截面积，`l2, S2` 为 active zone 长度/截面积，`Delta n` 为 active/passive transport-number difference。该诊断不会替代拟合 `alpha`，只把 `alpha` 反推成若由 transport-number contrast 解释所需的强度。

当前 Berea 最佳双长度候选给出：

- `passive_eta_equivalent_alpha_times_eta0 = 0.363`
- `titov_geometry_factor_median = 0.580`
- `transport_number_contrast_needed_median = 0.791`
- `transport_number_contrast_needed_mean = 0.898`
- `transport_number_contrast_needed_gt_one_fraction = 0.248`
- 在 passive anion transport number = 0.5 的中性对称电解质假设下，`neutral_passive_transport_contrast_valid_fraction = 0`

这说明：把 `alpha=36.337` 简单解释成 `eta_passive = alpha * eta0` 后，所需的 transport-number contrast 对普通中性 passive zone 图像太强，不能作为正式物理闭合。更可能的缺失项包括：

1. passive/wide zone 的真实电学截面积不是当前由 `center_zdc` 反推的有效面积；
2. Niu 单一 `eta0=0.01` 不适合直接拆成 active/passive 两个独立支路；
3. EDL/Stern-layer 表面电导和离子选择性需要显式进入 active-zone efficiency `a2`；
4. full-grid 场衰减对 active/passive 分支可能不同，不能长期用一个常数 attenuation 解释所有支路。

因此，当前最诚实的结论是：dual-length 模型已经把“低频肩部缺失”的根因从几何长度层面定位到了 active/passive 双时间尺度；但 `alpha` 的强度闭合还需要 EDL/transport-number 模块或独立约束，不能写成 Niu 2020 已给出的参数。

同日进一步检查 Titov 公式中的 active-zone efficiency `a2`。`a2` 表示 narrow/active zone 的有效 capillary conductivity 与 bulk solution conductivity 的比值，理论上可吸收表面电导和 EDL 选择性带来的 active-zone 导电增强。脚本新增了反推函数：给定目标 `Delta n`，求达到 `eta_required = alpha * eta0` 所需的 `a2`。

结果显示：

- 若目标 `Delta n = 0.5`，只有 `0.76%` 的 throat 有数学可行解；
- 这些可行 throat 的 `a2` 中位数约为 `36.7`；
- 若目标 `Delta n = 0.2`，可行比例为 `0`。

因此，仅靠 active-zone efficiency `a2` 也不能把 `alpha=36.337` 可靠闭合。该结果进一步支持：当前 passive 分支的最大不确定性很可能来自 passive/wide zone 的有效截面积、branch normalization、以及 active/passive 在 full-grid 场中的不同上尺度权重，而不是单独某个 transport-number 或 surface-conductivity 系数。

继续沿 passive/wide-zone 截面积假设做 sensitivity 诊断。脚本新增输出：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_passive_area_sensitivity.csv`

该表固定最佳 passive 长度 `pore1_plus_pore2` 和权重 `volume_over_passive_length`，只改变 passive branch 的 `Zdc = L/(sigma A)` 中的面积解释。结果：

| passive area source | alpha | peak-normalized RMSE | 所需 Delta n 中位数 | 说明 |
|---|---:|---:|---:|---|
| `center_zdc_area` | `36.34` | `0.02235` | `0.792` | 谱形最佳附近，但 transport-number 不物理 |
| `active_area` / `throat_shape_area` | `22.80` | `0.02197` | `0.583` | 谱形略优，但仍需偏强选择性 |
| `volume_over_passive_length_area` | `0.0164` | `0.02504` | `0.022` | 强度可物理闭合，但谱形略差 |

这个结果非常重要：`alpha` 之所以大，并不只是 active/passive transport-number 或 `a2` 的问题，而是 passive branch 的电学面积定义会把强度改变几个数量级。若用 passive volume/length 给出宽孔等效截面积，所需 transport-number contrast 回到很温和的 `~0.02`，且 `Delta n = 0.2` 或 `0.5` 下 `a2` 都有可行解；但该谱形不如 center-zdc/active-area 模式贴近 Niu Figure 8。因此下一步最有价值的验证不是继续拟合 `alpha`，而是：

1. 用 `volume_over_passive_length_area` 生成一个“物理可闭合 passive-area candidate”膜分量；
2. 跑同一 40 频 full-grid AC3D；
3. 比较它和 `center_zdc_area` best-fit 的 full-grid 低频肩部差异；
4. 决定正式候选应采用“更贴图的 center-zdc area”还是“更物理的 volume/length area”，或者采用由 pnextract pore body 截面/局部 field attenuation 约束的中间面积。

其中第 1 步已经产出可直接接 AC3D sweep 的 component spectrum：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_component_spectrum.csv`

该谱使用：

```text
passive_length_source = pore1_plus_pore2
passive_weight_source = volume_over_passive_length
passive_area_source   = volume_over_passive_length_area
alpha                 = 0.016355
```

它保留了膜局部谱峰频 `46415.888 Hz`，局部虚部峰值约 `7.12e-4 S/m`，同时把所需 transport-number contrast 降到约 `0.022`。因此它是目前最值得做 full-grid AC3D 验证的“物理可闭合”passive-area 候选。

已完成同一 Figure 8 40 个频率点的 full-grid AC3D 验证：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_dual_length_volume_area_x_figure8freq/`

对比数据：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_volume_area_fullgrid_40freq_vs_center_active_and_niu2020.csv`

summary：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_volume_area_fullgrid_40freq_vs_center_active_and_niu2020_summary.json`

对比图：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/dual_length_volume_area_fullgrid_40freq_vs_niu2020.png`

完整 40 点 full-grid 结果：

| 模型 | peak-normalized RMSE | 中位比值 | ratio 范围 | 论文峰频处比值 |
|---|---:|---:|---:|---:|
| active-only v11 | `0.05856` | `0.811` | `0.040 - 1.000` | `0.9868` |
| dual length, center-zdc area | `0.02231` | `0.920` | `0.820 - 1.120` | `0.9887` |
| dual length, volume/length area | `0.02510` | `0.907` | `0.772 - 1.158` | `0.9881` |

三条曲线的 full-grid 峰频都落在 `46415.888 Hz`。因此，`volume_over_passive_length_area` 没有破坏特征频率，也保持了峰值；它只是在全频带谱形上比 `center_zdc_area` 略差一点。更重要的是，`volume/length area` 把 passive branch 所需强度从 `alpha=36.337` 降到 `alpha=0.016355`，对应所需 transport-number contrast 中位数约 `0.022`，比 `center_zdc_area` 的 `~0.792` 更容易物理闭合。

这改变了当前判断：若只追求 Niu Figure 8 的数值贴合，`center_zdc_area` 最好；若追求可由 Titov/Buecker-Hoerdt 风格 active/passive 几何和迁移数差异解释，`volume_over_passive_length_area` 更合理。下一步代码完善不应再引入全局 `length_scale` 或 `zdc_scale`，而应把 passive 宽区有效截面积作为显式、可审计的膜几何字段：例如 `passive_area_m2 = passive_volume_m3 / passive_length_m`，并在 metadata 中标明它是 diagnostic active/passive 双长度模型，而不是 Niu 2020 已公开给出的原始参数。

同日把这一步落到可复用脚本输出中：`fit_niu2020_dual_length_membrane.py` 生成的 `dual_length_membrane_volume_area_component_spectrum.csv` 现在逐频率记录 `passive_area_source`、`passive_length_source`、`passive_weight_source`、`passive_zdc_source`、`passive_branch_alpha` 和 `membrane_geometry_mode`。因此后续 AC3D sweep 只看输入水相谱 CSV，也能知道这条曲线来自 `diagnostic_two_length_active_passive_volume_area`，不是无 provenance 的隐藏调参。

进一步把该信息接入结果包 provenance：`build_niu2020_berea_result_package.py` 新增 `write_diagnostic_membrane_model_summary()`，读取 diagnostic component spectrum 和 full-grid summary，写出：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/diagnostic_membrane_model_summary.json`

该 JSON 明确记录 `diagnostic_not_default_niu2020_parameter`、`diagnostic_two_length_active_passive_volume_area`、`volume_over_passive_length_area`、`alpha=0.0163545` 和 full-grid RMSE/峰频指标。诊断结果目录顶层也已写入 `manifest.json` 和 `README.md`，README 会说明该模型不是 Niu 2020 已公开给出的默认参数，也不是隐藏的 `length_scale` 或 `zdc_scale`。

同日进一步接入正式机制图脚本：`plot_niu2020_conductivity_mechanism_comparison.py` 新增可选参数 `--diagnostic-membrane-fullgrid-csv` 和 `--diagnostic-membrane-summary-json`。提供这两个路径时，脚本会把诊断膜候选作为 `diagnostic_membrane_volume_area` 写入 source data，并在虚部面板叠加为独立的 `Membrane diagnostic` 曲线；默认 `simulation_membrane` 仍保持原四机制 membrane sweep，不会被替换。

已基于 v11 full-grid 四机制包生成带诊断膜候选的对比图：

`results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/figures/niu2020_conductivity_mechanism_comparison_v11_x_with_diagnostic_membrane.png`

source data：

`results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/source_data/niu2020_conductivity_mechanism_comparison_v11_x_with_diagnostic_membrane_source_data.csv`

provenance：

`results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/provenance/niu2020_conductivity_mechanism_comparison_v11_x_with_diagnostic_membrane_provenance.md`

该 source data 含 `diagnostic_membrane_volume_area` 40 行；provenance 明确记录 `diagnostic_two_length_active_passive_volume_area`，并说明它不是隐藏的 `length_scale` 或 `zdc_scale`。

2026-06-18 继续补充：图脚本现在会同时派生 `diagnostic_all_volume_area = simulation_all - simulation_membrane + diagnostic_membrane_volume_area`，用于检验“把更物理的膜分支放回总响应后是否更接近实验”。同一输出的 summary JSON：

`results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/source_data/niu2020_conductivity_mechanism_comparison_v11_x_with_diagnostic_membrane_summary.json`

记录了两套指标：

```text
diagnostic_membrane_volume_area vs experiment_imag:
  common_frequency_count = 6
  peak_normalized_rmse   = 0.4453
  ratio_at_paper_peak    = 0.5418
  candidate_peak_freq    = 10000 Hz

diagnostic_all_volume_area vs experiment_imag:
  common_frequency_count = 6
  peak_normalized_rmse   = 0.3194
  ratio_at_paper_peak    = 0.8113
  candidate_peak_freq    = 10000 Hz
```

这个结果说明：单独膜分支不能直接代表 Figure 8 实验总虚部；把诊断膜分支替换回总响应后，曲线明显更接近实验，但仍未完全闭合。因此当前问题已经从“膜峰频为什么错”推进到“总响应中 pore/interfacial/membrane 的组合和 full-grid all 机制是否还缺少耦合或方向平均”的问题。

同日进一步做了真正 full-grid `all_diagnostic_volume_area` 验证，而不是只用有效电导结果代数替换。新增 `make_polarization_component_spectra.py --diagnostic-membrane-component`，生成 paper-mode all 输入谱：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/diagnostic_all_volume_area_component_spectra/polarization_spectra_all_diagnostic_volume_area.csv`

然后运行完整 `350^3` x-direction AC3D sweep：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/all_diagnostic_volume_area_x_figure8freq/sweep_results.csv`

并生成 full-grid all diagnostic 对比图和 summary：

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_conductivity_mechanism_comparison_fullgrid_all_diagnostic_volume_area.png`

`results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_conductivity_mechanism_comparison_fullgrid_all_diagnostic_volume_area_summary.json`

真实 full-grid all diagnostic 与实验虚部的共同频率指标为：

```text
simulation_all vs experiment_imag:
  common_frequency_count = 6
  peak_normalized_rmse   = 0.3195
  ratio_at_paper_peak    = 0.8115
  paper_peak_freq        = 10000 Hz
  candidate_peak_freq    = 10000 Hz
```

这个结果几乎等于前面的代数替换 `diagnostic_all_volume_area`，说明剩余差距不是“没有重新做 all 场求解”造成的。当前更可信的解释是：双长度膜分支已经修正了膜峰频和主要膜幅值，但 Figure 8 总虚部仍包含方向平均、pore/interfacial 参数、机制耦合或实验/模拟定义差异。下一步若继续追求总响应完全贴合，应优先审计 x/y/z 方向与方向平均、pore polarization 参数和 interfacial 高频介电项，而不是再改膜长度或 Zdc。

### 2. 把 Niu 单长度模型和诊断双长度模型分开

保留当前 Niu 模式作为正式复现默认：

```text
tau = relaxation_length_m^2 / (4D)
Zdc = zdc_length_m / (sigma_w * active_area_m2)
eta0 = 0.01
```

新增诊断模式：

```text
active_length_m, passive_length_m
active_area_m2, passive_area_m2
mobility_or_transport_number_contrast
```

先不必一次实现完整 Buecker-Hoerdt Eq. 16，但至少输出 SNP/LNP 两个极限的峰频预测，用来回答“峰频到底由哪个长度控制”。

### 3. 给 `eta0` 增加 derived/candidate 模式

短期仍默认 `eta0=0.01`。但新增：

- `eta_mode="niu_constant"`：复现 Niu
- `eta_mode="transport_number_contrast_diagnostic"`：由主动/被动区迁移数差异估算
- `eta_mode="edl_radius_diagnostic"`：由半径、Debye 长度、zeta/表面电荷估算

这些模式必须在结果目录和 metadata 中标为 diagnostic，直到有实验或论文参数支撑。

### 4. active aperture 的合理定位

`contactActiveApertureRadiusVoxels=0.17` 不能说是 Niu 论文明确给出的参数。它更合理的表述是：

“在 contact-split throat 中，几何 throat 半径用于网络/可视化；膜电流只允许通过接触处的有效离子选择窗口参与 `Zdc` 计算。”

这不是简单的 `zdc_scale`，因为它改变的是 contact throat 的电学截面积解释，而不是对所有 `Zdc` 后处理乘常数。但它仍然是有效参数，需要 provenance、敏感性和物理约束。

### 5. 加一组最小测试

建议新增/扩展测试：

- `relaxation_length_m` 改变时，峰频按 `1/L^2` 移动。
- `active_electrical_radius_m` 改变时，主要改变膜项幅值/`Zdc`，不应改变单元素峰频。
- `zdc_length_m` 和 `relaxation_length_m` 分离后，metadata 必须分别记录二者。
- `membrane_geometry_mode` 必须写入 source data/provenance。

## 对当前 Niu 峰频不一致的判断

“孔喉尺寸分布图和 Niu Figure 5 对得好”只能说明我们在统计意义上重建了 Niu 报告的孔/喉分布；它不能证明膜极化公式里的 `L` 已经映射正确。

最可能的原因是：

1. pnextract 的 `throat_length_m` 是几何/网络喉长；
2. Niu 公式里的膜 `L` 更接近 active ion-selective zone length；
3. contact-split 后，active zone 可能是接触窗口附近很短的一段，而不是完整 center-to-center 或完整 throat segment；
4. 文献中的双长度模型说明，峰频可能落在主动区、被动区或过渡区控制的不同分支；
5. Niu 没有公开这些映射细节，所以只能通过显式 provenance + 诊断模型缩小不确定性。

因此，下一步不建议恢复全局 `length_scale`。更好的做法是：在代码里把 `relaxation_length_m`、`zdc_length_m`、`active_area_m2` 和 `membrane_weight` 拆开，并把 v11 active aperture 写成一个可审计的 contact-split 电学几何模式。

## 2026-06-18 方向平均排查

为检查 Figure 8 总虚部剩余差距是否来自只取 x 方向，本轮补跑了 diagnostic all volume-area 模型的 y/z 方向 full-grid AC3D，并生成 x/y/z 方向平均：

```text
x sweep:
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/all_diagnostic_volume_area_x_figure8freq/sweep_results.csv
y sweep:
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/all_diagnostic_volume_area_y_figure8freq/sweep_results.csv
z sweep, reliable full40 descending warm-start:
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/all_diagnostic_volume_area_z_figure8freq_full40/sweep_results.csv
xyz mean:
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/all_diagnostic_volume_area_xyz_mean_figure8freq/sweep_results.csv
```

注意：最早的 z 临时目录 `all_diagnostic_volume_area_z_figure8freq/` 只有 30 个点，且最低频 `0.001 Hz` 未收敛到 `1e-5`，因此不参与方向平均。正式用于平均的是 `z_figure8freq_full40`，共 40 个频点，`info=0`，最大相对残差小于 `1e-5`。

方向平均摘要：

```text
direction_average_summary.json:
  frequency_count = 40
  all_info_zero = true
  max_relative_residual_norm = 9.993e-6
```

对 Figure 8 实验虚部的共同频率指标为：

```text
x-only full-grid all diagnostic:
  peak_normalized_rmse = 0.3195
  ratio_at_paper_peak  = 0.8115
  candidate_peak_freq  = 10000 Hz

xyz-mean full-grid all diagnostic:
  peak_normalized_rmse = 0.3536
  ratio_at_paper_peak  = 0.7250
  candidate_peak_freq  = 10000 Hz
```

因此，方向平均没有解释剩余差距，反而使总虚部峰值进一步低于实验；x 方向在当前 Berea 体数据和边界条件下更接近 Niu Figure 8。剩余差距更可能来自 pore/interfacial 参数、总响应实验/模拟定义、Niu 未公开的机制耦合或谱分量归一化，而不是膜长度、`Zdc`、或单纯的方向选择。下一步若继续追总响应，应优先审计 pore polarization 和 interfacial 高频介电背景，而不是继续调膜几何。

## 2026-06-18 追加：把 diagnostic all 的膜几何 provenance 固化进输入谱

此前 `make_polarization_component_spectra.py` 在生成
`polarization_spectra_all_diagnostic_volume_area.csv` 时，只把诊断膜分支的
`membrane_geometry_mode` 传入 all-spectrum。这样 full-grid sweep 虽然数值正确，但单看输入谱 CSV
无法知道 passive branch 的面积、长度、权重和 `Zdc` 来源。现已修正：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/diagnostic_all_volume_area_component_spectra/polarization_spectra_all_diagnostic_volume_area.csv
```

现在逐频率保留：

```text
membrane_geometry_mode = diagnostic_two_length_active_passive_volume_area
passive_area_source    = volume_over_passive_length_area
passive_length_source  = pore1_plus_pore2
passive_weight_source  = volume_over_passive_length
passive_zdc_source     = volume_over_passive_length_area
passive_branch_alpha   = 0.0163545307975375
```

同目录 `metadata.json` 也写入了 `diagnostic_membrane_geometry_mode`、
`diagnostic_membrane_passive_area_source` 和 `diagnostic_membrane_passive_branch_alpha`。
这一步不改变 `apparent_water_sigma_*` 数值，只提升可追溯性：后续 AC3D sweep 的输入谱本身就能说明这条
`all_diagnostic_volume_area` 曲线来自 volume/passive-length active-passive 双长度膜候选，而不是隐藏的全局
`length_scale` 或 `zdc_scale`。

## 2026-06-18 追加：总响应差距不能全部归因于膜几何

为避免把 Figure 8 总虚部的剩余差距继续误判为膜长度或 `Zdc` 问题，新增诊断脚本：

```text
code/scripts/sip_simulation/diagnose_niu2020_total_response_gap.py
```

该脚本同时比较三件事：

1. 我们的 candidate all full-grid sweep vs Niu Figure 8 的论文 `Simulation`；
2. 我们的 candidate all full-grid sweep 经 log-frequency 插值后 vs `Experiment`；
3. Niu 论文 `Simulation` 经 log-frequency 插值后 vs `Experiment`。

本轮针对 x-direction `all_diagnostic_volume_area` 生成：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_total_response_gap_diagnostic_all_volume_area_x.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_total_response_gap_diagnostic_all_volume_area_x_summary.json
```

关键频带指标如下：

```text
1-1e5 Hz, candidate all vs Niu paper Simulation:
  common_frequency_count = 16
  median_ratio           = 0.709
  peak_normalized_rmse   = 0.0432
  ratio_at_reference_peak= 1.087

1-1e5 Hz, candidate all vs Experiment:
  common_frequency_count = 16
  median_ratio           = 0.583
  peak_normalized_rmse   = 0.1250
  ratio_at_reference_peak= 0.613

1-1e5 Hz, Niu paper Simulation vs Experiment:
  common_frequency_count = 16
  median_ratio           = 0.717
  peak_normalized_rmse   = 0.1164
  ratio_at_reference_peak= 0.595
```

这说明剩余总响应差距有两层：第一，我们的 current all diagnostic 在 `1-1e5 Hz`
相对 Niu 论文 `Simulation` 仍偏低约 30%；第二，Niu 论文 `Simulation` 自身在同一频带相对实验也偏低。
因此，不能再把“与实验总虚部不完全重合”简单归因于膜极化长度或 `Zdc`。膜分量对 Niu membrane component
已经接近；下一步应把注意力转到 `pore polarization` 与 `interfacial polarization` 的参数、上尺度权重和机制耦合定义，
同时保留一个事实：如果目标是严格贴实验总曲线，可能需要超出 Niu 公开机制分解参数的标定，而这不能写成 Niu 默认物理输入。

## 2026-06-18 追加：组件级差距账本

为进一步定位 `candidate all vs Niu Simulation` 的低估来自哪条机制，新增组件诊断脚本：

```text
code/scripts/sip_simulation/diagnose_niu2020_component_response_gap.py
```

本轮输入：

```text
pore:
  results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/simulation_sweeps/pore_v11_x_figure8freq/sweep_results.csv
membrane:
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_dual_length_volume_area_x_figure8freq/sweep_results.csv
interfacial:
  results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/simulation_sweeps/interfacial_v11_x_figure8freq/sweep_results.csv
```

输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_component_response_gap_v11_pore_interfacial_dual_length_membrane_x.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_component_response_gap_v11_pore_interfacial_dual_length_membrane_x_summary.json
```

在 `1-1e5 Hz` 频带，相对 Niu Figure 8 论文组件：

```text
pore:
  median_ratio         = 0.812
  peak_normalized_rmse = 0.0918
  ratio_at_paper_peak  = 0.722

membrane, dual-length volume-area:
  median_ratio         = 0.925
  peak_normalized_rmse = 0.0334
  ratio_at_paper_peak  = 0.988
  paper_peak_freq      = candidate_peak_freq = 46415.888 Hz

interfacial:
  median_ratio         = 0.765
  peak_normalized_rmse = 0.0352
  ratio_at_paper_peak  = 0.940

component sum vs Niu Simulation:
  median_ratio         = 0.709
  peak_normalized_rmse = 0.0427
```

该结果把诊断重心进一步推开了：在当前候选模型中，膜分量已经是三条机制里最接近 Niu 论文组件的一条；
`candidate all` 相对 Niu `Simulation` 的中位低估主要不应继续由膜长度或 `Zdc` 承担。接下来最值得审计的是：

1. `pore polarization` 的几何权重、特征长度和体积卷积是否严格等价于 Niu 的实现；
2. `interfacial polarization` 的高频介电背景和边界条件是否完全按 Section 5.3 的机制分离定义进入 AC3D；
3. 三个独立机制分量相加与 `all` full-grid 求解之间是否存在非线性场分布差异，不能简单用局部谱代数相加解释。

## 2026-06-18 追加：pore polarization 的 local-to-full-grid 衰减诊断

继续审计 pore 分量后发现，Schwarz 局部 pore 公式本身并不是最可疑环节。代码中 pore 分量使用：

```text
tau_p = r^2 / (2D)
C_p*(r) = Sigma_s * i omega tau_p / (1 + i omega tau_p)
```

这与 Niu Equation 17-18 一致。用 Figure 5 的 `pore node relative volume` 直接计算局部水相增量时，pore 谱峰频与 Niu Figure 8 pore component 同落在 `0.464158883 Hz`；用 v11 网络孔体积分布时，局部谱峰移到 `1 Hz`，说明 v11 分布细节会轻微移动 pore 峰频，但没有造成数量级错误。

为区分局部 Schwarz 公式与 AC3D 场上尺度的贡献，新增诊断脚本：

```text
code/scripts/sip_simulation/diagnose_niu2020_pore_field_attenuation.py
```

生成的输入/输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_figure8_paper_pore_component.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/berea_v11_local_pore_water_increment_figure8freq.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_pore_field_attenuation_v11_x.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_pore_field_attenuation_v11_x_summary.json
```

在 `1-1e5 Hz`：

```text
median actual_field_attenuation      = 0.06185
median paper_required_attenuation    = 0.07589
median actual/required attenuation   = 0.81162
median effective_to_paper pore ratio = 0.81162
```

这说明 pore 分量低估可以拆成两步理解：

1. 局部 pore 谱的峰频和形状与 Niu Schwarz 模型基本一致；
2. 当前 x-direction full-grid AC3D 把局部 pore 水相增量折算为 REV 有效电导时，场衰减比 Niu paper pore component 隐含的衰减小约 `19%`。

因此，下一步若修正 pore 分量，不应引入 `pore_radius_scale` 这类后处理几何缩放；更合理的方向是审计 AC3D pore-only 机制的边界条件、solid=0 处理、孔隙连通场分布、方向选择，以及 Niu paper component 是否使用了与我们完全相同的 full-grid 上尺度定义。

## 2026-06-18 追加：interfacial polarization 的有效介电常数诊断

继续审计 interfacial 分量后发现，当前差距同样不像求解器收敛问题。新增诊断脚本：

```text
code/scripts/sip_simulation/diagnose_niu2020_interfacial_response_gap.py
```

该脚本把 interfacial-only sweep 的虚部有效电导换算为等效相对介电常数：

```text
epsilon_eff,imag-equivalent = sigma'' / (omega epsilon0)
```

并与 Niu Figure 8 的 `Interfacial polarization` 组件逐频比较。输入/输出为：

```text
input paper:
  data/Niu 2020data/Figure8.xlsx
input candidate:
  results/niu2020_berea_reproduction_20260617_contact_split_v2_v11_active_aperture/simulation_sweeps/interfacial_v11_x_figure8freq/sweep_results.csv
output:
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_interfacial_response_gap_v11_x.csv
  results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_interfacial_response_gap_v11_x_summary.json
```

在 `1-1e5 Hz`：

```text
median candidate/paper sigma'' ratio = 0.76527
median candidate/paper eps ratio     = 0.76488
median candidate effective eps_r     = 32.30
median Niu interfacial eps_r         = 42.81
max relative residual norm           = 9.98e-6
all solver info == 0                 = true
```

这说明 interfacial 分量偏低约 `23-24%`，但 sweep 残差和 `info` 状态没有显示数值未收敛。
因此，interfacial 的下一步不应是放宽 Krylov 容差或增加迭代数，而应审计以下物理/口径差异：

1. Niu Section 5.3 的 interfacial-only 机制是否只保留水/固 dc conductivity 与高频 permittivity，且孔隙水、固相介电常数取值与我们完全一致；
2. paper Figure 8 的 interfacial component 是否经过与实验/Simulation 相同的几何因子、方向或体积平均口径处理；
3. 当前 x-direction full-grid 边界条件是否会低估 Maxwell-Wagner 型高频介电上尺度，尤其是 Berea 各向异性和 y/z 方向响应已被证明更低时；
4. interfacial 分量是否需要在 metadata 中显式记录 `epsilon_w`、`epsilon_s`、`sigma_w`、`sigma_s` 与等效 `eps_r` 平台值，避免只看 `sigma''` 而忽略高频介电背景的口径差异。

和 pore 诊断合在一起看，当前剩余总响应差距的更合理解释是：膜分量已经基本闭合；pore 分量存在约 `19%` 的 full-grid 上尺度衰减差异；interfacial 分量存在约 `24%` 的有效介电常数差异。后两者叠加后，正好可以解释 `candidate all` 相对 Niu `Simulation` 仍约 `0.71` 的中位比例。

## 2026-06-18 追加：把膜诊断模型的文献依据写入脚本 provenance

为避免 active/passive 双长度模型变成“只有讨论里有依据、结果文件里看不到依据”的状态，本轮把结构化文献依据写入脚本输出：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
code/scripts/sip_simulation/build_niu2020_berea_result_package.py
```

新增的 provenance 字段为 `literature_basis_structured`。它逐条记录 DOI、模型角色、支持的物理假设，以及本地 PDF/text 路径。当前包括：

```text
10.1190/1.1438659
  active/passive impedance foundation
  支持 Marshall-Madden active/passive zone 阻抗框架

10.1016/S0926-9851(02)00168-4
  transport-number and geometry-factor chargeability
  支持 passive/active 长度、截面和迁移数差控制膜极化强度

10.1190/GEO2012-0548.1
  SNP/LNP two-time-scale interpretation
  支持 active/passive 两个时间常数，以及窄区/宽区长度分别控制特征时间的极限

10.1029/2019JB017679
  Stern/diffuse-layer caution for pore constrictions
  提醒孔喉收缩中的 Stern/diffuse layer 耦合可能影响膜极化，因此该双长度分支仍应保留 diagnostic 标记
```

实际刷新后的文件：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_candidate_metadata.json
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/diagnostic_membrane_model_summary.json
```

这一步不改变膜谱数值，也不把 `alpha` 升格为 Niu 2020 公开参数；它的意义是把“为什么允许 active/passive 双长度、为什么必须保留 diagnostic 标签”写进可复跑结果本身。后续若继续完善代码，应让任何使用 `diagnostic_two_length_active_passive_volume_area` 的图件、source data 或 result package 都能追溯到这些 DOI 和本地文献副本。

## 2026-06-18 追加：为 volume-area 膜候选单独输出物理解释账本

此前 `dual_length_membrane_best_branch_balance.csv` 对应的是最贴合 Niu membrane component 的 `center_zdc` 候选：

```text
passive_zdc_source = center_zdc
alpha              = 36.337
```

但我们更愿意保留为物理候选的是 `volume_over_passive_length_area`，因为它把 passive 宽区截面积定义为：

```text
passive_area_m2 = passive_volume_m3 / passive_length_m
```

这一路径不依赖隐藏的 `length_scale` 或 `zdc_scale`。为避免把 `center_zdc` 的 branch balance 错套到 volume-area 候选上，本轮新增：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  make_area_candidate_branch_balance()

results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_branch_balance.csv
```

并在：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_candidate_metadata.json
```

的 `outputs.volume_area_branch_balance` 中记录该 CSV 路径。

真实 Berea v11 结果为：

```text
membrane_geometry_mode                  = diagnostic_two_length_active_passive_volume_area
passive_area_source                     = volume_over_passive_length_area
passive_branch_alpha                    = 0.0163545307975376
passive_eta_equivalent_alpha_times_eta0 = 1.63545e-4
titov_geometry_factor_median            = 0.33728
transport_number_contrast_needed_median = 0.02202
active_efficiency_a2_needed_for_delta_n_0p5_valid_fraction = 1.0
active_efficiency_a2_needed_for_delta_n_0p2_valid_fraction = 1.0
```

这使当前判断更扎实：`volume_over_passive_length_area` 虽然谱形略不如 `center_zdc_area` 完美，但它需要的等效迁移数差只有约 `0.02`，且在 Titov 几何因子口径下有完整可行解；因此它比 `center_zdc_area + alpha=36` 更适合作为“有物理解释的诊断候选”。后续若要把它推进为正式候选，需要继续把 `alpha` 从拟合系数替换为可测或可约束的 transport-number / Stern-layer mobility / EDL 连通参数。

## 2026-06-18 追加：用 Titov 几何因子和 transport-number difference 正向生成膜谱

在 volume-area branch balance 的基础上，本轮把 `alpha=0.0163545` 进一步改写为一个显式 transport-number 参数化：

```text
alpha = median(Titov geometry factor) * (Delta n)^2 / eta0
```

其中 `eta0` 是代码中的基础 membrane polarizability。新增接口：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  alpha_from_transport_number_contrast()
  make_transport_number_parameterized_component_spectrum()
```

真实 Berea v11 输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_component_spectrum.csv
```

该谱逐频率记录：

```text
membrane_geometry_mode
  = diagnostic_two_length_active_passive_volume_area_transport_number_parameterized
passive_branch_alpha_source
  = titov_geometry_factor_median_transport_number_difference
passive_transport_number_difference
  = 0.02202044683139602
passive_titov_geometry_factor_median
  = 0.33727630773011213
passive_branch_alpha
  = 0.016354530814657642
```

与旧的 fitted-alpha volume-area 谱相比，`delta_sigma_membrane_imag_s_m` 最大差异约 `1.05e-13 S/m`，`passive_branch_alpha` 最大差异约 `1.71e-11`。因此这一步没有重新调数值，而是把同一条膜谱改写为：

```text
volume-area 几何 + Titov 几何因子中位数 + Delta n ≈ 0.022
```

这比直接写 `alpha=0.0163545` 更接近论文支持的物理口径。仍需注意：这里使用的是全局 median geometry factor，而不是逐 throat 的完整 EDL/mobility 求解；因此它仍应保留 `diagnostic` 标记。

同日同步更新 `make_polarization_component_spectra.py`，使 diagnostic all-spectrum 不再丢失这些物理来源列：

```text
passive_branch_alpha_source
passive_transport_number_difference
passive_titov_geometry_factor_median
```

重新生成后的组合谱：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/diagnostic_all_volume_area_component_spectra/polarization_spectra_all_diagnostic_volume_area.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/diagnostic_all_volume_area_component_spectra/metadata.json
```

现在 metadata 明确记录：

```text
diagnostic_membrane_geometry_mode
  = diagnostic_two_length_active_passive_volume_area_transport_number_parameterized
diagnostic_membrane_passive_branch_alpha_source
  = titov_geometry_factor_median_transport_number_difference
diagnostic_membrane_passive_transport_number_difference
  = 0.022020446831396
diagnostic_membrane_passive_titov_geometry_factor_median
  = 0.3372763077301121
```

由于 transport-number 参数化谱与旧 fitted-alpha volume-area 谱数值等价，已有 full-grid AC3D sweep 的数值结论不需要因为这一步重新解释为新拟合；变化主要是 provenance 从“alpha 拟合系数”推进到“Titov geometry factor + Delta n”的显式物理参数化。

## 2026-06-18 追加：把 Delta n 转换为 mobility/transport-number 约束

Bücker-Hördt 的 narrow-pore 解释常用 mobility contrast 或 transport-number contrast 描述 active zone 与 passive zone 的差异。为避免 `Delta n≈0.022` 仍停留在抽象参数，本轮把它继续写成 neutral-passive 参考下的 active-zone 阴离子迁移数和等效 mobility contrast。

脚本更新：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  make_transport_number_parameterized_component_spectrum()

code/scripts/sip_simulation/make_polarization_component_spectra.py
  DIAGNOSTIC_MEMBRANE_PROVENANCE_COLUMNS
```

新增并传播的字段：

```text
passive_anion_transport_number_reference
active_anion_transport_number_inferred
neutral_passive_mobility_contrast_inferred
```

真实 Berea v11 输出：

```text
passive_transport_number_difference        = 0.02202044683139602
passive_anion_transport_number_reference   = 0.5
active_anion_transport_number_inferred     = 0.47797955316860397
neutral_passive_mobility_contrast_inferred = 1.092139702150098
```

这意味着，在 neutral passive zone `t_- = 0.5` 的简化口径下，当前 volume-area 膜候选只需要 active zone 的阴离子迁移数从 `0.5` 降到约 `0.478`，等效 mobility contrast 约 `1.09`。这比先前 `center_zdc_area` 候选隐含的高 alpha/高 contrast 解释温和得多，也更容易作为物理候选继续推进。

这仍不是最终闭合：该转换假设 passive zone 为中性对称参考，并把 throat-wise geometry factor 用中位数压缩为一个全局参数。下一步若继续增强物理性，应从 Bücker 2019 的 Stern/diffuse-layer 讨论或相关表面迁移率资料中寻找 `t_-≈0.478` / mobility contrast `≈1.09` 是否处在合理范围，并判断是否需要逐喉而非全局中位数参数化。

## 2026-06-18 追加：逐 throat Titov geometry factor 诊断

为检查“用中位数 Titov geometry factor 生成全局 alpha”是否隐藏了几何分布效应，本轮新增逐 throat 版本：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  alpha_from_transport_number_contrast_per_throat()
  branch_local_delta_sigma_with_multipliers()
  make_transport_number_per_throat_component_spectrum()
```

新输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_per_throat_component_spectrum.csv
```

该版本保持同一个：

```text
Delta n = 0.022020446831396
neutral-passive mobility contrast = 1.092139702150098
```

但不再把 Titov geometry factor 压成中位数；而是逐 throat 使用：

```text
alpha_i = geometry_factor_i * Delta n^2 / eta0
```

Berea v11 的逐 throat alpha 分布为：

```text
alpha_i median = 0.0163545308146576
alpha_i mean   = 0.0207317406064893
alpha_i min    = 1.62526726645494e-5
alpha_i max    = 0.0484900076772605
```

与中位数-alpha 版本比较，膜水相谱虚部最大绝对差异约 `8.75e-5 S/m`，最大相对差异约 `0.871`。用同一个 `field_attenuation=0.0609263438471689` 做 local-to-effective 近似后，相对 Niu membrane component：

```text
median geometry-factor alpha:
  peak_normalized_rmse = 0.02568
  median_ratio         = 0.88503
  ratio_at_peak        = 0.98817
  peak_freq            = 46415.888 Hz

per-throat geometry-factor alpha_i:
  peak_normalized_rmse = 0.05368
  median_ratio         = 0.81832
  ratio_at_peak        = 0.98698
  peak_freq            = 46415.888 Hz
```

这说明逐 throat 几何分布并不破坏膜峰频和峰值，但会降低整体谱形贴合；因此当前不应贸然把 full-grid all-spectrum 从 median geometry-factor 版本切换到 per-throat 版本。更合理的解释是：逐 throat geometry factor 是更局部物理的方向，但还需要同时考虑逐 throat transport-number/mobility 差异、EDL 连通性或 field attenuation 的局部变化。当前结果包保留 per-throat 谱作为诊断对照，正式 diagnostic all-spectrum 暂继续使用数值等价于 Niu membrane component 的 median geometry-factor transport-number 参数化版本。

## 2026-06-18 追加：逐 throat 所需 Delta n_i / mobility contrast 表

为了判断“保持当前最贴近 Niu membrane component 的 volume-area 膜谱”在逐 throat 物理上是否需要离谱的 transport-number contrast，本轮新增逐喉反推表：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  make_required_transport_number_table()

results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_required_transport_number_by_throat.csv
```

该表对每条 throat 记录：

```text
titov_geometry_factor
passive_branch_alpha_target
transport_number_difference_required
active_anion_transport_number_required
neutral_passive_mobility_contrast_required
transport_number_difference_is_valid
```

真实 Berea v11 统计：

```text
rows           = 19272
valid_fraction = 0.999533

transport_number_difference_required:
  min    = 0.01279
  p05    = 0.01284
  median = 0.02200
  mean   = 0.04286
  p95    = 0.13621
  max    = 0.49289

active_anion_transport_number_required:
  min    = 0.00711
  p05    = 0.36379
  median = 0.47800
  mean   = 0.45714
  p95    = 0.48716
  max    = 0.48721

neutral_passive_mobility_contrast_required:
  min    = 1.0525
  p05    = 1.0527
  median = 1.0920
  mean   = 1.2455
  p95    = 1.7489
  max    = 139.67

fraction Delta n_i > 0.1 = 0.0970
fraction Delta n_i > 0.2 = 0.0186
```

这给出比单一 `Delta n≈0.022` 更细的判断：绝大多数 throat 只需要温和的 transport-number/mobility contrast，支持 volume-area 膜候选作为物理上可解释的诊断模型；但少数低 Titov geometry factor throat 会要求很大的 mobility contrast。后续若要把模型从 diagnostic 推向正式候选，需要处理这些尾部 throat，例如引入 EDL 连通阈值、surface mobility saturation、或逐 throat `Delta n_i` 的上限，而不是让少数几何异常 throat 决定全局参数。

## 2026-06-18 追加：逐 throat Delta n_i 物理上限诊断

为了检查“给逐 throat transport-number contrast 加物理上限”是否会破坏膜峰频/峰值，本轮新增 capped diagnostic：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  make_capped_transport_number_table()
  summarize_capped_transport_number_table()
  make_transport_number_capped_component_spectrum()
```

新增输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_cap_0p1_by_throat.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_cap_0p1_component_spectrum.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_cap_0p2_by_throat.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_cap_0p2_component_spectrum.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_cap_summary.csv
```

用同一个 `field_attenuation=0.0609263438471689` 对比 Niu membrane component：

```text
median geometry-factor alpha:
  peak_normalized_rmse = 0.02568
  median_ratio         = 0.88503
  ratio_at_peak        = 0.98817
  peak_freq            = 46415.888 Hz

per-throat uncapped Delta n_i:
  peak_normalized_rmse = 0.05368
  median_ratio         = 0.81832
  ratio_at_peak        = 0.98698
  peak_freq            = 46415.888 Hz

cap Delta n_i <= 0.1:
  capped_fraction      = 0.09745
  peak_normalized_rmse = 0.03688
  median_ratio         = 0.81868
  ratio_at_peak        = 0.98757
  peak_freq            = 46415.888 Hz

cap Delta n_i <= 0.2:
  capped_fraction      = 0.01904
  peak_normalized_rmse = 0.02950
  median_ratio         = 0.81887
  ratio_at_peak        = 0.98787
  peak_freq            = 46415.888 Hz
```

这个结果说明：少数需要大 `Delta n_i` 的 throat 确实主要影响谱形肩部，而不是膜峰频。`Delta n_i <= 0.2` 只封顶约 `1.9%` 的 throat，仍能把 RMSE 保持在接近 median-alpha 版本的水平；`Delta n_i <= 0.1` 封顶约 `9.7%` 的 throat，谱形略差但峰频/峰值仍稳定。

因此，后续更物理的正式候选可以考虑把逐 throat `Delta n_i` 写成有上限的 EDL/transport-number 函数，而不是继续使用全局 fitted alpha。需要强调的是：这个 capped diagnostic 仍不是 Niu 2020 明确给出的参数，它只是把 Bücker-Hördt/Titov 的 active-passive transport-number 框架推进到更可约束的数值形式。若要正式采用，应进一步用独立的表面电导、zeta/CEC、Debye 长度或离子迁移率资料约束 `Delta n_i` 的上限和 throat-wise 分布。

## 2026-06-18 追加：用 Debye 长度/EDL 覆盖比例约束逐 throat Delta n_i

Bücker & Hördt (2013a) 和 Bairlein et al. (2016) 都强调：膜极化的 active-zone 离子选择性不应是孤立拟合参数，而应与孔半径、EDL、Zeta potential、Debye length、Stern/diffuse-layer mobility 等有关。Titov et al. (2002) 也把 active/passive transport-number difference 与 EDL 占据孔体积/截面的物理图像联系起来。因此本轮在 capped diagnostic 之后，又新增了一个更接近物理约束的 throat-wise cap：

```text
edl_annulus_area_fraction = 1 - (1 - min(lambda_D_eff / r_throat, 1))^2
Delta n_i_cap = Delta n_max * edl_selectivity * edl_annulus_area_fraction
Delta n_i_used = min(Delta n_i_required, Delta n_i_cap)
alpha_i = Titov geometry factor_i * Delta n_i_used^2 / eta0
```

这不是说 Niu 2020 给出了 `lambda_D_eff`；相反，它明确把缺失参数暴露出来：若要让当前 volume-area 膜候选从 diagnostic 推向正式模型，需要用水化学或表面电化学资料约束 `lambda_D_eff` / `edl_selectivity`。

新增代码入口：

```text
code/scripts/sip_simulation/fit_niu2020_dual_length_membrane.py
  edl_annulus_area_fraction()
  make_edl_limited_transport_number_table()
  make_transport_number_edl_limited_component_spectrum()
```

新增输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_edl_10nm_by_throat.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_edl_10nm_component_spectrum.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_edl_100nm_by_throat.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_edl_100nm_component_spectrum.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_transport_number_edl_summary.csv
```

真实 Berea v11 结果如下：

```text
lambda_D_eff = 10 nm:
  edl_limited_fraction        = 0.51749
  median_edl_area_fraction    = 0.04158
  median_edl_delta_n_cap      = 0.02079
  median_required_delta_n     = 0.02202
  peak_normalized_rmse        = 0.05312
  median_ratio                = 0.81806
  ratio_at_peak               = 0.98695
  peak_freq                   = 46415.888 Hz

lambda_D_eff = 100 nm:
  edl_limited_fraction        = 0.02257
  median_edl_area_fraction    = 0.37603
  median_edl_delta_n_cap      = 0.18802
  median_required_delta_n     = 0.02202
  peak_normalized_rmse        = 0.02943
  median_ratio                = 0.81861
  ratio_at_peak               = 0.98784
  peak_freq                   = 46415.888 Hz
```

这个诊断说明两个问题：

1. 膜峰频仍由 active/passive 几何长度控制；EDL cap 主要改变幅值和低频肩部，不会把峰频再次推偏。
2. 若只采用 `~10 nm` 的有效 EDL 厚度，超过一半 throat 的 `Delta n_i` 会被限制，低频肩部明显不足；若采用 `~100 nm` 的等效选择层厚度，只有约 `2.3%` throat 被限制，谱形接近当前最佳 volume-area 版本。

因此，当前代码已经不再依赖任意 `length_scale` 或 `zdc_scale` 来对齐膜峰频。剩下的关键物理问题变成：Berea/Niu 水化学和矿物表面是否支持 `10-100 nm` 量级的等效 EDL/选择层厚度，或是否需要用 `edl_selectivity < 1`、Stern-layer mobility、surface conductance、CEC/zeta 数据共同闭合。这个方向比继续调几何长度更有论文支撑，也更容易写入正式 provenance。

## 2026-06-18 追加：由 Niu 水相电导率反推 Debye length

Niu 2020 正文给出 Berea 样品由 NaCl 溶液饱和，水相电导率为：

```text
sigma_w = 0.043 S/m
```

脚本新增水化学审计输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_niu_water_chemistry_debye_audit.csv
```

采用 NaCl 单价电解质和无限稀释摩尔电导率近似：

```text
Lambda_NaCl ~= 0.01264 S m2/mol
I ~= sigma_w / Lambda_NaCl / 1000
lambda_D = sqrt(epsilon_r epsilon_0 k_B T / (2 N_A e^2 1000 I))
```

得到：

```text
inferred ionic strength = 0.003402 mol/L
inferred Debye length   = 5.216 nm
10 nm effective layer   = 1.92 x Debye length
100 nm effective layer  = 19.17 x Debye length
```

这一步把上一节的 EDL 诊断解释得更严格：`10 nm` 在 Niu 水化学下约为 `2 lambda_D`，属于普通 diffuse-layer 量级，但它会限制超过一半 throat，导致膜低频肩部偏弱；`100 nm` 才能让谱形接近当前最佳 volume-area 膜谱，但它约等于 `19 lambda_D`，不能简单写成 Debye layer thickness。它更可能代表粗网格下的“等效选择层”：包含 Stern/diffuse-layer 耦合、surface conductance、局部矿物表面粗糙度/纳米孔隙、或 CT 未解析的界面异质性。

因此，下一步正式物理闭合不能把 `lambda_D_eff=100 nm` 直接当作 Niu 参数；更合理的做法是把它拆成：

```text
effective_selective_thickness = lambda_D * edl_thickness_multiplier
Delta n_i_cap = Delta n_max * edl_selectivity * annulus_fraction(r_i, effective_selective_thickness)
```

并用独立资料约束 `edl_thickness_multiplier` 与 `edl_selectivity`。这也解释了为什么只靠真实 Debye length 会低估膜幅值：Niu 的框架本身承认 CT 尺度无法直接解析纳米界面层，原文也指出相关过程发生在 several-to-tens-of-nanometers 的薄层中。

## 2026-06-18 追加：EDL 等效选择层厚度扫描

为避免只比较 `10 nm` 和 `100 nm` 两个手选值，本轮新增 EDL effective-thickness scan：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_edl_effective_thickness_scan.csv
```

固定：

```text
Niu-inferred lambda_D = 5.216 nm
edl_selectivity = 1.0
Delta n_max = 0.5
```

扫描 `5-200 nm` 的等效选择层厚度后得到：

```text
effective thickness  thickness/lambda_D  limited fraction  RMSE     ratio_at_peak
5 nm                 0.96                0.9956            0.05714  0.98681
10 nm                1.92                0.5175            0.05312  0.98695
20 nm                3.83                0.3195            0.04254  0.98732
30 nm                5.75                0.2108            0.03942  0.98743
50 nm                9.59                0.0985            0.03605  0.98757
75 nm                14.38               0.0436            0.03225  0.98772
100 nm               19.17               0.0226            0.02943  0.98784
150 nm               28.76               0.0080            0.02635  0.98799
200 nm               38.35               0.0034            0.02534  0.98806
```

所有扫描点的膜峰频仍为 `46415.888 Hz`，与 Niu membrane component 的峰频一致。这再次确认：膜特征频率已经由 active/passive 几何长度闭合；EDL/transport-number 参数主要控制幅值和低频肩部。

物理解释上，若 `edl_selectivity=1`，要达到接近当前最佳 volume-area 膜谱，需要约 `150-200 nm` 的等效选择层厚度，也就是 `~29-38 lambda_D`。这个厚度已经明显超过普通 diffuse-layer 厚度，因此不能写成“Debye length 本身”。它更可能是 CT 网格和 PNM throat 对纳米界面效应的上尺度等效，包含：

- Stern layer 与 diffuse layer 耦合；
- 表面电导和沿面迁移；
- 未解析的矿物表面粗糙度、微孔/纳米孔和 contact asperity；
- Niu 文中指出的纳米尺度固液界面异质性。

因此，下一步若要进一步物理化，不能只把 thickness 调到 `200 nm` 后作为参数固定；应做二维参数约束：`effective thickness` 与 `edl_selectivity` 可以互相补偿，真正应由表面电导、CEC/zeta、Stern mobility 或独立 salinity sensitivity 约束。

## 2026-06-18 追加：effective thickness × EDL selectivity 二维扫描

上一节的一维扫描默认 `edl_selectivity = 1.0`。本轮继续新增二维扫描：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_edl_thickness_selectivity_scan.csv
```

扫描范围：

```text
effective thickness = 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000 nm
edl_selectivity     = 0.25, 0.50, 0.75, 1.00
Niu-inferred lambda_D = 5.216 nm
```

按 peak-normalized RMSE 排名前几位：

```text
thickness  thickness/lambda_D  selectivity  limited fraction  RMSE     ratio_at_peak
500 nm     95.87               1.00         0.00047           0.02496  0.98815
300 nm     57.52               1.00         0.00104           0.02497  0.98813
500 nm     95.87               0.75         0.00182           0.02508  0.98809
200 nm     38.35               1.00         0.00343           0.02534  0.98806
1000 nm    191.73              0.50         0.00970           0.02677  0.98796
```

每个 `edl_selectivity` 下的最佳组合为：

```text
selectivity  best thickness  thickness/lambda_D  limited fraction  RMSE
0.25         500 nm          95.87               0.06123           0.03384
0.50         1000 nm         191.73              0.00970           0.02677
0.75         750 nm          143.80              0.00182           0.02508
1.00         500 nm          95.87               0.00047           0.02496
```

这个二维扫描把参数补偿关系说清楚了：

- `effective thickness` 和 `edl_selectivity` 确实能互相补偿；
- 但如果 `edl_selectivity` 太低，例如 `0.25`，即使厚度很大，最大可用 `Delta n_i` 仍不足，RMSE 停在 `~0.034`；
- `edl_selectivity >= 0.5` 且有效厚度达到数百纳米时，谱形接近未受限的 volume-area transport-number 版本；
- 所有组合的膜峰频仍保持在 `46415.888 Hz`，再次说明峰频不是由这个 EDL 参数调出来的。

因此，当前最物理的表述应是：为了复现 Niu membrane component，模型需要一个在 PNM/CT 尺度上的“有效离子选择区”，其作用强度相当于 `edl_selectivity >= 0.5` 且 `effective thickness` 为数百纳米；这不是普通 Debye length，而是把未解析 Stern/diffuse-layer 耦合、表面电导、纳米粗糙度和 contact asperity 汇总后的上尺度参数。正式采用前，必须继续用独立表面电化学参数或 salinity sensitivity 约束它。

## 2026-06-18 追加：最佳 EDL 候选谱与 throat 表输出

为了让后续机制图和 full-grid/diagnostic all-spectrum 可以直接读取一个候选，而不是人工从二维扫描表中挑选，本轮新增“最佳 EDL 候选”输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_best_edl_candidate_selection.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_best_edl_candidate_by_throat.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_best_edl_candidate_component_spectrum.csv
```

选择规则不是盲目取平台端最小 RMSE，而是：

```text
1. 找二维扫描中的最小 peak-normalized RMSE；
2. 允许 RMSE 容忍度 0.0005；
3. 在容忍范围内选择 effective thickness 最小的候选。
```

当前选出的候选为：

```text
effective_edl_thickness = 200 nm
Niu-inferred Debye length = 5.216 nm
thickness / lambda_D = 38.35
edl_selectivity = 1.0
edl_limited_fraction = 0.00343
peak_normalized_rmse = 0.02534
ratio_at_paper_peak = 0.98806
peak_frequency = 46415.888 Hz
```

这个候选比二维扫描中的纯最优平台点（例如 `500 nm, selectivity=1`）更保守，因为它用更小的等效选择层厚度取得几乎相同的膜谱误差。它仍然不是 Niu 2020 明确给出的参数，而是一个“有文献机制支撑、但还需要独立表面电化学约束”的 diagnostic candidate。后续如果接入正式机制图，应在 provenance 中保留该选择规则和“非默认 Niu 参数”的状态。

## 2026-06-18 追加：最佳 EDL 候选接入 diagnostic all-spectrum

为避免后续机制图继续读取旧 fitted-alpha 或手工替换膜项，本轮把最佳 EDL candidate component 接入 `make_polarization_component_spectra.py` 的 diagnostic all-spectrum 链路。新增输出目录：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/diagnostic_all_best_edl_component_spectra/
```

关键文件：

```text
polarization_spectra_all.csv
polarization_spectra_all_diagnostic_volume_area.csv
metadata.json
```

`metadata.json` 已保留最佳 EDL 参数和选择规则：

```text
diagnostic_membrane_geometry_mode = diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited
diagnostic_membrane_edl_debye_length_m = 5.215629e-09
diagnostic_membrane_edl_thickness_multiplier = 38.346284
diagnostic_membrane_edl_selectivity = 1.0
diagnostic_membrane_passive_transport_number_edl_limited_fraction = 0.003425
diagnostic_membrane_edl_selection_rule = smallest_effective_thickness_within_rmse_tolerance
diagnostic_membrane_edl_selected_peak_normalized_rmse = 0.025335
```

这一步的意义是：后续绘制机制对比图时，可以直接读取 `diagnostic_all_best_edl_component_spectra/polarization_spectra_all_diagnostic_volume_area.csv`，而不是在绘图脚本里临时替换膜项。它仍然是 diagnostic all-spectrum，不是 Niu 2020 原始默认参数；但相较旧的 fitted-alpha 版本，它的 provenance 已经推进到 Titov geometry factor + throat-wise transport-number + EDL effective selectivity。

## 2026-06-18 追加：机制对比图脚本保留 EDL provenance

同日进一步更新 `plot_niu2020_conductivity_mechanism_comparison.py`，使最终机制图的 source data 和 provenance 能保留 EDL-limited transport-number 参数，而不是只记录旧的 `passive_branch_alpha`。

新增可追踪字段包括：

```text
edl_debye_length_m
edl_thickness_multiplier
edl_selectivity
maximum_transport_number_difference
passive_transport_number_edl_limited_fraction
edl_selection_rule
edl_selection_rmse_tolerance
edl_selected_peak_normalized_rmse
```

这一步只修复“读入/绘图/写 provenance 时丢失 EDL 物理来源”的问题，不重新定义已有 full-grid sweep。当前本地已有的

```text
simulation_sweeps/all_diagnostic_volume_area_x_figure8freq/
simulation_sweeps/all_diagnostic_volume_area_xyz_mean_figure8freq/
```

仍对应旧的 `diagnostic_two_length_active_passive_volume_area` full-grid 结果；不能把它们改名或解释为 best-EDL full-grid 结果。若后续要画真正的 best-EDL full-grid 机制图，需要先用

```text
source_data/diagnostic_all_best_edl_component_spectra/polarization_spectra_all_diagnostic_volume_area.csv
```

作为 all-spectrum 输入重新运行 AC3D sweep，然后再把 sweep CSV 和 EDL metadata 交给绘图脚本。当前代码已经为这一步保留了 source/provenance 接口。

## 2026-06-18 追加：best-EDL membrane full-grid x-direction 验证

本轮已经把 best-EDL 膜候选推进到真正 full-grid AC3D 层，而不再只停留在局部 component spectrum。新增膜机制 sweep：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_best_edl_x_figure8freq/sweep_results.csv
```

运行设置与旧 `membrane_dual_length_volume_area_x_figure8freq` 保持一致：

```text
grid             = 350 x 350 x 350
direction        = x
frequencies      = Figure 8 对应 40 个频点
dtype            = complex64
preconditioner   = fft
fft_reference    = pore
rtol             = 1e-5
```

该 sweep 使用的水相谱为：

```text
source_data/dual_length_membrane_diagnostic/dual_length_membrane_volume_area_best_edl_candidate_component_spectrum.csv
```

并对应如下 EDL/Titov diagnostic 参数：

```text
membrane_geometry_mode = diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited
effective_edl_thickness = 200 nm
Debye length inferred from Niu sigma_w = 5.215629 nm
edl_thickness_multiplier = 38.346284
edl_selectivity = 1.0
transport-number cap = 0.5
selection rule = smallest_effective_thickness_within_rmse_tolerance
```

与 Niu Figure 8 的 membrane component 对比输出：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_x_vs_paper_component_source_data.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_x_vs_paper_component_summary.json
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_membrane_best_edl_fullgrid_x_vs_paper_component.png
```

关键结果：

```text
common_frequency_count       = 38
peak_normalized_rmse         = 0.02610
paper_peak_frequency_hz      = 46415.888336
candidate_peak_frequency_hz  = 46415.888336
ratio_at_paper_peak          = 0.98795
median_ratio                 = 0.88056
min_ratio                    = 0.75167
max_ratio                    = 1.00669
all_solver_info_zero         = true
max_relative_residual_norm   = 9.77e-06
```

这说明：在 full-grid 场求解之后，best-EDL 膜候选仍然保持了正确的膜特征频率，并且峰值幅度已经基本贴合 Niu Figure 8 的 membrane component。这个结论比此前局部谱卷积更强，因为它通过了 350³ AC3D 场求解。

同时，本轮还生成了一张机制对比图，把 v11 `all/pore/membrane/interfacial` 原机制曲线与 best-EDL full-grid membrane diagnostic overlay 放在同一图中：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_conductivity_mechanism_comparison_v11_x_with_best_edl_membrane.png
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_conductivity_mechanism_comparison_v11_x_with_best_edl_membrane_source_data.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/niu2020_conductivity_mechanism_comparison_v11_x_with_best_edl_membrane_provenance.md
```

该图件 provenance 明确记录 `EDL-limited transport-number parameters`，并继续标注 `diagnostic_not_default_niu2020_parameter`。因此，这不是恢复旧的 `length_scale` 或 `zdc_scale`，而是把缺失的膜选择性参数显式化为可审计的 Titov/EDL 上尺度参数。

当前剩余差距已经不主要在膜峰频或膜峰值，而在总响应：`pore`、`interfacial`、方向平均和 Figure 8 experimental/simulation 定义差异仍会影响 `all` 与实验曲线的对应关系。

## 2026-06-18 追加：true best-EDL all full-grid x-direction 验证

上一节的机制对比图仍可能被误解为“代数替换 all 曲线”。本轮进一步把 best-EDL 膜候选接入 paper-mode all spectrum 后，重新运行了真正的 full-grid all AC3D sweep：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/all_best_edl_x_figure8freq/sweep_results.csv
```

输入谱为：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/diagnostic_all_best_edl_component_spectra/polarization_spectra_all_diagnostic_volume_area.csv
```

运行设置：

```text
grid             = 350 x 350 x 350
direction        = x
frequencies      = Figure 8 对应 40 个频点
dtype            = complex64
preconditioner   = fft
fft_reference    = pore
rtol             = 1e-5
```

收敛检查：

```text
sweep_rows             = 40
all_solver_info_zero   = true
max_relative_residual  < 1e-5
```

新增机制图和 provenance：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_conductivity_mechanism_comparison_v11_x_with_true_best_edl_all.png
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_conductivity_mechanism_comparison_v11_x_with_true_best_edl_all_source_data.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/niu2020_conductivity_mechanism_comparison_v11_x_with_true_best_edl_all_provenance.md
```

该 provenance 明确写入：

```text
Diagnostic All Candidate
Full-grid CSV = simulation_sweeps/all_best_edl_x_figure8freq/sweep_results.csv
This optional curve is a true full-grid all-mechanism diagnostic candidate;
it is not an algebraic replacement of the default all curve.
```

用 `diagnose_niu2020_total_response_gap.py` 对 true best-EDL all sweep 与 Niu Figure 8 的 paper Simulation 和 Experiment 分别比较，输出为：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_total_response_gap_best_edl_all_x.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_total_response_gap_best_edl_all_x_summary.json
```

相对 Niu paper Simulation：

```text
common_frequency_count = 38
peak_normalized_rmse   = 0.05650
median_ratio           = 0.88190
ratio_at_reference_peak = 1.30961   # high-frequency dielectric-dominated peak at 1e10 Hz

<= 1e5 Hz:
peak_normalized_rmse   = 0.04650
median_ratio           = 0.66190
ratio_at_reference_peak = 1.08728

1 Hz to 1e5 Hz:
peak_normalized_rmse   = 0.04438
median_ratio           = 0.70337
ratio_at_reference_peak = 1.08728
```

相对 Niu Experiment（log-interpolated candidate）：

```text
common_frequency_count = 85
peak_normalized_rmse   = 0.03341
median_ratio           = 0.70166
ratio_at_reference_peak = 0.86468

<= 1e5 Hz:
peak_normalized_rmse   = 0.13161
median_ratio           = 0.44554
ratio_at_reference_peak = 0.61331

1 Hz to 1e5 Hz:
peak_normalized_rmse   = 0.12545
median_ratio           = 0.57516
ratio_at_reference_peak = 0.61331
```

对照 Niu paper Simulation 自身相对 Experiment：

```text
overall peak_normalized_rmse = 0.08070
overall median_ratio         = 0.64032
1 Hz to 1e5 Hz RMSE          = 0.11644
1 Hz to 1e5 Hz median_ratio  = 0.71705
```

解释：

1. best-EDL 膜项已经在 full-grid 膜机制层面对上了 Niu membrane component；
2. true best-EDL all full-grid 不是代数替换，已经通过 `350^3` 场求解；
3. all 响应在整体频带上与实验的 RMSE 低于 Niu paper Simulation vs Experiment 的 RMSE，但在 `1-1e5 Hz` 低中频段仍低于实验；
4. 因此剩余差距不能继续归咎于膜峰频或膜几何长度，下一步应集中检查 pore/interfacial 参数、方向平均、以及 Niu Figure 8 中 paper Simulation 与 Experiment 的定义差异。

## 2026-06-18 追加：best-EDL membrane 的 x/y/z 方向平均诊断

为确认 best-EDL 膜峰频是否依赖单一 `x` 方向求解，本轮继续运行了同一 membrane spectrum 在 `y`、`z` 方向的 full-grid AC3D sweep，并生成三向算术平均：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_best_edl_x_figure8freq/sweep_results.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_best_edl_y_figure8freq_corrected/sweep_results.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_best_edl_z_figure8freq/sweep_results.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_best_edl_xyz_mean_figure8freq/sweep_results.csv
```

`y` 方向原始 sweep 的最低频 `0.001 Hz` 未收敛：

```text
original_info                   = 1000
original_relative_residual_norm = 2.7937e-03
```

因此额外用邻近频点 `0.002154 Hz -> 0.001 Hz` 的顺序重跑最低频点，并只替换该失败行。修正记录保存在：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/simulation_sweeps/membrane_best_edl_y_figure8freq_corrected/correction_summary.json
```

修正后三个方向均满足：

```text
rows per direction       = 40
max_solver_info          = 0
max_relative_residual    < 1e-5
```

三向平均与 Niu Figure 8 membrane component 的对比输出为：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_membrane_best_edl_fullgrid_xyz_mean_vs_paper_component.png
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_xyz_mean_vs_paper_component_source_data.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_xyz_mean_vs_paper_component_summary.json
```

关键指标：

```text
common_frequency_count       = 38
peak_normalized_rmse         = 0.08345
paper_peak_frequency_hz      = 46415.888336
candidate_peak_frequency_hz  = 46415.888336
ratio_at_paper_peak          = 0.81316
median_ratio                 = 0.75232
min_ratio                    = 0.62212
max_ratio                    = 0.86576
```

解释：

1. 三向平均不会改变膜极化特征频率；峰频仍然与 Niu membrane component 一致；
2. 三向平均主要降低膜项幅值，峰值约为论文膜项的 `81%`；
3. 因此，当前“膜特征频率是否对得上”的问题已经不应再归因于方向平均、`length_scale` 或 `Zdc` 缩放；
4. 若采用 Niu 2020 正文中“across the REV”单一宏观电势梯度的口径，`x` 方向结果仍是当前最接近论文膜项的 full-grid 对照；三向平均应作为各向异性诊断，而不是自动替代论文口径。

## 2026-06-18 追加：结果包 manifest 已切换到 best-EDL provenance

此前 `results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/manifest.json`
和 `provenance/diagnostic_membrane_model_summary.json` 仍主要记录旧的 fitted-alpha
`diagnostic_two_length_active_passive_volume_area` 口径。代码现在已扩展
`build_niu2020_berea_result_package.py::write_diagnostic_membrane_model_summary()`：

1. 当 component spectrum 含 best-EDL 字段时，自动保留 `passive_branch_alpha_source`、
   `edl_debye_length_m`、`edl_thickness_multiplier`、`edl_selectivity`、
   `maximum_transport_number_difference`、`passive_transport_number_edl_limited_fraction`、
   `edl_selection_rule` 和 `edl_selected_peak_normalized_rmse`；
2. 当 full-grid summary JSON 是 best-EDL 对比常用的扁平指标格式时，也能写入
   `fullgrid_metrics`，包括 `peak_normalized_rmse`、`ratio_at_paper_peak`、
   `all_solver_info_zero` 和 `max_relative_residual_norm`；
3. README 会明确写出 `EDL-limited` 口径、Debye length、有效 EDL 厚度倍率和受
   EDL 限制的喉道比例。

实际结果包已刷新为：

```text
status = diagnostic_explicit_membrane_geometry_best_edl_package
membrane_geometry_mode = diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited
peak_normalized_rmse = 0.0261004613
ratio_at_paper_peak = 0.9879461153
edl_debye_length_m = 5.215629198e-09
edl_thickness_multiplier = 38.346284294
passive_transport_number_edl_limited_fraction = 0.0034246575
```

这一步不改变膜谱或场求解数值；它的意义是防止结果包继续显示旧的
`alpha` 诊断模型，从 provenance 层面把当前最佳膜模型明确固定为：

```text
Titov-style active/passive geometry factor
+ per-throat transport-number contrast
+ EDL-limited selectivity cap
+ no length_scale / no zdc_scale
```

## 2026-06-18 追加：膜分量对比已收束为可复跑脚本

此前 best-EDL membrane 与 Niu Figure 8 membrane component 的对比图、source data
和 summary 主要由临时内联 Python 生成。现在新增专用入口：

```text
code/scripts/sip_simulation/plot_niu2020_membrane_component_comparison.py
```

该脚本做四件事：

1. 只读取 `Figure8.xlsx` 中的 membrane component 列，不读取或冒充 paper Simulation 列；
2. 检查候选 full-grid sweep 的 `info` 与 `relative_residual_norm`，未收敛则拒绝输出；
3. 输出 membrane-only 对比图、source data 和 summary JSON；
4. 通过 `--candidate-metadata-csv` 把 best-EDL component spectrum 中的物理 provenance
   写入 summary，避免 full-grid sweep CSV 本身不含 EDL 字段时丢失模型来源。

已用该脚本重新生成：

```text
# x direction
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_membrane_best_edl_fullgrid_x_vs_paper_component.png
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_x_vs_paper_component_source_data.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_x_vs_paper_component_summary.json

# xyz mean
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/figures/niu2020_membrane_best_edl_fullgrid_xyz_mean_vs_paper_component.png
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_xyz_mean_vs_paper_component_source_data.csv
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/source_data/niu2020_membrane_best_edl_fullgrid_xyz_mean_vs_paper_component_summary.json
```

`x` 方向 summary 当前记录：

```text
common_frequency_count = 38
peak_normalized_rmse   = 0.0261004613
ratio_at_paper_peak    = 0.9879461153
all_solver_info_zero   = true
max_relative_residual_norm = 9.864e-06
paper_simulation_columns_used = false
```

并保留：

```text
membrane_geometry_mode = diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited
passive_branch_alpha_source = titov_geometry_factor_per_throat_edl_limited_transport_number_difference
edl_debye_length_m = 5.215629198e-09
edl_thickness_multiplier = 38.346284294
maximum_transport_number_difference = 0.5
edl_selection_rule = smallest_effective_thickness_within_rmse_tolerance
```

## 2026-06-18 追加：best-EDL 膜复现自动 verifier

为避免后续只凭图像判断“膜项对上”，新增自动审计入口：

```text
code/scripts/sip_simulation/verify_niu2020_best_edl_membrane_reproduction.py
```

默认审计对象为 x-direction best-EDL membrane full-grid sweep。它检查：

1. paper membrane component 峰频与 candidate 峰频是否一致；
2. `peak_normalized_rmse <= 0.03`；
3. `0.95 <= ratio_at_paper_peak <= 1.05`；
4. full-grid sweep 是否 `info=0` 且 `max_relative_residual_norm <= 1e-5`；
5. source data / summary 是否明确 `paper_simulation_columns_used = false`；
6. summary 是否包含 `diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited`
   和 EDL-limited provenance 字段。

真实结果包已生成报告：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/verify_niu2020_best_edl_membrane_reproduction_report.json
```

报告结论：

```text
passed = true
failed_checks = []
peak_normalized_rmse = 0.0261004613 <= 0.03
ratio_at_paper_peak = 0.9879461153 within [0.95, 1.05]
paper_peak_frequency_hz = candidate_peak_frequency_hz = 46415.888336
sweep rows = 40
max_info = 0
max_relative_residual_norm = 9.864e-06
paper_simulation_columns_used = false
best_edl_metadata_present = true
```

这份 verifier 把“膜极化模拟结果和 Niu 2020 membrane component 对得上”变成了可重复的文件级审计，而不是口头判断或图像观感。

## 2026-06-18 追加：机制对比图 provenance 已链接膜 verifier

`plot_niu2020_conductivity_mechanism_comparison.py` 现在支持：

```text
--diagnostic-membrane-verification-json
```

该参数会把 best-EDL membrane verifier 报告写入机制对比图 provenance。已刷新：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/niu2020_conductivity_mechanism_comparison_v11_x_with_true_best_edl_all_provenance.md
```

新增 provenance 小节：

```text
Diagnostic Membrane Verification
Report JSON = provenance/verify_niu2020_best_edl_membrane_reproduction_report.json
Passed = True
Peak-normalized RMSE = 0.02610046129914571
Ratio at paper membrane peak = 0.9879461152547286
Max relative residual norm = 9.863992636360334e-06
```

这一步的意义是：最终机制对比图不只显示 best-EDL diagnostic membrane/all 曲线，还能直接追溯到“膜项本身已经通过自动验证”的证据。这样审稿式复核时，可以从最终图件 provenance 反查到：

1. 哪个 membrane full-grid sweep 被验证；
2. 验证阈值是什么；
3. 是否使用 Figure8 paper Simulation 列；
4. best-EDL provenance 是否存在；
5. 求解器是否收敛。

## 2026-06-18 追加：best-EDL 膜参数中文配置文件

为了避免 best-EDL 参数只散落在 JSON metadata 和本文档中，结果包现在新增中文参数配置：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/configs/niu2020_best_edl_membrane_parameters_zh.py
```

该配置文件记录：

```text
membrane_geometry_mode
passive_branch_alpha_source
edl_debye_length_m
edl_thickness_multiplier
edl_selectivity
maximum_transport_number_difference
passive_transport_number_edl_limited_fraction
edl_selection_rule
edl_selected_peak_normalized_rmse
component_spectrum_csv
fullgrid_summary_json
verification_report_json
```

文件中用中文明确说明：该模型是 `diagnostic_not_default_niu2020_parameter`，不是
`length_scale`，不是 `zdc_scale`，也不是 Niu 2020 正文公开给出的默认参数。它的作用是把
Marshall-Madden active/passive membrane、Titov transport-number geometry factor、
Bucker-Hoerdt SNP/LNP 双时间尺度和 EDL-limited selectivity cap 串成一个可审计参数口径。

这个配置文件不改变任何数值结果；它只是把已经通过 verifier 的 best-EDL 参数固化为结果包内可读、可核对、可复跑的配置入口。

## 2026-06-18 追加：best-EDL 膜模型卡

结果包现在新增一页复核用模型卡：

```text
results/niu2020_berea_reproduction_20260617_v11_explicit_membrane_geometry_diagnostic/provenance/niu2020_best_edl_membrane_model_card.md
```

模型卡比长审计笔记更短，集中说明：

1. 验证目标是 Niu Figure 8 的 membrane component，不是总实验曲线本身；
2. best-EDL 模型身份、诊断属性和不属于 `length_scale/zdc_scale`；
3. Marshall-Madden、Titov、Bucker-Hoerdt、Bucker et al. 2019 的 DOI 和本地 PDF；
4. Debye length、effective EDL thickness、maximum transport-number difference 等关键参数；
5. full-grid AC3D membrane verifier 的通过结果；
6. x/y/z 方向平均主要影响幅值、不改变峰频；
7. 当前已知限制：all 响应仍受 pore/interfacial 机制影响，best-EDL 仍应保留 diagnostic 标签。

README 和 manifest 现在都链接该模型卡，便于从结果包首页快速进入“这个膜模型到底是什么”的复核入口。
