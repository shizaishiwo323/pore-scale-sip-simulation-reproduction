# AGENTS.md

本项目用于学习仓库中的论文资料，并尝试复现论文提出的孔隙尺度模拟框架。当前重点论文为：

- `../论文资料/JGR Solid Earth - 2020 - Niu - A Framework for Pore‐Scale Simulation of Effective Electrical Conductivity and Permittivity (1).pdf`
- `../论文资料/jgrb54470-sup-0001-2020jb020515-si.docx`
- `docs/references/JGR Solid Earth - 2020 - Niu - A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity.pdf`
- `docs/references/jgrb54470-sup-0001-2020jb020515-si.docx`
- `docs/references/Pore-network extraction from micro-computerized-tomography images.pdf`
- `docs/references/NISTIR 6269.pdf`
- `code/vendor/pnextract/`
- `docs/validation/A microfluidic chip for geoelectrical monitoring of critical zone processes.pdf`
- `docs/validation/Geophysical Research Letters - 2024 - Rembert - Microfluidics and Spectral Induced Polarization for Direct Observation and.pdf`
- `docs/validation/2024gl111271-sup-0002-data set si-s01.csv`
- `docs/validation/2024gl111271-sup-0003-data set si-s02.csv`
- `docs/validation/2024gl111271-sup-0004-data set si-s03.csv`
- `docs/validation/SIP_calcite_dissolution_simulation_parameters.md`

其中，Niu et al. (2020) 主论文和补充材料是当前复现的核心依据；`NISTIR 6269.pdf` 是 AC3D 数值框架和参考程序的基础资料；`Pore-network extraction from micro-computerized-tomography images.pdf` 与 `code/vendor/pnextract/` 是孔隙网络/微 CT 图像处理流程的重要辅助参考，应在涉及孔隙结构提取、分割假设、网络表征或几何连通性解释时一并查阅。

`docs/validation/` 中的 Lab on a Chip 与 GRL 微流控 SIP 论文、补充材料和数据表是二维微流控迁移与方解石溶蚀验证的主要参照。使用这些资料时，应区分论文实验测量、论文经验/半经验模型、以及本项目 AC2D 场求解结果，不能把三者混写为同一个模型。

当前本地仓库根目录：

- `C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\organized_gpu_ac3d_reproduction_20260527`

项目中的原始数据和论文图表数据位于：

- `../论文数据/microCT_Berea.raw`
- `paper_data/Figure5.xlsx`
- `paper_data/Figure6.xlsx`
- `paper_data/Figure7.xlsx`
- `paper_data/Figure8.xlsx`

## 当前项目文件概览

根目录当前包含论文资料、论文数据、复现代码、实验记录、输出结果和本说明文件。原始论文资料与原始数据均应视为只读输入。

| 路径 | 类型 | 大小 | 已知信息 | 使用注意 |
| --- | --- | ---: | --- | --- |
| `AGENTS.md` | Markdown 文档 | 12,887 bytes | 项目协作与复现说明 | 可按项目进展持续补充；文件大小会随内容更新变化 |
| `docs/references/JGR Solid Earth - 2020 - Niu - A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity.pdf` | PDF | 2,070,094 bytes | 当前项目内保存的 Niu et al. (2020) 主论文副本；题名为 *A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity of Porous Media in the Frequency Range From 1 mHz to 1 GHz* | 当前核心论文，优先阅读和引用；作为项目内只读参考资料，不要覆盖 |
| `docs/references/jgrb54470-sup-0001-2020jb020515-si.docx` | Word 文档 | 184,716 bytes | 当前项目内保存的 Niu et al. (2020) 补充材料副本 | 用于确认数据尺寸、参数、补充公式和额外实验细节；作为项目内只读参考资料，不要覆盖 |
| `docs/references/Pore-network extraction from micro-computerized-tomography images.pdf` | PDF | 950,843 bytes | 孔隙网络提取与 micro-CT 图像处理相关参考论文 | 用于辅助理解孔隙网络提取、图像分割、连通性与几何表征；不要把其中方法直接等同于 Niu et al. (2020) 的模拟设置，除非有明确对应证据 |
| `docs/references/NISTIR 6269.pdf` | PDF | 818,955 bytes | 当前项目内保存的 NISTIR 6269 副本；AC3D 参考程序和数值框架相关资料 | 用于追踪论文所用 AC3D 思路、参考代码背景和数值实现细节；作为项目内只读参考资料，不要覆盖 |
| `code/vendor/pnextract/` | 外部参考代码目录 | 5,731,780 bytes | 当前项目内保存的 pnextract 原始代码副本；包含源码、文档、第三方库和 `bin.7z` | 作为外部上游/参考代码保留原貌；不要在未记录来源和修改原因的情况下直接改写，项目自有包装或解析脚本应优先放在 `code/scripts/pore_network/`、`code/scripts/digital_rock_visualization/`、`code/scripts/sip_simulation/` 或 `code/src/` |
| `../论文资料/JGR Solid Earth - 2020 - Niu - A Framework for Pore‐Scale Simulation of Effective Electrical Conductivity and Permittivity (1).pdf` | PDF | 2,070,106 bytes | 18 页；题名为 *A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity of Porous Media in the Frequency Range From 1 mHz to 1 GHz*；作者 Qifei Niu, Chi Zhang, Manika Prasad；JGR Solid Earth 2020；DOI 相关标识 `10.1029/2020JB020515` | 当前核心论文，优先阅读和引用 |
| `../论文资料/jgrb54470-sup-0001-2020jb020515-si.docx` | Word 文档 | 184,716 bytes | 补充材料；内部包含 `word/document.xml`、页眉页脚、脚注/尾注、图片资源等 22 个 docx 组件 | 用于确认数据尺寸、参数、补充公式和额外实验细节 |
| `../论文资料/NISTIR 6269.pdf` | PDF | 818,955 bytes | 210 页；PDF 元数据标题为 `ir_cover.dvi` | 论文里面使用的模拟框架的AC3D参考代码来源 |
| `../论文数据/microCT_Berea.raw` | RAW 二进制数据 | 85,750,000 bytes | Berea sandstone microCT 数据；无自描述头信息；文件类型探测结果不可靠，应以论文/补充材料为准 | 读取前必须确认体素尺寸、数据类型、字节序、形状和相标签含义；不要直接覆盖或转换原文件 |
| `paper_data/Figure5.xlsx` | Excel 2007+ | 11,720 bytes | 1 个工作表：`Sheet1`；sheet XML 中有 50 个 row 标签 | 用于 Figure 5 数据复核和图表复现 |
| `paper_data/Figure6.xlsx` | Excel 2007+ | 13,019 bytes | 1 个工作表：`Sheet1`；sheet XML 中有 63 个 row 标签 | 用于 Figure 6 数据复核和图表复现 |
| `paper_data/Figure7.xlsx` | Excel 2007+ | 13,247 bytes | 1 个工作表：`Sheet1`；sheet XML 中有 87 个 row 标签 | 用于 Figure 7 数据复核和图表复现 |
| `paper_data/Figure8.xlsx` | Excel 2007+ | 18,267 bytes | 1 个工作表：`Sheet1`；sheet XML 中有 87 个 row 标签 | 用于 Figure 8 数据复核和图表复现 |

补充说明：

- `.DS_Store` 是 macOS 自动生成的桌面服务文件，不属于论文或复现资料。
- Excel 的 row 标签数量只表示当前 sheet XML 中检测到的行节点数量，不等同于已经完成的数据语义审计。
- `microCT_Berea.raw` 的文件大小可能与多种数据类型和网格形状组合匹配；在论文或补充材料未确认前，不要把任何维度推断写死到代码中。
- `docs/references/` 保存当前项目内可直接引用的文献副本；这些副本便于迁移和复现记录，但仍应作为只读参考资料处理。
- `code/vendor/pnextract/` 保存外部 pnextract 参考实现；如果需要修改，应优先复制或包装到项目自有代码中，并记录与原始版本的差异。

## 项目目标

1. 系统阅读并整理论文的研究问题、物理假设、数学模型、数值方法和实验设置。
2. 从论文和补充材料中还原模拟框架的关键流程，包括微观结构输入、孔隙/矿物相处理、电导率与介电常数计算、边界条件和结果后处理。
3. 使用仓库中的论文数据和 microCT 数据，逐步复现论文图表或核心数值结果。
4. 记录每一步复现的假设、参数、差异来源和验证结果，保证后续可以追踪、修改和重复运行。

## Niu 2020 三维 SIP 复现当前状态

当前 Niu 2020 Berea 复现的正式路线使用 `data/Niu 2020data/` 中的数据，不直接绘制论文工作簿里的 simulation/机制曲线作为本项目模拟结果。`microCT_Berea.raw` 按 `350 x 350 x 350`、little-endian `uint16` 读取，`1=孔隙/水相`、`2=固相`，体素边长 `2.8 um`，正式 full-grid AC3D 不降采样。

### 已确认的问题与修正

- 早期 pnextract 自动网络路线把 `throat_length_m` 和由 `R^2/(4G)` 反算的 `Zdc` 直接作为 Niu 膜极化模型的 `L/Zdc` 输入，导致膜极化峰从论文 Figure 8 的约 `4.6e4 Hz` 偏到约 `10 Hz`，低频虚部高出约 `70-80x`。这不是孔极化高频持续的主问题；孔极化高频主要是数值底噪/绘图尺度问题。
- 论文公式本身仍按 Titov 膜极化使用：`tau_m=L^2/(4D)`、`C_m*=1/Z_m*-1/Zdc`、`Delta sigma_w*=2C*/Lambda`。默认参数仍为 `D=1.3e-9 m^2/s`、`eta0=0.01`、`sigma_w=0.043 S/m`、`Lambda=2.7 um`、`Sigma_S=1.3e-9 S`。
- 当前修正采用显式、可追踪的膜极化输入缩放：`membrane_length_scale=0.0446683592150963`、`membrane_zdc_scale=10.0`。这表示当前 pnextract 几何与论文作者内部孔喉几何/电阻定义不完全一致；该缩放是诊断性修正，不应被写成论文原始隐藏参数。
- 旧 `results/niu2020_berea_full350_interfacial_fft_x/sweep_results.csv` 的低频 interfacial permittivity 趋势是数值残差被 `epsilon'=sigma''/(omega epsilon0)` 放大的结果，不是物理机制。正式 Figure 7 风格图应使用 `results/niu2020_berea_full350_interfacial_precision_merged/sweep_results.csv`：`f < 1 Hz` 使用由 complex128 probe 的可靠低频平台外推，`1-100 Hz` 使用 direct complex128 probe，`f > 100 Hz` 保留原 full350 AC3D sweep。该合并曲线只来自本项目求解结果和数值精度诊断，不使用论文 simulation/interfacial 列作为模拟输入。

### Niu Figure 7 机制分离约定

Niu et al. (2020) Figure 7 的 `Interfacial/Maxwell`、`Pore polarization`、`Membrane polarization` 和 `All` 不是用总曲线相减得到的增量分解，而是 Section 5.3 中“only consider one polarization at a time”的单机制材料赋值与场求解。后续画机制图、生成 component spectra、写 metadata 或解释 sample16/sample89 结果时必须按以下定义：

- `Interfacial / Maxwell`：水相和固相只赋予 dc conductivity 与高频介电常数；不加入孔极化 `Delta sigma_pore*`，不加入膜极化 `Delta sigma_membrane*`。该机制必须通过异质 solid-water 场求解体现 Maxwell-Wagner / interfacial polarization；若使用 `pnm-formation-factor` 快速代理，必须标注为 proxy，不能等同于完整 AC3D interfacial 结果。
- `Pore polarization`：水相赋值为 `sigma_w + Delta sigma_pore*`；固相赋值为 `0`；不加入 Maxwell 固相/水相高频介电背景，不加入膜极化。该曲线是“孔极化单机制模拟结果”，不是 `Pore - Maxwell`，也不是 `Maxwell + 孔极化扰动` 的普通叠加解释。
- `Membrane polarization`：水相赋值为 `sigma_w + Delta sigma_membrane*`；固相赋值为 `0`；不加入 Maxwell 固相/水相高频介电背景，不加入孔极化。该曲线是“膜极化单机制模拟结果”，不是 `Membrane - Maxwell`。
- `All`：水相赋值为 `sigma_w + 高频介电项 + Delta sigma_pore* + Delta sigma_membrane*`；固相赋值为高频介电项。该曲线用于与实验总响应对比，但不应被拆成简单线性相减的“贡献”。
- 若需要额外画 `Pore increment = Pore - Maxwell`、`Membrane increment = Membrane - Maxwell` 或其他差分诊断，文件名、图例和 metadata 必须显式写明 `increment/difference diagnostic`，不得标成 Niu Figure 7 的 `Pore polarization` 或 `Membrane polarization`。

### LKC 样品 SIP 绘图坐标规范

对 sample16/sample89 等 LKC 样品绘制实部/虚部频谱对比图时，默认采用与 Zhang/Fan 附图一致的 log-log 坐标和固定范围，便于跨样品和跨机制直接比较：

- 后续做 SIP 模拟时，`dynamic_pore_size_m` / `Lambda`、`sample length`、`sample diameter` 三个样品物性/几何参数应从 `data_inventory/ct_backed_samples_raw_copy_20260605/physical_properties/样品属性.xlsx` 查询；不要在脚本、metadata 或正文中沿用旧硬编码值，除非明确标注为 sensitivity/diagnostic。
- 横轴 `Frequency (Hz)`：`1e-4` 到 `1e5`。
- 实部纵轴 `sigma' (S/m)`：`1e-4` 到 `1e-1`。
- 虚部纵轴 `sigma'' (S/m)`：`1e-7` 到 `1e-3`。
- 纵轴应使用真正的对数坐标绘制原始正值 `sigma`，不要把数据先取 `log10(sigma)` 后再画成线性纵轴，除非图名和坐标轴明确标注为诊断图。
- 实验虚部若存在非正异常点，log 图只绘制 `sigma'' > 0` 的点，并在 metadata 或图例中说明过滤规则。
- sample16 的正式机制对比图必须使用 `LKC-16.csv` 的实验频率点本身，当前为 `0.1-20000 Hz`、`51` 个频点；不要把 `np.logspace(-3,4,80)` 之类扩展频率曲线直接标成正式实验同频对比。若为了显示孔极化峰形态需要扩展到实验外低频，图名、metadata 和正文必须标明为 `model diagnostic / extrapolated frequency extension`，并说明实验频带没有覆盖该部分。
- sample16 正式 PNM/AC3D 机制图应优先直接使用 pnextract 提取出的孔半径和孔喉长度，即 `pore_radius_scale=1`、`membrane_length_scale=1`。不要为了让孔/膜极化峰位贴合参考图而把这两个几何量缩放到经验值；若使用 `pore_radius_scale` 或 `membrane_length_scale`，只能作为明确标注的 sensitivity/diagnostic，不得作为论文正式参数。

### 正式代码入口

- `code/scripts/sip_simulation/compute_polarization_spectra.py`：生成孔/膜极化谱，支持显式 `--pore-radius-scale`、`--membrane-length-scale` 和 `--membrane-zdc-scale`，metadata 必须保留实际使用的缩放、有效 pore radius、有效 `L` 和有效 `Zdc` 统计。
- `code/scripts/sip_simulation/diagnose_niu2020_membrane_mismatch.py`：膜极化 mismatch 诊断与快速代理扫描，用于说明为什么需要 `L/Zdc` 修正；这不是最终 AC3D 场求解。
- `code/scripts/sip_simulation/make_polarization_component_spectra.py`：把 corrected 基础谱拆成 `interfacial`、`pore`、`membrane`、`all` 组件谱。
- `code/scripts/sip_simulation/run_ac3d_matrix_free_gpu_sweep.py`：正式 full-grid AC3D 求解入口，Niu 2020 常规 sweep 使用 `350^3` 全域、`direction=x`、`complex64`、`preconditioner=fft`、`rtol=1e-5`；低频 interfacial 需要单独使用 `complex128`、更严格 `rtol` 做精度诊断或使用已生成的 precision-merged 结果。
- `code/scripts/sip_simulation/plot_niu2020_figure7_style_corrected.py`：当前唯一推荐的 Niu Figure 7 风格机制复现图入口。实验散点只读 `Figure8.xlsx` 前三列；四条模拟曲线必须来自本项目 full350 AC3D `sweep_results.csv`。
- `code/scripts/sip_simulation/verify_niu2020_figure7_style_provenance.py`：验证 corrected Figure 7 风格图的数据来源，逐列确认 source data 与本项目 AC3D sweep 匹配，并记录论文 Figure8 simulation/机制列未作为绘图模拟源。

### 当前关键结果与 provenance

- Corrected Figure 7 风格图：`figures/niu2020/niu2020_figure7_style_corrected_reproduction.png`、`.svg`、`.pdf`。
- 图源数据：`results/source_data/niu2020_figure7_style_corrected_reproduction_source_data.csv`。
- 图源证明：`results/niu2020/niu2020_figure7_style_corrected_provenance.md` 和 `.json`。该证明应显示四条 `our_corrected_*` 曲线均 `matches_sweep_csv=true`，并显示 Figure8 中论文 simulation/pore/membrane/interfacial blocks 均 `used_as_plotted_simulation_source=false`。
- Corrected all sweep：`results/niu2020_berea_full350_all_scaled_membrane_fft_x/sweep_results.csv`。
- Corrected membrane sweep：`results/niu2020_berea_full350_membrane_scaled_fft_x/sweep_results.csv`。
- Unmodified pore sweep：`results/niu2020_berea_full350_pore_fft_x/sweep_results.csv`。
- Corrected interfacial sweep：`results/niu2020_berea_full350_interfacial_precision_merged/sweep_results.csv`。
- Original interfacial complex64 sweep：`results/niu2020_berea_full350_interfacial_fft_x/sweep_results.csv`，仅作为精度诊断和高频保留来源，不应直接作为正式 Figure 7 风格图的低频 interfacial 曲线。
- `results/niu2020_berea_full350_all_scaled_membrane_fft_x/ac3d_input_provenance.json` 和 `results/niu2020_berea_full350_membrane_scaled_fft_x/ac3d_input_provenance.json` 必须记录 `membrane_length_scale=0.0446683592150963`、`membrane_zdc_scale=10.0`、原始 `microCT_Berea.raw`、`350^3` shape、`2.8e-6 m` voxel size 和 full-resolution pnextract network。

### 禁止恢复的旧错误路线

- 不要恢复或新增以 `plot_niu2020_mechanism_components_vs_experiment.py`、`plot_niu2020_experiment_vs_ours.py` 为代表的旧 Niu 绘图入口；这些旧入口已删除，容易让未修正的 unscaled membrane 结果重新成为正式图。
- 不要把 `results/niu2020_berea_full350_all_fft_x/` 或 `results/niu2020_berea_full350_membrane_fft_x/` 中的未修正 unscaled 结果作为正式 Niu 复现图的模拟曲线。它们只能作为 mismatch 诊断和修正前对照。
- 不要把 `results/niu2020_berea_full350_interfacial_fft_x/` 的低频 interfacial 曲线直接绘入正式图；该旧曲线在低频受求解残差底噪影响，会产生错误的 `1/f` 型 permittivity 假象。
- 不要把 `data/Niu 2020data/Figure8.xlsx` 中的 Simulation、Pore polarization、Membrane polarization、Interfacial polarization 列直接绘成“我们的模拟结果”。这些列只能用于误差审计或论文对照；正式图必须有 provenance 证明模拟曲线来自本项目 AC3D sweep。

## 二维微流控 SIP 迁移状态

当前新增方向是把原三维孔隙尺度 SIP 框架迁移为二维微流控 AC2D 模拟，用于对照 Lab on a Chip/GRL 方解石溶蚀实验。该方向目前不是原 Niu et al. (2020) 三维 AC3D 的简单降维，而是一个针对二维通道图像、红黄相界面和 2.5 Hz 时间响应的独立验证工作流。

### 输入与几何约定

- 二维界面图像序列位于 `data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/interface_images/timestep_*.png`。
- 溶蚀时间和全局演化数据位于 `data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/global_evolution_log.csv`。
- 有效模拟域由红色区域和黄色区域的共同外接矩形确定；白色画布边界只作为显示背景，必须从尺寸标定、相统计和电学求解中剔除。
- `timestep_0001.png` 的有效域检测结果为 `1152 x 430 px`，对应实际尺寸 `0.4 x 0.15 cm`，像素尺度约为 `dx = 3.472 um/px`、`dy = 3.488 um/px`。
- 红色区域表示孔隙水/电解质通道，黄色区域表示方解石/颗粒固相。当前第一版不考虑气相，不建立气泡相，不计算 `Sw < 1`。
- 通道方向默认为图像水平方向 `x`。AC2D 当前采用左右边界宏观电势差、上下 no-flux 绝缘边界，不显式模拟 Lab on a Chip 的四电极位置和几何因子 `kG`。

### 当前机制边界

- Maxwell-Wagner 极化：已通过水相和方解石相的复电导率、介电常数差异进入 AC2D 场求解。这是相对比导致的界面极化。
- Schwarz/Debye 界面/颗粒极化：当前主线使用本项目自己的机理模型，形式为 `C_p* = i omega tau / (1 + i omega tau) * Sigma_s`、`tau = r^2 / (2D)`，再乘以二维方解石-水界面长度密度映射到体积复电导率。主线默认不再使用整个方解石投影等效半径作为 `r`，而使用可标定的局部特征长度 `r_char`，默认量级约 `10-20 um`。
- CEC/Waxman-Smits 界面电荷极化：仅作为论文经验/等效电路模型诊断路径保留，用于复核补充材料公式和量纲；不属于当前 AC2D 机理主模型，不应被标记为 AC2D 机制模拟结果。
- 孔极化和膜极化：来自三维/Niu 框架的探索性机制不宜直接用于二维微流控芯片论文对比。早期膜极化尝试导致 2.5 Hz 虚部异常放大，当前 Lab on a Chip 验证应优先使用 SI-S03 驱动的 `sigma_w(t)`、Maxwell-Wagner 相对比和 Schwarz/Debye 界面/颗粒极化。
- 术语上必须区分 Maxwell-Wagner 相对比界面极化、Schwarz 颗粒极化、CEC/EDL 界面电荷极化。三者都可能表现为界面相关响应，但参数来源、控制方程和物理含义不同。

### 当前代码入口

- `code/src/pore_scale_electrical/ac2d_solver.py`：二维复电导率场求解器。
- `code/src/pore_scale_electrical/microfluidic_2d.py`：微流控图像裁剪、相标签、几何统计和 AC2D 参数组装。
- `code/scripts/sip_simulation/run_ac2d_microfluidic_sweep.py`：批量运行二维微流控 AC2D 频谱和时间序列。当前默认 `driver_mode=si03`，只用 SI-S03 提供 `sigma_w(t)`，主机制为 `maxwell`、`interface`、`all`；旧 CEC/Waxman-Smits 路线应作为显式 `paper-cec` 诊断模式。
- `code/scripts/sip_simulation/calibrate_ac2d_mechanistic_params.py`：扫描 `Sigma_s` 和 Schwarz 特征长度 `r_char`，对机理型 AC2D 的 2.5 Hz 虚部量级做参数标定。
- `code/scripts/sip_simulation/plot_ac2d_microfluidic_paper_style.py`：生成类似论文多面板结构图的频谱与界面图像组合图。
- `code/scripts/sip_simulation/plot_ac2d_2p5hz_paper_comparison.py`：生成 2.5 Hz 实部/虚部随时间变化的论文数据对比图。
- `code/scripts/sip_simulation/compute_paper_consistent_2p5hz_waxman_smits.py`：用补充材料中的 `phi(t)`、`Sw(t)`、`sigma_w(t)`、`CEC(t)` 复核论文 Waxman-Smits 公式；这是论文公式诊断，不是 AC2D 场求解。
- `code/scripts/digital_rock_visualization/render_segmented_core_html.py`：针对分割后三维数字岩心 TIFF 生成 PyVista/VTK 离线交互 HTML。当前默认输入为 `data_inventory/ct_backed_samples_raw_copy_20260605/sample_89_Grainstone/CT_slices/89seged.tiff`，约定 `0=孔隙`、`255=固体`，自动识别所有像素值组分；HTML 右侧提供 `Surface mesh` 与 `Voxel volume` 两列组分显隐 checkbox、颜色选择器和 `Apply` 按钮，右上角显示按原始体素计数得到的孔隙率。当前 `pixel_size_y_um=1.7`，脚本默认按各向同性 `voxel_spacing_um_xyz=[1.7,1.7,1.7]` 写入 mesh/volume 和 metadata；若后续确认 x/z 尺度不同，应扩展参数而不是静默沿用各向同性假设。若目标是接近 ImageJ 的原始体数据分布，应优先使用 `--surface-mode none --voxel-mode volume --voxel-initial-visible solid`，避免把全分辨率数据先转成粗糙 surface mesh；若目标是轻量快速预览，可再配合 `--downsample`。
- `code/scripts/digital_rock_visualization/render_segmented_core_fiji3d_html.py`：参考 `C:\Users\imgw\Fiji.app\plugins\3D_Viewer-5.0.0.jar` 的 Fiji/ImageJ 3D Viewer 思路生成单 label-volume 离线交互 HTML。该脚本不覆盖 `render_segmented_core_html.py` 或 `run_segmented_core_pnextract_ballstick.py`，而是新增一条可视化路线：把原始 `0/255` 标签体作为一个 VTK volume actor 导出，通过网页端 LUT、Threshold、Transparency/Opacity 和 Color 控件模拟 Fiji 3D Viewer 的 `VOLUME` 模式与 `setThreshold`、`setTransparency`、`setColor` 行为。此路线用于避免全分辨率 surface mesh 的离散/块状感，默认 `--downsample 1 --initial-visible pore --interpolation linear`，因为样品 89 固体占比约 91.72%，默认显示孔隙相更利于观察内部结构。HTML 右上角还提供 `3D Clip` 面板，可选择 `X/Y/Z` 裁切平面、拖动 index slider/number input、选择保留 positive/negative 一侧，然后点击 `Apply` 才对 VTK volume mapper 应用 clipping plane，让三维体素本体像被切掉一半一样显示剩余部分；`Show full volume` 会清除 clipping plane。不要把它改成 2D canvas 预览切片，也不要改成拖动时实时裁切，因为全分辨率体数据会明显增加浏览器负担。
- `code/scripts/digital_rock_visualization/visualize_digital_rock_3d.py`：针对 RAW 数字岩心裁剪子体生成 marching-cubes 三维预览。原有 `--out` 仍输出 PNG；现支持可选 `--html-out`，会用 PyVista/VTK 把同一 mesh 导出为可鼠标旋转缩放的离线交互 HTML。该脚本适合做轻量 mesh 预览，不替代 `render_segmented_core_fiji3d_html.py` 的全分辨率 label-volume 查看。
- `3D Clip` 的 clipping plane 必须在视角拖动、缩放和后续 render 中保持稳定。网页端自定义 `vtkPlane` 的 `getOrigin()` 和 `getNormal()` 必须返回数组副本，而不是内部数组引用；否则 VTK.js 渲染时的坐标变换可能污染原始平面，导致 Apply 后一瞬间可见裁切、拖动视角后又看起来恢复完整体。当前脚本还通过 `window.segmentedCoreCurrentClipPlane` 和 render hook 持久化当前裁切状态，若 clipping plane 被清空或污染，会在下一次 render 前补回。
- `code/scripts/pore_network/run_segmented_core_pnextract_ballstick.py`：针对分割数字岩心生成 pnextract 输入、运行 `../pnextract/bin.7z` 解压出的 `pnextract.exe`、解析 node/link 网络文件，并调用 PyVista/VTK 球棍网络 HTML 渲染器。当前默认同样使用 `89seged.tiff`，`0=孔隙`，非孔隙组分映射为 pnextract 的 solid；默认 `--downsample 4` 以保证网络提取和 HTML 交互可用，默认 `--voxel-size-um 1.7`，因此 pnextract 输出的有效体素尺度为 `6.8 um`。如需全分辨率网络，可显式设置 `--downsample 1`，但应预期运行时间和 HTML 体积显著增加。
- `code/scripts/pore_network/render_berea_pore_network_html.py`：Berea/pnextract 球棍网络的 PyVista/VTK 离线 HTML 渲染器。除原有球体/管道材质渲染外，现支持 `--segmented-volume`、`--pore-value`、`--solid-value`，用于从原始分割体按像素计数计算孔隙率，并把孔隙率显示在 HTML 页面右上角。
- 旧的 `code/scripts/*.py` 入口目前保留为兼容 wrapper，会转发到分类目录中的真实脚本；新增功能或后续维护应优先修改分类目录里的真实脚本，不要把新逻辑写回 wrapper。

### 当前结果目录

- `results/ac2d_microfluidic/interface_images_v1/`：早期全机制结果，包含孔/膜探索项；膜极化虚部过大，不作为当前论文对比主结果。
- `results/ac2d_microfluidic/interface_images_maxwell_grain_v1/`：Maxwell-Wagner + 颗粒极化早期版本。
- `results/ac2d_microfluidic/interface_images_paper_particle_params_v1/`：采用论文颗粒参数 `Sigma_s` 和 `D` 的 Maxwell-Wagner + Schwarz/Debye 颗粒极化版本，是当前 AC2D 颗粒参数对比的主要结果。
- `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`：当前推荐的机理型 AC2D 主结果目录；使用 SI-S03 的 `sigma_w(t)`，不使用 CEC/Waxman-Smits 作为源项。
- `results/ac2d_microfluidic/mechanistic_calibration_v1/`：机理参数扫描结果目录，用于记录 `Sigma_s`、`r_char`、`tau` 和 2.5 Hz 误差。
- `results/ac2d_microfluidic/paper_consistent_2p5hz_waxman_smits.csv`：论文公式复核输出，不代表本项目二维场求解。
- `figures/ac2d_microfluidic/`：二维微流控 SIP 的结构图、频谱图和论文对比图应按用途集中保存在此目录下。
- `figures/segmented_cores/sample_89_Grainstone_89seged_solid255_pore0_interactive.html`：样品 89 分割数字岩心组分可视化 HTML。当前默认 `initial_visible=solid`，打开时只显示 `255` 固体的 surface mesh；voxel volume 默认不显示，可在右侧 `Voxel volume` 列分别勾选 `pore (0)` 或 `solid (255)`，再点击 `Apply` 观察体素/体绘制效果。
- `figures/segmented_cores/sample_89_Grainstone_89seged_fullres_voxel_volume_interactive.html`：样品 89 原始分辨率体渲染 HTML，使用 `--downsample 1 --surface-mode none --voxel-mode volume --voxel-initial-visible solid`，用于查看更接近 ImageJ 的原始体素结构分布；metadata 位于 `results/source_data/sample_89_Grainstone_89seged_fullres_voxel_volume_interactive_metadata.json`。
- `figures/segmented_cores/sample_89_Grainstone_89seged_downsample12_voxel_volume_interactive.html`：样品 89 推荐降采样体渲染 HTML，使用 `--downsample 12 --surface-mode none --voxel-mode volume --voxel-initial-visible solid`，用于在较小文件和更稳交互中查看同一组分分布；metadata 位于 `results/source_data/sample_89_Grainstone_89seged_downsample12_voxel_volume_interactive_metadata.json`。
- `figures/segmented_cores/sample_89_Grainstone_89seged_downsample6_voxel_volume_interactive.html`：样品 89 较细降采样体渲染 HTML，使用 `--downsample 6 --surface-mode none --voxel-mode volume --voxel-initial-visible solid`；文件仍较轻，但内置浏览器自动化加载时曾出现 native pipe 断开，优先把 `downsample12` 作为稳定预览版。
- `figures/segmented_cores/sample_89_Grainstone_89seged_fiji3d_viewer_style_volume_interactive.html`：样品 89 参考 Fiji 3D Viewer 的原始分辨率单 label-volume 交互 HTML，使用 `--downsample 1 --initial-visible pore --interpolation linear`，右侧面板提供相显隐、颜色、透明度控制，以及非实时 `3D Clip` 轴向体裁切 slider + `Apply` 控件；metadata 位于 `results/source_data/sample_89_Grainstone_89seged_fiji3d_viewer_style_volume_interactive_metadata.json`。
- `figures/segmented_cores/sample_89_Grainstone_89seged_fiji3d_viewer_style_downsample12_volume_interactive.html`：同一路线的轻量预览版，使用 `--downsample 12`，也包含同样的切片控件；metadata 位于 `results/source_data/sample_89_Grainstone_89seged_fiji3d_viewer_style_downsample12_volume_interactive_metadata.json`。
- `figures/segmented_cores/sample_89_Grainstone_89seged_pnextract_ballstick_interactive.html`：样品 89 基于 pnextract 的球棍孔隙网络交互 HTML。当前网络来自 `downsample=4` 的分割体，右上角孔隙率仍按原始全分辨率 TIFF 计算。
- `notebooks/sample_89_6seged_digital_rock_and_pore_network_workflow.ipynb`：针对 `data_inventory/ct_backed_samples_raw_copy_20260605/sample_89_Grainstone/CT_slices/segv2/6-seged.tiff` 的完整可复跑流程。该 notebook 在 Conda `ml` 环境中把多数标签 `1` 映射为固体 `255`，把标签 `2` 映射为孔隙 `0`，输出二值 TIFF/RAW、Fiji 风格全分辨率 HTML、pnextract 球棍网络 HTML，以及 `visualize_digital_rock_3d.py` 的 PNG 和交互 HTML 预览。当前体素统计为 `471,815 pore px / 27,000,000 total px`，孔隙率 `1.747%`。
- `figures/segmented_cores/sample_89_Grainstone_6seged_fiji3d_fullres_volume_interactive.html`：由上述 notebook 生成的 `6-seged` 全分辨率 Fiji/ImageJ 风格 label-volume 交互 HTML，输入二值体位于 `results/segmented_cores/sample_89_Grainstone_6seged_solid255_pore0.tiff`。
- `figures/segmented_cores/sample_89_Grainstone_6seged_pnextract_ballstick_interactive.html`：由上述 notebook 生成的 `6-seged` pnextract 球棍孔隙网络 HTML。当前网络使用 `downsample=4`，解析结果位于 `results/pnextract/sample_89_Grainstone_6seged/network_parsed/`，本次结果为 `42 pores / 17 throats`。
- `figures/segmented_cores/sample_89_Grainstone_6seged_visualize_digital_rock_pore_preview_interactive.html`：由 `visualize_digital_rock_3d.py --html-out` 生成的轻量 pore-phase mesh 交互 HTML；对应 PNG 预览为 `figures/segmented_cores/sample_89_Grainstone_6seged_visualize_digital_rock_pore_preview.png`。
- `results/source_data/sample_89_Grainstone_89seged_solid255_pore0_interactive_metadata.json`：样品 89 分割体 HTML 的输入、体素统计、组分 mesh、默认显隐、`pixel_size_y_um=1.7` 和材质参数记录。
- `results/source_data/sample_89_Grainstone_89seged_pnextract_ballstick_interactive_metadata.json`：样品 89 球棍网络 HTML 的输入 CSV、材质参数、孔隙率体素统计、pnextract 降采样和体素尺寸记录。
- `results/pnextract_inputs/sample_89_Grainstone_89seged/`：由 `run_segmented_core_pnextract_ballstick.py` 生成的 pnextract RAW/MHD 输入及 pnextract 原始输出文件。该目录是可再生结果，不应混入原始数据目录。
- `results/pnextract/sample_89_Grainstone_89seged/network_parsed/`：由 `parse_pnextract_network.py` 解析得到的 `pores.csv`、`throats.csv` 和 `network_summary.json`。

### 当前判断

- 当前推荐对比应使用 `AC2D SI03 机理模型`：SI-S03 只提供 `sigma_w(t)`，虚部来自 Maxwell-Wagner 和 Schwarz/Debye 机制；不要用 CEC/Waxman-Smits 等效电路作为 AC2D 主模型源项。
- 论文 CEC/Waxman-Smits 路径仍可用于验证论文经验模型是否能复算 SI-S02，但这只是诊断参考，不是本项目机制模型。
- 2.5 Hz 虚部量级对 `r_char` 很敏感。若 `D = 1.3e-9 m^2/s` 且希望 2.5 Hz 接近弛豫峰，`r_char` 应约为 `13 um`；使用整个方解石等效半径会让弛豫时间过长，导致虚部被压低。

## 工作原则

- 先读论文和补充材料，再写代码。不要在未确认论文定义、单位、边界条件和参数来源的情况下直接实现模型。
- 优先保持复现忠实性。若必须做简化，应在文档或代码注释中明确说明简化内容、原因和可能影响。
- 原始论文资料、原始数据和图表数据视为只读资产。不要覆盖、重命名或移动这些文件。
- 所有生成代码、笔记、转换数据、图表和中间结果应放在新建的清晰目录中，例如 `notes/`、`src/`、`scripts/`、`outputs/`、`figures/` 或 `experiments/`。
- 涉及数值计算时，应保存关键参数、随机种子、输入文件路径、输出文件路径和运行命令。
- 处理 `.raw`、`.xlsx`、`.pdf`、`.docx` 等文件时，优先使用可靠的解析库或标准工具，避免手写脆弱的二进制或表格解析逻辑。
- 第三方或上游参考代码放在 `code/vendor/` 下，默认视为可读参考而非项目自有实现；对其进行修改前应说明目的，并尽量保持改动最小、可追踪。

## 建议目录结构

如果后续需要组织复现工作，优先采用以下结构：

```text
notes/          论文阅读笔记、公式推导、方法拆解
src/            可复用的模拟框架代码
scripts/        一次性数据检查、转换、绘图和实验脚本
code/vendor/    第三方或上游参考代码副本，例如 pnextract
experiments/    复现实验配置、运行记录和参数文件
outputs/        模拟输出、中间数组、日志
figures/        复现图表和对比图
tests/          单元测试、数值回归测试和数据完整性检查
```

如创建新目录，应保持命名清楚、用途单一。不要把生成文件混入 `../论文资料/` 或 `../论文数据/`。

## 论文复现流程

1. 建立论文阅读笔记，至少覆盖：研究目标、输入数据、物理量定义、控制方程、边界条件、数值方法、主要参数、论文图表含义。
2. 检查补充材料，提取 microCT 数据尺寸、体素尺度、相分割方式、材料参数和图表数据来源。
3. 阅读 `NISTIR 6269.pdf`，将其作为 AC3D 参考程序和数值框架的背景资料；实现求解器时应区分 NISTIR 原始算法、Niu et al. (2020) 的改造使用方式和本项目的 GPU/FFT 实现。
4. 阅读 `Pore-network extraction from micro-computerized-tomography images.pdf`，并检查 `code/vendor/pnextract/`，将二者作为 micro-CT 孔隙网络提取和几何表征的背景参考；若其流程与 Niu et al. (2020) 不一致，应明确区分“背景参考”和“本论文复现依据”。
5. 对 `Figure5.xlsx` 至 `Figure8.xlsx` 做数据审计，记录每个工作表、列名、单位和对应论文图号。
6. 对 `microCT_Berea.raw` 做只读检查，确认形状、数据类型、取值范围和相标签含义；若论文未明确说明，不要擅自假定，需在笔记中标为待确认。
7. 先实现最小可验证模块，例如数据读取、相标签统计、简单体素可视化、边界条件构造，再逐步实现完整模拟。
8. 每完成一个复现步骤，应生成与论文结果的对比记录，包括图表、误差指标或解释性说明。
9. 对二维微流控工作流，应先确认图像相标签、白边剔除、像素尺度、时间轴和启用机制，再运行 AC2D 频谱或 2.5 Hz 时间响应对比。

## 代码与实验要求

- 本项目默认使用已经创建好的 Conda 环境：

```bash
conda activate C:\Users\imgw\.conda\envs\ml
```

- 运行 Python 脚本、安装依赖、执行测试或启动 notebook 前，应先激活该环境。
- 如果当前 shell 中 `conda` 不在 `PATH`，可直接调用 `C:\Users\imgw\.conda\envs\ml\python.exe`。
- 如需新增依赖，优先安装到 `C:\Users\imgw\.conda\envs\ml` 环境中，并记录依赖名称、用途和安装方式。
- Python 代码优先使用明确的函数边界和可测试模块，避免把完整流程写成不可复用的大脚本。
- 数值数组处理优先使用 `numpy`、`scipy`、`pandas`、`xarray`、`h5py`、`openpyxl` 等成熟库。
- 图表复现应标明数据来源、单位、坐标轴含义和与论文图号的对应关系。
- 对重型计算或大文件处理，先在小切片、小网格或抽样数据上验证逻辑，再运行完整数据。
- 不要把大型生成结果、缓存或临时文件无说明地堆在根目录。
- 如引入新的依赖，应说明用途，并尽量记录在 `requirements.txt`、`pyproject.toml` 或环境说明中。

## Berea 孔隙网络三维交互图规范

- `figures/berea_pore_network/berea_pore_network_75deg_specular_vivid_full.png` 是 Berea 孔隙网络可视化的目标静态风格参考。
- 若生成同类三维交互 HTML，应保留 PyVista/VTK 的真实三维 mesh 与材质光照路线，使用 `pyvista.Plotter.export_html` 导出 VTK.js offline HTML。
- 球体和管道的自然光泽必须来自绘图库/渲染器的材质参数，而不是叠加白色 marker、贴图点或其他后处理假高光。
- 当前推荐材质参数应与静态 PyVista 图一致：球体 `ambient=0.38`、`diffuse=0.76`、`specular=0.64`、`specular_power=46`、`smooth_shading=True`；管道 `ambient=0.42`、`diffuse=0.72`、`specular=0.42`、`specular_power=28`、`smooth_shading=True`。
- 推荐颜色保持球体 `#ff0000`、管道 `#004cff`、白色背景和灰色坐标盒；默认视角保持 `elev=18`、`azim=75`，并保留浏览器中的鼠标旋转、缩放和自适应显示。
- 不要为这类图退回 Plotly `scatter3d` marker 高光拼接方案，除非用户明确要求轻量文件优先并接受缺少真实球体材质高光。
- 由于真实 sphere/tube mesh 会嵌入 HTML，单文件体积可能明显大于 Plotly 版本；这是保留真实材质高光的可接受代价，应在结果说明中注明。

## 分割数字岩心三维可视化规范

- 当前样品 89 输入文件为 `data_inventory/ct_backed_samples_raw_copy_20260605/sample_89_Grainstone/CT_slices/89seged.tiff`。该文件是三维分割 TIFF 栈，当前已确认 `0` 表示孔隙，`255` 表示固体；若后续出现其他像素值，应把它们作为独立组分记录，而不是强行并入孔隙或固体。
- 分割体 HTML 可视化应使用 `code/scripts/digital_rock_visualization/render_segmented_core_html.py`，走 PyVista/VTK real mesh 与 volume actor 路线，导出离线单文件 HTML；不要改回 Plotly marker、二维切片拼接方案，或把体素显示伪装成另一套 surface mesh。
- 分割体 HTML 的所有组分 actor 必须在 Python 导出阶段保持 `Visibility=True`，以确保 PyVista/VTK 会把每个组分序列化到离线 scene；初始显隐只在 HTML 侧根据 metadata 和 `Apply` 逻辑设置。不要在导出前用 `actor.SetVisibility(False)` 隐藏默认不显示的组分，否则该组分会从 HTML scene 中缺失，后续无法通过 checkbox 切换出来。
- 分割体 HTML 控件依赖导出后注入的 VTK.js hook：`window.segmentedCoreRenderWindow`、`window.segmentedCoreActors` 和 `window.segmentedCoreVolumes`。`window.segmentedCoreActors` 应在 `o.synchronize(e.scene)` 完成后通过同步器上下文 `getInstance(id)` 从 `vtkOpenGLActor` scene id 注册；`window.segmentedCoreVolumes` 应从 `vtkVolume` scene id 注册。不要依赖 `renderer.getActors()` 这类在 PyVista 离线同步器里不稳定的路径。
- 分割体页面必须明确标注两种渲染列：`Surface mesh` 表示相界面三角面；`Voxel volume` 表示基于原始组分 mask 的体素/体绘制 volume actor。两列都必须支持 `pore (0)` 和 `solid (255)` 分别勾选、分别可视化，并通过 `Apply` 生效。
- 分割后数字岩心的默认颜色约定为孔隙 `#0077ff`、固相 `#ffff00`。普通分割体 HTML、Fiji/ImageJ 风格 volume HTML，以及 `visualize_digital_rock_3d.py --phase solid` 的 PNG/HTML 预览都应遵守该默认值；只有用户显式传入颜色参数或在网页控件中 Apply 新颜色时才覆盖。
- 若用户抱怨画面像“一个个像素格子”或过度离散，应先判断是否由 `downsample` 和 `Surface mesh` 等值面提取导致。接近 ImageJ 三维查看的导出优先使用 voxel volume 路线，尤其是 `--downsample 1 --surface-mode none --voxel-mode volume`；不要为了全分辨率查看强行生成 surface mesh。
- 若用户明确要求参考 Fiji/ImageJ 3D Viewer，应使用 `code/scripts/digital_rock_visualization/render_segmented_core_fiji3d_html.py` 新路线。该路线依据本机 Fiji `3D_Viewer-5.0.0.jar` 中的 `ContentConstants` 模式（`VOLUME`、`ORTHO`、`SURFACE`、`MULTIORTHO`）和 `Content`/`VoltexGroup` 的 `setThreshold`、`setTransparency`、`setColor`、`saturatedVolumeRendering` 行为，把分割体作为一个 label volume 并在 HTML 侧更新 LUT/opacity transfer function；不要删除或覆盖既有 `sample_89_Grainstone_89seged_solid255_pore0_interactive.html` 与 `sample_89_Grainstone_89seged_pnextract_ballstick_interactive.html` 对应脚本。3D 裁切查看通过同一个 VTK volume mapper 的 clipping plane 实现，不额外嵌入第二份 TIFF 数据；用户改变轴向或 index 后必须点击 `Apply` 才刷新三维裁切，拖动旋转视角后裁切面必须保持生效。
- 分割体 HTML 必须保留右上角孔隙率面板，孔隙率按原始全分辨率体素计数计算：`count(volume == 0) / volume.size`。样品 89 当前孔隙率为 `0.08279859909405538`，显示为 `8.28%`，对应 `31,235,076 pore px / 377,241,600 total px`。
- 组分显隐和颜色修改采用显式 `Apply` 按钮流程：用户先勾选/取消组分、选择颜色，再点击 `Apply`，脚本中的 `window.segmentedCoreApplyControls` 才把状态应用到 VTK actors。不要依赖实时 `input` 事件作为唯一触发方式，因为 VTK.js actor 初始化和浏览器原生颜色控件可能导致实时更新不稳定。
- 当前样品 89 的体素尺寸只补充了 `pixel_size_y_um = 1.7`。在没有独立 x/z 标定前，脚本按各向同性 `voxel_spacing_um_xyz=[1.7,1.7,1.7]` 处理；任何论文级几何量或物理单位解释都必须说明这一假设。
- 分割体到球棍网络应使用 `code/scripts/pore_network/run_segmented_core_pnextract_ballstick.py`，参考上游 `../pnextract` 的 maximal-ball 网络提取流程。脚本会把 `0` 映射为 pnextract pore/void，把非孔隙组分映射为 solid，写出 RAW/MHD，然后运行 pnextract、解析网络 CSV 并调用 `code/scripts/pore_network/render_berea_pore_network_html.py` 生成球棍 HTML。
- 球棍网络当前默认 `downsample=4`，这是为了让样品 89 的 pnextract 和 HTML 交互在本机可用；metadata 必须保留 `downsample`、`source_voxel_size_um`、`effective_voxel_size_um` 和输入路径，避免把该网络误读成全分辨率网络。
- 当前相关测试为 `tests/test_render_segmented_core_html.py`、`tests/test_render_berea_pore_network_html.py` 和 `tests/test_run_segmented_core_pnextract_ballstick.py`。修改这些可视化链路后，至少运行：

```powershell
C:\Users\imgw\.conda\envs\ml\python.exe -m pytest tests\test_render_berea_pore_network_html.py tests\test_render_segmented_core_html.py tests\test_render_segmented_core_fiji3d_html.py tests\test_run_segmented_core_pnextract_ballstick.py -q
```

## 验证标准

在声称完成某个复现目标前，应至少满足以下条件：

- 输入数据来源明确，关键参数可追踪。
- 运行命令或脚本入口明确。
- 输出结果保存到清晰路径。
- 与论文图表或数值结果有直接对比。
- 已说明无法完全一致的原因，例如论文参数缺失、数据预处理不明、随机性、网格尺寸差异或数值方法差异。
- 对二维微流控 AC2D 结果，必须说明是否考虑气相、是否显式考虑电极位置、是否启用 Maxwell-Wagner、Schwarz/Debye 颗粒极化、CEC/EDL 界面电荷项、孔极化或膜极化。
- 与 Lab on a Chip/GRL 数据对比时，应优先使用 `docs/validation/` 中的补充材料 CSV 和参数说明，并记录单位换算与几何假设。

## 文件安全限制

禁止批量删除文件或目录。

**不要使用：**

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件。

**正确示例：**

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量删除文件，应停止操作，并请求用户手动删除。

## 协作注意事项

- 在修改已有文件前，先确认文件用途和是否为原始资料。
- 不要删除、覆盖或压缩原始论文资料和原始数据。
- 对不确定的论文细节，应在笔记中标注“待确认”，不要把猜测写成事实。
- 若发现论文、补充材料和 Excel 数据之间存在矛盾，应保留证据路径，并记录具体差异。
- 任何自动化脚本默认应是可重复运行的；若会覆盖输出，应先显式提示或写入带时间戳/参数名的输出目录。
