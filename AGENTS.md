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
| `code/vendor/pnextract/` | 外部参考代码目录 | 5,731,780 bytes | 当前项目内保存的 pnextract 原始代码副本；包含源码、文档、第三方库和 `bin.7z` | 作为外部上游/参考代码保留原貌；不要在未记录来源和修改原因的情况下直接改写，项目自有包装或解析脚本应优先放在 `code/scripts/` 或 `code/src/` |
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
- `code/scripts/run_ac2d_microfluidic_sweep.py`：批量运行二维微流控 AC2D 频谱和时间序列。当前默认 `driver_mode=si03`，只用 SI-S03 提供 `sigma_w(t)`，主机制为 `maxwell`、`interface`、`all`；旧 CEC/Waxman-Smits 路线应作为显式 `paper-cec` 诊断模式。
- `code/scripts/calibrate_ac2d_mechanistic_params.py`：扫描 `Sigma_s` 和 Schwarz 特征长度 `r_char`，对机理型 AC2D 的 2.5 Hz 虚部量级做参数标定。
- `code/scripts/plot_ac2d_microfluidic_paper_style.py`：生成类似论文多面板结构图的频谱与界面图像组合图。
- `code/scripts/plot_ac2d_2p5hz_paper_comparison.py`：生成 2.5 Hz 实部/虚部随时间变化的论文数据对比图。
- `code/scripts/compute_paper_consistent_2p5hz_waxman_smits.py`：用补充材料中的 `phi(t)`、`Sw(t)`、`sigma_w(t)`、`CEC(t)` 复核论文 Waxman-Smits 公式；这是论文公式诊断，不是 AC2D 场求解。

### 当前结果目录

- `results/ac2d_microfluidic/interface_images_v1/`：早期全机制结果，包含孔/膜探索项；膜极化虚部过大，不作为当前论文对比主结果。
- `results/ac2d_microfluidic/interface_images_maxwell_grain_v1/`：Maxwell-Wagner + 颗粒极化早期版本。
- `results/ac2d_microfluidic/interface_images_paper_particle_params_v1/`：采用论文颗粒参数 `Sigma_s` 和 `D` 的 Maxwell-Wagner + Schwarz/Debye 颗粒极化版本，是当前 AC2D 颗粒参数对比的主要结果。
- `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`：当前推荐的机理型 AC2D 主结果目录；使用 SI-S03 的 `sigma_w(t)`，不使用 CEC/Waxman-Smits 作为源项。
- `results/ac2d_microfluidic/mechanistic_calibration_v1/`：机理参数扫描结果目录，用于记录 `Sigma_s`、`r_char`、`tau` 和 2.5 Hz 误差。
- `results/ac2d_microfluidic/paper_consistent_2p5hz_waxman_smits.csv`：论文公式复核输出，不代表本项目二维场求解。
- `figures/ac2d_microfluidic/`：二维微流控 SIP 的结构图、频谱图和论文对比图应按用途集中保存在此目录下。

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
