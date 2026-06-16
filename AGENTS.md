# AGENTS.md

本文件是 `sip模拟/` 项目的 agent 协作说明。当前外层仓库目录是 `C:\Users\imgw\Documents\Codex\SIP模拟`，真正的整理后项目根目录是：

```text
C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟
```

除非用户明确要求处理外层旧目录，后续代码、文档、数据清单、结果和图件都应优先在 `sip模拟/` 内读写。外层 `src/`、`scripts/`、`notes/`、`outputs/`、`figures/`、`pnextract/` 等目录是迁移前或同步中的历史/临时位置，不要把它们当作新的权威结构。

## 项目目标

本项目用于复现和扩展孔隙尺度 SIP/复电导率模拟，主线包括：

1. 复现 Niu et al. (2020) 的 Berea 砂岩三维 AC3D 复电导率/介电常数框架。
2. 使用 GPU matrix-free Krylov solver、FFT-Poisson 预条件和 warm start 完成 full `350^3` 扫频。
3. 按论文 Section 5.3 区分 `interfacial / pore / membrane / all` 单机制模拟，不把论文工作簿中的 simulation 列直接冒充为本项目模拟结果。
4. 将三维框架迁移到二维微流控 AC2D 场求解，用于对照 Lab on a Chip / GRL 方解石溶蚀 SIP 数据。
5. 维护数字岩心、分割体、pnextract 球棍网络和 Fiji/ImageJ 风格三维 HTML 可视化流程。

## 顶层结构

```text
sip模拟/
  AGENTS.md
  README.md
  DIRECTORY_TREE.txt
  MANIFEST.csv
  code/
    src/pore_scale_electrical/          # 核心 AC2D/AC3D 求解器和极化模型
    scripts/
      data_audit/                       # 原始数据审计
      digital_rock_visualization/       # 三维数字岩心和 Fiji 风格 HTML
      pore_network/                     # pnextract 输入、解析和球棍网络渲染
      sip_simulation/                   # Niu/AC2D/SIP 正式模拟、绘图和 provenance
    tests/                              # 贴近 code/src 和基础脚本的单元测试
    vendor/pnextract/                   # 上游/第三方 pnextract 参考代码副本
    pytest.ini
  configs/                              # 可复跑参数文件
  data/                                 # 当前项目内主数据副本
  data_inventory/                       # CT 样品清单、物性表和原始拷贝索引
  docs/
    notes/                              # 复现说明、审计、结果讨论
    plans/                              # 长线计划
    references/                         # Niu/NISTIR/pnextract 等参考文献副本
    validation/                         # 微流控 SIP 论文、补充材料和验证数据
    literature_review/                  # 写作和文献综述材料
    superpowers/                        # 历史计划记录
  environment/                          # Python/CuPy/GPU 环境快照
  figures/                              # 整理图件和交互 HTML
  notebooks/                            # 可复跑探索流程
  paper_data/                           # Figure 5-8 工作簿兼容副本
  results/                              # 模拟输出、source data、metadata、provenance
  tests/                                # 脚本/provenance/可视化高层测试
```

新增文件时优先遵守上面的结构。不要再向外层旧目录新增正式结果；如果为了兼容读取了外层旧文件，应在 metadata 或说明中写明来源。

## 只读输入与资料来源

以下路径默认视为只读输入，不要覆盖、移动或重命名：

- `data/Niu 2020data/microCT_Berea.raw`
- `data/Niu 2020data/microCT_Berea.tiff`
- `data/Niu 2020data/Figure5.xlsx`
- `data/Niu 2020data/Figure6.xlsx`
- `data/Niu 2020data/Figure7.xlsx`
- `data/Niu 2020data/Figure8.xlsx`
- `paper_data/Figure5.xlsx`
- `paper_data/Figure6.xlsx`
- `paper_data/Figure7.xlsx`
- `paper_data/Figure8.xlsx`
- `docs/references/JGR Solid Earth - 2020 - Niu - A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity.pdf`
- `docs/references/jgrb54470-sup-0001-2020jb020515-si.docx`
- `docs/references/NISTIR 6269.pdf`
- `docs/references/Pore-network extraction from micro-computerized-tomography images.pdf`
- `docs/validation/` 中的 Lab on a Chip / GRL 微流控论文、补充材料和 SI-S01/SI-S02/SI-S03 CSV。
- `data_inventory/ct_backed_samples_raw_copy_20260605/` 中的 CT 样品原始拷贝和物性表。
- `code/vendor/pnextract/`。这是上游参考代码副本，默认只读；需要项目自有封装时放到 `code/scripts/pore_network/` 或 `code/src/`。

如果必须生成转换版数据、裁剪数据或中间数组，写入 `results/`、`figures/`、`configs/` 或明确命名的子目录，并保留输入路径、参数和运行命令。

## 工作环境

默认 Python 环境：

```powershell
& 'C:\Users\imgw\.conda\envs\ml\python.exe' --version
```

常用工作目录：

```powershell
Set-Location 'C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟'
```

`code/pytest.ini` 设置了 `pythonpath = src`，所以运行 `code/tests` 时应在 `sip模拟/code` 下执行，或显式设置工作目录。高层 `sip模拟/tests` 多数会在测试内部处理脚本路径，但仍建议从 `sip模拟/` 根目录运行。

基础验证示例：

```powershell
Set-Location 'C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟\code'
& 'C:\Users\imgw\.conda\envs\ml\python.exe' -m pytest tests -q
```

高层脚本/provenance/可视化测试示例：

```powershell
Set-Location 'C:\Users\imgw\Documents\Codex\SIP模拟\sip模拟'
& 'C:\Users\imgw\.conda\envs\ml\python.exe' -m pytest tests -q
```

针对改动运行最小相关测试即可。涉及 GPU/full-grid 的脚本可能很重，先使用 smoke、小网格、单频或已有 metadata/provenance 测试。

## 代码入口约定

核心模块位于 `code/src/pore_scale_electrical/`：

- `ac3d_solver.py`：CPU/通用三维 AC3D 求解相关逻辑。
- `ac3d_gpu.py`：GPU matrix-free AC3D、FFT preconditioner、扫频相关核心。
- `ac2d_solver.py`：二维复电导率场求解器。
- `microfluidic_2d.py`：微流控图像裁剪、相标签、几何统计和 AC2D 参数组装。
- `polarization.py`：孔极化、膜极化和相关复电导率模型。

正式脚本优先使用分类目录中的真实入口：

- `code/scripts/sip_simulation/`：Niu 2020、AC2D、Figure 6-8、机制谱、校准、provenance。
- `code/scripts/pore_network/`：pnextract 输入准备、网络解析、Berea/样品球棍 HTML。
- `code/scripts/digital_rock_visualization/`：分割体、Fiji 3D Viewer 风格、轻量三维预览。
- `code/scripts/data_audit/`：raw / TIFF / 工作簿审计。

旧的 `code/scripts/*.py` 顶层脚本多为兼容 wrapper 或历史入口。新增功能和 bugfix 应优先落在分类目录的真实脚本里；只有兼容性确实需要时才同步 wrapper。

## Niu 2020 三维 AC3D 约定

当前 Niu 2020 Berea 正式路线使用 `data/Niu 2020data/`：

- `microCT_Berea.raw` 按 `350 x 350 x 350`、little-endian `uint16` 读取。
- 相标签：`1 = pore/water`，`2 = solid`。
- 体素边长：`2.8 um`。
- 正式 full-grid AC3D 不降采样，除非明确标为 smoke、diagnostic 或 sensitivity。

Figure 7/8 机制分离必须遵守 Section 5.3 的“only consider one polarization at a time”定义：

- `Interfacial / Maxwell`：水相和固相只赋予 dc conductivity 与高频介电常数；不加入孔极化或膜极化。
- `Pore polarization`：水相赋值为 `sigma_w + Delta sigma_pore*`；固相赋值为 `0`；不加入 Maxwell 固相/水相高频介电背景，不加入膜极化。
- `Membrane polarization`：水相赋值为 `sigma_w + Delta sigma_membrane*`；固相赋值为 `0`；不加入 Maxwell 背景，不加入孔极化。
- `All`：水相赋值为 `sigma_w + high-frequency permittivity + Delta sigma_pore* + Delta sigma_membrane*`；固相赋值为高频介电项。

不要把 `Figure8.xlsx` 中的 Simulation、Pore polarization、Membrane polarization、Interfacial polarization 列直接绘成“我们的模拟结果”。这些列只能用于论文对照、误差审计或 provenance 证明未使用。正式图必须能追溯到本项目 `sweep_results.csv`。

当前修正后的 membrane 诊断参数：

- `membrane_length_scale = 0.0446683592150963`
- `membrane_zdc_scale = 10.0`

这是一条显式、可追踪的诊断性修正，表示 pnextract 几何与论文作者内部孔喉几何/电阻定义不完全一致。不要把它写成论文隐藏原始参数。

推荐入口：

- `code/scripts/sip_simulation/compute_polarization_spectra.py`
- `code/scripts/sip_simulation/diagnose_niu2020_membrane_mismatch.py`
- `code/scripts/sip_simulation/make_polarization_component_spectra.py`
- `code/scripts/sip_simulation/run_ac3d_matrix_free_gpu_sweep.py`
- `code/scripts/sip_simulation/plot_niu2020_figure7_style_corrected.py`
- `code/scripts/sip_simulation/verify_niu2020_figure7_style_provenance.py`

关键结果/provenance：

- `figures/niu2020/niu2020_figure7_style_corrected_reproduction.png`
- `results/source_data/niu2020_figure7_style_corrected_reproduction_source_data.csv`
- `results/niu2020/niu2020_figure7_style_corrected_provenance.md`
- `results/niu2020_berea_full350_all_scaled_membrane_fft_x/sweep_results.csv`
- `results/niu2020_berea_full350_membrane_scaled_fft_x/sweep_results.csv`
- `results/niu2020_berea_full350_pore_fft_x/sweep_results.csv`
- `results/niu2020_berea_full350_interfacial_precision_merged/sweep_results.csv`

禁止恢复旧错误路线：

- 不要恢复或新增旧 `plot_niu2020_mechanism_components_vs_experiment.py`、`plot_niu2020_experiment_vs_ours.py` 作为正式入口。
- 不要把 unscaled `results/niu2020_berea_full350_all_fft_x/` 或 `results/niu2020_berea_full350_membrane_fft_x/` 作为正式模拟曲线。
- 不要把 `results/niu2020_berea_full350_interfacial_fft_x/` 的低频 interfacial 曲线直接绘入正式图；低频受求解残差底噪影响。

## 二维微流控 AC2D 约定

二维方向用于对照 Lab on a Chip / GRL 方解石溶蚀 SIP 实验，是独立验证工作流，不是 Niu 三维 AC3D 的简单降维。

输入：

- `data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/interface_images/timestep_*.png`
- `data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/global_evolution_log.csv`
- `docs/validation/2024gl111271-sup-0002-data set si-s01.csv`
- `docs/validation/2024gl111271-sup-0003-data set si-s02.csv`
- `docs/validation/2024gl111271-sup-0004-data set si-s03.csv`

几何与相标签：

- 有效模拟域由红色区域和黄色区域的共同外接矩形确定；白色画布边界只作显示背景，必须从尺寸标定、相统计和电学求解中剔除。
- `timestep_0001.png` 的有效域检测结果为 `1152 x 430 px`，对应 `0.4 x 0.15 cm`，像素尺度约为 `dx = 3.472 um/px`、`dy = 3.488 um/px`。
- 红色区域表示孔隙水/电解质通道，黄色区域表示方解石/颗粒固相。
- 当前第一版不考虑气相，不建立气泡相，不计算 `Sw < 1`。
- 通道方向默认为图像水平方向 `x`。AC2D 采用左右边界宏观电势差、上下 no-flux 绝缘边界，不显式模拟四电极位置和几何因子 `kG`。

机制边界：

- Maxwell-Wagner 极化通过水相/方解石相复电导率和介电常数差异进入场求解。
- Schwarz/Debye 颗粒极化主线使用 `C_p* = i omega tau / (1 + i omega tau) * Sigma_s`、`tau = r^2 / (2D)`，再由二维方解石-水界面长度密度映射到体积复电导率。
- `r_char` 是可标定局部特征长度，默认量级约 `10-20 um`；不要静默使用整个方解石投影等效半径。
- CEC/Waxman-Smits 只作为论文经验/等效电路公式复核路径，不属于当前 AC2D 主机制模型。

推荐结果目录：

- `results/ac2d_microfluidic/interface_images_si03_mechanistic_v1/`
- `results/ac2d_microfluidic/mechanistic_calibration_v1/`
- `figures/ac2d_microfluidic/`

## LKC/样品 SIP 与三维数字岩心

对 sample16/sample89 等 LKC 样品绘制 SIP 对比时：

- 物性/几何参数优先从 `data_inventory/ct_backed_samples_raw_copy_20260605/physical_properties/样品属性.xlsx` 查询。
- 横轴 `Frequency (Hz)` 默认 `1e-4` 到 `1e5`。
- 实部纵轴 `sigma' (S/m)` 默认 `1e-4` 到 `1e-1`。
- 虚部纵轴 `sigma'' (S/m)` 默认 `1e-7` 到 `1e-3`。
- 使用真正 log-log 坐标绘制原始正值，不要先 `log10` 再线性绘图，除非图名和轴标签明确写成诊断图。
- 实验虚部非正点只在 log 图中过滤，并在 metadata 或图例说明过滤规则。
- sample16 正式机制对比图必须使用 `LKC-16.csv` 实验频率点；实验频带外扩展只能标为 diagnostic / extrapolated。

样品 89 当前主要输入：

- `data_inventory/ct_backed_samples_raw_copy_20260605/sample_89_Grainstone/CT_slices/89seged.tiff`
- `data_inventory/ct_backed_samples_raw_copy_20260605/sample_89_Grainstone/CT_slices/segv2/6-seged.tiff`

标签约定：

- `0 = pore`
- `255 = solid`
- 其他像素值应作为独立组分记录，不要静默并入孔隙或固体。

分割数字岩心 HTML：

- 使用 `code/scripts/digital_rock_visualization/render_segmented_core_html.py` 生成 PyVista/VTK real mesh 与 volume actor 离线 HTML。
- 若目标接近 Fiji/ImageJ 3D Viewer，使用 `code/scripts/digital_rock_visualization/render_segmented_core_fiji3d_html.py`。
- 分割体 HTML 必须保留右上角孔隙率面板，孔隙率按原始全分辨率体素计数：`count(volume == 0) / volume.size`。
- 当前样品 89 `89seged.tiff` 孔隙率为 `8.28%`，对应 `31,235,076 pore px / 377,241,600 total px`。
- 当前只确认 `pixel_size_y_um = 1.7`；没有独立 x/z 标定前，脚本按各向同性 `voxel_spacing_um_xyz=[1.7,1.7,1.7]` 处理。论文级几何解释必须说明该假设。

pnextract 球棍网络：

- 使用 `code/scripts/pore_network/run_segmented_core_pnextract_ballstick.py` 生成 pnextract RAW/MHD 输入、运行 pnextract、解析 network CSV、渲染球棍 HTML。
- 默认 `downsample=4` 是为了本机运行和 HTML 交互可用；metadata 必须保留 `downsample`、`source_voxel_size_um`、`effective_voxel_size_um` 和输入路径。
- `figures/segmented_cores/` 中的 HTML 是可再生图件，不是原始输入。

Berea 孔隙网络可视化：

- `figures/berea_pore_network/berea_pore_network_75deg_specular_vivid_full.png` 是目标静态风格参考。
- 同类三维交互 HTML 应保留 PyVista/VTK 真实 sphere/tube mesh 与材质光照，使用 `pyvista.Plotter.export_html` 导出 VTK.js offline HTML。
- 不要退回 Plotly `scatter3d` marker 高光拼接方案，除非用户明确要求轻量文件优先并接受缺少真实球体材质高光。

## 文档和结果记录

每个复现实验或图件至少记录：

- 输入数据路径和版本。
- 关键参数、单位、随机种子和缩放因子。
- 运行命令或脚本入口。
- 输出路径。
- 与论文图表/实验数据的对应关系。
- 无法完全一致的原因，例如参数缺失、预处理不明、网格尺寸差异、数值精度、边界条件或模型简化。

建议位置：

- 阅读/推导/审计：`docs/notes/`
- 长线计划：`docs/plans/`
- 论文/补充资料副本：`docs/references/` 或 `docs/validation/`
- 运行配置：`configs/`
- 可再生数值结果和 metadata：`results/`
- 图件、HTML 和论文图：`figures/`
- notebook 探索：`notebooks/`

## 文件安全限制

禁止批量删除文件或目录。

不要使用：

- `del /s`
- `rd /s`
- `rmdir /s`
- `Remove-Item -Recurse`
- `rm -rf`

需要删除文件时，只能一次删除一个明确路径的文件：

```powershell
Remove-Item "C:\path\to\file.txt"
```

如果需要批量清理生成文件，应停止操作并请求用户确认或让用户手动删除。不要删除、覆盖或压缩原始论文资料和原始数据。

## 协作注意事项

- 修改已有文件前，先判断它是原始资料、第三方代码、兼容 wrapper、正式脚本还是可再生输出。
- 对不确定的论文细节，在 notes 或 metadata 中标注“待确认”，不要把猜测写成事实。
- 若论文、补充材料、Excel 工作簿和本项目结果存在矛盾，保留证据路径并记录具体差异。
- 自动化脚本应默认可重复运行；若会覆盖输出，应显式写入带参数名、版本名或时间语义的输出目录。
- 对重型 GPU/full-grid 计算，先在小切片、小网格、单频或 smoke 数据上验证逻辑。
- 修改可视化链路后，至少运行相关渲染/provenance 测试；修改求解器或极化模型后，至少运行对应 `code/tests`。
