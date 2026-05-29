# GPU AC3D Berea Reproduction Package

整理日期：2026-05-27

本目录是从原始复现仓库中抽取出的干净结果包，目的是把 GPU 加速求解、FFT-Poisson 预条件、扫频 warm start、机制拆分图 8 和关键计划文档集中到一个结构化位置，便于后续写论文、复核结果或继续开发。

## 项目简介

这个项目复现 Niu et al. (2020) 的孔尺度复电导率/介电常数模拟框架。核心思想是：

1. 从 Berea 砂岩二值 micro-CT 体数据构造水相/固相复电导率场。
2. 用孔极化和膜极化模型生成水相的频率相关附加复电导率。
3. 在 full `350^3` 体素网格上求解周期边界条件下的复数 AC3D 方程。
4. 用 GPU matrix-free Krylov solver、warm start 和 FFT-Poisson preconditioner 完成扫频。
5. 与论文 Figure 6-8 的实验/模拟数据对比，尤其对 Figure 8 做 interfacial、pore、membrane 和 all-polarization 机制拆分。

## 目录结构

```text
organized_gpu_ac3d_reproduction_20260527/
  README.md
  MANIFEST.csv
  code/
    src/pore_scale_electrical/      # 核心求解器与极化模型
    scripts/                        # GPU运行、绘图和谱生成脚本
    tests/                          # 单元测试
    vendor/pnextract/               # pnextract 原始参考代码副本
    pytest.ini
  docs/
    plans/                          # 两份主计划文档
    notes/                          # 关键复现说明和结果讨论
    references/                     # 论文、补充材料、NISTIR 和孔隙网络提取参考资料
  environment/                      # Python/CuPy/GPU环境快照
  figures/                          # 最新整理图件
  paper_data/                       # Figure 5-8 的论文表格数据
  results/
    source_data/                    # 汇总CSV和source data
    spectra/                        # paper-consistent机制输入谱
    sweeps/                         # GPU full-grid sweep原始结果目录
```

## 最重要的文件

- 两份计划文档：
  - `docs/plans/reproduction_framework_plan.md`
  - `docs/plans/simulation_optimization_plan.md`
- GPU核心代码：
  - `code/src/pore_scale_electrical/ac3d_gpu.py`
  - `code/src/pore_scale_electrical/ac3d_solver.py`
  - `code/src/pore_scale_electrical/polarization.py`
- GPU运行脚本：
  - `code/scripts/run_ac3d_matrix_free_gpu_single.py`
  - `code/scripts/run_ac3d_matrix_free_gpu_sweep.py`
  - `code/scripts/make_polarization_component_spectra.py`
- 参考资料与外部代码：
  - `docs/references/NISTIR 6269.pdf`
  - `docs/references/Pore-network extraction from micro-computerized-tomography images.pdf`
  - `code/vendor/pnextract/`
- 最新 Figure 8 机制拆分图：
  - `figures/paper_comparisons/figure8/figure8_paper_consistent_components_plus_all_vs_experiment_nature.svg`
  - `figures/paper_comparisons/figure8/figure8_paper_consistent_components_plus_all_vs_experiment_nature.png`
  - `figures/paper_comparisons/figure8/figure8_paper_consistent_components_plus_all_vs_experiment_nature.pdf`
  - `figures/paper_comparisons/figure8/figure8_paper_consistent_components_plus_all_vs_experiment_nature.tiff`
- 最新 Figure 8 source data：
  - `results/source_data/figure8_paper_consistent_components_plus_all_vs_experiment_source_data.csv`
- 最新 Figure 8 讨论：
  - `docs/notes/figure8_paper_consistent_components_plus_all_vs_experiment_discussion.md`

## 当前最终结论

GPU FFT-Poisson preconditioner 已经把 full `350^3` 单频求解从 CPU 版的长时间未充分收敛，推进到 GPU 上约十几秒到几十秒量级的收敛求解。代表性 12 频点 sweep 均达到 `rtol = 1e-5`。

Figure 8 的机制拆分已经按论文 Section 5.3 修正：

- `interfacial only`：水相和固相保留 dc conductivity + high-frequency permittivity，不加入 electrochemical increments。
- `pore only`：水相为 `sigma_w + Delta sigma_pore`，固相为 `0`。
- `membrane only`：水相为 `sigma_w + Delta sigma_membrane`，固相为 `0`。
- `all polarization`：水相为 `sigma_w + i omega epsilon_w + Delta sigma_pore + Delta sigma_membrane`，固相为 `i omega epsilon_s`。

修正后，pore-only 和 membrane-only 的高频虚部不再被界面/介电背景错误抬升；高频上升主要由 interfacial/all-polarization 曲线承担，和论文讨论一致。剩余主要差异集中在 membrane 贡献偏强，后续应重点检查 `Zdc`、throat length distribution 权重和 pnextract 几何估计。

## 使用环境

当前实际使用 Python 环境：

```text
C:\Users\imgw\.conda\envs\ml\python.exe
```

环境快照见：

- `environment/python_version.txt`
- `environment/pip_freeze.txt`
- `environment/gpu_check.txt`

## 复跑入口

注意：本整理包没有复制 full micro-CT 原始体数据 `microCT_Berea.raw`，因为它属于大体积原始输入。复跑时仍需从原仓库的 `论文数据/microCT_Berea.raw` 或相同路径读取。

示例：重新生成 paper-consistent 机制谱：

```powershell
& 'C:\Users\imgw\.conda\envs\ml\python.exe' code\scripts\make_polarization_component_spectra.py `
  --mode paper `
  --out-dir results\spectra\polarization_component_spectra_paper `
  --components interfacial pore membrane all
```

示例：full `350^3` GPU sweep：

```powershell
& 'C:\Users\imgw\.conda\envs\ml\python.exe' code\scripts\run_ac3d_matrix_free_gpu_sweep.py `
  --spectra results\spectra\polarization_component_spectra_paper\polarization_spectra_pore.csv `
  --dtype complex64 `
  --frequency-order ascending `
  --rtol 1e-5 `
  --maxiter 800 `
  --residual-every 50 `
  --progress-every 0 `
  --preconditioner fft `
  --out-dir results\sweeps\new_pore_sweep
```

## 验证状态

整理前完整测试通过：

```text
18 passed
```

测试文件已复制到 `code/tests/`。
