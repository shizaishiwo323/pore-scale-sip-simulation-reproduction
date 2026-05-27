# AC3D full-grid 模拟优化计划

日期：2026-05-20

## 背景

当前 full `350^3` Berea 单频求解已经跑通，但性能和收敛还不能支撑论文 Figure 6-8 所需的扫频模拟。

已完成的诊断运行：

- 输入：完整 Berea `350 x 350 x 350`
- 频率：`1 Hz`
- 方向：`x`
- 求解器：矩阵自由 `BiCGSTAB + Jacobi`
- 输出：`outputs/ac3d_matrix_free_full350_krylov_jacobi_20260520_retry1/matrix_free_single_result.json`
- 结果：`sigma_eff = 0.003593971376668855 + 0.00047860835566577547 i S/m`
- 终止：`iterations = 1000`、`info = 1000`
- 最终相对残差：`1.3499082792700935e-4`

主要问题：

1. 单频 full-grid 运行耗时约数小时。
2. `maxiter=1000` 后仍未达到 `rtol=1e-8`。
3. 当前没有残差历史，无法判断残差下降平台。
4. 当前 matvec 使用 `np.roll` 并重复计算 face conductance，产生大量临时数组和内存带宽压力。
5. Figure 6-8 需要扫频，不可能按当前速度直接跑 97 个 full-grid 频点。

## 总体目标

把当前“能跑通单频”的 Python 原型，升级为可用于扫频诊断和后续 GPU/native 加速的求解框架。

短期目标：

- 在 CPU 上加入残差历史、warm start、扫频调度。
- 减少 `np.roll` 临时数组和重复 face conductance 计算。
- 用 8-12 个代表频率完成可追踪的 full-grid 或代表性 REV 扫频诊断。

中期目标：

- 建立 GPU 原型，比较 CPU/GPU 在 `64^3`、`128^3`、full `350^3` 上的速度、显存和收敛稳定性。
- 比较 `complex64` 与 `complex128` 的误差和收敛。

长期目标：

- 支撑 Figure 6-8 全频段复现，包含 interfacial only、pore polarization only、membrane polarization only 和 all polarizations 四组结果。

## 阶段 1：CPU 求解器可观测性

状态：已完成，2026-05-20。

目的：先知道求解器为什么慢、残差怎么降。

任务：

1. 在 `solve_ac3d_matrix_free` 中暴露 `x0` 参数，传给 SciPy `bicgstab`。
2. 增加残差历史记录选项，例如：
   - `residual_every`
   - `residual_history_path`
   - `progress_every`
3. 每隔 N 次迭代计算一次 `||Ax-b|| / ||b||`，写入 CSV：
   - `iteration`
   - `elapsed_s`
   - `relative_residual_norm`
4. 记录最终状态：
   - `info`
   - `iterations`
   - `relative_residual_norm`
   - `rtol`
   - `atol`
   - `maxiter`
5. 在 `scripts/run_ac3d_matrix_free_single.py` 中增加命令行参数：
   - `--residual-every`
   - `--save-solution`
   - `--x0`
   - `--dtype`

预期产物：

- 更新 `src/pore_scale_electrical/ac3d_solver.py`
- 更新 `scripts/run_ac3d_matrix_free_single.py`
- 输出 `outputs/.../residual_history.csv`
- 输出可选初值文件 `solution.npy` 或 `solution.npz`

验收标准：

- `16^3` smoke run 能输出残差历史。
- `64^3` 或 `80^3` 代表子体积能画出残差曲线。
- full `350^3` 运行至少能每 10 或 20 步保存一次残差历史。

风险：

- 每次计算残差都需要额外一次 matvec；full-grid 不应每步计算，建议 `residual_every=10` 或 `20`。
- 保存 full-grid solution 约需 `350^3 * complex128 = 686 MB`，可接受但需要明确输出路径。

完成记录：

- `solve_ac3d_matrix_free` 已支持 `x0`、`residual_every` 和 `residual_callback`。
- `AC3DIterativeResult` 已新增 `residual_history` 字段。
- `scripts/run_ac3d_matrix_free_single.py` 已新增：
  - `--x0`
  - `--save-solution`
  - `--solution-path`
  - `--residual-every`
  - `--dtype`
  - `--atol`
  - `--matvec-backend`
- 单频 smoke 输出：
  - `outputs/ac3d_optimization_smoke_single/residual_history.csv`
  - `outputs/ac3d_optimization_smoke_single/solution.npy`
  - `outputs/ac3d_optimization_smoke_single_x0/matrix_free_single_result.json`
- 当前 `C:\Users\imgw\.conda\envs\ml` 环境未安装 `pytest`；已用轻量 runner 直接调用测试函数完成同等断言验证。

## 阶段 2：扫频 warm start

状态：已完成基础实现，2026-05-20。

目的：利用相邻频率解的连续性，减少每个频点的迭代次数。

任务：

1. 新增矩阵自由扫频脚本，例如：
   - `scripts/run_ac3d_matrix_free_sweep.py`
2. 频率按从低到高或从高到低排序运行。
3. 第一个频率使用 `x0 = 0`。
4. 后续频率使用上一个频率的解作为 `x0`。
5. 每个频率保存：
   - `result.json`
   - `residual_history.csv`
   - 可选 `solution.npy`
6. 支持断点续跑：
   - 若某频率已有 `result.json` 且 `--resume`，跳过或继续下一频率。
   - 若存在前一频率 `solution.npy`，自动作为初值。

建议首轮代表频率：

```text
1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8, 1e9 Hz
```

首轮容差建议：

- 诊断扫频：`rtol=1e-5` 或 `1e-6`
- 精算频点：`rtol=1e-8`

预期产物：

- `scripts/run_ac3d_matrix_free_sweep.py`
- `experiments/ac3d_matrix_free_sweep_*/config.yml`
- `outputs/ac3d_matrix_free_sweep_*/sweep_results.csv`
- `outputs/ac3d_matrix_free_sweep_*/frequency_*/result.json`
- `outputs/ac3d_matrix_free_sweep_*/frequency_*/residual_history.csv`

验收标准：

- 在 `64^3` 或 `128^3` 子体积上证明 warm start 比 cold start 迭代数更少。
- full `350^3` 至少完成 3 个相邻频点的 warm-start 诊断。
- 每个频点的失败、达到上限或收敛状态都能在总表中看出。

完成记录：

- 已新增 `scripts/run_ac3d_matrix_free_sweep.py`。
- 支持频率排序、`--resume`、`--save-solutions`、逐频点 `result.json`、逐频点 `residual_history.csv` 和总表 `sweep_results.csv`。
- 在同一进程中，后一个频点使用前一个频点的 `potential` 作为 `x0`。
- 小体积 smoke 已完成两个频点 `0.001 Hz` 和 `1 Hz`，第二个频点输出 `used_warm_start = true`。
- smoke 输出位于 `outputs/ac3d_optimization_smoke_sweep/`。

待补验证：

- 还需要用更有代表性的 `64^3` 或 `128^3` 子体积量化 warm start 相比 cold start 的迭代数收益。
- full `350^3` 的 3 个相邻频点诊断留到阶段 4 CPU benchmark 后执行。

## 阶段 3：去掉 `np.roll` 临时数组

状态：已完成基础 face-type/stencil 后端，2026-05-20。

目的：降低每次 matvec 的内存分配和内存带宽压力。

当前问题：

- `matrix_free_matvec` 每个轴都会调用多次 `np.roll`。
- `face_conductivity_forward` 每次 matvec 都重新计算调和平均 face conductance。
- full `350^3` 下每个 complex128 数组约 `686 MB`，几个临时数组就会显著拖慢。

优化思路：

1. 几何标签固定，只随频率改变材料复电导率。
2. 对每个方向的正向 face，只需要知道 face type：
   - pore-pore
   - solid-solid
   - pore-solid
3. 每个频率只计算三类 face conductance 标量：
   - `g_pp`
   - `g_ss`
   - `g_ps`
4. matvec 根据 face-type mask 或编码数组选择 conductance。
5. 优先实现切片版 stencil，避免 `np.roll`：
   - 内部区域用相邻切片差分。
   - 周期边界单独处理首尾切片。
6. 如内存允许，可预存三个方向的 face-type 编码：
   - `uint8`，每方向约 `42.9 MB`
   - 三方向约 `128.6 MB`

可能的数据结构：

```text
labels_uint8: shape (350, 350, 350)
face_type_x: uint8, 0=pp, 1=ss, 2=ps
face_type_y: uint8
face_type_z: uint8
```

预期产物：

- `build_face_type_codes(labels, pore_label, solid_label)`
- `matrix_free_matvec_face_types(face_types, conductances, vector)`
- 对应测试：
  - 与原 `np.roll` 版本在小网格上逐元素一致。
  - 均匀介质、串联/并联解析测试仍通过。

验收标准：

- `64^3` 上 matvec 结果与旧版本相对误差小于 `1e-12`。
- `128^3` 上单次 matvec 时间明显下降。
- full `350^3` 运行内存峰值降低或每迭代耗时下降。

风险：

- 周期边界处理容易出错，必须用小网格和解析算例测试。
- `face_type` mask 写法若产生布尔临时数组，也可能抵消优化；需要用编码数组和分支/切片谨慎实现。

完成记录：

- 已新增 compact face-type 数据结构和工具：
  - `FaceTypeConductivity`
  - `build_face_type_codes`
  - `face_type_conductance_values`
  - `face_type_conductivity`
  - `face_conductivity_arrays_from_types`
- 已新增无 `np.roll` 的 stencil 运算路径：
  - `matrix_free_matvec_faces`
  - `matrix_free_rhs_faces`
  - `jacobi_inverse_diagonal_faces`
  - `mean_current_density_faces`
  - `solve_ac3d_matrix_free_face_types`
- 新后端保留旧 `roll` 后端作为对照，可通过 `--matvec-backend roll|face-types` 选择。
- 已新增测试，确认 face-type matvec 与旧 `np.roll` matvec 在两相小网格上逐元素一致，face-type 求解器与直接求解器结果一致。

待补验证：

- 还需要阶段 4 benchmark 来量化 `64^3`、`128^3` 和 full `350^3` 的实际速度改善。

## 阶段 4：CPU benchmark 和扫频策略固化

目的：在进入 GPU 前，先有可靠 CPU baseline。

benchmark 尺寸：

- `32^3`
- `64^3`
- `128^3`
- 可选 `200^3`
- full `350^3`

记录指标：

- grid size
- dtype
- frequency
- solver
- preconditioner
- rtol
- maxiter
- iterations
- final residual
- elapsed total
- elapsed per iteration
- peak memory

预期产物：

- `scripts/benchmark_ac3d_matrix_free.py`
- `outputs/ac3d_benchmarks/cpu_benchmark.csv`
- `figures/ac3d_cpu_benchmark.png`

验收标准：

- 能量化旧 matvec 与 face-type matvec 的速度差异。
- 能判断 full-grid 扫频所需时间预算。

## 阶段 5：GPU 原型

状态：已完成基础原型与短迭代 benchmark，2026-05-20。

目的：利用 RTX 4070 Ti SUPER 加速 matvec 和 Krylov 迭代。

当前硬件：

- GPU：NVIDIA GeForce RTX 4070 Ti SUPER
- 显存：约 `16 GB`
- CUDA Driver：`12.7`

当前环境状态：

- 默认项目环境：`C:\Users\imgw\.conda\envs\ml`
- 已确认 `cupy-cuda12x = 13.6.0` 可用。
- CUDA runtime：`12090`
- GPU：`NVIDIA GeForce RTX 4070 Ti SUPER`
- `pip check`：无破损依赖。

任务：

1. 在 `ml` 环境中安装或确认 CuPy：

```powershell
conda activate C:\Users\imgw\.conda\envs\ml
pip install cupy-cuda12x
```

2. 新增 GPU 后端模块，例如：
   - `src/pore_scale_electrical/ac3d_gpu.py`
3. 先做最小 GPU matvec：
   - 输入 `labels` 或 `face_type`
   - 输入 `x`
   - 输出 `Ax`
4. 使用 `cupyx.scipy.sparse.linalg.bicgstab` 做原型。
5. 比较 `complex64` 和 `complex128`：
   - 速度
   - 显存
   - 收敛残差
   - 与 CPU 结果差异
6. 先跑：
   - `64^3`
   - `128^3`
7. 再尝试 full `350^3`。

完成记录：

- 已新增 `src/pore_scale_electrical/ac3d_gpu.py`。
- 已新增正式 GPU 运行入口：
  - `scripts/run_ac3d_matrix_free_gpu_single.py`
  - `scripts/run_ac3d_matrix_free_gpu_sweep.py`
- 已实现 GPU face-type 后端：
  - `GPUFaceTypeConductivity`
  - `face_type_conductivity_gpu`
  - `matrix_free_matvec_faces_gpu`
  - `matrix_free_matvec_face_types_gpu`
  - `matrix_free_rhs_faces_gpu`
  - `jacobi_inverse_diagonal_faces_gpu`
  - `mean_current_density_faces_gpu`
  - `solve_ac3d_matrix_free_gpu_face_types`
- 当前 CuPy 13.6.0 没有暴露 `cupyx.scipy.sparse.linalg.bicgstab`，只提供 `cg/cgs/gmres` 等接口；因此本阶段实现了一个本地 GPU BiCGSTAB 原型，以保持和 CPU 版相同 Krylov 家族。
- 已新增 `scripts/benchmark_ac3d_gpu.py`：
  - 支持 `--sizes`
  - 支持 `complex64/complex128`
  - 支持可选 CPU face-type 对照
  - 输出 `gpu_benchmark.csv`
  - 输出 `figures/ac3d_cpu_gpu_benchmark.png`
- `scripts/run_ac3d_matrix_free_gpu_single.py` 支持：
  - full-grid 单频 GPU 求解
  - `--x0`
  - `--save-solution`
  - `--residual-every`
  - `--dtype complex64|complex128`
  - 输出 `matrix_free_gpu_single_result.json`
  - 输出 `residual_history.csv`
- `scripts/run_ac3d_matrix_free_gpu_sweep.py` 支持：
  - full-grid 代表频点 sweep
  - 同一进程内 warm start
  - 每频点 `result.json`
  - 每频点 `residual_history.csv`
  - 总表 `sweep_results.csv`
- 已新增 `tests/test_ac3d_gpu.py`：
  - GPU matvec 与 CPU face-type matvec 在小网格上逐元素一致。
  - GPU 小网格求解与直接解一致。

验证结果：

- 单元测试：

```text
tests/test_ac3d_gpu.py + tests/test_ac3d_solver.py: 13 passed
```

- 非均一 `16^3` GPU/CPU 对比：
  - 输出：`outputs/ac3d_benchmarks_gpu_smoke_mixed/gpu_benchmark.csv`
  - `complex128`：GPU 与 CPU 有效电导绝对误差约 `1.28e-9`，potential 相对误差约 `5.17e-7`。
  - `complex64`：CPU SciPy BiCGSTAB 在第 8 步出现 breakdown (`info=-10`)，GPU 原型跑到 `maxiter=30`；说明 `complex64` 在当前问题上需谨慎，只适合性能探针或宽松诊断。

- `32^3/64^3/128^3` GPU-only 短迭代 benchmark：
  - 输出：`outputs/ac3d_benchmarks_gpu_stage5/gpu_benchmark.csv`
  - 图：`figures/ac3d_cpu_gpu_benchmark.png`
  - 设置：`1 Hz`、`rtol=1e-5`、`maxiter=20`、`crop_start=(0,0,0)`。
  - `128^3 complex64`：20 步约 `0.202 s`，约 `0.0101 s/iter`，显存增量约 `0.31 GiB`。
  - `128^3 complex128`：20 步约 `0.181 s`，约 `0.00905 s/iter`，显存增量约 `0.62 GiB`。

- full `350^3` GPU 探针：
  - 输出：`outputs/ac3d_benchmarks_gpu_full350_probe/gpu_benchmark.csv`
  - 设置：`complex64`、`1 Hz`、`maxiter=5`、`residual_every=1`。
  - 结果：5 步成功完成，约 `0.751 s`，约 `0.150 s/iter`。
  - 显存：总显存约 `15.99 GiB`，运行后可用显存从约 `14.74 GiB` 降至约 `8.32 GiB`，增量约 `6.41 GiB`。
  - 这只是显存/速度探针，`relative_residual_norm = 0.239`，不能作为收敛物理结果。

- full `350^3 complex64` 单频正式诊断：
  - 输出：`outputs/ac3d_gpu_full350_complex64_1hz_rtol1e-5/matrix_free_gpu_single_result.json`
  - 残差历史：`outputs/ac3d_gpu_full350_complex64_1hz_rtol1e-5/residual_history.csv`
  - 设置：`1 Hz`、`rtol=1e-5`、`maxiter=1000`、`residual_every=10`。
  - 结果：`iterations=1000`、`info=1000`，未达到 `rtol=1e-5`。
  - 最终残差：`4.6704574783890216e-4`。
  - 有效电导：`0.0036011177113702626 + 0.0004797702259475219 i S/m`。
  - 用时：build 约 `0.57 s`，solve 约 `78.50 s`，约 `0.0785 s/iter`。
  - 显存：增量约 `6.41 GiB`。

- full `350^3 complex128` 短探针：
  - 输出：`outputs/ac3d_gpu_full350_complex128_1hz_probe5/matrix_free_gpu_single_result.json`
  - 设置：`1 Hz`、`maxiter=5`。
  - 结果：5 步成功完成，残差 `0.23918517200210848`。
  - 用时：约 `5.46 s`，约 `1.09 s/iter`。
  - 显存：增量约 `13.41 GiB`，可用显存降至约 `1.33 GiB`；长跑存在较高 OOM 风险。

- `128^3 complex128` CPU/GPU 算子一致性：
  - 输出：`outputs/ac3d_gpu_cpu_consistency_128_complex128/consistency_result.json`
  - matvec 相对误差：`0.0`
  - RHS 相对误差：`0.0`
  - mean current 相对误差：约 `2.03e-15`

- full `350^3 complex64` 三频点 warm-start 诊断：
  - 输出：`outputs/ac3d_gpu_full350_complex64_warmstart_3freq_probe/sweep_results.csv`
  - 频点：`0.1 Hz`、`1 Hz`、`10 Hz`
  - 设置：`rtol=1e-4`、`maxiter=300`。
  - `0.1 Hz` cold start：300 步残差 `3.1075294150655717e-3`。
  - `1 Hz` warm start：300 步残差 `5.768575893917192e-4`。
  - `10 Hz` warm start：300 步残差 `3.157737899879776e-4`。
  - 说明：warm start 对相邻频率有明显收益，但当前 Jacobi + BiCGSTAB 仍未在 300 步内达到 `1e-4`。

- full-grid GPU 残差曲线：
  - `figures/ac3d_gpu_full350_residual_history.png`

显存估计：

- full `350^3` 单个 complex128 数组约 `686 MB`
- 单个 complex64 数组约 `343 MB`
- Krylov 迭代通常需要多条向量；complex128 full-grid 可能接近或超过可用显存，complex64 更可行。

预期产物：

- `src/pore_scale_electrical/ac3d_gpu.py`
- `scripts/benchmark_ac3d_gpu.py`
- `outputs/ac3d_benchmarks/gpu_benchmark.csv`
- `figures/ac3d_cpu_gpu_benchmark.png`

验收标准：

- `64^3` GPU 结果与 CPU 结果一致到可接受误差。
- `128^3` GPU 相比 CPU 有明确速度优势。
- full `350^3` 至少能完成若干迭代并记录显存占用。

当前验收状态：

- 小网格 GPU matvec 和 `complex128` 求解精度已通过测试。
- `64^3/128^3` 已完成 GPU 短迭代速度和显存记录。
- full `350^3 complex64` 已完成 1000 步单频诊断并记录残差历史。
- full `350^3 complex64` 已完成 3 个相邻频点 warm-start 诊断。
- full `350^3 complex128` 已完成 5 步可行性探针，但显存余量很小，不建议直接长跑。
- 待补：更强预条件器或更稳 Krylov 策略，否则仅靠 Jacobi 很难把 full-grid 快速推到 `1e-5` 或更低。

风险：

- `complex128` GPU 性能和显存压力可能不理想。
- `complex64` 可能导致 BiCGSTAB 收敛性变差，需要与 CPU 精度对比。
- `cupyx` 的 `bicgstab` 回调和残差记录能力需要单独验证。
- 已确认当前 CuPy 版本没有 `bicgstab`，因此需维护本地 GPU BiCGSTAB 或后续改用 GMRES/CGS 作为对照。

## 阶段 6：更强预条件器和 native 路线

状态：阶段 6A 已完成，2026-05-21。

目的：如果 Jacobi 不够，让 full-grid 在合理迭代数内收敛。

候选方向：

1. 频率连续 warm start。
2. Block Jacobi 或方向分裂预条件。
3. FFT/周期泊松型预条件器。
4. Algebraic multigrid，需确认是否支持 complex non-Hermitian 系统。
5. 复原 NISTIR 6269 / AC3D 原始 Fortran 迭代实现。
6. C++/CUDA stencil matvec + Krylov。

验收标准：

- full `350^3`、`1 Hz` 在更低 `maxiter` 内达到至少 `rtol=1e-5`。
- 关键频点可推进到 `rtol=1e-8` 或证明当前误差对 Figure 6-8 曲线影响可接受。

### 阶段 6A：GPU FFT Poisson preconditioner

目的：用周期边界条件下的常系数 Poisson 逆近似消除低频误差，替代过弱的 Jacobi 预条件器。

实现记录：

- 已在 `src/pore_scale_electrical/ac3d_gpu.py` 中新增：
  - `GPUPreconditioner`
  - `FFTReferenceConductance`
  - `poisson_reference_conductance_gpu`
  - `periodic_laplacian_eigenvalues_gpu`
  - `fft_poisson_inverse_denominator_gpu`
  - `make_fft_poisson_preconditioner_gpu`
- `solve_ac3d_matrix_free_gpu_face_types` 已支持：
  - `preconditioner="none"`
  - `preconditioner="jacobi"`
  - `preconditioner="fft"`
  - `fft_reference="mean-face"|"mean-abs"|"pore"`
- `scripts/run_ac3d_matrix_free_gpu_single.py` 和 `scripts/run_ac3d_matrix_free_gpu_sweep.py` 已暴露：
  - `--preconditioner none|jacobi|fft`
  - `--fft-reference mean-face|mean-abs|pore`
- 已新增测试，确认 FFT 预条件器在小网格上可用，且 GPU 求解仍与直接解一致。

算法说明：

- 预条件器近似为：

```text
M ≈ g_ref * L_periodic
M^{-1} r = ifftn(fftn(r - mean(r)) / (g_ref * lambda_k))
```

- `lambda_k` 为周期 7 点正 Laplacian 的谱特征值。
- 零频模设为 0，并保留第一个自由度的 gauge 修正。
- 默认 `g_ref` 使用三方向 face conductance 的复数加权平均 `mean-face`。

验证结果：

- `64^3 complex64`，`1 Hz`：
  - FFT：51 步达到 `9.211539121715463e-6`。
  - Jacobi：100 步仍为 `6.00149393763952e-3`。
  - 输出：
    - `outputs/ac3d_gpu_fft_smoke_64_complex64/`
    - `outputs/ac3d_gpu_jacobi_smoke_64_complex64/`

- `128^3 complex64`，`1 Hz`：
  - FFT：64 步达到 `9.630014299253041e-6`。
  - Jacobi：120 步仍为 `6.0058096861436774e-3`。
  - 输出：
    - `outputs/ac3d_gpu_fft_smoke_128_complex64/`
    - `outputs/ac3d_gpu_jacobi_smoke_128_complex64/`

- full `350^3 complex64`，`1 Hz`，cold start：
  - 输出：`outputs/ac3d_gpu_full350_complex64_1hz_fft_rtol1e-5/matrix_free_gpu_single_result.json`
  - 残差历史：`outputs/ac3d_gpu_full350_complex64_1hz_fft_rtol1e-5/residual_history.csv`
  - 设置：`rtol=1e-5`，`maxiter=250`，`preconditioner=fft`。
  - 结果：`iterations=159`，`info=0`，成功收敛。
  - 最终残差：`8.376333218379739e-6`。
  - 有效电导：`0.003590585641399417 + 0.0004781501913265306 i S/m`。
  - 求解耗时：约 `18.28 s`，约 `0.115 s/iter`。
  - 显存增量：约 `6.10 GiB`。

- 对照：full `350^3 complex64`，`1 Hz`，Jacobi：
  - 输出：`outputs/ac3d_gpu_full350_complex64_1hz_rtol1e-5/matrix_free_gpu_single_result.json`
  - 设置：`rtol=1e-5`，`maxiter=1000`，`preconditioner=jacobi`。
  - 结果：`iterations=1000`，`info=1000`，未收敛。
  - 最终残差：`4.6704574783890216e-4`。
  - 求解耗时：约 `78.50 s`。

- full `350^3 complex64`，三频点 FFT warm start：
  - 输出：`outputs/ac3d_gpu_full350_complex64_fft_warmstart_3freq_rtol1e-5/sweep_results.csv`
  - `0.1 Hz` cold start：170 步，残差 `9.67840688895565e-6`。
  - `1 Hz` warm start：2 步，残差 `9.75835164818174e-6`。
  - `10 Hz` warm start：1 步，残差 `9.607398370833645e-6`。

- 残差曲线：
  - `figures/ac3d_gpu_fft_full350_residual_history.png`

阶段 6A 结论：

- FFT Poisson 预条件器解决了当前 full-grid 收敛瓶颈。
- 对 full `350^3` 单频，FFT 预条件器把 Jacobi 的“1000 步未收敛”改为“159 步达到 `rtol=1e-5`”。
- 对代表频点 sweep，FFT + warm start 可以让相邻频点在 1-2 步内达到 `rtol=1e-5`。
- 下一步应把 Figure 6-8 的代表频点 sweep 改用 `preconditioner=fft`，并做 `complex64` 与关键频点 `complex128` 或 CPU/残差校验。

## 阶段 7：Figure 6-8 生产扫频

状态：第一轮 full-grid 代表频点 sweep 已完成，2026-05-21。

目的：把优化后的求解器用于论文图表复现。

运行分组：

1. interfacial only
2. pore polarization only
3. membrane polarization only
4. all polarizations

频率策略：

1. 第一轮：8-12 个代表频点，`rtol=1e-5` 或 `1e-6`。
2. 第二轮：对 Figure 6-8 曲线转折区加密。
3. 第三轮：关键频点精算到更严格容差。

输出：

- `outputs/ac3d_matrix_free_sweep_*/sweep_results.csv`
- `outputs/figure6_figure8_metrics.csv`
- `figures/reproduced_figure6.png`
- `figures/reproduced_figure7.png`
- `figures/reproduced_figure8.png`
- 每个频点的 `result.json`、`residual_history.csv`、运行日志。

验收标准：

- 每条曲线来源明确。
- 每个频点都有收敛状态。
- 图表能区分论文数据、复现数据和误差。
- 对不一致频段有明确原因分析。

第一轮完成记录：

- 新增 sweep 绘图脚本：
  - `scripts/plot_figure6_figure8_sweep_comparison.py`
- full `350^3 complex64`、GPU FFT Poisson 预条件器、12 个代表频点 sweep 已完成。
- 主 sweep 输出：
  - `outputs/ac3d_gpu_full350_complex64_fft_representative_12freq_rtol1e-5/sweep_results.csv`
- 针对中频收敛困难频点的 refine 输出：
  - `outputs/ac3d_gpu_full350_complex64_fft_refine_1e4_to_1e3_desc_rtol1e-5/sweep_results.csv`
- 合并后的 Figure 6-8 sweep 输入：
  - `outputs/figure6_figure8_stage7_gpu_fft_sweep.csv`
- 误差指标：
  - `outputs/figure6_figure8_stage7_gpu_fft_metrics.csv`
- 图表：
  - `figures/reproduced_figure6_gpu_fft_sweep.png`
  - `figures/reproduced_figure7_gpu_fft_sweep.png`
  - `figures/reproduced_figure8_gpu_fft_sweep.png`
  - `figures/ac3d_gpu_fft_stage7_sweep_convergence.png`

代表频点收敛状态：

```text
frequency_hz   iterations  residual_norm
1e-3           172         8.5292689e-6
1e-2           1           9.1135792e-6
1e-1           1           8.4577030e-6
1              1           8.4267738e-6
10             1           8.4613025e-6
100            1           9.9579478e-6
1e3            70          9.9582618e-6
1e4            418         9.3548197e-6
1e5            205         9.2542158e-6
1e6            84          9.5808810e-6
1e8            14          8.3620163e-6
1e9            11          2.4810072e-6
```

说明：

- 所有 12 个代表频点最终均 `info=0`，达到 `rtol=1e-5`。
- `1e3 Hz` 和 `1e4 Hz` 是当前最难收敛的中频段。
- 从低频升序 sweep 时，`1e3 Hz` 和 `1e4 Hz` 在 `maxiter=260` 内未达标。
- 反向局部 sweep `1e4 -> 1e3 Hz` 后：
  - `1e4 Hz` cold start 418 步达标。
  - `1e3 Hz` 用 `1e4 Hz` 解 warm start 后 70 步达标。
- 后续生产扫频建议采用分段频率路径，而不是简单单向升序：
  - 低频段：`1e-3 -> 100 Hz`
  - 中频段：从 `1e4 -> 1e3 Hz` 反向补算
  - 高频段：`1e5 -> 1e9 Hz`

第一轮曲线观察：

- full-grid GPU FFT 曲线已经能形成 Figure 6-8 的复现曲线，而不再是单点诊断。
- 实部电导随频率整体上升，趋势合理。
- 虚部电导和等效介电常数在高频端明显增大。
- 与论文 Figure 6-8 的实验/模拟曲线仍有幅值差异，尤其虚部与介电常数；这更可能来自物理模型分组、材料参数、极化机制拆分或方向/张量平均方式，而不再主要是线性求解未收敛问题。

## 建议执行顺序

1. 阶段 1：残差历史 + `x0` 支持。
2. 阶段 2：warm-start 扫频脚本。
3. 阶段 3：face-type/stencil matvec。
4. 阶段 4：CPU benchmark。
5. 阶段 5：GPU 原型。
6. 阶段 6：更强预条件器。
7. 阶段 7：Figure 6-8 生产扫频。

## 最近一次可执行任务

阶段 7 第一轮代表频点 full-grid sweep 已完成。下一步建议做：

```text
进入阶段 7 第二轮：围绕 1e3-1e5 Hz 中频转折区加密频点，并做 complex64/complex128 或 CPU 残差校验。
```

原因：

- 第一轮 12 个代表频点已经全部达到 `rtol=1e-5`。
- `1e3-1e4 Hz` 是当前收敛最难区间，也是曲线形态容易变化的中频段。
- Figure 6-8 的幅值差异需要通过物理分组、方向平均和关键频点精算来定位，而不是继续只优化单频求解器。
