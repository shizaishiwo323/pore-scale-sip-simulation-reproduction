下面这份可以作为你当前 **project-extracted 复现路线** 的补充计划，建议文件名直接定为：

```text
docs/plans/niu2020_dynamic_pore_size_supplement_plan.md
```

它的目标很单一：**从本项目自己的三维分割孔隙结构计算动态孔径 (\Lambda_{\mathrm{project}})，并替代当前写死的 Niu 2020 表值 (2.7,\mu m)**。Niu 2020 中 (\Lambda) 的用途是把孔/膜极化复电导 (C^*) 转成水相体积复电导率增量 (\Delta\sigma_w^*=2C^*/\Lambda)，并且对复杂微结构需要通过局部电场数值求解确定 (\Lambda)。 当前项目代码中 `PolarizationParameters.dynamic_pore_size_m` 仍默认等于 `2.7e-6`，且 `upscale_conductance_to_water_conductivity()` 直接用这个值做 (\Delta\sigma_w^*=2C^*/\Lambda)，所以这是一个会直接影响 pore/membrane/all 谱幅值的核心参数。

---

# 动态孔径补充实施计划

## 0. 计划定位

本补充不重跑 pnextract，也不重定义孔/膜极化模型；它只补齐一件事：

[
\boxed{
\Lambda_{\mathrm{project}}
==========================

\text{由本项目三维孔隙结构和 Laplace 场解计算得到的动态孔径}
}
]

它接入现有 project-extracted 路线的位置是：

```text
microCT / segmented volume
  → solve DC Laplace field
  → compute Lambda_project
  → compute pore/membrane polarization spectra using Lambda_project
  → generate interfacial / pore / membrane / all component spectra
  → full-grid AC3D sweeps
```

当前 project-extracted 计划已经要求不要把论文工作簿 simulation/component 列当作本项目模拟结果、不要做 post-extraction geometry scaling、正式图必须追溯到本项目 `sweep_results.csv`。动态孔径也应遵守同样的 provenance 规则：**Niu 的 2.7 µm 只能作为 paper reference，不作为 project-extracted formal 结果的默认值。**

---

## 1. 数学定义与离散目标

### 1.1 连续公式

Niu 2020 给出的上尺度公式是：

[
\boxed{
\Delta \sigma_w^*
=================

\frac{2C^*}{\Lambda}
}
]

其中 (C^*) 是孔极化或膜极化诱导的复电导，单位 S；(\Delta\sigma_w^*) 是加到水相上的体积复电导率增量，单位 S/m。Niu 进一步给出动态孔径定义：

[
\boxed{
\Lambda
=======

2
\frac{
\displaystyle \int_{\Omega_p} |\mathbf E(\mathbf x)|^2,dV
}{
\displaystyle \int_{\Gamma_{sw}} |\mathbf E(\mathbf x)|^2,dS
}
}
]

其中 (\Omega_p) 是孔隙水相体积，(\Gamma_{sw}) 是固液界面，(\mathbf E(\mathbf x)) 是局部电场；Niu 明确说明复杂微结构中 (\Lambda) 需要通过求解 Laplace 方程数值确定。

对应的 Laplace 场问题写成周期扰动形式：

[
\phi_d(\mathbf x)
=================

-\mathbf E_{0,d}\cdot\mathbf x + u_d(\mathbf x)
]

[
\mathbf E_d(\mathbf x)
======================

# -\nabla\phi_d

\mathbf E_{0,d}-\nabla u_d
]

[
\boxed{
\nabla\cdot
\left[
\sigma_p
\left(
\mathbf E_{0,d}-\nabla u_d
\right)
\right]
=0
\quad \text{in } \Omega_p
}
]

固液界面：

[
\boxed{
\mathbf n\cdot \mathbf E_d =0
\quad \text{on } \Gamma_{sw}
}
]

你当前 `ac3d_solver.py` 已经采用同一个核心形式 `div(sigma * (E - grad(u))) = 0`，因此本补充应复用现有 AC3D/DC 求解框架，而不是另写一个独立 Laplace 求解器。

---

### 1.2 三方向公式

对三个物理方向分别计算：

[
\mathbf E_{0,x}=E_0(1,0,0)
]

[
\mathbf E_{0,y}=E_0(0,1,0)
]

[
\mathbf E_{0,z}=E_0(0,0,1)
]

得到：

[
\boxed{
\Lambda_x
=========

2
\frac{
\displaystyle \int_{\Omega_p} |\mathbf E_x|^2,dV
}{
\displaystyle \int_{\Gamma_{sw}} |\mathbf E_x|^2,dS
}
}
]

[
\boxed{
\Lambda_y
=========

2
\frac{
\displaystyle \int_{\Omega_p} |\mathbf E_y|^2,dV
}{
\displaystyle \int_{\Gamma_{sw}} |\mathbf E_y|^2,dS
}
}
]

[
\boxed{
\Lambda_z
=========

2
\frac{
\displaystyle \int_{\Omega_p} |\mathbf E_z|^2,dV
}{
\displaystyle \int_{\Gamma_{sw}} |\mathbf E_z|^2,dS
}
}
]

标量值建议用能量积分先合并：

[
\boxed{
\Lambda_{\mathrm{iso}}
======================

2
\frac{
\displaystyle
\sum_{d\in{x,y,z}}
\int_{\Omega_p} |\mathbf E_d|^2,dV
}{
\displaystyle
\sum_{d\in{x,y,z}}
\int_{\Gamma_{sw}} |\mathbf E_d|^2,dS
}
}
]

这个标量作为后续 Niu-style pore/membrane 上尺度中的：

[
\boxed{
\Lambda_{\mathrm{project}}=\Lambda_{\mathrm{iso}}
}
]

同时保留 (\Lambda_x,\Lambda_y,\Lambda_z) 作为各向异性诊断。Niu 2020 没有在当前资料中明确说明如何把三方向 (\Lambda) 合并成一个标量，所以这里要在 provenance 中标注为：

```text
direction_aggregation = ratio_of_summed_field_energy_integrals
aggregation_status = project_rule_to_match_scalar_Niu_framework
```

---

### 1.3 体素离散公式

设体素边长为 (h)，孔隙体素集合为 (P)，固液界面体素面集合为 (F_{sw})。则：

[
\int_{\Omega_p} |\mathbf E|^2,dV
\approx
\sum_{i\in P} |\mathbf E_i|^2 h^3
]

[
\int_{\Gamma_{sw}} |\mathbf E|^2,dS
\approx
\sum_{f\in F_{sw}} |\mathbf E_f|^2 h^2
]

所以：

[
\boxed{
\Lambda
\approx
2
\frac{
\displaystyle \sum_{i\in P} |\mathbf E_i|^2 h^3
}{
\displaystyle \sum_{f\in F_{sw}} |\mathbf E_f|^2 h^2
}
}
]

也就是：

[
\boxed{
\Lambda
\approx
2h
\frac{
\displaystyle \sum_{i\in P} |\mathbf E_i|^2
}{
\displaystyle \sum_{f\in F_{sw}} |\mathbf E_f|^2
}
}
]

界面场 (\mathbf E_f) 不允许用 pore-solid 两侧电势差计算。固相电导为 0 时，固相电势是 gauge 固定产生的数值，不是物理固相电场。界面场应从孔隙侧场取切向分量：

[
\boxed{
\mathbf E_f
===========

## \mathbf E_i

(\mathbf E_i\cdot\mathbf n_f)\mathbf n_f
}
]

因此：

[
\boxed{
|\mathbf E_f|^2
===============

## |\mathbf E_i|^2

(\mathbf E_i\cdot\mathbf n_f)^2
}
]

这个细节是本实现中最重要的科学保护栏。

---

## 2. 当前项目状态与改动必要性

### 2.1 已有可复用部分

项目 README 已经说明当前主线是 Niu 2020 Berea 三维 AC3D 复现，核心包括 micro-CT 体数据、pore/membrane 极化模型、full `350^3` AC3D 方程和 GPU matrix-free Krylov solver。

配置文件中已经固定 Berea 输入：

```text
shape = 350 × 350 × 350
voxel_size = 2.8 µm
pore_label = 1
solid_label = 2
```

AC3D 求解器也已经实现有限体积/有限差分周期场解、面调和平均和有效电导率计算。

因此动态孔径计算应尽量复用：

```text
code/src/pore_scale_electrical/ac3d_solver.py
code/src/pore_scale_electrical/ac3d_gpu.py
code/scripts/sip_simulation/run_ac3d_matrix_free_gpu_single.py
```

而不是新增一套孤立数值内核。

### 2.2 必须补齐部分

当前 `compute_polarization_spectra.py` 会直接构造默认 `PolarizationParameters()`，因此默认会把 (2.7\ \mu m) 继续用于 spectra 计算。

这对 project-extracted formal 结果是不合格的。补充计划的核心改动是：

```text
project-extracted formal 模式必须读取 dynamic_pore_size.json
缺失则报错
禁止静默回退到 Niu Table 1 的 2.7 µm
```

---

## 3. 文件级实施计划

### 3.1 新增核心模块

新增：

```text
code/src/pore_scale_electrical/dynamic_pore_size.py
```

建议包含以下数据结构：

```python
@dataclass(frozen=True)
class DirectionalDynamicPoreSize:
    physical_direction: str
    array_axis: int
    volume_integral_v2_m: float
    surface_integral_v2: float
    dynamic_pore_size_m: float
    effective_conductivity_s_m: float
    residual_norm: float
    interface_face_count: int
    pore_voxel_count: int
    pore_volume_m3: float
    voxel_surface_area_m2: float
    uniform_field_proxy_m: float
```

核心函数：

```python
def physical_direction_to_array_axis(direction: str, axis_order: str) -> int:
    ...
```

当前项目内部有 `(z,y,x)` 表述，但求解器中 `x -> axis 0, y -> axis 1, z -> axis 2`。  因此动态孔径模块必须显式记录轴映射：

```text
axis_order = zyx
physical x -> array axis 2
physical y -> array axis 1
physical z -> array axis 0
```

不要在第一版中全局修改 `direction_to_axis()`，否则会影响已有 sweep provenance。

---

### 3.2 新增周期 active-domain 组件识别

修改：

```text
code/src/pore_scale_electrical/ac3d_active_domain.py
```

当前 `active_component_anchor_indices()` 用普通 6-connected `ndimage.label`，没有处理周期边界。

新增：

```python
def periodic_active_component_anchor_indices(active_mask: np.ndarray) -> tuple[int, ...]:
    ...
```

算法：

1. `ndimage.label(active_mask)` 得到普通连通分量；
2. 对 axis 0/1/2 三个方向分别检查周期边界；
3. 若边界相对体素均为孔隙，则 union 对应组件；
4. 合并后每个周期组件只保留一个 anchor；
5. GPU active-domain gauge 使用该周期组件 anchor。

新增测试：

```text
code/tests/test_ac3d_active_domain.py
```

测试目标：

```text
一个只通过周期边界连通的孔隙组件，anchor 数必须为 1。
```

现有测试只覆盖非周期连通与空 active domain。

---

### 3.3 新增 CLI 脚本

新增：

```text
code/scripts/sip_simulation/compute_dynamic_pore_size.py
```

建议参数：

```text
--raw
--shape 350 350 350
--dtype <u2
--axis-order zyx
--pore-label 1
--solid-label 2
--voxel-size-m 2.8e-6
--directions x y z
--backend cpu|gpu
--preconditioner none|jacobi|fft
--fft-reference pore|mean-face|mean-abs
--gauge-mode active-domain|auto
--solver-dtype complex64|complex128
--rtol
--maxiter
--crop-start
--crop-size
--run-mode smoke|diagnostic|formal
--save-potential
--out-dir
```

正式模式规则：

```text
run-mode=formal:
    crop-start/crop-size 必须为空
    directions 必须等于 x y z
    axis-order 必须显式提供
    pore-label/solid-label 必须显式提供
    必须写 dynamic_pore_size.json
```

Smoke/diagnostic 可以允许 crop，但输出必须标注：

```text
dynamic_pore_size_use = diagnostic_only_not_formal
```

---

### 3.4 接入极化谱计算

修改：

```text
code/scripts/sip_simulation/compute_polarization_spectra.py
```

新增参数：

```text
--parameter-mode niu2020-paper|project-extracted
--dynamic-pore-size-manifest results/.../dynamic_pore_size.json
```

规则：

```python
if parameter_mode == "niu2020-paper":
    params = PolarizationParameters(dynamic_pore_size_m=2.7e-6)
    lambda_source = "niu2020_table1"

if parameter_mode == "project-extracted":
    if manifest missing:
        raise ValueError("project-extracted spectra require dynamic pore size manifest")
    params = replace(
        PolarizationParameters(),
        dynamic_pore_size_m=manifest["lambda_iso_m"],
    )
    lambda_source = "project_extracted_microct_laplace_field"
```

metadata 中必须增加：

```json
{
  "dynamic_pore_size_m": "...",
  "dynamic_pore_size_source": "project_extracted_microct_laplace_field",
  "dynamic_pore_size_manifest": "...",
  "paper_reference_dynamic_pore_size_m": 2.7e-6,
  "paper_reference_used_as_project_value": false
}
```

---

### 3.5 接入 component spectra

修改：

```text
code/scripts/sip_simulation/make_polarization_component_spectra.py
```

要求：

```text
project-extracted mode 下，interfacial / pore / membrane / all 四个 component spectra
全部继承同一个 dynamic_pore_size_manifest
```

这里要特别防止：

```text
base spectra 用 project Lambda
component spectra 又回退到默认 2.7e-6
```

---

### 3.6 接入结果包 builder

修改：

```text
code/scripts/sip_simulation/build_niu2020_berea_result_package.py
```

在完整结果包中新增：

```text
results/<run_name>/dynamic_pore_size/
```

并在顶层 manifest 中加入：

```json
{
  "dynamic_pore_size": {
    "lambda_iso_m": "...",
    "lambda_x_m": "...",
    "lambda_y_m": "...",
    "lambda_z_m": "...",
    "source": "project_extracted_microct_laplace_field",
    "manifest": "dynamic_pore_size/dynamic_pore_size.json"
  }
}
```

项目 AGENTS 已要求正式结果包包含配置、数字岩心/孔网、source data、metadata、provenance 和 full-grid sweep，并且正式 sweep 不得散放到 `results/` 顶层。动态孔径目录也应属于同一完整结果包。

---

## 4. 结果目录结构

建议正式目录：

```text
results/niu2020_berea_project_extracted_v1/
  dynamic_pore_size/
    config.yml
    input_manifest.json
    dynamic_pore_size.json
    directional_integrals.csv
    diagnostics/
      lambda_direction_comparison.png
      surface_energy_slice_x.png
      surface_energy_slice_y.png
      surface_energy_slice_z.png
      volume_vs_surface_integral_check.csv
    solver/
      x/
        result.json
        residual_history.csv
        potential.npy
      y/
        result.json
        residual_history.csv
        potential.npy
      z/
        result.json
        residual_history.csv
        potential.npy
    provenance/
      dynamic_pore_size_provenance.md
```

`dynamic_pore_size.json` 建议字段：

```json
{
  "method": "johnson_niu_field_weighted_dynamic_pore_size_voxel_face_v1",
  "lambda_iso_m": 0.0,
  "lambda_x_m": 0.0,
  "lambda_y_m": 0.0,
  "lambda_z_m": 0.0,
  "lambda_uniform_proxy_m": 0.0,
  "paper_reference_lambda_m": 2.7e-6,
  "paper_reference_used_as_project_value": false,
  "axis_order": "zyx",
  "physical_to_array_axis": {"x": 2, "y": 1, "z": 0},
  "voxel_size_m": 2.8e-6,
  "pore_label": 1,
  "solid_label": 2,
  "pore_voxel_count": 0,
  "solid_voxel_count": 0,
  "interface_face_count": 0,
  "surface_estimator": "six_connected_voxel_faces_pore_side_tangential_field",
  "direction_aggregation": "ratio_of_summed_volume_and_surface_integrals",
  "input_sha256": "...",
  "git_commit": "...",
  "run_mode": "formal"
}
```

---

## 5. 数值流程细节

### 5.1 输入读取

复用或抽取已有 raw audit 逻辑。当前 `audit_microct_raw.py` 已经能读取 raw、推断/检查 shape、统计 label、输出中心切片和周期 face counts。

正式动态孔径脚本读取：

```python
volume = np.memmap(
    raw_path,
    dtype="<u2",
    mode="r",
    shape=(350, 350, 350),
    order="C",
)
```

检查：

```text
unique labels == {1, 2}
pore_label == 1
solid_label == 2
voxel_size_m == 2.8e-6
```

---

### 5.2 求解 DC 场

对每个方向 (d)：

```python
water_sigma = 1.0 + 0.0j
solid_sigma = 0.0 + 0.0j
E0 = 1.0  # V/m
```

用现有 GPU face-type 求解器：

```python
face_data = face_type_conductivity_gpu(
    labels,
    pore_label=1,
    solid_label=2,
    water_conductivity_s_m=1.0,
    solid_conductivity_s_m=0.0,
    dtype="complex64",
)

result = solve_ac3d_matrix_free_gpu_face_types(
    face_data,
    direction=array_axis,
    field_strength_v_m=1.0,
    voxel_size_m=2.8e-6,
    preconditioner="fft",
    fft_reference="pore",
    gauge_mode="active-domain",
    return_potential=True,
    rtol=1e-6,
)
```

`run_ac3d_matrix_free_gpu_single.py` 已支持 GPU face-type 构建、保存 residual history、保存 solution，因此可以作为动态孔径 CLI 的模板。

---

### 5.3 体积分计算

正式主值建议用有限体积能量恒等式：

[
I_{V,d}
=======

V_{\mathrm{cell}}
\frac{\sigma_{\mathrm{eff},d}}{\sigma_p}
E_0^2
]

其中：

```text
V_cell = n_voxels * h^3
sigma_p = 1 S/m
E0 = 1 V/m
```

同时从重建局部场计算：

[
I_{V,d}^{\mathrm{field}}
========================

\sum_{i\in P}
|\mathbf E_{d,i}|^2h^3
]

输出：

[
\mathrm{energy_identity_relative_error}
=======================================

\frac{|I_{V,d}^{\mathrm{field}}-I_{V,d}|}{I_{V,d}}
]

若误差太大，说明体素中心场重建或界面处理有问题，不能进入 formal。

---

### 5.4 界面积分计算

遍历六邻接 pore-solid 面。对每个界面面 (f)，从孔隙侧体素 (i) 获取局部场 (\mathbf E_i)，并取切向分量：

[
\mathbf E_f
===========

## \mathbf E_i

(\mathbf E_i\cdot\mathbf n_f)\mathbf n_f
]

[
I_{S,d}
=======

\sum_{f\in F_{sw}}
|\mathbf E_f|^2h^2
]

同时计算普通几何 proxy：

[
\Lambda_{\mathrm{geom}}
=======================

# \frac{2V_p}{S_{sw}}

# \frac{2N_p h^3}{N_{sw}h^2}

2h\frac{N_p}{N_{sw}}
]

该 proxy 只用于诊断，不能替代 (\Lambda_{\mathrm{project}})。Johnson/Niu 定义的 (\Lambda) 是场加权长度，不是单纯几何表面积比；相关文献也强调 (\Lambda) 由 Laplace 解确定，不能通过简单几何分析得到。

---

## 6. 与 pore/membrane spectra 的接入顺序

正式流程改为：

```text
Step 1: compute_dynamic_pore_size.py
Step 2: compute_polarization_spectra.py --parameter-mode project-extracted --dynamic-pore-size-manifest ...
Step 3: make_polarization_component_spectra.py
Step 4: run interfacial / pore / membrane / all AC3D sweeps
Step 5: plot mechanism comparison
```

注意：(\Lambda_{\mathrm{project}}) 只影响：

```text
delta_sigma_pore
delta_sigma_membrane
delta_sigma_total
apparent_water_sigma
```

不改变：

```text
pore radius distribution
throat length distribution
Zdc
sigma_w
epsilon_w
epsilon_s
```

更不能用 Lambda 重新缩放孔喉几何。当前项目已经把 post-extraction geometry scaling 作为红线禁止。

---

## 7. 测试计划

### 7.1 单元测试：公式与几何

新增：

```text
code/tests/test_dynamic_pore_size.py
```

测试 1：平板孔道

[
\Lambda = H
]

其中 (H=N_ph) 是孔道厚度。验收：

```text
relative error < 1e-6
```

测试 2：周期圆柱孔道

[
V=\pi R^2L,\quad S=2\pi RL
]

[
\Lambda = R
]

体素圆柱允许离散误差，但 (R=5,10,20) voxels 时误差应随分辨率下降。

测试 3：体素尺寸缩放

```text
h -> a h
Lambda -> a Lambda
```

测试 4：电场强度缩放

```text
E0 -> 10 E0
Lambda 不变
```

测试 5：水相电导率缩放

```text
sigma_p -> 0.1 sigma_p
Lambda 不变
```

测试 6：固相 gauge 污染

人为修改固相 potential，界面积分和 Lambda 不变。若变了，说明错误使用了 pore-solid 跨相电势差。

---

### 7.2 集成测试：CPU/GPU 一致性

在 `16^3` 或 `32^3` 子体上比较：

```text
CPU direct
CPU matrix-free
GPU matrix-free
```

验收：

```text
Lambda relative difference < 1e-4
```

---

### 7.3 provenance 测试

新增：

```text
tests/test_dynamic_pore_size_formal_provenance.py
```

检查：

```text
run_mode=formal 时必须有 x/y/z 三方向
paper_reference_used_as_project_value = false
dynamic_pore_size_source = project_extracted_microct_laplace_field
input_sha256 非空
axis_order 非空
```

---

### 7.4 spectra fail-closed 测试

修改或新增：

```text
tests/test_compute_polarization_spectra.py
```

检查：

```text
--parameter-mode project-extracted 但没有 --dynamic-pore-size-manifest
=> 必须报错

--parameter-mode niu2020-paper
=> 允许使用 2.7e-6，并记录 source=niu2020_table1
```

---

## 8. 运行分级

### 8.1 Smoke

目的：只检查流程，不得作为科学结果。

```powershell
& 'C:\Users\imgw\.conda\envs\ml\python.exe' code\scripts\sip_simulation\compute_dynamic_pore_size.py `
  --raw "data\Niu 2020data\microCT_Berea.raw" `
  --shape 350 350 350 `
  --dtype "<u2" `
  --axis-order zyx `
  --pore-label 1 `
  --solid-label 2 `
  --voxel-size-m 2.8e-6 `
  --crop-start 0 0 0 `
  --crop-size 48 48 48 `
  --directions x y z `
  --backend cpu `
  --run-mode smoke `
  --out-dir "results\niu2020_berea_project_extracted_v1_smoke\dynamic_pore_size"
```

输出必须写：

```text
formal_dynamic_pore_size = false
```

---

### 8.2 Diagnostic

目的：检查方向差异、收敛敏感性、界面积分稳定性。

```text
crop: 100³ 或多个 REV crop
directions: x/y/z
rtol: 1e-5, 1e-6, 1e-7
dtype: complex64 + selected complex128
```

验收：

```text
Lambda 随 rtol 收紧变化 < 0.5%
Lambda_x/y/z 没有异常离群
energy_identity_relative_error < 1%
```

---

### 8.3 Formal

目的：生成正式 project-extracted Lambda。

```powershell
& 'C:\Users\imgw\.conda\envs\ml\python.exe' code\scripts\sip_simulation\compute_dynamic_pore_size.py `
  --raw "data\Niu 2020data\microCT_Berea.raw" `
  --shape 350 350 350 `
  --dtype "<u2" `
  --axis-order zyx `
  --pore-label 1 `
  --solid-label 2 `
  --voxel-size-m 2.8e-6 `
  --directions x y z `
  --backend gpu `
  --preconditioner fft `
  --fft-reference pore `
  --gauge-mode active-domain `
  --solver-dtype complex64 `
  --rtol 1e-6 `
  --maxiter 1000 `
  --run-mode formal `
  --save-potential `
  --out-dir "results\niu2020_berea_project_extracted_v1\dynamic_pore_size"
```

随后生成 spectra：

```powershell
& 'C:\Users\imgw\.conda\envs\ml\python.exe' code\scripts\sip_simulation\compute_polarization_spectra.py `
  --source pnextract `
  --network-dir "results\niu2020_berea_project_extracted_v1\pore_network\network_parsed" `
  --parameter-mode project-extracted `
  --dynamic-pore-size-manifest "results\niu2020_berea_project_extracted_v1\dynamic_pore_size\dynamic_pore_size.json" `
  --out "results\niu2020_berea_project_extracted_v1\polarization_spectra\project_extracted_base_spectrum.csv"
```

---

## 9. 验收标准

动态孔径补充完成后，必须满足：

```text
1. dynamic_pore_size.json 存在，并包含 lambda_iso_m / lambda_x_m / lambda_y_m / lambda_z_m。
2. paper_reference_lambda_m = 2.7e-6，但 paper_reference_used_as_project_value = false。
3. x/y/z 三方向均有 solver result、residual history 和积分结果。
4. axis_order = zyx，且 physical_to_array_axis 明确记录。
5. interface_face_count、pore_voxel_count、lambda_uniform_proxy_m 同步输出。
6. spectra metadata 中 dynamic_pore_size_source 不再是默认值。
7. project-extracted formal 模式缺少 dynamic_pore_size_manifest 时直接报错。
8. 结果包 provenance 明确说明：该 Lambda 来自 project-extracted microCT Laplace field solve。
9. 后续 pore/membrane/all sweep 的 source data 能追溯到同一个 dynamic_pore_size.json。
```

---

## 10. 风险与边界条件

第一，`2.8 µm` 体素尺寸和论文参考 (\Lambda=2.7 µm) 同量级，这意味着计算值会非常受体素界面阶梯化影响。正式报告中必须同步给出 `lambda_uniform_proxy_m`、方向值和 sensitivity，而不能把差异解释成纯物理差异。

第二，体素面界面积分和 marching-cubes 曲面积分不能混用。第一版建议坚持 “six-connected voxel-face interface + pore-side tangential field”，因为它与当前有限体积 AC3D 算子一致。

第三，若某方向没有有效水相周期通路，则该方向 (\Lambda_d) 不应硬算成 0 或 inf 后参与标量平均，而应标记：

```text
direction_valid = false
reason = no_percolating_pore_path_under_periodic_active_domain
```

第四，(\Lambda_{\mathrm{project}}) 只负责 (2C^*/\Lambda) 上尺度，不应被用来调膜极化峰位；膜极化峰位主要由 (L^2/(4D))、(Z_{dc}) 和孔喉长度分布控制。Niu 的机制拆分中 pore/membrane-only 是把 (\Delta\sigma_w^*) 加到水相、固相置零，因此动态孔径进入的是 pore/membrane 体积增量，而不是 interfacial-only。

---

## 11. 最终补充口径

建议在项目文档中这样写：

> 本补充步骤用于计算 Niu et al. (2020) Equation 12 中的动态孔径 (\Lambda)。不同于直接采用论文 Table 1 的 (\Lambda=2.7,\mu m)，本项目在 project-extracted formal 路线中从 Berea microCT 分割体求解三方向 DC Laplace 场，并按 Johnson et al. 场加权定义计算 (\Lambda_x,\Lambda_y,\Lambda_z) 和能量合并标量 (\Lambda_{\mathrm{iso}})。该 (\Lambda_{\mathrm{iso}}) 随后用于将 project-extracted pore/membrane 复电导 (C^*) 转换为水相体积复电导率增量 (\Delta\sigma_w^*=2C^*/\Lambda_{\mathrm{project}})。论文给出的 2.7 µm 仅作为 reference comparison，不作为本项目正式模拟输入。
