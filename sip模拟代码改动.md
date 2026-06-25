可以精简成一句话：

**公式层面基本能做 Niu-framework 的 SIP 模拟；求解器层面还差“正式可信度验证”和“pore/membrane-only 零固相处理”。**

## 求解器最需要改的地方

### 1. GPU BiCGSTAB 必须增加 true residual

当前主要用递推残差判断收敛，正式结果不够稳。需要在 `code/src/pore_scale_electrical/ac3d_gpu.py`：

* `_gpu_bicgstab()`
* `solve_ac3d_matrix_free_gpu_face_types()`

增加：

```text
true_residual = ||b - A x|| / ||b||
recursive_residual
true_residual_passed
```

并写入 `sweep_results.csv`。

这是第一优先级。

---

### 2. pore-only / membrane-only 的零固相问题要处理

Niu 机制拆分要求：

```text
pore only: solid = 0
membrane only: solid = 0
```

当前这样做符合论文，但数值上会产生大量零空间：固体体素和孤立孔隙分量没有导电连接。

建议新增：

```text
code/src/pore_scale_electrical/ac3d_active_domain.py
```

只在导电水相连通域上求解 pore/membrane-only，或者对每个连通分量做 gauge fixing / nullspace projection。

这是第二优先级。

---

### 3. 低频 interfacial / all 需要 complex128 checkpoint

当前 full-grid 主要用 `complex64 + rtol=1e-5`。低频 (\sigma'') 很小，容易被残差底噪污染。

改动：

* 正式 sweep 仍可用 complex64；
* 但关键频点必须用 complex128 复核；
* 输出 complex64 vs complex128 相对误差。

建议检查频点：

```text
1e-3, 1e-2, 1e-1, 1, 1e3, 1e6, 1e9 Hz
```

---

### 4. 不允许 silent nearest-frequency 匹配

`run_ac3d_matrix_free_gpu_sweep.py` 当前会找最近的 spectrum row。正式复现应改成：

```text
formal mode: 频率必须精确匹配，否则报错
diagnostic mode: 允许插值，但必须记录
```

否则 sweep 的 requested frequency 和实际 used frequency 可能不一致。

---

### 5. all 机制必须重新跑场解，不能有效量拼接

正式求解器流程必须是：

```text
pore spectrum + membrane spectrum + dielectric term
→ 生成 all phase conductivity
→ 重新跑 AC3D
→ 得到 all sweep_results.csv
```

不能用：

```text
all - membrane + diagnostic_membrane
```

这种有效电导层代数替换。

---

### 6. 正式结果必须跑 x/y/z 三方向

当前很多结果是 `_fft_x`。正式 project-extracted 需要：

```text
sigma_xx
sigma_yy
sigma_zz
directional_mean
anisotropy_ratio
```

因为 project-extracted 孔网和 microCT 可能存在方向性。

---

## 最小改动顺序

按优先级做：

```text
1. GPU true residual
2. zero-solid active-domain / nullspace 处理
3. complex128 checkpoint
4. formal mode 禁止 nearest-frequency
5. all 机制强制独立 sweep
6. x/y/z 三方向 sweep 与方向平均
```

## 最终判断

**当前求解器可以用于 diagnostic/project-extracted SIP 模拟。**

但要作为正式复现结果，必须先补上：

```text
true residual + zero-solid 处理 + complex128 复核 + 独立 all sweep + 三方向验证
```

这几个不是代码风格问题，而是直接影响 (\sigma'')、膜极化峰位、低频 interfacial 曲线可信度的科学问题。
