# SIP 实验反演孔隙几何使用说明

此工具从实测频率、实部电导率和虚部电导率联合拟合非负孔半径分布、孔喉长度分布，以及各孔喉共用的等效面积。它输出能解释数据的一组条件解，并检查面积和长度边界的敏感性；不能仅凭频谱证明真实三维几何已恢复。

## Niu 2020 示例

在项目根目录运行：

```powershell
$env:PYTHONNOUSERSITE = '1'
& 'C:\Users\13404\.conda\envs\ml\python.exe' code/scripts/sip_simulation/invert_sip_geometry.py --niu-workbook 'data/Niu 2020data/Figure6.xlsx' --config configs/niu2020_sip_geometry_inversion.json --out results/niu2020_sip_geometry_inversion_20260912_diagnostic
```

输出包含：

- `source_data/pore_radius_distribution.csv`：孔半径、孔直径、归一化分箱权重及面积扫描的替代解包络。
- `source_data/throat_length_distribution.csv`：孔喉长度分布；不是孔喉半径分布。
- `source_data/parameter_summary.csv`：等效面积、圆形等效半径、分位尺寸、背景参数。
- `source_data/area_profile.csv`：固定不同面积后重新优化其余参数的拟合误差。
- `source_data/length_bound_sensitivity.csv`：改变孔喉长度上界时的面积与误差。
- `figures/geometry_histograms.png`：灰色细柱双联直方图。
- `figures/geometry_histograms_overlay.png`：蓝色Figure5与橙色条件反演的半透明叠加图。
- `figures/sip_joint_fit.png`：实部、虚部联合拟合、残差和机制分量。
- `README_zh.md`、`provenance/summary.json`：本次结果及限制。

输出目录可重复运行并更新其可再生文件；若需要保留多次参数实验，使用新的`--out`目录。输入实验、论文与CT数据不修改。

## 换成自己的实验

准备正值CSV，列名为：

```text
frequency_hz,sigma_real_s_m,sigma_imag_s_m
```

频率单位Hz，电导率单位S/m，正虚部约定。至少6个不同频率，建议足够密集覆盖主要弛豫峰。非正值或缺失值会报错，不会被静默删除；应先检查相位符号、电极效应和测量有效性。频带外点会记录在`excluded_frequency_rows.csv`。

复制中文JSON配置为该样品的配置，填写已知水电导率、D、膜极化率、表面电导、动态孔径Lambda和地层因子F；不要把Niu参数当作所有样品的通用值。把`figure5_validation`设为`null`，并根据独立信息设置尺寸上下界和拟合频带。

```powershell
$env:PYTHONNOUSERSITE = '1'
& 'C:\Users\13404\.conda\envs\ml\python.exe' code/scripts/sip_simulation/invert_sip_geometry.py --input 'data/my_sip.csv' --config 'configs/my_sip_inversion.json' --out 'results/my_sip_inversion_diagnostic'
```

`frequency_range_hz`、`radius_range_m`、`length_range_m`、`area_range_m2`可调整；这些是先验，不是拟合出的边界。误差尺度2%/5%为默认数值权重，不是已知测量误差，因此工具不输出统计置信区间。

## 如何判断能否解释为合理几何

先看实部和虚部残差是否存在系统偏差，再看分布是否堆在边界，最后检查面积扫描和长度边界扫描。仅有低拟合误差不够：不同尺寸分布可能产生几乎相同的频谱。

固定参数时，工具把孔核权重解释为归一化质量，孔喉核系数解释为面积乘质量。质量若要解释为体积分数，需要接受Niu分布模型的假设，不能直接称为孔隙率或孔喉个数。一个共同截面积只能给一个等效圆半径；没有足够数据反演独立截面积分布。

本实现是准静态有效体近似，拟合低频有效介电背景，未在每次迭代运行AC3D。研究级结论应由独立CT/NMR/压汞约束、参数敏感性以及选定几何的独立前向计算进一步验证。不要把反演拟合谱记为350³真实求解结果。
