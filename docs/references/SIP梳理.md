# Niu et al. (2020) 孔隙尺度 SIP 模拟框架整理

> 论文：Niu, Q., Zhang, C., & Prasad, M. (2020). *A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity of Porous Media in the Frequency Range From 1 mHz to 1 GHz*. Journal of Geophysical Research: Solid Earth.


> 公式说明：本文档已统一使用 Markdown/KaTeX/MathJax 常见语法：行内公式用 `$...$`，独立公式用 `$$...$$`，方便在 VSCode、Obsidian、Typora、GitHub Pages 等环境中渲染。

本文档整理该论文中 SIP / 宽频电学模拟的核心思想：作者不是直接在纳米尺度解析电双层中的离子迁移与扩散，而是把不同极化机制先转化为频率相关的复电导扰动，再上尺度到孔隙水相中，最后在真实微 CT 孔隙结构上求解复电导率场，从而得到样品整体的有效电导率和有效介电常数。

---

## 1. 论文模拟的总体目标

论文目标是建立一个孔隙尺度数值模拟框架，用来计算多孔地质材料在宽频范围内的：

- 有效电导率：$\sigma'_{\mathrm{eff}}$
- 有效介电常数：$\varepsilon'_{\mathrm{eff}}$
- 等价地，也可以得到有效复电导率：$\sigma^*_{\mathrm{eff}}$

作者重点考虑两大类极化：

1. **界面极化**，也就是 Maxwell-Wagner 型极化，来源于固相和水相之间电学性质不连续。
2. **电化学极化**，来源于矿物-水界面电双层 EDL 中离子的迁移、扩散和浓度重新分布。论文中具体把它拆成：
   - 孔极化 / pore polarization
   - 膜极化 / membrane polarization
   - 颗粒极化 / grain polarization 作为背景理论介绍，但实际 Berea sandstone 模拟中没有直接采用颗粒极化，而是采用孔极化和膜极化。

---

## 2. 基本电磁量定义：复电导率与复介电常数

论文先从总电流密度开始定义电导率和介电常数。

### 2.1 总电流密度

$$
J_t = J_c + J_d
$$

其中：

| 符号 | 含义 |
|---|---|
| $J_t$ | 总电流密度 |
| $J_c$ | 传导电流密度 |
| $J_d$ | 位移电流密度 |

传导电流来自孔隙水中离子迁移，位移电流来自材料极化响应。

### 2.2 欧姆导电项

$$
J_c = \sigma_{dc} E
$$

其中 $\sigma_{dc}$ 是直流电导率，$E$ 是电场强度。

### 2.3 位移电流项

$$
J_d = \frac{\partial D}{\partial t}
$$

$$
D = \varepsilon_d^* E
$$

其中：

$$
\varepsilon_d^* = \varepsilon_d' - i\varepsilon_d''
$$

这里的 $\varepsilon_d^*$ 是不包含欧姆损耗的复介电常数。

对于正弦电场：

$$
E = E_0 \exp(i\omega t)
$$

有：

$$
J_d = i\omega \varepsilon_d^* E
$$

所以总电流可以写成：

$$
J_t = (\sigma_{dc} + i\omega \varepsilon_d^*)E
$$

### 2.4 复电导率定义

定义：

$$
J_t = \sigma^* E
$$

则：

$$
\begin{aligned}
\sigma^* &= \sigma' + i\sigma'' \\
&= (\sigma_{dc}+\omega\varepsilon_d'') + i\omega\varepsilon_d'
\end{aligned}
$$

因此：

$$
\sigma' = \sigma_{dc}+\omega\varepsilon_d''
$$

$$
\sigma'' = \omega\varepsilon_d'
$$

论文中所谓 electrical conductivity，主要指复电导率实部 $\sigma'$。

### 2.5 复介电常数定义

定义：

$$
J_t = i\omega\varepsilon^*E
$$

则：

$$
\begin{aligned}
\varepsilon^* &= \varepsilon' - i\varepsilon'' \\
&= \varepsilon_d' - i\left(\varepsilon_d'' + \frac{\sigma_{dc}}{\omega}\right)
\end{aligned}
$$

论文中所谓 electrical permittivity，主要指复介电常数实部 $\varepsilon'$。

---

## 3. 论文涉及的几种极化机制

## 3.1 界面极化 interfacial polarization / Maxwell-Wagner polarization

### 物理图像

界面极化发生在两种电学性质不同的相之间，比如固体矿物相和孔隙水相。固相和水相的电导率、介电常数不同，当外加电场作用时，两相交界面附近的电流连续性和电场分布会发生调整，导致电荷在界面附近积累。

这种界面电荷积累形成感应偶极矩，使整个多孔介质表现出比简单体积平均更高的有效介电常数。

### 来源模型

论文提到的背景模型包括：

- Maxwell-Wagner 极化理论；
- Hanai 模型；
- differential effective medium theory；
- Bussian、Sen、Mendelson & Cohen 等相关理论。

但是论文的关键做法是：**在孔隙尺度数值模拟中，不再额外引入一个界面极化解析模型。**

原因是：真实微结构中已经显式区分了固相和水相，只要给固相和水相赋予不同的复电导率，固-液界面的电学不连续性会自然进入控制方程。因此界面极化是通过求解真实二相结构上的电势场自动产生的。

### 模拟中如何体现

固相：

$$
\sigma_s^* = i\omega\varepsilon_s
$$

水相：

$$
\sigma_w^* = \sigma_w + i\omega\varepsilon_w
$$

如果只模拟界面极化，则只给固相和水相赋予本征的高频介电常数和直流电导率，不加入孔极化或膜极化导致的 $\Delta\sigma_w^*$。

---

## 3.2 颗粒极化 grain polarization

### 物理图像

颗粒极化假设材料可以看成许多规则颗粒浸没在电解质中。矿物颗粒表面带负电，附近形成电双层。外加电场作用后，电双层内的离子沿颗粒表面重新分布，一侧富集、一侧亏损，从而在颗粒尺度上形成电偶极矩。

### 来源模型

论文重点提到 Schwarz (1962) 的模型。Schwarz 模型原本是为球形胶体颗粒在电解质中的低频介电弛豫建立的 Debye 型模型。其核心假设是：

- 电双层中的离子只能沿颗粒表面切向运动；
- 不考虑与体相溶液之间的法向交换；
- 适用于颗粒半径远大于 Debye 长度的情况。

### 与本文模拟的关系

论文并没有直接把 Berea sandstone 当成孤立颗粒体系来模拟颗粒极化。作者认为对于多孔岩石，连续固相不适合简单看成孤立颗粒；相反，孔隙空间的孔节点和孔喉网络更适合描述砂岩的极化过程。

但是作者借用了 Schwarz 模型的数学形式，把原来的颗粒半径替换为孔半径，用来近似描述孔极化。

---

## 3.3 孔极化 pore polarization

### 物理图像

孔极化认为极化主要发生在孔节点内部。孔壁附近存在电双层，外加电场作用后，孔壁附近反离子发生重新分布，在孔尺度上产生离子浓度梯度和电偶极矩。

对于岩石来说，孔极化比颗粒极化更自然，因为很多实验表明 SIP 弛豫时间更受孔径或孔喉尺度控制，而不是简单受颗粒大小控制。

### 来源模型公式

作者采用 Schwarz (1962) 模型的 Debye 型公式，但把颗粒半径换成孔半径 $r$：

$$
C_p^* = \frac{i\omega\tau_p}{1+i\omega\tau_p}\Sigma_S
$$

其中：

| 符号 | 含义 |
|---|---|
| $C_p^*$ | 单个孔的孔极化诱导复电导，单位 S |
| $\omega$ | 角频率，$\omega=2\pi f$ |
| $\tau_p$ | 孔极化弛豫时间 |
| $\Sigma_S$ | 与 EDL 有关的表面电导，单位 S |

弛豫时间为：

$$
\tau_p = \frac{r^2}{2D}
$$

其中：

| 符号 | 含义 |
|---|---|
| $r$ | 孔半径 |
| $D$ | EDL 中反离子的扩散系数 |

### 孔径分布卷积

真实岩石中不是一个孔，而是很多不同大小的孔。因此单孔响应要与孔径分布 $f(r)$ 卷积：

$$
C_P^* = f(r) \otimes C_p^*(r)
$$

其中 $C_P^*$ 是整个样品孔极化贡献的总复电导扰动。

### 本文参数

论文 Berea sandstone 示例中使用：

| 参数 | 数值 |
|---|---|
| $\Sigma_S$ | $1.3\times10^{-9}\ \mathrm{S}$ |
| $D$ | $1.3\times10^{-9}\ \mathrm{m^2/s}$ |
| Na$^+$ 迁移率 | $5.19\times10^{-8}\ \mathrm{m^2/(V\,s)}$ |

---

## 3.4 膜极化 membrane polarization

### 物理图像

膜极化发生在窄孔喉中。作者把孔隙空间简化成大孔节点和窄孔喉连接的网络。窄孔喉中，由于 EDL 占据相对较大的体积，孔喉表现出阳离子选择性，即阳离子的迁移数高于阴离子。

外加电场作用下，阳离子在孔喉一端富集，在另一端亏损，形成跨孔喉的离子浓度梯度和电偶极矩。这就是膜极化。

### 来源模型

论文采用 Titov et al. (2002) 的窄孔喉膜极化模型。模型输出的是孔喉的复阻抗 $Z_m^*$：

$$
Z_m^* = Z_{dc} \left[
1 - \eta_0 \left(
1 - \frac{1 - \exp(-2\sqrt{i\omega\tau_m})}{2\sqrt{i\omega\tau_m}}
\right)
\right]
$$

其中：

| 符号 | 含义 |
|---|---|
| $Z_m^*$ | 孔喉复阻抗 |
| $Z_{dc}$ | 孔喉直流阻抗 |
| $\eta_0$ | 极化率 / polarizability |
| $\tau_m$ | 膜极化弛豫时间 |
| $\omega$ | 角频率 |

弛豫时间为：

$$
\tau_m = \frac{L^2}{4D}
$$

其中：

| 符号 | 含义 |
|---|---|
| $L$ | 孔喉特征长度 |
| $D$ | EDL 中反离子扩散系数 |

因为 $Z_m^*$ 包含孔喉本身的直流导电部分，所以要减去直流分量，得到真正由膜极化引起的复电导扰动：

$$
C_m^* = \frac{1}{Z_m^*} - \frac{1}{Z_{dc}}
$$

真实样品有很多孔喉长度，因此与孔喉长度分布 $f(L)$ 卷积：

$$
C_M^* = f(L) \otimes C_m^*(L)
$$

### 本文参数

Berea sandstone 示例中：

| 参数 | 数值 |
|---|---|
| $\eta_0$ | 1% |
| $D$ | $1.3\times10^{-9}\ \mathrm{m^2/s}$ |
| $Z_{dc}$ | 根据孔喉几何尺寸、长度和孔隙水电导率计算 |

---

## 4. 不同极化贡献如何合并

孔极化和膜极化都先被表示为复电导扰动：

$$
C_P^*
$$

$$
C_M^*
$$

二者叠加得到总的电化学极化复电导：

$$
C^* = C_P^* + C_M^*
$$

注意这里的 $C^*$ 不是整个样品的有效复电导率，而是电化学极化在孔隙尺度上产生的复电导扰动，单位是 S。

---

## 5. 电化学极化如何上尺度到孔隙水相

为了把孔极化和膜极化纳入体素尺度的数值模拟，作者把复电导扰动 $C^*$ 转换成水相中的额外体积复电导率 $\Delta\sigma_w^*$：

$$
\Delta\sigma_w^* = \frac{2C^*}{\Lambda}
$$

其中：

| 符号 | 含义 |
|---|---|
| $\Delta\sigma_w^*$ | 添加到水相的额外复电导率，单位 S/m |
| $C^*$ | 电化学极化诱导的复电导扰动，单位 S |
| $\Lambda$ | dynamic pore size，动态孔径 |

动态孔径 $\Lambda$ 定义为：

$$
\frac{2}{\Lambda} = \frac{\int |E(x)|^2 dV}{\int |E(x)|^2 dS}
$$

等价写法：

$$
\Lambda = 2\frac{\int |E(x)|^2 dS}{\int |E(x)|^2 dV}
$$

其中：

| 符号 | 含义 |
|---|---|
| $E(x)$ | 局部电场 |
| $dV$ | 对孔隙空间体积积分 |
| $dS$ | 对固-液界面积分 |

Berea sandstone 示例中：

$$
\Lambda = 2.7\ \mu\mathrm{m}
$$

这个操作的物理意义是：孔极化和膜极化本来是局部孔节点或孔喉上的极化电导扰动，作者把它们等效成“孔隙水相额外获得了一部分频率相关的复电导率”。这样就可以在常规的体素电学模拟中处理，而不需要直接解析纳米尺度 EDL。

---

## 6. 最终进入孔隙尺度模拟的相属性

### 6.1 水相复电导率

加入电化学极化之后，水相表观复电导率为：

$$
\sigma_w^* = \sigma_w + i\omega\varepsilon_w + \Delta\sigma_w^*
$$

其中：

| 符号 | 含义 |
|---|---|
| $\sigma_w$ | 孔隙水直流电导率 |
| $\varepsilon_w$ | 水的高频介电常数 |
| $\Delta\sigma_w^*$ | 由孔极化和/或膜极化上尺度得到的额外复电导率 |

### 6.2 固相复电导率

对于非金属矿物，作者认为固相基本不导电，因此只有介电项：

$$
\sigma_s^* = i\omega\varepsilon_s
$$

其中 $\varepsilon_s$ 是固相高频介电常数。

Berea sandstone 示例中：

| 参数 | 数值 |
|---|---|
| $\sigma_w$ | $0.043\ \mathrm{S/m}$ |
| $\varepsilon_w$ | $80\varepsilon_0$ |
| $\varepsilon_s$ | $7\varepsilon_0$ |
| $\varepsilon_0$ | $8.85\times10^{-12}\ \mathrm{F/m}$ |

---

## 7. 核心控制方程

最终模拟是在真实微 CT 二值结构上求解电势场。控制方程是频域下的电流守恒方程：

$$
\nabla\cdot J_t = -\nabla\cdot\left(\sigma_x^*\nabla u\right)=0
$$

其中：

| 符号 | 含义 |
|---|---|
| $J_t$ | 总电流密度 |
| $\sigma_x^*$ | 空间位置 $x$ 处的复电导率 |
| $u$ | 电势 |
| $x$ | REV 内部空间位置 |

局部电场为：

$$
E = -\nabla u
$$

局部总电流密度为：

$$
J_t = \sigma_x^*E = -\sigma_x^*\nabla u
$$

相属性按体素位置赋值：

$$
\sigma_x^* =
\begin{cases}
 i\omega\varepsilon_s, & x \in \text{solid phase} \\
 \sigma_w + i\omega\varepsilon_w + \Delta\sigma_w^*, & x \in \text{water phase}
\end{cases}
$$

如果某次模拟不考虑电化学极化，则令：

$$
\Delta\sigma_w^*=0
$$

---

## 8. 边界条件与有效性质计算

### 8.1 边界条件

论文明确说明：在 REV 两端施加电势梯度，然后数值求解电势 $u$ 和总电流密度 $J_t$ 的空间分布。

因此可以概括为：

- 在一个方向上施加宏观电势差或电势梯度；
- 在体素结构内部求解电流守恒方程；
- 固相和水相界面不需要额外手动设置界面极化模型，界面处的电学不连续由 $\sigma_x^*$ 的空间突变自然体现。

需要注意：论文正文没有非常详细展开侧向边界条件的形式，例如是否严格采用周期边界、绝缘边界或某种混合处理。文中明确给出的关键信息是“across the REV applied electrical potential gradient”。所以复现时，最稳妥的写法是：以施加宏观电势梯度作为主边界条件，侧向边界处理需要结合 AC3D.F / NIST 代码或补充材料进一步确认。

### 8.2 有效复电导率计算

求解得到局部场之后，有效复电导率由体积平均电流密度和体积平均电场给出：

$$
\sigma_{\mathrm{eff}}^* = \frac{\langle J_t\rangle}{\langle E\rangle}
$$

其中：

| 符号 | 含义 |
|---|---|
| $\langle J_t\rangle$ | REV 体积平均总电流密度 |
| $\langle E\rangle$ | REV 体积平均电场 |
| $\sigma_{\mathrm{eff}}^*$ | 样品有效复电导率 |

由 $\sigma_{\mathrm{eff}}^*$ 可以得到：

$$
\sigma_{\mathrm{eff}}^*=\sigma'_{\mathrm{eff}}+i\sigma''_{\mathrm{eff}}
$$

有效介电常数实部可由：

$$
\sigma''_{\mathrm{eff}} = \omega\varepsilon'_{\mathrm{eff}}
$$

得到：

$$
\varepsilon'_{\mathrm{eff}} = \frac{\sigma''_{\mathrm{eff}}}{\omega}
$$

---

## 9. 不同极化模式如何单独模拟

论文为了区分不同极化的贡献，做了“只开启一个极化机制”的数值实验。

## 9.1 只模拟界面极化

操作方式：

- 固相赋值：

$$
\sigma_s^*=i\omega\varepsilon_s
$$

- 水相赋值：

$$
\sigma_w^*=\sigma_w+i\omega\varepsilon_w
$$

- 不加入孔极化和膜极化：

$$
\Delta\sigma_w^*=0
$$

这样，模拟中唯一能导致频散的机制就是固相和水相之间的电学性质差异，也就是界面极化。

## 9.2 只模拟孔极化

操作方式：

1. 用 Schwarz 型孔极化公式计算每个孔径对应的 $C_p^*(r)$：

$$
C_p^* = \frac{i\omega\tau_p}{1+i\omega\tau_p}\Sigma_S
$$

2. 用孔径分布卷积：

$$
C_P^* = f(r)\otimes C_p^*(r)
$$

3. 将其转换为水相额外复电导率：

$$
\Delta\sigma_w^* = \frac{2C_P^*}{\Lambda}
$$

4. 水相赋值为：

$$
\sigma_w^*=\sigma_w+\Delta\sigma_w^*
$$

5. 固相复电导率设为 0：

$$
\sigma_s^*=0
$$

这里作者在单独贡献分析中把固相设为 0，是为了隔离孔极化的贡献，不让固-液界面介电差异引入界面极化。

## 9.3 只模拟膜极化

操作方式：

1. 用 Titov 窄孔喉模型计算每个孔喉长度对应的 $Z_m^*(L)$：

$$
Z_m^* = Z_{dc} \left[
1 - \eta_0 \left(
1 - \frac{1 - \exp(-2\sqrt{i\omega\tau_m})}{2\sqrt{i\omega\tau_m}}
\right)
\right]
$$

2. 计算膜极化引起的复电导扰动：

$$
C_m^* = \frac{1}{Z_m^*} - \frac{1}{Z_{dc}}
$$

3. 用孔喉长度分布卷积：

$$
C_M^* = f(L)\otimes C_m^*(L)
$$

4. 上尺度到水相：

$$
\Delta\sigma_w^* = \frac{2C_M^*}{\Lambda}
$$

5. 水相赋值为：

$$
\sigma_w^*=\sigma_w+\Delta\sigma_w^*
$$

6. 固相复电导率设为 0：

$$
\sigma_s^*=0
$$

## 9.4 同时模拟所有极化

操作方式：

1. 孔极化：

$$
C_P^* = f(r)\otimes C_p^*(r)
$$

2. 膜极化：

$$
C_M^* = f(L)\otimes C_m^*(L)
$$

3. 叠加：

$$
C^*=C_P^*+C_M^*
$$

4. 上尺度：

$$
\Delta\sigma_w^*=\frac{2C^*}{\Lambda}
$$

5. 水相赋值：

$$
\sigma_w^*=\sigma_w+i\omega\varepsilon_w+\Delta\sigma_w^*
$$

6. 固相赋值：

$$
\sigma_s^*=i\omega\varepsilon_s
$$

7. 在微 CT 真实二相结构上求解：

$$
\nabla\cdot J_t=-\nabla\cdot(\sigma_x^*\nabla u)=0
$$

8. 由平均场计算：

$$
\sigma_{\mathrm{eff}}^*=\frac{\langle J_t\rangle}{\langle E\rangle}
$$

---

## 10. 论文模拟流程总结

可以把作者的模拟框架概括为以下流程：

### Step 1：输入微 CT 二值图像

把 Berea sandstone 的微 CT 图像分割为：

- 固相；
- 孔隙水相。

论文中使用的 REV 是 $350^3$ 个体素，体素尺寸为 $2.8\ \mu\mathrm{m}$。

### Step 2：提取几何统计量

从二值图像中提取：

- 孔隙度；
- 比表面积；
- 孔节点尺寸分布 $f(r)$；
- 孔喉长度分布 $f(L)$。

论文使用 Dong & Blunt (2009) 的孔隙网络提取算法来获得孔径分布和孔喉长度分布。

### Step 3：计算孔极化复电导

用 Schwarz 型公式计算单孔响应，再与孔径分布卷积：

$$
C_P^* = f(r)\otimes C_p^*(r)
$$

### Step 4：计算膜极化复电导

用 Titov 窄孔喉模型计算单孔喉响应，再与孔喉长度分布卷积：

$$
C_M^* = f(L)\otimes C_m^*(L)
$$

### Step 5：叠加电化学极化

$$
C^*=C_P^*+C_M^*
$$

### Step 6：上尺度到水相复电导率

$$
\Delta\sigma_w^*=\frac{2C^*}{\Lambda}
$$

### Step 7：构造每个体素的复电导率

水相：

$$
\sigma_w^*=\sigma_w+i\omega\varepsilon_w+\Delta\sigma_w^*
$$

固相：

$$
\sigma_s^*=i\omega\varepsilon_s
$$

### Step 8：求解控制方程

$$
\nabla\cdot J_t=-\nabla\cdot(\sigma_x^*\nabla u)=0
$$

### Step 9：计算有效复电导率

$$
\sigma_{\mathrm{eff}}^*=\frac{\langle J_t\rangle}{\langle E\rangle}
$$

### Step 10：输出有效电导率和有效介电常数

$$
\sigma'_{\mathrm{eff}}=\mathrm{Re}(\sigma_{\mathrm{eff}}^*)
$$

$$
\varepsilon'_{\mathrm{eff}}=\frac{\mathrm{Im}(\sigma_{\mathrm{eff}}^*)}{\omega}
$$

---

## 11. 不同极化机制在频率上的贡献

根据论文 Figure 7 的贡献分解：

| 极化机制 | 主要频率范围 | 作用特征 |
|---|---|---|
| 界面极化 | 高频段，约 $>10^5\ \mathrm{Hz}$ 附近 | 固-液两相电学不连续引起，控制高频介电常数变化 |
| 膜极化 | 中间频率，约 $10^2$ 到 $10^5\ \mathrm{Hz}$ | 窄孔喉阳离子选择性导致，对中频有效介电常数贡献明显 |
| 孔极化 | 低频段，约 $<10^2\ \mathrm{Hz}$ | 孔节点尺度 EDL 离子重新分布导致，低频介电常数显著升高 |

论文结论中强调：

- 高频段主要由界面极化控制；
- 中频段膜极化贡献显著，如果忽略膜极化，会低估有效介电常数；
- 低频段孔极化占主导；
- 模拟低频端与实验仍有差异，可能是因为模拟 REV 体积较小，没有包含更大孔隙产生的更长弛豫时间。

---

## 12. 这篇论文方法的核心思想

这篇论文最关键的思想可以概括为一句话：

**把难以直接解析的纳米尺度电双层电化学极化，通过孔径分布和孔喉长度分布转换成水相的频率相关复电导率扰动，再在真实微 CT 二相结构上求解电势场，从而同时考虑电化学极化和界面极化。**

更直观地说：

- 孔极化、膜极化不是直接在体素网格中解析离子浓度场；
- 它们先由解析/半解析模型给出 $C^*$；
- 再通过 $\Delta\sigma_w^*=2C^*/\Lambda$ 加到水相里；
- 界面极化不需要额外模型，因为真实固-液界面和两相电学性质差异已经在 $\sigma_x^*$ 场中；
- 最终统一通过一个复系数 Laplace 方程求解。

---

## 13. 对复现实现最重要的公式清单

| 编号 | 公式 | 用途 |
|---|---|---|
| 1 | $J_t=J_c+J_d$ | 总电流分解 |
| 2 | $J_c=\sigma_{dc}E$ | 传导电流 |
| 3 | $J_d=\partial D/\partial t$ | 位移电流 |
| 4 | $D=\varepsilon_d^*E$ | 电位移-电场关系 |
| 5 | $\sigma^*=\sigma'+i\sigma''$ | 复电导率定义 |
| 6 | $C_P^*=f(r)\otimes C_p^*(r)$ | 孔极化分布叠加 |
| 7 | $C_M^*=f(L)\otimes C_m^*(L)$ | 膜极化分布叠加 |
| 8 | $C^*=C_P^*+C_M^*$ | 电化学极化合并 |
| 9 | $\Delta\sigma_w^*=2C^*/\Lambda$ | 上尺度到水相 |
| 10 | $\sigma_w^*=\sigma_w+i\omega\varepsilon_w+\Delta\sigma_w^*$ | 水相表观复电导率 |
| 11 | $\nabla\cdot J_t=-\nabla\cdot(\sigma_x^*\nabla u)=0$ | 核心控制方程 |
| 12 | $\sigma_{\mathrm{eff}}^*=\langle J_t\rangle/\langle E\rangle$ | 有效复电导率计算 |
| 13 | $C_p^*=\frac{i\omega\tau_p}{1+i\omega\tau_p}\Sigma_S$ | 单孔孔极化模型 |
| 14 | $\tau_p=r^2/(2D)$ | 孔极化弛豫时间 |
| 15 | $Z_m^*=Z_{dc}[1-\eta_0(1-\frac{1-e^{-2\sqrt{i\omega\tau_m}}}{2\sqrt{i\omega\tau_m}})]$ | 膜极化复阻抗 |
| 16 | $\tau_m=L^2/(4D)$ | 膜极化弛豫时间 |
| 17 | $C_m^*=1/Z_m^*-1/Z_{dc}$ | 膜极化复电导扰动 |

---

## 14. 复现时需要特别注意的地方

1. **界面极化不是通过额外公式加进去的。** 只要真实微结构中有固相和水相，而且二者 $\sigma^*$ 不同，界面极化会从控制方程中自然出现。

2. **孔极化和膜极化是先算成复电导，再上尺度到水相。** 不是直接在三维体素里求解离子扩散方程。

3. **孔极化公式来自 Schwarz 模型的改写。** 原模型针对颗粒，论文把颗粒半径替换为孔半径，这是一个近似处理。

4. **膜极化使用 Titov 窄孔喉模型。** 其输出是复阻抗，必须先取倒数并减去直流导通项，才能得到极化诱导的复电导扰动。

5. **单独贡献分析中，作者会人为关闭其他机制。** 例如只算孔极化/膜极化时，固相复电导率设为 0，以避免界面极化混入。

6. **边界条件细节需要谨慎。** 论文正文明确说明施加跨 REV 的电势梯度，但没有在正文中充分展开侧边界的具体数值实现。若严格复现，需要进一步查看 AC3D.F / NIST 代码说明。

