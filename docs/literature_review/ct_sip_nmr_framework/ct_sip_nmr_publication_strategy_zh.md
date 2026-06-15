# CT-SIP-NMR 孔隙尺度模拟框架：文献调研与投稿策略

日期：2026-06-07

## 1. 检索范围

目标稿件是一篇方法/框架类论文，围绕以下工作展开：

- 以 micro-CT 或分割数字岩心几何作为孔隙尺度结构输入；
- 以实验室 SIP 或复电导率/复介电常数频谱作为电学验证数据；
- 以低场 NMR/T2 反演得到的孔径或流体储集约束作为独立孔隙结构验证；
- 使用自研孔隙尺度 AC 模拟框架计算有效复电导率/复介电常数，并将模拟频谱与实验结果对比。

第一轮英文检索词包括：

- `pore-scale simulation spectral induced polarization micro-CT`
- `complex conductivity porous media microtomography`
- `spectral induced polarization porosimetry pore size distribution NMR`
- `nuclear magnetic resonance complex conductivity permeability porous media`
- `digital rock physics effective properties CT segmentation`
- `microfluidics spectral induced polarization calcite dissolution precipitation`

检索使用了 OpenAlex、Crossref、DOI 记录、出版社页面和网页核验。随后调用了两个子智能体：一个负责审核候选论文的相关性，另一个负责下载可合法获取的 PDF，或将无法下载的条目标记为手动下载。

## 2. 主要证据链

### 2.1 文献明确证明的内容

已有文献明确表明，围绕以下方向已经形成了成熟但仍较分散的研究空间：

1. micro-CT/数字岩心成像与有效输运性质的孔隙尺度模拟；
2. SIP/复电导率作为对孔隙结构敏感的电学测量方法；
3. NMR 作为孔径和流体储集测量方法，可与 SIP 派生的特征长度尺度进行对比；
4. 微流控或直接成像实验可用于解释反应性孔隙尺度过程中的 SIP 响应。

Niu et al. (2020) 是最接近的直接先例：它建立了 CT 驱动的孔隙尺度 AC 电学模拟框架。Zhang et al. (2018) 和 Osterman et al. (2016) 支持 CT/NMR/SIP 多模态实验动机。Rembert et al. (2023, 2024) 和 Qiang et al. (2024) 则说明当前高水平研究正在走向“孔隙尺度直接观测 + 电学模拟或岩石物理建模”的方向。

### 2.2 文献支持但尚未完全解决的问题

文献支持 SIP 弛豫时间和虚部电导率与孔径或孔喉长度尺度相关，但具体物理长度尺度会依赖机制。因此，论文不宜声称 NMR、CT 和 SIP 测量的是同一个孔径。更稳健的表述是：

> NMR、CT 和 SIP 为孔体、孔喉、连通性、比表面积以及电化学极化长度尺度提供互补约束。

这正是机理型模拟框架可以贡献的地方：它可以检验哪些孔隙尺度几何假设和极化假设能够复现实测频谱。

### 2.3 应作为解释而非直接事实的内容

除非实验独立约束了对应的表面化学性质和孔隙尺度电荷分布，否则“该框架能够分离 Maxwell-Wagner、孔极化、膜极化、表面/EDL 极化、颗粒/界面极化”等说法应表述为基于模型的解释。

若要支撑一篇较强期刊论文，每一种机制都应具备：

- 清晰的数学源项或本构假设；
- 可追踪的参数和单位；
- 敏感性分析；
- 机制消融图，用于说明该机制是否确实为实验数据所需要。

## 3. 审核后的核心文献

### A. 用于定位创新性的核心文献

1. Niu, Zhang, and Prasad (2020), *A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity of Porous Media in the Frequency Range From 1 mHz to 1 GHz*, Journal of Geophysical Research: Solid Earth. DOI: https://doi.org/10.1029/2020JB020515  
   用途：CT 驱动 AC 电学孔隙尺度模拟的直接前作和基准。

2. Day-Lewis and Johnson (2022), *Pore-Scale Simulation of Spectral Induced Polarization*, OSTI technical report. DOI: https://doi.org/10.2172/1989486  
   用途：SIP 的孔网等效电路路线，可用于对比本项目的体素/场求解路线。

3. Zhang et al. (2018), *Enhanced pore space analysis by use of μ-CT, MIP, NMR, and SIP*, Solid Earth. DOI: https://doi.org/10.5194/se-9-1225-2018  
   用途：最接近 CT/NMR/SIP 实验组合的多模态孔隙空间表征文献。

4. Osterman et al. (2016), *A laboratory study to estimate pore geometric parameters of sandstones using complex conductivity and nuclear magnetic resonance for permeability prediction*, Water Resources Research. DOI: https://doi.org/10.1002/2015WR018472  
   用途：实验室 NMR 与复电导率联合比较，用于估计孔隙几何和渗透率。

5. Rembert et al. (2024), *Microfluidics and Spectral Induced Polarization for Direct Observation and Petrophysical Modeling of Calcite Dissolution*, Geophysical Research Letters. DOI: https://doi.org/10.1029/2024GL111271  
   用途：直接孔隙尺度成像 + SIP，用于反应性方解石溶蚀过程。

6. Rembert et al. (2023), *A microfluidic chip for geoelectrical monitoring of critical zone processes*, Lab on a Chip. DOI: https://doi.org/10.1039/D3LC00377A  
   用途：图像-SIP 耦合微流控实验平台。

7. Qiang et al. (2024), *Quantitative Evaluation of the Effect of Pore Fluids Distribution on Complex Conductivity Saturation Exponents*, Journal of Geophysical Research: Solid Earth. DOI: https://doi.org/10.1029/2024JB028689  
   用途：近期孔隙尺度流体分布 + 复电导率模拟研究。

8. Revil, Florsch, and Camerlynck (2014), *Spectral induced polarization porosimetry*, Geophysical Journal International. DOI: https://doi.org/10.1093/gji/ggu180  
   用途：SIP 弛豫/孔径分布理论和 SIP 孔隙测量框架。

9. Schwarz (1962), *A theory of the low-frequency dielectric dispersion of colloidal particles in electrolyte solution*, Journal of Physical Chemistry. DOI: https://doi.org/10.1021/j100818a067  
   用途：经典颗粒/界面极化模型背景。

### B. 方法支撑文献

10. Hu and Blunt (2009), *Pore-network extraction from micro-computerized-tomography images*, Physical Review E. DOI: https://doi.org/10.1103/PhysRevE.80.036307  
    用途：maximal-ball 孔网提取与几何表征。

11. Raeini, Bijeljic, and Blunt (2017), *Generalized network modeling: Network extraction as a coarse-scale discretization of the void space of porous media*, Physical Review E. DOI: https://doi.org/10.1103/PhysRevE.96.013312  
    用途：将孔隙空间网络提取理解为降阶离散化。

12. Andrae et al. (2012), *Digital rock physics benchmarks - Part I: Imaging and segmentation*, Computers & Geosciences. DOI: https://doi.org/10.1016/j.cageo.2012.09.005  
    用途：CT 成像与分割基准背景。

13. Andrae et al. (2012), *Digital rock physics benchmarks - Part II: Computing effective properties*, Computers & Geosciences. DOI: https://doi.org/10.1016/j.cageo.2012.09.008  
    用途：有效性质数值计算基准背景。

14. Blunt et al. (2012), *Pore-scale imaging and modelling*, Advances in Water Resources. DOI: https://doi.org/10.1016/j.advwatres.2012.03.003  
    用途：数字岩心/孔隙尺度建模总述。

15. Gostick et al. (2019), *PoreSpy: A Python Toolkit for Quantitative Analysis of Porous Media Images*, Journal of Open Source Software. DOI: https://doi.org/10.21105/joss.01296  
    用途：图像分析和孔隙结构指标计算。

16. Johnson, Koplik, and Schwartz (1986), *New Pore-Size Parameter Characterizing Transport in Porous Media*, Physical Review Letters. DOI: https://doi.org/10.1103/PhysRevLett.57.2564  
    用途：特征孔径/输运长度尺度理论。

17. Revil and Florsch (2010), *Determination of permeability from spectral induced polarization in granular media*, Geophysical Journal International. DOI: https://doi.org/10.1111/j.1365-246X.2010.04573.x  
    用途：SIP 弛豫与渗透率关系。

18. Revil et al. (2015), *Predicting permeability from the characteristic relaxation time and intrinsic formation factor of complex conductivity spectra*, Water Resources Research. DOI: https://doi.org/10.1002/2015WR017074  
    用途：弛豫时间、形成因子与渗透率关系。

19. Weller and Slater (2019), *Permeability estimation from induced polarization: an evaluation of geophysical length scales using an effective hydraulic radius concept*, Near Surface Geophysics. DOI: https://doi.org/10.1002/nsg.12071  
    用途：地球物理长度尺度和有效水力半径解释。

20. Zhang, Niu, and Zhang (2018), *Estimating pore-size distribution in carbonate reservoir rocks using joint inversion of NMR and complex conductivity data*, SEG Technical Program Expanded Abstracts. DOI: https://doi.org/10.1190/segam2018-2997894.1  
    用途：碳酸盐岩中 NMR 与复电导率联合反演孔径分布。

21. Revil et al. (2017), *Complex conductivity of soils*, Water Resources Research. DOI: https://doi.org/10.1002/2017WR020655  
    用途：表面导电和复电导率模型背景。

22. Xiong, Baychev, and Jivkov (2016), *Review of pore network modelling of porous media: Experimental characterisations, network constructions and applications to reactive transport*, Journal of Contaminant Hydrology. DOI: https://doi.org/10.1016/j.jconhyd.2016.07.002  
    用途：孔网模型背景及其局限。

### C. 讨论和背景文献

23. Kemna et al. (2012), *An overview of the spectral induced polarization method for near-surface applications*, Near Surface Geophysics. DOI: https://doi.org/10.3997/1873-0604.2012027  
    用途：SIP 方法背景。

24. Cnudde and Boone (2013), *High-resolution X-ray computed tomography in geosciences: A review of the current technology and applications*, Earth-Science Reviews. DOI: https://doi.org/10.1016/j.earscirev.2013.04.003  
    用途：CT 方法背景。

25. Anovitz and Cole (2015), *Characterization and Analysis of Porosity and Pore Structures*, Reviews in Mineralogy and Geochemistry. DOI: https://doi.org/10.2138/rmg.2015.80.04  
    用途：孔隙结构表征综述。

26. Jougnot et al. (2009), *Spectral induced polarization of partially saturated clay-rocks: a mechanistic approach*, Geophysical Journal International. DOI: https://doi.org/10.1111/j.1365-246X.2009.04426.x  
    用途：饱和度和表面过程机制背景。

27. Revil and Skold (2011), *Salinity dependence of spectral induced polarization in sands and sandstones*, Geophysical Journal International. DOI: https://doi.org/10.1111/j.1365-246X.2011.05181.x  
    用途：孔隙流体盐度敏感性。

28. Swanson et al. (2015), *Anomalous solute transport in saturated porous media: Relating transport model parameters to electrical and nuclear magnetic resonance properties*, Water Resources Research. DOI: https://doi.org/10.1002/2014WR015284  
    用途：NMR/电学性质与输运参数之间的联系。

29. Niu and Zhang (2019), *Permeability Prediction in Rocks Experiencing Mineral Precipitation and Dissolution: A Numerical Study*, Water Resources Research. DOI: https://doi.org/10.1029/2018WR024174  
    用途：矿物反应过程中孔隙结构演化与渗透率变化。

30. Tarasov and Titov (2013), *On the use of the Cole-Cole equations in spectral induced polarization*, Geophysical Journal International. DOI: https://doi.org/10.1093/gji/ggt251  
    用途：Cole-Cole 参数解释中的注意事项。

## 4. 下载状态

详细清单：

- `docs/literature_review/ct_sip_nmr_framework/download_manifest.md`
- `docs/literature_review/ct_sip_nmr_framework/download_manifest.csv`

PDF 目录：

- `docs/literature_review/ct_sip_nmr_framework/papers/`

下载子智能体给出的状态汇总：

- 从合法开放获取来源下载：5 篇
- 从项目本地 PDF 复制：4 篇
- 需要手动下载：21 篇

已下载或复制的 PDF：

- Niu et al. (2020), JGR Solid Earth
- Day-Lewis and Johnson (2022), OSTI report
- Zhang et al. (2018), Solid Earth
- Rembert et al. (2023), Lab on a Chip
- Rembert et al. (2024), GRL
- Hu and Blunt (2009), Physical Review E
- Gostick et al. (2019), JOSS
- Kemna et al. (2012), Near Surface Geophysics
- Cnudde and Boone (2013), Earth-Science Reviews

手动下载优先级：

1. Osterman et al. (2016), WRR, DOI: https://doi.org/10.1002/2015WR018472
2. Qiang et al. (2024), JGR Solid Earth, DOI: https://doi.org/10.1029/2024JB028689
3. Revil et al. (2014), GJI, DOI: https://doi.org/10.1093/gji/ggu180
4. Revil and Florsch (2010), GJI, DOI: https://doi.org/10.1111/j.1365-246X.2010.04573.x
5. Revil et al. (2015), WRR, DOI: https://doi.org/10.1002/2015WR017074
6. Andrae et al. (2012) Part I 和 Part II, DOI: https://doi.org/10.1016/j.cageo.2012.09.005 和 https://doi.org/10.1016/j.cageo.2012.09.008

## 5. 这项工作可以发展成什么类型的论文？

### 最强版本：方法框架类论文

一个较强的目标题目可以接近：

> A CT-, SIP-, and NMR-constrained pore-scale simulation framework for effective complex conductivity and permittivity of porous rocks

核心贡献：

- 直接使用真实三维 CT 几何，而不是只使用经验孔径分布；
- 使用对应样品的 SIP 和 NMR 数据进行多模态实验验证；
- 进行频率相关的孔隙尺度 AC 场求解；
- 做机制消融：Maxwell-Wagner、孔极化、膜极化、颗粒/界面极化和组合响应；
- 建立可复现工作流，清楚记录几何、参数、求解器和后处理来源。

如果模拟-实验对应关系足够强，最适合的目标期刊是：

1. Journal of Geophysical Research: Solid Earth  
   如果论文推进的是岩石物理或水文地球物理认识，而不仅是软件实现，这是最佳目标之一。

2. Water Resources Research  
   如果框架与孔隙尺度流动、输运、渗透率、溶蚀/沉淀或水-岩作用直接相关，这是很合适的目标。

3. Geophysical Journal International 或 Geophysics  
   如果论文强调 SIP 物理、极化机制和复电导率频谱解释，可考虑这两个期刊。

4. Computers & Geosciences  
   如果主要创新在计算框架、可复现代码、GPU/FFT 求解器、数据模型、可视化和工作流集成，这是稳妥目标。

5. Transport in Porous Media  
   如果核心信息是孔隙介质输运物理和上尺度，而不是地球物理方法开发，这是合适选择。

### 第二版本：数字岩心验证论文

如果实验匹配结果不错，但新的物理机制贡献较温和，可以表述为：

> Digital-rock validation of SIP mechanisms using co-registered CT, NMR, and complex conductivity measurements

合适目标期刊：

- Computers & Geosciences
- Transport in Porous Media
- Journal of Petroleum Science and Engineering / Geoenergy Science and Engineering
- Petrophysics
- Near Surface Geophysics

### 第三版本：反应性微流控方法论文

如果后续最强结果来自 AC2D/微流控方解石溶蚀工作流：

> Image-resolved AC simulation of microfluidic SIP responses during calcite dissolution

合适目标期刊：

- Geophysical Research Letters：适合短篇、机制明确且结果有新意的工作。
- Environmental Science & Technology：适合环境或反应过程机制为核心的工作。
- Lab on a Chip：适合包含实验平台或强微流控测量创新的工作。
- Water Resources Research：适合推进反应性输运解释的工作。

### 低风险版本：软件/数据/工作流论文

如果物理对比有价值，但还不足以支撑 JGR/WRR 级别的机制论文，可以表述为：

> An open workflow for CT-based pore-scale complex-conductivity simulation and SIP/NMR comparison

合适目标期刊：

- Computers & Geosciences
- SoftwareX
- Earth Science Informatics
- Journal of Open Source Software：仅当论文主要是软件说明，且代码足够完善、开放时适合。

## 6. 推荐投稿策略

最高价值路线是优先瞄准 JGR: Solid Earth 或 Water Resources Research，但前提是满足以下条件：

1. 至少两个样品，最好具有不同孔隙结构，并且具有同位或清晰配对的 CT、SIP 和 NMR 数据。
2. 模拟框架不仅能复现实部和虚部电导率的量级，还能复现频率趋势。
3. NMR/CT 孔径指标不是简单摆在 SIP 旁边，而是能约束参数或解释模型误差。
4. 机制消融能够说明每个被纳入的极化机制为何必要。
5. 不确定性需要显式呈现：分割阈值、体素尺寸、水相电导率、表面电导率、弛豫长度尺度、边界条件和几何因子。
6. 代码和数据来源足够清楚，使其他研究者能够复现主要图件。

如果上述条件只满足一部分，更现实的首投目标是 Computers & Geosciences 或 Transport in Porous Media。这并不是弱结果：对于自研框架而言，一篇干净的计算方法论文往往比一篇过度声称机制突破的地球物理论文更容易发表。

## 7. 建议论文结构

1. Introduction  
   缺口：CT/数字岩心、SIP 和 NMR 往往被分开使用，或只是经验关联；很少有研究用经过配对实验验证的孔隙尺度 AC 模拟框架把三者闭合起来。

2. Data and samples  
   CT 分割、SIP 实验设置、NMR 处理、样品几何、饱和状态和流体化学。

3. Simulation framework  
   几何输入、相标签、电导率赋值、极化项、AC 场方程、边界条件和有效性质提取。

4. Calibration and validation protocol  
   哪些参数来自测量，哪些参数需要拟合，哪些参数固定为文献值，哪些参数通过敏感性分析检验。

5. Results  
   CT/NMR 孔隙指标；模拟与实测 SIP 频谱对比；机制消融；不确定性包络。

6. Discussion  
   SIP 似乎感知的长度尺度；NMR 与 CT 在哪里一致或不一致；哪些机制是必要的；2D/3D、分割、表面化学和边界条件的局限。

7. Conclusions  
   该框架作为连接孔隙尺度结构、多模态实验和有效复电学性质的桥梁。

## 8. 一句话定位

如果实验和模拟能够较好对应，这项工作可以定位为一篇岩石物理/水文地球物理方法框架论文：一个可复现、CT 解析、由配对 SIP 和 NMR 实验约束并验证的孔隙尺度 AC 模拟框架，用于对复电导率频谱进行机制级解释。

