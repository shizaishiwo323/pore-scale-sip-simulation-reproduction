# Martin J. Blunt：数字岩心、CT 数字岩心与图像分割论文

统计日期：2026-07-18。作者主页：[Martin J. Blunt — Google Scholar](https://scholar.google.com/citations?user=vMSqj1AAAAAJ&hl=en)。

## 统计口径与结果

以题名和刊物信息为依据，剔除了同一工作的预印本/正式发表重复记录、会议摘要、无年份的重复条目，以及仅把 CT 用作实验观测、但不涉及数字岩心建模或图像处理方法的论文。得到 **24 篇**可直接用于本项目的相关正式论文：

| 类别 | 篇数 | 范围 |
|---|---:|---|
| 数字岩心与 micro-CT 图像直接建模 | 13 | 孔隙网络提取、直接数值模拟、渗流/反应输运、CT 驱动工作流 |
| 综述与领域定位 | 3 | 数字岩心/孔隙尺度成像建模的综述或观点文章 |
| 数字岩心图像分割与预处理 | 6 | 多相 CT/X-ray 图像分割、接触角提取、CT 去噪 |
| 数字岩心图像生成 | 2 | 3D 孔隙空间/多相孔隙尺度图像生成 |

其中最贴近当前 `pnextract + microCT + 孔隙尺度求解` 路线的基础论文是 Dong & Blunt (2009)、Mostaghimi et al. (2013)、Raeini et al. (2014, 2015) 与 Pereira Nunes et al. (2016)。最贴近多相 CT 分割的近期论文是 Mahdaviara et al. (2023)、Siavashi et al. (2024)、Gao et al. (2024)、Ye et al. (2025) 与 Zhang et al. (2026)。

> 引用次数为 Scholar 页面在统计日显示的数值；2026 年新论文的引用数未显示，记为“—”。

## 1. 数字岩心与 micro-CT 图像直接建模（13 篇）

| 年份 | 论文 | 主要相关性 | Scholar 引用 |
|---:|---|---|---:|
| 2007 | Okabe, H., & Blunt, M. J. *Pore space reconstruction of vuggy carbonates using microtomography and multiple-point statistics*. **Water Resources Research, 43**(12). | 用 microtomography 与多点统计重建孔隙空间。 | 238 |
| 2007 | Al-Kharusi, A. S., & Blunt, M. J. *Network extraction from sandstone and carbonate pore space images*. **Journal of Petroleum Science and Engineering, 56**(4), 219–231. | 从砂岩/碳酸盐孔隙图像提取网络。 | 442 |
| 2008 | Al-Kharusi, A. S., & Blunt, M. J. *Multiphase flow predictions from carbonate pore space images using extracted network models*. **Water Resources Research, 44**(6). | 图像提取网络上的多相流预测。 | 93 |
| 2009 | Dong, H., & Blunt, M. J. *Pore-network extraction from micro-computerized-tomography images*. **Physical Review E, 80**(3). | `pnextract` 方法论文；本项目最直接的网络提取来源。 | 1,617 |
| 2012 | Mostaghimi, P., Bijeljic, B., & Blunt, M. J. *Simulation of flow and dispersion on pore-space images*. **SPE Journal, 17**(4), 1131–1141. | 在孔隙空间图像上直接模拟流动与弥散。 | 187 |
| 2013 | Mostaghimi, P., Blunt, M. J., & Bijeljic, B. *Computations of absolute permeability on micro-CT images*. **Mathematical Geosciences, 45**(1), 103–125. | CT 体数据上绝对渗透率计算。 | 554 |
| 2014 | Raeini, A. Q., Blunt, M. J., & Bijeljic, B. *Direct simulations of two-phase flow on micro-CT images of porous media and upscaling of pore-scale forces*. **Advances in Water Resources, 74**, 116–126. | micro-CT 上两相流直接模拟与上尺度。 | 400 |
| 2015 | Raeini, A. Q., Bijeljic, B., & Blunt, M. J. *Modelling capillary trapping using finite-volume simulation of two-phase flow directly on micro-CT images*. **Advances in Water Resources, 83**, 102–110. | micro-CT 网格上的有限体积两相流与毛管捕集。 | 151 |
| 2016 | Pereira Nunes, J. P., Blunt, M. J., & Bijeljic, B. *Pore-scale simulation of carbonate dissolution in micro-CT images*. **Journal of Geophysical Research: Solid Earth, 121**(2), 558–576. | CT 数字岩心上的碳酸盐溶蚀模拟。 | 158 |
| 2019 | Raeini, A. Q., Yang, J., Bondino, I., Bultreys, T., Blunt, M. J., & Bijeljic, B. *Validating the generalized pore network model using micro-CT images of two-phase flow*. **Transport in Porous Media, 130**(2), 405–424. | 用两相 micro-CT 数据验证广义孔隙网络模型。 | 75 |
| 2023 | Li, M., Foroughi, S., Zhao, J., Bijeljic, B., & Blunt, M. J. *Image-based pore-scale modelling of the effect of wettability on breakthrough capillary pressure in gas diffusion layers*. **Journal of Power Sources, 584**, 233539. | 基于图像的孔隙尺度润湿性/突破压力建模。 | 19 |
| 2025 | Foroughi, S., Shojaei, M. J., Lane, N., Rashid, B., Lakshtanov, D., Ning, Y., et al. *A framework for multiphase pore-scale modeling based on micro-CT imaging*. **Transport in Porous Media, 152**(3), 18. | 多相 micro-CT 数字岩心建模框架。 | 22 |
| 2026 | Ganguli, S. S., Foroughi, S., Bijeljic, B., & Blunt, M. J. *Computing effective elastic properties from microtomographic images: An efficient 3D digital rock workflow*. **Journal of Geophysical Research: Solid Earth, 131**(6), e2026JB033780. | microtomography 图像的三维数字岩心弹性参数工作流。 | — |

## 2. 综述与领域定位（3 篇）

| 年份 | 论文 | 主要相关性 | Scholar 引用 |
|---:|---|---|---:|
| 2013 | Blunt, M. J., Bijeljic, B., Dong, H., Gharbi, O., Iglauer, S., Mostaghimi, P., et al. *Pore-scale imaging and modelling*. **Advances in Water Resources, 51**, 197–216. | 孔隙尺度成像、分割、网络提取和模拟的经典综述。 | 2,472 |
| 2021 | Wang, Y. D., Blunt, M. J., Armstrong, R. T., & Mostaghimi, P. *Deep learning in pore scale imaging and modeling*. **Earth-Science Reviews, 215**, 103555. | 深度学习用于孔隙尺度成像、分割、重建和模拟的综述。 | 315 |
| 2025 | Blunt, M. J., Sun, S., Boone, M. A., Zhang, L., & Cai, J. *Digital rock physics and fluid flow in the context of the energy transition*. **Advances in Geo-Energy Research, 18**(3), 299–302. | 数字岩心在能源转型中的领域定位与应用概述。 | 7 |

## 3. 数字岩心图像分割与预处理（6 篇）

| 年份 | 论文 | 主要相关性 | Scholar 引用 |
|---:|---|---|---:|
| 2023 | Mahdaviara, M., Shojaei, M. J., Siavashi, J., Sharifi, M., & Blunt, M. J. *Deep learning for multiphase segmentation of X-ray images of gas diffusion layers*. **Fuel, 345**, 128180. | 深度学习多相 X-ray 图像分割。 | 45 |
| 2024 | Siavashi, J., Mahdaviara, M., Shojaei, M. J., Sharifi, M., & Blunt, M. J. *Segmentation of two-phase flow X-ray tomography images to determine contact angle using deep autoencoders*. **Energy, 288**, 129698. | 深度自编码器分割两相 CT 图像，并用于接触角测量。 | 29 |
| 2024 | Gao, Y., Foroughi, S., Ma, Z., Yuan, S., Xiao, L., Bijeljic, B., et al. *Gradient information enhanced image segmentation and automatic in situ contact angle measurement applied to images of multiphase flow in porous media*. **Water Resources Research, 60**(9), e2023WR036869. | 梯度增强分割与原位接触角自动测量。 | 4 |
| 2025 | Ye, S., Song, X., Ma, Z., Yang, G., Zhou, L., Zhou, M., et al. *A noise-resistant and annotation-free supervoxel-based algorithm for rapid segmentation of multiphase X-ray images*. **Advances in Geo-Energy Research, 16**(1), 50–59. | 无标注、抗噪的多相 X-ray 超体素分割。 | 10 |
| 2026 | Zhang, R., Song, X., Zhu, L., Bijeljic, B., Li, G., & Blunt, M. J. *SAMamba3D: Adapting segment anything for generalizable three-dimensional segmentation of multiphase pore-scale images*. **Advances in Geo-Energy Research, 21**(2), 109–124. | 面向多相孔隙尺度图像的可泛化三维分割。 | — |
| 2026 | Kang, M., Ma, Q., Xiao, L., Liao, G., Bijeljic, B., & Blunt, M. J. *A conditional latent diffusion framework for dynamic CT image denoising in multiphase flow and reactive transport studies of porous media*. **Computers & Geosciences**, 106202. | 动态 CT 去噪；属于分割前预处理与数据质量提升。 | — |

## 4. 数字岩心图像生成（2 篇）

| 年份 | 论文 | 主要相关性 | Scholar 引用 |
|---:|---|---|---:|
| 2024 | Zhu, L., Bijeljic, B., & Blunt, M. J. *Generation of pore-space images using improved pyramid Wasserstein generative adversarial networks*. **Advances in Water Resources, 190**, 104748. | GAN 生成孔隙空间图像，可用于数字岩心扩增。 | 23 |
| 2025 | Zhu, L., Bijeljic, B., & Blunt, M. J. *Diffusion model-based generation of three-dimensional multiphase pore-scale images*. **Transport in Porous Media, 152**(3), 22. | 扩散模型生成三维多相孔隙尺度图像。 | 20 |

## 建议优先阅读顺序

1. Dong & Blunt (2009) → 明确 `pnextract` 的网络定义和提取逻辑。
2. Blunt et al. (2013) → 建立 CT 成像到孔隙尺度模拟的总体框架。
3. Mostaghimi et al. (2013)、Raeini et al. (2014, 2015) → 对照现有 micro-CT 体数据上的直接求解路线。
4. Pereira Nunes et al. (2016) → 与当前碳酸盐溶蚀/反应输运扩展最接近。
5. Gao et al. (2024)、Ye et al. (2025)、Zhang et al. (2026) → 作为后续多相 CT 分割方法的候选基线。
