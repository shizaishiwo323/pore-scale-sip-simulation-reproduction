# DRP-421 下载说明

- 数据集：3D Sandstone Sample Images
- 当前项目页：https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-421
- 数据 DOI：https://doi.org/10.17612/FRCM-CN23
- 许可：ODC-BY 1.0
- 作者：Yingzhi Cui、Christoph H. Arns、Igor Shikhov
- 机构：The University of New South Wales
- 下载日期：2026-08-15

## 数据内容

本目录保存项目公开的全部 7 个文件，均为 NetCDF 数据压缩包：Bentheimer、Castlegate 和 Leopard 砂岩的 800×800×800 CT 灰度体及分割体。体素分辨率和相标签见 `DOWNLOAD_MANIFEST.csv`。

## NMR 数据边界

项目公开文件树中没有独立的实验 NMR 衰减、T2 分布或反演结果文件。项目说明仅指出这些图像可用于通过模拟与实验匹配来反演 NMR 相属性。2022 年论文确实报告了 Bentheimer 岩心的实验 NMR T2 对比，但论文的数据声明只开放了 tomogram data。因此，本次没有把论文图中的曲线当作“开放原始 NMR 数据”。

相关开放获取论文 PDF 保存于：

`../../../docs/references/nmr/Cui_Shikhov_Arns_2022_NMR_Relaxation_Modelling.pdf`

论文 DOI：https://doi.org/10.1007/s11242-022-01752-0

PDF SHA-256：`25D6127D2B1263DC788BF31576E0A0311BD92185A4F9296BCFC8E9A548843FD0`

## 完整性验证

- 7/7 文件字节数与平台 API 一致。
- 7/7 ZIP 通过 Python `zipfile.testzip()` CRC 校验。
- 每个数据文件的 SHA-256 见 `DOWNLOAD_MANIFEST.csv`。
