# GFZ 多方法岩石物性数据集

- 数据集：Multi-method petrophysical laboratory data set for crushed carbonates and sandstone
- DOI：https://doi.org/10.5880/fidgeo.2021.044
- 作者：Jana H. Börner、Volker Herdegen、Jens-Uwe Repke、Klaus Spitzer
- 许可：CC BY 4.0
- 下载日期：2026-08-15
- 官方原始数据 ZIP 已保留，并完整解压到 `extracted/crushed_rock_data/`。

## CT、NMR 和 SIP 位置

- CT：`extracted/crushed_rock_data/07_microCT/`
- NMR：`extracted/crushed_rock_data/09_NMR/`
- SIP：`extracted/crushed_rock_data/10_SIP/`
- 官方数据说明：`2021-004_Boerner-et-al_Crushed-Rock-Data-Description.pdf`

三类数据共同覆盖的岩性是 Wellenkalk（WK）：

- CT：`muCT_WK.jpeg`
- NMR：`NMR_data.csv` 或 `NMR_data.xlsx` 中的 plug、WK2、WK3、WK7 数据列
- SIP：`SIP_data_plugs.csv` 中的 Wellenkalk WK 频谱，以及 `SIP_data_crushed.csv` 中的 WK 粒级数据

## 科学边界

开放的 CT 文件只是作者从三维重建体中提取的一张水平灰度切片，不是完整三维 CT 体。原实验重建体的体素分辨率约为 19.7 µm。

CT 使用 type-2 岩心；SIP 岩心数据也使用 type-2 岩心；NMR 使用 type-3 Wellenkalk 岩心及部分破碎粒级。资料没有提供唯一试件编号证明 CT 与 SIP 是同一根具体岩心，因此应表述为同岩性多方法数据，不应表述为同一测量体积的严格配准 CT–SIP–NMR 数据。

## 完整性

- 官方 ZIP：17,421,885 bytes
- ZIP CRC：通过
- 关键文件 SHA-256：见 `DOWNLOAD_MANIFEST.csv`
