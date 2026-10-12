# 本地输入资料清单

`local_inputs_sha256.csv` 记录截至 2026-10-12 本机存在、此前未被 Git 跟踪的 `data/` 文件：共 4,169 个，15,590,286,710 字节（约 14.52 GiB）。每条记录包含相对于项目根目录的路径、字节数和按原文件内容计算的 SHA-256；汇总及清单自身的 SHA-256 见 `local_inputs_summary.json`。

这些资料保留在本地，GitHub 上传清单，不上传其文件内容。此前已经被仓库跟踪的输入资料继续保留，未包含在这份新增清单中。清单不是下载地址，也不能仅凭清单恢复数据。

主要路径包括：

- `data/ct_backed_samples_raw_copy_20260605/`：CT、SIP、NMR 及样品物性资料。
- `data/Niu 2020data/`：本地 Berea 体数据、Avizo 文件和参考资料；已经跟踪的 Figure 工作簿仍在仓库中。
- `data/open_datasets/`：本地下载的较大公开数据文件；下载说明和此前跟踪的轻量数据仍在仓库中。
- `data/复现rembert模拟/`：供本地微流控工作流使用的历史模拟输入。

`category` 区分普通本地输入（`local_input`）、已有测量处理文件（`processed_measurement`）和历史模拟输入（`legacy_simulation_input`）。这份清单只标识文件，不为其科学正确性或来源权限提供认证。

复跑时将资料放回清单中的相对路径，然后核对字节数与 SHA-256。本次校验只读取原文件，没有修改、移动或删除输入。

求解器产生的 `results/`、`outputs/`、缓存和 notebook 运行输出不随本次 main 发布。notebook 的代码与说明仍保留。独立 `pnextract/` 源码和 Windows 运行文件已经纳入仓库；其 194 个与 `code/vendor/pnextract/` 对应的源码文件在统一换行符后内容一致。
