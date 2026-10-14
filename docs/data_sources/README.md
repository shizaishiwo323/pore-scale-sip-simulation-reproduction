# 输入资料清单

`local_inputs_sha256.csv` 是 2026-10-12 上传前的输入快照，记录当时尚未被 Git 跟踪的 `data/` 文件：共 4,169 个，15,590,286,710 字节（约 14.52 GiB）。每条记录包含相对于项目根目录的路径、字节数和按原文件内容计算的 SHA-256；汇总及清单自身的 SHA-256 见 `local_inputs_summary.json`。

自 2026-10-14 起，清单中的项目输入资料随 `main` 上传；大型 CT/体数据通过 Git LFS 保存。清单中的 `local_only` 是上传前快照状态，保留该字段以避免改写已发布的清单哈希，不代表当前 Git 状态。此前已经被仓库跟踪的轻量输入未包含在这份新增清单中。

主要路径包括：

- `data/ct_backed_samples_raw_copy_20260605/`：CT、SIP、NMR 及样品物性资料。
- `data/Niu 2020data/`：Berea 体数据、Avizo 文件、参考资料和 Figure 工作簿。
- `data/open_datasets/`：公开数据及下载说明。
- `data/复现rembert模拟/`：供微流控工作流使用的历史模拟输入。

`category` 区分普通本地输入（`local_input`）、已有测量处理文件（`processed_measurement`）和历史模拟输入（`legacy_simulation_input`）。这份清单只标识文件，不为其科学正确性或来源权限提供认证。

复跑时将资料放回清单中的相对路径，然后核对字节数与 SHA-256。本次校验只读取原文件，没有修改、移动或删除输入。

求解器产生的 `results/`、`outputs/`、缓存和 notebook 运行输出不随 main 发布。notebook 的代码与说明仍保留。独立 `pnextract/` 源码和 Windows 运行文件已经纳入仓库；其 194 个与 `code/vendor/pnextract/` 对应的源码文件在统一换行符后内容一致。

`data/dissolution_results*/`、`figures/`、临时目录、日志和根目录的临时笔记也仅在本地保存。`data/dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square/` 的 113 个历史输出已退出版本控制；它们也是微流控工作流的输入，复跑时仍需保留在该相对路径。历史版本可从提交 `00383e0` 恢复，未包含在上述新增输入校验清单中。

当前运行文件的实际 SHA-256 见 `pnextract_runtime_sha256.json`。其中 `pnextract.exe` 与随附上游校验文档中的哈希不同；上传的是本机已有运行文件，未重新编译，也未独立验证源码与该二进制完全对应。上游校验文档保持原样，实际文件应以本次记录的哈希核对。

三个 `pnextract/bin/*.exe` 以及 `data/` 中的 RAW、HDF5、TIFF、IM6 和 Avizo 体数据使用 Git LFS 保存；普通 Git 树中保存 LFS 指针。检出完整输入和运行文件前执行 `git lfs install` 和 `git lfs pull`。
