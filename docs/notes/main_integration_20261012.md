# main 集成与验证记录（2026-10-12）

## 提交范围

- 集成前 main：`021e91ccc4016eac55cc9bfd58e047499c7b71ae`。
- GitHub 开发分支：`codex/niu2020-reproduction-hardening`，原提交 `b2d0b4e0ec1be542a241c32cc65c8b1a4fb61959`。
- 本地最新求解器、极化模型、几何反演、样品运行及绘图脚本：`de86d4d78627dc301fefb410e3fabd584f6596db`。

开发分支包含集成前 main 的全部历史，因此采用快进合并，保留开发分支和本地最新代码。没有覆盖式选择旧版求解器，没有改写远端历史，也没有删除开发分支。

另纳入独立 pnextract 源码与运行文件、本地输入校验清单和发布范围说明。原始输入、既有求解结果保留在本地；本地原有缺失的输入未作为删除提交。发布的 notebook 清空执行计数和运行输出，保留代码、说明与图像附件。

## 验证

Python 使用 conda `ml`，设置 `PYTHONNOUSERSITE=1`。

在完整输入检出的集成工作区中运行：

```powershell
# 在 code/ 目录
python -X utf8 -m pytest tests -q
# 在项目根目录
python -X utf8 -m pytest tests -q
```

- `code/tests`：63 项通过，包含 CPU/GPU 小网格求解、膜极化和几何反演检查。
- 根目录 `tests`：149 项通过、9 项失败。
- 本次涉及的 Fiji HTML、Berea 网络 HTML、样品89运行和 GPU sweep 四组脚本测试：29 项通过。

9 项失败集中于 `test_build_segmented_core_pore_network_package.py`（3 项）和 `test_render_segmented_core_html.py`（6 项）：历史默认路径、旧版渲染接口及固定 vendor 可执行文件路径的假设与当前代码不一致。将集成前 `origin/main` 的源码和对应测试独立检出后，同样复现这 9 项失败（其余 14 项通过）。本次保留这些已存在问题，没有为通过旧测试而恢复旧渲染路线。

另完成新增及修改 Python 文件的语法检查、自有文件的提交空白检查、待发布文件大小与结果排除检查，以及本地输入原文件 SHA-256 校验。新纳入的 pnextract 上游源码保留其原有空白格式，没有为消除格式提示而改写第三方文件。

没有重新运行重型 full-grid 扫频，也没有重新编译 pnextract。小网格测试不能替代 full-grid 结果验证。
