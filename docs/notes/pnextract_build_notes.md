# pnextract 本机构建记录

日期：2026-05-20

## 结论

阶段 1 的构建流程已整理完成。当前本地仓库位于 `C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction`；由于 `pnextract/build/` 和 `pnextract/bin/pnextract*` 是可再生成的本地构建产物，当前克隆不包含旧机器上的二进制文件。如需在本机运行，请先用 Git Bash、MSYS2 或 WSL 重新执行本机构建脚本，生成 `pnextract/bin/pnextract.exe`。

可执行文件：

- `pnextract/bin/pnextract.exe`（Windows Git Bash/MSYS2/Cygwin）
- `pnextract/build/local/pnextract.exe`（Windows Git Bash/MSYS2/Cygwin）
- `pnextract/bin/pnextract`（Linux/macOS/WSL）
- `pnextract/build/local/pnextract`（Linux/macOS/WSL）

本机构建脚本：

- `scripts/build_pnextract_local.sh`

## 背景

仓库默认 `make -j` 构建没有直接通过，原因是默认构建链会先构建 bundled `zlib`/`libtiff`，并使用偏 Linux 静态链接的 toolchain。

已观察到的问题：

- 现代 CMake 对 bundled `zlib` 的旧 `cmake_minimum_required` 策略报错。
- 加入 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 后，默认 toolchain 仍强制 Linux/static，在非目标平台上链接测试可能失败，历史错误涉及 `crt0.o` 缺失。

本阶段采用本机 `g++` 直接编译 `pnextract` 核心源码，禁用 zlib/tiff/OpenMP 相关宏。因此重新生成的本地二进制支持未压缩 `.raw`/`.mhd` 输入，适合读取本项目的 `microCT_Berea.raw`。

## 构建命令

推荐使用脚本：

```bash
scripts/build_pnextract_local.sh
```

脚本核心命令：

```bash
g++ -std=c++17 -O2 -Wall -pedantic \
  -DRELEASE_DATE='"2026.05.20-local"' \
  -D_FILE_OFFSET_BITS=64 \
  -Ipnextract/src/include \
  -Ipnextract/src/libvoxel \
  -Ipnextract/src/pnm/pnextract \
  pnextract/src/pnm/pnextract/blockNet.cpp \
  pnextract/src/pnm/pnextract/nextract.cpp \
  pnextract/src/pnm/pnextract/medialSurf.cpp \
  pnextract/src/pnm/pnextract/writers_vtk.cpp \
  pnextract/src/pnm/pnextract/writers_vxl.cpp \
  pnextract/src/libvoxel/voxelImage.cpp \
  -o pnextract/build/local/pnextract.exe
```

在 Windows Git Bash/MSYS2/Cygwin 下，构建后脚本会复制：

```bash
cp pnextract/build/local/pnextract.exe pnextract/bin/pnextract.exe
chmod +x pnextract/bin/pnextract.exe
```

在 Linux/macOS/WSL 下输出文件不带 `.exe` 后缀。

## 编译警告

编译完成，但每个翻译单元都会出现两个历史代码警告：

- `src/include/typses.h:701`：`&` 和 `==` 的优先级警告。
- `src/libvoxel/voxelImageI.h:1772`：局部变量 `count` set but not used。

本阶段未修改上游源码。上述警告未阻止链接，也未影响烟雾测试。

## Usage 验证

命令：

```bash
pnextract/bin/pnextract.exe -h
```

结果：

```text
Pore Network Extraction: pnextract version 2026.05.20-local
Usage:
  pnextract vxlImage.mhd    #  extract network
  pnextract -g vxlImage.mhd # -generate vxlImage.mhd
```

## 烟雾测试

测试目录：

- `pnextract/build/local_smoke/`

测试输入：

- `smoke.raw`：人工生成的 `30^3`、`uint8` 二值图像。
- `smoke.mhd`：指向 `smoke.raw` 的 MHD 头文件。

运行命令：

```bash
cd pnextract/build/local_smoke
../../bin/pnextract.exe smoke.mhd > pnextract_smoke.log 2>&1
```

成功输出：

- `smoke_link1.dat`
- `smoke_link2.dat`
- `smoke_node1.dat`
- `smoke_node2.dat`
- `smoke_VElems.mhd`
- `smoke_VElems.raw`
- `smoke_pores.vtu`
- `smoke_throats.vtu`
- `smoke_throatsBalls.vtu`

日志结尾显示：

```text
smoke
***  4-2 pores, 1 throats,   ratio: -0.5  ***
end
```

说明核心入口、图像读取、孔网提取和网络文件写出均已跑通。这个人工图像很小，网络拓扑只用于 smoke test，不用于物理验证。

## 当前限制

- 当前本机二进制没有启用 zlib，因此不支持 `.raw.gz`。
- 当前本机二进制没有启用 libtiff，因此不支持直接读取 `.tif`。
- 当前本机二进制没有启用 OpenMP，完整 `350^3` Berea 运行可能较慢。

这些限制不阻塞下一阶段，因为本项目的 Berea 数据是未压缩 `.raw`。

