# 新设备部署指南

## 1. 建议平台

当前工程已在 Windows、Python 3.10、PyTorch 2.1 + CUDA 11.8 路线上验证。正式
训练需要 NVIDIA CUDA GPU。换设备时首先确保：

- NVIDIA 驱动可正常识别 GPU；
- Miniconda/Anaconda 可用；
- Visual Studio 2022 Build Tools 已安装“使用 C++ 的桌面开发”；
- CUDA Toolkit 与 PyTorch、MATLAB 的支持范围协调；
- MATLAB 已安装 Image Processing Toolbox（`imresize3`）以及编译 CUDA MEX
  所需组件。

MATLAB 与 PyTorch 对 CUDA/编译器的兼容范围不一定完全相同。优先选择两者都支持
的组合，不要直接照搬旧设备生成的 Python `.pyd`。本包已排除旧 `.pyd`，新设备
会重新编译。

## 2. 搬运后校验

复制完成后、运行 MATLAB 或编译任何扩展之前，先执行一次完整搬运校验：

```powershell
python experiment_control/verify_transfer.py
```

它覆盖初始代码、数据、MEX 和文档。若出现缺失或哈希不一致，应重新复制实验包，
不能直接开始训练。

完整搬运校验通过后，再运行日常结构验证：

```powershell
python experiment_control/validate_package.py
```

日常验证使用只覆盖不可变源码和文档的 `PACKAGE_CHECKSUMS.sha256`，因此以后重新
生成 MATLAB 数据、同步 Python 数据或在新设备编译 MEX/Python CUDA 扩展后仍可
继续使用。目标设备生成的 `.pyd`、`build/` 等会列入报告，但不会导致日常验证
失败；交付前才使用 `--require-portable-clean` 强制要求源码树无本机编译产物。
`verify_transfer.py` 则只要求在收到原始包时执行一次。

## 3. 创建 Python 环境

```powershell
cd PYTHON/NIR-BOS
conda env create -f environment.yml
conda activate huwei
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

预期 `torch.cuda.is_available()` 为 `True`。`environment.yml` 是当前已验证环境的
完整导出；若新设备上的 Conda 因精确构建号无法求解，应保留 Python 3.10、
PyTorch 2.1/CUDA 11.8 和 pip 依赖版本，针对目标平台去掉无法解析的构建号，而不
应静默升级所有科学计算库。

## 4. 编译 Python CUDA 扩展

可让第一次训练触发 JIT，也可以提前编译以尽早发现编译器问题：

```powershell
cd PYTHON/NIR-BOS/raymarching
python setup.py build_ext --inplace

cd ../../../claudedo/gradient_output/raymarching
python setup.py build_ext --inplace
```

两个路线拥有各自的扩展目录，需要分别确认。编译生成的 `.pyd` 属于目标设备，
不需要回写到开发仓或再次跨设备搬运。

## 5. MATLAB 初始化

从 MATLAB 中把工作目录切换到本包的 `MATLAB/`：

```matlab
step1_InitBOSLAB
```

如果随包携带的 `Mex_files/win64/*.mexw64` 与新设备兼容，可直接使用。否则按本机
MATLAB、CUDA 和 Visual Studio 组合运行：

```matlab
step2_Compile
```

`step2_Compile.m` 依赖包根目录下的 `Common/CUDA/`，因此不能只单独复制
`MATLAB/` 子目录。

## 6. 环境验收

返回包根目录，创建一个尚未训练的批次并执行门禁：

```powershell
python experiment_control/experiment.py create --batch-id device_acceptance_001
python experiment_control/experiment.py preflight --batch-id device_acceptance_001
```

门禁通过说明包结构、公共网格、严格控制组参数、日志、梯度数学自检和批次路径
均正常。该批次若只用于验收，不要继续训练；正式实验另建新批次号。
