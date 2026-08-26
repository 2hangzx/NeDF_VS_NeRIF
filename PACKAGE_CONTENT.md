# 正式实验包内容边界

## 已包含

- 完整 `MATLAB/`：BOSLAB 算法、工具、Phantom 1 数据、MEX 和精确 GT 接口；
- 完整 `Common/`：`step2_Compile.m` 实际引用的 CUDA/C++ 源码；
- 基线版 Python 代码、训练数据、环境文件和 raymarching CUDA 源码；
- 梯度版代码、Poisson 求解器、比较程序、验证器、文档和 CUDA 源码；
- 批次控制、中央保存路径、同批次中断恢复、已完成预算的跨批次延长辅助逻辑、
  部署手册和包校验和。

`TRANSFER_CHECKSUMS.sha256` 覆盖交付时的完整初始包；
`PACKAGE_CHECKSUMS.sha256` 只覆盖实验期间不应变化的源码与文档。这样 MATLAB 数据
重生成、MEX 重编译和 Python 数据同步不会破坏日常代码完整性检查。

## 有意排除

- 开发仓历史 `result/`、checkpoint、事件日志和大型比较 NPZ；
- `__pycache__`、`.pyc`、Python `build/` 和旧设备 `.pyd`；
- `claudedo/reconstruction_comparison_stages` 开发阶段快照；
- 根目录 `C++/` 独立工程。

独立 `C++/` 约 2.2 GiB，MATLAB→Python 正式流程没有引用它。MATLAB MEX 重编译
实际依赖的是体积约 0.5 MiB 的 `Common/CUDA/`，已包含。

## 数据策略

MATLAB 与 Python 各保留一份当前 Phantom 1 数据，使实验包在不重新生成数据时也
能直接训练。重新运行 MATLAB step3 后，使用 `experiment.py sync-data` 更新 Python
副本。精确 `flow_ground_truth.mat` 由比较程序直接从 MATLAB 数据目录读取。
