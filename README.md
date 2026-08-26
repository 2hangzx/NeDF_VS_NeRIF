# NIR-BOS 正式实验包

这是从开发仓整理出的独立、可搬运实验目录。将整个目录复制到新设备后，可完成：

```text
MATLAB 几何/投影数据与精确 GT
            ↓
数据同步到 Python
            ↓
按需训练基线版和/或梯度版（支持断点续训与跨批次延长预算）
            ↓
两路均完成时可选：折射率标量比较 + 三路折射率梯度比较
```

## 目录

```text
formal_experiment_package/
├── MATLAB/                    BOSLAB、测试数据、MEX 与 GT 导出接口
├── Common/                    MATLAB CUDA/MEX 编译依赖源码
├── PYTHON/NIR-BOS/            基线版代码、环境文件与训练数据
├── claudedo/gradient_output/  梯度版、比较程序与验证程序
├── experiment_control/        批次创建、训练、比较和完整性检查入口
├── experiments/               所有正式实验批次的集中保存位置
├── docs/                      新设备部署和全流程说明
├── PACKAGE_CHECKSUMS.sha256   实验期间不可变源码的日常校验
└── TRANSFER_CHECKSUMS.sha256  新设备收到原始包后的一次性完整校验
```

历史训练结果、checkpoint、缓存、Python 编译产物和 2.2 GiB 的独立 `C++/`
工程没有装入。MATLAB 实际编译所需的 `Common/CUDA` 已完整保留。

## 完整对比实验的最短入口

新设备第一次使用先阅读：

1. [`docs/NEW_DEVICE_SETUP.md`](docs/NEW_DEVICE_SETUP.md)
2. [`docs/FULL_WORKFLOW.md`](docs/FULL_WORKFLOW.md)
3. [`docs/BATCH_MANAGEMENT.md`](docs/BATCH_MANAGEMENT.md)

完成环境配置后，所有批次操作统一从包根目录执行：

```powershell
python experiment_control/experiment.py create --batch-id strict_control_run_001
python experiment_control/experiment.py preflight --batch-id strict_control_run_001
python experiment_control/experiment.py train-baseline --batch-id strict_control_run_001
python experiment_control/experiment.py train-gradient --batch-id strict_control_run_001
python experiment_control/experiment.py compare --batch-id strict_control_run_001
python experiment_control/experiment.py status --batch-id strict_control_run_001
```

训练中断后，从同批次同路线最近的完整 epoch checkpoint 恢复：

```powershell
python experiment_control/experiment.py resume-baseline --batch-id strict_control_run_001
python experiment_control/experiment.py resume-gradient --batch-id strict_control_run_001
```

若一条路线已经按原计划完整结束，后来才决定提高总迭代数，应创建声明新总预算的
子批次，并从父批次延长；不要用 `resume-*` 改写已完成批次：

```powershell
python experiment_control/experiment.py extend-baseline `
  --from-batch strict_control_10k_001 `
  --batch-id strict_control_20k_from_10k_001

python experiment_control/experiment.py extend-gradient `
  --from-batch strict_control_10k_001 `
  --batch-id strict_control_20k_from_10k_001
```

子批次必须事先按新的硬编码总预算创建并通过 `preflight`。完整的 10000→20000
操作顺序和学习率调度含义见 [`docs/FULL_WORKFLOW.md`](docs/FULL_WORKFLOW.md)。

训练参数仍在两版 `main_BOS.py` 中硬编码。控制程序只负责批次 ID、集中保存路径、
路线隔离和状态，不会自动改写网络或训练参数。基线版与梯度版可以任选其一、任意
顺序训练；只有执行 `compare` 时才要求同一批次的两路结果都已完整生成。

## 重要安全规则

- 每次实验使用新批次号；已存在批次不会被覆盖。
- 训练入口要求控制程序注入批次号，未指定批次时会主动退出。
- 基线版与梯度版分别保存到同一批次的不同子目录。
- `train-*` 只创建 scratch 路线；`resume-*` 只恢复已有路线 checkpoint。
- `extend-*` 只把已完成父批次延长到全新子批次，不修改父批次。
- 只训练一个模型时，完成对应训练命令后即可停止或归档，不需要运行比较。
- 比较前会核验两路 `experiment_manifest.json` 和 MAT SHA-256。
- 不要把失败或 `status != completed` 的运行用于正式结论。
