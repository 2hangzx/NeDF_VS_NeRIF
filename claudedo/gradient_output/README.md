# Gradient Output — 重构版 NIR-BOS

> 正式实验包提示：请从包根目录使用
> `python experiment_control/experiment.py ...` 创建批次、训练和比较。
> 正式入口要求批次号，本文中直接运行 `main_BOS.py` 和旧 `result/...` 路径仅用于
> 理解底层工程；正式流程以包根目录 `README.md` 与 `docs/FULL_WORKFLOW.md` 为准。
> 训练中断时使用控制器的 `resume-baseline` / `resume-gradient`，它们调用原始
> `Trainer` 的完整 checkpoint 恢复逻辑。已完成路线后来需要增加总预算时，使用
> `extend-baseline` / `extend-gradient` 写入新子批次，完整步骤见正式包
> `docs/FULL_WORKFLOW.md`。

## 与原版的区别

| | 原版 (NIR-BOS) | 本版 (Gradient Output) |
|---|---|---|
| **网络输出** | σ(x,y,z) 标量折射率 | ∇σ(x,y,z) 三通道折射率梯度 |
| **梯度获取方式** | 后处理：有限差分 + autograd | 网络直接输出 |
| **每次采样查询网络次数** | 1 + 6（有限差分）+ 1（autograd）= 8 次 | 1 次 |
| **激活函数** | custom_tanh 约束值域 | trainable_tanh = scale×tanh |
| **loss 监督** | 投影域 MSE（同） | 投影域 MSE（同） |
| **编码方式** | Fourier（同） | Fourier（同） |

## 目录结构

```
gradient_output/
├── main_BOS.py          ← 训练入口（修改：去掉 valbound、适配新网络）
├── experiment_logging.py ← 记录一次运行的配置、源码版本和结果哈希
├── activation.py        ← 激活函数（新增：trainable_tanh）
├── encoding.py          ← 位置编码（不改）
├── loss.py              ← 损失函数（不改）
├── nerf/
│   ├── network.py       ← NeRF 网络（核心改动：1→3 输出）
│   ├── renderer.py      ← 体渲染（核心改动：删有限差分+autograd）
│   ├── provider.py      ← 数据加载（不改）
│   ├── utils.py         ← 训练器、工具（适配新导出逻辑）
│   ├── poisson_solver.py ← 梯度场到标量场的 FFT 泊松反演
│   ├── mask3D_calculate.py ← 3D 掩码（不改）
│   └── clip_utils.py    ← CLIP 工具（不改）
├── raymarching/         ← CUDA 光线行进（不改，从原版复制）
├── compare_reconstructions.py ← 统一 flow 标量空间比较
├── compare_gradients.py ← 统一三分量梯度空间比较
├── validate_common_grid.py ← 只读检查两版与 MATLAB GT 的公共评价网格
├── validate_experiment_logging.py ← 独立验证实验清单及篡改检测
├── validate_training_readiness.py ← 批次声明与两版硬编码训练前只读门禁
├── TRAINING_EXPERIMENTS.md ← 正式训练分组、参数与运行方式
├── README.md            ← 本文件
├── DESIGN.md            ← 设计文档
└── CHANGELOG.md         ← 修改日志
```

## 快速开始

```bash
# 1. 激活环境
conda activate huwei

# 2. 运行（首次会 JIT 编译 CUDA 扩展）
cd gradient_output
python main_BOS.py
```

## 依赖

与原版相同（`PYTHON/NIR-BOS/environment.yml`），无需额外安装。

## 数据

训练数据位于 `../../PYTHON/NIR-BOS/data/Phantom 1/140x294x140/`（与原版共享）。

## 公共评价网格

基线版与梯度版继续人工维护各自 `main_BOS.py` 中的硬编码参数，不由程序自动
改写。两版导出场统一采用 MATLAB 精确 GT 的体素中心网格：

```text
ROInum       = [140, 294, 140]
ROIvoxelsize = 0.01360525
ROIsize      = ROInum * ROIvoxelsize / 2
             = [0.9523675, 1.99997175, 0.9523675]
```

`ROIsize` 表示三个坐标轴的半尺寸。在原有 `torch.linspace` 导出公式下，上述
关系保证数学定义中的相邻采样点间距等于 `ROIvoxelsize`；实际 float32 坐标只
包含正常的浮点舍入。修改硬编码后可运行只读验证，不会改变任何配置：

```bash
cd claudedo/gradient_output
python validate_common_grid.py
```

验证同时检查两版坐标数组以及 MATLAB `flow_ground_truth.mat` 中的 `roi_size`、
`roi_num` 和 `spacing`。训练时的光线采样和基线有限差分公式
`delta_r = bound / max_steps` 均保持原样；公共评价网格只约束推理导出与比较空间。

## 实验配置与结果溯源

两版仍按原项目方式在各自 `main_BOS.py` 中人工修改硬编码参数并直接运行，没有
引入配置文件或新的启动命令。每次训练或测试会额外生成 JSON 实验清单：

```text
<workspace>/experiment_manifests/<run_id>.json   # 历次运行，按时间保留
<workspace>/experiment_manifest_latest.json      # 当前最近一次运行
<workspace>/results/experiment_manifest.json     # 与本次导出结果放在一起
```

清单记录解析后的全部参数、实际 `sys.argv`、训练预算、网络层数/宽度/可训练参数量、
Python/PyTorch/CUDA 环境、入口及核心源码 SHA-256、checkpoint 请求、恢复前后的
epoch/global step、训练耗时，以及 checkpoint 和 MAT 结果的大小与 SHA-256。
它只在原流程旁写日志，不参与 loss、采样、优化或场导出计算。

两类比较脚本会自动读取结果旁的 `experiment_manifest.json`，并在报告的
`experiment_manifests` 字段中保存来源。若清单记录的 MAT 哈希与当前文件不符，
程序会拒绝比较；本功能加入前的旧结果仍可使用，但会明确标记为
`missing_legacy_manifest`。可先运行独立自检：

```bash
cd claudedo/gradient_output
python validate_experiment_logging.py
```

正式提高网络参数时，通过批次控制器给两条路线分配互不覆盖的 workspace。用户可
任选一个模型训练，也可按任意顺序训练两路；checkpoint、结果、日志和实验清单会
按路线天然成组，旧实验也能完整保留。

## 严格控制组训练母版

第四阶段已把两版入口人工设置为 `strict_control_v1`：共同使用 3 层×128 的
Fourier MLP 主干、30000 iterations、5e-3 学习率、256 rays、256 max_steps、
seed 0 和 scratch checkpoint。两路写入各自的新 workspace，不会覆盖旧结果。
3×128 仅是该 profile 的声明，不是验证器的网络容量下限；例如 2×64 smoke profile
同样可以通过，只要两版入口、profile 与参数量声明一致。代码中的 `num_layers`
包含输出层，因而 2×64 表示一个 64 神经元隐藏层。

正式训练前必须运行：

```bash
cd claudedo/gradient_output
python validate_common_grid.py
python validate_training_readiness.py --batch-id strict_control_run_001
```

详细参数、有限差分步长说明、运行方式、比较命令及后续各自调优组边界见
[`TRAINING_EXPERIMENTS.md`](TRAINING_EXPERIMENTS.md)。控制器调用各自
`main_BOS.py`；验证器只读检查，不会自动修改训练参数硬编码。

## 与基线版的统一 flow 空间对比

`compare_reconstructions.py` 会从梯度版导出的三分量梯度重新进行带体素尺度的
泊松反演，并与基线版 `sigmas0.mat` 在同一空间比较：

```
flow = (n / n0 - 1) / flow_max
```

为避免 Python 重采样与 MATLAB `imresize3` 的实现差异，首次比较前在 MATLAB
中运行独立接口（不会修改或执行完整的 step3）：

```matlab
cd MATLAB
step1_InitBOSLAB
export_phantom1_flow_ground_truth
```

接口会生成 `MATLAB/Test_data/Phantom 1/140x294x140/flow_ground_truth.mat`，
其中 `flow_gt` 已经过与 step3 相同的缩放、中心裁剪和归一化。对比脚本默认
要求该精确 GT；可用 `--skip-ground-truth` 只比较两种预测。

它只消除梯度积分天然缺失的加性常数，不进行 min-max 缩放，因此输出的 RMSE、
MAE 和 NCC 可用于比较两种重建的幅值与结构。默认命令：

```bash
python compare_reconstructions.py
```

结果写入 `result/gradient_test/comparison_flow_exact/`，包括 JSON 指标、对齐后的场和
中间切片图；命令结束前还会弹出一个带 Z 轴滑块的四栏窗口，依次显示基线 σ、
梯度→泊松 σ、GT σ 与“梯度版 − 基线版”误差。服务器或批处理运行可加
`--no-show` 禁用窗口。旧的 `compare_sigma.py` 仍适合观察梯度版与真值的视觉形状，
但其 min-max 对齐结果不应用于定量结论。

## 与基线版的统一梯度空间对比

`compare_gradients.py` 一次运行同时比较以下四个
`(140, 294, 140, 3)` 向量场：

- `baseline_grid_fd`：从基线版公共网格 `sigmas0` 使用
  `numpy.gradient(edge_order=2)` 计算；
- `baseline_auto`：基线标量网络导出的连续 autograd 梯度；
- `gradient_output`：梯度网络直接输出；
- MATLAB 精确 `flow_gt` 使用同一体素间距和差分算子得到的 ground-truth 梯度。

正式体素场准确度以 `baseline_grid_fd` 为主，因为它与 GT 使用相同的公共网格
差分算子；`baseline_auto` 用于诊断连续网络梯度在有限分辨率采样后的表现。两种
基线始终同时计算。兼容参数 `--baseline-source`（新名称
`--primary-baseline-source`）只控制旧字段 `baseline_gradient` 指向哪一种，默认
为 `grid-finite-difference`，不会关闭另一种基线评价。

共同空间定义为：

```text
g = gradient[(n / n0 - 1) / flow_max]
```

梯度没有加性常数自由度，所以该脚本不进行 gauge 对齐、min-max、轴交换、
符号翻转或幅值拟合。首次运行前同样需要生成精确 MATLAB GT：

```matlab
cd MATLAB
step1_InitBOSLAB
export_phantom1_flow_ground_truth
```

默认比较命令：

```bash
cd claudedo/gradient_output
python compare_gradients.py --no-show
```

可先运行解析场自检：

```bash
python compare_gradients.py --self-test
```

报告的 `vs_ground_truth` 分别给出三种预测相对 GT 的指标，`pairwise` 则给出
autograd—grid-FD 以及梯度版—两种基线的内部差异。每组同时包含每个分量的
MAE/RMSE/NCC、向量 L2 误差、梯度幅值误差、
方向夹角、curl 和泊松投影残差。评价区域使用中心裁剪后的 3D mask，并默认
向内腐蚀 2 个体素；方向指标只在 GT 梯度显著的 active 区域计算，防止零背景
稀释误差。

结果默认写入 `result/gradient_test/comparison_gradient_exact/`：

```text
gradient_comparison_report.json
gradient_metrics.csv
gradient_fields.npz
components_midplane.png
magnitude_midplane.png
angular_error_midplane.png
integrability_midplane.png
```

`gradient_fields.npz` 明确保存 `baseline_grid_fd`、`baseline_auto`、
`gradient_output`、`ground_truth_gradient`、两个 mask 和 ground-truth flow；另保留
`baseline_gradient` 作为指向当前主基线的兼容别名。

旧 `compare_gradient.py` 仅显示梯度幅值，未读取基线梯度，也没有正式定量报告；
它只保留用于复现旧图，不应用于新的基线版—梯度版结论。
