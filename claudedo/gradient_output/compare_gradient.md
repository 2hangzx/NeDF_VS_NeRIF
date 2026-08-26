# 折射率梯度比较工具说明

## 正式工具：`compare_gradients.py`

新的正式比较脚本在统一的归一化 flow 梯度空间中同时比较：

| 报告名称 | MAT 来源 | 含义 |
|---|---|---|
| `baseline_grid_fd` | 基线版 `sigmas0` | 公共网格上的 `numpy.gradient(edge_order=2)`，正式主基线 |
| `baseline_auto` | 基线版 `dsigmas_dxyz_auto0` | 标量网络对 NeRF 坐标的连续 autograd 梯度，诊断项 |
| `gradient_output` | 梯度版 `dsigmas_dxyz_auto0` | 梯度网络直接预测的三分量向量场 |
| Ground truth | `flow_gt` | MATLAB 精确预处理后，由 Python 统一差分得到的梯度 |

共同空间：

```text
flow = (n / n0 - 1) / flow_max
g = gradient(flow), spacing = (0.01360525, 0.01360525, 0.01360525)
```

公共体素中心网格固定为 `ROInum = [140, 294, 140]`、
`ROIsize = [0.9523675, 1.99997175, 0.9523675]`。两版参数由人工硬编码，比较前
运行 `python validate_common_grid.py`，只检查两版坐标和 MATLAB GT 是否一致，
不会自动修改训练参数。

梯度比较不做常数对齐、min-max、尺度拟合、轴交换或符号翻转。

两种基线在每次运行中始终同时评价。默认 `baseline_grid_fd` 还会写入历史兼容字段
`baseline_gradient`；`--primary-baseline-source exported` 可把兼容字段切换到
autograd，但不会改变正式 `pairwise`、`vs_ground_truth` 或图形中的双基线内容。

### 准备精确 GT

```matlab
cd MATLAB
step1_InitBOSLAB
export_phantom1_flow_ground_truth
```

### 运行

```bash
cd claudedo/gradient_output
conda activate huwei
python compare_gradients.py --self-test
python compare_gradients.py --no-show
```

如只希望快速验证指标和绘图链路，可跳过泊松诊断和大体积 NPZ：

```bash
python compare_gradients.py --no-show \
  --skip-poisson-diagnostics --skip-field-save
```

正式报告包括三分量、向量整体、幅值、方向角和可积性五类指标：

- `vs_ground_truth`：`baseline_grid_fd`、`baseline_auto` 和
  `gradient_output` 分别相对 GT；
- `pairwise`：autograd 相对 grid-FD、梯度版分别相对两种基线；
- `integrability`：三种预测和 GT 统一计算 curl 与泊松投影残差。

默认输出目录为 `result/gradient_test/comparison_gradient_exact/`。

## 实验来源校验

新训练/测试完成后，结果目录会同时保存 `experiment_manifest.json`。正式脚本会在
读取两路 MAT 时自动加载各自清单，核对清单中的结果 SHA-256，并将完整来源写入
报告的 `experiment_manifests` 字段。文件被替换或移动后内容发生变化时会立即
中止，防止参数与结果错配。

自动留痕功能加入前生成的 MAT 没有清单，仍可用于代码回归；报告会将其状态明确
标为 `missing_legacy_manifest`。这类旧数据不应被当作提高网络参数后的正式实验。

```bash
python validate_experiment_logging.py
```

该命令只在临时目录中验证清单字段、哈希关联和篡改检测，不执行训练。

## 历史工具：`compare_gradient.py`

旧脚本只显示梯度版与 GT 的 `|gradient|`，存在以下限制：

- 不读取基线版梯度；
- 丢失三分量符号和方向；
- 没有 3D mask、active-gradient 区域和正式指标；
- 缺失梯度变量时错误地回退到标量 `sigmas0`。

因此它只用于复现旧图，不应继续用于基线版与梯度版的定量比较。
