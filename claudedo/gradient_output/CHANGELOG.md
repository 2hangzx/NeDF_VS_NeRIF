# 修改日志

> 父文档：`DESIGN.md`
> 原版基准：`PYTHON/NIR-BOS/` 中对应文件

---

## 2026-08-26：解除固定网络容量下限

- 删除训练前验证器中的 `num_layers >= 3`、`hidden_dim >= 128` 固定门禁；
- 网络结构现在完全以批次 `declared_profile.json` 为权威，入口硬编码与 profile
  仍须逐项一致；
- 保留按结构计算并核对两路线参数量的检查，因此不会放宽未声明或填错结构；
- `strict_control_v1` 仍由其 profile 强制为 3×128，`smoke_test_v1` 可合法声明
  原始实现语义下的 2×64（一个 64 神经元隐藏层）；
- 修正 smoke profile 的预期参数量：baseline 2561、gradient_output 2694。

---

## 2026-08-25：已完成训练预算的受控跨批次延长

- 新增 `extend-baseline`、`extend-gradient`：从已完成父批次的同路线完整
  checkpoint 启动全新子批次，父批次保持不可变；
- 保留模型、optimizer、AMP scaler 与 EMA；按子批次硬编码学习率重置 optimizer
  当前学习率，并在剩余总预算上重建 `CosineAnnealingLR`；
- 新增父 checkpoint 清单登记/SHA-256、网络容量与核心源码兼容性门禁；
- 子批次实验清单记录父/子批次关系、原/目标/剩余预算和 scheduler 策略；
- 训练前验证器改为读取批次 `declared_profile.json`，训练参数仍由用户在两版入口
  人工硬编码，不引入新的配置式运行方式；
- 更新正式包 README、全流程、批次规则与封装验收记录，并完成 CPU 调度器和控制器
  回归测试。

---

## 2026-08-25：正式实验包训练路线解耦

- 移除“基线版完成后才能启动梯度版”的控制器限制；
- 基线版与梯度版现在可以任选其一、任意顺序训练；
- `compare` 仍要求同一批次两路结果完整，并继续核验实验清单和结果哈希；
- 同步更新正式包 README、全流程、批次管理、训练母版和封装验收记录。

---

## 2026-08-25：正式接入原始 checkpoint 断点续训

- 保留两版 `Trainer` 每个 epoch 的完整 checkpoint 保存格式和最近两个滚动文件；
- 新增 `resume-baseline`、`resume-gradient`，仅恢复同批次同路线完整 checkpoint；
- 新训练仍固定 scratch，恢复时显式注入 checkpoint，不改变网络和训练参数硬编码；
- 批次清单记录恢复文件路径、大小、SHA-256、恢复前状态和历史次数；
- 已完成路线拒绝追加训练，断电遗留 `running` 状态需要人工确认后显式放行。

---

## 已修改文件

| 文件 | 状态 | 说明 |
|------|------|------|
| `activation.py` | ✅ 已修改 | 新增 `TrainableTanh(nn.Module)` — `scale × tanh(x)`，scale 为可学习参数 |
| `nerf/network.py` | ✅ 已修改 | 输出 1→3；`custom_tanh`→`TrainableTanh`；`density()` 返回梯度 |
| `nerf/renderer.py` | ✅ 已修改 | 删有限差分循环+autograd；训练/推理模式直接用网络输出 |
| `nerf/utils.py` | ✅ 已修改 | `train_step` 去掉 `depth_auto` 分支；loss 简化为 `loss.mean()` |
| `main_BOS.py` | ✅ 已修改 | 数据路径适配新目录；workspace 改为 `result/gradient_test` |
| `compare_reconstructions.py` | ✅ 已新增 | 在统一 flow 标量空间比较基线版、梯度版与精确 GT |
| `compare_gradients.py` | ✅ 已新增 | 在统一三分量梯度空间比较两版与精确 GT |
| `validate_common_grid.py` | ✅ 已新增 | 只读验证两版硬编码导出网格及 MATLAB GT 元数据 |
| `experiment_logging.py`（两版） | ✅ 已新增 | 自动记录运行配置、模型规模、源码与结果哈希 |
| `validate_experiment_logging.py` | ✅ 已新增 | 独立验证实验清单写入、关联和篡改拒绝 |
| `validate_training_readiness.py` | ✅ 已新增 | 只读检查严格控制组训练配置和新 workspace |
| `TRAINING_EXPERIMENTS.md` | ✅ 已新增 | 固化实验分组、训练母版、运行方式和调优边界 |
| `MATLAB/.../export_phantom1_flow_ground_truth.m` | ✅ 已新增 | 独立导出 MATLAB 精确预处理后的 `flow_gt` |
| `README.md` / `compare_gradient.md` | ✅ 已更新 | 记录正式比较流程、指标、命令与旧工具限制 |

---

## 修改 1：activation.py — 新增 TrainableTanh

**原始源码**（文件末尾）：
```python
# 创建一个方便调用的 tanh 函数
custom_tanh = _tanh.apply
```

**改为**：
```python
custom_tanh = _tanh.apply


class TrainableTanh(nn.Module):
    """g = scale * tanh(x),  scale 为可学习参数"""
    def __init__(self, out_features, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((out_features,), init_scale))

    def forward(self, x):
        return self.scale * torch.tanh(x)
```

---

## 修改 2：nerf/network.py — 输出类型 1→3

**原始源码**（关键段）：

```python
class NeRFNetwork(NeRFRenderer):
    def __init__(self, ..., valbound = [-1.0, 0.0], ...):
        ...
        sigma_net = []
        for l in range(num_layers):
            ...
            if l == num_layers - 1:
                out_dim = 1       # 标量折射率
            ...
            if l == num_layers - 1:
                b = (valbound[0] + valbound[1]) / 2
                k = valbound[1] - b
                target_bias = math.atanh(-b / k)
                nn.init.constant_(layer.bias, target_bias)
        self.sigma_net = nn.ModuleList(sigma_net)
        self.valbound = valbound

    def forward(self, x, d):
        mask3D = self.mask3Ddata.maskinterp(x)
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
        sigma = custom_tanh(h[..., 0], self.valbound)
        return sigma * mask3D

    def density(self, x):
        ...
        return {'sigma': sigma * mask3D}
```

**改为**（关键段）：

```python
from activation import TrainableTanh

class NeRFNetwork(NeRFRenderer):
    def __init__(self, ..., valbound = [-1.0, 0.0], ...):  # valbound 保留但不用
        ...
        grad_net = []
        for l in range(num_layers):
            ...
            if l == num_layers - 1:
                out_dim = 3         # ∇σ_x, ∇σ_y, ∇σ_z
            ...
            if l == num_layers - 1:
                nn.init.constant_(layer.bias, 0.0)  # 梯度均值 0
        self.grad_net = nn.ModuleList(grad_net)
        self.grad_activation = TrainableTanh(out_features=3, init_scale=1.0)

    def forward(self, x, d):
        mask3D = self.mask3Ddata.maskinterp(x)              # [N]
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.grad_net[l](h)
        grad = self.grad_activation(h)                       # [N, 3]
        return grad * mask3D.unsqueeze(-1)                   # [N, 3]

    def density(self, x):
        return self.forward(x, x)                            # 直接调用 forward
```

---

## 修改 3：nerf/renderer.py — 删有限差分 + autograd

**原始源码**（run_cuda 训练模式，行 280-368）：

```python
sigmas = self(xyzs, dirs)
sigmas = self.density_scale * sigmas

# 有限差分 —— 3 个方向，每个方向额外查 2 次网络
delta_r = self.bound * 2 / max_steps / 2
dsigmas_dxyz = torch.zeros_like(xyzs)
for dim in range(3):
    ...  # 行 292-328，约 40 行

# autograd —— 额外构建计算图
dsigmas_dxyz_auto = torch.autograd.grad(
    outputs=sigmas, inputs=xyzs, ...)

# 两次 composite
weights_sum, depth, image = raymarching.composite_rays_train(dsigmas_dxyz, rgbs, ...)
weights_sum, depth_auto, image_auto = raymarching.composite_rays_train(dsigmas_dxyz_auto, rgbs, ...)
```

**改为**：

```python
gradient = self(xyzs, dirs)                          # [M, 3] 直接出梯度
gradient = self.density_scale * gradient

weights_sum, depth, image = raymarching.composite_rays_train(
    gradient, gradient, deltas, rays, T_thresh)
```

推理模式同样简化——不再需要 `requires_grad_(True)` 和 `autograd.grad`，直接用 `self()` 查 ∇σ。

---

## 修改 4：nerf/utils.py — 去掉 depth_auto 分支

**原始源码**（train_step）：

```python
pred_depth = outputs['depth']
pred_depth_auto = outputs['depth_auto']

loss = self.criterion(pred_depth, gt_rgb).mean(-1)
loss_auto = self.criterion(pred_depth_auto, gt_rgb).mean(-1)
...
loss = 1*loss.mean() + 0*loss_auto.mean()
```

**改为**：

```python
pred_depth = outputs['depth']

loss = self.criterion(pred_depth, gt_rgb).mean(-1)
...
loss = loss.mean()
```

---

## 运行时修复：`density()` 返回值类型变更

修改 2 将 `density()` 从返回 `{'sigma': tensor}` 改为直接返回 `tensor`。以下调用方需要同步修改：

| 文件 | 位置 | 修改 |
|------|------|------|
| `nerf/renderer.py` | `update_extra_state()` (2 处) | `.density(x)['sigma']` → `.density(x).norm(dim=-1)` |
| `nerf/utils.py` | `save_mesh() → query_func()` | `.density(x)['sigma']` → `.density(x).norm(dim=-1)` |

用 `.norm(dim=-1)` 取梯度幅值 |∇σ| 作为密度网格更新的标量信号，mesh 提取同理。

---

## 新增：nerf/poisson_solver.py — FFT 泊松求解器

**用途**：从网络输出的 ∇σ 重建 σ 标量场，用于 mesh 提取和与原版对比。

方程：∇²σ = div(∇σ)，边界条件 Neumann (∂σ/∂n = 0)

方法：3D FFT — `σ = IFFT[-FFT[div(g)] / |k|²]`，DC 分量置零。当前实现
保留训练时的 flow 数值幅值，不再默认归一化到 `[-1, 1]`。

接入点：`renderer.py` 推理模式 — 查询 ∇σ 后调用 `poisson_reconstruct(gradient0)` 生成 `sigmas0`。

---

## 修改 6：调参 — scale + 训练超参数

| 参数 | 改前 | 改后 | 原因 |
|------|------|------|------|
| `init_scale` | 1.0 | 50.0 | 真实 |∇σ| ~85，差两个数量级 |
| `--iters` | 50 | 1000 | 足够步数学出结构 |
| `--lr` | 2e-2 | 5e-3 | 梯度信号弱，小步更稳定 |
| `--num_rays` | 64 | 128 | 增加采样加速收敛 |

---

## 修改 5：main_BOS.py — 数据路径

```diff
- 'data/Phantom 1/140x294x140'
+ '../../PYTHON/NIR-BOS/data/Phantom 1/140x294x140'

- 'result/phantom 1/test'
+ 'result/gradient_test'
```

---

## 修改 7：统一 flow 空间的基线—梯度版重建比较（2026-08-18 16:27:06）

**修改文件**：

| 文件 | 修改 |
|------|------|
| `nerf/poisson_solver.py` | 新增 `spacing=(dx,dy,dz)` 与 `normalize` 参数；默认保留幅值。 |
| `nerf/renderer.py` | 导出时传入 `ROIvoxelsize`，使梯度到 σ 的反演使用 NeRF 坐标尺度。 |
| `compare_reconstructions.py` | 新增可复现实验脚本，统一比较基线 σ、泊松 σ 与 GT flow。 |

共同代数空间严格定义为 MATLAB `step3` 中的：

```text
flow = (n / n0 - 1) / flow_max
```

梯度场只能确定 σ 的相对值，缺少一个加性常数。比较脚本因此只在共同的
ROI 边界参考区移除该常数，**不会对任一重建做 min-max 或幅值缩放**。它同时
输出 MAE/RMSE/NCC、`curl(∇σ)` 及“预测梯度—泊松 σ 梯度”残差，用于区分
投影拟合和三维可积性。

`compare_reconstructions.py` 还提供带 Z 轴滑块的四栏窗口：基线 σ、
梯度→泊松 σ、GT σ、梯度版减基线版的误差。传入 `--no-show` 时保留批处理模式。

---

## 修改 8：精确 MATLAB GT 接口（2026-08-22）

- 新增 `MATLAB/Utilities/Demos/export_phantom1_flow_ground_truth.m`，独立复现
  step3 中 `n` 的 `imresize3(0.5)`、中心裁剪和 flow 归一化，只导出 Python
  对比所需的 `flow_ground_truth.mat`，不修改也不执行原始 step3。
- `compare_reconstructions.py` 默认读取 MATLAB 导出的 `flow_gt`，不再静默使用
  SciPy 插值近似 GT。
- `--skip-ground-truth` 可只比较基线版和梯度版；旧的原始 `n` 路径仅在显式
  指定 `--allow-legacy-ground-truth` 时启用，并输出警告。

---

## 修改 9：统一折射率梯度空间比较（2026-08-22）

**新增文件**：`compare_gradients.py`

比较对象统一为缩放 NeRF 坐标中的：

```text
g = gradient[(n / n0 - 1) / flow_max]
```

- 基线版默认读取 `dsigmas_dxyz_auto0`，即标量网络的连续 autograd 梯度；
- 梯度版读取同名变量，但其语义是网络直接预测的三分量梯度；
- GT 优先读取 `export_phantom1_flow_ground_truth.m` 导出的精确 `flow_gt`，再用
  与体素间距一致的 `numpy.gradient(edge_order=2)` 计算三分量梯度。

脚本明确禁止 gauge 对齐、min-max、幅值拟合、轴交换和符号翻转。正式指标包括：

1. `gx/gy/gz` 分量 MAE、RMSE、bias、NCC；
2. 向量 MAE-L2、RMSE-L2、relative-L2；
3. 梯度幅值误差；
4. active-gradient 区域内的余弦相似度、平均/中位/P95 夹角及角度命中率；
5. 三者统一计算的 curl 与泊松投影残差，用于衡量离散可积性。

评价 mask 默认向内腐蚀 2 个体素，排除硬 3D mask 和单边差分边界伪影；
active 区域默认阈值为 `0.05 × P99(|g_gt|)`，避免零背景稀释方向误差。

新增输出：JSON、扁平 CSV、压缩 NPZ、三分量切片、幅值切片、角度误差图和
curl 切片图。`--self-test` 使用解析二次势场、无旋场、旋转场及常数平移测试
差分与指标实现。

旧 `compare_gradient.py` 标记为历史幅值可视化工具，不再用于正式结论。

---

## 修改 10：公共评价网格精确统一（2026-08-24）

第一阶段遵循“最小改动、保留原项目运行方式”：两版训练参数继续在各自
`main_BOS.py` 中人工硬编码，不引入配置系统，也不自动改写网格。

原硬编码为：

```text
ROIsize = [0.9524, 2.0, 0.95274]
ROInum = [140, 294, 140]
ROIvoxelsize = 0.01360525
```

导出器使用从 `-ROIsize + ROIvoxelsize/2` 到
`+ROIsize - ROIvoxelsize/2` 的 `torch.linspace`。原 `ROIsize` 与另外两个参数
不满足 `2 * ROIsize = ROInum * ROIvoxelsize`，导致理论网格间距与声明间距略有差异。
两版现人工改为：

```text
ROIsize = [0.9523675, 1.99997175, 0.9523675]
```

新增 `validate_common_grid.py`，从两版源码 AST 中读取硬编码 `sys.argv`，重建
实际三个坐标轴，并与 MATLAB 精确 GT 的 `roi_size`、`roi_num`、`spacing`
逐项核对。该工具仅验证，不修改训练或评价参数。

项目环境中另用真实 `torch.linspace(dtype=float32)` 复算：两版坐标逐元素一致，
三个轴的平均步长均为 `0.013605250045657158`；最大单步舍入误差在 x/z 轴为
`4.66e-08`、y 轴为 `1.32e-07`。

本阶段没有修改 renderer、网络结构、训练预算、光线采样或基线训练有限差分
`delta_r = bound / max_steps`。

---

## 修改 11：双基线梯度同时比较（2026-08-24）

`compare_gradients.py` 不再在基线 autograd 与网格有限差分之间二选一。每次运行
固定生成并评价：

```text
baseline_grid_fd
baseline_auto
gradient_output
ground_truth_gradient（存在精确 GT 时）
```

- `baseline_grid_fd` 从基线 `sigmas0` 使用公共 spacing 和
  `numpy.gradient(edge_order=2)` 得到，作为正式体素场主基线；
- `baseline_auto` 读取基线 `dsigmas_dxyz_auto0`，作为连续模型诊断；
- `--primary-baseline-source` 及旧别名 `--baseline-source` 只选择历史兼容字段
  `baseline_gradient`，默认改为 `grid-finite-difference`，不会停用另一种基线。

JSON/CSV 新增结构化分组：

- `vs_ground_truth`：三种预测分别对 GT；
- `pairwise`：autograd 对 grid-FD、梯度版分别对两种基线；
- `integrability`：三种预测与 GT 的 curl 和泊松投影残差。

NPZ 新增 `baseline_grid_fd` 与 `baseline_auto`，并保留 `baseline_gradient` 兼容别名。
分量、幅值、角度、curl 切片图及交互查看器均同时显示两种基线。

同时修复 Windows GBK 控制台执行 `compare_gradients.py --help` 时，argparse 因
说明文本中的数学符号无法编码而退出的问题；模块文档仍保留完整数学记号。

本阶段只修改比较工具和说明文档，没有修改训练、推理导出、网络或 ground truth。

---

## 修改 12：实验配置、源码与结果自动留痕（2026-08-24）

为避免后续提高两版网络参数后无法确认“某个 MAT 究竟由哪组参数和源码生成”，
在两版入口中以旁路方式接入内容一致的 `experiment_logging.py`。原硬编码参数、
启动命令、训练和导出调用顺序均保留。

每次运行会在 workspace 保存带 `run_id` 的历史清单和 latest 清单；导出完成后，
还会把同一清单写入 `results/experiment_manifest.json`。记录范围包括：

1. 全部解析参数、实际硬编码 argv 和名义训练预算；
2. 网络类型、层数、宽度、可训练参数量与模型结构文本；
3. Python、PyTorch、CUDA、设备信息和核心源码 SHA-256；
4. checkpoint 请求、恢复前后 epoch/global step、训练耗时；
5. checkpoint 与 MAT 结果文件清单及 SHA-256。

`compare_reconstructions.py` 和 `compare_gradients.py` 会把两路清单写入比较报告；
有清单时验证输入 MAT 哈希，不一致立即报错。旧结果没有清单时继续兼容，但报告
明确写入 `missing_legacy_manifest`，不会把其参数来源误当成当前源码。

新增 `validate_experiment_logging.py`，在临时目录验证两版记录器一致、13 项字段
完整、结果关联成功以及篡改后的 MAT 被拒绝。该测试不启动训练，不写正式 workspace。

---

## 修改 13：严格控制组高容量训练母版（2026-08-24）

按 `讨论1.md` 的人工统一原则，在两版 `main_BOS.py` 中建立
`strict_control_v1`。两版共同配置为：

```text
Fourier MLP trunk = 3 layers × 128 hidden units
iters = 30000, lr = 5e-3
num_rays = 256, max_steps = 256
seed = 0, bound = 2, checkpoint = scratch
```

网络容量由原 2×64 提高到 3×128。按当前网络层定义，基线版预计 21,505 个
可训练参数，梯度版预计 21,766 个；差异来自不可避免的 1 通道/3 通道输出头和
梯度版的 3 个 trainable scale，隐藏主干完全一致。

两版 workspace 改为新的、互不覆盖的 route-specific 目录。新增只读工具
`validate_training_readiness.py`，从两个入口的 AST 读取硬编码，检查相同物理数据、
网络主干、训练预算、采样、优化器、调度器、公共网格、scratch 和 workspace
安全性。它不会把参数自动写回源码。

本配置下，基线训练 loss 的原有限差分仍为
`delta_r = bound / max_steps = 0.0078125`；公共网格比较差分仍为
`ROIvoxelsize = 0.01360525`。二者作用域不同，没有通过新参数耦合。

新增 `TRAINING_EXPERIMENTS.md`，明确先运行严格控制组，完成后再在共同预算上限
下建立各自调优组。本阶段未启动训练，也未修改 network、renderer、utils、loss、
比较指标或 MATLAB ground truth。
