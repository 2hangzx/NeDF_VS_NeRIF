# 基线版—梯度版训练实验母版

> 本文件保留母版设计依据。正式实验包已经把保存路径统一迁移到
> `experiments/<batch-id>/`，请使用包根目录的 `experiment_control/experiment.py`；
> 下文旧的 route-local workspace 命令不再作为正式执行入口。

## 1. 实验分组

依据 `claudedo/codex_comu/讨论1.md`，正式实验分为两类：

| 分组 | 目的 | 配置原则 | 当前状态 |
| --- | --- | --- | --- |
| 严格控制组 | 判断相同训练设置下哪条路线更有效 | 网络主干、迭代数、学习率、采样、优化器等相同 | `strict_control_v1` 已就绪 |
| 各自调优组 | 判断两条路线各自合理调优后的上限 | 相同数据、ROI、评价网格和总预算上限，路线内调参 | 等严格控制组结果后建立 |

严格控制组必须先完成。各自调优组不得复用或覆盖严格控制组 workspace；应使用
新的实验名，并在修改后重新运行训练前验证和保存新阶段快照。

## 2. 当前严格控制组 v1

两版 `main_BOS.py` 继续使用原项目的硬编码 `sys.argv`。正式包控制器仍调用各自的
`main_BOS.py`，只注入批次 workspace；没有引入训练参数配置系统或自动改写逻辑。

| 参数 | 基线版 | 梯度版 |
| --- | ---: | ---: |
| Fourier MLP 主干 | 3 层 × 128 | 3 层 × 128 |
| 预期可训练参数量 | 21,505 | 21,766 |
| iterations | 30,000 | 30,000 |
| learning rate | 5e-3 | 5e-3 |
| rays / iteration | 256 | 256 |
| max steps / ray | 256 | 256 |
| seed | 0 | 0 |
| optimizer | Adam, betas=(0.9, 0.99), eps=1e-8 | 相同 |
| scheduler | CosineAnnealingLR, eta_min=1e-6 | 相同 |
| checkpoint | scratch | scratch |

这里的 3×128 是 `strict_control_v1` 自身声明的实验配置，不是验证器施加的全局容量
下限。其他 profile 可以声明 2×64 等结构，只要两版入口硬编码与 profile 一致；
参数量由验证器自动计算和报告，不要求写入 JSON。按当前 `network.py` 的计数语义，
`num_layers` 包含输出层，因此
`num_layers=2, hidden_dim=64` 实际是一个 64 神经元隐藏层加一个输出层；若要表达
两个 64 神经元隐藏层，应使用 `num_layers=3, hidden_dim=64`。

参数量的小差异只来自方法定义所必需的输出头：基线输出 1 个标量，梯度版输出
3 个分量并带 3 个可训练 scale；隐藏层数和宽度完全一致。名义训练预算为
7,680,000 条 rays、最多 1,966,080,000 个 ray samples。occupancy 跳过和提前终止
会使实际采样量低于该上界。

工作目录分别为：

```text
PYTHON/NIR-BOS/result/experiments/strict_control_v1/baseline
claudedo/gradient_output/result/experiments/strict_control_v1/gradient_output
```

二者都是新目录，不会覆盖旧训练结果。基线每个采样点需要额外有限差分/自动微分
查询，因此相同迭代和采样预算不意味着相同墙钟时间；实验清单会分别记录训练耗时，
这正是两种路线计算效率差异的一部分。

## 3. 两种有限差分步长

严格控制组中：

```text
基线训练 loss 有限差分：delta_r = bound / max_steps = 2 / 256 = 0.0078125
公共网格梯度比较差分：  delta_eval = ROIvoxelsize = 0.01360525
```

前者仍由原 renderer 公式和 `max_steps` 决定，同时服务于基线训练的 ray marching
设置；后者只由统一导出网格决定。第四阶段没有把二者绑定成同一个参数，也没有
修改 renderer。梯度版训练没有基线版的标量网络有限差分，但使用相同 max_steps
进行 ray marching。

## 4. 训练前检查

在任何一次正式训练前先运行：

```bash
cd claudedo/gradient_output
python validate_common_grid.py
python validate_training_readiness.py --batch-id strict_control_run_001
```

第二个命令只读解析两版源码，检查共同数据、网络主干、预算、采样、优化器、
调度器、scratch 和新 workspace。第一次运行要求 workspace 不存在，以防误覆盖；
训练已经开始后可用以下命令复查其余配置：

```bash
python validate_training_readiness.py --batch-id strict_control_run_001 --allow-existing-workspaces
```

## 5. 正式运行方式

创建批次并通过 `preflight` 后，可以任选一个模型训练，也可以按任意顺序完成两路：

```powershell
python experiment_control/experiment.py train-baseline --batch-id strict_control_run_001
python experiment_control/experiment.py train-gradient --batch-id strict_control_run_001
```

两条命令彼此没有顺序依赖。只需要一个模型结果时，仅运行对应命令即可；同一块
GPU 上不建议并发运行，这是显存与性能建议，不是训练门禁。

若某一路线在完整结束前中断，正式包保留原始 `Trainer` 的完整 checkpoint 机制。
使用包根目录的 `resume-baseline` 或 `resume-gradient` 恢复同批次同路线最近的
`ngp_epXXXX.pth`；不要把 `--ckpt` 人工改成另一条路线的文件。

每一路完成后至少检查：

```text
checkpoints/*.pth
results/sigmas0.mat
results/experiment_manifest.json
experiment_manifest_latest.json
log_ngp.txt
train_time.txt
```

若运行中断，不要把未完成清单对应的结果用于正式比较。应检查清单中的 `status`、
`trainer_state` 和 checkpoint，再决定恢复原批次还是启用新的实验目录。恢复只用于
完成原预算。

若路线已经完成，后来仅决定提高总迭代数，可保持网络结构和核心计算代码不变，
建立声明新总预算的子 profile/子批次，再用控制器的 `extend-baseline` 或
`extend-gradient` 从父批次完整 checkpoint 延长。它保留模型、Adam、scaler 和
EMA，按子批次硬编码 `--lr` 开始第二阶段，并在目标总步数减 checkpoint global
step 的剩余区间重建 cosine scheduler。该结果属于有明确父 checkpoint 的两阶段
训练，不应描述为从 scratch 的同预算实验。详细操作见正式包
`docs/FULL_WORKFLOW.md` 第八节。

如果改变网络结构、renderer、loss 或其他核心算法，则不属于预算延长，必须从
scratch 使用新 profile 和新批次。

## 6. 严格控制组比较命令

需要对比且同一批次两路均成功完成后，从正式实验包根目录运行：

```powershell
python experiment_control/experiment.py compare --batch-id strict_control_run_001
```

比较脚本将读取两个结果旁的实验清单并核对 SHA-256。若任一 MAT 与清单不一致，
正式比较会被拒绝。

## 7. 各自调优组边界

各自调优组在严格控制组完成后再确定具体参数。允许两版分别调整 learning rate、
iterations、num_rays 和 max_steps，但必须：

1. 使用同一数据、seed 方案、ROI、公共评价网格、MATLAB GT 和评价指标；
2. 事先声明统一的预算上限，并同时报告名义 rays、名义最大 samples 和实际训练时间；
3. 使用 `scratch` 和全新的 route-specific workspace；
4. 保留每次尝试的实验清单，不能只保留表现最好而来源不明的 MAT；
5. 将调参组结论与严格控制组分开报告。

`max_steps` 在基线版还决定训练 loss 的有限差分步长，因此各自调优时若改变它，
必须在报告中同时写明新的 `bound / max_steps`，不能只把它描述成采样数量变化。
