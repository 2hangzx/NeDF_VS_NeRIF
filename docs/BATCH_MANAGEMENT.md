# 实验批次与保存路径规则

## 目录策略

每个批次集中保存在：

```text
experiments/<batch-id>/
├── batch_manifest.json
├── declared_profile.json
├── metadata/preflight/
├── baseline/                         仅训练基线版时生成
│   ├── checkpoints/*.pth
│   ├── results/sigmas0.mat
│   ├── results/experiment_manifest.json
│   ├── experiment_manifest_latest.json
│   ├── log_ngp.txt
│   └── train_time.txt
├── gradient_output/                  仅训练梯度版时生成
│   ├── checkpoints/*.pth
│   ├── results/sigmas0.mat
│   ├── results/experiment_manifest.json
│   ├── experiment_manifest_latest.json
│   ├── log_ngp.txt
│   └── train_time.txt
└── comparisons/
    ├── flow/                         执行 compare 后生成
    └── gradient/                     执行 compare 后生成
```

训练代码不再把正式结果写进 `PYTHON/NIR-BOS/result` 或
`claudedo/gradient_output/result`。两版入口仍保留训练参数硬编码，只在解析完成后
从 `NIR_BOS_BATCH_ID` 获取中央 workspace；该变量由控制程序设置。

## 生命周期

```text
                    ┌→ train-baseline  ↔ resume-baseline ─┐
create → preflight ┤                                     ├→ 两路均完成时可选 compare
                    └→ train-gradient ↔ resume-gradient ─┘

completed 父批次 ─→ 新 profile → create 子批次 → preflight
                                      ├→ extend-baseline ─┐
                                      └→ extend-gradient ─┴→ 按需 compare
```

两个训练分支彼此独立，可任选一个、任意顺序运行。只训练一路时不需要进入
`compare`；只有比较命令要求两路结果都完整存在。任一步失败都会写入
`batch_manifest.json`，`status` 命令只读展示当前状态。

## 不覆盖原则

- `create` 拒绝已存在的批次目录。
- `train-baseline` 和 `train-gradient` 固定从 scratch 开始，并拒绝已有 workspace。
- `resume-baseline` 和 `resume-gradient` 反向要求 workspace 与完整 checkpoint 已存在。
- resume 只能选择当前批次当前路线下的 `ngp_epXXXX.pth`，不能跨路线加载。
- 已经 `completed` 的路线不能在同一批次追加训练。
- `extend-*` 是唯一允许读取另一批次 checkpoint 的入口；来源必须是同一路线的已完成
  父批次，输出必须是路线状态为 `pending` 且没有 workspace 的新子批次。

这样既保留原项目断点续训能力，也能避免“其他路线 checkpoint + 新参数 + 同一结果
目录”形成无法解释的实验。

## checkpoint 与恢复审计

原始 `Trainer` 每个 epoch 保存一次完整 checkpoint，默认滚动保留最近两个。完整
文件含网络、optimizer、scheduler、AMP scaler、EMA、epoch、global step 和统计量。
恢复入口默认选择 epoch 编号最大的文件，也接受同一路线内的明确文件名。

每次恢复会在 `batch_manifest.json` 的路线状态中追加 `resume_history`，记录：

- 恢复前状态；
- checkpoint 的包内相对路径；
- 文件大小和 SHA-256；
- 恢复请求时间。

正常异常退出会记录 `failed`，控制台 Ctrl+C 会记录 `interrupted`。断电后若状态仍为
`running`，必须先确认旧进程已经终止，再显式使用 `--allow-stale-running`。恢复从
最近 epoch 边界继续，中断 epoch 内未保存的计算不会保留。

## 已完成预算的跨批次延长

`resume-*` 与 `extend-*` 的语义不同：

| 命令 | 来源与目标 | 总预算 | scheduler |
|---|---|---|---|
| `resume-*` | 同批次、同路线 | 不变；完成原声明 | 完整加载并继续原状态 |
| `extend-*` | 已完成父批次 → 新子批次 | 子批次声明更大的总步数 | 保留训练状态，在剩余步数上重建 cosine |

延长启动前会检查父 checkpoint 的 SHA-256、完整状态、模型层数/宽度/参数量，以及
`network.py`、`renderer.py`、`utils.py` 与父实验清单中的源码哈希。允许变化的是子
profile 和 `main_BOS.py` 中声明的新总预算及第二阶段起始学习率，不允许借延长入口
更换网络或核心计算逻辑。

子批次路线的 `extension_source` 会记录父批次、父 run ID、checkpoint 路径与哈希、
checkpoint global step、目标总步数、剩余步数、父 scheduler 状态以及新 scheduler
策略。父批次目录不写入、不删除；子批次可单独归档，也能清楚追溯来源。

## 新种子与重复实验

当前母版 seed 固定为 0。完全相同配置的重复运行仍应使用不同批次号，例如：

```text
strict_control_repeat_001
strict_control_repeat_002
```

如果要改变 seed 或其他硬编码，则已经形成新实验配置。应同时：

1. 修改两版正式包入口；
2. 新建 profile JSON；
3. 让 profile 完整声明训练、评价网格、参数量、预算与 extension policy；训练前
   验证器会动态读取该批次的 `declared_profile.json`，无需修改验证器源码；
4. 重新生成包校验和；
5. 使用新批次号，并在 `--note` 写明差异。

## 各自调优组

各自调优组不能和严格控制组混在同一批次。建议命名：

```text
tuned_baseline_<profile>_<run>
tuned_gradient_<profile>_<run>
```

若两路调优配置不同，控制程序需要增加对应 profile/门禁后再运行。共同数据、ROI、
MATLAB GT、评价网格和最终指标不能改变；预算上限和 `bound/max_steps` 必须在批次
说明中明确记录。
