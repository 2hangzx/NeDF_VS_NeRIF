# MATLAB → Python 正式实验全流程

以下命令均假设已激活 `huwei` 环境，并从正式实验包根目录开始。

## 一、生成或确认 MATLAB 数据

在 MATLAB 中：

```matlab
cd('<正式实验包>/MATLAB')
step1_InitBOSLAB

% 第一次且现有 MEX 不兼容时运行
step2_Compile

% 生成 Phantom 1 投影、mask、transforms 等数据
step3_generate_phantom1_synthetic_data

% 导出 Python 定量比较使用的精确 flow ground truth
export_phantom1_flow_ground_truth
```

如果只想使用包内已经生成的数据，可跳过 step3，但仍建议重新运行轻量的
`export_phantom1_flow_ground_truth`，确认 MATLAB 工具链和 GT 元数据正常。

## 二、同步 MATLAB 数据到 Python

```powershell
python experiment_control/experiment.py sync-data
python experiment_control/validate_package.py
```

`sync-data` 从 `MATLAB/Test_data/Phantom 1` 复制到
`PYTHON/NIR-BOS/data/Phantom 1`。它不会删除 MATLAB 数据。Python 训练和两类比较
随后读取同一轮生成的数据。

## 三、创建唯一实验批次

```powershell
python experiment_control/experiment.py create `
  --batch-id strict_control_run_001 `
  --note "3x128 strict control, first formal run"
```

批次号只能包含字母、数字、点、下划线和连字符。建议包含组别、序号或日期，例如：

```text
strict_control_run_001
strict_control_seed1_001
tuned_baseline_v1_001
```

## 四、正式训练前门禁

```powershell
python experiment_control/experiment.py preflight --batch-id strict_control_run_001
```

它依次检查：

1. 包完整性和数据副本；
2. MATLAB GT 与两版公共评价网格；
3. 两版严格控制组硬编码和批次路径；
4. 实验清单写入与结果篡改检测；
5. 解析梯度、curl 和常数平移自检。

任何一步失败都不要启动正式训练。

## 五、按需训练基线版（可选）

```powershell
python experiment_control/experiment.py train-baseline `
  --batch-id strict_control_run_001
```

控制程序会设置批次环境并在下面创建 workspace：

```text
experiments/strict_control_run_001/baseline/
```

结束后会核验 `results/sigmas0.mat`、`results/experiment_manifest.json` 及其中的
`route=baseline`、`status=completed`。中途失败时批次状态会记录为 failed。

## 六、按需训练梯度版（可选）

梯度版与基线版没有强制先后关系，可以只训练梯度版，也可以先训练梯度版：

```powershell
python experiment_control/experiment.py train-gradient `
  --batch-id strict_control_run_001
```

输出位置：

```text
experiments/strict_control_run_001/gradient_output/
```

两路可任选其一，也可按任意顺序完成。如果在同一块 GPU 上训练两路，建议不要同时
运行，以免显存竞争；这只是资源建议，不是控制程序的顺序限制。

如果当前实验只需要获得一个模型的结果，完成对应训练命令后即可使用 `status`
检查并归档该路线，无需执行后面的比较步骤。

## 七、训练中断时从 checkpoint 恢复（按需）

两版原始 `Trainer` 都会在每个 epoch 结束时写入完整 checkpoint，并默认滚动保留
最近两个：

```text
experiments/<batch-id>/baseline/checkpoints/ngp_epXXXX.pth
experiments/<batch-id>/gradient_output/checkpoints/ngp_epXXXX.pth
```

完整 checkpoint 包含模型、optimizer、学习率调度器、AMP scaler、EMA、epoch、
global step 和训练统计。基线版中断后执行：

```powershell
python experiment_control/experiment.py resume-baseline `
  --batch-id strict_control_run_001
```

梯度版中断后执行：

```powershell
python experiment_control/experiment.py resume-gradient `
  --batch-id strict_control_run_001
```

默认选择同一路线 epoch 编号最大的完整 checkpoint。也可以明确指定文件名：

```powershell
python experiment_control/experiment.py resume-baseline `
  --batch-id strict_control_run_001 `
  --checkpoint ngp_ep0012.pth
```

正常异常退出或 Ctrl+C 会分别记录为 `failed` 或 `interrupted`。整机断电可能使清单
仍停留在 `running`；确认原进程已经不存在后才可增加 `--allow-stale-running`。
checkpoint 只在 epoch 边界保存，因此中断 epoch 内尚未保存的步数需要重算。

恢复只用于完成原批次已经声明的训练预算，不会增加 `iters`。若已完成的实验后来
需要增加总迭代数，请使用下一节的跨批次延长流程。

## 八、已完成实验需要增加总迭代数时跨批次延长（按需）

典型场景是父批次原计划 10000 iterations 且已经 `completed`，后来决定从其完整
checkpoint 接着训练到总计 20000。这里的 20000 是目标总步数，不是再增加 20000。

首先保留 10000 批次不动，为新预算准备新的硬编码和 profile：

1. 复制 `experiment_control/profiles/strict_control_v1.json`，例如命名为
   `strict_control_20k_v1.json`；
2. 修改新 profile 的 `profile_id`、`training.iterations`，并按
   `iterations × num_rays`、`iterations × num_rays × max_steps` 更新
   `nominal_budget`；
3. 在两版 `main_BOS.py` 的原有硬编码参数中把 `--iters` 改为 20000；如需改变第二
   阶段起始学习率，同时修改两版 `--lr` 和新 profile 的 `training.learning_rate`；
4. 网络结构必须与父 checkpoint 相同。不要修改 `nerf/network.py`、
   `nerf/renderer.py` 或 `nerf/utils.py`；
5. 这些是用户明确授权的实验配置修改，修改完后运行
   `python experiment_control/generate_checksums.py` 更新当前实验包的完整性基线。

随后创建并检查全新的子批次：

```powershell
python experiment_control/experiment.py create `
  --batch-id strict_control_20k_from_10k_001 `
  --profile strict_control_20k_v1 `
  --note "extend completed 10k parent to total 20k"

python experiment_control/experiment.py preflight `
  --batch-id strict_control_20k_from_10k_001
```

按需要延长任意路线：

```powershell
python experiment_control/experiment.py extend-baseline `
  --from-batch strict_control_10k_001 `
  --batch-id strict_control_20k_from_10k_001

python experiment_control/experiment.py extend-gradient `
  --from-batch strict_control_10k_001 `
  --batch-id strict_control_20k_from_10k_001
```

`extend-*` 会核验父路线已完成、checkpoint 已登记且 SHA-256 一致、网络结构及核心
计算源码兼容，并把父 checkpoint 路径和预算关系写入子批次清单。模型权重、Adam
状态、AMP scaler 与 EMA 会保留；优化器学习率重置为子批次硬编码的 `--lr`，并在
`目标 iterations - checkpoint global_step` 的剩余区间上新建
`CosineAnnealingLR(T_max=剩余步数, eta_min=1e-6)`。父 checkpoint 的保存记录会
从子训练统计中分离，子批次滚动保存不会删除父文件。

因此，延长实验是可审计的“两阶段学习率计划”，不等价于从 scratch 直接训练
20000，也不等价于一开始就用 `T_max=20000` 连续训练。报告结果时应明确写成
“10k checkpoint + 10k extension”。

扩展中的子批次若中断且已经产生自己的 checkpoint，之后对该子批次使用普通
`resume-baseline` 或 `resume-gradient`；它会继续保存过的新调度器状态。若中断发生
在第一个子 checkpoint 之前，保留失败批次作为记录，另建新子批次再次延长。

只延长一个模型完全允许，完成后即可归档，不需要运行比较。若要比较，两路都应
延长到同一个子批次并完整结束。

## 九、需要对比时再运行两类正式比较

只有本步骤要求同一批次的基线版与梯度版结果都完整存在：

```powershell
python experiment_control/experiment.py compare `
  --batch-id strict_control_run_001
```

默认不弹交互窗口，结果集中保存到：

```text
experiments/strict_control_run_001/comparisons/flow/
experiments/strict_control_run_001/comparisons/gradient/
```

本地需要交互切片窗口时加 `--show`；磁盘不足、仅临时调试时可加
`--skip-field-save`，正式归档建议保留完整梯度 NPZ。

## 十、查看状态与归档

```powershell
python experiment_control/experiment.py status --batch-id strict_control_run_001
```

完成到当前阶段的批次目录包含：

- 批次清单和声明配置；
- 已训练路线的 checkpoint、训练日志、训练耗时和实验清单；
- 如发生过恢复，批次清单包含 checkpoint 路径、大小、SHA-256 和恢复历史；
- 如由父批次延长，清单还包含父/子批次号、父 checkpoint 哈希、原/目标/剩余预算
  和调度器重启策略；
- 已训练路线的 MAT 结果及 SHA-256；
- 若执行过比较，则包含标量和梯度比较报告、CSV、NPZ 与图片；
- 训练前门禁报告。

归档时复制整个 `experiments/<batch-id>/`，不要只复制 `sigmas0.mat`。
