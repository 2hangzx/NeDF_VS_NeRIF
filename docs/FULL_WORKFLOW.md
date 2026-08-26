# MATLAB → Python 通用实验全流程

本文说明 `formal_experiment_package` 中任意实验配置从准备、声明、预检、训练、恢复、
扩展、比较到归档的完整顺序。它既适用于严格控制组，也适用于冒烟测试、不同随机
种子、不同训练预算和各自调优组。

以下命令均假设：

- 已激活项目 Python 环境（默认环境名为 `huwei`）；
- 当前目录为 `formal_experiment_package/` 包根目录；
- 所有命令均由 `experiment_control/experiment.py` 统一管理；
- 每个实验使用唯一的 `batch-id`，不手工复用已有批次目录。

## 总体顺序

```text
新设备一次性验收
        ↓
生成或确认 MATLAB 数据 → 同步到 Python → 数据与包验证
        ↓
选择已有 profile，或设计一个新 profile
        ↓
若配置有变化：同步修改两份 main_BOS.py → 更新校验和
        ↓
create 新批次 → preflight
        ↓
按需 train-baseline 和/或 train-gradient
        ↓                         ↓
正常完成                     中断/失败
        ↓                         ↓
按需 compare                 同批次 resume-*
        └──────────────┬──────────┘
                       ↓
                  status 与归档

已完成父批次需要增加总预算：
新 profile + 新硬编码 + 新校验和 → create 子批次 → preflight → extend-*
```

## 一、新设备收到实验包后的一次性验收

此步骤只在新设备首次收到原始实验包时执行，而且必须在修改源码、profile、数据或
重新生成校验和之前执行：

```powershell
python experiment_control/verify_transfer.py
python experiment_control/validate_package.py
```

`verify_transfer.py` 验证收到的文件是否与交付时一致。开始本地配置修改或数据同步
后，不再用原始搬运校验和判断本地工作副本。

新设备的 Python、CUDA、编译器和 MATLAB 配置见 `docs/NEW_DEVICE_SETUP.md`。

## 二、生成或确认 MATLAB 数据

如果使用包内已有 Phantom 1 数据，可以跳过重新生成投影；如果需要重新生成数据，
在 MATLAB 中执行：

```matlab
cd('<正式实验包>/MATLAB')
step1_InitBOSLAB

% 仅在首次编译或现有 MEX 与当前设备不兼容时运行
step2_Compile

% 生成投影、mask、transforms 等数据
step3_generate_phantom1_synthetic_data

% 导出 Python 定量比较所需的精确 flow ground truth
export_phantom1_flow_ground_truth
```

即使复用包内投影数据，也建议运行轻量的 `export_phantom1_flow_ground_truth`，确认
MATLAB 工具链、GT 和元数据能够正常生成。

## 三、同步并验证训练数据

将 MATLAB 数据同步到两版 Python 共用的数据目录：

```powershell
python experiment_control/experiment.py sync-data
python experiment_control/validate_package.py
```

`sync-data` 从 `MATLAB/Test_data/Phantom 1` 复制到
`PYTHON/NIR-BOS/data/Phantom 1`，不会删除 MATLAB 原数据。基线版、梯度版及比较程序
随后读取同一轮数据。

每次重新生成 MATLAB 投影或 GT 后，都要重新执行本步骤。只修改训练超参数时不需要
重新同步数据。

## 四、确定本次实验配置

### 4.1 使用已有配置

已有配置位于：

```text
experiment_control/profiles/<profile-id>.json
```

例如严格控制组使用：

```text
experiment_control/profiles/strict_control_v1.json
```

使用已有 profile 前，仍需确认两份入口当前的硬编码与该 profile 完全一致：

```text
PYTHON/NIR-BOS/main_BOS.py
claudedo/gradient_output/main_BOS.py
```

如果当前硬编码已经匹配，可以直接进入“六、创建批次”。

### 4.2 创建新配置

凡是改变下列任一内容，都应视为新实验配置：

- 随机种子、迭代数、学习率；
- rays 数、ray-marching steps；
- 编码方式、网络层数或隐藏层宽度；
- `bound`、`scale`、`dt_gamma`、mask 开关；
- ROI、评价网格或输出范围；
- 其他会改变训练预算、网络参数量或结果解释的设置。

创建新配置时，按以下顺序操作：

1. 复制最接近的现有 profile，例如：

   ```powershell
   Copy-Item experiment_control/profiles/strict_control_v1.json `
     experiment_control/profiles/<new-profile-id>.json
   ```

2. 修改新 profile 的 `profile_id`、`purpose`、`training` 和 `evaluation_grid`。

3. 更新名义预算：

   ```text
   total_rays       = iterations × num_rays
   max_ray_samples  = iterations × num_rays × max_steps
   ```

4. 将同一组训练参数同步写入两份 `main_BOS.py`。严格对照实验要求两路除输出结构和
   研究变量外保持一致。

profile 是实验的声明，`main_BOS.py` 是实际执行配置。两者必须一致；只修改其中一处
会导致 `preflight` 失败。参数量由验证器根据网络结构自动计算并写入报告，profile
不需要设置 `expected_trainable_parameters`。

## 五、参数或源码变化后更新完整性基线

新建或修改 profile、修改两份入口或修改其他受校验源码后，执行：

```powershell
python experiment_control/generate_checksums.py
```

必须在所有配置修改完成之后、创建批次之前执行。若生成校验和后又修改了受控文件，
需要再次生成。

仅创建批次、训练、同步数据或写入 `experiments/` 结果，不需要为每次运行重新生成
校验和。

## 六、创建唯一实验批次

使用已确定的 profile 创建批次：

```powershell
python experiment_control/experiment.py create `
  --batch-id <unique-batch-id> `
  --profile <profile-id> `
  --note "<本批次目的和关键差异>"
```

例如：

```powershell
python experiment_control/experiment.py create `
  --batch-id strict_control_seed0_001 `
  --profile strict_control_v1 `
  --note "strict control, seed 0, first run"
```

批次号只能包含字母、数字、点、下划线和连字符。推荐包含配置、种子、序号或日期：

```text
strict_control_seed0_001
smoke_cuda_001
tuned_baseline_hash_v1_001
```

`create` 会把 profile 复制到：

```text
experiments/<batch-id>/declared_profile.json
```

该副本是本批次的固定声明。创建后再修改源 profile，不会更新已经创建的批次。若
声明有误，不要手工改写批次副本；保留或标记错误批次，修正配置后使用新批次号重新
创建。

## 七、执行训练前门禁

```powershell
python experiment_control/experiment.py preflight `
  --batch-id <batch-id>
```

`preflight` 依次检查：

1. 包完整性和必需文件；
2. MATLAB 与 Python 数据副本；
3. MATLAB GT 与两版公共评价网格；
4. 批次声明与两份 `main_BOS.py` 的硬编码；
5. 两版网络主干、训练预算、采样参数和保存路径；
6. 实验清单写入、结果篡改检测和 checkpoint 扩展机制；
7. 解析梯度、curl、常数平移等数学自检。

任何检查失败都不要启动训练。修正配置或源码后，重新生成校验和，并为尚未开始的
实验创建一个新的批次最为稳妥。

## 八、启动训练路线

基线版和梯度版彼此独立，可以只训练一路，也可以按任意顺序训练两路。

### 8.1 基线版

```powershell
python experiment_control/experiment.py train-baseline `
  --batch-id <batch-id>
```

输出目录：

```text
experiments/<batch-id>/baseline/
```

### 8.2 梯度版

```powershell
python experiment_control/experiment.py train-gradient `
  --batch-id <batch-id>
```

输出目录：

```text
experiments/<batch-id>/gradient_output/
```

同一块 GPU 上建议串行训练两路，避免显存竞争。只需要一个模型结果时，训练对应路线
即可，无需运行另一条路线或 `compare`。

训练结束后控制器会核验：

```text
results/sigmas0.mat
results/experiment_manifest.json
```

并要求实验清单中的 `route` 正确且 `status=completed`。

## 九、训练期间的配置锁定规则

两份 `main_BOS.py` 是当前实验包的全局执行配置，不由每个批次单独加载。因此从某个
批次通过 `preflight` 开始，直到该批次所需路线完成或明确放弃期间：

- 不要切换两份入口到另一个 profile；
- 不要用同一实验包交叉训练两个不同硬编码配置的批次；
- 需要恢复的批次必须继续使用它原先声明的网络和训练配置；
- 若需并行维护不同配置，应复制出独立实验包，每个副本固定一种配置。

切换到另一个配置时，应先同步修改两份入口、确认对应 profile、重新生成校验和，
然后使用新批次号执行 `create → preflight`。

## 十、训练中断后的同批次恢复

训练每个 epoch 结束时保存完整 checkpoint，并默认滚动保留最近两个：

```text
experiments/<batch-id>/baseline/checkpoints/ngp_epXXXX.pth
experiments/<batch-id>/gradient_output/checkpoints/ngp_epXXXX.pth
```

checkpoint 包含模型、optimizer、scheduler、AMP scaler、EMA、epoch、global step 和
训练统计。

恢复基线版：

```powershell
python experiment_control/experiment.py resume-baseline `
  --batch-id <batch-id>
```

恢复梯度版：

```powershell
python experiment_control/experiment.py resume-gradient `
  --batch-id <batch-id>
```

默认选择当前批次、当前路线 epoch 最大的完整 checkpoint，也可指定文件名：

```powershell
python experiment_control/experiment.py resume-baseline `
  --batch-id <batch-id> `
  --checkpoint ngp_ep0012.pth
```

异常退出或 Ctrl+C 会分别记录为 `failed` 或 `interrupted`。整机断电后清单可能仍为
`running`；只有确认旧进程已经终止，才能显式执行：

```powershell
python experiment_control/experiment.py resume-baseline `
  --batch-id <batch-id> `
  --allow-stale-running
```

checkpoint 只在 epoch 边界保存，中断 epoch 内未保存的计算需要重算。`resume-*` 只
完成原批次已声明的总预算，不增加 `iterations`，也不能跨批次或跨路线加载。

## 十一、已完成批次增加总预算

已完成路线不能在原批次内继续追加训练。需要增加总迭代数时，使用“已完成父批次 →
全新子批次”的扩展流程。

### 11.1 准备新的目标配置

1. 复制父批次对应 profile，创建新的 profile ID；
2. 将 `training.iterations` 改成目标总步数，而不是额外增加的步数；
3. 更新 `nominal_budget`；
4. 同步修改两份 `main_BOS.py` 的 `--iters`；
5. 如需新的第二阶段起始学习率，同步修改 profile 与两份入口的 `--lr`；
6. 保持父 checkpoint 的网络结构不变，不修改相关核心计算逻辑；
7. 运行 `python experiment_control/generate_checksums.py`。

### 11.2 创建并预检子批次

```powershell
python experiment_control/experiment.py create `
  --batch-id <child-batch-id> `
  --profile <extended-profile-id> `
  --note "extend <parent-batch-id> to total <target-iterations> iterations"

python experiment_control/experiment.py preflight `
  --batch-id <child-batch-id>
```

### 11.3 延长所需路线

```powershell
python experiment_control/experiment.py extend-baseline `
  --from-batch <parent-batch-id> `
  --batch-id <child-batch-id>

python experiment_control/experiment.py extend-gradient `
  --from-batch <parent-batch-id> `
  --batch-id <child-batch-id>
```

可只延长其中一路。`extend-*` 要求父路线已完成、checkpoint 已登记且 SHA-256 一致，
并要求子批次网络与父 checkpoint 兼容。

扩展会保留模型、Adam、AMP scaler 和 EMA 状态；优化器学习率重置为子 profile 声明
的起始学习率，并在：

```text
目标 iterations - 父 checkpoint global_step
```

对应的剩余区间上重建 cosine scheduler。因此扩展是可审计的两阶段学习率计划，不
等价于从 scratch 直接训练到目标步数。报告时应明确写出父 checkpoint 与扩展预算。

扩展子批次中断后，如果已经生成子批次自己的 checkpoint，使用普通 `resume-*`
继续该子批次；不要再次从父批次重复执行 `extend-*`。

## 十二、两路结果比较

只有在同一批次的基线版和梯度版均为 `completed` 时，才能执行：

```powershell
python experiment_control/experiment.py compare `
  --batch-id <batch-id>
```

输出位置：

```text
experiments/<batch-id>/comparisons/flow/
experiments/<batch-id>/comparisons/gradient/
```

默认不弹出交互窗口。本地需要交互切片时可加 `--show`；临时调试且磁盘不足时可加
`--skip-field-save`，正式归档建议保留完整梯度 NPZ。

如果只训练或只延长一路，不需要也不能执行 `compare`。

## 十三、查看状态与验收结果

任何阶段都可以只读查看：

```powershell
python experiment_control/experiment.py status `
  --batch-id <batch-id>
```

至少检查：

- `preflight.status` 是否为 `passed`；
- 所需路线是否为 `completed`；
- `experiment_manifest.json` 的 profile、route、最终 global step 和源码哈希；
- `results/sigmas0.mat` 是否存在且已登记 SHA-256；
- 日志中是否出现 NaN、CUDA 错误或非预期恢复；
- 若执行比较，比较报告、CSV、NPZ 和图片是否完整。

不要把 `failed`、`interrupted`、`running` 或实验清单未完成的运行用于正式结论。

## 十四、归档与下一配置

归档时复制整个目录：

```text
experiments/<batch-id>/
```

不要只复制 `sigmas0.mat`。完整批次还包含：

- `batch_manifest.json` 与 `declared_profile.json`；
- preflight 报告；
- checkpoint、日志、训练时间和实验清单；
- 恢复或扩展历史及 checkpoint SHA-256；
- 结果 MAT 文件及哈希；
- 可选的标量和梯度比较产物。

开始下一种配置前，重新执行以下配置闭环：

```text
选择/新建 profile
→ 同步两份 main_BOS.py
→ 更新 nominal_budget 和参数量声明
→ generate_checksums.py
→ create 新批次
→ preflight
→ train / resume / extend
→ status / compare / archive
```

不要通过修改已创建批次的 `declared_profile.json`、覆盖旧 workspace 或在已完成批次
中追加训练来切换配置。
