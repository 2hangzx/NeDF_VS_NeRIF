# 设计文档：Gradient Output 架构

> 对应于 `claudedo/gradient_output_analysis.md` 中的需求分析

## 一、为什么这样设计

### 原版瓶颈

原版网络输出标量折射率 σ(x,y,z)，梯度通过后处理获取：

```
σ = network(xyz)                    # 1 次查询
∇σ_x = [σ(x+Δ) - σ(x-Δ)] / 2Δ      # 额外 2 次查询
∇σ_y = [σ(y+Δ) - σ(y-Δ)] / 2Δ      # 额外 2 次查询
∇σ_z = [σ(z+Δ) - σ(z-Δ)] / 2Δ      # 额外 2 次查询
                                    # 共计 7 次查询 每个采样点
```

再加上 autograd 梯度的额外计算图构建，训练时每个采样点需要多次网络推理，计算密集且内存占用大。

### 新方案

```
∇σ = network(xyz)                   # 1 次查询，直接输出 3 通道
```

训练效率提升约 **7 倍**（单步），显存占用减少（无需中间激活用于 autograd）。

## 二、已确认的设计决策

| # | 决策 | 理由 |
|---|------|------|
| 1 | **tanh + 可学习 scale** | ∇σ 值域无界，tanh 约束到 [-1,1] 再乘以可学习的幅值 scale，兼顾稳定性和表达能力 |
| 2 | **仅投影域监督** | 训练数据（PNG）只包含投影域标签，不加 3D 域监督可避免准备 ∇σ_gt 的额外工作 |
| 3 | **泊松重建 σ** | 导出时从 ∇σ 网格解泊松方程 ∇²σ = div(g) 恢复标量场，供 mesh 提取。阶段二实现 |
| 4 | **沿用 Fourier 编码** | 编码方式与输出类型无关；Hash 编码无 CUDA 加速会更慢 |

## 三、网络架构

```
(x, y, z)  →  FourierEncoder  →  MLP (2层×64维)  →  Linear(64, 3)
                                                      ↓
                                               scale × tanh(x)
                                                      ↓
                                                 ∇σ (3通道)
```

与原版的区别仅在最后一层的宽度（1→3）和激活函数（custom_tanh→trainable_tanh）。

## 四、体渲染路径

```
原版:
  self(xyzs) → σ → 有限差分 → dsigmas_dxyz → composite_rays_train → depth
                → autograd  → dsigmas_dxyz_auto → composite_rays_train → depth_auto

新版:
  self(xyzs) → ∇σ → composite_rays_train → depth
  (一步到位，depth ≡ depth_auto，无需两个分支)
```

## 五、损失函数

```python
loss = MSE(depth, gt_rgb)
```

与原版等价——`depth` 是沿光线积分的 ∇σ 得到偏转角投影，`gt_rgb` 是归一化偏转角投影标签。同尺度，可直接 MSE。

## 六、导出

推理模式下，直接在公共规则网格 (140×294×140) 上查询网络。公共网格间距为
`0.01360525`，半尺寸为 `[0.9523675, 1.99997175, 0.9523675]`，与 MATLAB
精确 ground truth 一致：

```python
gradient0 = self(xyzs0, xyzs0)    # → (140, 294, 140, 3)
```

相当于网络预测的折射率梯度场。σ 标量场通过泊松重建恢复（阶段二）。

## 七、统一梯度评价

公共网格上保留三种预测梯度：

```text
baseline_grid_fd = gradient(baseline sigmas0, public spacing)
baseline_auto    = baseline scalar network autograd
gradient_output  = gradient network direct output
```

其中 `baseline_grid_fd` 与 ground truth 使用同一离散差分算子，是正式体素场主基线；
`baseline_auto` 不被丢弃，而是作为连续模型梯度诊断。两者以及梯度版输出在同一
mask、active-gradient 区域和指标集合中同时报告，不进行尺度或符号拟合。

## 八、实验可追溯性

两版入口在模型创建后初始化相同的旁路记录器：

```text
硬编码 argv + 解析参数 + 模型规模 + 运行环境 + 源码 SHA-256
                              ↓
              workspace/experiment_manifests/<run_id>.json
                              ↓ 训练/测试及导出完成
              results/experiment_manifest.json + 产物 SHA-256
                              ↓
              两类比较报告加载清单并校验输入 MAT
```

记录器不向训练器回传任何值，也不改变 loss、优化器、有限差分步长、光线采样或
导出网格。历史清单按 `run_id` 保留，latest 清单便于快速查看，结果旁清单负责把
某次导出与其参数和源码绑定。旧结果缺少清单时允许回归运行但明确标记；新结果
有清单而哈希不匹配时拒绝比较。

## 九、正式训练实验设计

正式实验先运行严格控制组，再运行各自调优组：

```text
共同数据、GT、ROI、公共网格与评价指标
                   │
        ┌──────────┴──────────┐
        │                     │
严格控制组                 各自调优组
相同主干/预算/采样          共同预算上限内分别调参
回答方法差异               回答合理最佳上限
```

`strict_control_v1` 使用共同 3×128 Fourier MLP 主干和
30000×256 rays、每 ray 最多 256 steps 的名义预算。输出头因任务定义分别为 1 和
3 通道，不强行做无意义的参数量完全相等；小幅参数差异在实验清单中显式记录。

两版均从 scratch 开始并写入新的独立 workspace。训练前只读门禁验证入口硬编码
及优化器/调度器 AST，不改变原项目运行方式。基线训练有限差分继续由
`bound / max_steps` 决定，公共评价差分继续由 `ROIvoxelsize` 决定。
