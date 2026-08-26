# 正式实验包封装验收记录

验收日期：2026-08-25

本记录针对开发设备上生成的原始正式实验包。验收期间只创建了临时批次
`packaging_validation_001`，没有启动基线版或梯度版正式训练；验收结束后该临时
批次已从交付目录删除。

## 已通过项目

- 完整搬运清单：全部文件 SHA-256 一致；
- 不可变源码清单：全部文件 SHA-256 一致；
- MATLAB 与 Python 公共 Phantom 1 数据：71 个文件逐一 SHA-256 一致；
- MATLAB、Common、两版 Python、CUDA 源码和控制脚本必需路径齐全；
- 交付源码树无旧设备 `.pyd`、`.pyc`、`build/`、`result/` 等产物；
- 公共比较网格验证：10/10 通过；
- 两路线批次声明、训练参数、checkpoint 恢复、预算延长与路径门禁：33/33 通过；
- 实验日志与结果清单自检：13/13 通过；
- 折射率梯度计算与比较数学自检通过；
- 中央控制器完整 `preflight` 通过并正确写入批次状态；
- 未注入批次号时训练入口拒绝运行；
- 基线版与梯度版训练入口彼此独立，对比入口仍要求两路完整结果。
- 两版原始 `Trainer` 的完整 checkpoint CPU 往返测试通过，模型、optimizer、
  scheduler、scaler、epoch 和 global step 均正确恢复；
- resume 控制器测试通过：scratch 覆盖拒绝、同路线恢复、恢复历史与 SHA-256、
  stale-running 人工确认以及跨目录 checkpoint 拒绝均符合设计。
- extension 独立 CPU 自检 11/11 通过：模型/Adam 状态保留，学习率按子批次声明重置，
  `T_max` 按剩余预算重建，父 checkpoint 保存记录从子批次滚动列表分离；
- extension 控制器测试通过：只接受已完成父路线及完整 checkpoint，校验清单登记与
  SHA-256、网络容量、核心源码哈希、父子批次隔离和目标预算递增，并正确注入独立
  扩展模式环境。

最终清单的精确文件数以包根目录的 `PACKAGE_CHECKSUMS.sha256` 和
`TRANSFER_CHECKSUMS.sha256` 行数为准。换机后应先运行一次
`python experiment_control/verify_transfer.py`，再进行环境配置与实验。
