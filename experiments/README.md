# 正式实验批次目录

不要手工在这里新建零散结果。使用：

```powershell
python experiment_control/experiment.py create --batch-id <唯一批次号>
```

控制程序会创建批次清单、声明配置、两路 workspace 路径和比较路径。归档实验时
复制整个批次目录。训练中断时不要移动或重命名路线内的 `checkpoints/`，应从包根
目录使用对应的 `resume-baseline` 或 `resume-gradient` 命令恢复。

已完成路线后来需要增加总迭代数时，不要修改原批次。先为新总预算创建并检查一个
新子批次，再使用 `extend-baseline --from-batch <父批次>` 或
`extend-gradient --from-batch <父批次>`。子批次清单会记录完整来源，父批次保持
可复现、可独立归档。
