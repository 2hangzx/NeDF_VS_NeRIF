"""已完成短预算实验的延长训练辅助逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


REQUIRED_FULL_CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "lr_scheduler",
    "scaler",
    "ema",
    "epoch",
    "global_step",
    "stats",
}


def inspect_full_checkpoint(path: Path) -> dict[str, Any]:
    """只读检查 checkpoint 是否足以进行状态完整的延长训练。"""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint 不是状态字典: {path}")
    missing = sorted(REQUIRED_FULL_CHECKPOINT_KEYS - payload.keys())
    if missing:
        raise ValueError(f"checkpoint 缺少完整训练状态 {missing}: {path}")

    epoch = int(payload["epoch"])
    global_step = int(payload["global_step"])
    if epoch < 0 or global_step <= 0:
        raise ValueError(
            f"checkpoint 的 epoch/global_step 不适合延长训练: "
            f"epoch={epoch}, global_step={global_step}")

    scheduler = payload["lr_scheduler"]
    optimizer = payload["optimizer"]
    if not isinstance(scheduler, dict) or not isinstance(optimizer, dict):
        raise ValueError(f"checkpoint 的优化器/调度器状态格式无效: {path}")
    if not optimizer.get("param_groups"):
        raise ValueError(f"checkpoint 的优化器没有参数组: {path}")
    if not optimizer.get("state"):
        raise ValueError(f"checkpoint 没有已训练的 optimizer moments: {path}")
    if "T_max" not in scheduler or "last_epoch" not in scheduler:
        raise ValueError(
            f"checkpoint 不是可识别的 CosineAnnealingLR 完整状态: {path}")
    if not isinstance(payload["stats"], dict):
        raise ValueError(f"checkpoint 的训练统计格式无效: {path}")
    return {
        "epoch": epoch,
        "global_step": global_step,
        "contains_ema": True,
        "saved_scheduler_class": "CosineAnnealingLR",
        "saved_scheduler_t_max": scheduler.get("T_max"),
        "saved_scheduler_last_epoch": scheduler.get("last_epoch"),
        "saved_optimizer_lrs": [
            float(group["lr"]) for group in optimizer.get("param_groups", [])
        ],
    }


def restart_cosine_scheduler_for_extension(
        trainer: Any,
        target_iterations: int,
        restart_learning_rate: float,
        eta_min: float = 1e-6,
) -> dict[str, Any]:
    """保留模型/优化器状态，为新增预算重启余弦调度器。"""
    source_global_step = int(trainer.global_step)
    remaining_iterations = int(target_iterations) - source_global_step
    if remaining_iterations <= 0:
        raise ValueError(
            "延长训练的目标 iterations 必须大于 checkpoint global_step: "
            f"target={target_iterations}, checkpoint={source_global_step}")
    if restart_learning_rate <= 0 or eta_min < 0:
        raise ValueError("延长训练学习率必须为正，eta_min 不能为负")
    if eta_min >= restart_learning_rate:
        raise ValueError("eta_min 必须小于延长训练的重启学习率")

    previous_scheduler = trainer.lr_scheduler
    previous_lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]
    previous_checkpoint_records = len(trainer.stats.get("checkpoints", []))
    if not trainer.optimizer.state:
        raise RuntimeError(
            "延长训练未检测到已加载的 optimizer moments；"
            "拒绝把不完整恢复误记为状态连续训练")

    # 父 checkpoint 中的路径属于父批次；清空后可避免子批次滚动保存时删除父文件。
    trainer.stats["checkpoints"] = []
    trainer.stats["results"] = []
    trainer.stats["valid_loss"] = []
    trainer.stats["best_result"] = None

    for group in trainer.optimizer.param_groups:
        group["lr"] = float(restart_learning_rate)
        group["initial_lr"] = float(restart_learning_rate)

    trainer.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=remaining_iterations,
        eta_min=float(eta_min),
    )
    details = {
        "policy": "cosine_restart_over_remaining_iterations",
        "source_epoch": int(trainer.epoch),
        "source_global_step": source_global_step,
        "target_total_iterations": int(target_iterations),
        "remaining_iterations": remaining_iterations,
        "restart_learning_rate": float(restart_learning_rate),
        "eta_min": float(eta_min),
        "loaded_scheduler_t_max": getattr(previous_scheduler, "T_max", None),
        "loaded_scheduler_last_epoch": getattr(
            previous_scheduler, "last_epoch", None),
        "loaded_optimizer_lrs": previous_lrs,
        "parent_checkpoint_records_detached": previous_checkpoint_records,
        "optimizer_moments_preserved": True,
        "ema_and_scaler_preserved": True,
    }
    trainer.log(f"[INFO] Extension scheduler restarted: {details}")
    return details
