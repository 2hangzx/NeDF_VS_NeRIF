"""CPU 自检：完整 checkpoint、剩余预算调度器和父/子批次扩展控制。"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

import batch_paths
import experiment
from training_extension import (
    inspect_full_checkpoint,
    restart_cosine_scheduler_for_extension,
)


def main() -> None:
    checks: dict[str, bool] = {}
    original_experiments_root = batch_paths.EXPERIMENTS_ROOT
    original_package_root = experiment.PACKAGE_ROOT
    original_run = experiment.run

    try:
        with tempfile.TemporaryDirectory(prefix="nir_bos_extension_") as temporary:
            test_root = Path(temporary).resolve()
            batch_paths.EXPERIMENTS_ROOT = test_root / "experiments"
            experiment.PACKAGE_ROOT = test_root

            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10000, eta_min=1e-6)
            loss = model(torch.ones(1, 2)).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            parent = "extension_parent_10k"
            child = "extension_child_30k"
            with redirect_stdout(io.StringIO()):
                for batch_id in (parent, child):
                    experiment.create_batch(SimpleNamespace(
                        batch_id=batch_id,
                        profile="strict_control_v1",
                        note="automated extension validation",
                    ))

            parent_profile_path = (
                experiment.batch_root(parent) / "declared_profile.json")
            parent_profile = json.loads(
                parent_profile_path.read_text(encoding="utf-8"))
            parent_profile["training"]["iterations"] = 10000
            parent_profile["nominal_budget"] = {
                "total_rays": 10000 * 256,
                "max_ray_samples": 10000 * 256 * 256,
            }
            experiment.write_json_atomic(parent_profile_path, parent_profile)

            parent_workspace = experiment.route_workspace(parent, "baseline")
            checkpoint_dir = parent_workspace / "checkpoints"
            result_dir = parent_workspace / "results"
            checkpoint_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "ngp_ep0100.pth"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "scaler": {},
                "ema": {},
                "epoch": 100,
                "global_step": 10000,
                "stats": {
                    "checkpoints": [str(checkpoint)],
                    "results": [1.0],
                    "valid_loss": [1.0],
                    "best_result": 1.0,
                },
            }, checkpoint)
            checkpoint_hash = experiment.file_sha256(checkpoint)

            checkpoint_info = inspect_full_checkpoint(checkpoint)
            checks["full_checkpoint_is_accepted"] = (
                checkpoint_info["global_step"] == 10000
                and checkpoint_info["saved_scheduler_t_max"] == 10000
                and bool(checkpoint_info["saved_optimizer_lrs"])
            )

            optimizer_state_before = {
                parameter: {
                    key: value.clone() if torch.is_tensor(value) else value
                    for key, value in state.items()
                }
                for parameter, state in optimizer.state.items()
            }
            trainer = SimpleNamespace(
                global_step=10000,
                epoch=100,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                stats={
                    "checkpoints": [str(checkpoint)],
                    "results": [1.0],
                    "valid_loss": [1.0],
                    "best_result": 1.0,
                },
                log=lambda _: None,
            )
            details = restart_cosine_scheduler_for_extension(
                trainer, 20000, 0.005, eta_min=1e-6)
            checks["remaining_budget_scheduler_is_rebuilt"] = (
                details["remaining_iterations"] == 10000
                and trainer.lr_scheduler.T_max == 10000
                and trainer.lr_scheduler.last_epoch == 0
            )
            checks["optimizer_moments_are_preserved"] = all(
                torch.equal(
                    optimizer_state_before[parameter]["exp_avg"],
                    optimizer.state[parameter]["exp_avg"],
                )
                for parameter in optimizer_state_before
            )
            checks["parent_checkpoint_records_are_detached"] = (
                trainer.stats["checkpoints"] == []
                and trainer.stats["results"] == []
                and checkpoint.is_file()
                and experiment.file_sha256(checkpoint) == checkpoint_hash
            )

            source_hashes = {}
            for relative in (
                "nerf/network.py", "nerf/renderer.py", "nerf/utils.py",
            ):
                path = experiment.BASELINE_DIR / relative
                source_hashes[str(path.resolve())] = experiment.file_sha256(path)
            parent_options = {
                "iters": 10000,
                "lr": 0.005,
                "seed": 0,
                "bound": 2.0,
                "scale": 0.00054421,
                "dt_gamma": 0.0,
                "num_rays": 256,
                "max_steps": 256,
                "fp16": True,
                "cuda_ray": True,
                "maskflag": True,
                "ROIsize": [0.9523675, 1.99997175, 0.9523675],
                "ROInum": [140, 294, 140],
                "ROIvoxelsize": 0.01360525,
                "valbound": [-1.0, 3.0],
            }
            experiment.write_json_atomic(
                result_dir / "experiment_manifest.json", {
                    "run_id": "extension_parent_test",
                    "route": "baseline",
                    "status": "completed",
                    "options": parent_options,
                    "model": {
                        "num_layers": 3,
                        "hidden_dim": 128,
                        "encoding": "Fourier",
                        "density_scale": 1,
                        "bound": 2.0,
                        "trainable_parameter_count": 21505,
                    },
                    "source_sha256": source_hashes,
                    "artifacts": {"checkpoints": [{
                        "name": checkpoint.name,
                        "sha256": checkpoint_hash,
                    }]},
                })
            (result_dir / "sigmas0.mat").write_bytes(b"parent test result")
            experiment.update_batch(parent, lambda data: (
                data.update({"status": "baseline_completed"}),
                data["routes"]["baseline"].update({"status": "completed"}),
            ))
            experiment.update_batch(child, lambda data: data.update({
                "status": "ready",
                "preflight": {"status": "passed"},
            }))

            selected, source_record = experiment.verify_extension_source(
                parent, child, "baseline", "latest")
            checks["parent_source_is_audited"] = (
                selected == checkpoint.resolve()
                and source_record["checkpoint_sha256"] == checkpoint_hash
                and source_record["remaining_iterations"] == 20000
                and source_record["source_declared_iterations"] == 10000
                and source_record["target_declared_iterations"] == 30000
            )

            environment = experiment.command_env(
                child, selected, extension=True)
            checks["extension_environment_is_isolated"] = (
                environment[experiment.BATCH_ENV] == child
                and environment[experiment.CHECKPOINT_ENV] == str(selected)
                and environment[experiment.EXTENSION_ENV] == "1"
            )

            try:
                experiment.verify_extension_source(
                    parent, parent, "baseline", "latest")
            except ValueError:
                checks["same_batch_extension_is_rejected"] = True
            else:
                checks["same_batch_extension_is_rejected"] = False

            def fake_run(
                    command, cwd, batch_id=None, checkpoint=None,
                    extension=False,
            ):
                checks["extend_route_injects_expected_launch"] = (
                    command[-1] == "main_BOS.py"
                    and cwd == experiment.BASELINE_DIR
                    and batch_id == child
                    and checkpoint == selected
                    and extension is True
                )
                child_results = (
                    experiment.route_workspace(child, "baseline") / "results")
                child_results.mkdir(parents=True)
                (child_results / "sigmas0.mat").write_bytes(b"child test result")
                experiment.write_json_atomic(
                    child_results / "experiment_manifest.json", {
                        "run_id": "extension_child_test",
                        "route": "baseline",
                        "status": "completed",
                        "trainer_state": {"final": {"global_step": 30000}},
                        "extension": {
                            "policy": (
                                "cosine_restart_over_remaining_iterations"),
                            "source_global_step": 10000,
                            "remaining_iterations": 20000,
                        },
                    })

            experiment.run = fake_run
            with redirect_stdout(io.StringIO()):
                experiment.extend_route(SimpleNamespace(
                    from_batch=parent,
                    batch_id=child,
                    checkpoint="latest",
                ), "baseline")
            child_manifest = experiment.load_batch_manifest(child)
            child_state = child_manifest["routes"]["baseline"]
            checks["child_lineage_and_completion_are_recorded"] = (
                child_state["status"] == "completed"
                and child_state["launch_mode"] == "extension"
                and child_state["extension_source"]["parent_batch_id"] == parent
                and child_state["extension_source"]["remaining_iterations"]
                == 20000
            )
            checks["parent_checkpoint_remains_immutable"] = (
                checkpoint.is_file()
                and experiment.file_sha256(checkpoint) == checkpoint_hash
            )
            commands = experiment.build_parser()._subparsers._group_actions[0].choices
            checks["both_extension_commands_are_exposed"] = all(
                name in commands for name in (
                    "extend-baseline", "extend-gradient"))
    finally:
        batch_paths.EXPERIMENTS_ROOT = original_experiments_root
        experiment.PACKAGE_ROOT = original_package_root
        experiment.run = original_run
        os.environ.pop(experiment.EXTENSION_ENV, None)

    report = {"passed": all(checks.values()), "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"训练预算延长自检失败: {failed}")


if __name__ == "__main__":
    main()
