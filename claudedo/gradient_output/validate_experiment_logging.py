"""在临时目录自检两版实验清单记录与结果完整性关联。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch

from compare_reconstructions import load_experiment_manifest


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
BASELINE_LOGGER = PROJECT_ROOT / 'PYTHON/NIR-BOS/experiment_logging.py'
GRADIENT_LOGGER = SCRIPT_DIR / 'experiment_logging.py'


def load_recorder(path: Path):
    spec = importlib.util.spec_from_file_location('experiment_logging_under_test', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'无法加载实验记录器: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExperimentRecorder


def main() -> None:
    if BASELINE_LOGGER.read_bytes() != GRADIENT_LOGGER.read_bytes():
        raise AssertionError('两版 experiment_logging.py 内容不一致')

    ExperimentRecorder = load_recorder(GRADIENT_LOGGER)
    with tempfile.TemporaryDirectory(prefix='nir_bos_manifest_test_') as directory:
        workspace = Path(directory)
        options = SimpleNamespace(
            workspace=str(workspace),
            iters=12,
            num_rays=34,
            max_steps=56,
            ckpt='scratch',
            seed=7,
            ROIsize=[0.9523675, 1.99997175, 0.9523675],
            ROInum=[140, 294, 140],
            ROIvoxelsize=0.01360525,
        )
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.Linear(5, 3),
        )
        recorder = ExperimentRecorder(
            route='self_test', mode='train', opt=options, model=model,
            device=torch.device('cpu'), entrypoint=str(GRADIENT_LOGGER))

        log_path = workspace / 'log_ngp.txt'
        log_path.write_text('self-test log\n', encoding='utf-8')
        trainer = SimpleNamespace(
            epoch=0, global_step=0, log_path=str(log_path))
        recorder.mark_trainer_initialized(trainer)
        recorder.record_training_time(1.25)

        checkpoint_dir = workspace / 'checkpoints'
        checkpoint_dir.mkdir()
        (checkpoint_dir / 'ngp_ep0001.pth').write_bytes(b'dummy-checkpoint')
        results_dir = workspace / 'results'
        results_dir.mkdir()
        result_path = results_dir / 'sigmas0.mat'
        result_path.write_bytes(b'dummy-mat-content')

        trainer.epoch = 1
        trainer.global_step = 12
        recorder.complete(trainer)

        manifest_path = results_dir / 'experiment_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        checks = {
            'two_route_loggers_identical': True,
            'manifest_completed': manifest['status'] == 'completed',
            'options_recorded': manifest['options']['seed'] == 7,
            'parameter_count_recorded': (
                manifest['model']['trainable_parameter_count'] == 38),
            'nominal_ray_budget_recorded': (
                manifest['budget']['nominal_total_rays'] == 12 * 34),
            'initial_state_recorded': (
                manifest['trainer_state']['initial']['global_step'] == 0),
            'final_state_recorded': (
                manifest['trainer_state']['final']['global_step'] == 12),
            'training_time_recorded': (
                manifest['timing']['training_seconds'] == 1.25),
            'source_hashes_recorded': bool(manifest['source_sha256']),
            'checkpoint_hash_recorded': bool(
                manifest['artifacts']['checkpoints'][0].get('sha256')),
            'result_hash_recorded': bool(
                manifest['artifacts']['results'][0].get('sha256')),
        }

        provenance = load_experiment_manifest(result_path)
        checks['comparison_loader_verifies_result'] = (
            provenance['result_integrity'] == 'sha256_verified')

        result_path.write_bytes(b'tampered')
        try:
            load_experiment_manifest(result_path)
        except ValueError:
            checks['tampered_result_rejected'] = True
        else:
            checks['tampered_result_rejected'] = False

        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise AssertionError(f'实验清单自检失败: {failed}')

        print(json.dumps({
            'status': 'passed',
            'checks': checks,
        }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
