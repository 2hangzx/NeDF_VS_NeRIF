"""记录一次训练/测试运行的配置、源码和结果来源，不改变计算流程。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    record = {
        'name': path.name,
        'path': str(path.resolve()),
        'size_bytes': stat.st_size,
        'modified_at': datetime.fromtimestamp(
            stat.st_mtime).astimezone().isoformat(timespec='seconds'),
    }
    if include_hash:
        record['sha256'] = _sha256(path)
    return record


def _inventory(directory: Path, hash_suffixes: set[str]) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    return [
        _file_record(path, path.suffix.lower() in hash_suffixes)
        for path in sorted(directory.rglob('*')) if path.is_file()
    ]


class ExperimentRecorder:
    """在 workspace 中维护带 run_id 的 JSON 实验清单。"""

    def __init__(self, route: str, mode: str, opt: Any, model: Any,
                 device: Any, entrypoint: str):
        self.route = route
        self.mode = mode
        self.opt = opt
        self.workspace = Path(opt.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = self.workspace / 'experiment_manifests'
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')
        self.path = self.manifest_dir / f'{self.run_id}.json'
        self.latest_path = self.workspace / 'experiment_manifest_latest.json'

        entrypoint_path = Path(entrypoint).resolve()
        source_files = [
            entrypoint_path,
            entrypoint_path.parent / 'experiment_logging.py',
            entrypoint_path.parent / 'nerf/network.py',
            entrypoint_path.parent / 'nerf/renderer.py',
            entrypoint_path.parent / 'nerf/utils.py',
        ]
        source_hashes = {
            str(path): _sha256(path) for path in source_files if path.is_file()
        }
        options = _json_safe(vars(opt))
        iters = int(getattr(opt, 'iters', 0))
        num_rays = int(getattr(opt, 'num_rays', 0))
        max_steps = int(getattr(opt, 'max_steps', 0))

        cuda_available = torch.cuda.is_available()
        runtime = {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': torch.__version__,
            'cuda_available': cuda_available,
            'torch_cuda': torch.version.cuda,
            'device': str(device),
        }
        if cuda_available and torch.device(device).type == 'cuda':
            runtime['gpu_name'] = torch.cuda.get_device_name(device)

        self.data = {
            'schema_version': 1,
            'run_id': self.run_id,
            'route': route,
            'mode': mode,
            'status': 'created',
            'created_at': _now(),
            'working_directory': str(Path.cwd().resolve()),
            'workspace': str(self.workspace),
            'hardcoded_argv': list(sys.argv),
            'options': options,
            'budget': {
                'requested_iterations': iters,
                'rays_per_iteration': num_rays,
                'max_steps_per_ray': max_steps,
                'nominal_total_rays': iters * num_rays,
                'nominal_max_ray_samples': iters * num_rays * max_steps,
                'note': 'upper-bound proxy; occupancy and ray termination change actual samples',
            },
            'model': {
                'class': f'{model.__class__.__module__}.{model.__class__.__name__}',
                'num_layers': getattr(model, 'num_layers', None),
                'hidden_dim': getattr(model, 'hidden_dim', None),
                'encoding': getattr(model, 'encoding_str', None),
                'density_scale': getattr(model, 'density_scale', None),
                'bound': getattr(model, 'bound', None),
                'trainable_parameter_count': sum(
                    parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad),
                'representation': str(model),
            },
            'runtime': runtime,
            'source_sha256': source_hashes,
            'checkpoint_request': getattr(opt, 'ckpt', None),
            'trainer_state': {},
            'timing': {},
            'artifacts': {},
        }
        self._write()

    def _write_path(self, path: Path) -> None:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        temporary.replace(path)

    def _write(self) -> None:
        self._write_path(self.path)
        self._write_path(self.latest_path)

    def mark_trainer_initialized(self, trainer: Any) -> None:
        self.data['status'] = 'initialized'
        self.data['trainer_state']['initial'] = {
            'epoch': int(trainer.epoch),
            'global_step': int(trainer.global_step),
            'resumed_from_checkpoint': bool(trainer.epoch or trainer.global_step),
        }
        self._write()

    def record_extension(self, details: dict[str, Any]) -> None:
        self.data['extension'] = _json_safe(details)
        self._write()

    def record_training_time(self, elapsed_seconds: float) -> None:
        self.data['timing']['training_seconds'] = float(elapsed_seconds)
        self._write()

    def complete(self, trainer: Any) -> None:
        self.data['status'] = 'completed'
        self.data['completed_at'] = _now()
        self.data['trainer_state']['final'] = {
            'epoch': int(trainer.epoch),
            'global_step': int(trainer.global_step),
        }
        self.data['artifacts'] = {
            'checkpoints': _inventory(
                self.workspace / 'checkpoints', {'.pth'}),
            'results': _inventory(self.workspace / 'results', {'.mat'}),
            'log': _file_record(Path(trainer.log_path), False)
            if getattr(trainer, 'log_path', None)
            and Path(trainer.log_path).is_file() else None,
            'train_time': _file_record(self.workspace / 'train_time.txt', False)
            if (self.workspace / 'train_time.txt').is_file() else None,
        }
        self._write()

        results_dir = self.workspace / 'results'
        if results_dir.is_dir():
            self._write_path(results_dir / 'experiment_manifest.json')
