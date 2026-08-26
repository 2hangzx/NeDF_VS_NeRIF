"""在统一 flow 代数空间中比较基线版与梯度版的折射率重建。

两版的共同空间为 MATLAB step3 中定义的
    flow = (n / n0 - 1) / flow_max

梯度版仅预测 ∇flow，泊松反演后存在一个不可观测的加性常数。本脚本只用
共同背景参考区补偿该常数，不对任一预测做 min-max 或幅值缩放，因此 RMSE
和 MAE 能反映真实的幅值重建误差。

示例（从任意工作目录均可运行）：
    conda activate huwei
    python claudedo/gradient_output/compare_reconstructions.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom
import torch

from nerf.poisson_solver import poisson_reconstruct


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description='在统一 flow 空间比较基线版与梯度版折射率重建。')
    parser.add_argument('--baseline', type=Path,
                        default=PROJECT_ROOT / 'PYTHON/NIR-BOS/result/phantom 1/test/results/sigmas0.mat',
                        help='基线版 sigmas0.mat')
    parser.add_argument('--gradient', type=Path,
                        default=SCRIPT_DIR / 'result/gradient_test/results/sigmas0.mat',
                        help='梯度版 sigmas0.mat（必须含 dsigmas_dxyz_auto0）')
    parser.add_argument('--ground-truth', type=Path,
                        default=PROJECT_ROOT / 'MATLAB/Test_data/Phantom 1/140x294x140/flow_ground_truth.mat',
                        help='由 export_phantom1_flow_ground_truth.m 导出的精确 flow GT')
    parser.add_argument('--skip-ground-truth', action='store_true',
                        help='跳过所有 GT 指标，只比较基线版与梯度版')
    parser.add_argument('--allow-legacy-ground-truth', action='store_true',
                        help='允许读取原始 n_GroundTruth.mat 并用 SciPy 近似 MATLAB imresize3（仅兼容旧流程）')
    parser.add_argument('--mask', type=Path,
                        default=PROJECT_ROOT / 'PYTHON/NIR-BOS/data/Phantom 1/140x294x140/3Dmask.mat',
                        help='可选 3Dmask.mat；尺寸不一致时自动退回边界参考区')
    parser.add_argument('--spacing', type=float, nargs=3,
                        default=(0.01360525, 0.01360525, 0.01360525),
                        metavar=('DX', 'DY', 'DZ'),
                        help='NeRF 坐标系中三个方向的体素中心间距')
    parser.add_argument('--pad', type=int, default=8, help='泊松反演的反射填充宽度')
    parser.add_argument('--output-dir', type=Path,
                        default=SCRIPT_DIR / 'result/gradient_test/comparison_flow_exact',
                        help='报告、对齐场和切片图的输出目录')
    parser.add_argument('--no-show', action='store_true',
                        help='仅生成文件，不弹出交互式切片窗口（适用于服务器/批处理）')
    return parser.parse_args()


def load_field(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'找不到 {key}: {path}')
    values = sio.loadmat(path)
    if key not in values:
        raise KeyError(f'{path} 未包含变量 {key!r}；实际变量: {sorted(k for k in values if not k.startswith("__"))}')
    return np.asarray(values[key]).squeeze().astype(np.float32, copy=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_experiment_manifest(result_path: Path) -> dict:
    """读取结果 MAT 同目录的训练清单；旧结果缺失清单时保持兼容。"""
    manifest_path = result_path.parent / 'experiment_manifest.json'
    if not manifest_path.is_file():
        return {
            'status': 'missing_legacy_manifest',
            'expected_file': str(manifest_path.resolve()),
            'note': 'result predates automatic experiment manifests or was moved alone',
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'实验清单无法读取: {manifest_path}: {error}') from error

    result_records = manifest.get('artifacts', {}).get('results', [])
    matching = next(
        (record for record in result_records
         if record.get('name') == result_path.name),
        None,
    )
    integrity = 'not_recorded'
    if matching is not None and matching.get('sha256'):
        actual_hash = file_sha256(result_path)
        expected_hash = matching['sha256']
        if actual_hash != expected_hash:
            raise ValueError(
                f'结果文件与实验清单 SHA-256 不一致: {result_path}; '
                f'expected={expected_hash}, actual={actual_hash}')
        integrity = 'sha256_verified'

    return {
        'status': 'loaded',
        'file': str(manifest_path.resolve()),
        'result_integrity': integrity,
        'manifest': manifest,
    }


def common_mask(mask_path: Path, shape: tuple[int, int, int]) -> tuple[np.ndarray, str]:
    """返回评价区域；优先使用同尺寸 mask，否则使用整个有限域。"""
    if mask_path.is_file():
        values = sio.loadmat(mask_path)
        mask = values.get('maskback')
        if mask is not None:
            mask = np.asarray(mask).squeeze()
            if tuple(mask.shape) == shape:
                return mask.astype(bool), f'maskback: {mask_path}'
            # 历史数据中保存过 294^3 的 maskback；其 y 轴已匹配，而 x/z
            # 比 ROI 更大。按 MATLAB step3 的同一中心裁剪规则取共同 ROI。
            if mask.ndim == 3 and all(source >= target for source, target in zip(mask.shape, shape)):
                starts = [(source - target) // 2 for source, target in zip(mask.shape, shape)]
                slices = tuple(slice(start, start + target) for start, target in zip(starts, shape))
                return mask[slices].astype(bool), f'center-cropped maskback: {mask_path}'
    return np.ones(shape, dtype=bool), 'entire volume (mask absent or shape mismatch)'


def border_mask(shape: tuple[int, int, int], width: int = 3) -> np.ndarray:
    """掩码不可用时，以 ROI 外壳估计梯度积分丢失的常数项。"""
    border = np.zeros(shape, dtype=bool)
    border[:width] = border[-width:] = True
    border[:, :width] = border[:, -width:] = True
    border[:, :, :width] = border[:, :, -width:] = True
    return border


def align_gauge(field: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """仅移除常数自由度；绝不改变场的幅值。"""
    offset = float(np.mean(field[reference]))
    return field - offset, offset


def metrics(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    residual = pred[valid] - target[valid]
    pred_values, target_values = pred[valid], target[valid]
    corr = np.corrcoef(pred_values, target_values)[0, 1]
    return {
        'mae': float(np.mean(np.abs(residual))),
        'rmse': float(np.sqrt(np.mean(residual ** 2))),
        'bias': float(np.mean(residual)),
        'ncc': float(corr) if np.isfinite(corr) else float('nan'),
    }


def flow_ground_truth(path: Path, shape: tuple[int, int, int],
                      allow_legacy: bool = False) -> tuple[np.ndarray, dict]:
    """优先读取 MATLAB 直接导出的 flow；原始 n 重采样仅作显式兼容。"""
    if not path.is_file():
        raise FileNotFoundError(
            f'找不到精确 GT: {path}\n'
            '请先在 MATLAB 中运行 export_phantom1_flow_ground_truth，'
            '或使用 --skip-ground-truth 跳过 GT 指标。')

    values = sio.loadmat(path)
    flow_key = next((key for key in ('flow_gt', 'flow') if key in values), None)
    if flow_key is not None:
        flow = np.asarray(values[flow_key]).squeeze().astype(np.float32, copy=False)
        if flow.shape != shape:
            raise ValueError(f'精确 GT 尺寸应为 {shape}，实际为 {flow.shape}: {path}')
        info = {
            'source': 'exact MATLAB-preprocessed flow',
            'file': str(path.resolve()),
            'variable': flow_key,
        }
        if 'spacing' in values:
            info['spacing'] = np.asarray(values['spacing']).squeeze().astype(float).tolist()
        if 'flow_max' in values:
            info['flow_max'] = float(np.asarray(values['flow_max']).squeeze())
        if 'n0' in values:
            info['n0'] = float(np.asarray(values['n0']).squeeze())
        return flow, info

    if 'n' not in values:
        available = sorted(key for key in values if not key.startswith('__'))
        raise KeyError(f'{path} 不含 flow_gt、flow 或 n；实际变量: {available}')
    if not allow_legacy:
        raise RuntimeError(
            f'{path} 只包含原始折射率 n。为避免跨软件重采样误差，'
            '请运行 MATLAB 接口 export_phantom1_flow_ground_truth。'
            '如需复现旧结果，请显式添加 --allow-legacy-ground-truth。')

    warnings.warn(
        '正在使用 SciPy zoom(order=1) 近似 MATLAB imresize3；'
        '由此得到的 GT 指标仅用于兼容旧结果。', RuntimeWarning)
    n = np.asarray(values['n']).squeeze().astype(np.float32, copy=False)
    n_small = zoom(n, zoom=(0.5, 0.5, 0.5), order=1, prefilter=False)
    starts = [(size - target) // 2 for size, target in zip(n_small.shape, shape)]
    if any(start < 0 for start in starts):
        raise ValueError(f'GT 尺寸 {n_small.shape} 小于重建尺寸 {shape}')
    slices = tuple(slice(start, start + target) for start, target in zip(starts, shape))
    n_roi = n_small[slices]
    n0 = 296.15 * (1.00027 - 1) / 1100 + 1
    flow0 = n_roi / n0 - 1
    flow_max = abs(float(np.min(flow0)))
    flow = (flow0 / flow_max).astype(np.float32)
    return flow, {
        'source': 'legacy SciPy approximation of MATLAB imresize3',
        'file': str(path.resolve()),
        'variable': 'n',
        'flow_max': flow_max,
        'n0': n0,
    }


def gradient_diagnostics(gradient: np.ndarray, sigma: np.ndarray,
                         spacing: tuple[float, float, float], valid: np.ndarray) -> dict[str, float]:
    """报告梯度可积性，而不是把泊松平滑后的 σ 当作唯一结论。"""
    d_sigma = np.gradient(sigma, *spacing, edge_order=1)
    residual = np.stack(d_sigma, axis=-1) - gradient
    gx, gy, gz = (gradient[..., index] for index in range(3))
    dgx_dx, dgx_dy, dgx_dz = np.gradient(gx, *spacing, edge_order=1)
    dgy_dx, dgy_dy, dgy_dz = np.gradient(gy, *spacing, edge_order=1)
    dgz_dx, dgz_dy, dgz_dz = np.gradient(gz, *spacing, edge_order=1)
    curl = np.stack((dgz_dy - dgy_dz, dgx_dz - dgz_dx, dgy_dx - dgx_dy), axis=-1)
    return {
        'gradient_poisson_residual_rmse': float(np.sqrt(np.mean(residual[valid] ** 2))),
        'curl_rmse': float(np.sqrt(np.mean(curl[valid] ** 2))),
    }


def save_midplane(output_path: Path, baseline: np.ndarray, gradient: np.ndarray,
                  ground_truth: np.ndarray | None, valid: np.ndarray):
    z = baseline.shape[2] // 2
    fields = [('Baseline σ', baseline), ('Gradient → Poisson σ', gradient)]
    if ground_truth is not None:
        fields.append(('Ground truth σ', ground_truth))
    difference = gradient - baseline
    fields.append(('Gradient − baseline', difference))

    value_fields = [field for _, field in fields[:-1]]
    vmin = min(float(field[valid].min()) for field in value_fields)
    vmax = max(float(field[valid].max()) for field in value_fields)
    err_max = max(float(np.abs(difference[valid]).max()), 1e-8)
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5), constrained_layout=True)
    for axis, (title, field) in zip(np.atleast_1d(axes), fields):
        if title == 'Gradient − baseline':
            image = axis.imshow(field[:, :, z].T, origin='lower', cmap='RdBu_r',
                                vmin=-err_max, vmax=err_max)
        else:
            image = axis.imshow(field[:, :, z].T, origin='lower', cmap='viridis',
                                vmin=vmin, vmax=vmax)
        axis.set_title(f'{title}\nZ={z}')
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def show_slice_viewer(baseline: np.ndarray, gradient: np.ndarray,
                      ground_truth: np.ndarray | None):
    """显示四栏交互式切片视图，所有 σ 均已在同一 flow 空间对齐。"""
    if ground_truth is None:
        raise ValueError('交互式四栏视图需要 ground truth；请提供 --ground-truth。')

    difference = gradient - baseline
    common_min = min(float(baseline.min()), float(gradient.min()), float(ground_truth.min()))
    common_max = max(float(baseline.max()), float(gradient.max()), float(ground_truth.max()))
    error_max = max(float(np.abs(difference).max()), 1e-8)
    z_initial = baseline.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(19, 6))
    fig.subplots_adjust(bottom=0.18, wspace=0.42)
    fields = [
        ('Baseline σ', baseline, 'viridis', common_min, common_max),
        ('Gradient → Poisson σ', gradient, 'viridis', common_min, common_max),
        ('Ground truth σ', ground_truth, 'viridis', common_min, common_max),
        ('Gradient − baseline', difference, 'RdBu_r', -error_max, error_max),
    ]
    images = []
    for axis, (title, field, cmap, vmin, vmax) in zip(axes, fields):
        image = axis.imshow(field[:, :, z_initial].T, origin='lower', cmap=cmap,
                            vmin=vmin, vmax=vmax)
        axis.set_title(f'{title}\nZ={z_initial}')
        axis.set_xlabel('X')
        axis.set_ylabel('Y')
        fig.colorbar(image, ax=axis, shrink=0.78)
        images.append(image)

    slider_axis = fig.add_axes([0.16, 0.065, 0.68, 0.035])
    slider = Slider(slider_axis, 'Z slice', 0, baseline.shape[2] - 1,
                    valinit=z_initial, valstep=1)

    def update(z):
        z = int(z)
        for image, (_, field, _, _, _) in zip(images, fields):
            image.set_data(field[:, :, z].T)
        for axis, (title, _, _, _, _) in zip(axes, fields):
            axis.set_title(f'{title}\nZ={z}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def main():
    args = parse_args()
    baseline = load_field(args.baseline, 'sigmas0')
    gradient_data = sio.loadmat(args.gradient)
    if 'dsigmas_dxyz_auto0' not in gradient_data:
        raise KeyError(f'{args.gradient} 缺少直接预测梯度 dsigmas_dxyz_auto0')
    gradient = np.asarray(gradient_data['dsigmas_dxyz_auto0']).squeeze().astype(np.float32, copy=False)
    if baseline.ndim != 3 or gradient.shape != baseline.shape + (3,):
        raise ValueError(f'需要 σ 为 (H,W,D)、梯度为 (H,W,D,3)，实际为 {baseline.shape} 与 {gradient.shape}')

    # 重新反演，避免使用旧版本在 [-1, 1] 中归一化过的 sigmas0。
    gradient_sigma = poisson_reconstruct(torch.from_numpy(gradient), spacing=tuple(args.spacing),
                                         pad=args.pad, normalize=False).cpu().numpy()
    valid, valid_source = common_mask(args.mask, baseline.shape)
    finite = np.isfinite(baseline) & np.isfinite(gradient_sigma)
    valid &= finite
    reference = border_mask(baseline.shape) & valid
    if not np.any(reference):
        reference = valid

    baseline_aligned, baseline_offset = align_gauge(baseline, reference)
    gradient_aligned, gradient_offset = align_gauge(gradient_sigma, reference)
    report = {
        'space': 'flow = (n / n0 - 1) / flow_max',
        'baseline_file': str(args.baseline.resolve()),
        'gradient_file': str(args.gradient.resolve()),
        'experiment_manifests': {
            'baseline': load_experiment_manifest(args.baseline),
            'gradient_output': load_experiment_manifest(args.gradient),
        },
        'spacing': list(args.spacing),
        'evaluation_mask': valid_source,
        'gauge_reference': '3-voxel ROI border',
        'baseline_offset_removed': baseline_offset,
        'gradient_offset_removed': gradient_offset,
        'baseline_vs_gradient': metrics(baseline_aligned, gradient_aligned, valid),
        'gradient_integrability': gradient_diagnostics(gradient, gradient_aligned, tuple(args.spacing), valid),
    }

    ground_truth = None
    if not args.skip_ground_truth:
        ground_truth, gt_info = flow_ground_truth(
            args.ground_truth, baseline.shape,
            allow_legacy=args.allow_legacy_ground_truth)
        if 'spacing' in gt_info and not np.allclose(
                np.asarray(gt_info['spacing']), np.asarray(args.spacing),
                rtol=1e-6, atol=1e-8):
            raise ValueError(
                f'MATLAB GT spacing {gt_info["spacing"]} 与比较参数 '
                f'{list(args.spacing)} 不一致；请统一坐标空间后再比较。')
        ground_truth, gt_offset = align_gauge(ground_truth, reference)
        report['ground_truth'] = gt_info
        report['ground_truth_offset_removed'] = gt_offset
        report['baseline_vs_ground_truth'] = metrics(baseline_aligned, ground_truth, valid)
        report['gradient_vs_ground_truth'] = metrics(gradient_aligned, ground_truth, valid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved_fields = {
        'baseline': baseline_aligned,
        'gradient_poisson': gradient_aligned,
        'gradient_direct': gradient,
        'valid_mask': valid,
    }
    if ground_truth is not None:
        saved_fields['ground_truth'] = ground_truth
    np.savez_compressed(args.output_dir / 'flow_fields_aligned.npz', **saved_fields)
    with (args.output_dir / 'comparison_report.json').open('w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=True)
    save_midplane(args.output_dir / 'comparison_midplane.png', baseline_aligned,
                   gradient_aligned, ground_truth, valid)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'已输出统一 flow 空间的比较结果：{args.output_dir}')
    if not args.no_show:
        show_slice_viewer(baseline_aligned, gradient_aligned, ground_truth)


if __name__ == '__main__':
    main()
