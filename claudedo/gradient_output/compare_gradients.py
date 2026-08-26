"""在统一的 ∇flow 空间中比较双基线梯度、梯度版和 MATLAB ground truth。

共同标量空间：
    flow = (n / n0 - 1) / flow_max
共同梯度空间：
    g = ∇_(x',y',z') flow

基线版同时评价两种梯度：从公共规则网格 ``sigmas0`` 统一有限差分得到的
``baseline_grid_fd``，以及导出的连续网络 autograd 梯度 ``baseline_auto``。
梯度版的 ``dsigmas_dxyz_auto0`` 则是网络直接输出。四个向量场已经位于相同的
NeRF 坐标和 flow 幅值空间，因此本脚本不做 gauge 对齐、min-max、符号翻转、
轴交换或幅值拟合。正式体素场主基线是 ``baseline_grid_fd``；
``baseline_auto`` 用于连续模型和采样分辨率诊断。

示例：
    python compare_gradients.py --no-show
    python compare_gradients.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_erosion
import torch

from compare_reconstructions import (
    common_mask,
    flow_ground_truth,
    load_experiment_manifest,
)
from nerf.poisson_solver import poisson_reconstruct


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
COMPONENT_NAMES = ('gx', 'gy', 'gz')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            '在统一的 flow 梯度空间中比较双基线、梯度版与 MATLAB ground truth。'))
    parser.add_argument(
        '--baseline', type=Path,
        default=PROJECT_ROOT / 'PYTHON/NIR-BOS/result/phantom 1/test/results/sigmas0.mat',
        help='基线版 sigmas0.mat（必须同时包含 sigmas0 和 autograd 梯度）')
    parser.add_argument(
        '--gradient', type=Path,
        default=SCRIPT_DIR / 'result/gradient_test/results/sigmas0.mat',
        help='梯度版 sigmas0.mat（读取网络直接输出的梯度）')
    parser.add_argument('--baseline-key', default='dsigmas_dxyz_auto0',
                        help='基线版 MAT 中的三分量梯度变量名')
    parser.add_argument('--gradient-key', default='dsigmas_dxyz_auto0',
                        help='梯度版 MAT 中的三分量梯度变量名')
    parser.add_argument(
        '--primary-baseline-source', '--baseline-source',
        dest='primary_baseline_source',
        choices=('exported', 'grid-finite-difference'),
        default='grid-finite-difference',
        help=('选择兼容字段 baseline_gradient 使用哪种主基线；两种基线始终都会评价。'
              '旧参数名 --baseline-source 继续可用'))
    parser.add_argument(
        '--ground-truth', type=Path,
        default=PROJECT_ROOT / 'MATLAB/Test_data/Phantom 1/140x294x140/flow_ground_truth.mat',
        help='由 export_phantom1_flow_ground_truth.m 导出的精确 flow GT')
    parser.add_argument('--skip-ground-truth', action='store_true',
                        help='没有 GT 时仅比较基线版和梯度版')
    parser.add_argument('--allow-legacy-ground-truth', action='store_true',
                        help='显式允许从原始 n 用 SciPy 近似 MATLAB imresize3')
    parser.add_argument(
        '--mask', type=Path,
        default=PROJECT_ROOT / 'PYTHON/NIR-BOS/data/Phantom 1/140x294x140/3Dmask.mat',
        help='共同评价区域的 3Dmask.mat')
    parser.add_argument('--spacing', type=float, nargs=3,
                        default=(0.01360525, 0.01360525, 0.01360525),
                        metavar=('DX', 'DY', 'DZ'),
                        help='NeRF 坐标系中的体素间距')
    parser.add_argument('--erosion', type=int, default=2,
                        help='评价 mask 向内腐蚀层数，排除硬 mask 和差分边界伪影')
    parser.add_argument('--active-ratio', type=float, default=0.05,
                        help='有效方向区域阈值：ratio × GT 梯度幅值的 P99')
    parser.add_argument('--pad', type=int, default=8,
                        help='泊松可积性诊断的反射填充宽度')
    parser.add_argument('--skip-poisson-diagnostics', action='store_true',
                        help='仅计算 curl，跳过较耗时的泊松投影残差')
    parser.add_argument('--skip-field-save', action='store_true',
                        help='不保存体积较大的 gradient_fields.npz')
    parser.add_argument(
        '--output-dir', type=Path,
        default=SCRIPT_DIR / 'result/gradient_test/comparison_gradient_exact',
        help='JSON、CSV、NPZ 和切片图输出目录')
    parser.add_argument('--no-show', action='store_true',
                        help='不弹出交互式切片窗口')
    parser.add_argument('--self-test', action='store_true',
                        help='运行解析场数值自检，不读取项目结果')
    return parser.parse_args()


def load_mat_field(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'找不到输入文件: {path}')
    values = sio.loadmat(path)
    if key not in values:
        available = sorted(name for name in values if not name.startswith('__'))
        raise KeyError(f'{path} 不包含 {key!r}；实际变量: {available}')
    return np.asarray(values[key]).squeeze().astype(np.float32, copy=False)


def validate_scalar(field: np.ndarray, name: str):
    if field.ndim != 3:
        raise ValueError(f'{name} 应为三维标量场，实际 shape={field.shape}')


def validate_vector(field: np.ndarray, name: str):
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(f'{name} 应为 (H,W,D,3)，实际 shape={field.shape}')


def spatial_gradient(field: np.ndarray,
                     spacing: tuple[float, float, float]) -> np.ndarray:
    """使用同一二阶边界差分算子从规则网格标量场构造三分量梯度。"""
    validate_scalar(field, 'scalar field')
    components = np.gradient(field, *spacing, edge_order=2)
    return np.stack(components, axis=-1).astype(np.float32, copy=False)


def load_baseline_gradients(
        path: Path, autograd_key: str,
        spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """一次读取基线 MAT，并同时返回公共网格差分和连续 autograd 梯度。"""
    if not path.is_file():
        raise FileNotFoundError(f'找不到输入文件: {path}')
    values = sio.loadmat(path)
    required = ('sigmas0', autograd_key)
    missing = [key for key in required if key not in values]
    if missing:
        available = sorted(name for name in values if not name.startswith('__'))
        raise KeyError(f'{path} 缺少 {missing}；实际变量: {available}')

    sigma = np.asarray(values['sigmas0']).squeeze().astype(np.float32, copy=False)
    baseline_auto = np.asarray(values[autograd_key]).squeeze().astype(
        np.float32, copy=False)
    validate_scalar(sigma, 'baseline sigmas0')
    validate_vector(baseline_auto, 'baseline autograd gradient')
    if baseline_auto.shape != sigma.shape + (3,):
        raise ValueError(
            '基线 sigmas0 和 autograd 梯度 shape 不一致: '
            f'{sigma.shape} vs {baseline_auto.shape}')

    baseline_grid_fd = spatial_gradient(sigma, spacing)
    information = {
        'file': str(path.resolve()),
        'grid_finite_difference': {
            'source': 'finite difference of exported baseline sigmas0',
            'variable': 'sigmas0',
            'operator': 'numpy.gradient(edge_order=2)',
            'spacing': list(spacing),
            'role': 'primary voxel-field baseline',
        },
        'autograd': {
            'source': 'exported autograd gradient of baseline scalar network',
            'variable': autograd_key,
            'role': 'continuous-model diagnostic',
        },
    }
    return baseline_grid_fd, baseline_auto, information


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if first.size < 2 or float(np.std(first)) == 0 or float(np.std(second)) == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def scalar_metrics(prediction: np.ndarray, target: np.ndarray,
                   valid: np.ndarray) -> dict:
    pred_values = prediction[valid].astype(np.float64, copy=False)
    target_values = target[valid].astype(np.float64, copy=False)
    residual = pred_values - target_values
    target_rms = float(np.sqrt(np.mean(target_values ** 2)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return {
        'mae': float(np.mean(np.abs(residual))),
        'rmse': rmse,
        'bias': float(np.mean(residual)),
        'ncc': safe_correlation(pred_values, target_values),
        'nrmse_by_target_rms': rmse / target_rms if target_rms > 0 else None,
    }


def direction_metrics(prediction: np.ndarray, target: np.ndarray,
                      active: np.ndarray) -> dict:
    pred = prediction[active].astype(np.float64, copy=False)
    truth = target[active].astype(np.float64, copy=False)
    if pred.shape[0] == 0:
        return {'voxel_count': 0}

    pred_norm = np.linalg.norm(pred, axis=-1)
    truth_norm = np.linalg.norm(truth, axis=-1)
    denominator = pred_norm * truth_norm
    cosine = np.zeros_like(denominator)
    np.divide(np.sum(pred * truth, axis=-1), denominator,
              out=cosine, where=denominator > 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return {
        'voxel_count': int(angle.size),
        'mean_cosine': float(np.mean(cosine)),
        'median_cosine': float(np.median(cosine)),
        'mean_angle_deg': float(np.mean(angle)),
        'median_angle_deg': float(np.median(angle)),
        'p95_angle_deg': float(np.percentile(angle, 95)),
        'fraction_angle_le_10_deg': float(np.mean(angle <= 10)),
        'fraction_angle_le_20_deg': float(np.mean(angle <= 20)),
        'fraction_angle_le_30_deg': float(np.mean(angle <= 30)),
    }


def vector_metrics(prediction: np.ndarray, target: np.ndarray,
                   valid: np.ndarray, active: np.ndarray) -> dict:
    residual = prediction[valid].astype(np.float64, copy=False) - \
        target[valid].astype(np.float64, copy=False)
    residual_norm = np.linalg.norm(residual, axis=-1)
    target_values = target[valid].astype(np.float64, copy=False)
    target_energy = float(np.sum(target_values ** 2))
    residual_energy = float(np.sum(residual ** 2))

    component_report = {}
    for index, name in enumerate(COMPONENT_NAMES):
        component_report[name] = scalar_metrics(
            prediction[..., index], target[..., index], valid)

    pred_magnitude = np.linalg.norm(prediction, axis=-1)
    target_magnitude = np.linalg.norm(target, axis=-1)
    return {
        'voxel_count': int(np.count_nonzero(valid)),
        'components': component_report,
        'vector': {
            'mae_l2': float(np.mean(residual_norm)),
            'rmse_l2': float(np.sqrt(np.mean(residual_norm ** 2))),
            'relative_l2': (
                float(np.sqrt(residual_energy / target_energy))
                if target_energy > 0 else None),
        },
        'magnitude': scalar_metrics(pred_magnitude, target_magnitude, valid),
        'direction': direction_metrics(prediction, target, active),
    }


def prepare_masks(mask_path: Path, shape: tuple[int, int, int],
                  fields: list[np.ndarray], reference_gradient: np.ndarray,
                  reference_name: str, erosion: int,
                  active_ratio: float) -> tuple[np.ndarray, np.ndarray, dict]:
    if erosion < 0:
        raise ValueError('--erosion 必须大于等于 0')
    if active_ratio < 0:
        raise ValueError('--active-ratio 必须大于等于 0')

    base_mask, mask_source = common_mask(mask_path, shape)
    core = (binary_erosion(base_mask, iterations=erosion)
            if erosion > 0 else base_mask.copy())
    finite = np.ones(shape, dtype=bool)
    for field in fields:
        finite &= np.all(np.isfinite(field), axis=-1)
    valid = core & finite
    if not np.any(valid):
        raise ValueError('腐蚀、有限值过滤后没有可用于评价的体素')

    magnitude = np.linalg.norm(reference_gradient, axis=-1)
    robust_peak = float(np.percentile(magnitude[valid], 99))
    threshold = active_ratio * robust_peak
    active = valid & (magnitude > max(threshold, 1e-12))
    if not np.any(active):
        raise ValueError('active-gradient 区域为空；请检查 GT 或降低 --active-ratio')

    return valid, active, {
        'source': mask_source,
        'erosion_iterations': erosion,
        'mask_voxel_count_before_erosion': int(np.count_nonzero(base_mask)),
        'valid_core_voxel_count': int(np.count_nonzero(valid)),
        'active_reference': reference_name,
        'active_ratio_of_p99': active_ratio,
        'reference_magnitude_p99': robust_peak,
        'active_threshold': threshold,
        'active_voxel_count': int(np.count_nonzero(active)),
    }


def curl_field(gradient: np.ndarray,
               spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx, gy, gz = (gradient[..., index] for index in range(3))
    curl_x = (np.gradient(gz, spacing[1], axis=1, edge_order=2) -
              np.gradient(gy, spacing[2], axis=2, edge_order=2))
    curl_y = (np.gradient(gx, spacing[2], axis=2, edge_order=2) -
              np.gradient(gz, spacing[0], axis=0, edge_order=2))
    curl_z = (np.gradient(gy, spacing[0], axis=0, edge_order=2) -
              np.gradient(gx, spacing[1], axis=1, edge_order=2))
    return (curl_x.astype(np.float32, copy=False),
            curl_y.astype(np.float32, copy=False),
            curl_z.astype(np.float32, copy=False))


def integrability_metrics(gradient: np.ndarray,
                          spacing: tuple[float, float, float],
                          valid: np.ndarray, pad: int,
                          skip_poisson: bool) -> tuple[dict, np.ndarray]:
    curl_x, curl_y, curl_z = curl_field(gradient, spacing)
    curl_squared = curl_x ** 2 + curl_y ** 2 + curl_z ** 2
    curl_magnitude = np.sqrt(curl_squared).astype(np.float32, copy=False)
    gradient_magnitude = np.linalg.norm(gradient, axis=-1)
    curl_rms_norm = float(np.sqrt(np.mean(curl_squared[valid])))
    gradient_rms_norm = float(np.sqrt(np.mean(gradient_magnitude[valid] ** 2)))
    characteristic_spacing = float(np.mean(spacing))
    report = {
        'curl_rmse_per_component': float(np.sqrt(np.mean(curl_squared[valid]) / 3)),
        'curl_rms_norm': curl_rms_norm,
        'max_curl_norm': float(np.max(curl_magnitude[valid])),
        'curl_relative': (
            curl_rms_norm / (gradient_rms_norm / characteristic_spacing)
            if gradient_rms_norm > 0 else None),
    }

    if skip_poisson:
        report['poisson_projection'] = None
        return report, curl_magnitude

    sigma = poisson_reconstruct(
        torch.from_numpy(np.ascontiguousarray(gradient)),
        spacing=spacing, pad=pad, normalize=False).cpu().numpy()
    projected = spatial_gradient(sigma, spacing)
    np.subtract(projected, gradient, out=projected)
    residual = projected[valid].astype(np.float64, copy=False)
    gradient_values = gradient[valid].astype(np.float64, copy=False)
    residual_energy = float(np.sum(residual ** 2))
    gradient_energy = float(np.sum(gradient_values ** 2))
    residual_norm = np.linalg.norm(residual, axis=-1)
    report['poisson_projection'] = {
        'rmse_l2': float(np.sqrt(np.mean(residual_norm ** 2))),
        'relative_l2': (
            float(np.sqrt(residual_energy / gradient_energy))
            if gradient_energy > 0 else None),
    }
    return report, curl_magnitude


def robust_abs_limit(fields: list[np.ndarray], valid: np.ndarray,
                     percentile: float = 99.0) -> float:
    limits = [float(np.percentile(np.abs(field[valid]), percentile)) for field in fields]
    return max(max(limits), 1e-8)


def masked_slice(field: np.ndarray, valid: np.ndarray, z_index: int) -> np.ndarray:
    values = field[:, :, z_index].T.copy()
    values[~valid[:, :, z_index].T] = np.nan
    return values


def save_component_figure(path: Path, baseline_grid_fd: np.ndarray,
                          baseline_auto: np.ndarray, gradient: np.ndarray,
                          ground_truth: np.ndarray | None, valid: np.ndarray):
    z_index = baseline_grid_fd.shape[2] // 2
    if ground_truth is None:
        columns = [
            ('Baseline grid-FD', baseline_grid_fd, False),
            ('Baseline autograd', baseline_auto, False),
            ('Gradient output', gradient, False),
            ('Autograd - grid-FD', baseline_auto - baseline_grid_fd, True),
            ('Gradient - grid-FD', gradient - baseline_grid_fd, True),
        ]
    else:
        columns = [
            ('Baseline grid-FD', baseline_grid_fd, False),
            ('Baseline autograd', baseline_auto, False),
            ('Gradient output', gradient, False),
            ('Ground truth', ground_truth, False),
            ('Grid-FD - GT', baseline_grid_fd - ground_truth, True),
            ('Autograd - GT', baseline_auto - ground_truth, True),
            ('Gradient - GT', gradient - ground_truth, True),
        ]

    fig, axes = plt.subplots(3, len(columns), figsize=(4.2 * len(columns), 11),
                             constrained_layout=True)
    value_fields = [baseline_grid_fd, baseline_auto, gradient] + (
        [] if ground_truth is None else [ground_truth])
    error_fields = [entry[1] for entry in columns if entry[2]]
    for component, component_name in enumerate(COMPONENT_NAMES):
        value_limit = robust_abs_limit(
            [field[..., component] for field in value_fields], valid)
        error_limit = robust_abs_limit(
            [field[..., component] for field in error_fields], valid)
        for column, (title, field, is_error) in enumerate(columns):
            limit = error_limit if is_error else value_limit
            image = axes[component, column].imshow(
                masked_slice(field[..., component], valid, z_index),
                origin='lower', cmap='RdBu_r', vmin=-limit, vmax=limit)
            axes[component, column].set_title(f'{title} {component_name}\nZ={z_index}')
            fig.colorbar(image, ax=axes[component, column], shrink=0.72)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_magnitude_figure(path: Path, baseline_grid_fd: np.ndarray,
                          baseline_auto: np.ndarray, gradient: np.ndarray,
                          ground_truth: np.ndarray | None, valid: np.ndarray):
    z_index = baseline_grid_fd.shape[2] // 2
    grid_fd_mag = np.linalg.norm(baseline_grid_fd, axis=-1)
    auto_mag = np.linalg.norm(baseline_auto, axis=-1)
    gradient_mag = np.linalg.norm(gradient, axis=-1)
    if ground_truth is None:
        columns = [
            ('|Baseline grid-FD|', grid_fd_mag, False),
            ('|Baseline autograd|', auto_mag, False),
            ('|Gradient output|', gradient_mag, False),
            ('|Autograd| - |grid-FD|', auto_mag - grid_fd_mag, True),
            ('|Gradient| - |grid-FD|', gradient_mag - grid_fd_mag, True),
        ]
        value_fields = [grid_fd_mag, auto_mag, gradient_mag]
    else:
        truth_mag = np.linalg.norm(ground_truth, axis=-1)
        columns = [
            ('|Baseline grid-FD|', grid_fd_mag, False),
            ('|Baseline autograd|', auto_mag, False),
            ('|Gradient output|', gradient_mag, False),
            ('|Ground truth|', truth_mag, False),
            ('|Grid-FD| - |GT|', grid_fd_mag - truth_mag, True),
            ('|Autograd| - |GT|', auto_mag - truth_mag, True),
            ('|Gradient| - |GT|', gradient_mag - truth_mag, True),
        ]
        value_fields = [grid_fd_mag, auto_mag, gradient_mag, truth_mag]

    value_max = max(float(np.percentile(field[valid], 99)) for field in value_fields)
    error_limit = robust_abs_limit([field for _, field, is_error in columns if is_error], valid)
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 4.5),
                             constrained_layout=True)
    for axis, (title, field, is_error) in zip(np.atleast_1d(axes), columns):
        if is_error:
            image = axis.imshow(masked_slice(field, valid, z_index), origin='lower',
                                cmap='RdBu_r', vmin=-error_limit, vmax=error_limit)
        else:
            image = axis.imshow(masked_slice(field, valid, z_index), origin='lower',
                                cmap='viridis', vmin=0, vmax=max(value_max, 1e-8))
        axis.set_title(f'{title}\nZ={z_index}')
        fig.colorbar(image, ax=axis, shrink=0.72)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def angular_error_map(prediction: np.ndarray, target: np.ndarray,
                      active: np.ndarray) -> np.ndarray:
    dot = np.sum(prediction * target, axis=-1)
    denominator = np.linalg.norm(prediction, axis=-1) * np.linalg.norm(target, axis=-1)
    cosine = np.zeros_like(dot)
    np.divide(dot, denominator, out=cosine, where=denominator > 1e-12)
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).astype(np.float32)
    angle[~active] = np.nan
    return angle


def save_angle_figure(path: Path, baseline_grid_fd: np.ndarray,
                      baseline_auto: np.ndarray, gradient: np.ndarray,
                      ground_truth: np.ndarray | None, active: np.ndarray):
    z_index = baseline_grid_fd.shape[2] // 2
    if ground_truth is None:
        maps = [
            ('Autograd vs grid-FD', angular_error_map(
                baseline_auto, baseline_grid_fd, active)),
            ('Gradient vs grid-FD', angular_error_map(
                gradient, baseline_grid_fd, active)),
            ('Gradient vs autograd', angular_error_map(
                gradient, baseline_auto, active)),
        ]
    else:
        maps = [
            ('Grid-FD vs GT', angular_error_map(
                baseline_grid_fd, ground_truth, active)),
            ('Autograd vs GT', angular_error_map(
                baseline_auto, ground_truth, active)),
            ('Gradient vs GT', angular_error_map(gradient, ground_truth, active)),
            ('Autograd vs grid-FD', angular_error_map(
                baseline_auto, baseline_grid_fd, active)),
            ('Gradient vs grid-FD', angular_error_map(
                gradient, baseline_grid_fd, active)),
        ]
    finite_values = [field[np.isfinite(field)] for _, field in maps]
    angle_max = max(
        max(float(np.percentile(values, 95)), 1.0) if values.size else 1.0
        for values in finite_values)
    angle_max = min(angle_max, 180.0)
    fig, axes = plt.subplots(1, len(maps), figsize=(5 * len(maps), 4.8),
                             constrained_layout=True)
    for axis, (title, field) in zip(np.atleast_1d(axes), maps):
        image = axis.imshow(field[:, :, z_index].T, origin='lower', cmap='magma',
                            vmin=0, vmax=angle_max)
        axis.set_title(f'{title} angle (deg)\nZ={z_index}')
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_integrability_figure(path: Path, curl_maps: dict[str, np.ndarray],
                              valid: np.ndarray):
    z_index = valid.shape[2] // 2
    limit = max(float(np.percentile(values[valid], 99)) for values in curl_maps.values())
    fig, axes = plt.subplots(1, len(curl_maps), figsize=(5 * len(curl_maps), 4.8),
                             constrained_layout=True)
    for axis, (title, field) in zip(np.atleast_1d(axes), curl_maps.items()):
        image = axis.imshow(masked_slice(field, valid, z_index), origin='lower',
                            cmap='inferno', vmin=0, vmax=max(limit, 1e-8))
        axis.set_title(f'{title} |curl(g)|\nZ={z_index}')
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def show_slice_viewer(baseline_grid_fd: np.ndarray, baseline_auto: np.ndarray,
                      gradient: np.ndarray, ground_truth: np.ndarray | None,
                      valid: np.ndarray):
    fields = [
        ('Baseline grid-FD', baseline_grid_fd, False),
        ('Baseline autograd', baseline_auto, False),
        ('Gradient output', gradient, False),
    ]
    if ground_truth is not None:
        fields.append(('Ground truth', ground_truth, False))
    fields.extend([
        ('Autograd - grid-FD', baseline_auto - baseline_grid_fd, True),
        ('Gradient - grid-FD', gradient - baseline_grid_fd, True),
    ])

    z_initial = baseline_grid_fd.shape[2] // 2
    fig, axes = plt.subplots(1, len(fields), figsize=(4.6 * len(fields), 6))
    fig.subplots_adjust(left=0.12, bottom=0.22, wspace=0.42)
    selected = {'name': 'gx'}

    def field_component(field: np.ndarray, name: str) -> np.ndarray:
        if name == '|g|':
            return np.linalg.norm(field, axis=-1)
        return field[..., COMPONENT_NAMES.index(name)]

    images = []
    for axis, (title, field, _) in zip(axes, fields):
        values = field_component(field, selected['name'])
        image = axis.imshow(masked_slice(values, valid, z_initial), origin='lower',
                            cmap='RdBu_r')
        axis.set_title(f'{title} {selected["name"]}\nZ={z_initial}')
        fig.colorbar(image, ax=axis, shrink=0.72)
        images.append(image)

    slider_axis = fig.add_axes([0.22, 0.08, 0.68, 0.035])
    slider = Slider(slider_axis, 'Z slice', 0, baseline_grid_fd.shape[2] - 1,
                    valinit=z_initial, valstep=1)
    radio_axis = fig.add_axes([0.015, 0.035, 0.075, 0.13])
    radio = RadioButtons(radio_axis, (*COMPONENT_NAMES, '|g|'), active=0)

    def update(_):
        z_index = int(slider.val)
        display_fields = []
        for _, field, _ in fields:
            display_fields.append(field_component(field, selected['name']))
        value_fields = [values for values, (_, _, is_error) in zip(display_fields, fields)
                        if not is_error]
        error_fields = [values for values, (_, _, is_error) in zip(display_fields, fields)
                        if is_error]
        value_limit = robust_abs_limit(value_fields, valid)
        difference_limit = robust_abs_limit(error_fields, valid)
        for index, (image, axis, (title, _, is_error)) in enumerate(
                zip(images, axes, fields)):
            values = display_fields[index]
            image.set_data(masked_slice(values, valid, z_index))
            if selected['name'] == '|g|' and not is_error:
                image.set_cmap('viridis')
                image.set_clim(0, value_limit)
            elif selected['name'] == '|g|' and is_error:
                image.set_cmap('magma')
                image.set_clim(0, difference_limit)
            else:
                image.set_cmap('RdBu_r')
                limit = difference_limit if is_error else value_limit
                image.set_clim(-limit, limit)
            axis.set_title(f'{title} {selected["name"]}\nZ={z_index}')
        fig.canvas.draw_idle()

    def select_component(label):
        selected['name'] = label
        update(None)

    slider.on_changed(update)
    radio.on_clicked(select_component)
    update(None)
    plt.show()


def flatten_report(values: dict, prefix: str = ''):
    for key, value in values.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            yield from flatten_report(value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, value


def run_self_test():
    spacing = (0.1, 0.08, 0.12)
    x = np.arange(25, dtype=np.float32) * spacing[0] - 1.2
    y = np.arange(27, dtype=np.float32) * spacing[1] - 1.04
    z = np.arange(23, dtype=np.float32) * spacing[2] - 1.32
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    sigma = xx ** 2 + 2 * yy ** 2 + 3 * zz ** 2
    expected = np.stack((2 * xx, 4 * yy, 6 * zz), axis=-1)
    calculated = spatial_gradient(sigma, spacing)
    gradient_error = float(np.max(np.abs(calculated - expected)))
    if gradient_error > 1e-4:
        raise AssertionError(f'解析梯度误差过大: {gradient_error}')

    valid = np.ones(sigma.shape, dtype=bool)
    active = valid & (np.linalg.norm(expected, axis=-1) > 1e-6)
    identical = vector_metrics(expected, expected, valid, active)
    if identical['vector']['rmse_l2'] != 0 or identical['direction']['mean_angle_deg'] > 1e-6:
        raise AssertionError('相同向量场的指标自检失败')

    curl_x, curl_y, curl_z = curl_field(expected, spacing)
    curl_error = float(max(np.max(np.abs(curl_x)), np.max(np.abs(curl_y)),
                           np.max(np.abs(curl_z))))
    if curl_error > 1e-4:
        raise AssertionError(f'无旋场 curl 自检失败: {curl_error}')

    rotational = np.stack((-yy, xx, np.zeros_like(xx)), axis=-1)
    _, _, rotational_curl_z = curl_field(rotational, spacing)
    rotational_error = float(np.max(np.abs(rotational_curl_z - 2)))
    if rotational_error > 1e-4:
        raise AssertionError(f'旋转场 curl 自检失败: {rotational_error}')

    shifted = spatial_gradient(sigma + 17.0, spacing)
    gauge_error = float(np.max(np.abs(shifted - calculated)))
    if gauge_error > 1e-3:
        raise AssertionError(f'梯度常数不变性自检失败: {gauge_error}')
    print(json.dumps({
        'status': 'passed',
        'analytic_gradient_max_abs_error': gradient_error,
        'curl_free_max_abs_error': curl_error,
        'rotational_curl_max_abs_error': rotational_error,
        'constant_shift_gradient_max_abs_error': gauge_error,
    }, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    spacing = tuple(float(value) for value in args.spacing)
    if any(value <= 0 for value in spacing):
        raise ValueError('--spacing 的三个值都必须大于 0')

    baseline_grid_fd, baseline_auto, baseline_info = load_baseline_gradients(
        args.baseline, args.baseline_key, spacing)
    gradient = load_mat_field(args.gradient, args.gradient_key)
    validate_vector(gradient, 'gradient-output gradient')
    if baseline_grid_fd.shape != gradient.shape:
        raise ValueError(
            '基线和梯度版 shape 不一致: '
            f'{baseline_grid_fd.shape} vs {gradient.shape}')
    shape = baseline_grid_fd.shape[:3]

    if args.primary_baseline_source == 'exported':
        primary_baseline = baseline_auto
        primary_baseline_name = 'baseline_auto'
    else:
        primary_baseline = baseline_grid_fd
        primary_baseline_name = 'baseline_grid_fd'

    ground_truth = None
    ground_truth_flow = None
    ground_truth_info = None
    fields = [baseline_grid_fd, baseline_auto, gradient]
    if not args.skip_ground_truth:
        ground_truth_flow, ground_truth_info = flow_ground_truth(
            args.ground_truth, shape,
            allow_legacy=args.allow_legacy_ground_truth)
        if 'spacing' in ground_truth_info and not np.allclose(
                np.asarray(ground_truth_info['spacing']), np.asarray(spacing),
                rtol=1e-6, atol=1e-8):
            raise ValueError(
                f'MATLAB GT spacing {ground_truth_info["spacing"]} 与比较参数 '
                f'{list(spacing)} 不一致')
        ground_truth = spatial_gradient(ground_truth_flow, spacing)
        fields.append(ground_truth)

    reference_gradient = (
        ground_truth if ground_truth is not None else baseline_grid_fd)
    reference_name = (
        'ground_truth' if ground_truth is not None else 'baseline_grid_fd')
    valid, active, mask_info = prepare_masks(
        args.mask, shape, fields, reference_gradient,
        reference_name, args.erosion, args.active_ratio)

    pairwise_metrics = {
        'baseline_auto_vs_grid_fd': vector_metrics(
            baseline_auto, baseline_grid_fd, valid, active),
        'gradient_output_vs_baseline_grid_fd': vector_metrics(
            gradient, baseline_grid_fd, valid, active),
        'gradient_output_vs_baseline_auto': vector_metrics(
            gradient, baseline_auto, valid, active),
    }

    report = {
        'space': 'gradient of flow = (n / n0 - 1) / flow_max in scaled NeRF coordinates',
        'normalization': 'none (no gauge, min-max, sign/axis transform, or amplitude fitting)',
        'shape': list(shape),
        'component_order': list(COMPONENT_NAMES),
        'spacing': list(spacing),
        'comparison_policy': {
            'primary_baseline': primary_baseline_name,
            'primary_reason': (
                'voxel-field accuracy uses the same public-grid finite-difference '
                'operator as ground truth'),
            'baseline_auto_role': 'continuous-model diagnostic',
            'both_baselines_always_evaluated': True,
        },
        'baseline': baseline_info,
        'gradient_output': {
            'source': 'direct output of gradient network',
            'file': str(args.gradient.resolve()),
            'variable': args.gradient_key,
        },
        'experiment_manifests': {
            'baseline': load_experiment_manifest(args.baseline),
            'gradient_output': load_experiment_manifest(args.gradient),
        },
        'mask': mask_info,
        'pairwise': pairwise_metrics,
        # 保留旧字段，目标是旧脚本的 gradient；正式报告请使用 pairwise。
        'baseline_vs_gradient': vector_metrics(
            primary_baseline, gradient, valid, active),
    }
    if ground_truth is not None:
        vs_ground_truth = {
            'baseline_grid_fd': vector_metrics(
                baseline_grid_fd, ground_truth, valid, active),
            'baseline_auto': vector_metrics(
                baseline_auto, ground_truth, valid, active),
            'gradient_output': vector_metrics(
                gradient, ground_truth, valid, active),
        }
        report['ground_truth'] = ground_truth_info
        report['vs_ground_truth'] = vs_ground_truth
        # 保留旧字段，其 baseline 含义由 comparison_policy.primary_baseline 指明。
        report['baseline_vs_ground_truth'] = vs_ground_truth[primary_baseline_name]
        report['gradient_vs_ground_truth'] = vs_ground_truth['gradient_output']

    integrability_report = {}
    curl_maps = {}
    named_fields = [
        ('baseline_grid_fd', baseline_grid_fd),
        ('baseline_auto', baseline_auto),
        ('gradient_output', gradient),
    ]
    if ground_truth is not None:
        named_fields.append(('ground_truth', ground_truth))
    for name, field in named_fields:
        print(f'[integrability] {name} ...', flush=True)
        diagnostics, curl_magnitude = integrability_metrics(
            field, spacing, valid, args.pad, args.skip_poisson_diagnostics)
        integrability_report[name] = diagnostics
        curl_maps[name] = curl_magnitude
    report['integrability'] = integrability_report

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'gradient_comparison_report.json').open(
            'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
    with (args.output_dir / 'gradient_metrics.csv').open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(('metric', 'value'))
        writer.writerows(flatten_report(report))

    if not args.skip_field_save:
        saved_fields = {
            'baseline_grid_fd': baseline_grid_fd,
            'baseline_auto': baseline_auto,
            # 兼容阶段 01 的字段名；具体来源记录在 comparison_policy。
            'baseline_gradient': primary_baseline,
            'gradient_output': gradient,
            'valid_core': valid,
            'active_gradient': active,
        }
        if ground_truth is not None:
            saved_fields['ground_truth_gradient'] = ground_truth
            saved_fields['ground_truth_flow'] = ground_truth_flow
        np.savez_compressed(args.output_dir / 'gradient_fields.npz', **saved_fields)

    save_component_figure(
        args.output_dir / 'components_midplane.png',
        baseline_grid_fd, baseline_auto, gradient, ground_truth, valid)
    save_magnitude_figure(
        args.output_dir / 'magnitude_midplane.png',
        baseline_grid_fd, baseline_auto, gradient, ground_truth, valid)
    save_angle_figure(
        args.output_dir / 'angular_error_midplane.png',
        baseline_grid_fd, baseline_auto, gradient, ground_truth, active)
    save_integrability_figure(
        args.output_dir / 'integrability_midplane.png', curl_maps, valid)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'已输出统一梯度空间比较结果：{args.output_dir}')
    if not args.no_show:
        show_slice_viewer(
            baseline_grid_fd, baseline_auto, gradient, ground_truth, valid)


if __name__ == '__main__':
    main()
