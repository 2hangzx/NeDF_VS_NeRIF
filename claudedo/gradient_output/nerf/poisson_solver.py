"""
3D FFT 泊松求解器：从折射率梯度 ∇σ 重建标量场 σ

方程：∇²σ = div(g)
边界条件：Neumann (∂σ/∂n = 0)

原理：
  FFT[∇²σ] = FFT[div(g)]
  -|k|² · FFT[σ] = FFT[div(g)]
  σ = IFFT[ -FFT[div(g)] / |k|² ]      (k ≠ 0)
  DC 分量设为零（梯度不携带绝对偏移信息）

用法：
  from nerf.poisson_solver import poisson_reconstruct
  sigma = poisson_reconstruct(gradient, spacing=(dx, dy, dz))
  # sigma 保留网络训练时的 flow 数值尺度；仅缺少不可观测的加性常数。
"""
import torch
import torch.nn.functional as F


def _freq_grid(n, spacing, device):
    """生成一维频率平方分量 (2πk/L)²。"""
    k = torch.fft.fftfreq(n, d=spacing, device=device)
    return (2 * torch.pi * k) ** 2                             # k²


def poisson_reconstruct(gradient, spacing=(1.0, 1.0, 1.0), pad=8,
                        normalize=False):
    """
    从 3D 梯度场通过 FFT 泊松求解重建标量场 σ

    Args:
        gradient: torch.Tensor or np.ndarray, (H, W, D, 3)  梯度场 [∂σ/∂x, ∂σ/∂y, ∂σ/∂z]
        spacing:  (dx, dy, dz)，梯度所对应坐标系中的体素中心间距。
                  对 NIR-BOS 默认 ROI，该值约为 (0.01360525,) * 3。
        pad:      int, 每边反射填充量，缓解 Gibbs 振铃
        normalize: 是否额外归一化到 [-1, 1]。默认 False，以保留 flow
                   空间的幅值，供与基线版 σ 进行定量比较。

    Returns:
        sigma: torch.Tensor, (H, W, D) 重建的标量场。均值为零；这是
               梯度场无法恢复的常数自由度，而非额外的数值归一化。
    """
    if isinstance(gradient, torch.Tensor):
        grad = gradient.detach()
        was_numpy = False
    else:
        import numpy as np
        grad = torch.from_numpy(np.asarray(gradient, dtype=np.float32))
        was_numpy = True

    device = grad.device
    H, W, D = grad.shape[:3]
    if grad.ndim != 4 or grad.shape[-1] != 3:
        raise ValueError(f'gradient must have shape (H, W, D, 3), got {tuple(grad.shape)}')
    if len(spacing) != 3 or any(float(step) <= 0 for step in spacing):
        raise ValueError(f'spacing must contain three positive values, got {spacing}')
    dx, dy, dz = (float(step) for step in spacing)

    # ── 反射填充（抑制 FFT 周期边界假象）──
    grad = grad.permute(3, 0, 1, 2).unsqueeze(0)              # (1, 3, H, W, D)
    grad = F.pad(grad, [pad, pad, pad, pad, pad, pad], mode='reflect')

    Ph, Pw, Pd = grad.shape[2:]
    gx, gy, gz = grad[:, 0], grad[:, 1], grad[:, 2]           # each (1, Ph, Pw, Pd)

    # ── 散度 = ∂gx/∂x + ∂gy/∂y + ∂gz/∂z ──
    # 在频域通过乘以 ik 实现导数
    kx2 = _freq_grid(Ph, dx, device).view(1, -1, 1, 1)        # (1, Ph, 1, 1)
    ky2 = _freq_grid(Pw, dy, device).view(1, 1, -1, 1)        # (1, 1, Pw, 1)
    kz2 = _freq_grid(Pd, dz, device).view(1, 1, 1, -1)        # (1, 1, 1, Pd)
    k2  = kx2 + ky2 + kz2                                      # |k|²

    kx = 2j * torch.pi * torch.fft.fftfreq(Ph, d=dx, device=device).view(1, -1, 1, 1)
    ky = 2j * torch.pi * torch.fft.fftfreq(Pw, d=dy, device=device).view(1, 1, -1, 1)
    kz = 2j * torch.pi * torch.fft.fftfreq(Pd, d=dz, device=device).view(1, 1, 1, -1)

    # FFT 求散度：div = ∂gx/∂x + ∂gy/∂y + ∂gz/∂z
    div_hat = (kx * torch.fft.fftn(gx, dim=(-3, -2, -1)) +
               ky * torch.fft.fftn(gy, dim=(-3, -2, -1)) +
               kz * torch.fft.fftn(gz, dim=(-3, -2, -1)))

    # 泊松方程求解：σ_hat = -div_hat / |k|²，k=0 处设为 0
    with torch.no_grad():
        mask = k2 > 1e-12
        sigma_hat = torch.zeros_like(div_hat)
        sigma_hat[mask] = -div_hat[mask] / k2[mask]

    sigma = torch.fft.ifftn(sigma_hat, dim=(-3, -2, -1)).real

    # ── 去掉填充 ──
    sigma = sigma[:, pad:pad + H, pad:pad + W, pad:pad + D].squeeze(0).squeeze(0)

    # 梯度不能确定 σ 的常数项；固定为零均值，保留其余数值尺度。
    sigma = sigma - sigma.mean()
    if normalize:
        scale = torch.maximum(sigma.abs().max(), torch.tensor(1e-8, device=device))
        sigma = sigma / scale

    return sigma.numpy() if was_numpy else sigma
