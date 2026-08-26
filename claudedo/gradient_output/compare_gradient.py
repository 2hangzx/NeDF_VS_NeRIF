"""
对比 |∇σ| — CGLS 真值梯度幅值 vs NeRF 梯度输出预测

用法:
  cd claudedo/gradient_output
  python compare_gradient.py [结果目录]
  默认: python compare_gradient.py result/gradient_test

依赖: numpy, scipy, matplotlib
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys, os

# ── 路径 ──
workspace  = sys.argv[1] if len(sys.argv) > 1 else "result/gradient_test"
pred_file  = os.path.join(workspace, "results", "sigmas0.mat")
gt_file    = os.path.join("../../MATLAB/Test_data/Phantom 1/n_GroundTruth.mat")

for f, label in [(pred_file, "预测"), (gt_file, "真实值")]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{label}文件不存在: {f}")

# ── 加载 ──
pred = sio.loadmat(pred_file)
gt   = sio.loadmat(gt_file)

sigmas0     = pred['sigmas0'].squeeze()               # σ 预测 (泊松重建, 140,294,140)
gradient0   = pred.get('dsigmas_dxyz_auto0')          # ∇σ 预测 (140,294,140,3)
if gradient0 is not None:
    gradient0 = gradient0.squeeze()
if gradient0 is not None:
    grad_pred_mag = np.sqrt(gradient0[...,0]**2 + gradient0[...,1]**2 + gradient0[...,2]**2)
else:
    grad_pred_mag = sigmas0                           # 回退

n_gt        = gt['n'].squeeze()                       # 原始折射率
n_gt        = n_gt[::2, ::2, ::2]                     # 降采样到 ~(140,294,140)

# ── 裁剪到一致 ──
crop = lambda arr, sh: arr[
    max(0,(arr.shape[0]-sh[0])//2):max(0,(arr.shape[0]-sh[0])//2)+sh[0],
    max(0,(arr.shape[1]-sh[1])//2):max(0,(arr.shape[1]-sh[1])//2)+sh[1],
    max(0,(arr.shape[2]-sh[2])//2):max(0,(arr.shape[2]-sh[2])//2)+sh[2]]
n_gt_crop = crop(n_gt, sigmas0.shape)
print(f"GT: {n_gt.shape} -> crop -> {n_gt_crop.shape}")
print(f"Pred: {sigmas0.shape}")

# ── 对齐变换（同 step3）──
T0 = 1100
n0 = 296.15 * (1.00027 - 1) / T0 + 1
flow0   = n_gt_crop / n0 - 1
flow_max = abs(np.nanmin(flow0))
flow_gt = flow0 / flow_max                               # GT 折射率（归一化）
print(f"n0={n0:.7f}  flow_max={flow_max:.6f}")

# ── 计算 GT 梯度幅值 ──
voxel_size = 0.01360525  # 与 step3 一致
gx_gt, gy_gt, gz_gt = np.gradient(flow_gt, voxel_size, voxel_size, voxel_size)
grad_gt_mag = np.sqrt(gx_gt**2 + gy_gt**2 + gz_gt**2)   # |∇σ_gt|

# grad_pred_mag 已在上面从 gradient0 计算

print(f"GT  |∇σ| range: [{grad_gt_mag.min():.4f}, {grad_gt_mag.max():.4f}]")
print(f"Pred |∇σ| range: [{grad_pred_mag.min():.4f}, {grad_pred_mag.max():.4f}]")

# ── 误差 ──
error = grad_pred_mag - grad_gt_mag
emax  = max(abs(error.min()), abs(error.max()))
print(f"Error range: [{error.min():.4f}, {error.max():.4f}]  max|err|={emax:.4f}")

# ── 可视化 ──
sh = sigmas0.shape
vmin = min(grad_gt_mag.min(), grad_pred_mag.min())
vmax = max(grad_gt_mag.max(), grad_pred_mag.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Gradient Output: |∇σ| — GT vs Prediction", fontsize=14)
z0 = sh[2] // 2

im_gt  = axes[0].imshow(grad_gt_mag[:, :, z0].T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
axes[0].set_title(f"|∇σ| GT (Z={z0})")
plt.colorbar(im_gt, ax=axes[0], shrink=0.7)

im_pr  = axes[1].imshow(grad_pred_mag[:, :, z0].T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
axes[1].set_title(f"|∇σ| Pred (Z={z0})")
plt.colorbar(im_pr, ax=axes[1], shrink=0.7)

im_err = axes[2].imshow(error[:, :, z0].T, origin='lower', cmap='RdBu_r', vmin=-emax, vmax=emax)
axes[2].set_title(f"Error (Z={z0})")
plt.colorbar(im_err, ax=axes[2], shrink=0.7)

# ── 切片滑块 ──
slider_ax = plt.axes([0.15, 0.01, 0.7, 0.03])
slice_slider = Slider(slider_ax, 'Z slice', 0, sh[2]-1, valinit=z0, valstep=1)

def update(z):
    z = int(z)
    im_gt.set_data(grad_gt_mag[:, :, z].T)
    im_pr.set_data(grad_pred_mag[:, :, z].T)
    im_err.set_data(error[:, :, z].T)
    axes[0].set_title(f"|∇σ| GT (Z={z})")
    axes[1].set_title(f"|∇σ| Pred (Z={z})")
    axes[2].set_title(f"Error (Z={z})")
    fig.canvas.draw_idle()

slice_slider.on_changed(update)
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.show()
print("Done.")
