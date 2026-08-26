"""
查看泊松重建 σ 场 vs GT σ 场

泊松求解 ∇²σ = div(∇σ) 丢失了绝对偏移和尺度因子，
因此泊松 σ 需要与 GT 对齐后才能比较。

用法:
  cd claudedo/gradient_output
  python compare_sigma.py [结果目录]
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys, os

workspace = sys.argv[1] if len(sys.argv) > 1 else "result/gradient_test"
pred_file = os.path.join(workspace, "results", "sigmas0.mat")
gt_file   = os.path.join("../../MATLAB/Test_data/Phantom 1/n_GroundTruth.mat")

for f, label in [(pred_file, "预测"), (gt_file, "真实值")]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{label}文件不存在: {f}")

# ── 加载 ──
pred   = sio.loadmat(pred_file)
gt     = sio.loadmat(gt_file)
sigma_p = pred['sigmas0'].squeeze()                    # 泊松 σ (140,294,140)
n_gt    = gt['n'].squeeze()[::2, ::2, ::2]

def crop(arr, sh):
    dh = max(0, (arr.shape[0] - sh[0]) // 2)
    dw = max(0, (arr.shape[1] - sh[1]) // 2)
    dd = max(0, (arr.shape[2] - sh[2]) // 2)
    return arr[dh:dh+sh[0], dw:dw+sh[1], dd:dd+sh[2]]

n_gt = crop(n_gt, sigma_p.shape)

# ── GT σ（同 step3 变换）──
T0, n0 = 1100, 296.15 * (1.00027 - 1) / T0 + 1
flow0   = n_gt / n0 - 1
flow_max = abs(np.nanmin(flow0))
sigma_gt = flow0 / flow_max                            # GT σ ∈ [-1, ~0.15]

print(f"Poisson σ 原始范围: [{sigma_p.min():.4f}, {sigma_p.max():.4f}]")
print(f"GT σ 范围:          [{sigma_gt.min():.4f}, {sigma_gt.max():.4f}]")

# ── 对齐：泊松 σ 缩放到与 GT 同范围 ──
p_min, p_max = sigma_p.min(), sigma_p.max()
g_min, g_max = sigma_gt.min(), sigma_gt.max()
sigma_p_aligned = (sigma_p - p_min) / (p_max - p_min) * (g_max - g_min) + g_min

print(f"Poisson σ 对齐后范围: [{sigma_p_aligned.min():.4f}, {sigma_p_aligned.max():.4f}]")

# ── 误差 ──
error = sigma_p_aligned - sigma_gt
emax  = max(abs(error.min()), abs(error.max()))
print(f"Error (对齐后): [{error.min():.4f}, {error.max():.4f}]  max|err|={emax:.4f}")

# ── 可视化 ──
sh  = sigma_p.shape
vmin, vmax = g_min, g_max
z0 = sh[2] // 2

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Gradient → Poisson → σ vs GT σ", fontsize=14)

axes[0].imshow(sigma_gt[:, :, z0].T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
axes[0].set_title("GT σ")
axes[1].imshow(sigma_p[:, :, z0].T, origin='lower', cmap='hot')
axes[1].set_title("Poisson σ (raw)")
axes[2].imshow(sigma_p_aligned[:, :, z0].T, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
axes[2].set_title("Poisson σ (aligned)")
im_err = axes[3].imshow(error[:, :, z0].T, origin='lower', cmap='RdBu_r', vmin=-emax, vmax=emax)
axes[3].set_title(f"Error (Z={z0})")
plt.colorbar(im_err, ax=axes[3], shrink=0.7)

# ── 滑块 ──
slider_ax = plt.axes([0.15, 0.01, 0.7, 0.03])
slider = Slider(slider_ax, 'Z slice', 0, sh[2]-1, valinit=z0, valstep=1)

def update(z):
    z = int(z)
    axes[0].images[0].set_data(sigma_gt[:, :, z].T)
    axes[1].images[0].set_data(sigma_p[:, :, z].T)
    axes[2].images[0].set_data(sigma_p_aligned[:, :, z].T)
    axes[3].images[0].set_data(error[:, :, z].T)
    axes[0].set_title(f"GT σ (Z={z})")
    axes[1].set_title(f"Poisson σ raw (Z={z})")
    axes[2].set_title(f"Poisson σ aligned (Z={z})")
    axes[3].set_title(f"Error (Z={z})")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.show()
print("Done.")
