# plot_slices.m — 3D 折射率场切片可视化工具

> **日期**：2026-06-11
> **位置**：`MATLAB/Utilities/plot_slices.m`
> **依赖**：MATLAB（无额外工具箱）

---

## 一、功能

从训练输出的 `sigmas0.mat`（三维折射率场）中提取并绘制二维切片图，自动保存为 PNG。

## 二、用法

```matlab
% 默认：X / Y / Z 三方向中截面各一张
plot_slices('path/to/sigmas0.mat')

% 指定方向和层号
plot_slices('sigmas0.mat', 'z', 70)

% 多切片（数组）
plot_slices('sigmas0.mat', 'y', [50, 100, 150])

% 等间隔遍历
plot_slices('sigmas0.mat', 'x', 20:20:120)

% 自定义颜色范围 + 输出目录
plot_slices('sigmas0.mat', 'z', 70, 'clim', [-0.5, 2.5], 'outdir', './my_figs')
```

## 三、参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `sigmas_path` | 字符串 | `sigmas0.mat` 文件路径（必填） |
| `direction` | `'x'` / `'y'` / `'z'` | 切片方向（可选，默认三个方向都出） |
| `indices` | 整数或数组 | 切片层号（可选，默认中截面） |
| `'clim'` | `[cmin, cmax]` | 颜色范围（键值对，可选） |
| `'outdir'` | 字符串 | 输出目录（键值对，默认 `slices/` 子目录） |

## 四、输出

| 内容 | 说明 |
|------|------|
| 文件格式 | PNG，150 DPI |
| 存放位置 | `sigmas0.mat` 同目录下的 `slices/` 文件夹 |
| 命名规则 | `slice_x070.png`、`slice_y_mid.png` 等 |
| 图面内容 | `imagesc` 热力图 + `jet` colormap + colorbar + 坐标标签 |

## 五、实现要点

- 使用 `inputParser` 解析可选参数，支持灵活调用
- `squeeze` + 转置保证切片方向与物理坐标对齐
- `exportgraphics` 保存矢量质量 PNG，无需弹出窗口（`Visible='off'`）
- 自动越界保护：索引超出范围时 clamp 到边界
- 零依赖：仅需 MATLAB 基础功能，不需要任何工具箱

## 六、示例输出

```text
[plot_slices] Loaded sigmas0: 140 x 294 x 140
[plot_slices] Value range: [-0.9706, 2.9982]
[plot_slices] Saved: .../slices/slice_x070.png
[plot_slices] Saved: .../slices/slice_y147.png
[plot_slices] Saved: .../slices/slice_z070.png
[plot_slices] Done. 3 slice(s) saved to .../slices
```
