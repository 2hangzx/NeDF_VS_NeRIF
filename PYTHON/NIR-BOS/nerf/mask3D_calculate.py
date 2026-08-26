import os
import torch
import scipy.io as sio

class Mask3DDataset:
    def __init__(self, opt, device,**kwargs):
        super().__init__()
        self.bound = opt.bound
        self.device = device
        self.root_path = opt.path
        mask3D_f_path = os.path.join(self.root_path, '3Dmask.mat')
        if opt.maskflag:
            mask_data = sio.loadmat(mask3D_f_path)  # 读取 .mat 文件
            self.mask3D = mask_data['maskback']  # 提取你想要的变量
        else:
            self.mask3D = None

        if self.mask3D is not None:
            self.mask3D = torch.from_numpy(self.mask3D)  # [N, H, W, C]
            self.mask3D = self.mask3D.to(self.device)

    def maskinterp(self, xyzs):
        if self.mask3D is not None:
            grid_size = self.mask3D.shape[0]
            x_min, x_max = -self.bound, self.bound
            xi = ((xyzs[:, 0] - x_min) / (x_max - x_min) * (grid_size - 1)).long().clamp(0, grid_size - 1)
            yi = ((xyzs[:, 1] - x_min) / (x_max - x_min) * (grid_size - 1)).long().clamp(0, grid_size - 1)
            zi = ((xyzs[:, 2] - x_min) / (x_max - x_min) * (grid_size - 1)).long().clamp(0, grid_size - 1)
            # lin = torch.linspace(-self.bound, self.bound, grid_size, device=self.device)
            # D, H, W = self.mask3D.shape
            # xi = torch.searchsorted(lin.contiguous(), xyzs[:, 0].float(), right=False)
            # yi = torch.searchsorted(lin.contiguous(), xyzs[:, 1].float(), right=False)
            # zi = torch.searchsorted(lin.contiguous(), xyzs[:, 2].float(), right=False)

            # xi = xi.clamp(0, D - 1)
            # yi = yi.clamp(0, H - 1)
            # zi = zi.clamp(0, W - 1)

            mask3D_interp = self.mask3D[xi, yi, zi]
        else:
            mask3D_interp = torch.ones(xyzs.shape[0], device=xyzs.device)

        return mask3D_interp



