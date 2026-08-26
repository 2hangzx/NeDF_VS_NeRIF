import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import  trunc_exp, neg_trunc_exp, sigmoid, neg_sigmoid, custom_tanh, TrainableTanh
from .renderer import NeRFRenderer
import math

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 bound=1,
                 mask3Ddata = None,
                 valbound = [-1.0, 0.0],        # 保留参数签名兼容，不再用于激活约束
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoding_str = encoding
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        # ---- gradient network ----
        grad_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 3                        # ∇σ_x, ∇σ_y, ∇σ_z
            else:
                out_dim = hidden_dim

            bias_flag = (l == num_layers - 1)
            layer = nn.Linear(in_dim, out_dim, bias=bias_flag)

            if l != num_layers - 1:
                nn.init.kaiming_uniform_(layer.weight, a=0.01, nonlinearity="leaky_relu")
                if bias_flag:
                    nn.init.constant_(layer.bias, 0.0)
            else:
                # 梯度均值为 0，bias 初始化为 0
                nn.init.constant_(layer.bias, 0.0)

            grad_net.append(layer)

        self.grad_net = nn.ModuleList(grad_net)
        self.grad_activation = TrainableTanh(out_features=3, init_scale=50.0)
        self.mask3Ddata = mask3Ddata

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], not used (gradient network is view-independent)
        mask3D = self.mask3Ddata.maskinterp(x)              # [N]
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.grad_net[l](h)
            if l != self.num_layers - 1:
                h = F.leaky_relu(h, negative_slope=0.01, inplace=True)

        grad = self.grad_activation(h)                       # [N, 3]
        return grad * mask3D.unsqueeze(-1)                   # [N, 3]

    def density(self, x):
        # 直接调用 forward（gradient 网络无方向依赖性）
        return self.forward(x, x)

    def density_grad(self, x):
        """显式返回梯度，供体渲染使用"""
        return self.forward(x, x)


    # optimizer utils
    def get_params(self, lr):
        if self.encoding_str == "Hash":
            params = [
                {'params': self.encoder.parameters(), 'lr': lr},
                {'params': self.grad_net.parameters(), 'lr': lr},
                {'params': self.grad_activation.parameters(), 'lr': lr},
            ]
        elif self.encoding_str == "Fourier":
            params = [
                {'params': self.grad_net.parameters(), 'lr': lr},
                {'params': self.grad_activation.parameters(), 'lr': lr},
            ]
        return params
