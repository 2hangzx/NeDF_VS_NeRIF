import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import  trunc_exp, neg_trunc_exp, sigmoid, neg_sigmoid, custom_tanh
from .renderer import NeRFRenderer
import math

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics", #方向通常使用球谐函数（sphere_harmonics）进行编码
                 num_layers=2,
                 hidden_dim=64,
                 # geo_feat_dim=15,
                 # num_layers_color=2,
                 # hidden_dim_color=64,
                 bound=1,
                 mask3Ddata = None,
                 valbound = [-1.0, 0.0],
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.geo_feat_dim = geo_feat_dim
        self.encoding_str = encoding
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 #+ self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            bias_flag = (l == num_layers - 1)
            layer = nn.Linear(in_dim, out_dim, bias=bias_flag)

            # 初始化
            if l != num_layers - 1:
                nn.init.kaiming_uniform_(layer.weight, a=0.01, nonlinearity="leaky_relu")
                if bias_flag:
                    nn.init.constant_(layer.bias, 0.0)
            else:
                # 最后一层 bias 初始化保证输出 sigma ≈ 0
                b = (valbound[0] + valbound[1]) / 2
                k = valbound[1] - b
                target_bias = math.atanh(-b / k)  # ≈ -0.347
                nn.init.constant_(layer.bias, target_bias)

            sigma_net.append(layer)

        self.sigma_net = nn.ModuleList(sigma_net)
        self.valbound = valbound
        self.mask3Ddata = mask3Ddata

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        mask3D = self.mask3Ddata.maskinterp(x)
        # sigma
        # ROIsize = torch.tensor(self.ROIsize, device=x.device).float()
        # x_norm = x / ROIsize
        x = self.encoder(x , bound=self.bound)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                # h = F.relu(h, inplace=True)
                h = F.leaky_relu(h, negative_slope=0.01, inplace=True)

        sigma = custom_tanh(h[..., 0],self.valbound)

        return sigma*mask3D

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        mask3D = self.mask3Ddata.maskinterp(x)
        # ROIsize = torch.tensor(self.ROIsize, device=x.device).float()
        # x_norm = x / ROIsize
        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                # h = F.relu(h, inplace=True)
                h = F.leaky_relu(h, negative_slope=0.01, inplace=True)

        sigma = custom_tanh(h[..., 0],self.valbound)

        return {
            'sigma': sigma*mask3D,
            # 'geo_feat': geo_feat,
        }


    # optimizer utils
    def get_params(self, lr):
        if self.encoding_str == "Hash":
            params = [
                {'params': self.encoder.parameters(), 'lr': lr}, #Fourier do not need
                {'params': self.sigma_net.parameters(), 'lr': lr},
            ]
        elif self.encoding_str == "Fourier":
            params = [
                {'params': self.sigma_net.parameters(), 'lr': lr},
            ]
        return params
