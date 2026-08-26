import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd 

class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))

trunc_exp = _trunc_exp.apply

class _neg_trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return -torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return -g * torch.exp(x.clamp(-15, 15))

neg_trunc_exp = _neg_trunc_exp.apply

class _sigmoid(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # 强制输入为float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sigmoid(x)  # 计算 Sigmoid

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]  # 恢复输入
        sig = torch.sigmoid(x)  # 计算 Sigmoid
        grad_input = g * sig * (1 - sig)  # Sigmoid 的反向传播公式
        return grad_input  # 返回梯度

# 创建一个方便使用的 trunc_exp 函数
sigmoid = _sigmoid.apply

class _neg_sigmoid(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # 强制输入为float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # return -torch.sigmoid(x)  # 计算 -sigmoid(x)
        return -(torch.sigmoid(x) - 0.5) * 2

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]  # 恢复输入
        sig = torch.sigmoid(x)  # 计算 Sigmoid
        # grad_input = -g * sig * (1 - sig)  # Sigmoid 的反向传播公式
        grad_input =  -2 * g * sig * (1 - sig)
        return grad_input  # 返回梯度

# 创建一个方便使用的 neg_sigmoid 函数
neg_sigmoid = _neg_sigmoid.apply

class _tanh(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # 保证输入是 float32
    def forward(ctx, x, valbound):
        if not torch.is_tensor(valbound):
            valbound = torch.tensor(valbound, dtype=x.dtype, device=x.device)
        b = ( valbound[0] + valbound[1] ) / 2
        k = valbound[1] - b
        ctx.save_for_backward(x,k,b)
        return k*torch.tanh(x)+b  # 前向计算 tanh(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        (x, k ,b) = ctx.saved_tensors  # 取回保存的输入
        y = torch.tanh(x)
        grad_input = g * k * (1 - y * y)  # tanh 的导数：1 - tanh^2(x)
        grad_valbound = None
        return grad_input, grad_valbound


# 创建一个方便调用的 tanh 函数
custom_tanh = _tanh.apply
#
# class _safe_sigmoid(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float32)
#     def forward(ctx, x, valbound):
#         vmin, vmax = valbound
#         ctx.save_for_backward(x)
#         sig = torch.sigmoid(x)
#         # 缩放到 [vmin, vmax]，并让 x=0 输出 0
#         y = (vmax - vmin) * (sig - 0.5)
#         return y
#
#     @staticmethod
#     @custom_bwd
#     def backward(ctx, g):
#         x = ctx.saved_tensors[0]
#         sig = torch.sigmoid(x)
#         grad_input = g * (sig * (1 - sig)) * (ctx.saved_tensors[0].new_zeros(1) + 1) * (1)
#         # 注意只返回 x 的梯度，valbound 不参与梯度
#         return grad_input, None
#
# safe_sigmoid = _safe_sigmoid.apply