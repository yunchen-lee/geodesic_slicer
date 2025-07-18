import torch
from torch import nn
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers (2024 Neural Slicer)
def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation (2024 Neural Slicer)
class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)



# siren network (2024 Neural Slicer)
class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=1., w0_initial=30., use_bias=True,
                 final_activation=None):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias,
                                activation=final_activation, is_last=True)

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)
        i = 0
        for layer, mod in zip(self.layers, mods):
            i += 1
            x = layer(x)

            if exists(mod):
               x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)


# 修改版本 2
class Siren(nn.Module):
    class Sine(nn.Module):
        def __init__(self, w0=1.):
            super().__init__()
            self.w0 = w0

        def forward(self, x):
            return torch.sin(self.w0 * x)

    def __init__(self, dim_in, dim_out, w0=1., c=6., is_first=False, use_bias=True, activation=None, is_last=False):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.is_last = is_last
        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# 修改版本 2
class SimpleSirenNet(nn.Module):
    def __init__(self, dim_in=1, dim_hidden=256, dim_out=1, num_layers=5, w0=1., w0_initial=30., use_bias=True, final_activation=None):
        super().__init__()
        self.layers = nn.ModuleList()

        # 第一層 Siren，使用 w0_initial 進行初始化，並標記為 is_first
        self.layers.append(Siren(dim_in, dim_hidden, w0=w0_initial, is_first=True, use_bias=use_bias))

        # 中間的 Siren 層，使用 w0
        for _ in range(num_layers - 1): # 這裡的 num_layers - 1 是指中間 Siren 層的數量
            self.layers.append(Siren(dim_hidden, dim_hidden, w0=w0, use_bias=use_bias))
        
        self.final_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_layer(x)


class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)
