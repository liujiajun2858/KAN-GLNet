import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
from functools import partial
from timm.layers import trunc_normal_
from timm.layers import CondConv2d
from typing import Optional, Callable, Union



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

######################################## MS-Block start ########################################

class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k) -> None:
        super().__init__()

        self.in_conv = Conv(inc, ouc, 1)
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))


class MSBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes, in_expand_ratio=3., mid_expand_ratio=2., layers_num=3,
                 in_down_ratio=2.) -> None:
        super().__init__()

        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)
        self.in_conv = Conv(inc, in_channel)

        self.mid_convs = []
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(int(layers_num))]
            self.mid_convs.append(nn.Sequential(*mid_convs))
        self.mid_convs = nn.ModuleList(self.mid_convs)
        self.out_conv = Conv(in_channel, ouc, 1)

        self.attention = None

    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, ...]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        out = self.out_conv(out)
        if self.attention is not None:
            out = self.attention(out)
        return out


######################################## MS-Block end ########################################

######################################## iRMB ########################################
import torch
import torch.nn as nn
from torch.nn import init

########SE
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class iRMB(nn.Module):
    def __init__(self, dim_in, dim_out, norm_in=True, has_skip=True, exp_ratio=1.0,
                 act=True, v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0, dim_head=16, window_size=7,
                 attn_s=True, qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim_in) if norm_in else nn.Identity()
        self.act = Conv.default_act if act else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
        self.attn_s = attn_s
        if self.attn_s:
            assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
            self.dim_head = dim_head
            self.window_size = window_size
            self.num_head = dim_in // dim_head
            self.scale = self.dim_head ** -0.5
            self.attn_pre = attn_pre
            self.qk = nn.Conv2d(dim_in, int(dim_in * 2), 1, bias=qkv_bias)
            self.v = nn.Sequential(
                nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                self.act
            )
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            if v_proj:
                self.v = nn.Sequential(
                    nn.Conv2d(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias),
                    self.act
                )
            else:
                self.v = nn.Identity()
        self.conv_local = Conv(dim_mid, dim_mid, k=dw_ks, s=stride, d=dilation, g=dim_mid)
        self.se = SEAttention(dim_mid, reduction=se_ratio) if se_ratio > 0.0 else nn.Identity()

        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Conv2d(dim_mid, dim_out, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        B, C, H, W = x.shape
        if self.attn_s:
            # padding
            if self.window_size <= 0:
                window_size_W, window_size_H = W, H
            else:
                window_size_W, window_size_H = self.window_size, self.window_size
            pad_l, pad_t = 0, 0
            pad_r = (window_size_W - W % window_size_W) % window_size_W
            pad_b = (window_size_H - H % window_size_H) % window_size_H
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
            n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
            x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
            # attention
            b, c, h, w = x.shape
            qk = self.qk(x)
            qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head,
                           dim_head=self.dim_head).contiguous()
            q, k = qk[0], qk[1]
            attn_spa = (q @ k.transpose(-2, -1)) * self.scale
            attn_spa = attn_spa.softmax(dim=-1)
            attn_spa = self.attn_drop(attn_spa)
            if self.attn_pre:
                x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ x
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
                x_spa = self.v(x_spa)
            else:
                v = self.v(x)
                v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
                x_spa = attn_spa @ v
                x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h,
                                  w=w).contiguous()
            # unpadding
            x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        else:
            x = self.v(x)

        x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

        x = self.proj_drop(x)
        x = self.proj(x)

        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x



######################################## iRMB  ########################################

######################################## Dynamic Group Convolution Shuffle Transformer start ########################################

class DGCST(nn.Module):
    # Dynamic Group Convolution Shuffle Transformer
    def __init__(self, c1, c2) -> None:
        super().__init__()

        self.c = c2 // 4
        self.gconv = Conv(self.c, self.c, g=self.c)
        self.conv1 = Conv(c1, c2, 1)
        self.conv2 = nn.Sequential(
            Conv(c2, c2, 1),
            Conv(c2, c2, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = torch.split(x, [self.c, x.size(1) - self.c], 1)

        x1 = self.gconv(x1)

        # shuffle
        b, n, h, w = x1.size()
        b_n = b * n // 2
        y = x1.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        y = torch.cat((y[0], y[1]), 1)

        x = torch.cat([y, x2], 1)
        return x + self.conv2(x)

######################################## Dynamic Group Convolution Shuffle Transformer end ########################################

######################################## StartNet start ########################################

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

######################################## StartNet end ########################################

######################################## Edge information enhancement module start ########################################
class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]

class EIEStem(nn.Module):
    def __init__(self, inc, hidc, ouc) -> None:
        super().__init__()

        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.sobel_branch(x), self.pool_branch(x)], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class EIEM(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x = self.conv2(x_feature + x)
        return x
######################################## Edge information enhancement module end ########################################


######################################## TransNeXt Convolutional GLU start ########################################

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                      groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x
######################################## TransNeXt Convolutional GLU ########################################

######################################## vHeat start ########################################

class Mlp_Heat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Heat2D(nn.Module):
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }

    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    =>
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)
    """

    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        # del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)

        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1)  # B, H, W, C

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1).type_as(x))
        x = F.conv1d(x.contiguous().view(-1, W, C),
                     weight_cosm.contiguous().view(M, W, 1).type_as(x)).contiguous().view(B, N, M, -1)

        if not self.training:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp.type_as(x))
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp)  # exp decay

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1).type_as(x))
        x = F.conv1d(x.contiguous().view(-1, M, C),
                     weight_cosm.t().contiguous().view(W, M, 1).type_as(x)).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)

        x = x * nn.functional.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class HeatBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            res: int = 14,
            infer_mode=False,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            use_checkpoint: bool = False,
            drop: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            mlp_ratio: float = 4.0,
            post_norm=True,
            layer_scale=None,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp_Heat(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                                channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        self.infer_mode = infer_mode

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)

        self.freq_embed = nn.Parameter(torch.zeros(res, res, hidden_dim), requires_grad=True)
        trunc_normal_(self.freq_embed, std=0.02)
        self.op.infer_init_heat2d(self.freq_embed)

    def _forward(self, x: torch.Tensor):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, self.freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.op(self.norm1(x), self.freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, self.freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))  # FFN
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), self.freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if not self.training:
            self.op.infer_init_heat2d(self.freq_embed)

        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

######################################## vHeat end ########################################


######################################## Global-to-Local Spatial Aggregation Module start ########################################

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    @staticmethod
    def last_zero_init(m: Union[nn.Module, nn.Sequential]) -> None:
        try:
            from mmengine.model import kaiming_init, constant_init
            if isinstance(m, nn.Sequential):
                constant_init(m[-1], val=0)
            else:
                constant_init(m, val=0)
        except ImportError as e:
            pass

    def reset_parameters(self):
        try:
            from mmengine.model import kaiming_init
            if self.pooling_type == 'att':
                kaiming_init(self.conv_mask, mode='fan_in')
                self.conv_mask.inited = True

            if self.channel_add_conv is not None:
                self.last_zero_init(self.channel_add_conv)
            if self.channel_mul_conv is not None:
                self.last_zero_init(self.channel_mul_conv)
        except ImportError as e:
            pass

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class GLFNChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(GLFNChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class GLFNSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(GLFNSpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class GLFNConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = Conv(in_features, hidden_features, 1, act=nn.ReLU(inplace=True))
        self.conv2 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv3 = Conv(hidden_features, hidden_features, 1, act=nn.ReLU(inplace=True))
        self.conv4 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv5 = Conv(hidden_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv6 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ca = GLFNChannelAttention(64)
        self.sa = GLFNSpatialAttention()
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1


class GLFN(nn.Module):
    def __init__(self, input_dim=512, embed_dim=32):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv1_1 = Conv(embed_dim * 2, input_dim, 1)
        self.conv1_1_1 = Conv(input_dim // 2, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim // 2, embed_dim, 1)
        self.GlobelBlock = ContextBlock(inplanes=embed_dim, ratio=2)
        self.local = GLFNConvBranch(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.size()
        x_0, x_1 = x.chunk(2, dim=1)

        # local block
        local = self.local(self.local_11conv(x_0))

        # Globel block
        Globel = self.GlobelBlock(self.global_11conv(x_1))

        # concat Globel + local
        x = torch.cat([local, Globel], dim=1)
        x = self.conv1_1(x)

        return x

# 初始化 GLSA 类
glsa = GLFN(64)

# 生成随机输入数据
input_data = torch.randn(1, 64, 16, 32)  # 例如 (batch_size=1, input_dim=64, height=16, width=16)

# 前向传播
output = glsa(input_data)

print(output.shape)