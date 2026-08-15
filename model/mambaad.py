import math
import numpy as np
from typing import Optional, Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.ops as ops

from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import Bottleneck

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from model import get_model
from model import MODEL
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from hilbert import decode, encode
from pyzorder import ZOrderIndexer

from loss.adaptive_mc_loss import AdaptiveMCLoss

# ==============================================================================
# 基础卷积组件
# ==============================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)
        return x

# ==============================================================================
# Mamba 核心组件
# ==============================================================================
class HSCANS(nn.Module):
    def __init__(self, size=16, dim=2, scan_type='scan'):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)

        if 'sweep' == scan_type:
            locs_flat = indexes
        elif 'scan' == scan_type:
            indexes = indexes.reshape(size, size)
            for i in np.arange(1, size, step=2):
                indexes[i, :] = indexes[i, :][::-1]
            locs_flat = indexes.reshape(-1)
        elif 'zorder' == scan_type:
            zi = ZOrderIndexer((0, size - 1), (0, size - 1))
            locs_flat = []
            for z in indexes:
                r, c = zi.rc(int(z))
                locs_flat.append(c * size + r)
            locs_flat = np.array(locs_flat)
        elif 'zigzag' == scan_type:
            indexes = indexes.reshape(size, size)
            locs_flat = []
            for i in range(2 * size - 1):
                if i % 2 == 0:
                    start_col = max(0, i - size + 1)
                    end_col = min(i, size - 1)
                    for j in range(start_col, end_col + 1):
                        locs_flat.append(indexes[i - j, j])
                else:
                    start_row = max(0, i - size + 1)
                    end_row = min(i, size - 1)
                    for j in range(start_row, end_row + 1):
                        locs_flat.append(indexes[j, i - j])
            locs_flat = np.array(locs_flat)
        elif 'hilbert' == scan_type:
            bit = int(math.log2(size))
            locs = decode(indexes, dim, bit)
            locs_flat = self.flat_locs_hilbert(locs, dim, bit)
        else:
            raise Exception('invalid encoder mode')

        locs_flat_inv = np.argsort(locs_flat)
        index_flat = torch.LongTensor(locs_flat.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        index_flat_inv = torch.LongTensor(locs_flat_inv.astype(np.int64)).unsqueeze(0).unsqueeze(1)
        self.index_flat = nn.Parameter(index_flat, requires_grad=False)
        self.index_flat_inv = nn.Parameter(index_flat_inv, requires_grad=False)

    def flat_locs_hilbert(self, locs, num_dim, num_bit):
        ret = []
        l = 2 ** num_bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(num_dim):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)
        return np.array(ret).astype(np.uint64)

    def __call__(self, img):
        img_encode = self.encode(img)
        return img_encode

    def encode(self, img):
        img_encode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(
            2, self.index_flat_inv.expand(img.shape), img)
        return img_encode

    def decode(self, img):
        img_decode = torch.zeros(img.shape, dtype=img.dtype, device=img.device).scatter_(
            2, self.index_flat.expand(img.shape), img)
        return img_decode

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            size=8,
            scan_type='scan',
            num_direction=8,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.num_direction = num_direction

        x_proj_weight = [nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight
                         for _ in range(self.num_direction)]
        self.x_proj_weight = nn.Parameter(torch.stack(x_proj_weight, dim=0))

        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.num_direction)]
        self.dt_projs_weight = nn.Parameter(torch.stack([dt_proj.weight for dt_proj in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([dt_proj.bias for dt_proj in dt_projs], dim=0))

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.num_direction, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.num_direction, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.scans = HSCANS(size=size, scan_type=scan_type)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = self.num_direction
        xs = []
        if K >= 2:
            xs.append(self.scans.encode(x.view(B, -1, L)))
        if K >= 4:
            xs.append(self.scans.encode(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)))
        if K >= 8:
            xs.append(self.scans.encode(torch.rot90(x, k=1, dims=(2, 3)).contiguous().view(B, -1, L)))
            xs.append(self.scans.encode(
                torch.transpose(torch.rot90(x, k=1, dims=(2, 3)), dim0=2, dim1=3).contiguous().view(B, -1, L)))

        xs = torch.stack(xs, dim=1).view(B, K // 2, -1, L)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, K // 2:K], dims=[-1]).view(B, K // 2, -1, L)
        ys = []
        if K >= 2:
            ys.append(self.scans.decode(out_y[:, 0]))
            ys.append(self.scans.decode(inv_y[:, 0]))
        if K >= 4:
            ys.append(
                torch.transpose(self.scans.decode(out_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B,
                                                                                                                    -1,
                                                                                                                    L))
            ys.append(
                torch.transpose(self.scans.decode(inv_y[:, 1]).view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B,
                                                                                                                    -1,
                                                                                                                    L))
        if K >= 8:
            ys.append(
                torch.rot90(self.scans.decode(out_y[:, 2]).view(B, -1, W, H), k=3, dims=(2, 3)).contiguous().view(B, -1,
                                                                                                                  L))
            ys.append(
                torch.rot90(self.scans.decode(inv_y[:, 2]).view(B, -1, W, H), k=3, dims=(2, 3)).contiguous().view(B, -1,
                                                                                                                  L))
            ys.append(
                torch.rot90(torch.transpose(self.scans.decode(out_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3,
                            dims=(2, 3)).contiguous().view(B, -1, L))
            ys.append(
                torch.rot90(torch.transpose(self.scans.decode(inv_y[:, 3]).view(B, -1, W, H), dim0=2, dim1=3), k=3,
                            dims=(2, 3)).contiguous().view(B, -1, L))
        y = sum(ys)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class HSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            size: int = 8,
            scan_type='scan',
            num_direction=4,
            use_adaln: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.use_adaln = bool(use_adaln)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, size=size,
                                   scan_type=scan_type, num_direction=num_direction, **kwargs)
        self.drop_path = DropPath(drop_path)

        if self.use_adaln:
            cond_dim = hidden_dim
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_dim, hidden_dim * 2, bias=True)
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, input: torch.Tensor, c=None):
        x_norm = self.ln_1(input)

        if self.use_adaln and c is not None:
            gamma_c, beta_c = self.adaLN_modulation(c).chunk(2, dim=1)
            gamma_c = gamma_c.unsqueeze(1).unsqueeze(1)
            beta_c = beta_c.unsqueeze(1).unsqueeze(1)
            x_norm = x_norm * (1 + gamma_c) + beta_c

        x = input + self.drop_path(self.self_attention(x_norm))
        return x

# ==============================================================================
# 无瓶颈直通版：跨层动态与可变形注意力模块 (完全契合 93.8 的检查点)
# ==============================================================================
class DeformableAttnRes(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.offset_mask_conv = nn.Conv2d(
            channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=self.padding
        )

        self.deform_conv = ops.DeformConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False
        )

        self.dropout = nn.Dropout2d(0.1)

        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)

    def forward(self, x_query, x_pool=None):
        if x_pool is None:
            x_pool = x_query

        out = self.offset_mask_conv(x_query)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        fused_feature = self.deform_conv(x_pool, offset, mask)
        out_res = self.dropout(fused_feature)

        return out_res

class LSSModule(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            depth: int = 2,
            size: int = 8,
            scan_type: str = 'scan',
            num_direction: int = 8,
            use_selective_scan: bool = True,
            use_cnn_branch: bool = True,
            use_deformable_pool: bool = True,
            add_outer_residual: bool = True,
            use_adaln: bool = True,
            local_kernel_sizes=(5, 7),
            local_dilations=(1, 1),
            **kwargs,
    ):
        super().__init__()
        self.use_selective_scan = bool(use_selective_scan)
        self.use_cnn_branch = bool(use_cnn_branch)
        self.use_deformable_pool = bool(use_deformable_pool)
        self.add_outer_residual = bool(add_outer_residual)
        self.use_adaln = bool(use_adaln)
        if len(local_kernel_sizes) != 2 or len(local_dilations) != 2:
            raise ValueError('LSS local branch expects exactly two kernel sizes and two dilation values.')
        self.local_kernel_sizes = tuple(int(value) for value in local_kernel_sizes)
        self.local_dilations = tuple(int(value) for value in local_dilations)
        for kernel_size, dilation in zip(self.local_kernel_sizes, self.local_dilations):
            if kernel_size <= 0 or kernel_size % 2 == 0:
                raise ValueError(f'Local kernel sizes must be positive odd integers, got {kernel_size}.')
            if dilation <= 0:
                raise ValueError(f'Local dilation values must be positive integers, got {dilation}.')
        self.local_effective_receptive_fields = tuple(
            kernel_size + (kernel_size - 1) * (dilation - 1)
            for kernel_size, dilation in zip(self.local_kernel_sizes, self.local_dilations)
        )
        self.smm_blocks = nn.ModuleList()
        if self.use_selective_scan:
            self.smm_blocks = nn.ModuleList([
                HSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate,
                         d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction,
                         use_adaln=self.use_adaln, **kwargs)
                for i in range(depth)])

        # The same stage input is processed by two configurable depth-wise
        # convolutions in parallel with the HSS path. The defaults (5x5 and
        # 7x7, dilation 1) preserve the original MambaAD LSS exactly.
        if self.use_cnn_branch:
            kernel_5, kernel_7 = self.local_kernel_sizes
            dilation_5, dilation_7 = self.local_dilations
            padding_5 = dilation_5 * (kernel_5 - 1) // 2
            padding_7 = dilation_7 * (kernel_7 - 1) // 2
            self.conv1b7 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU(),
            )
            self.conv1a7 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU(),
            )
            self.conv1b5 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU(),
            )
            self.conv1a5 = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU(),
            )
            self.conv55 = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_5,
                    stride=1,
                    padding=padding_5,
                    dilation=dilation_5,
                    bias=False,
                    groups=hidden_dim,
                ),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU(),
            )
            self.conv77 = nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_7,
                    stride=1,
                    padding=padding_7,
                    dilation=dilation_7,
                    bias=False,
                    groups=hidden_dim,
                ),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU(),
            )
            self.finalconv11 = nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, stride=1)

        if self.use_deformable_pool:
            self.query_norm = nn.InstanceNorm2d(hidden_dim)
            self.deform_attn = DeformableAttnRes(channels=hidden_dim, kernel_size=3)
            self.deform_act = nn.SiLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, input: torch.Tensor, c=None, pool_feat=None):
        if not self.use_selective_scan and not self.use_cnn_branch and not self.use_deformable_pool:
            return input

        out_ssm = input

        if self.use_selective_scan:
            for blk in self.smm_blocks:
                out_ssm = blk(out_ssm, c)

        out_ssm_permuted = out_ssm.permute(0, 3, 1, 2).contiguous()

        # Official LSS fusion: HSS, 5x5 CNN and 7x7 CNN all receive the same
        # stage input, then a 1x1 convolution mixes their concatenated channels.
        if self.use_cnn_branch:
            input_conv = input.permute(0, 3, 1, 2).contiguous()
            out_77 = self.conv1a7(self.conv77(self.conv1b7(input_conv)))
            out_55 = self.conv1a5(self.conv55(self.conv1b5(input_conv)))
            output = torch.cat((out_ssm_permuted, out_55, out_77), dim=1)
            output = self.finalconv11(output)
        else:
            output = out_ssm_permuted

        if self.use_deformable_pool:
            if pool_feat is not None:
                v_pool = pool_feat.permute(0, 3, 1, 2).contiguous()
            else:
                v_pool = input.permute(0, 3, 1, 2).contiguous()

            q = self.query_norm(output)
            deform_residual = self.deform_attn(x_query=q, x_pool=v_pool)
            deform_residual = self.deform_act(deform_residual)

            output = output + deform_residual

        output = output.permute(0, 2, 3, 1).contiguous()
        return output + input if self.add_outer_residual else output

class LSSLayer_up(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            size=8,
            scan_type='scan',
            num_direction=4,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        if depth % 3 == 0:
            self.blocks = nn.ModuleList([
                LSSModule(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=3,
                    num_direction=num_direction,
                )
                for i in range(depth // 3)])
        elif depth % 2 == 0:
            self.blocks = nn.ModuleList([
                LSSModule(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    size=size,
                    scan_type=scan_type,
                    depth=2,
                    num_direction=num_direction,
                )
                for i in range(depth // 2)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x, c=None):
        if self.upsample is not None:
            x = self.upsample(x)

        dynamic_feature_pool = None

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, c, dynamic_feature_pool)
            else:
                x = blk(x, c, dynamic_feature_pool)

            if i == 0:
                dynamic_feature_pool = x.clone()

        return x

class MambaUPNet(nn.Module):
    def __init__(self, dims_decoder=[512, 256, 128, 64], depths_decoder=[3, 4, 6, 3], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, scan_type='scan', num_direction=4):
        super().__init__()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = LSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                size=8 * 2 ** (i_layer),
                scan_type=scan_type,
                num_direction=num_direction,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, c=None):
        x = rearrange(x, 'b c h w -> b h w c')
        out_features = []
        for i, layer in enumerate(self.layers_up):
            x = layer(x, c)
            if i != 0:
                out_features.insert(0, rearrange(x, 'b h w c -> b c h w'))
        return out_features

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

# ==============================================================================
# 上层架构组装 (Architecture Assembly)
# ==============================================================================
class MFF_OCE(nn.Module):
    def __init__(self, block, layers, width_per_group=64, norm_layer=None):
        super(MFF_OCE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 128, layers, stride=2)

        self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
        self.bn1 = norm_layer(32 * block.expansion)
        self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
        self.bn2 = norm_layer(64 * block.expansion)
        self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
        self.bn21 = norm_layer(32 * block.expansion)
        self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bn31 = norm_layer(64 * block.expansion)
        self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bnf = norm_layer(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation,
                  norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        sv_features = self.bn_layer(sv_features)

        return sv_features.contiguous()

class MAMBAADOfficial(nn.Module):
    def __init__(self, model_t, model_s):
        super(MAMBAADOfficial, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = MambaUPNet(
            depths_decoder=model_s['depths_decoder'],
            scan_type=model_s['scan_type'],
            num_direction=model_s['num_direction'],
        )
        self.frozen_layers = ['net_t']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        feats_s = self.net_s(self.mff_oce(feats_t))
        return feats_t, feats_s


@MODEL.register_module
def mambaad_official(pretrained=False, **kwargs):
    model = MAMBAADOfficial(**kwargs)
    return model


class MAMBAAD(nn.Module):
    def __init__(
            self,
            model_t,
            model_s,
            bottleneck_dim=512,
            biomedclip_model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            class_prompt_template='A medical image of {class_name}',
            class_prompts=None,
    ):
        super(MAMBAAD, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = MambaUPNet(depths_decoder=model_s['depths_decoder'], scan_type=model_s['scan_type'],
                                num_direction=model_s['num_direction'])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.text_proj = nn.Linear(bottleneck_dim, 512)

        # The AdaLN condition must come from a frozen semantic prior rather
        # than an industrially learned class table. A learned embedding only
        # reflects the source industrial domain and is not a valid medical
        # semantic prior at cross-domain zero-shot inference time.
        self.semantic_conditioner = FrozenBiomedTextEncoder(
            biomedclip_model_name,
            prompt_normal=None,
            prompt_abnormal=None,
            class_prompt_template=class_prompt_template,
            class_prompts=class_prompts,
        )

        self.frozen_layers = ['net_t', 'semantic_conditioner']

    def _project_global_semantics(self, bottleneck_feats: torch.Tensor) -> torch.Tensor:
        global_tokens = self.gap(bottleneck_feats)
        global_tokens = torch.flatten(global_tokens, 1)
        f_global = self.text_proj(global_tokens)
        return F.normalize(f_global, p=2, dim=1)

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def _encode_semantic_condition(self, cls_names, batch_size, device):
        # Frozen BiomedCLIP prompt embeddings are used as the only AdaLN
        # condition source so the decoder sees a semantic prior instead of a
        # trainable source-domain embedding.
        with torch.no_grad():
            return self.semantic_conditioner.encode_class_prompts(
                cls_names=cls_names,
                batch_size=batch_size,
                device=device,
            )

    def forward(self, imgs, cls_names=None, return_teacher_features=False):
        """
        Args:
            imgs: Input image tensor of shape ``(B, 3, H, W)``.
            cls_names: Optional class-name list used for conditional decoding.
            return_teacher_features: When ``True``, also returns the frozen
                teacher features needed by the original MambaAD training and
                evaluation pipeline.

        Returns:
            - Default: ``(reconstructed_features, f_global)``
            - If ``return_teacher_features=True``:
              ``(teacher_features, reconstructed_features, f_global)``
        """
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        fused_feats = self.mff_oce(feats_t)
        f_global = self._project_global_semantics(fused_feats)

        c_embed = None
        if cls_names is not None:
            c_embed = self._encode_semantic_condition(
                cls_names=cls_names,
                batch_size=imgs.shape[0],
                device=imgs.device,
            )

        reconstructed_features = self.net_s(fused_feats, c_embed)
        if return_teacher_features:
            return feats_t, reconstructed_features, f_global
        return reconstructed_features, f_global

@MODEL.register_module
def mambaad(pretrained=False, **kwargs):
    model = MAMBAAD(**kwargs)
    return model


class FrozenVisualSequenceEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, imgs):
        feats = self.backbone(imgs)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim != 4:
            raise ValueError(f'Expected visual encoder to output a 4D feature map, got {tuple(feats.shape)}.')

        bsz, channels, height, width = feats.shape
        seq = feats.flatten(2).transpose(1, 2).contiguous()
        return seq, (height, width), channels


class FrozenBiomedTextEncoder(nn.Module):
    def __init__(self, model_name, prompt_normal=None, prompt_abnormal=None,
                 class_prompt_template='A medical image of {class_name}', class_prompts=None):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                'BiomedCLIP zero-shot inference requires the `open_clip` package to be installed.'
            ) from exc

        self.model_name = model_name
        self.text_encoder, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.class_prompt_template = class_prompt_template
        self.class_prompt_map = None
        self.normal_prompt_map = None
        self.abnormal_prompt_map = None
        self.class_names = []
        if prompt_normal is not None or prompt_abnormal is not None:
            if prompt_normal is None or prompt_abnormal is None:
                raise ValueError('`prompt_normal` and `prompt_abnormal` must be provided together.')
            self.class_names, _, _ = self._build_prompt_pairs(prompt_normal, prompt_abnormal)
            self.normal_prompt_map = self._normalize_prompt_config(prompt_normal, 'prompt_normal')
            self.abnormal_prompt_map = self._normalize_prompt_config(prompt_abnormal, 'prompt_abnormal')

        if class_prompts is not None:
            class_prompt_map = self._normalize_prompt_config(class_prompts, 'class_prompts')
            if '__shared__' not in class_prompt_map and self.class_names:
                missing = sorted(set(self.class_names) - set(class_prompt_map.keys()))
                if missing:
                    raise ValueError(f'`class_prompts` is missing class keys: {missing}.')
            self.class_prompt_map = class_prompt_map
            if not self.class_names and '__shared__' not in class_prompt_map:
                self.class_names = sorted(class_prompt_map.keys())

        with torch.no_grad():
            sample_prompt = self._get_sample_class_prompt()
            sample_tokens = self.tokenizer([sample_prompt])
            self.text_dim = int(self.text_encoder.encode_text(sample_tokens).shape[-1])

    def _normalize_prompt_config(self, prompt_config, name):
        if isinstance(prompt_config, str):
            return {'__shared__': prompt_config}
        if isinstance(prompt_config, dict):
            if not prompt_config:
                raise ValueError(f'`{name}` must not be an empty dict.')
            return {str(key).lower(): value for key, value in prompt_config.items()}
        raise TypeError(f'`{name}` must be a string or dict, got {type(prompt_config).__name__}.')

    def _resolve_prompt_template(self, prompt_template, cls_name):
        if '{cls_name}' in prompt_template:
            return prompt_template.format(cls_name=cls_name)
        if '{class_name}' in prompt_template:
            return prompt_template.format(class_name=cls_name)
        return prompt_template

    def _build_prompt_pairs(self, prompt_normal, prompt_abnormal):
        normal_map = self._normalize_prompt_config(prompt_normal, 'prompt_normal')
        abnormal_map = self._normalize_prompt_config(prompt_abnormal, 'prompt_abnormal')

        if '__shared__' in normal_map and '__shared__' in abnormal_map:
            cls_names = ['__shared__']
        elif '__shared__' in normal_map:
            cls_names = list(abnormal_map.keys())
            normal_map = {name: normal_map['__shared__'] for name in cls_names}
        elif '__shared__' in abnormal_map:
            cls_names = list(normal_map.keys())
            abnormal_map = {name: abnormal_map['__shared__'] for name in cls_names}
        else:
            cls_names = sorted(normal_map.keys())
            if set(cls_names) != set(abnormal_map.keys()):
                raise ValueError('`prompt_normal` and `prompt_abnormal` must have the same class keys.')

        normal_prompts = [self._resolve_prompt_template(normal_map[name], name) for name in cls_names]
        abnormal_prompts = [self._resolve_prompt_template(abnormal_map[name], name) for name in cls_names]
        return cls_names, normal_prompts, abnormal_prompts

    def _get_sample_class_prompt(self):
        sample_name = self.class_names[0] if len(self.class_names) > 0 else 'organ'
        if self.class_prompt_map is None:
            return self.class_prompt_template.format(class_name=sample_name)
        if '__shared__' in self.class_prompt_map:
            return self._resolve_prompt_template(self.class_prompt_map['__shared__'], sample_name)
        return self._resolve_prompt_template(self.class_prompt_map[sample_name], sample_name)

    def _build_semantic_prompts(self, cls_names, prompt_map, batch_size):
        if cls_names is None:
            cls_names = [self.class_names[0] if len(self.class_names) > 0 else 'organ'] * batch_size
        elif isinstance(cls_names, str):
            cls_names = [cls_names] * batch_size
        elif len(cls_names) != batch_size:
            raise ValueError(f'Expected {batch_size} class names, got {len(cls_names)}.')

        prompts = []
        for class_name in cls_names:
            class_key = str(class_name).lower()
            if prompt_map is None:
                prompts.append(self.class_prompt_template.format(class_name=class_key))
            elif '__shared__' in prompt_map:
                prompts.append(self._resolve_prompt_template(prompt_map['__shared__'], class_key))
            else:
                if class_key not in prompt_map:
                    raise KeyError(
                        f'No prompt found for class `{class_name}`. Available classes: {sorted(prompt_map.keys())}.'
                    )
                prompts.append(self._resolve_prompt_template(prompt_map[class_key], class_key))
        return prompts

    def encode_class_prompts(self, cls_names=None, batch_size=1, device=None):
        prompt_class = self._build_semantic_prompts(cls_names, self.class_prompt_map, batch_size)
        prompt_tokens = self.tokenizer(prompt_class)
        token_device = device if device is not None else next(self.text_encoder.parameters()).device
        prompt_tokens = prompt_tokens.to(token_device)
        class_embedding = self.text_encoder.encode_text(prompt_tokens)
        return F.normalize(class_embedding, p=2, dim=-1)

    def forward(self, cls_names=None, batch_size=1):
        normal_prompts = self._build_semantic_prompts(cls_names, self.normal_prompt_map, batch_size)
        abnormal_prompts = self._build_semantic_prompts(cls_names, self.abnormal_prompt_map, batch_size)
        token_device = next(self.text_encoder.parameters()).device
        normal_tokens = self.tokenizer(normal_prompts).to(token_device)
        abnormal_tokens = self.tokenizer(abnormal_prompts).to(token_device)
        t_norm = F.normalize(self.text_encoder.encode_text(normal_tokens), p=2, dim=-1)
        t_abn = F.normalize(self.text_encoder.encode_text(abnormal_tokens), p=2, dim=-1)
        return t_norm, t_abn


class DepthRMSNorm(nn.Module):
    """RMSNorm used by depth attention without requiring a recent PyTorch build."""

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x * inv_rms.to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


class DepthAttentionResidual(nn.Module):
    """Official-style single-query attention over a list of depth sources.

    Every source has shape [B, H, W, D]. Sources are stacked on a new depth
    axis N, scored by one learned pseudo-query, and mixed with a softmax over N.
    """

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm = DepthRMSNorm(self.hidden_dim, eps=eps)
        self.proj = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, sources):
        if not isinstance(sources, (list, tuple)) or len(sources) == 0:
            raise ValueError('DepthAttentionResidual expects at least one source tensor.')

        reference_shape = sources[0].shape
        if len(reference_shape) != 4 or reference_shape[-1] != self.hidden_dim:
            raise ValueError(
                f'Expected sources shaped [B, H, W, {self.hidden_dim}], got {reference_shape}.'
            )
        if any(source.shape != reference_shape for source in sources[1:]):
            raise ValueError('All depth-attention sources must have the same shape.')

        # Complete historical representations: [N, B, H, W, D].
        V = torch.stack(list(sources), dim=0)
        K = self.norm(V)

        # Official AttnRes semantics: one learned pseudo-query per mixer,
        # no input-dependent query projection and no value projection.
        query = self.proj.weight.squeeze(0)
        logits = torch.einsum(
            'd, n b h w d -> n b h w',
            query,
            K,
        )
        weights = logits.softmax(dim=0)
        mixed = torch.einsum(
            'n b h w, n b h w d -> b h w d',
            weights,
            V,
        )
        return mixed, weights


class PDARCSSD(nn.Module):
    """Patch-wise Depth-Attention Residual CSSD.

    LSS keeps the original parallel HSS/Mamba and multi-kernel CNN paths.
    Attention Residual only changes how each stage retrieves complete earlier
    LSS representations along network depth.
    """

    def __init__(self, hidden_dim, grid_size, depths=(1, 1, 1, 1), d_state=16, drop_path_rate=0.0,
                 attn_drop_rate=0.0, scan_type='scan', num_direction=8,
                 use_selective_scan=True, use_cnn_branch=True, use_deformable_pool=False,
                 local_receptive_field_schedule=None):
        super().__init__()
        if not isinstance(depths, (list, tuple)) or len(depths) == 0:
            raise ValueError('`depths` must be a non-empty list or tuple.')
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        self.use_selective_scan = bool(use_selective_scan)
        self.use_cnn_branch = bool(use_cnn_branch)
        self.use_deformable_pool = bool(use_deformable_pool)

        if local_receptive_field_schedule is None:
            # Backward-compatible original LSS: every stage uses 5x5 and 7x7
            # depth-wise convolutions without dilation.
            stage_kernel_sizes = [(5, 7) for _ in depths]
            stage_dilations = [(1, 1) for _ in depths]
            self.local_receptive_field_schedule = tuple((5, 7) for _ in depths)
        else:
            if len(local_receptive_field_schedule) != len(depths):
                raise ValueError(
                    '`local_receptive_field_schedule` must contain one pair for each PDAR-LSS stage.'
                )
            normalized_schedule = []
            for stage_idx, receptive_fields in enumerate(local_receptive_field_schedule):
                if len(receptive_fields) != 2:
                    raise ValueError(
                        f'PDAR-LSS stage {stage_idx + 1} must define exactly two receptive fields.'
                    )
                receptive_fields = tuple(int(value) for value in receptive_fields)
                if any(value <= 0 or value % 2 == 0 for value in receptive_fields):
                    raise ValueError(
                        'Local receptive fields must be positive odd integers, '
                        f'got {receptive_fields} at stage {stage_idx + 1}.'
                    )
                normalized_schedule.append(receptive_fields)

            # A 3x3 depth-wise kernel with dilation d has effective receptive
            # field 2d+1. Using the same 3x3 kernel at every stage changes only
            # the spatial view, not the depth-wise convolution parameter count.
            self.local_receptive_field_schedule = tuple(normalized_schedule)
            stage_kernel_sizes = [(3, 3) for _ in depths]
            stage_dilations = [
                tuple((receptive_field - 1) // 2 for receptive_field in stage_fields)
                for stage_fields in self.local_receptive_field_schedule
            ]

        stage_drop_paths = torch.linspace(0, drop_path_rate, len(depths)).tolist()
        self.stages = nn.ModuleList([
            LSSModule(
                hidden_dim=self.hidden_dim,
                drop_path=stage_drop_paths[idx],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                depth=depth,
                size=self.grid_size,
                scan_type=scan_type,
                num_direction=num_direction,
                use_selective_scan=self.use_selective_scan,
                use_cnn_branch=self.use_cnn_branch,
                use_deformable_pool=self.use_deformable_pool,
                add_outer_residual=False,
                use_adaln=False,
                local_kernel_sizes=stage_kernel_sizes[idx],
                local_dilations=stage_dilations[idx],
            )
            for idx, depth in enumerate(depths)
        ])

        # Stage 1 has only F0 and therefore needs no learned selector. Stage i
        # (i > 1) attends over the complete history [F0, ..., F{i-1}].
        self.depth_mixers = nn.ModuleList([
            DepthAttentionResidual(self.hidden_dim)
            for _ in range(max(0, len(depths) - 1))
        ])
        self.final_depth_mixer = DepthAttentionResidual(self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, v_raw, semantic_embedding, spatial_shape, return_debug=False):
        bsz, num_tokens, feat_dim = v_raw.shape
        height, width = spatial_shape
        if height != width:
            raise ValueError(f'PDARCSSD currently expects square token grids, got {(height, width)}.')
        if height * width != num_tokens:
            raise ValueError(
                f'Spatial shape {(height, width)} does not match sequence length {num_tokens}.'
            )
        if feat_dim != self.hidden_dim:
            raise ValueError(f'Expected feature dim {self.hidden_dim}, got {feat_dim}.')

        x0 = v_raw.view(bsz, height, width, feat_dim)
        history = [x0]
        stage_weights = []
        pool_feat = None

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 0:
                stage_input = x0
                weights = x0.new_ones((1, bsz, height, width))
            else:
                stage_input, weights = self.depth_mixers[stage_idx - 1](history)

            stage_output = stage(stage_input, semantic_embedding, pool_feat)
            history.append(stage_output)
            stage_weights.append(weights)
            if pool_feat is None:
                pool_feat = stage_output

        final_context, final_weights = self.final_depth_mixer(history)
        context = self.out_norm(final_context)
        context_tokens = context.view(bsz, num_tokens, feat_dim)

        if not return_debug:
            return context_tokens
        return context_tokens, {
            'depth_stage_weights': tuple(weight.permute(1, 0, 2, 3) for weight in stage_weights),
            'depth_final_weights': final_weights.permute(1, 0, 2, 3),
            'depth_final_context': context_tokens,
        }


class CSSD(nn.Module):
    def __init__(self, hidden_dim, grid_size, depths=(3, 4, 6, 3), d_state=16, drop_path_rate=0.2,
                 attn_drop_rate=0.0, scan_type='scan', num_direction=8,
                 use_selective_scan=True, use_cnn_branch=True, use_deformable_pool=True):
        super().__init__()
        if not isinstance(depths, (list, tuple)) or len(depths) == 0:
            raise ValueError('`depths` must be a non-empty list or tuple.')
        self.use_selective_scan = bool(use_selective_scan)
        self.use_cnn_branch = bool(use_cnn_branch)
        self.use_deformable_pool = bool(use_deformable_pool)

        stage_drop_paths = torch.linspace(0, drop_path_rate, len(depths)).tolist()
        self.stages = nn.ModuleList([
            LSSModule(
                hidden_dim=hidden_dim,
                drop_path=stage_drop_paths[idx],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                depth=depth,
                size=grid_size,
                scan_type=scan_type,
                num_direction=num_direction,
                use_selective_scan=self.use_selective_scan,
                use_cnn_branch=self.use_cnn_branch,
                use_deformable_pool=self.use_deformable_pool,
                add_outer_residual=True,
                use_adaln=True,
            )
            for idx, depth in enumerate(depths)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, v_raw, semantic_embedding, spatial_shape):
        bsz, num_tokens, feat_dim = v_raw.shape
        height, width = spatial_shape
        if height != width:
            raise ValueError(f'CSSD currently expects square token grids, got {(height, width)}.')
        if height * width != num_tokens:
            raise ValueError(
                f'Spatial shape {(height, width)} does not match sequence length {num_tokens}.'
            )

        x = v_raw.view(bsz, height, width, feat_dim)
        pool_feat = None
        for stage in self.stages:
            x = stage(x, semantic_embedding, pool_feat)
            if pool_feat is None:
                pool_feat = x
        x = self.out_norm(x)
        return x.view(bsz, num_tokens, feat_dim)


class FrozenBiomedCLIPPatchEncoder(nn.Module):
    def __init__(
            self,
            model_name,
            prompt_normal,
            prompt_abnormal,
            input_mean=(0.485, 0.456, 0.406),
            input_std=(0.229, 0.224, 0.225),
            biomed_mean=(0.48145466, 0.4578275, 0.40821073),
            biomed_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                'BiomedCLIP patch localization requires the `open_clip` package to be installed.'
            ) from exc

        self.model_name = model_name
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.normal_prompt_map = self._normalize_prompt_config(prompt_normal, 'prompt_normal')
        self.abnormal_prompt_map = self._normalize_prompt_config(prompt_abnormal, 'prompt_abnormal')
        self.class_names = sorted(
            set(k for k in self.normal_prompt_map.keys() if k != '__shared__')
            | set(k for k in self.abnormal_prompt_map.keys() if k != '__shared__')
        )
        if '__shared__' not in self.normal_prompt_map and '__shared__' not in self.abnormal_prompt_map:
            if set(self.normal_prompt_map.keys()) != set(self.abnormal_prompt_map.keys()):
                raise ValueError('`prompt_normal` and `prompt_abnormal` must have the same class keys.')

        self.register_buffer('input_mean', torch.tensor(input_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('input_std', torch.tensor(input_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('biomed_mean', torch.tensor(biomed_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('biomed_std', torch.tensor(biomed_std).view(1, 3, 1, 1), persistent=False)

        with torch.no_grad():
            sample_tokens = self.tokenizer(['A medical image']).to(next(self.model.parameters()).device)
            self.text_dim = int(self.model.encode_text(sample_tokens).shape[-1])
            self.text_token_dim = self._infer_text_token_dim()
            image_size = getattr(self.model.visual, 'image_size', 224)
            if isinstance(image_size, (list, tuple)):
                image_size = image_size[0]
            self.image_size = int(image_size)

    def _normalize_prompt_config(self, prompt_config, name):
        if isinstance(prompt_config, str):
            return {'__shared__': prompt_config}
        if isinstance(prompt_config, dict):
            if not prompt_config:
                raise ValueError(f'`{name}` must not be an empty dict.')
            return {str(key).lower(): value for key, value in prompt_config.items()}
        raise TypeError(f'`{name}` must be a string or dict, got {type(prompt_config).__name__}.')

    def _format_prompt(self, prompt, key):
        if '{cls_name}' in prompt:
            return prompt.format(cls_name=key)
        if '{class_name}' in prompt:
            return prompt.format(class_name=key)
        return prompt

    def _resolve_prompts(self, prompt_map, cls_name):
        key = str(cls_name).lower()
        if key in prompt_map:
            prompt = prompt_map[key]
        elif '__shared__' in prompt_map:
            prompt = prompt_map['__shared__']
        elif 'good' in prompt_map:
            prompt = prompt_map['good']
        elif self.class_names:
            prompt = prompt_map[self.class_names[0]]
        else:
            raise KeyError(f'No prompt found for class `{cls_name}`.')

        if isinstance(prompt, (list, tuple)):
            if len(prompt) == 0:
                raise ValueError(f'Prompt list for class `{cls_name}` must not be empty.')
            return [self._format_prompt(str(item), key) for item in prompt]
        return [self._format_prompt(str(prompt), key)]

    def _resolve_prompt(self, prompt_map, cls_name):
        return self._resolve_prompts(prompt_map, cls_name)[0]

    def _expand_cls_names(self, cls_names, batch_size):
        if cls_names is None:
            default_name = 'good' if ('good' in self.normal_prompt_map or 'good' in self.abnormal_prompt_map) else (
                self.class_names[0] if self.class_names else '__shared__'
            )
            return [default_name] * batch_size
        if isinstance(cls_names, str):
            return [cls_names] * batch_size
        if len(cls_names) != batch_size:
            raise ValueError(f'Expected {batch_size} class names, got {len(cls_names)}.')
        return [str(name) for name in cls_names]

    def _prepare_images(self, imgs):
        imgs_01 = (imgs * self.input_std.to(dtype=imgs.dtype) + self.input_mean.to(dtype=imgs.dtype)).clamp(0.0, 1.0)
        if imgs_01.shape[-2:] != (self.image_size, self.image_size):
            imgs_01 = F.interpolate(
                imgs_01,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
        return (imgs_01 - self.biomed_mean.to(dtype=imgs.dtype)) / self.biomed_std.to(dtype=imgs.dtype)

    def _pick_feature_tensor(self, features):
        if isinstance(features, dict):
            for key in ['x_norm_patchtokens', 'x_norm', 'tokens', 'last_hidden_state']:
                if key in features:
                    return features[key]
            return next(reversed(features.values()))
        if isinstance(features, (list, tuple)):
            return features[-1]
        return features

    def _tokens_from_feature_tensor(self, features):
        features = self._pick_feature_tensor(features)
        if features.ndim == 4:
            bsz, channels, height, width = features.shape
            return features.permute(0, 2, 3, 1).reshape(bsz, height * width, channels), (height, width)
        if features.ndim != 3:
            raise RuntimeError(f'Expected patch features with 3 or 4 dims, got {tuple(features.shape)}.')

        num_tokens = features.shape[1]
        trunk = getattr(self.model.visual, 'trunk', self.model.visual)
        prefix_candidates = [
            int(getattr(trunk, 'num_prefix_tokens', 0) or 0),
            int(getattr(trunk, 'num_tokens', 0) or 0),
            1,
            0,
        ]
        for prefix in prefix_candidates:
            patch_count = num_tokens - prefix
            side = int(round(math.sqrt(patch_count)))
            if patch_count > 0 and side * side == patch_count:
                return features[:, prefix:, :], (side, side)
        side = int(math.floor(math.sqrt(num_tokens)))
        patch_count = side * side
        if patch_count <= 0:
            raise RuntimeError(f'Cannot infer patch grid from {num_tokens} tokens.')
        return features[:, -patch_count:, :], (side, side)

    def _get_projection_module(self):
        visual = getattr(self.model, 'visual', None)
        candidates = []
        if visual is not None:
            head = getattr(visual, 'head', None)
            if head is not None:
                candidates.extend([getattr(head, 'proj', None), getattr(head, 'fc', None)])
            candidates.extend([getattr(visual, 'proj', None), getattr(visual, 'projection', None)])
            trunk = getattr(visual, 'trunk', None)
            if trunk is not None:
                candidates.extend([getattr(trunk, 'proj', None), getattr(trunk, 'head', None)])
        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    def _apply_projection(self, tokens, projection):
        if projection is None or isinstance(projection, torch.nn.Identity):
            return tokens
        if isinstance(projection, torch.nn.Linear):
            flat = tokens.reshape(-1, tokens.shape[-1])
            return projection(flat).reshape(tokens.shape[0], tokens.shape[1], -1)
        if isinstance(projection, torch.nn.Parameter):
            return tokens @ projection
        if torch.is_tensor(projection):
            return tokens @ projection
        return tokens

    def encode_text_pairs(self, cls_names, batch_size, device, selection_kwargs=None, return_selection_debug=False):
        cls_names = self._expand_cls_names(cls_names, batch_size)
        normal_prompt_sets = [self._resolve_prompts(self.normal_prompt_map, name) for name in cls_names]
        abnormal_prompt_sets = [self._resolve_prompts(self.abnormal_prompt_map, name) for name in cls_names]
        return self.encode_prompt_sets(
            normal_prompt_sets,
            abnormal_prompt_sets,
            device=device,
            selection_kwargs=selection_kwargs,
            return_selection_debug=return_selection_debug,
        )

    def _normalize_prompt_value(self, prompt_value, name):
        if isinstance(prompt_value, str):
            return [prompt_value]
        if isinstance(prompt_value, (list, tuple)):
            if len(prompt_value) == 0:
                raise ValueError(f'`{name}` must not be an empty list.')
            return [str(item) for item in prompt_value]
        raise TypeError(f'`{name}` must be a string or list, got {type(prompt_value).__name__}.')

    def _same_class_consistency(self, features):
        if features.shape[0] <= 1:
            return features.new_zeros((features.shape[0],))
        sim = features @ features.t()
        eye = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
        return sim.masked_fill(eye, 0.0).sum(dim=1) / float(features.shape[0] - 1)

    def _select_prompt_features(self, normal_features, abnormal_features, selection_kwargs):
        kwargs = dict(selection_kwargs or {})
        enabled = bool(kwargs.get('enabled', False))
        if not enabled:
            return (
                F.normalize(normal_features.mean(dim=0), p=2, dim=-1),
                F.normalize(abnormal_features.mean(dim=0), p=2, dim=-1),
                {},
            )

        topk = int(kwargs.get('topk', 3))
        topk_normal = max(1, min(int(kwargs.get('topk_normal', topk)), normal_features.shape[0]))
        topk_abnormal = max(1, min(int(kwargs.get('topk_abnormal', topk)), abnormal_features.shape[0]))
        margin_weight = float(kwargs.get('margin_weight', 1.0))
        consistency_weight = float(kwargs.get('consistency_weight', 0.0))

        cross_sim = normal_features @ abnormal_features.t()
        normal_margin = (1.0 - cross_sim).mean(dim=1)
        abnormal_margin = (1.0 - cross_sim).mean(dim=0)
        normal_consistency = self._same_class_consistency(normal_features)
        abnormal_consistency = self._same_class_consistency(abnormal_features)
        normal_score = margin_weight * normal_margin + consistency_weight * normal_consistency
        abnormal_score = margin_weight * abnormal_margin + consistency_weight * abnormal_consistency

        normal_idx = torch.topk(normal_score, k=topk_normal).indices
        abnormal_idx = torch.topk(abnormal_score, k=topk_abnormal).indices
        selected_normal = normal_features.index_select(0, normal_idx)
        selected_abnormal = abnormal_features.index_select(0, abnormal_idx)
        t_norm = F.normalize(selected_normal.mean(dim=0), p=2, dim=-1)
        t_abn = F.normalize(selected_abnormal.mean(dim=0), p=2, dim=-1)
        selected_cross = selected_normal @ selected_abnormal.t()
        debug = {
            'prompt_selection_enabled': normal_features.new_tensor(1.0),
            'prompt_selection_normal_count': normal_features.new_tensor(float(topk_normal)),
            'prompt_selection_abnormal_count': normal_features.new_tensor(float(topk_abnormal)),
            'prompt_selection_candidate_normal_count': normal_features.new_tensor(float(normal_features.shape[0])),
            'prompt_selection_candidate_abnormal_count': abnormal_features.new_tensor(float(abnormal_features.shape[0])),
            'prompt_selection_normal_score_mean': normal_score.index_select(0, normal_idx).mean(),
            'prompt_selection_abnormal_score_mean': abnormal_score.index_select(0, abnormal_idx).mean(),
            'prompt_selection_normal_margin_mean': normal_margin.index_select(0, normal_idx).mean(),
            'prompt_selection_abnormal_margin_mean': abnormal_margin.index_select(0, abnormal_idx).mean(),
            'prompt_selection_selected_cross_cos_mean': selected_cross.mean(),
            'prompt_selection_selected_margin_mean': 1.0 - selected_cross.mean(),
            'prompt_selection_all_cross_cos_mean': cross_sim.mean(),
            'prompt_selection_all_margin_mean': 1.0 - cross_sim.mean(),
        }
        return t_norm, t_abn, debug

    def encode_prompt_sets(self, normal_prompt_sets, abnormal_prompt_sets, device, selection_kwargs=None, return_selection_debug=False):
        if len(normal_prompt_sets) != len(abnormal_prompt_sets):
            raise ValueError('Normal and abnormal prompt sets must have the same batch length.')
        flat_prompts = []
        slices = []
        for prompt_set in list(normal_prompt_sets) + list(abnormal_prompt_sets):
            start = len(flat_prompts)
            flat_prompts.extend([str(prompt) for prompt in prompt_set])
            slices.append((start, len(flat_prompts)))
        tokens = self.tokenizer(flat_prompts).to(device)
        features = F.normalize(self.model.encode_text(tokens), p=2, dim=-1)
        pooled = []
        selection_debugs = []
        for start, end in slices:
            pooled.append((start, end))
        batch_size = len(normal_prompt_sets)
        normal_pooled = []
        abnormal_pooled = []
        for idx in range(batch_size):
            normal_start, normal_end = pooled[idx]
            abnormal_start, abnormal_end = pooled[idx + batch_size]
            t_norm, t_abn, debug = self._select_prompt_features(
                features[normal_start:normal_end],
                features[abnormal_start:abnormal_end],
                selection_kwargs,
            )
            normal_pooled.append(t_norm)
            abnormal_pooled.append(t_abn)
            if debug:
                selection_debugs.append(debug)
        pooled = torch.cat(
            [torch.stack(normal_pooled, dim=0), torch.stack(abnormal_pooled, dim=0)],
            dim=0,
        )
        if return_selection_debug:
            merged_debug = {}
            if selection_debugs:
                for key in selection_debugs[0]:
                    merged_debug[key] = torch.stack([debug[key] for debug in selection_debugs]).mean()
            else:
                merged_debug['prompt_selection_enabled'] = pooled.new_tensor(0.0)
            return pooled[:batch_size], pooled[batch_size:], merged_debug
        return pooled[:batch_size], pooled[batch_size:]

    def encode_static_text_pairs(self, prompt_normal, prompt_abnormal, batch_size, device):
        normal_prompts = self._normalize_prompt_value(prompt_normal, 'prompt_normal')
        abnormal_prompts = self._normalize_prompt_value(prompt_abnormal, 'prompt_abnormal')
        t_norm, t_abn = self.encode_prompt_sets([normal_prompts], [abnormal_prompts], device=device)
        return t_norm.expand(batch_size, -1).contiguous(), t_abn.expand(batch_size, -1).contiguous()

    def _get_hf_text_tower(self):
        text_tower = getattr(self.model, 'text', None)
        transformer = getattr(text_tower, 'transformer', None) if text_tower is not None else None
        if transformer is not None and hasattr(transformer, 'get_input_embeddings'):
            return text_tower, transformer
        transformer = getattr(self.model, 'text_encoder', None)
        if transformer is not None and hasattr(transformer, 'get_input_embeddings'):
            return self.model, transformer
        return None, None

    def _infer_text_token_dim(self):
        _, transformer = self._get_hf_text_tower()
        if transformer is None:
            return self.text_dim
        embedding = transformer.get_input_embeddings()
        weight = getattr(embedding, 'weight', None)
        if weight is None or weight.ndim != 2:
            return self.text_dim
        return int(weight.shape[1])

    def _project_hf_text_features(self, text_tower, pooled):
        proj = getattr(text_tower, 'proj', None)
        if proj is None or isinstance(proj, torch.nn.Identity):
            return pooled
        if isinstance(proj, torch.nn.Module):
            return proj(pooled)
        if isinstance(proj, torch.nn.Parameter) or torch.is_tensor(proj):
            return pooled @ proj
        raise RuntimeError(f'Unsupported BiomedCLIP text projection type: {type(proj).__name__}.')

    def _pool_hf_text_features(self, text_tower, outputs, attention_mask):
        if isinstance(outputs, (list, tuple)):
            last_hidden_state = outputs[0]
            pooled = outputs[1] if len(outputs) > 1 and outputs[1] is not None and outputs[1].ndim == 2 else None
        else:
            last_hidden_state = outputs.last_hidden_state
            pooled = getattr(outputs, 'pooler_output', None)
        pooler = getattr(text_tower, 'pooler', None)
        if pooler is not None:
            for args in ((outputs, attention_mask), (last_hidden_state, attention_mask), (last_hidden_state,)):
                try:
                    return pooler(*args)
                except TypeError:
                    continue
        if pooled is not None:
            return pooled
        return last_hidden_state[:, 0]

    def _encode_text_with_prompt_tokens(self, token_ids, prompt_tokens):
        text_tower, transformer = self._get_hf_text_tower()
        if transformer is None:
            raise RuntimeError(
                'TIPS-style learnable token-prefix prompts require a HuggingFace-style '
                'BiomedCLIP text tower with `inputs_embeds` support.'
            )

        embedding = transformer.get_input_embeddings()
        token_ids = token_ids.to(next(self.model.parameters()).device)
        prompt_tokens = prompt_tokens.to(device=token_ids.device, dtype=embedding.weight.dtype)
        input_embeds = embedding(token_ids)

        config = getattr(transformer, 'config', None)
        pad_token_id = int(getattr(config, 'pad_token_id', 0) or 0)
        attention_mask = (token_ids != pad_token_id).long()
        prompt_len = min(int(prompt_tokens.shape[0]), max(int(input_embeds.shape[1]) - 1, 0))
        if prompt_len > 0:
            prompt = prompt_tokens[:prompt_len].unsqueeze(0).expand(input_embeds.shape[0], -1, -1)
            prompt_mask = attention_mask.new_ones((attention_mask.shape[0], prompt_len))
            input_embeds = torch.cat(
                [input_embeds[:, :1], prompt, input_embeds[:, 1:input_embeds.shape[1] - prompt_len]],
                dim=1,
            )
            attention_mask = torch.cat(
                [attention_mask[:, :1], prompt_mask, attention_mask[:, 1:attention_mask.shape[1] - prompt_len]],
                dim=1,
            )

        try:
            outputs = transformer(inputs_embeds=input_embeds, attention_mask=attention_mask, return_dict=True)
        except TypeError:
            outputs = transformer(inputs_embeds=input_embeds, attention_mask=attention_mask)
        pooled = self._pool_hf_text_features(text_tower, outputs, attention_mask)
        return self._project_hf_text_features(text_tower, pooled)

    def encode_prompt_sets_with_prompt_tokens(self, prompt_sets, prompt_tokens, device):
        flat_prompts = []
        slices = []
        for prompt_set in prompt_sets:
            start = len(flat_prompts)
            flat_prompts.extend([str(prompt) for prompt in prompt_set])
            slices.append((start, len(flat_prompts)))
        token_ids = self.tokenizer(flat_prompts).to(device)
        features = F.normalize(self._encode_text_with_prompt_tokens(token_ids, prompt_tokens), p=2, dim=-1)
        pooled = []
        for start, end in slices:
            pooled.append(F.normalize(features[start:end].mean(dim=0), p=2, dim=-1))
        return torch.stack(pooled, dim=0)

    def encode_prompt_pairs_with_prompt_tokens(
            self,
            normal_prompt_sets,
            abnormal_prompt_sets,
            normal_prompt_tokens,
            abnormal_prompt_tokens,
            device,
    ):
        if len(normal_prompt_sets) != len(abnormal_prompt_sets):
            raise ValueError('Normal and abnormal prompt sets must have the same batch length.')
        t_norm = self.encode_prompt_sets_with_prompt_tokens(normal_prompt_sets, normal_prompt_tokens, device)
        t_abn = self.encode_prompt_sets_with_prompt_tokens(abnormal_prompt_sets, abnormal_prompt_tokens, device)
        return t_norm, t_abn

    def encode_image_and_patches(self, imgs):
        biomed_imgs = self._prepare_images(imgs)
        image_features = F.normalize(self.model.encode_image(biomed_imgs), p=2, dim=-1)
        visual = self.model.visual
        trunk = getattr(visual, 'trunk', visual)
        if hasattr(trunk, 'forward_features'):
            features = trunk.forward_features(biomed_imgs)
        else:
            features = trunk(biomed_imgs)
        tokens, grid_shape = self._tokens_from_feature_tensor(features)
        tokens = self._apply_projection(tokens, self._get_projection_module())
        tokens = F.normalize(tokens, p=2, dim=-1)
        return image_features, tokens, grid_shape


class MAMBAADBiomedCLIPLocalAdapter(nn.Module):
    def __init__(
            self,
            model_s,
            biomedclip_model_name,
            prompt_normal,
            prompt_abnormal,
            image_size=256,
            local_loss_kwargs=None,
            input_mean=(0.485, 0.456, 0.406),
            input_std=(0.229, 0.224, 0.225),
            biomed_mean=(0.48145466, 0.4578275, 0.40821073),
            biomed_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super().__init__()
        self.image_size = image_size
        self.biomedclip = FrozenBiomedCLIPPatchEncoder(
            biomedclip_model_name,
            prompt_normal=prompt_normal,
            prompt_abnormal=prompt_abnormal,
            input_mean=input_mean,
            input_std=input_std,
            biomed_mean=biomed_mean,
            biomed_std=biomed_std,
        )
        self.visual_dim, self.grid_size = self._infer_visual_spec(image_size)
        self.local_adapter = CSSD(
            hidden_dim=self.visual_dim,
            grid_size=self.grid_size,
            depths=model_s.get('depths_decoder', [3, 4, 6, 3]),
            d_state=model_s.get('d_state', 16),
            drop_path_rate=model_s.get('drop_path_rate', 0.2),
            attn_drop_rate=model_s.get('attn_drop_rate', 0.0),
            scan_type=model_s.get('scan_type', 'scan'),
            num_direction=model_s.get('num_direction', 8),
            use_selective_scan=model_s.get('use_selective_scan', True),
            use_deformable_pool=model_s.get('use_deformable_pool', True),
        )
        self.local_head = nn.Linear(self.visual_dim, 1)
        nn.init.normal_(self.local_head.weight, std=0.02)
        nn.init.zeros_(self.local_head.bias)
        self.local_loss_kwargs = dict(local_loss_kwargs or {})
        self.eval_adapter_mode = 'trained'
        self.last_adapter_debug = {}

        self._freeze_module(self.biomedclip)
        self._set_requires_grad(self.local_adapter, True)
        self._set_requires_grad(self.local_head, True)

    def _infer_visual_spec(self, image_size):
        was_training = self.biomedclip.training
        self.biomedclip.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            _, tokens, spatial_shape = self.biomedclip.encode_image_and_patches(dummy)
        self.biomedclip.train(was_training)
        if tokens.shape[-1] != self.biomedclip.text_dim:
            raise ValueError(
                f'BiomedCLIP patch token dim ({tokens.shape[-1]}) must match text dim '
                f'({self.biomedclip.text_dim}).'
            )
        if spatial_shape[0] != spatial_shape[1]:
            raise ValueError(f'CSSD requires a square patch grid, got {spatial_shape}.')
        return int(tokens.shape[-1]), int(spatial_shape[0])

    def _freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            if param.is_floating_point() or param.is_complex():
                param.requires_grad = requires_grad

    def train(self, mode=True):
        self.training = mode
        self.biomedclip.eval()
        self.local_adapter.train(mode)
        self.local_head.train(mode)
        return self

    def set_eval_adapter_mode(self, mode):
        mode = str(mode).lower()
        if mode not in ('trained', 'bypass', 'random'):
            raise ValueError(f'Invalid eval_adapter_mode={mode}. Expected trained, bypass, or random.')
        self.eval_adapter_mode = mode

    def reset_adapter_parameters(self, seed=None):
        params = list(self.local_adapter.parameters()) + list(self.local_head.parameters())
        device = params[0].device if params else torch.device('cpu')
        devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
            self._reset_module_parameters(self.local_adapter)
            self.local_head.reset_parameters()
            nn.init.zeros_(self.local_head.bias)

    def _reset_module_parameters(self, module):
        for child in module.children():
            self._reset_module_parameters(child)
        reset = getattr(module, 'reset_parameters', None)
        if callable(reset):
            reset()

    def adapter_param_norm(self):
        total_sq = 0.0
        total_params = 0
        for module in [self.local_adapter, self.local_head]:
            for param in module.parameters():
                if not param.is_floating_point():
                    continue
                value = param.detach().float()
                total_sq += float(torch.sum(value * value).cpu())
                total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def _foreground_masks(self, imgs, target_shape):
        cfg = self.local_loss_kwargs
        threshold = float(cfg.get('foreground_threshold', 8.0 / 255.0))
        input_mean = self.biomedclip.input_mean.to(device=imgs.device, dtype=imgs.dtype)
        input_std = self.biomedclip.input_std.to(device=imgs.device, dtype=imgs.dtype)
        imgs_01 = (imgs * input_std + input_mean).clamp(0.0, 1.0)
        foreground = imgs_01.max(dim=1, keepdim=True).values > threshold
        if foreground.shape[-2:] != target_shape:
            foreground = F.interpolate(foreground.float(), size=target_shape, mode='nearest') > 0.5
        erode_iters = int(cfg.get('foreground_erode_iters', 1))
        interior = foreground.float()
        for _ in range(max(erode_iters, 0)):
            interior = 1.0 - F.max_pool2d(1.0 - interior, kernel_size=3, stride=1, padding=1)
        interior = interior > 0.5
        edge = foreground & ~interior
        background = ~foreground
        return foreground, interior, edge, background

    def _masked_softplus_mean(self, logits, mask):
        mask = mask.to(dtype=logits.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (F.softplus(logits) * mask).sum() / denom

    def _masked_topk_softplus(self, logits, mask, ratio):
        bsz = logits.shape[0]
        flat_logits = logits.flatten(1)
        flat_mask = mask.flatten(1).bool()
        losses = []
        for idx in range(bsz):
            values = flat_logits[idx][flat_mask[idx]]
            if values.numel() == 0:
                values = flat_logits[idx]
            k = max(1, int(values.numel() * float(ratio)))
            losses.append(F.softplus(values.topk(k).values).mean())
        return torch.stack(losses).mean()

    def _localization_losses(self, logits, imgs):
        cfg = self.local_loss_kwargs
        foreground, _, edge, background = self._foreground_masks(imgs, logits.shape[-2:])
        topk_ratio = float(cfg.get('normal_topk_ratio', 0.01))
        normal_topk = self._masked_topk_softplus(logits, foreground, topk_ratio)
        background_loss = self._masked_softplus_mean(logits, background)
        edge_loss = self._masked_softplus_mean(logits, edge)
        weights = {
            'normal_topk': float(cfg.get('normal_topk_loss_weight', 0.1)),
            'background': float(cfg.get('background_loss_weight', 0.05)),
            'edge': float(cfg.get('edge_loss_weight', 0.05)),
        }
        total = (
            weights['normal_topk'] * normal_topk
            + weights['background'] * background_loss
            + weights['edge'] * edge_loss
        )
        return {
            'total': total,
            'loss_total': total,
            'loss_normal_topk': normal_topk,
            'loss_background': background_loss,
            'loss_edge': edge_loss,
            'loss_normal_topk_weighted': weights['normal_topk'] * normal_topk,
            'loss_background_weighted': weights['background'] * background_loss,
            'loss_edge_weighted': weights['edge'] * edge_loss,
            'foreground_ratio': foreground.float().mean().detach(),
            'edge_ratio': edge.float().mean().detach(),
            'background_ratio': background.float().mean().detach(),
        }

    def _adapter_logits(self, tokens, spatial_shape, image_shape, t_norm=None, t_abn=None):
        if self.eval_adapter_mode == 'bypass':
            if t_norm is None or t_abn is None:
                raise RuntimeError('bypass mode requires text priors for normal_minus_abnormal map.')
            sim_normal = torch.einsum('bld,bd->bl', tokens, F.normalize(t_norm, p=2, dim=-1))
            sim_abnormal = torch.einsum('bld,bd->bl', tokens, F.normalize(t_abn, p=2, dim=-1))
            patch_logits = sim_normal - sim_abnormal
            refined = tokens
        else:
            refined = self.local_adapter(tokens, None, spatial_shape)
            patch_logits = self.local_head(refined).squeeze(-1)

        height, width = spatial_shape
        logits = patch_logits.view(patch_logits.shape[0], height, width)
        logits = F.interpolate(
            logits.unsqueeze(1),
            size=image_shape,
            mode='bilinear',
            align_corners=False,
        )
        with torch.no_grad():
            delta = (refined.detach() - tokens.detach()).float()
            raw = tokens.detach().float()
            refined_detached = refined.detach().float()
            self.last_adapter_debug = {
                'adapter_feature_delta_l2': delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_feature_delta_abs': delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': raw.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_refined_l2': refined_detached.pow(2).sum(dim=-1).sqrt().mean(dim=1),
            }
        return logits.squeeze(1)

    def forward(
            self,
            imgs,
            cls_names=None,
            score_cls_names=None,
            adapter_cls_names=None,
            return_anomaly_map=False,
            compute_label_free=True,
    ):
        del adapter_cls_names
        if score_cls_names is None:
            score_cls_names = cls_names
        if self.training and self.eval_adapter_mode != 'trained':
            raise RuntimeError('eval_adapter_mode is only for test/eval; use trained mode during training.')

        with torch.no_grad():
            image_features, tokens, spatial_shape = self.biomedclip.encode_image_and_patches(imgs)
            t_norm, t_abn = self.biomedclip.encode_text_pairs(
                score_cls_names,
                batch_size=imgs.shape[0],
                device=imgs.device,
            )
            image_score = torch.sum(image_features * t_abn, dim=1) - torch.sum(image_features * t_norm, dim=1)

        anomaly_map = self._adapter_logits(
            tokens.detach(),
            spatial_shape,
            (imgs.shape[2], imgs.shape[3]),
            t_norm=t_norm,
            t_abn=t_abn,
        )

        if self.training:
            if not compute_label_free:
                return {
                    'anomaly_map': anomaly_map,
                    'image_score': image_score.detach(),
                }
            out = self._localization_losses(anomaly_map.unsqueeze(1), imgs)
            if return_anomaly_map:
                out.update(anomaly_map=anomaly_map, image_score=image_score.detach())
            return out

        return anomaly_map, image_score.detach()


class BiomedCLIPLocalConvDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128, 64), dropout=0.0):
        super().__init__()
        hidden_dims = list(hidden_dims)
        layers = []
        prev_dim = in_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Conv2d(prev_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),
                nn.GELU(),
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            if idx < len(hidden_dims) - 1:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            prev_dim = hidden_dim
        self.decoder = nn.Sequential(*layers)
        self.head = nn.Conv2d(prev_dim, 1, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens, spatial_shape, image_shape):
        bsz, num_tokens, feat_dim = tokens.shape
        height, width = spatial_shape
        if height * width != num_tokens:
            raise ValueError(f'Spatial shape {(height, width)} does not match token count {num_tokens}.')
        x = tokens.transpose(1, 2).reshape(bsz, feat_dim, height, width)
        logits = self.head(self.decoder(x))
        if logits.shape[-2:] != image_shape:
            logits = F.interpolate(logits, size=image_shape, mode='bilinear', align_corners=False)
        return logits.squeeze(1)


class ARCCCalibration(nn.Module):
    """Anomaly-response-guided context calibration for local anomaly logits."""

    def __init__(
            self,
            in_dim,
            use_response=True,
            use_foreground=True,
            use_edge=True,
            kernel_size=3,
            hidden_dim=None,
            lambda_init=0.1,
    ):
        super().__init__()
        self.use_response = bool(use_response)
        self.use_foreground = bool(use_foreground)
        self.use_edge = bool(use_edge)
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError('ARCC deformable kernel_size must be odd.')
        self.kernel_size = kernel_size
        self.lambda_init = float(lambda_init)
        extra_channels = int(self.use_response) + int(self.use_foreground) + int(self.use_edge)
        offset_in_dim = in_dim + extra_channels
        offset_channels = 3 * kernel_size * kernel_size
        self.offset_head = nn.Sequential(
            nn.Conv2d(offset_in_dim, offset_in_dim, kernel_size=3, padding=1, groups=1, bias=False),
            nn.GELU(),
            nn.Conv2d(offset_in_dim, offset_channels, kernel_size=3, padding=1),
        )
        self.deform_context = ops.DeformConv2d(
            in_dim,
            in_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        hidden_dim = int(hidden_dim or max(64, in_dim // 2))
        self.calibration_head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(32, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        nn.init.zeros_(self.calibration_head[-1].weight)
        nn.init.zeros_(self.calibration_head[-1].bias)

    def forward(self, feature_map, local_logits, foreground=None, edge=None, image_shape=None):
        target_shape = feature_map.shape[-2:]
        guidance = [feature_map]
        if self.use_response:
            response = local_logits.unsqueeze(1) if local_logits.ndim == 3 else local_logits
            if response.shape[-2:] != target_shape:
                response = F.interpolate(response, size=target_shape, mode='bilinear', align_corners=False)
            guidance.append(response)
        if self.use_foreground:
            if foreground is None:
                foreground = torch.ones(
                    feature_map.shape[0],
                    1,
                    target_shape[0],
                    target_shape[1],
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                )
            elif foreground.shape[-2:] != target_shape:
                foreground = F.interpolate(foreground.to(dtype=feature_map.dtype), size=target_shape, mode='nearest')
            guidance.append(foreground.to(dtype=feature_map.dtype))
        if self.use_edge:
            if edge is None:
                edge = torch.zeros(
                    feature_map.shape[0],
                    1,
                    target_shape[0],
                    target_shape[1],
                    device=feature_map.device,
                    dtype=feature_map.dtype,
                )
            elif edge.shape[-2:] != target_shape:
                edge = F.interpolate(edge.to(dtype=feature_map.dtype), size=target_shape, mode='nearest')
            guidance.append(edge.to(dtype=feature_map.dtype))

        offset_mask = self.offset_head(torch.cat(guidance, dim=1))
        offset_channels = 2 * self.kernel_size * self.kernel_size
        offset = offset_mask[:, :offset_channels]
        mod_mask = torch.sigmoid(offset_mask[:, offset_channels:])
        context = self.deform_context(feature_map, offset, mod_mask)
        calibration = self.calibration_head(context)
        if image_shape is not None and calibration.shape[-2:] != image_shape:
            calibration = F.interpolate(calibration, size=image_shape, mode='bilinear', align_corners=False)
        return calibration.squeeze(1), mod_mask


class TextGuidedLocalRelationBranch(nn.Module):
    def __init__(
            self,
            in_dim,
            hidden_dim=256,
            dropout=0.0,
            semantic_direction='abnormal_minus_normal',
    ):
        super().__init__()
        self.semantic_direction = str(semantic_direction)
        self.proj = nn.Sequential(
            nn.Linear(in_dim * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, in_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(in_dim, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _semantic_score(self, tokens, t_norm, t_abn):
        t_norm = F.normalize(t_norm, p=2, dim=-1)
        t_abn = F.normalize(t_abn, p=2, dim=-1)
        if t_norm.ndim == 3:
            sim_normal = torch.einsum('bld,bld->bl', tokens, t_norm)
            sim_abnormal = torch.einsum('bld,bld->bl', tokens, t_abn)
        else:
            sim_normal = torch.einsum('bld,bd->bl', tokens, t_norm)
            sim_abnormal = torch.einsum('bld,bd->bl', tokens, t_abn)
        if self.semantic_direction in ('normal_minus_abnormal', 'normal-abnormal', 'old'):
            return sim_normal - sim_abnormal
        return sim_abnormal - sim_normal

    def forward(self, tokens, spatial_shape, image_shape, t_norm, t_abn):
        bsz, num_tokens, feat_dim = tokens.shape
        height, width = spatial_shape
        if height * width != num_tokens:
            raise ValueError(f'Spatial shape {(height, width)} does not match token count {num_tokens}.')

        semantic = self._semantic_score(tokens, t_norm, t_abn)
        token_map = tokens.transpose(1, 2).reshape(bsz, feat_dim, height, width)
        neighbor_map = F.avg_pool2d(token_map, kernel_size=3, stride=1, padding=1)
        neighbor_tokens = neighbor_map.flatten(2).transpose(1, 2)

        semantic_map = semantic.view(bsz, 1, height, width)
        neighbor_semantic = F.avg_pool2d(semantic_map, kernel_size=3, stride=1, padding=1).flatten(2).squeeze(1)
        semantic_diff = semantic - neighbor_semantic

        relation_input = torch.cat(
            [
                tokens,
                neighbor_tokens,
                tokens - neighbor_tokens,
                semantic.unsqueeze(-1),
                semantic_diff.unsqueeze(-1),
            ],
            dim=-1,
        )
        relation_tokens = self.proj(relation_input)
        logits = self.head(relation_tokens).squeeze(-1)
        logit_map = logits.view(bsz, height, width)
        semantic_map = semantic.view(bsz, height, width)
        semantic_gate = torch.sigmoid(semantic_map)

        if logit_map.shape[-2:] != image_shape:
            logit_map = F.interpolate(
                logit_map.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            semantic_map = F.interpolate(
                semantic_map.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            semantic_gate = F.interpolate(
                semantic_gate.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        return logit_map, relation_tokens, semantic_map, semantic_gate


class MAMBAADBiomedCLIPDualBranchAdapter(MAMBAADBiomedCLIPLocalAdapter):
    def __init__(
            self,
            model_s,
            biomedclip_model_name,
            prompt_normal,
            prompt_abnormal,
            image_size=256,
            local_loss_kwargs=None,
            text_guidance_kwargs=None,
            image_branch_kwargs=None,
            decoder_kwargs=None,
            input_mean=(0.485, 0.456, 0.406),
            input_std=(0.229, 0.224, 0.225),
            biomed_mean=(0.48145466, 0.4578275, 0.40821073),
            biomed_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        super().__init__(
            model_s=model_s,
            biomedclip_model_name=biomedclip_model_name,
            prompt_normal=prompt_normal,
            prompt_abnormal=prompt_abnormal,
            image_size=image_size,
            local_loss_kwargs=local_loss_kwargs,
            input_mean=input_mean,
            input_std=input_std,
            biomed_mean=biomed_mean,
            biomed_std=biomed_std,
        )
        self.text_guidance_kwargs = dict(text_guidance_kwargs or {})
        self.image_branch_kwargs = dict(image_branch_kwargs or {})
        self.decoder_kwargs = dict(decoder_kwargs or {})

        self.loc_decoder = BiomedCLIPLocalConvDecoder(
            self.visual_dim,
            hidden_dims=self.decoder_kwargs.get('hidden_dims', (256, 128, 64)),
            dropout=float(self.decoder_kwargs.get('dropout', 0.0)),
        )
        self.local_prompt_bank_size = max(1, int(self.text_guidance_kwargs.get('num_local_prompt_banks', 1)))
        self.local_prompt_bank_temperature = max(
            1.0e-4,
            float(self.text_guidance_kwargs.get('prompt_bank_temperature', 1.0)),
        )
        self.prompt_bank_diversity_weight = float(self.text_guidance_kwargs.get('prompt_bank_diversity_weight', 0.0))
        self.prompt_bank_class_orthogonal_weight = float(
            self.text_guidance_kwargs.get('prompt_bank_class_orthogonal_weight', 0.0)
        )
        self.local_prompt_bank_init_std = float(self.text_guidance_kwargs.get('prompt_bank_init_std', 0.0))
        self.local_text_delta_normal = nn.Parameter(torch.zeros(self.local_prompt_bank_size, self.visual_dim))
        self.local_text_delta_abnormal = nn.Parameter(torch.zeros(self.local_prompt_bank_size, self.visual_dim))
        if self.local_prompt_bank_size > 1 and self.local_prompt_bank_init_std > 0:
            nn.init.normal_(self.local_text_delta_normal, std=self.local_prompt_bank_init_std)
            nn.init.normal_(self.local_text_delta_abnormal, std=self.local_prompt_bank_init_std)
        self.local_prompt_router = nn.Linear(self.visual_dim, self.local_prompt_bank_size)
        nn.init.zeros_(self.local_prompt_router.weight)
        nn.init.zeros_(self.local_prompt_router.bias)
        self.local_prompt_token_count = max(0, int(self.text_guidance_kwargs.get('num_local_prompt_tokens', 8)))
        self.local_prompt_token_init_std = float(self.text_guidance_kwargs.get('prompt_token_init_std', 0.02))
        self.local_prompt_token_dim = int(getattr(self.biomedclip, 'text_token_dim', self.visual_dim))
        self.local_prompt_tokens_normal = nn.Parameter(torch.empty(self.local_prompt_token_count, self.local_prompt_token_dim))
        self.local_prompt_tokens_abnormal = nn.Parameter(torch.empty(self.local_prompt_token_count, self.local_prompt_token_dim))
        self._init_local_prompt_tokens()
        self.semantic_scale = nn.Parameter(torch.tensor(float(self.text_guidance_kwargs.get('gate_scale_init', 1.0))))
        self.semantic_bias = nn.Parameter(torch.tensor(float(self.text_guidance_kwargs.get('gate_bias_init', 0.0))))
        self.semantic_eta = nn.Parameter(torch.tensor(float(self.text_guidance_kwargs.get('gate_eta_init', 0.1))))
        self.image_score_beta = float(self.image_branch_kwargs.get('image_score_beta', 0.25))
        self.cssd_topk_ratio = float(self.image_branch_kwargs.get('topk_ratio', 0.05))
        self.cssd_image_loss_weight = float(self.image_branch_kwargs.get('loss_weight', 0.1))
        self.use_cssd_image_branch = bool(self.image_branch_kwargs.get('use_cssd', True))
        self.image_score_source = str(self.image_branch_kwargs.get('image_score_source', 'global')).lower()
        self.map_topk_ratio = float(self.image_branch_kwargs.get('map_topk_ratio', self.cssd_topk_ratio))
        self.text_reg_weight = float(self.text_guidance_kwargs.get('prototype_reg_weight', 0.05))
        self.prompt_token_norm_weight = float(self.text_guidance_kwargs.get('prompt_token_norm_weight', 0.0))
        self.fixed_prompt_selection_kwargs = dict(self.text_guidance_kwargs.get('fixed_prompt_selection_kwargs', {}))
        self.pathology_axis_kwargs = dict(self.text_guidance_kwargs.get('pathology_axis_kwargs', {}))
        self.pathology_axis_loss_weight = float(self.pathology_axis_kwargs.get('loss_weight', 0.0))
        self.text_prompt_mode = str(self.text_guidance_kwargs.get('text_prompt_mode', 'decoupled')).lower()
        self.global_prompt_mode = str(self.text_guidance_kwargs.get('global_prompt_mode', 'fixed')).lower()
        self.local_prompt_mode = str(self.text_guidance_kwargs.get('local_prompt_mode', 'learnable_delta')).lower()
        self.local_prompt_source = str(self.text_guidance_kwargs.get('local_prompt_source', 'class')).lower()
        self.local_prompt_source_map = {
            str(key).lower(): str(value).lower()
            for key, value in dict(self.text_guidance_kwargs.get('local_prompt_source_map', {})).items()
        }
        self.local_prompt_blend_weight = float(self.text_guidance_kwargs.get('local_prompt_blend_weight', 0.5))
        self.local_prompt_token_text_mode = str(
            self.text_guidance_kwargs.get('local_prompt_token_text_mode', 'tips_state_class')
        ).lower()
        self.local_prompt_token_class = str(self.text_guidance_kwargs.get('local_prompt_token_class', 'object'))
        self.local_prompt_token_state_normal = str(self.text_guidance_kwargs.get('local_prompt_token_state_normal', 'perfect'))
        self.local_prompt_token_state_abnormal = str(self.text_guidance_kwargs.get('local_prompt_token_state_abnormal', 'broken'))
        self.local_prompt_token_template = str(self.text_guidance_kwargs.get('local_prompt_token_template', '{state} {class_name}'))
        self.stop_local_prompt_image_grad = bool(self.text_guidance_kwargs.get('stop_local_prompt_image_grad', True))
        self.local_prompt_normal = self.text_guidance_kwargs.get(
            'local_prompt_normal',
            [
                'A normal local medical image patch with consistent tissue texture and no focal abnormal signal.',
                'A normal anatomical region with preserved local structure and no suspicious bright or dark lesion.',
            ],
        )
        self.local_prompt_abnormal = self.text_guidance_kwargs.get(
            'local_prompt_abnormal',
            [
                'An abnormal local medical image patch containing a focal lesion or abnormal tissue signal.',
                'A suspicious anatomical region with disrupted local structure, abnormal texture, or pathological contrast.',
            ],
        )
        self._validate_prompt_modes()

        self._set_requires_grad(self.loc_decoder, True)
        self._set_requires_grad(self.local_adapter, True)
        # The old linear head is kept for checkpoint compatibility but is not used by this v2 route.
        self._set_requires_grad(self.local_head, False)
        self._configure_prompt_grad_flags()
        self.last_prompt_debug = {}
        self._last_prompt_bank_weights = None
        self._fixed_text_pair_cache = {}
        self._last_fixed_prompt_selection_debug = {}
        self._pathology_axis_cache = {}

    def _validate_prompt_modes(self):
        text_modes = {'decoupled', 'shared', 'legacy'}
        global_modes = {'fixed', 'shared'}
        local_modes = {'learnable_delta', 'learnable_token_prefix', 'fixed', 'shared'}
        local_sources = {'class', 'generic', 'blend'}
        token_text_modes = {'tips_state_class', 'source_prompts'}
        if self.text_prompt_mode not in text_modes:
            raise ValueError(f'Invalid text_prompt_mode={self.text_prompt_mode}. Expected one of {sorted(text_modes)}.')
        if self.text_prompt_mode == 'legacy':
            self.text_prompt_mode = 'shared'
        if self.global_prompt_mode not in global_modes:
            raise ValueError(f'Invalid global_prompt_mode={self.global_prompt_mode}. Expected one of {sorted(global_modes)}.')
        if self.local_prompt_mode not in local_modes:
            raise ValueError(f'Invalid local_prompt_mode={self.local_prompt_mode}. Expected one of {sorted(local_modes)}.')
        if self.local_prompt_source not in local_sources:
            raise ValueError(f'Invalid local_prompt_source={self.local_prompt_source}. Expected one of {sorted(local_sources)}.')
        if self.local_prompt_token_text_mode not in token_text_modes:
            raise ValueError(
                f'Invalid local_prompt_token_text_mode={self.local_prompt_token_text_mode}. '
                f'Expected one of {sorted(token_text_modes)}.'
            )
        invalid_sources = {
            key: value for key, value in self.local_prompt_source_map.items()
            if value not in local_sources
        }
        if invalid_sources:
            raise ValueError(f'Invalid local_prompt_source_map entries: {invalid_sources}. Expected one of {sorted(local_sources)}.')
        self.local_prompt_blend_weight = max(0.0, min(1.0, self.local_prompt_blend_weight))

    def _configure_prompt_grad_flags(self):
        learnable_delta = self.local_prompt_mode in ('learnable_delta', 'shared') or self.text_prompt_mode == 'shared'
        learnable_prefix = self.local_prompt_mode == 'learnable_token_prefix'
        for param in [self.local_text_delta_normal, self.local_text_delta_abnormal]:
            param.requires_grad = bool(learnable_delta)
        for param in [self.local_prompt_tokens_normal, self.local_prompt_tokens_abnormal]:
            param.requires_grad = bool(learnable_prefix)
        self._set_requires_grad(self.local_prompt_router, bool(learnable_delta and self.local_prompt_bank_size > 1))

    def _init_local_prompt_tokens(self):
        with torch.no_grad():
            if self.local_prompt_token_count <= 0:
                return
            self.local_prompt_tokens_normal.normal_(std=self.local_prompt_token_init_std)
            self.local_prompt_tokens_abnormal.normal_(std=self.local_prompt_token_init_std)

    def load_compatible_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        compat_messages = []
        legacy_to_local = {
            'text_delta_normal': 'local_text_delta_normal',
            'text_delta_abnormal': 'local_text_delta_abnormal',
        }
        for old_key, new_key in legacy_to_local.items():
            if new_key not in state_dict and old_key in state_dict:
                state_dict[new_key] = state_dict[old_key]
                compat_messages.append(f'mapped legacy `{old_key}` -> `{new_key}`')
            if old_key in state_dict:
                del state_dict[old_key]

        current_state = self.state_dict()
        for key in ['local_text_delta_normal', 'local_text_delta_abnormal']:
            if key not in state_dict or key not in current_state:
                continue
            source = state_dict[key]
            target = current_state[key]
            if tuple(source.shape) == tuple(target.shape):
                continue
            if source.ndim == 1 and target.ndim == 2 and source.shape[0] == target.shape[1]:
                state_dict[key] = source.unsqueeze(0).expand(target.shape[0], -1).clone()
                compat_messages.append(f'expanded `{key}` from {tuple(source.shape)} to {tuple(target.shape)}')
            elif source.ndim == 2 and target.ndim == 2 and source.shape[1] == target.shape[1]:
                if source.shape[0] >= target.shape[0]:
                    state_dict[key] = source[:target.shape[0]].clone()
                else:
                    repeat = math.ceil(float(target.shape[0]) / float(source.shape[0]))
                    state_dict[key] = source.repeat(repeat, 1)[:target.shape[0]].clone()
                compat_messages.append(f'resized `{key}` from {tuple(source.shape)} to {tuple(target.shape)}')

        incompatible = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {
            'local_text_delta_normal',
            'local_text_delta_abnormal',
            'local_prompt_tokens_normal',
            'local_prompt_tokens_abnormal',
            'local_prompt_router.weight',
            'local_prompt_router.bias',
        }
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        critical_missing = [key for key in missing if key not in allowed_missing]
        critical_unexpected = [key for key in unexpected if not key.startswith('local_prompt_router.')]
        if missing:
            compat_messages.append(f'missing keys: {missing}')
        if unexpected:
            compat_messages.append(f'unexpected keys: {unexpected}')
        if compat_messages:
            print('==> PromptCheckpointCompat ' + '; '.join(compat_messages))
        if strict and (critical_missing or critical_unexpected):
            raise RuntimeError(
                'Checkpoint is incompatible after prompt migration. '
                f'critical_missing={critical_missing}, critical_unexpected={critical_unexpected}'
            )
        return incompatible

    def train(self, mode=True):
        self.training = mode
        self.biomedclip.eval()
        self.local_adapter.train(mode)
        self.loc_decoder.train(mode)
        self.local_prompt_router.train(mode)
        self.local_head.eval()
        return self

    def reset_adapter_parameters(self, seed=None):
        params = (
            list(self.local_adapter.parameters())
            + list(self.loc_decoder.parameters())
            + list(self.local_prompt_router.parameters())
        )
        device = params[0].device if params else torch.device('cpu')
        devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
            self._reset_module_parameters(self.local_adapter)
            self._reset_module_parameters(self.loc_decoder)
            self.local_prompt_router.reset_parameters()
            nn.init.zeros_(self.local_prompt_router.weight)
            nn.init.zeros_(self.local_prompt_router.bias)
            with torch.no_grad():
                self.local_text_delta_normal.zero_()
                self.local_text_delta_abnormal.zero_()
                if self.local_prompt_bank_size > 1 and self.local_prompt_bank_init_std > 0:
                    self.local_text_delta_normal.normal_(std=self.local_prompt_bank_init_std)
                    self.local_text_delta_abnormal.normal_(std=self.local_prompt_bank_init_std)
                if self.local_prompt_token_count > 0:
                    self.local_prompt_tokens_normal.normal_(std=self.local_prompt_token_init_std)
                    self.local_prompt_tokens_abnormal.normal_(std=self.local_prompt_token_init_std)

    def adapter_param_norm(self):
        total_sq = 0.0
        total_params = 0
        for module in [self.local_adapter, self.loc_decoder, self.local_prompt_router]:
            for param in module.parameters():
                if not param.is_floating_point():
                    continue
                value = param.detach().float()
                total_sq += float(torch.sum(value * value).cpu())
                total_params += value.numel()
        for param in [
            self.local_text_delta_normal,
            self.local_text_delta_abnormal,
            self.local_prompt_tokens_normal,
            self.local_prompt_tokens_abnormal,
            self.semantic_scale,
            self.semantic_bias,
            self.semantic_eta,
        ]:
            value = param.detach().float()
            total_sq += float(torch.sum(value * value).cpu())
            total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def _learnable_text_pairs(self, t_norm, t_abn, tokens=None):
        delta_norm = self.local_text_delta_normal.to(device=t_norm.device, dtype=t_norm.dtype)
        delta_abn = self.local_text_delta_abnormal.to(device=t_abn.device, dtype=t_abn.dtype)
        bank_norm = F.normalize(t_norm.unsqueeze(1) + delta_norm.unsqueeze(0), p=2, dim=-1)
        bank_abn = F.normalize(t_abn.unsqueeze(1) + delta_abn.unsqueeze(0), p=2, dim=-1)
        if self.local_prompt_bank_size <= 1 or tokens is None:
            self._last_prompt_bank_weights = None
            return bank_norm[:, 0], bank_abn[:, 0]
        router_logits = self.local_prompt_router(tokens.to(dtype=t_norm.dtype)) / self.local_prompt_bank_temperature
        router_weights = F.softmax(router_logits, dim=-1)
        learn_norm = F.normalize(torch.einsum('blk,bkd->bld', router_weights, bank_norm), p=2, dim=-1)
        learn_abn = F.normalize(torch.einsum('blk,bkd->bld', router_weights, bank_abn), p=2, dim=-1)
        self._last_prompt_bank_weights = router_weights.detach()
        return learn_norm, learn_abn

    @property
    def text_delta_normal(self):
        return self.local_text_delta_normal

    @property
    def text_delta_abnormal(self):
        return self.local_text_delta_abnormal

    def _fixed_text_pairs(self, cls_names, batch_size, device):
        cls_names = self.biomedclip._expand_cls_names(cls_names, batch_size)
        normal_features = []
        abnormal_features = []
        selection_debugs = []
        with torch.no_grad():
            for cls_name in cls_names:
                cache_key = (str(device), str(cls_name).lower())
                cached = self._fixed_text_pair_cache.get(cache_key)
                if cached is None:
                    t_norm, t_abn, selection_debug = self.biomedclip.encode_text_pairs(
                        [cls_name],
                        batch_size=1,
                        device=device,
                        selection_kwargs=self.fixed_prompt_selection_kwargs,
                        return_selection_debug=True,
                    )
                    cached = (t_norm[0].detach(), t_abn[0].detach(), selection_debug)
                    self._fixed_text_pair_cache[cache_key] = cached
                normal_features.append(cached[0].to(device=device))
                abnormal_features.append(cached[1].to(device=device))
                if len(cached) > 2 and cached[2]:
                    selection_debugs.append(cached[2])
        self._last_fixed_prompt_selection_debug = {}
        if selection_debugs:
            for key in selection_debugs[0]:
                self._last_fixed_prompt_selection_debug[key] = torch.stack([
                    debug[key].to(device=device) for debug in selection_debugs
                ]).mean()
        return torch.stack(normal_features, dim=0).detach(), torch.stack(abnormal_features, dim=0).detach()

    def _local_sources_for_batch(self, cls_names, batch_size):
        cls_names = self.biomedclip._expand_cls_names(cls_names, batch_size)
        return [self.local_prompt_source_map.get(str(name).lower(), self.local_prompt_source) for name in cls_names]

    def _format_local_prompt_token_text(self, state):
        return self.local_prompt_token_template.format(
            state=state,
            class_name=self.local_prompt_token_class,
            cls_name=self.local_prompt_token_class,
            class_text=self.local_prompt_token_class,
        )

    def _local_prompt_sets_for_batch(self, cls_names, batch_size):
        cls_names = self.biomedclip._expand_cls_names(cls_names, batch_size)
        if self.local_prompt_mode == 'learnable_token_prefix' and self.local_prompt_token_text_mode == 'tips_state_class':
            normal_prompt = self._format_local_prompt_token_text(self.local_prompt_token_state_normal)
            abnormal_prompt = self._format_local_prompt_token_text(self.local_prompt_token_state_abnormal)
            return [[normal_prompt] for _ in cls_names], [[abnormal_prompt] for _ in cls_names]

        sources = self._local_sources_for_batch(cls_names, batch_size)
        generic_normal = self.biomedclip._normalize_prompt_value(self.local_prompt_normal, 'local_prompt_normal')
        generic_abnormal = self.biomedclip._normalize_prompt_value(self.local_prompt_abnormal, 'local_prompt_abnormal')
        normal_sets = []
        abnormal_sets = []
        for name, source in zip(cls_names, sources):
            class_normal = self.biomedclip._resolve_prompts(self.biomedclip.normal_prompt_map, name)
            class_abnormal = self.biomedclip._resolve_prompts(self.biomedclip.abnormal_prompt_map, name)
            if source == 'class':
                normal_sets.append(class_normal)
                abnormal_sets.append(class_abnormal)
            elif source == 'generic':
                normal_sets.append(generic_normal)
                abnormal_sets.append(generic_abnormal)
            else:
                normal_sets.append(class_normal + generic_normal)
                abnormal_sets.append(class_abnormal + generic_abnormal)
        return normal_sets, abnormal_sets

    def _local_base_text_pairs(self, cls_names, batch_size, device, global_base_norm=None, global_base_abn=None):
        if self.text_prompt_mode == 'shared' or self.local_prompt_mode == 'shared':
            if global_base_norm is not None and global_base_abn is not None:
                return global_base_norm.detach(), global_base_abn.detach()
            return self._fixed_text_pairs(cls_names, batch_size, device)

        sources = self._local_sources_for_batch(cls_names, batch_size)
        if len(set(sources)) == 1 and sources[0] == 'class':
            if global_base_norm is not None and global_base_abn is not None:
                return global_base_norm.detach(), global_base_abn.detach()
            return self._fixed_text_pairs(cls_names, batch_size, device)

        need_class = any(source in ('class', 'blend') for source in sources)
        class_norm, class_abn = None, None
        if need_class:
            if global_base_norm is not None and global_base_abn is not None:
                class_norm, class_abn = global_base_norm.detach(), global_base_abn.detach()
            else:
                class_norm, class_abn = self._fixed_text_pairs(cls_names, batch_size, device)

        with torch.no_grad():
            generic_norm, generic_abn = self.biomedclip.encode_static_text_pairs(
                self.local_prompt_normal,
                self.local_prompt_abnormal,
                batch_size=batch_size,
                device=device,
            )
        generic_norm, generic_abn = generic_norm.detach(), generic_abn.detach()
        if len(set(sources)) == 1 and sources[0] == 'generic':
            return generic_norm, generic_abn

        local_norms = []
        local_abns = []
        for idx, source in enumerate(sources):
            if source == 'class':
                local_norms.append(class_norm[idx])
                local_abns.append(class_abn[idx])
            elif source == 'generic':
                local_norms.append(generic_norm[idx])
                local_abns.append(generic_abn[idx])
            else:
                class_weight = self.local_prompt_blend_weight
                generic_weight = 1.0 - class_weight
                local_norms.append(F.normalize(class_weight * class_norm[idx] + generic_weight * generic_norm[idx], p=2, dim=-1))
                local_abns.append(F.normalize(class_weight * class_abn[idx] + generic_weight * generic_abn[idx], p=2, dim=-1))
        return torch.stack(local_norms, dim=0).detach(), torch.stack(local_abns, dim=0).detach()

    def _local_text_pairs(self, base_t_norm, base_t_abn, cls_names=None, batch_size=None, device=None, tokens=None):
        if self.local_prompt_mode == 'fixed':
            self._last_prompt_bank_weights = None
            return F.normalize(base_t_norm.detach(), p=2, dim=-1), F.normalize(base_t_abn.detach(), p=2, dim=-1)
        if self.local_prompt_mode == 'learnable_token_prefix':
            if cls_names is None or batch_size is None or device is None:
                raise ValueError('TIPS-style local prompt tokens require cls_names, batch_size, and device.')
            self._last_prompt_bank_weights = None
            normal_sets, abnormal_sets = self._local_prompt_sets_for_batch(cls_names, batch_size)
            return self.biomedclip.encode_prompt_pairs_with_prompt_tokens(
                normal_sets,
                abnormal_sets,
                self.local_prompt_tokens_normal,
                self.local_prompt_tokens_abnormal,
                device=device,
            )
        return self._learnable_text_pairs(base_t_norm.detach(), base_t_abn.detach(), tokens=tokens)

    def _global_text_pairs(self, fixed_t_norm, fixed_t_abn, local_t_norm, local_t_abn):
        if self.text_prompt_mode == 'shared' or self.global_prompt_mode == 'shared':
            if local_t_norm.ndim == 3:
                local_t_norm = F.normalize(local_t_norm.mean(dim=1), p=2, dim=-1)
                local_t_abn = F.normalize(local_t_abn.mean(dim=1), p=2, dim=-1)
            if self.stop_local_prompt_image_grad:
                return local_t_norm.detach(), local_t_abn.detach()
            return local_t_norm, local_t_abn
        return fixed_t_norm.detach(), fixed_t_abn.detach()

    def _text_similarity(self, tokens, text):
        text = F.normalize(text, p=2, dim=-1)
        if text.ndim == 2:
            return torch.einsum('bld,bd->bl', tokens, text)
        if text.ndim == 3:
            if text.shape[:2] != tokens.shape[:2]:
                raise ValueError(
                    f'Patch-conditioned text shape {tuple(text.shape)} must match token batch/grid '
                    f'{tuple(tokens.shape[:2])}.'
                )
            return torch.einsum('bld,bld->bl', tokens, text)
        raise ValueError(f'Text prototypes must be [B, D] or [B, L, D], got {tuple(text.shape)}.')

    def _record_prompt_debug(
            self,
            image_features,
            tokens,
            fixed_global_norm,
            fixed_global_abn,
            local_base_norm,
            local_base_abn,
            local_norm,
            local_abn,
            global_score,
            cssd_image_score,
            image_score,
    ):
        with torch.no_grad():
            global_sim_normal = torch.sum(image_features.detach() * fixed_global_norm.detach(), dim=1)
            global_sim_abnormal = torch.sum(image_features.detach() * fixed_global_abn.detach(), dim=1)
            local_sim_normal = self._text_similarity(tokens.detach(), local_norm.detach())
            local_sim_abnormal = self._text_similarity(tokens.detach(), local_abn.detach())
            if local_norm.ndim == 3:
                local_base_norm_for_cos = local_base_norm.detach().unsqueeze(1)
                local_base_abn_for_cos = local_base_abn.detach().unsqueeze(1)
            else:
                local_base_norm_for_cos = local_base_norm.detach()
                local_base_abn_for_cos = local_base_abn.detach()
            local_base_sim = (
                torch.sum(F.normalize(local_base_norm_for_cos, p=2, dim=-1) * F.normalize(local_norm.detach(), p=2, dim=-1), dim=-1)
                + torch.sum(F.normalize(local_base_abn_for_cos, p=2, dim=-1) * F.normalize(local_abn.detach(), p=2, dim=-1), dim=-1)
            ) * 0.5
            fixed_proto_cos = torch.sum(
                F.normalize(fixed_global_norm.detach(), p=2, dim=-1)
                * F.normalize(fixed_global_abn.detach(), p=2, dim=-1),
                dim=-1,
            )
            local_proto_cos = torch.sum(
                F.normalize(local_norm.detach(), p=2, dim=-1)
                * F.normalize(local_abn.detach(), p=2, dim=-1),
                dim=-1,
            )
            fixed_local_normal_cos = torch.sum(
                F.normalize(fixed_global_norm.detach(), p=2, dim=-1)
                * F.normalize(local_norm.detach() if local_norm.ndim == 2 else local_norm.detach().mean(dim=1), p=2, dim=-1),
                dim=-1,
            )
            fixed_local_abnormal_cos = torch.sum(
                F.normalize(fixed_global_abn.detach(), p=2, dim=-1)
                * F.normalize(local_abn.detach() if local_abn.ndim == 2 else local_abn.detach().mean(dim=1), p=2, dim=-1),
                dim=-1,
            )
            local_patch_gap = local_sim_abnormal - local_sim_normal
            local_patch_normal_margin = local_sim_normal - local_sim_abnormal
            local_patch_gap_flat = local_patch_gap.float().reshape(-1)
            local_patch_gap_positive = local_patch_gap_flat[local_patch_gap_flat > 0]
            local_patch_gap_negative = local_patch_gap_flat[local_patch_gap_flat <= 0]
            local_patch_gap_top1 = local_patch_gap.topk(
                max(1, int(local_patch_gap.shape[1] * 0.01)),
                dim=1,
            ).values.mean()
            local_patch_gap_top5 = local_patch_gap.topk(
                max(1, int(local_patch_gap.shape[1] * 0.05)),
                dim=1,
            ).values.mean()
            local_direction = F.normalize(
                (local_abn.detach() if local_abn.ndim == 2 else local_abn.detach().mean(dim=1))
                - (local_norm.detach() if local_norm.ndim == 2 else local_norm.detach().mean(dim=1)),
                p=2,
                dim=-1,
            )
            fixed_direction = F.normalize(
                fixed_global_abn.detach() - fixed_global_norm.detach(),
                p=2,
                dim=-1,
            )
            local_fixed_direction_cos = torch.sum(local_direction * fixed_direction, dim=-1)
            token_norm_normal = self.local_prompt_tokens_normal.detach().float().norm(dim=-1)
            token_norm_abnormal = self.local_prompt_tokens_abnormal.detach().float().norm(dim=-1)
            tensors = [
                fixed_global_norm,
                fixed_global_abn,
                local_norm,
                local_abn,
                global_score,
                cssd_image_score,
                image_score,
            ]
            nonfinite = any(not torch.isfinite(t.detach()).all().item() for t in tensors)
            self.last_prompt_debug = {
                'fixed_global_normal_norm': fixed_global_norm.detach().norm(dim=-1).mean(),
                'fixed_global_abnormal_norm': fixed_global_abn.detach().norm(dim=-1).mean(),
                'local_normal_norm': local_norm.detach().norm(dim=-1).mean(),
                'local_abnormal_norm': local_abn.detach().norm(dim=-1).mean(),
                'local_delta_normal_norm': self.local_text_delta_normal.detach().float().norm(),
                'local_delta_abnormal_norm': self.local_text_delta_abnormal.detach().float().norm(),
                'local_prompt_tokens_normal_norm': self.local_prompt_tokens_normal.detach().float().norm(),
                'local_prompt_tokens_abnormal_norm': self.local_prompt_tokens_abnormal.detach().float().norm(),
                'global_sim_normal_mean': global_sim_normal.mean(),
                'global_sim_abnormal_mean': global_sim_abnormal.mean(),
                'local_patch_sim_normal_mean': local_sim_normal.mean(),
                'local_patch_sim_abnormal_mean': local_sim_abnormal.mean(),
                'fixed_proto_cos_mean': fixed_proto_cos.mean(),
                'fixed_proto_margin_mean': (1.0 - fixed_proto_cos).mean(),
                'local_proto_cos_mean': local_proto_cos.mean(),
                'local_proto_margin_mean': (1.0 - local_proto_cos).mean(),
                'fixed_local_normal_cos_mean': fixed_local_normal_cos.mean(),
                'fixed_local_abnormal_cos_mean': fixed_local_abnormal_cos.mean(),
                'local_patch_gap_mean': local_patch_gap.mean(),
                'local_patch_gap_std': local_patch_gap.float().std(unbiased=False),
                'local_patch_gap_q95': torch.quantile(local_patch_gap_flat, 0.95),
                'local_patch_gap_q99': torch.quantile(local_patch_gap_flat, 0.99),
                'local_patch_gap_top1_mean': local_patch_gap_top1,
                'local_patch_gap_top5_mean': local_patch_gap_top5,
                'local_patch_gap_positive_mean': (
                    local_patch_gap_positive.mean()
                    if local_patch_gap_positive.numel()
                    else image_score.new_tensor(0.0)
                ),
                'local_patch_gap_negative_mean': (
                    local_patch_gap_negative.mean()
                    if local_patch_gap_negative.numel()
                    else image_score.new_tensor(0.0)
                ),
                'local_patch_abnormal_win_ratio': (local_patch_gap > 0).float().mean(),
                'local_patch_normal_margin_mean': local_patch_normal_margin.mean(),
                'local_fixed_direction_cos_mean': local_fixed_direction_cos.mean(),
                'local_base_proto_cos_mean': local_base_sim.mean(),
                'global_score_mean': global_score.detach().mean(),
                'cssd_image_score_mean_debug': cssd_image_score.detach().mean(),
                'image_score_mean': image_score.detach().mean(),
                'local_prompt_token_normal_norm_mean': token_norm_normal.mean() if token_norm_normal.numel() else image_score.new_tensor(0.0),
                'local_prompt_token_normal_norm_std': token_norm_normal.std(unbiased=False) if token_norm_normal.numel() else image_score.new_tensor(0.0),
                'local_prompt_token_abnormal_norm_mean': token_norm_abnormal.mean() if token_norm_abnormal.numel() else image_score.new_tensor(0.0),
                'local_prompt_token_abnormal_norm_std': token_norm_abnormal.std(unbiased=False) if token_norm_abnormal.numel() else image_score.new_tensor(0.0),
                'local_prompt_bank_size': torch.tensor(
                    float(self.local_prompt_bank_size),
                    device=image_score.device,
                    dtype=image_score.dtype,
                ),
                'local_prompt_token_count': torch.tensor(
                    float(self.local_prompt_token_count),
                    device=image_score.device,
                    dtype=image_score.dtype,
                ),
                'local_prompt_token_prefix_enabled': torch.tensor(
                    float(self.local_prompt_mode == 'learnable_token_prefix'),
                    device=image_score.device,
                    dtype=image_score.dtype,
                ),
                'fixed_global_requires_grad': torch.tensor(
                    float(fixed_global_norm.requires_grad or fixed_global_abn.requires_grad),
                    device=image_score.device,
                    dtype=image_score.dtype,
                ),
                'local_delta_requires_grad': torch.tensor(
                    float(
                        self.local_text_delta_normal.requires_grad
                        or self.local_text_delta_abnormal.requires_grad
                    ),
                    device=image_score.device,
                    dtype=image_score.dtype,
                ),
                'local_prompt_token_requires_grad': torch.tensor(
                    float(
                        self.local_prompt_tokens_normal.requires_grad
                        or self.local_prompt_tokens_abnormal.requires_grad
                    ),
                    device=image_score.device,
                    dtype=image_score.dtype,
                ),
                'prompt_nonfinite': torch.tensor(float(nonfinite), device=image_score.device, dtype=image_score.dtype),
            }
            weights = self._last_prompt_bank_weights
            if weights is not None:
                entropy = -(weights * weights.clamp_min(1.0e-8).log()).sum(dim=-1)
                usage = weights.mean(dim=(0, 1))
                self.last_prompt_debug.update({
                    'prompt_bank_entropy': entropy.mean(),
                    'prompt_bank_usage_max': usage.max(),
                    'prompt_bank_usage_min': usage.min(),
                })
            if self._last_fixed_prompt_selection_debug:
                self.last_prompt_debug.update(self._last_fixed_prompt_selection_debug)

    def _text_prototype_regularization(self, t_norm, t_abn, learn_norm, learn_abn):
        if learn_norm.ndim == 3:
            t_norm = t_norm.unsqueeze(1)
            t_abn = t_abn.unsqueeze(1)
        loss_norm = 1.0 - torch.sum(F.normalize(t_norm, p=2, dim=-1) * learn_norm, dim=-1)
        loss_abn = 1.0 - torch.sum(F.normalize(t_abn, p=2, dim=-1) * learn_abn, dim=-1)
        return (loss_norm + loss_abn).mean()

    def _prompt_bank_diversity_regularization(self, base_t_norm, base_t_abn):
        if self.local_prompt_bank_size <= 1:
            return base_t_norm.new_tensor(0.0)
        delta_norm = self.local_text_delta_normal.to(device=base_t_norm.device, dtype=base_t_norm.dtype)
        delta_abn = self.local_text_delta_abnormal.to(device=base_t_abn.device, dtype=base_t_abn.dtype)
        bank_norm = F.normalize(base_t_norm.unsqueeze(1) + delta_norm.unsqueeze(0), p=2, dim=-1)
        bank_abn = F.normalize(base_t_abn.unsqueeze(1) + delta_abn.unsqueeze(0), p=2, dim=-1)
        eye = torch.eye(self.local_prompt_bank_size, device=base_t_norm.device, dtype=torch.bool).unsqueeze(0)
        sim_norm = torch.matmul(bank_norm, bank_norm.transpose(1, 2)).masked_fill(eye, 0.0)
        sim_abn = torch.matmul(bank_abn, bank_abn.transpose(1, 2)).masked_fill(eye, 0.0)
        denom = float(self.local_prompt_bank_size * max(self.local_prompt_bank_size - 1, 1))
        return (sim_norm.pow(2).sum(dim=(1, 2)) + sim_abn.pow(2).sum(dim=(1, 2))).mean() / (2.0 * denom)

    def _prompt_bank_class_orthogonal_regularization(self, fixed_norm, fixed_abn, local_norm, local_abn):
        if self.prompt_bank_class_orthogonal_weight <= 0:
            return fixed_norm.new_tensor(0.0)
        class_direction = F.normalize(fixed_abn.detach() - fixed_norm.detach(), p=2, dim=-1)
        local_direction = F.normalize(local_abn - local_norm, p=2, dim=-1)
        if local_direction.ndim == 3:
            class_direction = class_direction.unsqueeze(1)
        return torch.sum(local_direction * class_direction, dim=-1).abs().mean()

    def _prompt_token_norm_regularization(self):
        if self.prompt_token_norm_weight <= 0 or self.local_prompt_token_count <= 0:
            return self.local_prompt_tokens_normal.new_tensor(0.0)
        normal_loss = self.local_prompt_tokens_normal.float().pow(2).mean()
        abnormal_loss = self.local_prompt_tokens_abnormal.float().pow(2).mean()
        return 0.5 * (normal_loss + abnormal_loss)

    def _pathology_axis_text_pairs(self, batch_size, device):
        normal_prompts = self.pathology_axis_kwargs.get(
            'normal_prompts',
            [
                'A local medical region with regular tissue texture.',
                'A local anatomical patch with preserved structure.',
            ],
        )
        abnormal_prompts = self.pathology_axis_kwargs.get(
            'abnormal_prompts',
            [
                'A local medical region with focal pathological tissue.',
                'A local anatomical patch with lesion, abnormal signal, or disrupted structure.',
            ],
        )
        cache_key = (
            str(device),
            tuple(str(prompt) for prompt in self.biomedclip._normalize_prompt_value(normal_prompts, 'pathology_axis_normal')),
            tuple(str(prompt) for prompt in self.biomedclip._normalize_prompt_value(abnormal_prompts, 'pathology_axis_abnormal')),
        )
        cached = self._pathology_axis_cache.get(cache_key)
        if cached is None:
            with torch.no_grad():
                axis_norm, axis_abn = self.biomedclip.encode_static_text_pairs(
                    normal_prompts,
                    abnormal_prompts,
                    batch_size=1,
                    device=device,
                )
            cached = (axis_norm[0].detach(), axis_abn[0].detach())
            self._pathology_axis_cache[cache_key] = cached
        axis_norm = cached[0].to(device=device).unsqueeze(0).expand(batch_size, -1)
        axis_abn = cached[1].to(device=device).unsqueeze(0).expand(batch_size, -1)
        return axis_norm.detach(), axis_abn.detach()

    def _pathology_axis_regularization(self, local_norm, local_abn, batch_size, device):
        if self.pathology_axis_loss_weight <= 0:
            return local_norm.new_tensor(0.0), {'pathology_axis_enabled': local_norm.new_tensor(0.0)}
        axis_norm, axis_abn = self._pathology_axis_text_pairs(batch_size, device)
        local_direction = F.normalize(local_abn - local_norm, p=2, dim=-1)
        if local_direction.ndim == 3:
            local_direction = F.normalize(local_direction.mean(dim=1), p=2, dim=-1)
        axis_direction = F.normalize(axis_abn - axis_norm, p=2, dim=-1)
        axis_cos = torch.sum(local_direction * axis_direction, dim=-1)
        loss_axis = (1.0 - axis_cos).mean()
        debug = {
            'pathology_axis_enabled': local_norm.new_tensor(1.0),
            'pathology_axis_cos_mean': axis_cos.detach().mean(),
            'pathology_axis_margin_mean': (
                1.0 - torch.sum(F.normalize(axis_norm, p=2, dim=-1) * F.normalize(axis_abn, p=2, dim=-1), dim=-1)
            ).detach().mean(),
        }
        return loss_axis, debug

    def _semantic_gate(self, tokens, t_norm, t_abn, spatial_shape, image_shape):
        sim_normal = self._text_similarity(tokens, t_norm)
        sim_abnormal = self._text_similarity(tokens, t_abn)
        # Prior diagnostic showed normal-minus-abnormal is the better localization direction.
        semantic = sim_normal - sim_abnormal
        gate = torch.sigmoid(self.semantic_scale * semantic + self.semantic_bias)
        height, width = spatial_shape
        semantic_map = semantic.view(semantic.shape[0], height, width)
        gate_map = gate.view(gate.shape[0], height, width)
        if gate_map.shape[-2:] != image_shape:
            semantic_map = F.interpolate(
                semantic_map.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
            gate_map = F.interpolate(
                gate_map.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        return semantic_map, gate_map

    def _cssd_image_branch(self, tokens, spatial_shape, image_shape, t_norm, t_abn, detach_text=False):
        if (not self.use_cssd_image_branch) or self.eval_adapter_mode == 'bypass':
            refined = tokens
        else:
            refined = self.local_adapter(tokens, None, spatial_shape)
        if detach_text:
            t_norm = t_norm.detach()
            t_abn = t_abn.detach()
        sim_normal = self._text_similarity(refined, t_norm)
        sim_abnormal = self._text_similarity(refined, t_abn)
        patch_scores = sim_abnormal - sim_normal
        topk = max(1, int(patch_scores.shape[1] * self.cssd_topk_ratio))
        cssd_image_score = patch_scores.topk(topk, dim=1).values.mean(dim=1)
        height, width = spatial_shape
        cssd_map = patch_scores.view(patch_scores.shape[0], height, width)
        cssd_map = F.interpolate(
            cssd_map.unsqueeze(1),
            size=image_shape,
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        with torch.no_grad():
            delta = (refined.detach() - tokens.detach()).float()
            raw = tokens.detach().float()
            refined_detached = refined.detach().float()
            self.last_adapter_debug = {
                'adapter_feature_delta_l2': delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_feature_delta_abs': delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': raw.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_refined_l2': refined_detached.pow(2).sum(dim=-1).sqrt().mean(dim=1),
            }
        return cssd_image_score, cssd_map

    def _localization_map(self, tokens, spatial_shape, image_shape, t_norm, t_abn, imgs=None):
        cnn_logits = self.loc_decoder(tokens, spatial_shape, image_shape)
        semantic_map, semantic_gate = self._semantic_gate(tokens, t_norm, t_abn, spatial_shape, image_shape)
        if bool(self.text_guidance_kwargs.get('enable_gate', True)):
            eta = torch.clamp(self.semantic_eta, min=0.0, max=2.0)
            anomaly_map = cnn_logits * (1.0 + eta * semantic_gate)
        else:
            anomaly_map = cnn_logits
        return anomaly_map, cnn_logits, semantic_map, semantic_gate

    def _map_topk_image_score(self, anomaly_map, imgs, ratio):
        if anomaly_map.ndim == 3:
            anomaly_map_4d = anomaly_map.unsqueeze(1)
        else:
            anomaly_map_4d = anomaly_map
        if imgs is not None:
            foreground, _, _, _ = self._foreground_masks(imgs, anomaly_map_4d.shape[-2:])
        else:
            foreground = torch.ones_like(anomaly_map_4d, dtype=torch.bool)
        flat_map = anomaly_map_4d.flatten(1)
        flat_fg = foreground.flatten(1).bool()
        scores = []
        for idx in range(flat_map.shape[0]):
            values = flat_map[idx][flat_fg[idx]]
            if values.numel() == 0:
                values = flat_map[idx]
            k = max(1, int(values.numel() * float(ratio)))
            scores.append(values.topk(k).values.mean())
        return torch.stack(scores, dim=0)

    def _compose_image_score(self, global_score, cssd_image_score, anomaly_map, imgs):
        if self.image_score_source == 'map_topk':
            return self._map_topk_image_score(anomaly_map, imgs, self.map_topk_ratio)
        if self.image_score_source == 'global_only':
            return global_score
        return global_score + self.image_score_beta * cssd_image_score

    def _image_bce_loss(self, image_score, label):
        target = torch.full_like(image_score, float(label))
        return F.binary_cross_entropy_with_logits(image_score, target)

    def forward(
            self,
            imgs,
            cls_names=None,
            score_cls_names=None,
            adapter_cls_names=None,
            return_anomaly_map=False,
            compute_label_free=True,
    ):
        del adapter_cls_names
        if score_cls_names is None:
            score_cls_names = cls_names
        if self.training and self.eval_adapter_mode != 'trained':
            raise RuntimeError('eval_adapter_mode is only for test/eval; use trained mode during training.')

        with torch.no_grad():
            image_features, tokens, spatial_shape = self.biomedclip.encode_image_and_patches(imgs)
        fixed_global_norm, fixed_global_abn = self._fixed_text_pairs(score_cls_names, imgs.shape[0], imgs.device)
        local_base_norm, local_base_abn = self._local_base_text_pairs(
            score_cls_names,
            imgs.shape[0],
            imgs.device,
            global_base_norm=fixed_global_norm,
            global_base_abn=fixed_global_abn,
        )
        t_norm, t_abn = self._local_text_pairs(
            local_base_norm,
            local_base_abn,
            cls_names=score_cls_names,
            batch_size=imgs.shape[0],
            device=imgs.device,
            tokens=tokens.detach(),
        )
        global_t_norm, global_t_abn = self._global_text_pairs(fixed_global_norm, fixed_global_abn, t_norm, t_abn)
        global_score = torch.sum(image_features.detach() * global_t_abn, dim=1) - torch.sum(image_features.detach() * global_t_norm, dim=1)
        detach_local_text_for_image = bool(self.training and self.stop_local_prompt_image_grad)
        cssd_image_score, cssd_map = self._cssd_image_branch(
            tokens.detach(),
            spatial_shape,
            (imgs.shape[2], imgs.shape[3]),
            t_norm=t_norm,
            t_abn=t_abn,
            detach_text=detach_local_text_for_image,
        )
        anomaly_map, cnn_map, semantic_map, semantic_gate = self._localization_map(
            tokens.detach(),
            spatial_shape,
            (imgs.shape[2], imgs.shape[3]),
            t_norm=t_norm,
            t_abn=t_abn,
            imgs=imgs,
        )
        image_score = self._compose_image_score(global_score, cssd_image_score, anomaly_map, imgs)
        self._record_prompt_debug(
            image_features,
            tokens,
            fixed_global_norm,
            fixed_global_abn,
            local_base_norm,
            local_base_abn,
            t_norm,
            t_abn,
            global_score,
            cssd_image_score,
            image_score,
        )

        if self.training:
            if not compute_label_free:
                return {
                    'anomaly_map': anomaly_map,
                    'image_score': image_score.detach(),
                    'global_image_score': global_score.detach(),
                    'cssd_image_score': cssd_image_score,
                    'cnn_anomaly_map': cnn_map,
                    'semantic_map': semantic_map,
                    'semantic_gate': semantic_gate,
                }
            out = self._localization_losses(anomaly_map.unsqueeze(1), imgs)
            loss_img_normal = self._image_bce_loss(cssd_image_score, 0.0)
            loss_text_reg = self._text_prototype_regularization(local_base_norm.detach(), local_base_abn.detach(), t_norm, t_abn)
            loss_prompt_bank_div = self._prompt_bank_diversity_regularization(
                local_base_norm.detach(),
                local_base_abn.detach(),
            )
            loss_prompt_class_orth = self._prompt_bank_class_orthogonal_regularization(
                fixed_global_norm.detach(),
                fixed_global_abn.detach(),
                t_norm,
                t_abn,
            )
            loss_prompt_token_norm = self._prompt_token_norm_regularization()
            loss_pathology_axis, pathology_axis_debug = self._pathology_axis_regularization(
                t_norm,
                t_abn,
                imgs.shape[0],
                imgs.device,
            )
            out['total'] = (
                out['total']
                + self.cssd_image_loss_weight * loss_img_normal
                + self.text_reg_weight * loss_text_reg
                + self.prompt_bank_diversity_weight * loss_prompt_bank_div
                + self.prompt_bank_class_orthogonal_weight * loss_prompt_class_orth
                + self.prompt_token_norm_weight * loss_prompt_token_norm
                + self.pathology_axis_loss_weight * loss_pathology_axis
            )
            out['loss_total'] = out['total']
            out['loss_cssd_image_normal'] = loss_img_normal
            out['loss_cssd_image_normal_weighted'] = self.cssd_image_loss_weight * loss_img_normal
            out['loss_text_proto_reg'] = loss_text_reg
            out['loss_text_proto_reg_weighted'] = self.text_reg_weight * loss_text_reg
            out['loss_prompt_bank_diversity'] = loss_prompt_bank_div
            out['loss_prompt_bank_diversity_weighted'] = self.prompt_bank_diversity_weight * loss_prompt_bank_div
            out['loss_prompt_class_orthogonal'] = loss_prompt_class_orth
            out['loss_prompt_class_orthogonal_weighted'] = (
                self.prompt_bank_class_orthogonal_weight * loss_prompt_class_orth
            )
            out['loss_prompt_token_norm'] = loss_prompt_token_norm
            out['loss_prompt_token_norm_weighted'] = self.prompt_token_norm_weight * loss_prompt_token_norm
            out['loss_pathology_axis'] = loss_pathology_axis
            out['loss_pathology_axis_weighted'] = self.pathology_axis_loss_weight * loss_pathology_axis
            out['cssd_image_score_mean'] = cssd_image_score.detach().mean()
            out['semantic_gate_mean'] = semantic_gate.detach().mean()
            out.update(self.last_prompt_debug)
            out.update(pathology_axis_debug)
            if return_anomaly_map:
                out.update(anomaly_map=anomaly_map, image_score=image_score.detach())
            return out

        return anomaly_map, image_score.detach()


class MAMBAADBiomedCLIPTGLRANoMamba(MAMBAADBiomedCLIPDualBranchAdapter):
    def __init__(
            self,
            model_s,
            biomedclip_model_name,
            prompt_normal,
            prompt_abnormal,
            image_size=256,
            local_loss_kwargs=None,
            text_guidance_kwargs=None,
            image_branch_kwargs=None,
            decoder_kwargs=None,
            relation_kwargs=None,
            input_mean=(0.485, 0.456, 0.406),
            input_std=(0.229, 0.224, 0.225),
            biomed_mean=(0.48145466, 0.4578275, 0.40821073),
            biomed_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        image_branch_kwargs = dict(image_branch_kwargs or {})
        image_branch_kwargs['use_cssd'] = False
        super().__init__(
            model_s=model_s,
            biomedclip_model_name=biomedclip_model_name,
            prompt_normal=prompt_normal,
            prompt_abnormal=prompt_abnormal,
            image_size=image_size,
            local_loss_kwargs=local_loss_kwargs,
            text_guidance_kwargs=text_guidance_kwargs,
            image_branch_kwargs=image_branch_kwargs,
            decoder_kwargs=decoder_kwargs,
            input_mean=input_mean,
            input_std=input_std,
            biomed_mean=biomed_mean,
            biomed_std=biomed_std,
        )
        self.relation_kwargs = dict(relation_kwargs or {})
        self.relation_branch = TextGuidedLocalRelationBranch(
            self.visual_dim,
            hidden_dim=int(self.relation_kwargs.get('hidden_dim', 256)),
            dropout=float(self.relation_kwargs.get('dropout', 0.0)),
            semantic_direction=self.relation_kwargs.get('semantic_direction', 'abnormal_minus_normal'),
        )
        self.relation_eta = nn.Parameter(torch.tensor(float(self.relation_kwargs.get('gate_eta_init', 0.1))))

        self._set_requires_grad(self.local_adapter, False)
        self._set_requires_grad(self.loc_decoder, False)
        self._set_requires_grad(self.relation_branch, True)

    def train(self, mode=True):
        self.training = mode
        self.biomedclip.eval()
        self.local_adapter.eval()
        self.loc_decoder.eval()
        self.local_head.eval()
        self.local_prompt_router.train(mode)
        self.relation_branch.train(mode)
        return self

    def adapter_param_norm(self):
        total_sq = 0.0
        total_params = 0
        for module in [self.relation_branch, self.local_prompt_router]:
            for param in module.parameters():
                if not param.is_floating_point():
                    continue
                value = param.detach().float()
                total_sq += float(torch.sum(value * value).cpu())
                total_params += value.numel()
        for param in [
            self.local_text_delta_normal,
            self.local_text_delta_abnormal,
            self.local_prompt_tokens_normal,
            self.local_prompt_tokens_abnormal,
            self.semantic_scale,
            self.semantic_bias,
            self.semantic_eta,
            self.relation_eta,
        ]:
            value = param.detach().float()
            total_sq += float(torch.sum(value * value).cpu())
            total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def reset_adapter_parameters(self, seed=None):
        params = list(self.relation_branch.parameters()) + list(self.local_prompt_router.parameters())
        device = params[0].device if params else torch.device('cpu')
        devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
            self._reset_module_parameters(self.relation_branch)
            self.local_prompt_router.reset_parameters()
            nn.init.zeros_(self.local_prompt_router.weight)
            nn.init.zeros_(self.local_prompt_router.bias)
            with torch.no_grad():
                self.local_text_delta_normal.zero_()
                self.local_text_delta_abnormal.zero_()
                if self.local_prompt_bank_size > 1 and self.local_prompt_bank_init_std > 0:
                    self.local_text_delta_normal.normal_(std=self.local_prompt_bank_init_std)
                    self.local_text_delta_abnormal.normal_(std=self.local_prompt_bank_init_std)
                if self.local_prompt_token_count > 0:
                    self.local_prompt_tokens_normal.normal_(std=self.local_prompt_token_init_std)
                    self.local_prompt_tokens_abnormal.normal_(std=self.local_prompt_token_init_std)

    def _localization_map(self, tokens, spatial_shape, image_shape, t_norm, t_abn, imgs=None):
        relation_logits, relation_tokens, semantic_map, semantic_gate = self.relation_branch(
            tokens,
            spatial_shape,
            image_shape,
            t_norm=t_norm,
            t_abn=t_abn,
        )
        if bool(self.text_guidance_kwargs.get('enable_gate', True)):
            eta = torch.clamp(self.relation_eta, min=0.0, max=2.0)
            anomaly_map = relation_logits * (1.0 + eta * semantic_gate)
        else:
            anomaly_map = relation_logits
        with torch.no_grad():
            delta = (relation_tokens.detach() - tokens.detach()).float()
            self.last_adapter_debug = {
                'adapter_feature_delta_l2': delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_feature_delta_abs': delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': tokens.detach().float().pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_refined_l2': relation_tokens.detach().float().pow(2).sum(dim=-1).sqrt().mean(dim=1),
            }
        return anomaly_map, relation_logits, semantic_map, semantic_gate


class MAMBAADBiomedCLIPTGLRAFull(MAMBAADBiomedCLIPTGLRANoMamba):
    def __init__(
            self,
            model_s,
            biomedclip_model_name,
            prompt_normal,
            prompt_abnormal,
            image_size=256,
            local_loss_kwargs=None,
            text_guidance_kwargs=None,
            image_branch_kwargs=None,
            decoder_kwargs=None,
            relation_kwargs=None,
            fusion_kwargs=None,
            input_mean=(0.485, 0.456, 0.406),
            input_std=(0.229, 0.224, 0.225),
            biomed_mean=(0.48145466, 0.4578275, 0.40821073),
            biomed_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        image_branch_kwargs = dict(image_branch_kwargs or {})
        image_branch_kwargs['use_cssd'] = True
        super().__init__(
            model_s=model_s,
            biomedclip_model_name=biomedclip_model_name,
            prompt_normal=prompt_normal,
            prompt_abnormal=prompt_abnormal,
            image_size=image_size,
            local_loss_kwargs=local_loss_kwargs,
            text_guidance_kwargs=text_guidance_kwargs,
            image_branch_kwargs=image_branch_kwargs,
            decoder_kwargs=decoder_kwargs,
            relation_kwargs=relation_kwargs,
            input_mean=input_mean,
            input_std=input_std,
            biomed_mean=biomed_mean,
            biomed_std=biomed_std,
        )
        self.use_cssd_image_branch = True
        self.fusion_kwargs = dict(fusion_kwargs or {})
        fusion_hidden_dim = int(self.fusion_kwargs.get('hidden_dim', 512))
        fusion_dropout = float(self.fusion_kwargs.get('dropout', 0.0))
        self.global_local_fusion = nn.Sequential(
            nn.Linear(self.visual_dim * 3 + 1, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout) if fusion_dropout > 0 else nn.Identity(),
            nn.Linear(fusion_hidden_dim, self.visual_dim),
            nn.GELU(),
        )
        self.fusion_head = nn.Linear(self.visual_dim, 1)
        self.fusion_residual_scale = nn.Parameter(torch.tensor(float(self.fusion_kwargs.get('residual_scale_init', 0.1))))
        nn.init.normal_(self.fusion_head.weight, std=0.02)
        nn.init.zeros_(self.fusion_head.bias)

        self._set_requires_grad(self.local_adapter, True)
        self._set_requires_grad(self.loc_decoder, False)
        self._set_requires_grad(self.global_local_fusion, True)
        self._set_requires_grad(self.fusion_head, True)

    def train(self, mode=True):
        self.training = mode
        self.biomedclip.eval()
        self.local_adapter.train(mode)
        self.loc_decoder.eval()
        self.local_head.eval()
        self.relation_branch.train(mode)
        self.global_local_fusion.train(mode)
        self.fusion_head.train(mode)
        return self

    def adapter_param_norm(self):
        total_sq = 0.0
        total_params = 0
        for module in [self.local_adapter, self.relation_branch, self.global_local_fusion, self.fusion_head, self.local_prompt_router]:
            for param in module.parameters():
                if not param.is_floating_point():
                    continue
                value = param.detach().float()
                total_sq += float(torch.sum(value * value).cpu())
                total_params += value.numel()
        for param in [
            self.local_text_delta_normal,
            self.local_text_delta_abnormal,
            self.local_prompt_tokens_normal,
            self.local_prompt_tokens_abnormal,
            self.semantic_scale,
            self.semantic_bias,
            self.semantic_eta,
            self.relation_eta,
            self.fusion_residual_scale,
        ]:
            value = param.detach().float()
            total_sq += float(torch.sum(value * value).cpu())
            total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def reset_adapter_parameters(self, seed=None):
        params = (
            list(self.local_adapter.parameters())
            + list(self.relation_branch.parameters())
            + list(self.global_local_fusion.parameters())
            + list(self.fusion_head.parameters())
            + list(self.local_prompt_router.parameters())
        )
        device = params[0].device if params else torch.device('cpu')
        devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
            self._reset_module_parameters(self.local_adapter)
            self._reset_module_parameters(self.relation_branch)
            self._reset_module_parameters(self.global_local_fusion)
            self.fusion_head.reset_parameters()
            nn.init.zeros_(self.fusion_head.bias)
            self.local_prompt_router.reset_parameters()
            nn.init.zeros_(self.local_prompt_router.weight)
            nn.init.zeros_(self.local_prompt_router.bias)
            with torch.no_grad():
                self.local_text_delta_normal.zero_()
                self.local_text_delta_abnormal.zero_()
                if self.local_prompt_bank_size > 1 and self.local_prompt_bank_init_std > 0:
                    self.local_text_delta_normal.normal_(std=self.local_prompt_bank_init_std)
                    self.local_text_delta_abnormal.normal_(std=self.local_prompt_bank_init_std)
                if self.local_prompt_token_count > 0:
                    self.local_prompt_tokens_normal.normal_(std=self.local_prompt_token_init_std)
                    self.local_prompt_tokens_abnormal.normal_(std=self.local_prompt_token_init_std)

    def _localization_map(self, tokens, spatial_shape, image_shape, t_norm, t_abn, imgs=None):
        if self.eval_adapter_mode == 'bypass':
            global_tokens = tokens
        else:
            global_tokens = self.local_adapter(tokens, None, spatial_shape)
        _, relation_tokens, semantic_map, semantic_gate = self.relation_branch(
            tokens,
            spatial_shape,
            image_shape,
            t_norm=t_norm,
            t_abn=t_abn,
        )
        semantic_patch = self.relation_branch._semantic_score(tokens, t_norm, t_abn).unsqueeze(-1)
        fusion_input = torch.cat([tokens, global_tokens, relation_tokens, semantic_patch], dim=-1)
        fusion_delta = self.global_local_fusion(fusion_input)
        residual_scale = torch.clamp(self.fusion_residual_scale, min=0.0, max=2.0)
        fused_tokens = tokens + residual_scale * fusion_delta
        patch_logits = self.fusion_head(fused_tokens).squeeze(-1)
        height, width = spatial_shape
        fused_map = patch_logits.view(patch_logits.shape[0], height, width)
        if fused_map.shape[-2:] != image_shape:
            fused_map = F.interpolate(
                fused_map.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)
        if bool(self.text_guidance_kwargs.get('enable_gate', True)):
            eta = torch.clamp(self.relation_eta, min=0.0, max=2.0)
            anomaly_map = fused_map * (1.0 + eta * semantic_gate)
        else:
            anomaly_map = fused_map
        with torch.no_grad():
            global_delta = (global_tokens.detach() - tokens.detach()).float()
            relation_delta = (relation_tokens.detach() - tokens.detach()).float()
            fused_delta = (fused_tokens.detach() - tokens.detach()).float()
            self.last_adapter_debug = {
                'adapter_feature_delta_l2': fused_delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_feature_delta_abs': fused_delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': tokens.detach().float().pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_refined_l2': fused_tokens.detach().float().pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_global_delta_l2': global_delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_relation_delta_l2': relation_delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
            }
        return anomaly_map, fused_map, semantic_map, semantic_gate


class MAMBAADBiomedCLIPCNNGlobalAuxAdapter(MAMBAADBiomedCLIPDualBranchAdapter):
    def __init__(
            self,
            model_s,
            biomedclip_model_name,
            prompt_normal,
            prompt_abnormal,
            image_size=256,
            local_loss_kwargs=None,
            text_guidance_kwargs=None,
            image_branch_kwargs=None,
            decoder_kwargs=None,
            global_aux_kwargs=None,
            arcc_kwargs=None,
            input_mean=(0.485, 0.456, 0.406),
            input_std=(0.229, 0.224, 0.225),
            biomed_mean=(0.48145466, 0.4578275, 0.40821073),
            biomed_std=(0.26862954, 0.26130258, 0.27577711),
    ):
        image_branch_kwargs = dict(image_branch_kwargs or {})
        image_branch_kwargs['use_cssd'] = True
        super().__init__(
            model_s=model_s,
            biomedclip_model_name=biomedclip_model_name,
            prompt_normal=prompt_normal,
            prompt_abnormal=prompt_abnormal,
            image_size=image_size,
            local_loss_kwargs=local_loss_kwargs,
            text_guidance_kwargs=text_guidance_kwargs,
            image_branch_kwargs=image_branch_kwargs,
            decoder_kwargs=decoder_kwargs,
            input_mean=input_mean,
            input_std=input_std,
            biomed_mean=biomed_mean,
            biomed_std=biomed_std,
        )
        self.global_aux_kwargs = dict(global_aux_kwargs or {})
        self.arcc_kwargs = dict(arcc_kwargs or {})
        self.global_gate_scale = nn.Parameter(torch.tensor(float(self.global_aux_kwargs.get('gate_scale_init', 1.0))))
        self.global_gate_bias = nn.Parameter(torch.tensor(float(self.global_aux_kwargs.get('gate_bias_init', 0.0))))
        self.global_eta = nn.Parameter(torch.tensor(float(self.global_aux_kwargs.get('gate_eta_init', 0.05))))
        self.use_arcc = bool(self.arcc_kwargs.get('use_arcc', False))
        self.arcc_lambda = nn.Parameter(torch.tensor(float(self.arcc_kwargs.get('lambda_init', 0.1))))
        if self.use_arcc:
            self.arcc = ARCCCalibration(
                self.visual_dim,
                use_response=bool(self.arcc_kwargs.get('use_response', True)),
                use_foreground=bool(self.arcc_kwargs.get('use_foreground', True)),
                use_edge=bool(self.arcc_kwargs.get('use_edge', True)),
                kernel_size=int(self.arcc_kwargs.get('kernel_size', 3)),
                hidden_dim=self.arcc_kwargs.get('hidden_dim', None),
                lambda_init=float(self.arcc_kwargs.get('lambda_init', 0.1)),
            )
        else:
            self.arcc = None
        self._cached_global_tokens = None
        self._cached_global_patch_scores = None

    def train(self, mode=True):
        super().train(mode)
        if self.arcc is not None:
            self.arcc.train(mode)
        return self

    def adapter_param_norm(self):
        total_sq, total_params = super().adapter_param_norm()
        total_sq = total_sq * total_sq
        for param in [self.global_gate_scale, self.global_gate_bias, self.global_eta, self.arcc_lambda]:
            value = param.detach().float()
            total_sq += float(torch.sum(value * value).cpu())
            total_params += value.numel()
        if self.arcc is not None:
            for param in self.arcc.parameters():
                if not param.is_floating_point():
                    continue
                value = param.detach().float()
                total_sq += float(torch.sum(value * value).cpu())
                total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def reset_adapter_parameters(self, seed=None):
        super().reset_adapter_parameters(seed=seed)
        if self.arcc is None:
            return
        params = list(self.arcc.parameters())
        device = params[0].device if params else torch.device('cpu')
        devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
            self._reset_module_parameters(self.arcc)

    def _cssd_image_branch(self, tokens, spatial_shape, image_shape, t_norm, t_abn, detach_text=False):
        if self.eval_adapter_mode == 'bypass':
            refined = tokens
        else:
            refined = self.local_adapter(tokens, None, spatial_shape)
        if detach_text:
            t_norm = t_norm.detach()
            t_abn = t_abn.detach()
        sim_normal = self._text_similarity(refined, t_norm)
        sim_abnormal = self._text_similarity(refined, t_abn)
        patch_scores = sim_abnormal - sim_normal
        topk = max(1, int(patch_scores.shape[1] * self.cssd_topk_ratio))
        cssd_image_score = patch_scores.topk(topk, dim=1).values.mean(dim=1)
        height, width = spatial_shape
        cssd_map = patch_scores.view(patch_scores.shape[0], height, width)
        cssd_map = F.interpolate(
            cssd_map.unsqueeze(1),
            size=image_shape,
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        self._cached_global_tokens = refined
        self._cached_global_patch_scores = None if detach_text else patch_scores
        with torch.no_grad():
            delta = (refined.detach() - tokens.detach()).float()
            raw = tokens.detach().float()
            refined_detached = refined.detach().float()
            self.last_adapter_debug = {
                'adapter_feature_delta_l2': delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_feature_delta_abs': delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': raw.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_refined_l2': refined_detached.pow(2).sum(dim=-1).sqrt().mean(dim=1),
            }
        return cssd_image_score, cssd_map

    def _tokens_to_feature_map(self, tokens, spatial_shape):
        height, width = spatial_shape
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], height, width)

    def _apply_arcc(self, cnn_logits, global_tokens, spatial_shape, image_shape, imgs):
        feature_map = self._tokens_to_feature_map(global_tokens, spatial_shape)
        foreground = edge = None
        if imgs is not None:
            foreground, _, edge, _ = self._foreground_masks(imgs, spatial_shape)
        g_cal, mod_mask = self.arcc(
            feature_map,
            cnn_logits,
            foreground=foreground,
            edge=edge,
            image_shape=image_shape,
        )
        arcc_lambda = torch.clamp(self.arcc_lambda, min=0.0, max=2.0)
        anomaly_map = cnn_logits + arcc_lambda * cnn_logits * torch.tanh(g_cal)
        return anomaly_map, g_cal, mod_mask, arcc_lambda

    def _localization_map(self, tokens, spatial_shape, image_shape, t_norm, t_abn, imgs=None):
        cnn_logits = self.loc_decoder(tokens, spatial_shape, image_shape)
        semantic_map, semantic_gate = self._semantic_gate(tokens, t_norm, t_abn, spatial_shape, image_shape)

        global_tokens = self._cached_global_tokens
        global_patch_scores = self._cached_global_patch_scores
        if global_tokens is None:
            if self.eval_adapter_mode == 'bypass':
                global_tokens = tokens
            else:
                global_tokens = self.local_adapter(tokens, None, spatial_shape)
        if global_patch_scores is None:
            sim_normal = self._text_similarity(global_tokens, t_norm)
            sim_abnormal = self._text_similarity(global_tokens, t_abn)
            global_patch_scores = sim_abnormal - sim_normal

        height, width = spatial_shape
        global_gate = torch.sigmoid(
            self.global_gate_scale * global_patch_scores.view(global_patch_scores.shape[0], height, width)
            + self.global_gate_bias
        )
        if global_gate.shape[-2:] != image_shape:
            global_gate = F.interpolate(
                global_gate.unsqueeze(1),
                size=image_shape,
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

        g_cal = None
        arcc_mod_mask = None
        arcc_lambda = None
        if self.use_arcc and self.arcc is not None:
            anomaly_map, g_cal, arcc_mod_mask, arcc_lambda = self._apply_arcc(
                cnn_logits,
                global_tokens,
                spatial_shape,
                image_shape,
                imgs,
            )
        elif bool(self.text_guidance_kwargs.get('enable_gate', True)):
            semantic_eta = torch.clamp(self.semantic_eta, min=0.0, max=2.0)
            global_eta = torch.clamp(self.global_eta, min=0.0, max=2.0)
            anomaly_map = cnn_logits * (1.0 + semantic_eta * semantic_gate + global_eta * global_gate)
        else:
            anomaly_map = cnn_logits

        with torch.no_grad():
            global_delta = (global_tokens.detach() - tokens.detach()).float()
            raw = tokens.detach().float()
            self.last_adapter_debug.update({
                'adapter_global_delta_l2': global_delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_global_delta_abs': global_delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': raw.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'global_gate_mean': global_gate.detach().flatten(1).mean(dim=1),
                'arcc_enabled': torch.full(
                    (tokens.shape[0],),
                    float(self.use_arcc),
                    device=tokens.device,
                    dtype=tokens.dtype,
                ),
            })
            if g_cal is not None:
                self.last_adapter_debug.update({
                    'arcc_calibration_mean': g_cal.detach().flatten(1).mean(dim=1),
                    'arcc_calibration_abs_mean': g_cal.detach().abs().flatten(1).mean(dim=1),
                    'arcc_modulation_mean': arcc_mod_mask.detach().flatten(1).mean(dim=1),
                    'arcc_lambda': torch.full(
                        (tokens.shape[0],),
                        float(arcc_lambda.detach().cpu()),
                        device=tokens.device,
                        dtype=tokens.dtype,
                    ),
                })
        return anomaly_map, cnn_logits, semantic_map, semantic_gate


class MAMBAADZeroShot(nn.Module):
    def __init__(self, model_t, model_s, biomedclip_model_name, prompt_normal, prompt_abnormal, image_size=256,
                 adaptive_mc_kwargs=None, class_prompt_template='A medical image of {class_name}', class_prompts=None,
                 image_score_topk_ratio=0.01, anomaly_score_direction='normal_minus_abnormal'):
        super().__init__()
        self.visual_encoder = FrozenVisualSequenceEncoder(get_model(model_t))
        self.text_encoder = FrozenBiomedTextEncoder(
            biomedclip_model_name,
            prompt_normal=prompt_normal,
            prompt_abnormal=prompt_abnormal,
            class_prompt_template=class_prompt_template,
            class_prompts=class_prompts,
        )

        visual_dim, spatial_shape = self._infer_visual_spec(image_size)
        if visual_dim != self.text_encoder.text_dim:
            raise ValueError(
                f'Visual feature dim ({visual_dim}) must match text dim ({self.text_encoder.text_dim}). '
                'Use a visual encoder stage whose channel width matches BiomedCLIP.'
            )
        if spatial_shape[0] != spatial_shape[1]:
            raise ValueError(f'CSSD requires a square feature grid, got {spatial_shape}.')

        self.grid_size = spatial_shape[0]
        self.cssd = CSSD(
            hidden_dim=visual_dim,
            grid_size=self.grid_size,
            depths=model_s.get('depths_decoder', [3, 4, 6, 3]),
            d_state=model_s.get('d_state', 16),
            drop_path_rate=model_s.get('drop_path_rate', 0.2),
            attn_drop_rate=model_s.get('attn_drop_rate', 0.0),
            scan_type=model_s.get('scan_type', 'scan'),
            num_direction=model_s.get('num_direction', 8),
            use_selective_scan=model_s.get('use_selective_scan', True),
            use_deformable_pool=model_s.get('use_deformable_pool', True),
        )
        self.label_free_loss = AdaptiveMCLoss(**(adaptive_mc_kwargs or {}))
        self.image_score_topk_ratio = image_score_topk_ratio
        self.anomaly_score_direction = self._normalize_anomaly_score_direction(anomaly_score_direction)
        self.eval_adapter_mode = 'trained'
        self.last_adapter_debug = {}

        self._freeze_module(self.visual_encoder)
        self._freeze_module(self.text_encoder)
        self._set_requires_grad(self.cssd, True)

    def _infer_visual_spec(self, image_size):
        was_training = self.visual_encoder.training
        self.visual_encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            _, spatial_shape, visual_dim = self.visual_encoder(dummy)
        self.visual_encoder.train(was_training)
        return visual_dim, spatial_shape

    def _freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def _set_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            # 只有浮点数或复数才允许开启梯度求导，跳过底层的整数索引(LongTensor)
            if param.is_floating_point() or param.is_complex():
                param.requires_grad = requires_grad

    def _normalize_anomaly_score_direction(self, direction):
        direction = str(direction).lower()
        aliases = {
            'normal_minus_abnormal': 'normal_minus_abnormal',
            'normal-abnormal': 'normal_minus_abnormal',
            'normal_abnormal': 'normal_minus_abnormal',
            'abnormal_minus_normal': 'abnormal_minus_normal',
            'abnormal-normal': 'abnormal_minus_normal',
            'abnormal_normal': 'abnormal_minus_normal',
        }
        if direction not in aliases:
            raise ValueError(
                f'Invalid anomaly_score_direction={direction}. '
                'Expected normal_minus_abnormal or abnormal_minus_normal.'
            )
        return aliases[direction]

    def set_eval_adapter_mode(self, mode):
        mode = str(mode).lower()
        if mode not in ('trained', 'bypass', 'random'):
            raise ValueError(f'Invalid eval_adapter_mode={mode}. Expected trained, bypass, or random.')
        self.eval_adapter_mode = mode

    def reset_adapter_parameters(self, seed=None):
        device = next(self.cssd.parameters()).device
        devices = [device.index] if device.type == 'cuda' and device.index is not None else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(int(seed))
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
            self._reset_module_parameters(self.cssd)

    def _reset_module_parameters(self, module):
        for child in module.children():
            self._reset_module_parameters(child)
        reset = getattr(module, 'reset_parameters', None)
        if callable(reset):
            reset()

    def adapter_param_norm(self):
        total_sq = 0.0
        total_params = 0
        for param in self.cssd.parameters():
            if not param.is_floating_point():
                continue
            value = param.detach().float()
            total_sq += float(torch.sum(value * value).cpu())
            total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def train(self, mode=True):
        self.training = mode
        self.visual_encoder.eval()
        self.text_encoder.eval()
        self.cssd.train(mode)
        self.label_free_loss.train(mode)
        return self

    def _anomaly_outputs(self, v_refined, t_norm, t_abn, spatial_shape, image_shape):
        v_refined = F.normalize(v_refined, p=2, dim=-1)
        t_norm = F.normalize(t_norm, p=2, dim=-1)
        t_abn = F.normalize(t_abn, p=2, dim=-1)
        sim_normal = torch.einsum('bld,bd->bl', v_refined, t_norm)
        sim_abnormal = torch.einsum('bld,bd->bl', v_refined, t_abn)
        if self.anomaly_score_direction == 'abnormal_minus_normal':
            scores = sim_abnormal - sim_normal
        else:
            scores = sim_normal - sim_abnormal
        height, width = spatial_shape
        anomaly_map = scores.view(scores.shape[0], height, width)
        anomaly_map = F.interpolate(
            anomaly_map.unsqueeze(1),
            size=image_shape,
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        flat_map = anomaly_map.flatten(1)
        if self.image_score_topk_ratio is None:
            image_score = flat_map.max(dim=1).values
        else:
            topk = max(1, int(flat_map.shape[1] * self.image_score_topk_ratio))
            image_score = flat_map.topk(topk, dim=1).values.mean(dim=1)
        return anomaly_map, image_score

    def forward(
            self,
            imgs,
            cls_names=None,
            score_cls_names=None,
            adapter_cls_names=None,
            return_anomaly_map=False,
            compute_label_free=True,
    ):
        if score_cls_names is None:
            score_cls_names = cls_names
        if adapter_cls_names is None:
            adapter_cls_names = cls_names
        with torch.no_grad():
            v_raw, spatial_shape, _ = self.visual_encoder(imgs)
            t_norm, t_abn = self.text_encoder(score_cls_names, batch_size=imgs.shape[0])
            # Frozen BiomedCLIP semantics are the only condition vector used by
            # AdaLN. This keeps semantic conditioning aligned with medical
            # language priors instead of source-domain learned embeddings.
            semantic_embedding = self.text_encoder.encode_class_prompts(
                cls_names=adapter_cls_names,
                batch_size=imgs.shape[0],
                device=imgs.device,
            )

        if self.training and self.eval_adapter_mode != 'trained':
            raise RuntimeError('eval_adapter_mode is only for test/eval; use trained mode during training.')

        # Semantic-conditioning pipeline:
        # BiomedCLIP text prior -> AdaLN modulation in LSS/HSS blocks ->
        # CSSD spatial refinement -> anomaly localization.
        if self.eval_adapter_mode == 'bypass':
            v_refined = v_raw
        else:
            v_refined = self.cssd(v_raw, semantic_embedding, spatial_shape)
        with torch.no_grad():
            delta = (v_refined.detach() - v_raw.detach()).float()
            raw = v_raw.detach().float()
            refined = v_refined.detach().float()
            self.last_adapter_debug = {
                'adapter_feature_delta_l2': delta.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_feature_delta_abs': delta.abs().mean(dim=(1, 2)),
                'adapter_raw_l2': raw.pow(2).sum(dim=-1).sqrt().mean(dim=1),
                'adapter_refined_l2': refined.pow(2).sum(dim=-1).sqrt().mean(dim=1),
            }
        f_global = v_refined.mean(dim=1)

        if self.training:
            if not compute_label_free:
                anomaly_map, image_score = self._anomaly_outputs(
                    v_refined,
                    t_norm,
                    t_abn,
                    spatial_shape,
                    (imgs.shape[2], imgs.shape[3]),
                )
                return {
                    'anomaly_map': anomaly_map,
                    'image_score': image_score,
                }
            total, normal_align, margin, token_consistency, stats = self.label_free_loss(
                v_refined,
                v_raw,
                t_norm,
                t_abn,
                f_global=f_global,
            )
            out = {
                'total': total,
                'loss_total': total,
                'normal_align': normal_align,
                'loss_normal_align': normal_align,
                'margin': margin,
                'loss_adaptive_margin': margin,
                'token_consistency': token_consistency,
                'loss_token_consistency': token_consistency,
                **stats,
            }
            if return_anomaly_map:
                anomaly_map, image_score = self._anomaly_outputs(
                    v_refined,
                    t_norm,
                    t_abn,
                    spatial_shape,
                    (imgs.shape[2], imgs.shape[3]),
                )
                out.update(anomaly_map=anomaly_map, image_score=image_score)
            return out

        return self._anomaly_outputs(
            v_refined,
            t_norm,
            t_abn,
            spatial_shape,
            (imgs.shape[2], imgs.shape[3]),
        )


@MODEL.register_module
def mambaad_biomedclip_local_adapter(pretrained=False, **kwargs):
    model = MAMBAADBiomedCLIPLocalAdapter(**kwargs)
    return model


@MODEL.register_module
def mambaad_biomedclip_dual_branch_adapter(pretrained=False, **kwargs):
    model = MAMBAADBiomedCLIPDualBranchAdapter(**kwargs)
    return model


@MODEL.register_module
def mambaad_biomedclip_tglra_no_mamba(pretrained=False, **kwargs):
    model = MAMBAADBiomedCLIPTGLRANoMamba(**kwargs)
    return model


@MODEL.register_module
def mambaad_biomedclip_tglra_full(pretrained=False, **kwargs):
    model = MAMBAADBiomedCLIPTGLRAFull(**kwargs)
    return model


@MODEL.register_module
def mambaad_biomedclip_cnn_global_aux_adapter(pretrained=False, **kwargs):
    model = MAMBAADBiomedCLIPCNNGlobalAuxAdapter(**kwargs)
    return model


@MODEL.register_module
def mambaad_zsad(pretrained=False, **kwargs):
    model = MAMBAADZeroShot(**kwargs)
    return model
