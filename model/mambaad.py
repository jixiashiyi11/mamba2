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
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, size=size,
                                   scan_type=scan_type, num_direction=num_direction, **kwargs)
        self.drop_path = DropPath(drop_path)

        cond_dim = 512
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, input: torch.Tensor, c=None):
        x_norm = self.ln_1(input)

        if c is not None:
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
            **kwargs,
    ):
        super().__init__()
        self.smm_blocks = nn.ModuleList([
            HSSBlock(hidden_dim=hidden_dim, drop_path=drop_path, norm_layer=norm_layer, attn_drop_rate=attn_drop_rate,
                     d_state=d_state, size=size, scan_type=scan_type, num_direction=num_direction, **kwargs)
            for i in range(depth)])

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
        out_ssm = input

        for blk in self.smm_blocks:
            out_ssm = blk(out_ssm, c)

        out_ssm_permuted = out_ssm.permute(0, 3, 1, 2).contiguous()

        if pool_feat is not None:
            v_pool = pool_feat.permute(0, 3, 1, 2).contiguous()
        else:
            v_pool = input.permute(0, 3, 1, 2).contiguous()

        q = self.query_norm(out_ssm_permuted)
        deform_residual = self.deform_attn(x_query=q, x_pool=v_pool)
        deform_residual = self.deform_act(deform_residual)

        output = out_ssm_permuted + deform_residual

        output = output.permute(0, 2, 3, 1).contiguous()
        return output + input

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


class CSSD(nn.Module):
    def __init__(self, hidden_dim, grid_size, depths=(3, 4, 6, 3), d_state=16, drop_path_rate=0.2,
                 attn_drop_rate=0.0, scan_type='scan', num_direction=8):
        super().__init__()
        if not isinstance(depths, (list, tuple)) or len(depths) == 0:
            raise ValueError('`depths` must be a non-empty list or tuple.')

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

    def _resolve_prompt(self, prompt_map, cls_name):
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

        if '{cls_name}' in prompt:
            return prompt.format(cls_name=key)
        if '{class_name}' in prompt:
            return prompt.format(class_name=key)
        return prompt

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

    def encode_text_pairs(self, cls_names, batch_size, device):
        cls_names = self._expand_cls_names(cls_names, batch_size)
        normal_prompts = [self._resolve_prompt(self.normal_prompt_map, name) for name in cls_names]
        abnormal_prompts = [self._resolve_prompt(self.abnormal_prompt_map, name) for name in cls_names]
        tokens = self.tokenizer(normal_prompts + abnormal_prompts).to(device)
        features = F.normalize(self.model.encode_text(tokens), p=2, dim=-1)
        return features[:batch_size], features[batch_size:]

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
        self.text_delta_normal = nn.Parameter(torch.zeros(self.visual_dim))
        self.text_delta_abnormal = nn.Parameter(torch.zeros(self.visual_dim))
        self.semantic_scale = nn.Parameter(torch.tensor(float(self.text_guidance_kwargs.get('gate_scale_init', 1.0))))
        self.semantic_bias = nn.Parameter(torch.tensor(float(self.text_guidance_kwargs.get('gate_bias_init', 0.0))))
        self.semantic_eta = nn.Parameter(torch.tensor(float(self.text_guidance_kwargs.get('gate_eta_init', 0.1))))
        self.image_score_beta = float(self.image_branch_kwargs.get('image_score_beta', 0.25))
        self.cssd_topk_ratio = float(self.image_branch_kwargs.get('topk_ratio', 0.05))
        self.cssd_image_loss_weight = float(self.image_branch_kwargs.get('loss_weight', 0.1))
        self.text_reg_weight = float(self.text_guidance_kwargs.get('prototype_reg_weight', 0.05))

        self._set_requires_grad(self.loc_decoder, True)
        self._set_requires_grad(self.local_adapter, True)
        # The old linear head is kept for checkpoint compatibility but is not used by this v2 route.
        self._set_requires_grad(self.local_head, False)

    def train(self, mode=True):
        self.training = mode
        self.biomedclip.eval()
        self.local_adapter.train(mode)
        self.loc_decoder.train(mode)
        self.local_head.eval()
        return self

    def adapter_param_norm(self):
        total_sq = 0.0
        total_params = 0
        for module in [self.local_adapter, self.loc_decoder]:
            for param in module.parameters():
                if not param.is_floating_point():
                    continue
                value = param.detach().float()
                total_sq += float(torch.sum(value * value).cpu())
                total_params += value.numel()
        for param in [self.text_delta_normal, self.text_delta_abnormal, self.semantic_scale, self.semantic_bias, self.semantic_eta]:
            value = param.detach().float()
            total_sq += float(torch.sum(value * value).cpu())
            total_params += value.numel()
        return math.sqrt(total_sq), total_params

    def _learnable_text_pairs(self, t_norm, t_abn):
        delta_norm = self.text_delta_normal.to(device=t_norm.device, dtype=t_norm.dtype).unsqueeze(0)
        delta_abn = self.text_delta_abnormal.to(device=t_abn.device, dtype=t_abn.dtype).unsqueeze(0)
        learn_norm = F.normalize(t_norm + delta_norm, p=2, dim=-1)
        learn_abn = F.normalize(t_abn + delta_abn, p=2, dim=-1)
        return learn_norm, learn_abn

    def _text_prototype_regularization(self, t_norm, t_abn, learn_norm, learn_abn):
        loss_norm = 1.0 - torch.sum(F.normalize(t_norm, p=2, dim=-1) * learn_norm, dim=-1)
        loss_abn = 1.0 - torch.sum(F.normalize(t_abn, p=2, dim=-1) * learn_abn, dim=-1)
        return (loss_norm + loss_abn).mean()

    def _semantic_gate(self, tokens, t_norm, t_abn, spatial_shape, image_shape):
        sim_normal = torch.einsum('bld,bd->bl', tokens, F.normalize(t_norm, p=2, dim=-1))
        sim_abnormal = torch.einsum('bld,bd->bl', tokens, F.normalize(t_abn, p=2, dim=-1))
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

    def _cssd_image_branch(self, tokens, spatial_shape, image_shape, t_norm, t_abn):
        if self.eval_adapter_mode == 'bypass':
            refined = tokens
        else:
            refined = self.local_adapter(tokens, None, spatial_shape)
        sim_normal = torch.einsum('bld,bd->bl', refined, F.normalize(t_norm, p=2, dim=-1))
        sim_abnormal = torch.einsum('bld,bd->bl', refined, F.normalize(t_abn, p=2, dim=-1))
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

    def _localization_map(self, tokens, spatial_shape, image_shape, t_norm, t_abn):
        cnn_logits = self.loc_decoder(tokens, spatial_shape, image_shape)
        semantic_map, semantic_gate = self._semantic_gate(tokens, t_norm, t_abn, spatial_shape, image_shape)
        if bool(self.text_guidance_kwargs.get('enable_gate', True)):
            eta = torch.clamp(self.semantic_eta, min=0.0, max=2.0)
            anomaly_map = cnn_logits * (1.0 + eta * semantic_gate)
        else:
            anomaly_map = cnn_logits
        return anomaly_map, cnn_logits, semantic_map, semantic_gate

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
            base_t_norm, base_t_abn = self.biomedclip.encode_text_pairs(
                score_cls_names,
                batch_size=imgs.shape[0],
                device=imgs.device,
            )
        t_norm, t_abn = self._learnable_text_pairs(base_t_norm.detach(), base_t_abn.detach())
        global_score = torch.sum(image_features.detach() * t_abn, dim=1) - torch.sum(image_features.detach() * t_norm, dim=1)
        cssd_image_score, cssd_map = self._cssd_image_branch(
            tokens.detach(),
            spatial_shape,
            (imgs.shape[2], imgs.shape[3]),
            t_norm=t_norm,
            t_abn=t_abn,
        )
        anomaly_map, cnn_map, semantic_map, semantic_gate = self._localization_map(
            tokens.detach(),
            spatial_shape,
            (imgs.shape[2], imgs.shape[3]),
            t_norm=t_norm,
            t_abn=t_abn,
        )
        image_score = global_score + self.image_score_beta * cssd_image_score

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
            loss_text_reg = self._text_prototype_regularization(base_t_norm.detach(), base_t_abn.detach(), t_norm, t_abn)
            out['total'] = out['total'] + self.cssd_image_loss_weight * loss_img_normal + self.text_reg_weight * loss_text_reg
            out['loss_total'] = out['total']
            out['loss_cssd_image_normal'] = loss_img_normal
            out['loss_cssd_image_normal_weighted'] = self.cssd_image_loss_weight * loss_img_normal
            out['loss_text_proto_reg'] = loss_text_reg
            out['loss_text_proto_reg_weighted'] = self.text_reg_weight * loss_text_reg
            out['cssd_image_score_mean'] = cssd_image_score.detach().mean()
            out['semantic_gate_mean'] = semantic_gate.detach().mean()
            if return_anomaly_map:
                out.update(anomaly_map=anomaly_map, image_score=image_score.detach())
            return out

        return anomaly_map, image_score.detach()


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
def mambaad_zsad(pretrained=False, **kwargs):
    model = MAMBAADZeroShot(**kwargs)
    return model
