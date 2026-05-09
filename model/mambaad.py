
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

# 尝试导入 load_state_dict_from_url
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# 假设这些是你项目本地的依赖
from model import get_model
from model import MODEL
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from hilbert import decode, encode
from pyzorder import ZOrderIndexer


# ==============================================================================
# 基础卷积组件 (Base Convolutional Components)
# ==============================================================================
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def deconv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False,
                              dilation=dilation)


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
# Mamba 核心组件 (Mamba Core Components)
# ==============================================================================
class HSCANS(nn.Module):
    def __init__(self, size=16, dim=2, scan_type='scan'):
        super().__init__()
        size = int(size)
        max_num = size ** dim
        indexes = np.arange(max_num)

        if 'sweep' == scan_type:  # ['sweep', 'scan', 'zorder', 'zigzag', 'hilbert']
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

        # ====================================================
        # 💡 AdaLN 控制台：生成通道级别的 gamma 和 beta
        # ====================================================
        cond_dim = 512
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2, bias=True)
        )
        # 顶级技巧：零初始化
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, input: torch.Tensor, c=None):
        # 1. 过普通的安检门
        x_norm = self.ln_1(input)

        # 2. 💡 如果有器官密码传过来，开启 AdaLN 通道洗脑！
        if c is not None:
            gamma_c, beta_c = self.adaLN_modulation(c).chunk(2, dim=1)
            gamma_c = gamma_c.unsqueeze(1).unsqueeze(1)
            beta_c = beta_c.unsqueeze(1).unsqueeze(1)
            x_norm = x_norm * (1 + gamma_c) + beta_c

        # 3. Mamba 戴着滤镜去扫图找病灶
        x = input + self.drop_path(self.self_attention(x_norm))
        return x


# ==============================================================================
# 跨层动态与可变形注意力模块 (Deformable Cross-Layer Attention Modules)
# ==============================================================================
class SpatialAdaptiveAffineGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.conv_gamma = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_beta = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.zeros_(self.conv_beta.bias)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.l1_penalty = 0.0

    def forward(self, f_global, cnn_feat):
        gamma_spatial = self.conv_gamma(f_global)
        beta_spatial = self.conv_beta(f_global)
        cnn_norm = self.norm(cnn_feat)

        max_cnn, _ = torch.max(cnn_norm, dim=1, keepdim=True)
        avg_cnn = torch.mean(cnn_norm, dim=1, keepdim=True)

        max_global, _ = torch.max(f_global, dim=1, keepdim=True)
        avg_global = torch.mean(f_global, dim=1, keepdim=True)

        pool_concat = torch.cat([max_cnn, avg_cnn, max_global, avg_global], dim=1)

        w_xy = self.spatial_attention(pool_concat)

        self.l1_penalty = torch.abs(gamma_spatial).mean() + torch.abs(beta_spatial).mean()

        modulation_term = gamma_spatial * cnn_norm + beta_spatial

        # 保底机制
        w_xy_soft = 0.2 + 0.8 * w_xy

        out = cnn_norm + w_xy_soft * modulation_term
        return out

import torch
import torch.nn as nn
import torchvision.ops as ops

class DeformableAttnRes(nn.Module):
    """
    带瓶颈设计 (Bottleneck) 的跨层可变形注意力残差模块
    """
    # 🌟 重点 1：这里加上了 ratio 参数！
    def __init__(self, channels, ratio=0.5, kernel_size=3):
        super().__init__()
        self.channels = channels
        # 根据 ratio 算出内部通道数 (沙漏最窄的地方)
        self.inner_channels = int(channels * ratio)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 🌟 重点 2：降维安检门 (大幅压缩通道，逼迫网络抛弃背景噪点)
        self.reduce_conv = nn.Conv2d(channels, self.inner_channels, kernel_size=1, bias=False)
        self.reduce_norm = nn.InstanceNorm2d(self.inner_channels)

        # 司令部预测网络 (注意：这里的输入变成了压缩后的 inner_channels)
        self.offset_mask_conv = nn.Conv2d(
            self.inner_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=self.padding
        )

        # 核心 DeformConv (在极其纯净的 inner_channels 里抓取)
        self.deform_conv = ops.DeformConv2d(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False
        )

        # 🌟 重点 3：升维出口 (把抓回来的纯净特征放大回原通道数)
        self.expand_conv = nn.Conv2d(self.inner_channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(0.1)

        # 初始化权重为 0，保证初始状态下残差分支不干扰主干道
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)
        nn.init.zeros_(self.expand_conv.weight)

    def forward(self, x_query, x_pool=None):
        if x_pool is None:
            x_pool = x_query

        # A. 探长和干警同时“瘦身” (经过降维安检门，抛弃噪点)
        q_reduced = self.reduce_norm(self.reduce_conv(x_query))
        v_reduced = self.reduce_norm(self.reduce_conv(x_pool))

        # B. 探长下达指令 (基于精简后的特征)
        out = self.offset_mask_conv(q_reduced)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # C. 干警出击抓取 (只在纯净的通道里抓)
        fused_feature = self.deform_conv(v_reduced, offset, mask)

        # D. 带着战利品“膨胀”回原维度 (为了和原通道完美相加)
        out_res = self.expand_conv(fused_feature)
        out_res = self.dropout(out_res)

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

        # 🗑️ 历史包袱全部清空：没有 CNN，没有 Concat，没有 1x1 降维卷积！

        # ✨ 终极形态：Deformable Cross-Layer Module (跨层形变模块)
        # 将原有的 Sequential 拆开，方便我们分别传入 query 和 pool
        self.query_norm = nn.InstanceNorm2d(hidden_dim)
        self.deform_attn = DeformableAttnRes(channels=hidden_dim, ratio=0.25, kernel_size=3)
        self.deform_act = nn.SiLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # 💡 核心改动：增加 pool_feat 参数，接收来自浅层的特征
    def forward(self, input: torch.Tensor, c=None, pool_feat=None):
        out_ssm = input

        # Mamba 绿线：主干道不断向前推进
        for blk in self.smm_blocks:
            out_ssm = blk(out_ssm, c)

        # 深层特征 (作为探长司令部)
        out_ssm_permuted = out_ssm.permute(0, 3, 1, 2).contiguous()

        # 📦 获取浅层特征池 (如果没传，说明是第一层，就抓自己)
        if pool_feat is not None:
            v_pool = pool_feat.permute(0, 3, 1, 2).contiguous()
        else:
            v_pool = input.permute(0, 3, 1, 2).contiguous()

        # ==========================================
        # 🎯 执行 Deformable AttnRes (对应图纸 1,2,3步)
        # ==========================================
        # 1. Learned Query: 深层特征归一化，准备发号施令
        q = self.query_norm(out_ssm_permuted)

        # 2. Deformable Sampling: 深层做 Query，去浅层池(v_pool)里抓高清像素
        deform_residual = self.deform_attn(x_query=q, x_pool=v_pool)
        deform_residual = self.deform_act(deform_residual)

        # 3. Weighting & Summation (纯粹的残差相加 ⊕)
        output = out_ssm_permuted + deform_residual

        # 维度还原
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

        # 🌟 建立 Dynamic Feature Pool (动态特征池)
        dynamic_feature_pool = None

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                # 传入 pool
                x = checkpoint.checkpoint(blk, x, c, dynamic_feature_pool)
            else:
                # 传入 pool
                x = blk(x, c, dynamic_feature_pool)

            # 📥 关键操作：在第一层 (Layer 1) 跑完后，把它的高清输出存入特征池！
            # 这样后面的 Layer 2, Layer 3 就能跨层去抓取它的特征了
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
    def __init__(self, model_t, model_s):
        super(MAMBAAD, self).__init__()
        self.net_t = get_model(model_t)
        self.mff_oce = MFF_OCE(Bottleneck, 3)
        self.net_s = MambaUPNet(depths_decoder=model_s['depths_decoder'], scan_type=model_s['scan_type'],
                                num_direction=model_s['num_direction'])

        self.frozen_layers = ['net_t']

        self.cond_dim = 512
        self.class_embedding = nn.Embedding(3, self.cond_dim)
        self.name_to_id = {'brain': 0, 'liver': 1, 'retinal': 2}

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

    def forward(self, imgs, cls_names=None):
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        fused_feats = self.mff_oce(feats_t)

        c_embed = None
        if cls_names is not None:
            class_ids = [self.name_to_id.get(str(name).lower(), 0) for name in cls_names]
            class_ids = torch.tensor(class_ids, dtype=torch.long, device=imgs.device)
            c_embed = self.class_embedding(class_ids)

        feats_s = self.net_s(fused_feats, c_embed)
        return feats_t, feats_s


@MODEL.register_module
def mambaad(pretrained=False, **kwargs):
    model = MAMBAAD(**kwargs)
    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params

    vmunet = MambaUPNet([512, 256, 128, 64], [3, 4, 6, 3])
    bs = 1
    reso = 8
    x = torch.randn(bs, 512, reso, reso).cuda()
    net = vmunet.cuda()
    net.eval()
    y = net(x)
    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6

    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
        print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(
            flops, params, bs * cnt / (t_e - t_s)))
