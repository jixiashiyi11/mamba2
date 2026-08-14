"""
PDAR-LSS 核心代码说明版
=========================

这个文件用于发给 GPT 或给老师查看“PDAR 的代码逻辑”。
它对应项目中 model/mambaad.py 的 DepthRMSNorm、
DepthAttentionResidual 和 PDARCSSD。

名称：PDAR-LSS = Patch-wise Depth-Attention Residual over LSS stages

核心思想：
    普通顺序 LSS：F0 -> LSS1 -> F1 -> LSS2 -> F2 -> ...
    PDAR-LSS：第 i 个 LSS 前，从全部历史 {F0,...,F(i-1)} 中，
              为每一个空间 patch 学习深度权重，再融合为 Hi。

注意：
    1. 注意力只沿“网络深度 N”做 Softmax，不在空间 H/W 上做注意力。
    2. LSS 中的 HSS/Mamba 负责长程建模，5x5/7x7 CNN 负责局部建模。
    3. PDAR 保存的是 HSS 与 CNN 融合后的完整 LSS 输出，不是纯 HSS 输出。
"""

from functools import partial

import torch
import torch.nn as nn

# 项目真实的 Mamba/LSS stage。它的输入、输出均为 [B, H, W, D]。
from model.mambaad import LSSModule


class DepthRMSNorm(nn.Module):
    """对每个 token 的最后一维 D 做 RMSNorm，形状不变。"""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_dim))  # [D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [..., D]，例如 [N, B, H, W, D]
        inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x * inv_rms.to(dtype=x.dtype)
        # 输出: 与 x 同形状，例如 [N, B, H, W, D]
        return normalized * self.weight.to(dtype=x.dtype)


class DepthAttentionResidual(nn.Module):
    """从多个历史 LSS 特征中做逐 patch 的深度选择。

    输入：sources = [F0, F1, ..., F(N-1)]
          每个 Fn 的形状均为 [B, H, W, D]。
    输出：mixed   = H_i，形状 [B, H, W, D]，作为下一 LSS 的输入。
          weights = alpha，形状 [N, B, H, W]，仅用于解释/可视化。

    q 是本模块独立学习的固定 query，形状 [D]。
    q 不来自文本、mask 或 A_raw；它与每个历史深度的 K 做点积。
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.norm = DepthRMSNorm(hidden_dim, eps=eps)

        # weight 的原始形状是 [1, D]；取 squeeze 后 q 为 [D]。
        self.proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, sources):
        if not isinstance(sources, (list, tuple)) or len(sources) == 0:
            raise ValueError("sources must be a non-empty list of [B,H,W,D] tensors.")

        reference_shape = sources[0].shape
        if len(reference_shape) != 4 or reference_shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected [B,H,W,{self.hidden_dim}], got {reference_shape}.")
        if any(source.shape != reference_shape for source in sources[1:]):
            raise ValueError("All historical features must have identical shape.")

        # 1) 沿新深度维 N 堆叠。
        # V: [N, B, H, W, D]
        V = torch.stack(list(sources), dim=0)

        # 2) 只用归一化后的 K 来公平比较不同深度的特征。
        # K: [N, B, H, W, D]
        K = self.norm(V)

        # 3) 取出当前 depth mixer 自己的可学习 query。
        # q: [D]
        q = self.proj.weight.squeeze(0)

        # 4) 对“同一空间位置”上的每一个历史深度求点积：
        # score[n,b,h,w] = q dot K[n,b,h,w,:]
        # scores: [N, B, H, W]
        scores = torch.einsum("d, n b h w d -> n b h w", q, K)

        # 5) 只在深度 N 上归一化，故每个 (b,h,w) 都有 sum_n alpha=1。
        # alpha: [N, B, H, W]
        alpha = scores.softmax(dim=0)

        # 6) 用 alpha 加权“原始 V”，而不是归一化后的 K。
        # H_i[b,h,w,:] = sum_n alpha[n,b,h,w] * V[n,b,h,w,:]
        # mixed: [B, H, W, D]
        mixed = torch.einsum("n b h w, n b h w d -> b h w d", alpha, V)
        return mixed, alpha


class PDARLSSForReview(nn.Module):
    """四个 LSS stage 的完整 PDAR 信息传递逻辑。

    输入：
        v_raw:              [B, T, D]，本项目中 T=24*24=576，D=768。
        semantic_embedding: [B, D]，传给 LSS 内部；PDAR 本身不用于 q。
        spatial_shape:      (H,W)，本项目为 (24,24)。

    输出：
        context_tokens: [B, T, D]。
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        grid_size: int = 24,
        depths=(1, 1, 1, 1),
        d_state: int = 16,
        drop_path_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        scan_type: str = "scan",
        num_direction: int = 8,
        use_selective_scan: bool = True,
        use_cnn_branch: bool = True,
        use_deformable_pool: bool = False,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)

        stage_drop_paths = torch.linspace(0, drop_path_rate, len(depths)).tolist()
        self.stages = nn.ModuleList([
            LSSModule(
                hidden_dim=self.hidden_dim,
                drop_path=stage_drop_paths[index],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                depth=depth,
                size=self.grid_size,
                scan_type=scan_type,
                num_direction=num_direction,
                use_selective_scan=use_selective_scan,
                # 原版 MambaAD LSS：HSS 与 5x5/7x7 CNN 并行后融合。
                use_cnn_branch=use_cnn_branch,
                use_deformable_pool=use_deformable_pool,
                # 防止历史 F 被 LSS stage 再机械加一次。
                add_outer_residual=False,
                # 当前 PDAR 实验的设定；标准 CSSD 保持 True。
                use_adaln=False,
            )
            for index, depth in enumerate(depths)
        ])

        # Stage 2/3/4 各有一个独立 q；final mixer 也有独立 q。
        self.depth_mixers = nn.ModuleList([
            DepthAttentionResidual(self.hidden_dim)
            for _ in range(len(depths) - 1)
        ])
        self.final_depth_mixer = DepthAttentionResidual(self.hidden_dim)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, v_raw, semantic_embedding, spatial_shape, return_debug=False):
        B, T, D = v_raw.shape
        H, W = spatial_shape
        if H * W != T or D != self.hidden_dim:
            raise ValueError("v_raw and spatial_shape do not match the configured PDAR dimensions.")

        # F0: [B,T,D] -> [B,H,W,D]，即本项目 [B,24,24,768]。
        F0 = v_raw.view(B, H, W, D)
        history = [F0]
        stage_weights = []
        pool_feat = None

        for stage_index, lss_stage in enumerate(self.stages):
            if stage_index == 0:
                # Stage 1 没有更早历史，直接接收 F0。
                stage_input = F0
                alpha = F0.new_ones((1, B, H, W))
            else:
                # Stage 2: history=[F0,F1]，N=2，输出 H2。
                # Stage 3: history=[F0,F1,F2]，N=3，输出 H3。
                # Stage 4: history=[F0,F1,F2,F3]，N=4，输出 H4。
                stage_input, alpha = self.depth_mixers[stage_index - 1](history)

            # LSS 内部先融合 HSS、5x5 CNN 和 7x7 CNN；因此写入 history
            # 的 F1/F2/... 都是完整 LSS 表示，而不是单独的 HSS 表示。
            stage_output = lss_stage(stage_input, semantic_embedding, pool_feat)
            history.append(stage_output)
            stage_weights.append(alpha)

            # 保留项目原有 LSS pool 的用法。
            if pool_feat is None:
                pool_feat = stage_output

        # 最终从 [F0,F1,F2,F3,F4] 做一次 N=5 的深度融合。
        final_context, final_alpha = self.final_depth_mixer(history)
        context = self.out_norm(final_context)           # [B,H,W,D]
        context_tokens = context.view(B, T, D)           # [B,576,768]

        if not return_debug:
            return context_tokens
        return context_tokens, {
            "depth_stage_weights": tuple(
                weight.permute(1, 0, 2, 3) for weight in stage_weights
            ),
            "depth_final_weights": final_alpha.permute(1, 0, 2, 3),
            "depth_final_context": context_tokens,
        }


"""
老师常问的三句话回答：

Q1. q 与谁算相似度？
A1. q 与同一 patch 位置、不同历史 LSS 深度的 RMSNorm(Fn) 做点积；
    它不与文本、mask、A_raw 或其他空间位置做相似度。

Q2. Softmax 在哪里做？
A2. alpha = softmax(scores, dim=0)，dim=0 是历史深度 N；
    对每个 batch/patch，所有历史层的权重相加为 1。

Q3. PDAR 改了什么？
A3. 改的是 LSS stage 之间的信息传递：从“只传上一层”改为
    “每个 patch 自适应融合全部历史层”；每个历史 Fn 已经包含
    HSS 长程信息与 5x5/7x7 CNN 局部信息。
"""
