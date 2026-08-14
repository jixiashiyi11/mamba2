import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualPatchAdapter(nn.Module):
    """AA-CLIP-style lightweight residual MLP adapter for patch tokens."""

    def __init__(self, dim, bottleneck=4, adapter_scale=0.1, min_hidden_dim=64):
        super().__init__()
        hidden_dim = max(dim // bottleneck, min_hidden_dim)
        self.adapter_scale = float(adapter_scale)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, semantic_embedding=None, spatial_shape=None):
        return F.normalize(x + self.adapter_scale * self.net(x), dim=-1)


class CSSDPatchAdapter(nn.Module):
    """Wrapper that reuses MambaAD's CSSD token refiner on CLIP patch tokens."""

    def __init__(
        self,
        dim,
        grid_size,
        depths=(1, 1, 1, 1),
        d_state=16,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        scan_type="scan",
        num_direction=8,
        use_selective_scan=True,
        use_cnn_branch=True,
        use_deformable_pool=True,
        adapter_scale=0.1,
    ):
        super().__init__()
        from model.mambaad import CSSD

        self.adapter_scale = float(adapter_scale)
        self.cssd = CSSD(
            hidden_dim=dim,
            grid_size=grid_size,
            depths=depths,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            scan_type=scan_type,
            num_direction=num_direction,
            use_selective_scan=use_selective_scan,
            use_cnn_branch=use_cnn_branch,
            use_deformable_pool=use_deformable_pool,
        )

    def forward(self, x, semantic_embedding, spatial_shape):
        refined = self.cssd(x, semantic_embedding, spatial_shape)
        return F.normalize(x + self.adapter_scale * (refined - x), dim=-1)


class LocalGlobalPatchAdapter(nn.Module):
    """CNN local refinement + CSSD/Mamba global refinement for CLIP patch tokens."""

    def __init__(
        self,
        dim,
        grid_size,
        depths=(1, 1, 1, 1),
        d_state=16,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        scan_type="scan",
        num_direction=8,
        use_selective_scan=True,
        use_cnn_branch=True,
        use_deformable_pool=False,
        adapter_scale=0.1,
        local_kernel_size=3,
        local_scale=1.0,
        global_scale=1.0,
        fusion_type="add",
    ):
        super().__init__()
        if local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd.")
        from model.mambaad import CSSD

        self.adapter_scale = float(adapter_scale)
        self.local_scale = float(local_scale)
        self.global_scale = float(global_scale)
        self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in ("add", "concat"):
            raise ValueError("fusion_type must be add or concat.")
        padding = local_kernel_size // 2

        self.local_norm = nn.LayerNorm(dim)
        self.local_dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=local_kernel_size,
            padding=padding,
            groups=dim,
        )
        self.local_pwconv = nn.Conv2d(dim, dim, kernel_size=1)
        self.local_act = nn.GELU()
        self.local_out_norm = nn.LayerNorm(dim)

        self.cssd = CSSD(
            hidden_dim=dim,
            grid_size=grid_size,
            depths=depths,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            scan_type=scan_type,
            num_direction=num_direction,
            use_selective_scan=use_selective_scan,
            use_cnn_branch=use_cnn_branch,
            use_deformable_pool=use_deformable_pool,
        )
        self.fusion_norm = nn.LayerNorm(dim)
        if self.fusion_type == "concat":
            self.fusion_proj = nn.Linear(dim * 2, dim)
        else:
            self.fusion_proj = None

    def _local_refine(self, x, spatial_shape):
        bsz, num_tokens, dim = x.shape
        height, width = spatial_shape
        if height * width != num_tokens:
            raise ValueError(f"Spatial shape {(height, width)} does not match sequence length {num_tokens}.")

        local = self.local_norm(x).view(bsz, height, width, dim)
        local = local.permute(0, 3, 1, 2).contiguous()
        local = self.local_dwconv(local)
        local = self.local_act(local)
        local = self.local_pwconv(local)
        local = local.permute(0, 2, 3, 1).contiguous().view(bsz, num_tokens, dim)
        return self.local_out_norm(local)

    def forward(self, x, semantic_embedding, spatial_shape):
        local_delta = self._local_refine(x, spatial_shape)
        global_refined = self.cssd(x, semantic_embedding, spatial_shape)
        global_delta = global_refined - x
        if self.fusion_type == "concat":
            fused_delta = self.fusion_proj(
                F.gelu(torch.cat([self.local_scale * local_delta, self.global_scale * global_delta], dim=-1))
            )
        else:
            fused_delta = self.local_scale * local_delta + self.global_scale * global_delta
        fused_delta = self.fusion_norm(fused_delta)
        return F.normalize(x + self.adapter_scale * fused_delta, dim=-1)


class MultiLayerLocalPatchAdapter(nn.Module):
    """Use low/mid CLIP patch tokens to locally refine the final CLIP patch tokens."""

    def __init__(
        self,
        dim,
        grid_size,
        adapter_scale=0.1,
        local_kernel_size=3,
        local_layers=2,
        hidden_dim=None,
    ):
        super().__init__()
        if local_kernel_size % 2 == 0:
            raise ValueError("local_kernel_size must be odd.")
        self.adapter_scale = float(adapter_scale)
        self.local_layers = int(local_layers)
        if self.local_layers < 1:
            raise ValueError("local_layers must be >= 1.")
        hidden_dim = int(hidden_dim or dim)
        padding = local_kernel_size // 2

        self.input_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(self.local_layers)])
        self.fuse = nn.Conv2d(dim * self.local_layers, hidden_dim, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=local_kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(dim)

    def _tokens_to_map(self, x, spatial_shape, norm):
        bsz, num_tokens, dim = x.shape
        height, width = spatial_shape
        if height * width != num_tokens:
            raise ValueError(f"Spatial shape {(height, width)} does not match sequence length {num_tokens}.")
        x = norm(x).view(bsz, height, width, dim)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, last_patch, local_patches, spatial_shape):
        if len(local_patches) < self.local_layers:
            raise ValueError(f"Expected at least {self.local_layers} local patch feature maps.")
        local_patches = list(local_patches[: self.local_layers])
        maps = [
            self._tokens_to_map(patch, spatial_shape, norm)
            for patch, norm in zip(local_patches, self.input_norms)
        ]
        local = self.fuse(F.gelu(torch.cat(maps, dim=1)))
        local = self.dwconv(local)
        local = self.act(local)
        local = self.out_proj(local)
        bsz, dim, height, width = local.shape
        local_delta = local.permute(0, 2, 3, 1).contiguous().view(bsz, height * width, dim)
        local_delta = self.out_norm(local_delta)
        refined = F.normalize(last_patch + self.adapter_scale * local_delta, dim=-1)
        return refined, local_delta


class MambaResponseContext(nn.Module):
    """Build PDAR + HSS/CNN context from CLIP's final patch tokens by default."""

    def __init__(
        self,
        dim,
        grid_size,
        depths=(1, 1, 1, 1),
        d_state=16,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        scan_type="scan",
        num_direction=8,
        use_selective_scan=True,
        use_cnn_branch=True,
        use_deformable_pool=False,
        context_scale=1.0,
        cssd_type="pdar",
    ):
        super().__init__()
        from model.mambaad import CSSD, PDARCSSD

        self.context_scale = float(context_scale)
        self.cssd_type = str(cssd_type).lower()
        if self.cssd_type not in ("standard", "cssd", "pdar", "pdar_cssd"):
            raise ValueError(
                f"Unsupported cssd_type={cssd_type}. Expected standard, cssd, pdar, or pdar_cssd."
            )
        self.use_pdar_cssd = self.cssd_type in ("pdar", "pdar_cssd")
        self.input_norm = nn.LayerNorm(dim)
        cssd_cls = PDARCSSD if self.use_pdar_cssd else CSSD
        self.cssd = cssd_cls(
            hidden_dim=dim,
            grid_size=grid_size,
            depths=depths,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            scan_type=scan_type,
            num_direction=num_direction,
            use_selective_scan=use_selective_scan,
            use_cnn_branch=use_cnn_branch,
            use_deformable_pool=use_deformable_pool,
        )
        self.out_norm = nn.LayerNorm(dim)
        self.prior_head = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, tokens, spatial_shape):
        bsz, num_tokens, dim = tokens.shape
        height, width = spatial_shape
        if height * width != num_tokens:
            raise ValueError(f"Spatial shape {(height, width)} does not match sequence length {num_tokens}.")

        x = self.input_norm(tokens)
        semantic_embedding = F.normalize(tokens.mean(dim=1), dim=-1)
        cssd_debug = {}
        if self.use_pdar_cssd:
            context, cssd_debug = self.cssd(
                x,
                semantic_embedding,
                spatial_shape,
                return_debug=True,
            )
        else:
            context = self.cssd(x, semantic_embedding, spatial_shape)
        context = self.out_norm(context)
        context_tokens = F.normalize(tokens + self.context_scale * (context - tokens), dim=-1)

        # The auxiliary head supervises the complete AttnRes-fused context, while
        # the multi-channel context_tokens remain the features entering ARCC.
        aux_tokens = cssd_debug.get("depth_final_context", context_tokens)
        aux_map = aux_tokens.transpose(1, 2).reshape(bsz, dim, height, width)
        context_logits = self.prior_head(aux_map).squeeze(1)
        debug = {
            "mamba_context_logits": context_logits,
            # Backward-compatible name used by existing visualization code.
            "mamba_global_prior": context_logits,
        }
        if self.use_pdar_cssd:
            final_weights = cssd_debug["depth_final_weights"]
            stage_weights = cssd_debug["depth_stage_weights"]
            weights_float = final_weights.float().clamp_min(1e-8)
            entropy = -(weights_float * weights_float.log()).sum(dim=1)
            if final_weights.shape[1] > 1:
                entropy = entropy / math.log(final_weights.shape[1])
            debug.update(
                {
                    "mamba_depth_weights": final_weights,
                    "mamba_depth_weight_mean": final_weights.mean(dim=(0, 2, 3)),
                    # Per-image spatial means. Stage i has i historical sources:
                    # stage 1 -> F0, stage 2 -> F0/F1, ..., final -> F0/.../F4.
                    "mamba_depth_stage_weight_means": tuple(
                        weight.mean(dim=(2, 3)) for weight in stage_weights
                    ),
                    "mamba_depth_final_weight_means": final_weights.mean(dim=(2, 3)),
                    "dbg_mamba_depth_entropy": entropy.mean(),
                    "dbg_mamba_depth_max_weight": final_weights.amax(dim=1).mean(),
                }
            )
            if not self.training:
                # The trainer summarizes these maps immediately and only saves the
                # small regional means. Training does not retain these extra tensors.
                debug["mamba_depth_stage_weights"] = tuple(stage_weights)
        return context_tokens, debug
