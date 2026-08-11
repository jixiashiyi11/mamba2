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
    """Build global context from CLIP's final patch tokens, without text input."""

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
        use_deformable_pool=False,
        context_scale=1.0,
    ):
        super().__init__()
        from model.mambaad import CSSD

        self.context_scale = float(context_scale)
        self.input_norm = nn.LayerNorm(dim)
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
        context = self.cssd(x, semantic_embedding, spatial_shape)
        context = self.out_norm(context)
        context_tokens = F.normalize(tokens + self.context_scale * (context - tokens), dim=-1)
        context_map = context_tokens.transpose(1, 2).reshape(bsz, dim, height, width)
        global_prior = self.prior_head(context_map)
        return context_tokens, {
            "mamba_global_prior": global_prior.squeeze(1),
        }
