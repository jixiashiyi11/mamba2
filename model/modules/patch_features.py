import torch
import torch.nn.functional as F


def project_clip_patch_tokens(visual, patch_tokens, num_tokens_with_cls):
    """Project CLIP ViT patch tokens into the CLIP text embedding space."""
    if patch_tokens.shape[1] == num_tokens_with_cls:
        patch_tokens = patch_tokens[:, 1:, :]
    patch_tokens = visual.ln_post(patch_tokens)
    patch_tokens = patch_tokens @ visual.proj
    return F.normalize(patch_tokens.float(), dim=-1)


def fuse_layer_scores(layer_scores, mode="mean", weights=None):
    """Fuse per-layer patch anomaly scores with a deterministic rule."""
    stacked = torch.stack(layer_scores, dim=0)
    mode = str(mode).lower()
    if mode == "mean":
        return stacked.mean(dim=0)
    if mode == "max":
        return stacked.max(dim=0).values
    if mode == "weighted_mean":
        if weights is None:
            raise ValueError("weights must be provided when mode='weighted_mean'.")
        weights = torch.as_tensor(weights, device=stacked.device, dtype=stacked.dtype)
        if weights.numel() != stacked.shape[0]:
            raise ValueError(f"Expected {stacked.shape[0]} layer weights, got {weights.numel()}.")
        weights = weights / weights.sum().clamp_min(1e-12)
        return (stacked * weights.view(-1, 1, 1)).sum(dim=0)
    raise ValueError(f"Unsupported multi-level fusion mode: {mode}")
