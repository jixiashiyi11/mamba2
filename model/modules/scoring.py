import torch
import torch.nn.functional as F


def abnormal_minus_normal(features, text_prototypes):
    """Score features against [normal, abnormal] prototypes."""
    if features.dim() == 2:
        sims = torch.einsum("bd,bkd->bk", features, text_prototypes)
        return sims[:, 1] - sims[:, 0], sims
    if features.dim() == 3:
        sims = torch.einsum("bld,bkd->blk", features, text_prototypes)
        return sims[:, :, 1] - sims[:, :, 0], sims
    raise ValueError(f"Expected 2D or 3D features, got shape {tuple(features.shape)}")


def mean_topk_score(score_map, topk_ratio=0.01):
    flat = score_map.flatten(1)
    if topk_ratio is None:
        return flat.max(dim=1).values
    topk = max(1, int(flat.shape[1] * float(topk_ratio)))
    return flat.topk(topk, dim=1).values.mean(dim=1)


def upsample_patch_map(patch_score, grid_size, image_shape):
    h, w = grid_size
    patch_map = patch_score.reshape(patch_score.shape[0], h, w)
    return F.interpolate(
        patch_map.unsqueeze(1),
        size=image_shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

