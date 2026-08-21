import torch
import torch.nn.functional as F


def apply_mamba_probability_veto(raw_logits, bidirectional_logits, support, alpha):
    """Apply an independent, no-amplification Mamba veto in probability space."""
    suppressed_logits = torch.minimum(raw_logits, bidirectional_logits)
    suppressed_prob = torch.sigmoid(suppressed_logits.float())
    support = support.float().clamp(0.0, 1.0)
    alpha = alpha.float().clamp(0.0, 1.0)
    veto = alpha * (1.0 - support)
    final_prob = suppressed_prob * (1.0 - veto)
    final_logits = torch.logit(final_prob.clamp(1e-6, 1.0 - 1e-6))
    return final_logits.to(raw_logits.dtype), suppressed_logits, veto.to(raw_logits.dtype)


def resize_mamba_patch_targets(masks, output_size, mode="nearest"):
    """Convert pixel masks to Mamba patch labels without losing tiny defects."""
    targets = masks if masks.ndim == 4 else masks.unsqueeze(1)
    if targets.shape[-2:] == tuple(output_size):
        return targets.clamp(0.0, 1.0)
    if mode == "adaptive_max":
        if any(source < target for source, target in zip(targets.shape[-2:], output_size)):
            raise ValueError(
                "adaptive_max mask conversion only supports downsampling, "
                f"got {tuple(targets.shape[-2:])} -> {tuple(output_size)}."
            )
        targets = F.adaptive_max_pool2d(targets, output_size=output_size)
    elif mode == "nearest":
        targets = F.interpolate(targets, size=output_size, mode="nearest")
    else:
        raise ValueError(
            "mamba_context_mask_pool must be 'nearest' or 'adaptive_max', "
            f"got {mode!r}."
        )
    return targets.clamp(0.0, 1.0)
