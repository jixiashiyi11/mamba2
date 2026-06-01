import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LOSS


@LOSS.register_module
class AdaptiveMCLoss(nn.Module):
    """
    Text-guided adaptive margin contrastive loss with MIL pooling.

    Args:
        m_base: Base hinge margin.
        alpha: Margin scaling factor for ambiguity-aware adaptation.
        mil_topk_ratio: Top-k ratio used for MIL pooling on patch scores.
        mil_weight: Weight applied to the MIL term.
        eps: Numerical stability term used by normalization.
    """

    def __init__(self, m_base=0.2, alpha=0.3, mil_topk_ratio=0.1, mil_weight=1.0, eps=1e-6):
        super(AdaptiveMCLoss, self).__init__()
        self.m_base = m_base
        self.alpha = alpha
        self.mil_topk_ratio = mil_topk_ratio
        self.mil_weight = mil_weight
        self.eps = eps

    def forward(self, v_refined, t_norm, t_abn, f_global=None, labels=None):
        """
        Args:
            v_refined: Refined patch features of shape ``(B, L, D)``. For
                backward compatibility, ``(B, D)`` is also accepted and will
                be treated as the global feature input.
            t_norm: Normal text prior of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            t_abn: Abnormal text prior of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            f_global: Global image features of shape ``(B, D)``.
            labels: Optional binary labels of shape ``(B,)`` where
                ``0 -> normal`` and ``1 -> abnormal``. If ``None``, all
                samples are treated as normal.

        Returns:
            Scalar mean adaptive margin contrastive loss.
        """
        if f_global is None:
            f_global = v_refined
            v_refined = None

        if f_global.ndim != 2:
            raise ValueError(f'Expected f_global to have shape (B, D), got {tuple(f_global.shape)}.')

        batch_size, feat_dim = f_global.shape
        t_norm = self._prepare_text_prior(t_norm, batch_size, feat_dim, 't_norm', f_global.device, f_global.dtype)
        t_abn = self._prepare_text_prior(t_abn, batch_size, feat_dim, 't_abn', f_global.device, f_global.dtype)

        f_global = F.normalize(f_global, p=2, dim=1, eps=self.eps)
        t_norm = F.normalize(t_norm, p=2, dim=1, eps=self.eps)
        t_abn = F.normalize(t_abn, p=2, dim=1, eps=self.eps)

        s_pos = torch.sum(f_global * t_norm, dim=1)
        s_neg = torch.sum(f_global * t_abn, dim=1)

        confusion = torch.abs(s_pos - s_neg)
        adaptive_margin = self.m_base + self.alpha * (1.0 - confusion)

        loss_normal = F.relu(s_neg - s_pos + adaptive_margin)
        loss_abnormal = F.relu(s_pos - s_neg + adaptive_margin)

        if labels is None:
            labels = torch.zeros(batch_size, device=f_global.device, dtype=torch.long)
        else:
            labels = labels.to(device=f_global.device)
            if labels.ndim != 1 or labels.shape[0] != batch_size:
                raise ValueError(f'Expected labels to have shape ({batch_size},), got {tuple(labels.shape)}.')
            labels = labels.long()

        total_loss = torch.where(labels > 0, loss_abnormal, loss_normal)

        if v_refined is not None:
            if v_refined.ndim != 3 or v_refined.shape[0] != batch_size or v_refined.shape[2] != feat_dim:
                raise ValueError(
                    f'Expected v_refined to have shape ({batch_size}, L, {feat_dim}), got {tuple(v_refined.shape)}.'
                )

            patch_feats = F.normalize(v_refined, p=2, dim=-1, eps=self.eps)
            patch_abn = torch.einsum('bld,bd->bl', patch_feats, t_abn)
            patch_norm = torch.einsum('bld,bd->bl', patch_feats, t_norm)
            patch_delta = patch_abn - patch_norm

            topk = max(1, int(math.ceil(v_refined.shape[1] * self.mil_topk_ratio)))
            mil_score = patch_delta.topk(topk, dim=1).values.mean(dim=1)

            mil_loss_normal = F.relu(mil_score + adaptive_margin)
            mil_loss_abnormal = F.relu(adaptive_margin - mil_score)
            mil_loss = torch.where(labels > 0, mil_loss_abnormal, mil_loss_normal)
            total_loss = total_loss + self.mil_weight * mil_loss

        return total_loss.mean()

    def _prepare_text_prior(self, tensor, batch_size, feat_dim, name, device, dtype):
        if tensor is None:
            raise ValueError(f'{name} must not be None.')

        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != feat_dim:
            raise ValueError(
                f'Expected {name} to have shape (1, {feat_dim}), ({batch_size}, {feat_dim}), '
                f'or ({feat_dim},), got {tuple(tensor.shape)}.'
            )
        if tensor.shape[0] == 1:
            tensor = tensor.expand(batch_size, -1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(
                f'Expected {name} batch dimension to be 1 or {batch_size}, got {tuple(tensor.shape)}.'
            )
        return tensor
