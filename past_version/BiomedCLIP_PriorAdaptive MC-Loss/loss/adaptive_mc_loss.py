import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LOSS


@LOSS.register_module
class AdaptiveMCLoss(nn.Module):
    """
    Text-guided adaptive margin contrastive loss.

    Args:
        m_base: Base hinge margin.
        alpha: Margin scaling factor for ambiguity-aware adaptation.
        eps: Numerical stability term used by normalization.
    """

    def __init__(self, m_base=0.2, alpha=0.3, eps=1e-6):
        super(AdaptiveMCLoss, self).__init__()
        self.m_base = m_base
        self.alpha = alpha
        self.eps = eps

    def forward(self, f_global, t_norm, t_abn, labels=None):
        """
        Args:
            f_global: Image features of shape ``(B, D)``.
            t_norm: Normal text prior of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            t_abn: Abnormal text prior of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            labels: Optional binary labels of shape ``(B,)`` where
                ``0 -> normal`` and ``1 -> abnormal``. If ``None``, all
                samples are treated as normal.

        Returns:
            Scalar mean adaptive margin contrastive loss.
        """
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

        selected_losses = torch.where(labels > 0, loss_abnormal, loss_normal)
        return selected_losses.mean()

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
