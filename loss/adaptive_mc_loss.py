import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LOSS


@LOSS.register_module
class AdaptiveMCLoss(nn.Module):
    """
    Text-guided adaptive margin contrastive loss with semantic compactness and
    hard negative mining.

    Args:
        m_base: Base hinge margin.
        alpha: Margin scaling factor for ambiguity-aware adaptation.
        beta: Compactness loss weight for normal samples.
        eps: Numerical stability term used by normalization.
    """

    def __init__(self, m_base=0.2, alpha=0.3, beta=0.1, eps=1e-6):
        super(AdaptiveMCLoss, self).__init__()
        self.m_base = m_base
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, f_global, t_norm, t_abn, labels=None, return_details=False):
        """
        Args:
            f_global: Image features of shape ``(B, D)``.
            t_norm: Normal text priors of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            t_abn: Abnormal text priors of shape ``(K, D)``, ``(1, D)``, or ``(D,)``.
            labels: Optional binary labels of shape ``(B,)`` where
                ``0 -> normal`` and ``1 -> abnormal``. If ``None``, all
                samples are treated as normal.
            return_details: When ``True``, returns a dictionary with per-sample
                losses and similarities in addition to the mean loss.

        Returns:
            Mean loss tensor, or a detail dictionary when ``return_details=True``.
        """
        if f_global.ndim != 2:
            raise ValueError(f'Expected f_global to have shape (B, D), got {tuple(f_global.shape)}.')

        batch_size, feat_dim = f_global.shape
        if labels is None:
            labels = torch.zeros(batch_size, device=f_global.device, dtype=torch.long)
        else:
            labels = labels.to(device=f_global.device, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != batch_size:
                raise ValueError(f'Expected labels to have shape ({batch_size},), got {tuple(labels.shape)}.')

        f_global = F.normalize(f_global, p=2, dim=1, eps=self.eps)
        t_norm = self._prepare_positive_priors(t_norm, batch_size, feat_dim, 't_norm', f_global.device, f_global.dtype)
        t_abn = self._prepare_negative_priors(t_abn, feat_dim, 't_abn', f_global.device, f_global.dtype)
        t_norm = F.normalize(t_norm, p=2, dim=1, eps=self.eps)
        t_abn = F.normalize(t_abn, p=2, dim=1, eps=self.eps)

        s_pos = torch.sum(f_global * t_norm, dim=1)
        s_neg_all = torch.matmul(f_global, t_abn.transpose(0, 1))
        s_neg_max, _ = torch.max(s_neg_all, dim=1)

        confusion = torch.abs(s_pos - s_neg_max)
        adaptive_margin = self.m_base + self.alpha * (1.0 - confusion)

        normal_mask = labels == 0
        if normal_mask.any():
            loss_compact = 1.0 - s_pos[normal_mask]
            loss_compact = loss_compact.mean()
        else:
            loss_compact = s_pos.new_tensor(0.0)

        loss_normal = F.relu(s_neg_max - s_pos + adaptive_margin) + self.beta * loss_compact
        loss_abnormal = F.relu(s_pos - s_neg_max + adaptive_margin)

        selected_losses = torch.where(normal_mask, loss_normal, loss_abnormal)
        mean_loss = selected_losses.mean()

        if not return_details:
            return mean_loss

        return {
            'loss': mean_loss,
            'selected_losses': selected_losses,
            'loss_compact': loss_compact,
            's_pos': s_pos,
            's_neg_max': s_neg_max,
            's_neg_all': s_neg_all,
            'adaptive_margin': adaptive_margin,
        }

    def _prepare_positive_priors(self, tensor, batch_size, feat_dim, name, device, dtype):
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

    def _prepare_negative_priors(self, tensor, feat_dim, name, device, dtype):
        if tensor is None:
            raise ValueError(f'{name} must not be None.')

        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != feat_dim:
            raise ValueError(
                f'Expected {name} to have shape (K, {feat_dim}) or ({feat_dim},), got {tuple(tensor.shape)}.'
            )
        return tensor
