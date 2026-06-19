import torch
import torch.nn as nn
import torch.nn.functional as F

from . import LOSS


@LOSS.register_module
class AdaptiveMCLoss(nn.Module):
    """
    Label-free adaptive semantic margin loss for normal-only AD.

    Args:
        base_margin: Base margin for normal-vs-abnormal text separation.
        alpha: Dynamic margin scale for semantic confusion.
        lambda_normal_align: Weight applied to global normal text alignment.
        lambda_margin: Weight applied to adaptive margin loss.
        lambda_cons: Weight applied to token-level consistency.
        lambda_score_separation: Weight applied to normal-only anomaly-score suppression.
        eps: Numerical stability term used by normalization.
    """

    def __init__(
            self,
            base_margin=0.05,
            alpha=0.1,
            lambda_normal_align=1.0,
            lambda_margin=0.05,
            lambda_cons=1.0,
            lambda_score_separation=0.0,
            score_topk_ratio=0.1,
            score_temperature=0.1,
            score_target=0.0,
            anomaly_score_direction='normal_minus_abnormal',
            eps=1e-6,
            m_base=None,
            mil_topk_ratio=None,
            **_unused_kwargs,
    ):
        super(AdaptiveMCLoss, self).__init__()
        self.base_margin = base_margin if m_base is None else m_base
        self.alpha = alpha
        self.lambda_normal_align = lambda_normal_align
        self.lambda_margin = lambda_margin
        self.lambda_cons = lambda_cons
        self.lambda_score_separation = lambda_score_separation
        self.score_topk_ratio = score_topk_ratio if mil_topk_ratio is None else mil_topk_ratio
        self.score_temperature = score_temperature
        self.score_target = score_target
        self.anomaly_score_direction = self._normalize_anomaly_score_direction(anomaly_score_direction)
        self.eps = eps

    def _normalize_anomaly_score_direction(self, direction):
        direction = str(direction).lower()
        aliases = {
            'normal_minus_abnormal': 'normal_minus_abnormal',
            'normal-abnormal': 'normal_minus_abnormal',
            'normal_abnormal': 'normal_minus_abnormal',
            'abnormal_minus_normal': 'abnormal_minus_normal',
            'abnormal-normal': 'abnormal_minus_normal',
            'abnormal_normal': 'abnormal_minus_normal',
        }
        if direction not in aliases:
            raise ValueError(
                f'Invalid anomaly_score_direction={direction}. '
                'Expected normal_minus_abnormal or abnormal_minus_normal.'
            )
        return aliases[direction]

    def forward(self, v_refined, v_raw, t_norm, t_abn, f_global=None):
        """
        Args:
            v_refined: Refined visual tokens of shape ``(B, L, D)``.
            v_raw: Raw frozen visual tokens of shape ``(B, L, D)``.
            t_norm: Normal text prior of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            t_abn: Abnormal text prior of shape ``(1, D)``, ``(B, D)``, or ``(D,)``.
            f_global: Optional refined global visual feature of shape ``(B, D)``.

        Returns:
            ``(total, loss_normal_align, loss_margin,
            loss_token_consistency, stats)``.
        """
        if v_refined.ndim != 3:
            raise ValueError(f'Expected v_refined to have shape (B, L, D), got {tuple(v_refined.shape)}.')
        if v_raw.ndim != 3 or v_raw.shape != v_refined.shape:
            raise ValueError(f'Expected v_raw to have shape {tuple(v_refined.shape)}, got {tuple(v_raw.shape)}.')

        if f_global is None:
            f_global = v_refined.mean(dim=1)
        if f_global.ndim != 2:
            raise ValueError(f'Expected f_global to have shape (B, D), got {tuple(f_global.shape)}.')

        batch_size, feat_dim = f_global.shape
        t_norm = self._prepare_text_prior(t_norm, batch_size, feat_dim, 't_norm', f_global.device, f_global.dtype)
        t_abn = self._prepare_text_prior(t_abn, batch_size, feat_dim, 't_abn', f_global.device, f_global.dtype)

        f_global = F.normalize(f_global, p=2, dim=1, eps=self.eps)
        t_norm = F.normalize(t_norm, p=2, dim=1, eps=self.eps)
        t_abn = F.normalize(t_abn, p=2, dim=1, eps=self.eps)

        s_norm = torch.sum(f_global * t_norm, dim=1)
        s_abn = torch.sum(f_global * t_abn, dim=1)

        loss_normal_align = 1.0 - s_norm.mean()

        gap = torch.abs(s_norm - s_abn)
        confusion = (1.0 - gap).clamp(min=0.0, max=1.0)
        adaptive_margin = self.base_margin + self.alpha * confusion.detach()
        loss_margin = F.relu(adaptive_margin + s_abn - s_norm).mean()

        refined_tokens = F.normalize(v_refined, p=2, dim=-1, eps=self.eps)
        raw_tokens = F.normalize(v_raw.detach(), p=2, dim=-1, eps=self.eps)
        loss_token_consistency = 1.0 - torch.sum(refined_tokens * raw_tokens, dim=-1).mean()

        token_sim_normal = torch.sum(refined_tokens * t_norm.unsqueeze(1), dim=-1)
        token_sim_abnormal = torch.sum(refined_tokens * t_abn.unsqueeze(1), dim=-1)
        if self.anomaly_score_direction == 'abnormal_minus_normal':
            token_scores = token_sim_abnormal - token_sim_normal
        else:
            token_scores = token_sim_normal - token_sim_abnormal
        loss_score_separation = self._normal_score_suppression(token_scores)

        total_loss = (
                self.lambda_normal_align * loss_normal_align
                + self.lambda_margin * loss_margin
                + self.lambda_cons * loss_token_consistency
                + self.lambda_score_separation * loss_score_separation
        )
        stats = {
            'sim_normal_mean': s_norm.detach().mean(),
            'sim_abnormal_mean': s_abn.detach().mean(),
            'adaptive_margin_mean': adaptive_margin.detach().mean(),
            'score_train_topk_mean': self._topk_scores(token_scores).detach().mean(),
            'loss_score_separation': loss_score_separation.detach(),
        }
        return total_loss, loss_normal_align, loss_margin, loss_token_consistency, stats

    def _topk_scores(self, token_scores):
        if self.score_topk_ratio is None:
            return token_scores.max(dim=1).values
        topk = max(1, int(token_scores.shape[1] * self.score_topk_ratio))
        return token_scores.topk(topk, dim=1).values.mean(dim=1)

    def _normal_score_suppression(self, token_scores):
        selected_scores = self._topk_scores(token_scores)
        temperature = max(float(self.score_temperature), self.eps)
        return F.softplus((selected_scores - self.score_target) / temperature).mean() * temperature

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
