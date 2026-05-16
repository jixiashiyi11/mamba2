import torch
import torch.nn.functional as F

from loss.adaptive_mc_loss import AdaptiveMCLoss


def warmup_lambda(epoch, warmup_epochs, start=0.01, end=1.0):
    if warmup_epochs <= 0:
        return end
    progress = min(max(epoch / float(warmup_epochs), 0.0), 1.0)
    return start + (end - start) * progress


def train_one_epoch(model, dataloader, optimizer, device, t_norm, t_abn, epoch, warmup_epochs=10):
    """
    Minimal joint-training example for MambaAD + Adaptive MC-Loss.

    Args:
        model: Modified ``MAMBAAD`` model.
        dataloader: Yields ``(images, labels)`` where labels use ``0`` for
            normal and ``1`` for abnormal. Pass ``None`` for unsupervised
            training.
        optimizer: PyTorch optimizer.
        device: Target device, e.g. ``torch.device("cuda")``.
        t_norm: Frozen BiomedCLIP normal prior with shape ``(512,)`` or ``(1, 512)``.
        t_abn: Frozen BiomedCLIP abnormal prior with shape ``(512,)`` or ``(1, 512)``.
        epoch: Current epoch index.
        warmup_epochs: Number of epochs used to warm up the adaptive-loss weight.
    """
    model.train()
    ada_mc_loss = AdaptiveMCLoss().to(device)

    with torch.no_grad():
        t_norm = F.normalize(t_norm.to(device=device, dtype=torch.float32).view(1, -1), p=2, dim=1)
        t_abn = F.normalize(t_abn.to(device=device, dtype=torch.float32).view(1, -1), p=2, dim=1)

    lambda_weight = warmup_lambda(epoch, warmup_epochs, start=0.01, end=1.0)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = None if labels is None else labels.to(device, non_blocking=True).long().view(-1)

        optimizer.zero_grad(set_to_none=True)

        teacher_feats, recon_feats, f_global = model(
            images,
            return_teacher_features=True,
        )

        l_recon = sum(F.mse_loss(pred, target) for target, pred in zip(teacher_feats, recon_feats))
        l_adamc = ada_mc_loss(f_global, t_norm, t_abn, labels)
        l_total = l_recon + lambda_weight * l_adamc

        l_total.backward()
        optimizer.step()

        # For pixel-space decoders, replace the reconstruction term with:
        # recon_img, f_global = model(images)
        # l_recon = F.mse_loss(recon_img, images)
