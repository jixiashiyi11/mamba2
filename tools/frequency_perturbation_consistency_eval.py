import argparse
import csv
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import get_cfg
from trainer import get_trainer
from util.net import init_training
from util.util import init_checkpoint, run_pre


def _parse_csv_floats(text):
    return [float(x.strip()) for x in str(text).split(',') if x.strip()]


def _float_tag(value):
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')


def _cfg_path_to_module_path(cfg_path):
    return cfg_path.split('.')[0].replace('/', '.')


def _build_cfg(args):
    opt_terminal = Namespace(
        cfg_path=args.cfg,
        mode='test',
        sleep=-1,
        memory=-1,
        dist_url='env://',
        logger_rank=0,
        opts=[],
    )
    cfg = get_cfg(opt_terminal)
    if args.resume_dir:
        cfg.trainer.resume_dir = args.resume_dir
    if args.checkpoint:
        cfg.model.kwargs['checkpoint_path'] = args.checkpoint
    # The E2 checkpoint was saved before optional frequency-branch scalars were
    # added to the model class. They are disabled for this config, so non-strict
    # loading preserves the trained ARCC weights while ignoring those new keys.
    cfg.model.kwargs['strict'] = False
    cfg.mode = 'test'
    cfg.cfg_path = _cfg_path_to_module_path(args.cfg)
    cfg.debug_eval = False
    if args.batch_size is not None:
        cfg.trainer.data.batch_size_test = int(args.batch_size)
    if args.num_workers is not None:
        cfg.trainer.data.num_workers_per_gpu = int(args.num_workers)
    return cfg


def _net_module(trainer):
    return trainer.net.module if hasattr(trainer.net, 'module') else trainer.net


def _normalize_paths(paths, batch_size):
    if paths is None:
        return [''] * batch_size
    if isinstance(paths, (list, tuple)):
        out = [str(item) for item in paths]
    else:
        out = [str(paths)]
    if len(out) < batch_size:
        out.extend([''] * (batch_size - len(out)))
    return out[:batch_size]


def _imagenet_mean_std(imgs):
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return mean, std


def _to_unit(imgs):
    mean, std = _imagenet_mean_std(imgs)
    return (imgs * std + mean).clamp(0.0, 1.0)


def _to_model_space(imgs_01, reference):
    mean, std = _imagenet_mean_std(reference)
    return (imgs_01.clamp(0.0, 1.0) - mean) / std


def _foreground_mask_from_imgs(imgs, threshold):
    imgs_01 = _to_unit(imgs)
    return imgs_01.max(dim=1).values > float(threshold)


def _square_box_from_mask(mask, margin_ratio):
    height, width = mask.shape
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return 0, height, 0, width
    y0 = int(idx[:, 0].min().item())
    y1 = int(idx[:, 0].max().item()) + 1
    x0 = int(idx[:, 1].min().item())
    x1 = int(idx[:, 1].max().item()) + 1
    box_h = max(1, y1 - y0)
    box_w = max(1, x1 - x0)
    side = max(box_h, box_w)
    side = int(math.ceil(side * (1.0 + 2.0 * float(margin_ratio))))
    side = min(max(1, side), max(height, width))
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    y0 = max(0, min(y0, height - side))
    x0 = max(0, min(x0, width - side))
    return y0, y0 + side, x0, x0 + side


def _foreground_crop_zoom(imgs, foreground, margin_ratio):
    batch_size, _, height, width = imgs.shape
    crops = []
    boxes = []
    for idx in range(batch_size):
        y0, y1, x0, x1 = _square_box_from_mask(foreground[idx].bool(), margin_ratio)
        crop = imgs[idx:idx + 1, :, y0:y1, x0:x1]
        crop = F.interpolate(crop, size=(height, width), mode='bilinear', align_corners=False)
        crops.append(crop)
        boxes.append((y0, y1, x0, x1))
    return torch.cat(crops, dim=0), boxes


def _restore_crop_map(crop_map, boxes, output_shape):
    if crop_map.ndim == 4:
        crop_map = crop_map.squeeze(1)
    batch_size = crop_map.shape[0]
    height, width = output_shape
    out = crop_map.new_zeros((batch_size, height, width))
    for idx, (y0, y1, x0, x1) in enumerate(boxes):
        patch = F.interpolate(
            crop_map[idx:idx + 1].unsqueeze(1),
            size=(y1 - y0, x1 - x0),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)
        out[idx, y0:y1, x0:x1] = patch
    return out


def _radial_masks(height, width, device, low_cut, mid_cut):
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device))
    yy, xx = torch.meshgrid(fy, fx, indexing='ij')
    radius = torch.sqrt(xx * xx + yy * yy)
    radius = radius / radius.max().clamp_min(1e-8)
    low = radius <= float(low_cut)
    mid = (radius > float(low_cut)) & (radius <= float(mid_cut))
    high = radius > float(mid_cut)
    return low, mid, high


def _frequency_attenuate(imgs, low_cut, mid_cut, low_scale=1.0, mid_scale=1.0, high_scale=1.0):
    """Apply mild frequency-domain perturbation while preserving image size.

    This is intentionally a perturbation, not synthetic anomaly generation. It
    asks whether the model's existing heatmap remains stable when low/mid/high
    image bands are softly attenuated.
    """
    imgs_01 = _to_unit(imgs)
    _, _, height, width = imgs_01.shape
    spectrum = torch.fft.fftshift(torch.fft.fft2(imgs_01, dim=(-2, -1)), dim=(-2, -1))
    low, mid, high = _radial_masks(height, width, imgs.device, low_cut, mid_cut)
    scale = torch.ones((height, width), device=imgs.device, dtype=imgs_01.dtype)
    scale[low] = float(low_scale)
    scale[mid] = float(mid_scale)
    scale[high] = float(high_scale)
    filtered = spectrum * scale.view(1, 1, height, width)
    out = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)))
    return _to_model_space(out, imgs)


def _local_frequency_candidates(imgs):
    """Local frequency-like perturbations using spatial filters.

    The candidates keep the same spatial layout and only softly change local
    detail bands. They are used either as a fixed perturbation set or selected
    per organ in organ-aware mode.
    """
    imgs_01 = _to_unit(imgs)
    low = F.avg_pool2d(imgs_01, kernel_size=21, stride=1, padding=10)
    mid_context = F.avg_pool2d(imgs_01, kernel_size=7, stride=1, padding=3)
    high_residual = imgs_01 - mid_context
    mid_residual = mid_context - low
    low_center = low - low.flatten(2).mean(dim=-1).view(low.shape[0], low.shape[1], 1, 1)

    lowpass = low + 0.35 * mid_residual
    highdrop = imgs_01 - 0.80 * high_residual
    freqsmooth = low + 0.70 * mid_residual + 0.35 * high_residual
    liver_mildsmooth = low + 0.85 * mid_residual + 0.55 * high_residual
    liver_midkeep = low + 1.00 * mid_residual + 0.45 * high_residual
    liver_highdrop_mild = imgs_01 - 0.45 * high_residual
    retinal_lowdrift = imgs_01 + 0.12 * low_center
    retinal_midpreserve = low + 0.95 * mid_residual + 0.85 * high_residual
    retinal_illumination = (imgs_01 * (1.0 + 0.12 * low_center)).clamp(0.0, 1.0)

    return {
        'local_lowpass': _to_model_space(lowpass, imgs),
        'local_highdrop': _to_model_space(highdrop, imgs),
        'local_freqsmooth': _to_model_space(freqsmooth, imgs),
        'local_liver_mildsmooth': _to_model_space(liver_mildsmooth, imgs),
        'local_liver_midkeep': _to_model_space(liver_midkeep, imgs),
        'local_liver_highdrop_mild': _to_model_space(liver_highdrop_mild, imgs),
        'local_retinal_lowdrift': _to_model_space(retinal_lowdrift, imgs),
        'local_retinal_midpreserve': _to_model_space(retinal_midpreserve, imgs),
        'local_retinal_illumination': _to_model_space(retinal_illumination, imgs),
    }


def _local_frequency_perturbations(imgs):
    candidates = _local_frequency_candidates(imgs)
    return {
        'local_lowpass': candidates['local_lowpass'],
        'local_highdrop': candidates['local_highdrop'],
        'local_freqsmooth': candidates['local_freqsmooth'],
    }


def _organ_aware_frequency_perturbations(imgs, organs):
    candidates = _local_frequency_candidates(imgs)
    mapping = {
        'brain': ('local_lowpass', 'local_freqsmooth', 'local_highdrop'),
        'liver': ('local_liver_mildsmooth', 'local_liver_midkeep', 'local_liver_highdrop_mild'),
        'retinal': ('local_retinal_lowdrift', 'local_retinal_midpreserve', 'local_retinal_illumination'),
        'breast': ('local_liver_mildsmooth', 'local_liver_midkeep', 'local_liver_highdrop_mild'),
        'default': ('local_freqsmooth', 'local_liver_mildsmooth', 'local_retinal_lowdrift'),
    }
    out = {}
    for view_idx in range(3):
        batch_view = torch.empty_like(imgs)
        for idx, organ in enumerate(organs):
            keys = mapping.get(str(organ).lower(), mapping['default'])
            batch_view[idx:idx + 1] = candidates[keys[view_idx]][idx:idx + 1]
        out[f'organ_aware_view{view_idx + 1}'] = batch_view
    return out


def _fft_frequency_perturbations(imgs, low_cut, mid_cut):
    return {
        'fft_lowpass': _frequency_attenuate(
            imgs,
            low_cut,
            mid_cut,
            low_scale=1.0,
            mid_scale=0.55,
            high_scale=0.15,
        ),
        'fft_highdrop': _frequency_attenuate(
            imgs,
            low_cut,
            mid_cut,
            low_scale=1.0,
            mid_scale=1.0,
            high_scale=0.20,
        ),
        'fft_freqsmooth': _frequency_attenuate(
            imgs,
            low_cut,
            mid_cut,
            low_scale=1.0,
            mid_scale=0.75,
            high_scale=0.35,
        ),
    }


def _frequency_perturbations(imgs, low_cut, mid_cut, mode, organs=None):
    mode = str(mode).lower()
    if mode == 'local':
        return _local_frequency_perturbations(imgs)
    if mode == 'organ_aware':
        if organs is None:
            raise ValueError('organ_aware perturbation requires organs.')
        return _organ_aware_frequency_perturbations(imgs, organs)
    if mode == 'fft':
        return _fft_frequency_perturbations(imgs, low_cut, mid_cut)
    if mode == 'both':
        out = _local_frequency_perturbations(imgs)
        out.update(_fft_frequency_perturbations(imgs, low_cut, mid_cut))
        return out
    raise ValueError(f'Unsupported perturb_mode={mode}')


def _forward_model(trainer, imgs):
    score_cls_names, adapter_cls_names = trainer._get_model_cls_names()
    out = trainer.net(
        imgs,
        cls_names=score_cls_names,
        adapter_cls_names=adapter_cls_names,
    )
    if isinstance(out, dict):
        return out['anomaly_map'], out['image_score']
    return out


def _robust_norm_per_image(score_map, foreground, low_q=0.01, high_q=0.99):
    if score_map.ndim == 4:
        score_map = score_map.squeeze(1)
    foreground = foreground.bool()
    out = torch.zeros_like(score_map)
    for idx in range(score_map.shape[0]):
        values = score_map[idx][foreground[idx]]
        if values.numel() < 8:
            values = score_map[idx].reshape(-1)
        lo = torch.quantile(values.float(), float(low_q))
        hi = torch.quantile(values.float(), float(high_q))
        out[idx] = ((score_map[idx] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return out


def _frequency_consistency(raw_map, perturbed_maps, foreground, temperature):
    raw_norm = _robust_norm_per_image(raw_map, foreground)
    if not perturbed_maps:
        return torch.ones_like(raw_norm), raw_norm
    consistency_terms = []
    support_terms = []
    for perturbed in perturbed_maps:
        pert_norm = _robust_norm_per_image(perturbed, foreground)
        consistency_terms.append(torch.exp(-torch.abs(raw_norm - pert_norm) / max(float(temperature), 1e-6)))
        support_terms.append((pert_norm + 1e-4) / (raw_norm + 1e-4))
    consistency = torch.stack(consistency_terms, dim=0).mean(dim=0)
    support = torch.stack(support_terms, dim=0).mean(dim=0).clamp(0.0, 1.0)
    return (consistency * support).clamp(0.0, 1.0), raw_norm


def _extract_patch_token_grid(trainer, imgs):
    module = _net_module(trainer)
    if not hasattr(module, 'biomedclip'):
        raise RuntimeError('Self-anchor calibration requires a model with `biomedclip.encode_image_and_patches`.')
    _, tokens, spatial_shape = module.biomedclip.encode_image_and_patches(imgs)
    tokens = F.normalize(tokens.detach(), p=2, dim=-1)
    return tokens.reshape(tokens.shape[0], spatial_shape[0], spatial_shape[1], tokens.shape[-1]), spatial_shape


def _resize_map_to_shape(score_map, target_shape):
    if score_map.ndim == 3:
        score_map = score_map.unsqueeze(1)
    return F.interpolate(score_map.float(), size=target_shape, mode='bilinear', align_corners=False).squeeze(1)


def _self_anchor_unexplained(
        token_grid,
        raw_map,
        foreground,
        low_ratio,
        max_anchors,
        topm,
):
    """Per-image self-reference from low-risk foreground patches.

    This does not build a target-domain bank: anchors are selected separately
    inside each test image from low ARCC-response foreground patches.
    """
    batch_size, height, width, feat_dim = token_grid.shape
    raw_grid = _resize_map_to_shape(raw_map, (height, width))
    if foreground.shape[-2:] != (height, width):
        foreground = F.interpolate(
            foreground.unsqueeze(1).float(),
            size=(height, width),
            mode='nearest',
        ).squeeze(1) > 0.5

    unexplained = torch.zeros((batch_size, height, width), device=token_grid.device, dtype=token_grid.dtype)
    normality = torch.zeros_like(unexplained)
    anchor_mask = torch.zeros((batch_size, height, width), device=token_grid.device, dtype=torch.bool)
    anchor_counts = []

    flat_tokens = token_grid.reshape(batch_size, height * width, feat_dim)
    flat_scores = raw_grid.reshape(batch_size, height * width)
    flat_fg = foreground.reshape(batch_size, height * width).bool()

    for idx in range(batch_size):
        fg_idx = torch.nonzero(flat_fg[idx], as_tuple=False).flatten()
        if fg_idx.numel() < 2:
            fg_idx = torch.arange(height * width, device=token_grid.device)

        low_k = max(1, int(math.ceil(float(fg_idx.numel()) * float(low_ratio))))
        low_k = min(low_k, int(fg_idx.numel()))
        low_order = torch.topk(flat_scores[idx, fg_idx], k=low_k, largest=False).indices
        anchor_idx = fg_idx[low_order]

        max_count = max(1, int(max_anchors))
        if anchor_idx.numel() > max_count:
            order = torch.linspace(
                0,
                anchor_idx.numel() - 1,
                steps=max_count,
                device=anchor_idx.device,
            ).round().long()
            anchor_idx = anchor_idx[order]

        anchors = flat_tokens[idx, anchor_idx]
        anchor_mask[idx].view(-1)[anchor_idx] = True
        anchor_counts.append(int(anchor_idx.numel()))
        sim = flat_tokens[idx] @ anchors.T
        m = min(max(1, int(topm)), sim.shape[1])
        sim_topm = sim.topk(m, dim=1).values.mean(dim=1)
        normality[idx] = sim_topm.reshape(height, width)
        unexplained[idx] = (1.0 - sim_topm).reshape(height, width)

    unexplained_norm = _robust_norm_per_image(unexplained, foreground)
    normality_out = normality
    anchor_mask_out = anchor_mask.float()
    if normality_out.shape[-2:] != raw_map.shape[-2:]:
        normality_out = F.interpolate(
            normality_out.unsqueeze(1),
            size=raw_map.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        unexplained_norm = F.interpolate(
            unexplained_norm.unsqueeze(1),
            size=raw_map.shape[-2:],
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
        anchor_mask_out = F.interpolate(
            anchor_mask_out.unsqueeze(1),
            size=raw_map.shape[-2:],
            mode='nearest',
        ).squeeze(1)
    return unexplained_norm.clamp(0.0, 1.0), normality_out, anchor_counts, anchor_mask_out > 0.5


def _topk_score(values, mask, ratio):
    if values.ndim == 4:
        values = values.squeeze(1)
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).bool()
    out = []
    for idx in range(values.shape[0]):
        selected = flat_values[idx][flat_mask[idx]]
        if selected.numel() == 0:
            selected = flat_values[idx]
        k = max(1, int(math.ceil(selected.numel() * ratio)))
        out.append(selected.topk(k).values.mean())
    return torch.stack(out, dim=0)


def _safe_metric(fn, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(fn(y_true, y_score))


def _f1_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.nanmax(f1))


def _write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(records, pixel_store):
    rows = []
    variants = sorted(pixel_store.keys())
    organs = sorted({r['organ'] for r in records})
    for variant in variants:
        for organ in organs + ['Avg']:
            sub = records if organ == 'Avg' else [r for r in records if r['organ'] == organ]
            labels = np.array([r['label'] for r in sub], dtype=np.int64)
            scores = np.array([r[f'{variant}_image_score'] for r in sub], dtype=np.float64)
            px_labels = np.concatenate(pixel_store[variant][organ]['labels']) if pixel_store[variant][organ]['labels'] else np.array([])
            px_scores = np.concatenate(pixel_store[variant][organ]['scores']) if pixel_store[variant][organ]['scores'] else np.array([])
            rows.append({
                'variant': variant,
                'organ': organ,
                'n': len(sub),
                'image_AUROC': _safe_metric(roc_auc_score, labels, scores),
                'image_AP': _safe_metric(average_precision_score, labels, scores),
                'image_F1': _f1_max(labels, scores),
                'pixel_AUROC': _safe_metric(roc_auc_score, px_labels, px_scores),
                'pixel_AP': _safe_metric(average_precision_score, px_labels, px_scores),
                'pixel_F1_max': _f1_max(px_labels, px_scores),
            })
    return rows


def _to_numpy_img(img):
    arr = _to_unit(img.unsqueeze(0)).squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _colorize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))
    if hi > lo:
        x = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    else:
        x = np.zeros_like(values)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _binary_panel(mask):
    mask = np.asarray(mask, dtype=np.float32)
    mask = (mask > 0.5).astype(np.uint8) * 255
    return np.repeat(mask[..., None], 3, axis=2)


def _save_panel(
        path,
        img,
        mask,
        raw_map,
        consistency,
        fpc_map,
        foreground=None,
        anchor_mask=None,
        self_anchor=None,
        sar_map=None,
):
    img_np = _to_numpy_img(img)
    mask_np = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    panels = [
        img_np,
        np.repeat(mask_np[..., None], 3, axis=2),
        _colorize(raw_map),
    ]
    if foreground is not None:
        panels.append(_binary_panel(foreground))
    if anchor_mask is not None:
        panels.append(_binary_panel(anchor_mask))
    panels.extend([
        _colorize(consistency),
        _colorize(fpc_map),
    ])
    if self_anchor is not None and sar_map is not None:
        panels.extend([
            _colorize(self_anchor),
            _colorize(sar_map),
        ])
    height, width = img_np.shape[:2]
    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.BILINEAR), (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Frequency Perturbation Consistency eval for anomaly maps.'
    )
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--resume-dir', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-test-batches', type=int, default=0)
    parser.add_argument('--low-cut', type=float, default=0.15)
    parser.add_argument('--mid-cut', type=float, default=0.35)
    parser.add_argument('--perturb-mode', default='organ_aware', choices=['organ_aware', 'local', 'fft', 'both'])
    parser.add_argument('--consistency-temperature', type=float, default=0.25)
    parser.add_argument('--fpc-lambdas', default='0.25,0.5,0.75')
    parser.add_argument('--self-anchor-lambdas', default='')
    parser.add_argument('--self-anchor-low-ratio', type=float, default=0.30)
    parser.add_argument('--self-anchor-max-anchors', type=int, default=64)
    parser.add_argument('--self-anchor-topm', type=int, default=5)
    parser.add_argument('--sar-gated-lambdas', default='')
    parser.add_argument('--sar-gated-taus', default='0.6,0.7,0.8')
    parser.add_argument('--sar-gated-alphas', default='8,12')
    parser.add_argument('--enable-foreground-crop', action='store_true')
    parser.add_argument('--crop-margin-ratio', type=float, default=0.15)
    parser.add_argument('--disable-self-anchor', action='store_true')
    parser.add_argument('--map-topk-ratio', type=float, default=0.01)
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--vis-per-organ', type=int, default=20)
    parser.add_argument('--vis-organs', default='')
    parser.add_argument('--stop-after-vis-filled', action='store_true')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fpc_lambdas = _parse_csv_floats(args.fpc_lambdas)
    self_anchor_lambdas = _parse_csv_floats(args.self_anchor_lambdas) if args.self_anchor_lambdas else list(fpc_lambdas)
    sar_gated_lambdas = _parse_csv_floats(args.sar_gated_lambdas) if args.sar_gated_lambdas else list(self_anchor_lambdas)
    sar_gated_taus = _parse_csv_floats(args.sar_gated_taus)
    sar_gated_alphas = _parse_csv_floats(args.sar_gated_alphas)
    vis_organs = {x.strip().lower() for x in str(args.vis_organs).split(',') if x.strip()}

    cfg = _build_cfg(args)
    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)
    trainer = get_trainer(cfg)
    trainer.net.eval()

    data_name = getattr(getattr(cfg, 'data_test', cfg.data), 'name', getattr(cfg.data, 'name', 'data'))
    output_dir = args.output_dir or os.path.join(cfg.logdir, f'frequency_perturbation_consistency_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    records = []
    pixel_store = defaultdict(lambda: defaultdict(lambda: {'labels': [], 'scores': []}))
    vis_counts = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(trainer.test_loader, start=1):
            if args.max_test_batches and batch_idx > args.max_test_batches:
                break
            trainer.set_input(batch)
            trainer.forward()

            imgs = trainer.imgs
            masks = trainer.imgs_mask
            masks_2d = masks.squeeze(1) if masks.ndim == 4 else masks
            raw_map = trainer.anomaly_map.detach()
            raw_map = raw_map.squeeze(1) if raw_map.ndim == 4 else raw_map
            if masks_2d.shape[-2:] != raw_map.shape[-2:]:
                masks_2d = F.interpolate(
                    masks_2d.unsqueeze(1).float(),
                    size=raw_map.shape[-2:],
                    mode='nearest',
                ).squeeze(1)
            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), imgs.shape[0])

            foreground = _foreground_mask_from_imgs(imgs, args.foreground_threshold)
            if foreground.shape[-2:] != raw_map.shape[-2:]:
                foreground = F.interpolate(foreground.unsqueeze(1).float(), size=raw_map.shape[-2:], mode='nearest').squeeze(1) > 0.5

            perturbed_maps = []
            for perturbed_imgs in _frequency_perturbations(
                    imgs,
                    args.low_cut,
                    args.mid_cut,
                    args.perturb_mode,
                    organs=organs,
            ).values():
                perturbed_map, _ = _forward_model(trainer, perturbed_imgs)
                perturbed_map = perturbed_map.detach()
                perturbed_maps.append(perturbed_map.squeeze(1) if perturbed_map.ndim == 4 else perturbed_map)

            consistency, raw_positive = _frequency_consistency(
                raw_map,
                perturbed_maps,
                foreground,
                args.consistency_temperature,
            )
            variants = {
                'raw': raw_map,
                'raw_positive': raw_positive,
                'fpc_direct': raw_positive * consistency,
            }
            for lam in fpc_lambdas:
                variants[f'fpc_soft_l{lam:g}'] = raw_map - float(lam) * (1.0 - consistency) * raw_positive

            self_unexplained = None
            self_normality = None
            self_anchor_mask = None
            self_anchor_counts = [0] * imgs.shape[0]
            if not args.disable_self_anchor:
                token_grid, _ = _extract_patch_token_grid(trainer, imgs)
                self_unexplained, self_normality, self_anchor_counts, self_anchor_mask = _self_anchor_unexplained(
                    token_grid,
                    raw_map,
                    foreground,
                    args.self_anchor_low_ratio,
                    args.self_anchor_max_anchors,
                    args.self_anchor_topm,
                )
                variants['sar_direct'] = raw_positive * self_unexplained
                for lam in self_anchor_lambdas:
                    variants[f'sar_soft_l{lam:g}'] = raw_positive * (1.0 + float(lam) * self_unexplained)
                for tau in sar_gated_taus:
                    for alpha in sar_gated_alphas:
                        high_response_gate = torch.sigmoid(float(alpha) * (raw_positive - float(tau)))
                        for lam in sar_gated_lambdas:
                            name = (
                                f"sar_gated_t{_float_tag(tau)}"
                                f"_a{_float_tag(alpha)}"
                                f"_l{_float_tag(lam)}"
                            )
                            variants[name] = raw_positive * (1.0 + float(lam) * self_unexplained * high_response_gate)

            if args.enable_foreground_crop:
                crop_imgs, crop_boxes = _foreground_crop_zoom(imgs, foreground, args.crop_margin_ratio)
                crop_raw_map, _ = _forward_model(trainer, crop_imgs)
                crop_raw_map = crop_raw_map.detach()
                crop_raw_map = crop_raw_map.squeeze(1) if crop_raw_map.ndim == 4 else crop_raw_map
                crop_foreground = _foreground_mask_from_imgs(crop_imgs, args.foreground_threshold)
                if crop_foreground.shape[-2:] != crop_raw_map.shape[-2:]:
                    crop_foreground = F.interpolate(
                        crop_foreground.unsqueeze(1).float(),
                        size=crop_raw_map.shape[-2:],
                        mode='nearest',
                    ).squeeze(1) > 0.5
                crop_raw_positive = _robust_norm_per_image(crop_raw_map, crop_foreground)
                variants['crop_raw_positive'] = _restore_crop_map(
                    crop_raw_positive,
                    crop_boxes,
                    raw_map.shape[-2:],
                )
                if not args.disable_self_anchor:
                    crop_token_grid, _ = _extract_patch_token_grid(trainer, crop_imgs)
                    crop_unexplained, _, _, _ = _self_anchor_unexplained(
                        crop_token_grid,
                        crop_raw_map,
                        crop_foreground,
                        args.self_anchor_low_ratio,
                        args.self_anchor_max_anchors,
                        args.self_anchor_topm,
                    )
                    variants['crop_sar_direct'] = _restore_crop_map(
                        crop_raw_positive * crop_unexplained,
                        crop_boxes,
                        raw_map.shape[-2:],
                    )

            variant_scores = {
                name: _topk_score(score_map, foreground, args.map_topk_ratio)
                for name, score_map in variants.items()
            }

            for idx, organ in enumerate(organs):
                row = {
                    'organ': organ,
                    'img_path': img_paths[idx],
                    'mask_path': mask_paths[idx],
                    'label': labels[idx],
                    'mask_sum': int((masks_2d[idx] > 0.5).sum().detach().cpu()),
                    'perturb_mode': args.perturb_mode,
                    'consistency_mean': float(consistency[idx].mean().detach().cpu()),
                    'consistency_top1_mean': float(
                        _topk_score(consistency[idx:idx + 1], foreground[idx:idx + 1], args.map_topk_ratio)[0].detach().cpu()
                    ),
                    'self_anchor_count': int(self_anchor_counts[idx]) if self_anchor_counts else 0,
                    'self_unexplained_mean': (
                        float(self_unexplained[idx].mean().detach().cpu()) if self_unexplained is not None else float('nan')
                    ),
                    'self_unexplained_top1_mean': (
                        float(
                            _topk_score(
                                self_unexplained[idx:idx + 1],
                                foreground[idx:idx + 1],
                                args.map_topk_ratio,
                            )[0].detach().cpu()
                        ) if self_unexplained is not None else float('nan')
                    ),
                    'self_normality_mean': (
                        float(self_normality[idx].mean().detach().cpu()) if self_normality is not None else float('nan')
                    ),
                }
                for variant_name, score_map in variants.items():
                    row[f'{variant_name}_image_score'] = float(variant_scores[variant_name][idx].detach().cpu())
                    row[f'{variant_name}_map_mean'] = float(score_map[idx].mean().detach().cpu())
                    pixel_store[variant_name][organ]['labels'].append(
                        masks_2d[idx].detach().cpu().numpy().reshape(-1).astype(np.uint8)
                    )
                    pixel_store[variant_name][organ]['scores'].append(
                        score_map[idx].detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                    pixel_store[variant_name]['Avg']['labels'].append(
                        masks_2d[idx].detach().cpu().numpy().reshape(-1).astype(np.uint8)
                    )
                    pixel_store[variant_name]['Avg']['scores'].append(
                        score_map[idx].detach().cpu().numpy().reshape(-1).astype(np.float32)
                    )
                records.append(row)

                organ_key = str(organ).lower()
                should_save_vis = (
                    args.vis_per_organ > 0
                    and (not vis_organs or organ_key in vis_organs)
                    and vis_counts[organ] < args.vis_per_organ
                )
                if should_save_vis:
                    name = os.path.splitext(os.path.basename(img_paths[idx]))[0] or f'batch{batch_idx}_item{idx}'
                    _save_panel(
                        os.path.join(output_dir, 'debug_vis', organ, f'{vis_counts[organ]:03d}_{name}_label{labels[idx]}.png'),
                        imgs[idx],
                        masks_2d[idx],
                        raw_map[idx].detach().cpu().numpy(),
                        consistency[idx].detach().cpu().numpy(),
                        variants['fpc_direct'][idx].detach().cpu().numpy(),
                        foreground[idx].detach().cpu().numpy(),
                        None if self_anchor_mask is None else self_anchor_mask[idx].detach().cpu().numpy(),
                        None if self_unexplained is None else self_unexplained[idx].detach().cpu().numpy(),
                        None if 'sar_direct' not in variants else variants['sar_direct'][idx].detach().cpu().numpy(),
                    )
                    vis_counts[organ] += 1

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} test batches')
            if args.stop_after_vis_filled and args.vis_per_organ > 0 and vis_organs:
                if all(vis_counts[name] >= args.vis_per_organ for name in vis_organs):
                    print(f'Stopped after collecting requested visualizations: {sorted(vis_organs)}')
                    break

    if not records:
        raise RuntimeError('No test records were processed.')

    record_fields = [
        'organ', 'img_path', 'mask_path', 'label', 'mask_sum',
        'perturb_mode', 'consistency_mean', 'consistency_top1_mean',
        'self_anchor_count', 'self_unexplained_mean',
        'self_unexplained_top1_mean', 'self_normality_mean',
    ]
    for variant_name in sorted(pixel_store.keys()):
        record_fields.extend([f'{variant_name}_image_score', f'{variant_name}_map_mean'])
    records_path = os.path.join(output_dir, 'fpc_records.csv')
    _write_csv(records_path, records, record_fields)

    metrics = _summarize(records, pixel_store)
    metrics_path = os.path.join(output_dir, 'fpc_metrics.csv')
    _write_csv(
        metrics_path,
        metrics,
        [
            'variant', 'organ', 'n',
            'image_AUROC', 'image_AP', 'image_F1',
            'pixel_AUROC', 'pixel_AP', 'pixel_F1_max',
        ],
    )

    print(f'Output: {output_dir}')
    print(f'Records: {records_path}')
    print(f'Metrics: {metrics_path}')
    for row in metrics:
        if row['organ'] == 'Avg':
            print(
                f"{row['variant']}: image_AUROC={row['image_AUROC']:.4f} "
                f"pixel_AUROC={row['pixel_AUROC']:.4f} "
                f"pixel_AP={row['pixel_AP']:.4f} F1={row['pixel_F1_max']:.4f}"
            )


if __name__ == '__main__':
    main()
