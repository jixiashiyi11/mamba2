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


DEFAULT_ORGAN_PRIORS = {
    'brain': (0.6087, 0.2226, 0.1687),
    'liver': (0.3202, 0.2998, 0.3800),
    'retinal': (0.3001, 0.3059, 0.3940),
    'breast': (0.35, 0.35, 0.30),
}


def _parse_csv_floats(text):
    return [float(item.strip()) for item in str(text).split(',') if item.strip()]


def _parse_organ_priors(text):
    priors = dict(DEFAULT_ORGAN_PRIORS)
    if not text:
        return priors
    for chunk in str(text).split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, values = chunk.split(':', 1)
        vals = _parse_csv_floats(values)
        if len(vals) != 3:
            raise ValueError(f'Invalid prior for {name}: expected low,mid,high.')
        total = sum(vals)
        if total <= 0:
            raise ValueError(f'Invalid prior for {name}: sum must be positive.')
        priors[name.strip()] = tuple(v / total for v in vals)
    return priors


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
    cfg.mode = 'test'
    cfg.cfg_path = _cfg_path_to_module_path(args.cfg)
    cfg.debug_eval = False
    if args.batch_size is not None:
        cfg.trainer.data.batch_size_test = int(args.batch_size)
    if args.num_workers is not None:
        cfg.trainer.data.num_workers_per_gpu = int(args.num_workers)
    return cfg


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


def _foreground_mask_from_imgs(imgs, threshold):
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    return imgs_01.max(dim=1).values > float(threshold)


def _hann2d(height, width, device, dtype):
    if height <= 1 or width <= 1:
        return torch.ones((height, width), device=device, dtype=dtype)
    wy = torch.hann_window(height, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(width, periodic=False, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


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


def _positive_foreground_signal(score_map, region_mask):
    if score_map.ndim == 4:
        score_map = score_map.squeeze(1)
    region_mask = region_mask.bool()
    flat = score_map.reshape(score_map.shape[0], -1)
    flat_mask = region_mask.reshape(region_mask.shape[0], -1)
    out = torch.zeros_like(score_map)
    for idx in range(score_map.shape[0]):
        values = flat[idx][flat_mask[idx]]
        if values.numel() == 0:
            values = flat[idx]
        base = values.min()
        shifted = (score_map[idx] - base).clamp_min(0.0)
        out[idx] = shifted * region_mask[idx].to(dtype=score_map.dtype)
    return out


def _band_ratios(signal, low_cut, mid_cut):
    if signal.ndim == 2:
        signal = signal.unsqueeze(0)
    batch, height, width = signal.shape
    window = _hann2d(height, width, signal.device, signal.dtype)
    centered = signal - signal.flatten(1).mean(dim=1).view(batch, 1, 1)
    spectrum = torch.fft.fftshift(torch.fft.fft2(centered * window), dim=(-2, -1))
    energy = torch.abs(spectrum) ** 2
    low_mask, mid_mask, high_mask = _radial_masks(height, width, signal.device, low_cut, mid_cut)
    total = energy.flatten(1).sum(dim=1).clamp_min(1e-12)
    low = energy[:, low_mask].sum(dim=1) / total
    mid = energy[:, mid_mask].sum(dim=1) / total
    high = energy[:, high_mask].sum(dim=1) / total
    return torch.stack([low, mid, high], dim=1)


def _frequency_components(signal, low_cut, mid_cut):
    if signal.ndim == 2:
        signal = signal.unsqueeze(0)
    _, height, width = signal.shape
    spectrum = torch.fft.fftshift(torch.fft.fft2(signal), dim=(-2, -1))
    low_mask, mid_mask, high_mask = _radial_masks(height, width, signal.device, low_cut, mid_cut)
    comps = []
    for mask in [low_mask, mid_mask, high_mask]:
        filtered = torch.zeros_like(spectrum)
        filtered[:, mask] = spectrum[:, mask]
        comp = torch.real(torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1))))
        comps.append(comp)
    return comps


def _match_mean_std(source, reference, region_mask):
    out = torch.zeros_like(source)
    flat_region = region_mask.reshape(region_mask.shape[0], -1).bool()
    for idx in range(source.shape[0]):
        src = source[idx]
        ref = reference[idx]
        mask = flat_region[idx].reshape_as(src)
        src_vals = src[mask] if mask.any() else src.reshape(-1)
        ref_vals = ref[mask] if mask.any() else ref.reshape(-1)
        src_norm = (src - src_vals.mean()) / src_vals.std().clamp_min(1e-6)
        out[idx] = src_norm * ref_vals.std().clamp_min(1e-6) + ref_vals.mean()
    return out


def _calibrate_to_frequency_prior(raw_map, foreground, organs, priors, lambdas, low_cut, mid_cut):
    signal = _positive_foreground_signal(raw_map, foreground)
    low, mid, high = _frequency_components(signal, low_cut, mid_cut)
    comps = [low, mid, high]
    variants = {'raw': raw_map.squeeze(1) if raw_map.ndim == 4 else raw_map}
    for lam in lambdas:
        recon = torch.zeros_like(signal)
        for idx, organ in enumerate(organs):
            target = priors.get(str(organ), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
            weighted = sum(float(target[band]) * comps[band][idx] for band in range(3))
            recon[idx] = weighted
        recon = _match_mean_std(recon, variants['raw'], foreground)
        variants[f'freqcal_l{lam:g}'] = (1.0 - float(lam)) * variants['raw'] + float(lam) * recon
    return variants


def _kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-8, 1.0)
    q = np.clip(q, 1e-8, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


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


def _summarize_frequency(records):
    rows = []
    for organ in sorted({r['organ'] for r in records}) + ['Avg']:
        sub = records if organ == 'Avg' else [r for r in records if r['organ'] == organ]
        if not sub:
            continue
        rows.append({
            'organ': organ,
            'n': len(sub),
            'raw_low_mean': float(np.mean([float(r['raw_low']) for r in sub])),
            'raw_mid_mean': float(np.mean([float(r['raw_mid']) for r in sub])),
            'raw_high_mean': float(np.mean([float(r['raw_high']) for r in sub])),
            'target_low_mean': float(np.mean([float(r['target_low']) for r in sub])),
            'target_mid_mean': float(np.mean([float(r['target_mid']) for r in sub])),
            'target_high_mean': float(np.mean([float(r['target_high']) for r in sub])),
            'raw_target_kl_mean': float(np.mean([float(r['raw_target_kl']) for r in sub])),
            'gt_low_mean': float(np.nanmean([float(r['gt_low']) for r in sub])),
            'gt_mid_mean': float(np.nanmean([float(r['gt_mid']) for r in sub])),
            'gt_high_mean': float(np.nanmean([float(r['gt_high']) for r in sub])),
        })
    return rows


def _to_numpy_img(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = (img.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


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


def _save_panel(path, img, mask, raw_map, cal_map):
    img_np = _to_numpy_img(img)
    mask_np = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    panels = [
        img_np,
        np.repeat(mask_np[..., None], 3, axis=2),
        _colorize(raw_map),
        _colorize(cal_map),
    ]
    width, height = img_np.shape[1], img_np.shape[0]
    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.BILINEAR), (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Eval-only frequency consistency diagnostics and map calibration for anomaly maps.'
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
    parser.add_argument('--lambdas', default='0.25,0.5')
    parser.add_argument('--organ-priors', default='')
    parser.add_argument('--map-topk-ratio', type=float, default=0.01)
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--vis-per-organ', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    priors = _parse_organ_priors(args.organ_priors)
    lambdas = _parse_csv_floats(args.lambdas)

    cfg = _build_cfg(args)
    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)
    trainer = get_trainer(cfg)
    trainer.net.eval()

    data_name = getattr(getattr(cfg, 'data_test', cfg.data), 'name', getattr(cfg.data, 'name', 'data'))
    output_dir = args.output_dir or os.path.join(cfg.logdir, f'frequency_map_consistency_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    records = []
    pixel_store = defaultdict(lambda: defaultdict(lambda: {'labels': [], 'scores': []}))
    vis_counts = defaultdict(int)
    best_vis_variant = f'freqcal_l{lambdas[0]:g}' if lambdas else 'raw'

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
            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), imgs.shape[0])

            foreground = _foreground_mask_from_imgs(imgs, args.foreground_threshold)
            if foreground.shape[-2:] != raw_map.shape[-2:]:
                foreground = F.interpolate(foreground.unsqueeze(1).float(), size=raw_map.shape[-2:], mode='nearest').squeeze(1) > 0.5

            variants = _calibrate_to_frequency_prior(
                raw_map,
                foreground,
                organs,
                priors,
                lambdas,
                args.low_cut,
                args.mid_cut,
            )
            raw_signal = _positive_foreground_signal(raw_map, foreground)
            raw_ratios = _band_ratios(raw_signal, args.low_cut, args.mid_cut)

            gt_region = masks_2d > 0.5
            if gt_region.shape[-2:] != raw_map.shape[-2:]:
                gt_region = F.interpolate(gt_region.unsqueeze(1).float(), size=raw_map.shape[-2:], mode='nearest').squeeze(1) > 0.5
            gt_signal = _positive_foreground_signal(raw_map, gt_region)
            gt_ratios = _band_ratios(gt_signal, args.low_cut, args.mid_cut)

            variant_scores = {
                name: _topk_score(score_map, foreground, args.map_topk_ratio)
                for name, score_map in variants.items()
            }

            for idx, organ in enumerate(organs):
                target = np.asarray(priors.get(organ, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)), dtype=np.float64)
                raw_ratio = raw_ratios[idx].detach().cpu().numpy()
                gt_ratio = gt_ratios[idx].detach().cpu().numpy() if int(gt_region[idx].sum()) > 0 else np.array([np.nan, np.nan, np.nan])
                row = {
                    'organ': organ,
                    'img_path': img_paths[idx],
                    'mask_path': mask_paths[idx],
                    'label': labels[idx],
                    'mask_sum': int(gt_region[idx].sum().detach().cpu()),
                    'raw_low': float(raw_ratio[0]),
                    'raw_mid': float(raw_ratio[1]),
                    'raw_high': float(raw_ratio[2]),
                    'target_low': float(target[0]),
                    'target_mid': float(target[1]),
                    'target_high': float(target[2]),
                    'raw_target_kl': _kl_divergence(raw_ratio, target),
                    'gt_low': float(gt_ratio[0]),
                    'gt_mid': float(gt_ratio[1]),
                    'gt_high': float(gt_ratio[2]),
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

                if args.vis_per_organ > 0 and vis_counts[organ] < args.vis_per_organ:
                    name = os.path.splitext(os.path.basename(img_paths[idx]))[0] or f'batch{batch_idx}_item{idx}'
                    _save_panel(
                        os.path.join(output_dir, 'debug_vis', organ, f'{vis_counts[organ]:03d}_{name}_label{labels[idx]}.png'),
                        imgs[idx],
                        masks_2d[idx],
                        raw_map[idx].detach().cpu().numpy(),
                        variants.get(best_vis_variant, raw_map)[idx].detach().cpu().numpy(),
                    )
                    vis_counts[organ] += 1

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} test batches')

    if not records:
        raise RuntimeError('No test records were processed.')

    record_fields = [
        'organ', 'img_path', 'mask_path', 'label', 'mask_sum',
        'raw_low', 'raw_mid', 'raw_high',
        'target_low', 'target_mid', 'target_high',
        'raw_target_kl',
        'gt_low', 'gt_mid', 'gt_high',
    ]
    for variant_name in sorted(pixel_store.keys()):
        record_fields.extend([f'{variant_name}_image_score', f'{variant_name}_map_mean'])
    _write_csv(os.path.join(output_dir, 'frequency_map_records.csv'), records, record_fields)

    metrics = _summarize(records, pixel_store)
    metric_fields = [
        'variant', 'organ', 'n',
        'image_AUROC', 'image_AP', 'image_F1',
        'pixel_AUROC', 'pixel_AP', 'pixel_F1_max',
    ]
    _write_csv(os.path.join(output_dir, 'frequency_map_metrics.csv'), metrics, metric_fields)

    freq_summary = _summarize_frequency(records)
    _write_csv(
        os.path.join(output_dir, 'frequency_map_summary.csv'),
        freq_summary,
        [
            'organ', 'n',
            'raw_low_mean', 'raw_mid_mean', 'raw_high_mean',
            'target_low_mean', 'target_mid_mean', 'target_high_mean',
            'raw_target_kl_mean',
            'gt_low_mean', 'gt_mid_mean', 'gt_high_mean',
        ],
    )

    print(f'Output: {output_dir}')
    print(f'Records: {os.path.join(output_dir, "frequency_map_records.csv")}')
    print(f'Metrics: {os.path.join(output_dir, "frequency_map_metrics.csv")}')
    print(f'Frequency summary: {os.path.join(output_dir, "frequency_map_summary.csv")}')
    for row in metrics:
        if row['organ'] == 'Avg':
            print(
                f"{row['variant']}: image_AUROC={row['image_AUROC']:.4f} "
                f"pixel_AUROC={row['pixel_AUROC']:.4f} "
                f"pixel_AP={row['pixel_AP']:.4f} F1={row['pixel_F1_max']:.4f}"
            )


if __name__ == '__main__':
    main()
