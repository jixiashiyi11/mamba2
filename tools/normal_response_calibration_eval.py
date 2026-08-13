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
    return [float(item.strip()) for item in str(text).split(',') if item.strip()]


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
        cfg.trainer.data.batch_size_per_gpu_test = int(args.batch_size)
    if args.train_batch_size is not None:
        cfg.trainer.data.batch_size = int(args.train_batch_size)
        cfg.trainer.data.batch_size_per_gpu = int(args.train_batch_size)
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


def _infer_organ(path, cls_name=''):
    text = f'{cls_name}/{path}'.lower()
    if any(key in text for key in ['brain', 'brats', 'oasis', 'ixi']):
        return 'brain'
    if any(key in text for key in ['liver', 'msd_liver', 'task03']):
        return 'liver'
    if any(key in text for key in ['retinal', 'retina', 'fundus', 'oct', 'dr_lesion']):
        return 'retinal'
    return 'global'


def _foreground_mask(imgs, threshold):
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    return imgs_01.max(dim=1).values > float(threshold)


def _erode(mask, iters):
    x = mask.float().unsqueeze(1)
    for _ in range(max(int(iters), 0)):
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return x.squeeze(1) > 0.5


def _resize_bool(mask, target_hw):
    if mask.shape[-2:] == target_hw:
        return mask.bool()
    return F.interpolate(mask.float().unsqueeze(1), size=target_hw, mode='nearest').squeeze(1) > 0.5


def _sample_tensor(values, max_count, generator):
    values = values.detach().flatten()
    if values.numel() == 0:
        return values.detach().cpu()
    if values.numel() <= max_count:
        return values.detach().cpu()
    idx = torch.randperm(values.numel(), device=values.device, generator=generator)[:max_count]
    return values.index_select(0, idx).detach().cpu()


def _append_capped(store, key, values, max_total, max_per_region, generator):
    sampled = _sample_tensor(values, max_per_region, generator)
    if sampled.numel() == 0:
        return
    store[key].append(sampled)
    total = sum(item.numel() for item in store[key])
    if total <= max_total:
        return
    merged = torch.cat(store[key], dim=0)
    idx = torch.randperm(merged.numel(), generator=torch.Generator().manual_seed(int(total)))[:max_total]
    store[key] = [merged.index_select(0, idx)]


def _quantile_from_store(store, key, q, fallback=None):
    values = store.get(key, [])
    if not values:
        if fallback is not None:
            return fallback
        return 0.0
    arr = torch.cat(values, dim=0).float().numpy()
    if arr.size == 0:
        return fallback if fallback is not None else 0.0
    return float(np.quantile(arr, q))


def _collect_normal_response_stats(trainer, args):
    store = defaultdict(list)
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(args.seed + 17)

    with torch.no_grad():
        for batch_idx, batch in enumerate(trainer.train_loader, start=1):
            if args.max_train_batches and batch_idx > args.max_train_batches:
                break
            trainer.set_input(batch)
            trainer.forward()
            imgs = trainer.imgs
            raw_map = trainer.anomaly_map.detach()
            if raw_map.ndim == 4:
                raw_map = raw_map.squeeze(1)
            fg = _foreground_mask(imgs, args.foreground_threshold)
            fg = _resize_bool(fg, raw_map.shape[-2:])
            interior = _erode(fg, args.foreground_erode_iters)
            edge = fg & ~interior
            background = ~fg

            paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            cls_names = [str(x) for x in getattr(trainer, 'cls_name', [''] * imgs.shape[0])]
            for idx in range(imgs.shape[0]):
                organ = _infer_organ(paths[idx], cls_names[idx])
                regions = {
                    'all': torch.ones_like(fg[idx], dtype=torch.bool),
                    'foreground': fg[idx],
                    'interior': interior[idx],
                    'edge': edge[idx],
                    'background': background[idx],
                }
                for region_name, region_mask in regions.items():
                    values = raw_map[idx][region_mask]
                    for organ_key in ['global', organ]:
                        _append_capped(
                            store,
                            (organ_key, region_name),
                            values,
                            args.max_samples_per_bucket,
                            args.max_samples_per_region,
                            generator,
                        )
            if batch_idx % 20 == 0:
                print(f'Collected normal response from {batch_idx} train batches')

    thresholds = {}
    for q in args.quantiles:
        global_fg = _quantile_from_store(store, ('global', 'foreground'), q, fallback=0.0)
        for organ in ['global', 'brain', 'liver', 'retinal']:
            for region_name in ['all', 'foreground', 'interior', 'edge', 'background']:
                fallback = global_fg if region_name != 'all' else _quantile_from_store(store, ('global', 'all'), q, fallback=global_fg)
                thresholds[(organ, region_name, q)] = _quantile_from_store(
                    store,
                    (organ, region_name),
                    q,
                    fallback=fallback,
                )
    sample_counts = [
        {'organ': organ, 'region': region, 'n_samples': sum(v.numel() for v in values)}
        for (organ, region), values in sorted(store.items())
    ]
    return thresholds, sample_counts


def _threshold_tensor(raw_map, foreground, interior, edge, organ, thresholds, q, mode):
    if mode == 'global':
        value = thresholds[('global', 'foreground', q)]
        return raw_map.new_full(raw_map.shape, float(value))
    if mode == 'organ':
        value = thresholds.get((organ, 'foreground', q), thresholds[('global', 'foreground', q)])
        return raw_map.new_full(raw_map.shape, float(value))
    if mode == 'region':
        t = raw_map.new_full(raw_map.shape, float(thresholds.get((organ, 'background', q), thresholds[('global', 'background', q)])))
        t = torch.where(edge, raw_map.new_tensor(float(thresholds.get((organ, 'edge', q), thresholds[('global', 'edge', q)]))), t)
        t = torch.where(interior, raw_map.new_tensor(float(thresholds.get((organ, 'interior', q), thresholds[('global', 'interior', q)]))), t)
        t = torch.where(foreground & ~(interior | edge), raw_map.new_tensor(float(thresholds.get((organ, 'foreground', q), thresholds[('global', 'foreground', q)]))), t)
        return t
    raise ValueError(f'Unsupported calibration mode: {mode}')


def _make_variants(raw_map, foreground, interior, edge, organs, thresholds, quantiles, eps=1e-6):
    variants = {'raw': raw_map}
    for q in quantiles:
        for mode in ['global', 'organ', 'region']:
            relu_maps = []
            z_maps = []
            for idx, organ in enumerate(organs):
                t_hi = _threshold_tensor(raw_map[idx], foreground[idx], interior[idx], edge[idx], organ, thresholds, q, mode)
                t_lo = _threshold_tensor(raw_map[idx], foreground[idx], interior[idx], edge[idx], organ, thresholds, 0.5, mode)
                relu_maps.append(torch.relu(raw_map[idx] - t_hi) * foreground[idx].float())
                denom = (t_hi - t_lo).abs().clamp_min(eps)
                z_maps.append(torch.relu((raw_map[idx] - t_lo) / denom) * foreground[idx].float())
            suffix = str(q).replace('.', 'p')
            variants[f'{mode}_relu_q{suffix}'] = torch.stack(relu_maps, dim=0)
            variants[f'{mode}_tail_q{suffix}'] = torch.stack(z_maps, dim=0)
    return variants


def _topk_score(values, mask, ratio):
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.reshape(mask.shape[0], -1).bool()
    out = []
    for idx in range(flat_values.shape[0]):
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
            labels = np.asarray([r['label'] for r in sub], dtype=np.int64)
            scores = np.asarray([r[f'{variant}_image_score'] for r in sub], dtype=np.float64)
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


def _save_panel(path, img, mask, raw_map, calibrated_map):
    img_np = _to_numpy_img(img)
    mask_np = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    panels = [
        img_np,
        np.repeat(mask_np[..., None], 3, axis=2),
        _colorize(raw_map),
        _colorize(calibrated_map),
    ]
    width, height = img_np.shape[1], img_np.shape[0]
    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.BILINEAR), (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description='Eval-only normal response calibration for E2/ARCC maps.')
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--resume-dir', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--train-batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-test-batches', type=int, default=0)
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--foreground-erode-iters', type=int, default=3)
    parser.add_argument('--quantiles', default='0.5,0.9,0.95,0.975,0.99')
    parser.add_argument('--map-topk-ratio', type=float, default=0.01)
    parser.add_argument('--max-samples-per-region', type=int, default=512)
    parser.add_argument('--max-samples-per-bucket', type=int, default=200000)
    parser.add_argument('--vis-per-organ', type=int, default=20)
    parser.add_argument('--vis-variant', default='region_relu_q0p95')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.quantiles = sorted(set(_parse_csv_floats(args.quantiles) + [0.5]))

    cfg = _build_cfg(args)
    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)
    trainer = get_trainer(cfg)
    trainer.net.eval()

    data_name = getattr(getattr(cfg, 'data_test', cfg.data), 'name', getattr(cfg.data, 'name', 'data'))
    output_dir = args.output_dir or os.path.join(cfg.logdir, f'normal_response_calibration_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    print('Collecting normal response thresholds from normal train images...')
    thresholds, sample_counts = _collect_normal_response_stats(trainer, args)

    threshold_rows = []
    for (organ, region, q), value in sorted(thresholds.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        threshold_rows.append({'organ': organ, 'region': region, 'quantile': q, 'threshold': value})
    _write_csv(os.path.join(output_dir, 'normal_response_thresholds.csv'), threshold_rows)
    _write_csv(os.path.join(output_dir, 'normal_response_sample_counts.csv'), sample_counts)

    pixel_store = defaultdict(lambda: defaultdict(lambda: {'labels': [], 'scores': []}))
    records = []
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
            if raw_map.ndim == 4:
                raw_map = raw_map.squeeze(1)

            fg = _foreground_mask(imgs, args.foreground_threshold)
            fg = _resize_bool(fg, raw_map.shape[-2:])
            interior = _erode(fg, args.foreground_erode_iters)
            edge = fg & ~interior

            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), imgs.shape[0])
            variants = _make_variants(raw_map, fg, interior, edge, organs, thresholds, args.quantiles)
            variant_scores = {
                name: _topk_score(score_map, fg, args.map_topk_ratio)
                for name, score_map in variants.items()
            }

            for idx, organ in enumerate(organs):
                row = {
                    'organ': organ,
                    'img_path': img_paths[idx],
                    'mask_path': mask_paths[idx],
                    'label': labels[idx],
                    'mask_sum': int((masks_2d[idx] > 0.5).sum().detach().cpu()),
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
                    vis_map = variants.get(args.vis_variant, raw_map)[idx].detach().cpu().numpy()
                    _save_panel(
                        os.path.join(output_dir, 'debug_vis', organ, f'{vis_counts[organ]:03d}_{name}_label{labels[idx]}.png'),
                        imgs[idx],
                        masks_2d[idx],
                        raw_map[idx].detach().cpu().numpy(),
                        vis_map,
                    )
                    vis_counts[organ] += 1

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} test batches')

    if not records:
        raise RuntimeError('No test records were processed.')

    record_fields = ['organ', 'img_path', 'mask_path', 'label', 'mask_sum']
    for variant_name in sorted(pixel_store.keys()):
        record_fields.extend([f'{variant_name}_image_score', f'{variant_name}_map_mean'])
    _write_csv(os.path.join(output_dir, 'normal_response_calibration_records.csv'), records, record_fields)

    metrics = _summarize(records, pixel_store)
    metric_fields = [
        'variant', 'organ', 'n',
        'image_AUROC', 'image_AP', 'image_F1',
        'pixel_AUROC', 'pixel_AP', 'pixel_F1_max',
    ]
    _write_csv(os.path.join(output_dir, 'normal_response_calibration_metrics.csv'), metrics, metric_fields)

    print(f'Output: {output_dir}')
    print(f'Thresholds: {os.path.join(output_dir, "normal_response_thresholds.csv")}')
    print(f'Records: {os.path.join(output_dir, "normal_response_calibration_records.csv")}')
    print(f'Metrics: {os.path.join(output_dir, "normal_response_calibration_metrics.csv")}')
    for row in metrics:
        if row['organ'] == 'Avg':
            print(
                f"{row['variant']}: image_AUROC={row['image_AUROC']:.4f} "
                f"pixel_AUROC={row['pixel_AUROC']:.4f} "
                f"pixel_AP={row['pixel_AP']:.4f} F1={row['pixel_F1_max']:.4f}"
            )


if __name__ == '__main__':
    main()
