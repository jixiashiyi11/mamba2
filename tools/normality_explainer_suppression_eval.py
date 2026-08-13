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
    if args.batch_size is not None:
        cfg.trainer.data.batch_size_test = int(args.batch_size)
    if args.num_workers is not None:
        cfg.trainer.data.num_workers_per_gpu = int(args.num_workers)
    cfg.debug_eval = False
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


def _extract_tokens(module, imgs):
    with torch.no_grad():
        _, tokens, spatial_shape = module.biomedclip.encode_image_and_patches(imgs)
    return F.normalize(tokens.detach(), p=2, dim=-1), spatial_shape


def _foreground_regions(module, imgs, spatial_shape):
    if hasattr(module, '_foreground_masks'):
        foreground, interior, edge, background = module._foreground_masks(imgs, spatial_shape)
        return {
            'interior': interior.squeeze(1).bool(),
            'edge': edge.squeeze(1).bool(),
            'background': background.squeeze(1).bool(),
            'foreground': foreground.squeeze(1).bool(),
        }
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    foreground = imgs_01.max(dim=1, keepdim=True).values > (8.0 / 255.0)
    foreground = F.interpolate(foreground.float(), size=spatial_shape, mode='nearest') > 0.5
    interior = 1.0 - F.max_pool2d(1.0 - foreground.float(), kernel_size=3, stride=1, padding=1)
    interior = interior > 0.5
    edge = foreground & ~interior
    background = ~foreground
    return {
        'interior': interior.squeeze(1).bool(),
        'edge': edge.squeeze(1).bool(),
        'background': background.squeeze(1).bool(),
        'foreground': foreground.squeeze(1).bool(),
    }


def _sample_rows(tokens, mask, max_count, generator):
    flat_tokens = tokens.reshape(-1, tokens.shape[-1])
    flat_mask = mask.reshape(-1)
    idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return flat_tokens.new_zeros((0, flat_tokens.shape[-1]))
    if idx.numel() > max_count:
        perm = torch.randperm(idx.numel(), device=idx.device, generator=generator)[:max_count]
        idx = idx[perm]
    return flat_tokens.index_select(0, idx).detach().cpu()


def _simple_kmeans(samples, k, iters=20, seed=0):
    if samples.numel() == 0:
        return samples
    samples = F.normalize(samples.float(), p=2, dim=-1)
    if samples.shape[0] <= k:
        return samples
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    init_idx = torch.randperm(samples.shape[0], generator=generator)[:k]
    centers = samples.index_select(0, init_idx).contiguous()
    for _ in range(max(1, int(iters))):
        sim = samples @ centers.T
        assign = sim.argmax(dim=1)
        new_centers = []
        for cluster_idx in range(k):
            part = samples[assign == cluster_idx]
            if part.numel() == 0:
                new_centers.append(centers[cluster_idx])
            else:
                new_centers.append(part.mean(dim=0))
        centers = F.normalize(torch.stack(new_centers, dim=0), p=2, dim=-1)
    return centers


def _build_normal_prototypes(trainer, args):
    module = _net_module(trainer)
    module.eval()
    buckets = defaultdict(list)
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(args.seed + 17)
    max_per_batch = max(1, int(args.max_samples_per_region_per_batch))

    with torch.no_grad():
        for batch_idx, batch in enumerate(trainer.train_loader, start=1):
            if args.max_train_batches and batch_idx > args.max_train_batches:
                break
            trainer.set_input(batch)
            imgs = trainer.imgs
            tokens, spatial_shape = _extract_tokens(module, imgs)
            regions = _foreground_regions(module, imgs, spatial_shape)
            token_grid = tokens.reshape(tokens.shape[0], spatial_shape[0], spatial_shape[1], tokens.shape[-1])
            for name in ['interior', 'edge', 'background', 'foreground']:
                sampled = _sample_rows(token_grid, regions[name], max_per_batch, generator)
                if sampled.numel() > 0:
                    buckets[name].append(sampled)

    prototypes = {}
    stats_samples = {}
    for name in ['interior', 'edge', 'background', 'foreground']:
        if not buckets[name]:
            continue
        samples = torch.cat(buckets[name], dim=0)
        if samples.shape[0] > args.max_region_samples:
            generator_cpu = torch.Generator()
            generator_cpu.manual_seed(args.seed + len(name))
            idx = torch.randperm(samples.shape[0], generator=generator_cpu)[:args.max_region_samples]
            samples = samples.index_select(0, idx)
        centers = _simple_kmeans(
            samples,
            k=int(args.prototypes_per_region),
            iters=int(args.kmeans_iters),
            seed=args.seed + len(name) * 13,
        )
        prototypes[name] = F.normalize(centers, p=2, dim=-1).to(trainer.device)
        stats_samples[name] = F.normalize(samples, p=2, dim=-1).to(trainer.device)

    if not prototypes:
        raise RuntimeError('Failed to build normal prototypes from train_loader.')

    thresholds = {}
    for name, samples in stats_samples.items():
        sims = []
        proto = prototypes[name]
        for start in range(0, samples.shape[0], args.sim_chunk_size):
            sim = samples[start:start + args.sim_chunk_size] @ proto.T
            sims.append(sim.max(dim=1).values.detach().cpu())
        sims = torch.cat(sims, dim=0).numpy()
        thresholds[name] = {
            'low': float(np.quantile(sims, args.explain_low_quantile)),
            'high': float(np.quantile(sims, args.explain_high_quantile)),
            'mean': float(np.mean(sims)),
        }
    return prototypes, thresholds


def _nearest_similarity(tokens, prototypes, chunk_size):
    flat = tokens.reshape(-1, tokens.shape[-1])
    values = []
    for start in range(0, flat.shape[0], chunk_size):
        sim = flat[start:start + chunk_size] @ prototypes.T
        values.append(sim.max(dim=1).values)
    return torch.cat(values, dim=0).reshape(tokens.shape[:-1])


def _normal_explainability(module, imgs, prototypes, thresholds, args):
    tokens, spatial_shape = _extract_tokens(module, imgs)
    token_grid = tokens.reshape(tokens.shape[0], spatial_shape[0], spatial_shape[1], tokens.shape[-1])
    regions = _foreground_regions(module, imgs, spatial_shape)
    explain = torch.zeros(token_grid.shape[:3], device=imgs.device, dtype=token_grid.dtype)
    sim_map = torch.zeros_like(explain)

    for name in ['interior', 'edge', 'background']:
        proto_name = name if name in prototypes else 'foreground'
        if proto_name not in prototypes:
            continue
        sim = _nearest_similarity(token_grid, prototypes[proto_name], args.sim_chunk_size)
        stat = thresholds[proto_name]
        denom = max(float(stat['high'] - stat['low']), 1e-6)
        prob = ((sim - float(stat['low'])) / denom).clamp(0.0, 1.0)
        mask = regions[name].to(dtype=torch.bool)
        explain = torch.where(mask, prob, explain)
        sim_map = torch.where(mask, sim, sim_map)

    explain = F.interpolate(
        explain.unsqueeze(1),
        size=imgs.shape[-2:],
        mode='bilinear',
        align_corners=False,
    ).squeeze(1).clamp(0.0, 1.0)
    sim_map = F.interpolate(
        sim_map.unsqueeze(1),
        size=imgs.shape[-2:],
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)
    return explain, sim_map


def _foreground_mask_from_imgs(imgs, threshold):
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    return imgs_01.max(dim=1).values > float(threshold)


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
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(fn(y_true, y_score))


def _f1_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.nanmax(f1))


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


def _write_csv(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_numpy_img(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = (img.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _colorize(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi > lo:
        x = (values - lo) / (hi - lo)
    else:
        x = np.zeros_like(values)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _save_panel(path, img, mask, raw_map, explain_map, final_map):
    img_np = _to_numpy_img(img)
    mask_np = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    maps = [_colorize(raw_map), _colorize(explain_map), _colorize(final_map)]
    panels = [img_np, np.repeat(mask_np[..., None], 3, axis=2)] + maps
    width = img_np.shape[1]
    height = img_np.shape[0]
    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.BILINEAR), (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Eval-only normality explainer suppression for ARCC/CNN anomaly maps.'
    )
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--resume-dir', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-test-batches', type=int, default=0)
    parser.add_argument('--prototypes-per-region', type=int, default=32)
    parser.add_argument('--max-region-samples', type=int, default=50000)
    parser.add_argument('--max-samples-per-region-per-batch', type=int, default=256)
    parser.add_argument('--kmeans-iters', type=int, default=20)
    parser.add_argument('--explain-low-quantile', type=float, default=0.05)
    parser.add_argument('--explain-high-quantile', type=float, default=0.95)
    parser.add_argument('--gammas', default='0.5,1.0,2.0')
    parser.add_argument('--map-topk-ratio', type=float, default=0.01)
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--sim-chunk-size', type=int, default=4096)
    parser.add_argument('--vis-per-organ', type=int, default=20)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _build_cfg(args)
    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)
    trainer = get_trainer(cfg)
    trainer.net.eval()
    module = _net_module(trainer)
    module.eval()

    data_name = getattr(getattr(cfg, 'data_test', cfg.data), 'name', getattr(cfg.data, 'name', 'data'))
    output_dir = args.output_dir or os.path.join(cfg.logdir, f'normality_explainer_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    print('Building normal prototypes...')
    prototypes, thresholds = _build_normal_prototypes(trainer, args)
    proto_rows = []
    for name, proto in prototypes.items():
        proto_rows.append({
            'region': name,
            'num_prototypes': int(proto.shape[0]),
            'dim': int(proto.shape[1]),
            'sim_low': thresholds[name]['low'],
            'sim_high': thresholds[name]['high'],
            'sim_mean': thresholds[name]['mean'],
        })
    _write_csv(os.path.join(output_dir, 'normal_prototypes_summary.csv'), proto_rows)

    gammas = _parse_csv_floats(args.gammas)
    variants = ['raw'] + [f'suppressed_gamma{gamma:g}' for gamma in gammas]
    pixel_store = {
        variant: defaultdict(lambda: {'labels': [], 'scores': []})
        for variant in variants
    }
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
            if masks.ndim == 4:
                masks_2d = masks.squeeze(1)
            else:
                masks_2d = masks
            raw_map = trainer.anomaly_map.detach()
            explain, normal_sim = _normal_explainability(module, imgs, prototypes, thresholds, args)
            foreground = _foreground_mask_from_imgs(imgs, args.foreground_threshold)
            if foreground.shape[-2:] != raw_map.shape[-2:]:
                foreground = F.interpolate(foreground.unsqueeze(1).float(), size=raw_map.shape[-2:], mode='nearest').squeeze(1) > 0.5

            map_by_variant = {'raw': raw_map}
            for gamma in gammas:
                suppress_weight = torch.pow((1.0 - explain).clamp(0.0, 1.0), float(gamma))
                map_by_variant[f'suppressed_gamma{gamma:g}'] = raw_map * suppress_weight

            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), imgs.shape[0])

            for variant, score_map in map_by_variant.items():
                image_score = _topk_score(score_map, foreground, args.map_topk_ratio)
                for organ in set(organs):
                    pixel_store[variant][organ]
                for idx, organ in enumerate(organs):
                    pixel_store[variant][organ]['labels'].append(masks_2d[idx].detach().cpu().numpy().reshape(-1).astype(np.uint8))
                    pixel_store[variant][organ]['scores'].append(score_map[idx].detach().cpu().numpy().reshape(-1).astype(np.float32))
                    pixel_store[variant]['Avg']['labels'].append(masks_2d[idx].detach().cpu().numpy().reshape(-1).astype(np.uint8))
                    pixel_store[variant]['Avg']['scores'].append(score_map[idx].detach().cpu().numpy().reshape(-1).astype(np.float32))

                    if variant == 'raw':
                        row = {
                            'organ': organ,
                            'img_path': img_paths[idx],
                            'mask_path': mask_paths[idx],
                            'label': labels[idx],
                            'raw_image_score': float(image_score[idx].detach().cpu()),
                            'raw_map_mean': float(raw_map[idx].mean().detach().cpu()),
                            'normal_explain_mean': float(explain[idx].mean().detach().cpu()),
                            'normal_sim_mean': float(normal_sim[idx].mean().detach().cpu()),
                        }
                        records.append(row)
                    else:
                        records[-imgs.shape[0] + idx][f'{variant}_image_score'] = float(image_score[idx].detach().cpu())
                        records[-imgs.shape[0] + idx][f'{variant}_map_mean'] = float(score_map[idx].mean().detach().cpu())

            for idx, organ in enumerate(organs):
                if args.vis_per_organ > 0 and vis_counts[organ] < args.vis_per_organ:
                    best_variant = f'suppressed_gamma{gammas[0]:g}'
                    name = os.path.splitext(os.path.basename(img_paths[idx]))[0] or f'batch{batch_idx}_item{idx}'
                    _save_panel(
                        os.path.join(
                            output_dir,
                            'debug_vis',
                            organ,
                            f'{vis_counts[organ]:03d}_{name}_label{labels[idx]}.png',
                        ),
                        imgs[idx],
                        masks_2d[idx],
                        raw_map[idx].detach().cpu().numpy(),
                        explain[idx].detach().cpu().numpy(),
                        map_by_variant[best_variant][idx].detach().cpu().numpy(),
                    )
                    vis_counts[organ] += 1

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} test batches')

    if not records:
        raise RuntimeError('No test records were processed.')

    # Ensure all variant image-score fields exist for every row.
    for row in records:
        for variant in variants:
            row.setdefault(f'{variant}_image_score', float('nan'))
            row.setdefault(f'{variant}_map_mean', float('nan'))

    record_fields = [
        'organ', 'img_path', 'mask_path', 'label',
        'normal_explain_mean', 'normal_sim_mean',
    ]
    for variant in variants:
        record_fields.extend([f'{variant}_image_score', f'{variant}_map_mean'])
    _write_csv(os.path.join(output_dir, 'normality_explainer_records.csv'), records, record_fields)

    metrics = _summarize(records, pixel_store)
    metric_fields = [
        'variant', 'organ', 'n',
        'image_AUROC', 'image_AP', 'image_F1',
        'pixel_AUROC', 'pixel_AP', 'pixel_F1_max',
    ]
    _write_csv(os.path.join(output_dir, 'normality_explainer_metrics.csv'), metrics, metric_fields)

    print(f'Output: {output_dir}')
    print(f'Records: {os.path.join(output_dir, "normality_explainer_records.csv")}')
    print(f'Metrics: {os.path.join(output_dir, "normality_explainer_metrics.csv")}')
    for row in metrics:
        if row['organ'] == 'Avg':
            print(
                f"{row['variant']}: image_AUROC={row['image_AUROC']:.4f} "
                f"pixel_AUROC={row['pixel_AUROC']:.4f} "
                f"pixel_AP={row['pixel_AP']:.4f} F1={row['pixel_F1_max']:.4f}"
            )


if __name__ == '__main__':
    main()
