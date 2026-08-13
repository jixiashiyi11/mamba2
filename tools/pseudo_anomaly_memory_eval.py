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


def _extract_tokens(module, imgs):
    with torch.no_grad():
        _, tokens, spatial_shape = module.biomedclip.encode_image_and_patches(imgs)
    tokens = F.normalize(tokens.detach(), p=2, dim=-1)
    return tokens.reshape(tokens.shape[0], spatial_shape[0], spatial_shape[1], tokens.shape[-1]), spatial_shape


def _foreground_regions(module, imgs, spatial_shape, threshold):
    if hasattr(module, '_foreground_masks'):
        foreground, interior, edge, background = module._foreground_masks(imgs, spatial_shape)
        return {
            'foreground': foreground.squeeze(1).bool(),
            'interior': interior.squeeze(1).bool(),
            'edge': edge.squeeze(1).bool(),
            'background': background.squeeze(1).bool(),
        }
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    foreground = imgs_01.max(dim=1, keepdim=True).values > threshold
    foreground = F.interpolate(foreground.float(), size=spatial_shape, mode='nearest') > 0.5
    interior = 1.0 - F.max_pool2d(1.0 - foreground.float(), kernel_size=3, stride=1, padding=1)
    interior = interior > 0.5
    edge = foreground & ~interior
    background = ~foreground
    return {
        'foreground': foreground.squeeze(1).bool(),
        'interior': interior.squeeze(1).bool(),
        'edge': edge.squeeze(1).bool(),
        'background': background.squeeze(1).bool(),
    }


def _foreground_mask_from_imgs(imgs, threshold):
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    return imgs_01.max(dim=1).values > threshold


def _resize_scores_to_grid(score_map, spatial_shape):
    if score_map.ndim == 3:
        score_map = score_map.unsqueeze(1)
    return F.interpolate(score_map.float(), size=spatial_shape, mode='bilinear', align_corners=False).squeeze(1)


def _topk_bool(response, candidate_mask, ratio, largest=True):
    batch_size, height, width = response.shape
    out = torch.zeros((batch_size, height, width), device=response.device, dtype=torch.bool)
    flat_response = response.reshape(batch_size, -1)
    flat_candidate = candidate_mask.reshape(batch_size, -1)
    for idx in range(batch_size):
        candidate_idx = torch.nonzero(flat_candidate[idx], as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            continue
        k = max(1, int(math.ceil(float(candidate_idx.numel()) * float(ratio))))
        k = min(k, int(candidate_idx.numel()))
        chosen_local = torch.topk(flat_response[idx, candidate_idx], k=k, largest=largest).indices
        chosen = candidate_idx[chosen_local]
        out[idx].view(-1)[chosen] = True
    return out


def _sample_rows(token_grid, mask, max_count, generator):
    flat_tokens = token_grid.reshape(-1, token_grid.shape[-1])
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
    centers = samples.index_select(0, torch.randperm(samples.shape[0], generator=generator)[:k]).contiguous()
    for _ in range(max(1, int(iters))):
        sim = samples @ centers.T
        assign = sim.argmax(dim=1)
        new_centers = []
        for cluster_idx in range(k):
            part = samples[assign == cluster_idx]
            new_centers.append(centers[cluster_idx] if part.numel() == 0 else part.mean(dim=0))
        centers = F.normalize(torch.stack(new_centers, dim=0), p=2, dim=-1)
    return centers


def _nearest_similarity(token_grid, prototypes, chunk_size):
    flat = token_grid.reshape(-1, token_grid.shape[-1])
    values = []
    for start in range(0, flat.shape[0], chunk_size):
        sim = flat[start:start + chunk_size] @ prototypes.T
        values.append(sim.max(dim=1).values)
    return torch.cat(values, dim=0).reshape(token_grid.shape[:-1])


def _quantile(values, q, default):
    if not values:
        return float(default)
    arr = torch.cat([v.detach().cpu().reshape(-1) for v in values], dim=0).numpy()
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def _build_memories(trainer, args):
    module = _net_module(trainer)
    module.eval()
    generator = torch.Generator(device=trainer.device)
    generator.manual_seed(args.seed + 101)
    normal_samples = []
    pseudo_samples = []
    memory_response_stats = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(trainer.train_loader, start=1):
            if args.max_train_batches and batch_idx > args.max_train_batches:
                break
            trainer.set_input(batch)
            trainer.forward()
            imgs = trainer.imgs
            raw_map = trainer.anomaly_map.detach()
            token_grid, spatial_shape = _extract_tokens(module, imgs)
            regions = _foreground_regions(module, imgs, spatial_shape, args.foreground_threshold)
            response = _resize_scores_to_grid(raw_map, spatial_shape)

            fg = regions['foreground']
            interior = regions['interior'] & fg
            edge = regions['edge'] & fg
            bg = regions['background']

            high_fg = _topk_bool(response, fg, args.pseudo_top_ratio, largest=True)
            low_fg = _topk_bool(response, interior if interior.any() else fg, args.normal_low_ratio, largest=False)
            high_edge = _topk_bool(response, edge, args.edge_pseudo_top_ratio, largest=True) if edge.any() else torch.zeros_like(high_fg)
            high_bg = _topk_bool(response, bg, args.background_pseudo_top_ratio, largest=True) if bg.any() else torch.zeros_like(high_fg)

            pseudo_mask = high_fg | high_edge | high_bg
            normal_mask = low_fg & ~pseudo_mask

            normal_rows = _sample_rows(token_grid, normal_mask, args.max_samples_per_batch, generator)
            pseudo_rows = _sample_rows(token_grid, pseudo_mask, args.max_samples_per_batch, generator)
            if normal_rows.numel() > 0:
                normal_samples.append(normal_rows)
            if pseudo_rows.numel() > 0:
                pseudo_samples.append(pseudo_rows)

            for name, mask in [('normal_low', normal_mask), ('pseudo_high', pseudo_mask), ('foreground', fg)]:
                vals = response[mask]
                if vals.numel() > 0:
                    memory_response_stats[name].append(vals.detach().cpu())

            if batch_idx % 20 == 0:
                print(f'Collected memory from {batch_idx} train batches')

    if not normal_samples:
        raise RuntimeError('Failed to collect normal memory samples.')
    if not pseudo_samples:
        raise RuntimeError('Failed to collect pseudo-anomaly memory samples.')

    normal_samples = torch.cat(normal_samples, dim=0)
    pseudo_samples = torch.cat(pseudo_samples, dim=0)
    if normal_samples.shape[0] > args.max_memory_samples:
        idx = torch.randperm(normal_samples.shape[0], generator=torch.Generator().manual_seed(args.seed + 1))[:args.max_memory_samples]
        normal_samples = normal_samples.index_select(0, idx)
    if pseudo_samples.shape[0] > args.max_memory_samples:
        idx = torch.randperm(pseudo_samples.shape[0], generator=torch.Generator().manual_seed(args.seed + 2))[:args.max_memory_samples]
        pseudo_samples = pseudo_samples.index_select(0, idx)

    normal_proto = _simple_kmeans(normal_samples, args.normal_prototypes, args.kmeans_iters, args.seed + 11)
    pseudo_proto = _simple_kmeans(pseudo_samples, args.pseudo_prototypes, args.kmeans_iters, args.seed + 23)
    normal_proto = F.normalize(normal_proto, p=2, dim=-1).to(trainer.device)
    pseudo_proto = F.normalize(pseudo_proto, p=2, dim=-1).to(trainer.device)

    # Estimate similarity normalization on the sampled memories themselves.
    normal_sim_chunks = []
    pseudo_sim_chunks = []
    normal_samples_dev = F.normalize(normal_samples, p=2, dim=-1).to(trainer.device)
    pseudo_samples_dev = F.normalize(pseudo_samples, p=2, dim=-1).to(trainer.device)
    for start in range(0, normal_samples_dev.shape[0], args.sim_chunk_size):
        normal_sim_chunks.append((normal_samples_dev[start:start + args.sim_chunk_size] @ normal_proto.T).max(dim=1).values.detach().cpu())
    for start in range(0, pseudo_samples_dev.shape[0], args.sim_chunk_size):
        pseudo_sim_chunks.append((pseudo_samples_dev[start:start + args.sim_chunk_size] @ pseudo_proto.T).max(dim=1).values.detach().cpu())

    normal_sims = torch.cat(normal_sim_chunks, dim=0).numpy()
    pseudo_sims = torch.cat(pseudo_sim_chunks, dim=0).numpy()
    thresholds = {
        'normal_sim_low': float(np.quantile(normal_sims, args.sim_low_quantile)),
        'normal_sim_high': float(np.quantile(normal_sims, args.sim_high_quantile)),
        'pseudo_sim_low': float(np.quantile(pseudo_sims, args.sim_low_quantile)),
        'pseudo_sim_high': float(np.quantile(pseudo_sims, args.sim_high_quantile)),
        'normal_low_response_mean': _quantile(memory_response_stats['normal_low'], 0.50, 0.0),
        'pseudo_high_response_mean': _quantile(memory_response_stats['pseudo_high'], 0.50, 0.0),
    }

    return {
        'normal': normal_proto,
        'pseudo': pseudo_proto,
        'thresholds': thresholds,
        'num_normal_samples': int(normal_samples.shape[0]),
        'num_pseudo_samples': int(pseudo_samples.shape[0]),
    }


def _memory_maps(module, imgs, memories, args, target_hw):
    token_grid, spatial_shape = _extract_tokens(module, imgs)
    normal_sim = _nearest_similarity(token_grid, memories['normal'], args.sim_chunk_size)
    pseudo_sim = _nearest_similarity(token_grid, memories['pseudo'], args.sim_chunk_size)
    thresholds = memories['thresholds']

    normal_denom = max(thresholds['normal_sim_high'] - thresholds['normal_sim_low'], 1e-6)
    pseudo_denom = max(thresholds['pseudo_sim_high'] - thresholds['pseudo_sim_low'], 1e-6)
    normal_explain = ((normal_sim - thresholds['normal_sim_low']) / normal_denom).clamp(0.0, 1.0)
    normal_distance = 1.0 - normal_explain
    pseudo_prob = ((pseudo_sim - thresholds['pseudo_sim_low']) / pseudo_denom).clamp(0.0, 1.0)

    def upsample(x):
        return F.interpolate(x.unsqueeze(1), size=target_hw, mode='bilinear', align_corners=False).squeeze(1)

    return {
        'normal_distance': upsample(normal_distance),
        'normal_explain': upsample(normal_explain),
        'pseudo_prob': upsample(pseudo_prob),
        'normal_sim': upsample(normal_sim),
        'pseudo_sim': upsample(pseudo_sim),
    }


def _per_image_scale(raw_map, foreground):
    flat = raw_map.reshape(raw_map.shape[0], -1)
    mask = foreground.reshape(foreground.shape[0], -1).bool()
    scales = []
    for idx in range(raw_map.shape[0]):
        values = flat[idx][mask[idx]]
        if values.numel() < 2:
            values = flat[idx]
        scales.append(values.std().clamp_min(1e-6))
    return torch.stack(scales, dim=0).view(-1, 1, 1)


def _make_variants(raw_map, memory_maps, foreground, lambdas, betas):
    variants = {'raw': raw_map}
    scale = _per_image_scale(raw_map, foreground)
    pseudo = memory_maps['pseudo_prob']
    normal_distance = memory_maps['normal_distance']

    for lam in lambdas:
        variants[f'pseudo_sub_l{lam:g}'] = raw_map - float(lam) * pseudo * scale
        variants[f'pseudo_veto_l{lam:g}'] = raw_map * (1.0 - float(lam) * pseudo).clamp_min(0.0)
    for beta in betas:
        variants[f'normaldist_add_b{beta:g}'] = raw_map + float(beta) * normal_distance * scale
    for beta in betas:
        for lam in lambdas:
            variants[f'normaldist_pseudo_b{beta:g}_l{lam:g}'] = (
                raw_map
                + float(beta) * normal_distance * (1.0 - pseudo) * scale
                - float(lam) * pseudo * scale
            )
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


def _save_panel(path, img, mask, raw_map, pseudo_map, final_map):
    img_np = _to_numpy_img(img)
    mask_np = (mask.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    panels = [
        img_np,
        np.repeat(mask_np[..., None], 3, axis=2),
        _colorize(raw_map),
        _colorize(pseudo_map),
        _colorize(final_map),
    ]
    width, height = img_np.shape[1], img_np.shape[0]
    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.BILINEAR), (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def _region_mean(values, mask):
    selected = values[mask]
    if selected.numel() == 0:
        return float('nan')
    return float(selected.mean().detach().cpu())


def main():
    parser = argparse.ArgumentParser(
        description='Eval-only pseudo-anomaly memory suppression using normal-only train images.'
    )
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--resume-dir', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-test-batches', type=int, default=0)
    parser.add_argument('--normal-prototypes', type=int, default=64)
    parser.add_argument('--pseudo-prototypes', type=int, default=64)
    parser.add_argument('--max-memory-samples', type=int, default=60000)
    parser.add_argument('--max-samples-per-batch', type=int, default=512)
    parser.add_argument('--kmeans-iters', type=int, default=20)
    parser.add_argument('--pseudo-top-ratio', type=float, default=0.03)
    parser.add_argument('--normal-low-ratio', type=float, default=0.20)
    parser.add_argument('--edge-pseudo-top-ratio', type=float, default=0.20)
    parser.add_argument('--background-pseudo-top-ratio', type=float, default=0.05)
    parser.add_argument('--lambdas', default='0.25,0.5,1.0')
    parser.add_argument('--betas', default='0.25,0.5')
    parser.add_argument('--map-topk-ratio', type=float, default=0.01)
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--sim-low-quantile', type=float, default=0.05)
    parser.add_argument('--sim-high-quantile', type=float, default=0.95)
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
    output_dir = args.output_dir or os.path.join(cfg.logdir, f'pseudo_anomaly_memory_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    print('Building pseudo-anomaly memory from normal train images...')
    memories = _build_memories(trainer, args)
    _write_csv(
        os.path.join(output_dir, 'memory_summary.csv'),
        [{
            'normal_prototypes': int(memories['normal'].shape[0]),
            'pseudo_prototypes': int(memories['pseudo'].shape[0]),
            'num_normal_samples': memories['num_normal_samples'],
            'num_pseudo_samples': memories['num_pseudo_samples'],
            **memories['thresholds'],
        }],
    )

    lambdas = _parse_csv_floats(args.lambdas)
    betas = _parse_csv_floats(args.betas)
    pixel_store = defaultdict(lambda: defaultdict(lambda: {'labels': [], 'scores': []}))
    records = []
    vis_counts = defaultdict(int)
    best_vis_variant = f'normaldist_pseudo_b{betas[0]:g}_l{lambdas[0]:g}' if betas and lambdas else 'raw'

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
            mem_maps = _memory_maps(module, imgs, memories, args, raw_map.shape[-2:])
            foreground = _foreground_mask_from_imgs(imgs, args.foreground_threshold)
            if foreground.shape[-2:] != raw_map.shape[-2:]:
                foreground = F.interpolate(foreground.unsqueeze(1).float(), size=raw_map.shape[-2:], mode='nearest').squeeze(1) > 0.5

            variants = _make_variants(raw_map, mem_maps, foreground, lambdas, betas)
            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), imgs.shape[0])

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
                    'pseudo_prob_mean': float(mem_maps['pseudo_prob'][idx].mean().detach().cpu()),
                    'normal_distance_mean': float(mem_maps['normal_distance'][idx].mean().detach().cpu()),
                    'pseudo_prob_gt_mean': _region_mean(mem_maps['pseudo_prob'][idx], masks_2d[idx] > 0.5),
                    'pseudo_prob_fg_mean': _region_mean(mem_maps['pseudo_prob'][idx], foreground[idx]),
                    'normal_distance_gt_mean': _region_mean(mem_maps['normal_distance'][idx], masks_2d[idx] > 0.5),
                    'normal_distance_fg_mean': _region_mean(mem_maps['normal_distance'][idx], foreground[idx]),
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
                    final_for_vis = variants.get(best_vis_variant, raw_map)[idx].detach().cpu().numpy()
                    _save_panel(
                        os.path.join(output_dir, 'debug_vis', organ, f'{vis_counts[organ]:03d}_{name}_label{labels[idx]}.png'),
                        imgs[idx],
                        masks_2d[idx],
                        raw_map[idx].detach().cpu().numpy(),
                        mem_maps['pseudo_prob'][idx].detach().cpu().numpy(),
                        final_for_vis,
                    )
                    vis_counts[organ] += 1

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} test batches')

    if not records:
        raise RuntimeError('No test records were processed.')

    record_fields = [
        'organ', 'img_path', 'mask_path', 'label', 'mask_sum',
        'pseudo_prob_mean', 'normal_distance_mean',
        'pseudo_prob_gt_mean', 'pseudo_prob_fg_mean',
        'normal_distance_gt_mean', 'normal_distance_fg_mean',
    ]
    for variant_name in sorted(pixel_store.keys()):
        record_fields.extend([f'{variant_name}_image_score', f'{variant_name}_map_mean'])
    _write_csv(os.path.join(output_dir, 'pseudo_anomaly_memory_records.csv'), records, record_fields)

    metrics = _summarize(records, pixel_store)
    metric_fields = [
        'variant', 'organ', 'n',
        'image_AUROC', 'image_AP', 'image_F1',
        'pixel_AUROC', 'pixel_AP', 'pixel_F1_max',
    ]
    _write_csv(os.path.join(output_dir, 'pseudo_anomaly_memory_metrics.csv'), metrics, metric_fields)

    print(f'Output: {output_dir}')
    print(f'Records: {os.path.join(output_dir, "pseudo_anomaly_memory_records.csv")}')
    print(f'Metrics: {os.path.join(output_dir, "pseudo_anomaly_memory_metrics.csv")}')
    for row in metrics:
        if row['organ'] == 'Avg':
            print(
                f"{row['variant']}: image_AUROC={row['image_AUROC']:.4f} "
                f"pixel_AUROC={row['pixel_AUROC']:.4f} "
                f"pixel_AP={row['pixel_AP']:.4f} F1={row['pixel_F1_max']:.4f}"
            )


if __name__ == '__main__':
    main()
