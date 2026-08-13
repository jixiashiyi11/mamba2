import argparse
import csv
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict, deque
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


def _foreground_mask(imgs, threshold):
    mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
    return imgs_01.max(dim=1, keepdim=True).values > float(threshold)


def _erode(mask, iters):
    x = mask.float()
    for _ in range(max(int(iters), 0)):
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return x > 0.5


def _resize_bool(mask, target_hw):
    if mask.shape[-2:] == target_hw:
        return mask.bool()
    return F.interpolate(mask.float(), size=target_hw, mode='nearest') > 0.5


def _resize_float(values, target_hw):
    if values.shape[-2:] == target_hw:
        return values.float()
    return F.interpolate(values.float(), size=target_hw, mode='bilinear', align_corners=False)


def _forward_model(trainer, imgs, cls_names=None):
    if cls_names is None:
        score_cls_names, adapter_cls_names = trainer._get_model_cls_names()
    else:
        score_cls_names = cls_names
        adapter_cls_names = cls_names
    out = trainer.net(
        imgs,
        cls_names=score_cls_names,
        adapter_cls_names=adapter_cls_names,
    )
    if isinstance(out, dict):
        return out['anomaly_map'], out['image_score']
    return out


def _odd_kernel(value):
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _apply_deletion(imgs, delete_mask, fill_value='blur', blur_kernel=21):
    mask = _resize_bool(delete_mask, imgs.shape[-2:])
    if fill_value in ('zero', 'mean'):
        fill = torch.zeros_like(imgs)
    elif fill_value == 'black':
        raw_mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        raw_std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        fill = (torch.zeros_like(imgs) - raw_mean) / raw_std
    elif fill_value == 'blur':
        kernel = _odd_kernel(blur_kernel)
        fill = F.avg_pool2d(imgs, kernel_size=kernel, stride=1, padding=kernel // 2)
    elif fill_value == 'local_mean':
        kernel = _odd_kernel(blur_kernel)
        valid = (~mask).to(dtype=imgs.dtype)
        numerator = F.avg_pool2d(imgs * valid, kernel_size=kernel, stride=1, padding=kernel // 2)
        denominator = F.avg_pool2d(valid, kernel_size=kernel, stride=1, padding=kernel // 2).clamp_min(1e-6)
        fill = numerator / denominator
    else:
        raise ValueError(f'Unsupported fill_value={fill_value}')
    return torch.where(mask.expand_as(imgs), fill, imgs)


def _topk_image_score(score_map, foreground, ratio):
    if score_map.ndim == 3:
        score_map = score_map.unsqueeze(1)
    foreground = _resize_bool(foreground, score_map.shape[-2:])
    flat_score = score_map.flatten(1)
    flat_fg = foreground.flatten(1)
    out = []
    for idx in range(flat_score.shape[0]):
        values = flat_score[idx][flat_fg[idx]]
        if values.numel() == 0:
            values = flat_score[idx]
        k = max(1, int(math.ceil(values.numel() * float(ratio))))
        out.append(values.topk(k).values.mean())
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


def _connected_components(binary):
    binary = np.asarray(binary, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            coords = []
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if ny < 0 or nx < 0 or ny >= height or nx >= width:
                            continue
                        if binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            components.append(coords)
    return components


def _component_mask(coords, shape):
    mask = np.zeros(shape, dtype=bool)
    if coords:
        ys, xs = zip(*coords)
        mask[np.asarray(ys), np.asarray(xs)] = True
    return mask


def _component_stats(comp_mask, raw_map, foreground, edge):
    area = int(comp_mask.sum())
    fg_area = max(int(foreground.sum()), 1)
    ys, xs = np.where(comp_mask)
    if ys.size == 0:
        return None
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    bbox_area = max(h * w, 1)
    compactness = float(area / bbox_area)
    edge_overlap = float((comp_mask & edge).sum() / max(area, 1))
    fg_overlap = float((comp_mask & foreground).sum() / max(area, 1))
    return {
        'area': area,
        'area_ratio': float(area / fg_area),
        'bbox_y1': int(ys.min()),
        'bbox_x1': int(xs.min()),
        'bbox_y2': int(ys.max()),
        'bbox_x2': int(xs.max()),
        'compactness': compactness,
        'edge_overlap': edge_overlap,
        'foreground_overlap': fg_overlap,
        'mean_score': float(raw_map[comp_mask].mean()),
        'max_score': float(raw_map[comp_mask].max()),
    }


def _proposal_score(stats, drop, args):
    area = stats['area_ratio']
    too_small = max(0.0, float(args.min_area_ratio) - area) / max(float(args.min_area_ratio), 1e-6)
    too_large = max(0.0, area - float(args.max_area_ratio)) / max(float(args.max_area_ratio), 1e-6)
    area_penalty = min(1.0, too_small + too_large)
    shape_score = (
        float(stats['foreground_overlap'])
        * (1.0 - float(stats['edge_overlap']))
        * (1.0 - area_penalty)
        * max(float(stats['compactness']), 0.05)
    )
    drop_score = math.tanh(float(drop) / max(float(args.drop_scale), 1e-6))
    return (
        float(args.heat_weight) * float(stats['mean_score'])
        + float(args.shape_weight) * shape_score
        + float(args.drop_weight) * drop_score
        - float(args.edge_penalty_weight) * float(stats['edge_overlap'])
        - float(args.area_penalty_weight) * area_penalty
    ), shape_score, area_penalty, drop_score


def _score_direction(score, direction):
    if direction in ('original', 'model'):
        return score
    if direction in ('reverse', 'neg'):
        return -score
    raise ValueError(f'Unsupported score_direction={direction}')


def _make_region_maps(raw_map, proposals, shape, args):
    component_only = np.zeros(shape, dtype=np.float32)
    verified = np.zeros(shape, dtype=np.float32)
    soft_weight = np.ones(shape, dtype=np.float32)
    scores = np.asarray([float(proposal['verify_score']) for proposal in proposals], dtype=np.float32)
    if scores.size > 0:
        score_min = float(scores.min())
        score_max = float(scores.max())
        if score_max > score_min:
            soft_scores = (scores - score_min) / (score_max - score_min)
        else:
            soft_scores = np.full_like(scores, 0.5, dtype=np.float32)
    else:
        soft_scores = np.asarray([], dtype=np.float32)
    for proposal in proposals:
        proposal['soft_norm_score'] = 0.0
        proposal['soft_weight'] = 1.0
    for proposal, soft_score in zip(proposals, soft_scores):
        mask = proposal['mask']
        component_only[mask] = raw_map[mask]
        scale = max(float(proposal['verify_score']), 0.0)
        verified[mask] = raw_map[mask] * scale
        weight = 1.0 - float(args.soft_beta) + (float(args.soft_alpha) + float(args.soft_beta)) * float(soft_score)
        weight = float(np.clip(weight, float(args.soft_min_weight), float(args.soft_max_weight)))
        soft_weight[mask] = weight
        proposal['soft_norm_score'] = float(soft_score)
        proposal['soft_weight'] = weight
    soft_verified = raw_map * soft_weight
    return component_only, verified, soft_verified


def _normalize_for_vis(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _colorize(values):
    x = _normalize_for_vis(values)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _to_numpy_img(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = (img.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _save_panel(path, img, mask, raw_map, component_map, verified_map, soft_map):
    img_np = _to_numpy_img(img)
    mask_np = (mask.detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8) * 255
    panels = [
        img_np,
        np.repeat(mask_np[..., None], 3, axis=2),
        _colorize(raw_map),
        _colorize(component_map),
        _colorize(verified_map),
        _colorize(soft_map),
    ]
    height, width = img_np.shape[:2]
    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel).resize((width, height), Image.BILINEAR), (idx * width, 0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


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
    variants = sorted(pixel_store.keys())
    organs = sorted({row['organ'] for row in records})
    rows = []
    for variant in variants:
        for organ in organs + ['Avg']:
            sub = records if organ == 'Avg' else [row for row in records if row['organ'] == organ]
            labels = np.asarray([row['label'] for row in sub], dtype=np.int64)
            image_scores = np.asarray([row[f'{variant}_image_score'] for row in sub], dtype=np.float64)
            px_labels = np.concatenate(pixel_store[variant][organ]['labels']) if pixel_store[variant][organ]['labels'] else np.array([])
            px_scores = np.concatenate(pixel_store[variant][organ]['scores']) if pixel_store[variant][organ]['scores'] else np.array([])
            rows.append({
                'variant': variant,
                'organ': organ,
                'n': len(sub),
                'image_AUROC': _safe_metric(roc_auc_score, labels, image_scores),
                'image_AP': _safe_metric(average_precision_score, labels, image_scores),
                'image_F1': _f1_max(labels, image_scores),
                'pixel_AUROC': _safe_metric(roc_auc_score, px_labels, px_scores),
                'pixel_AP': _safe_metric(average_precision_score, px_labels, px_scores),
                'pixel_F1_max': _f1_max(px_labels, px_scores),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Eval-only region-verified anomaly localization.')
    parser.add_argument('-c', '--cfg', required=True)
    parser.add_argument('--resume-dir', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--foreground-erode-iters', type=int, default=3)
    parser.add_argument('--component-percentile', type=float, default=95.0)
    parser.add_argument('--min-component-pixels', type=int, default=4)
    parser.add_argument('--max-proposals', type=int, default=8)
    parser.add_argument('--keep-top-components', type=int, default=3)
    parser.add_argument('--map-topk-ratio', type=float, default=0.01)
    parser.add_argument('--fill-value', default='blur', choices=['black', 'mean', 'zero', 'blur', 'local_mean'])
    parser.add_argument('--blur-kernel', type=int, default=21)
    parser.add_argument('--score-direction', default='original', choices=['original', 'model', 'reverse', 'neg'])
    parser.add_argument('--drop-scale', type=float, default=0.5)
    parser.add_argument('--heat-weight', type=float, default=1.0)
    parser.add_argument('--shape-weight', type=float, default=0.5)
    parser.add_argument('--drop-weight', type=float, default=0.5)
    parser.add_argument('--edge-penalty-weight', type=float, default=0.5)
    parser.add_argument('--area-penalty-weight', type=float, default=0.25)
    parser.add_argument('--min-area-ratio', type=float, default=0.0003)
    parser.add_argument('--max-area-ratio', type=float, default=0.20)
    parser.add_argument('--soft-alpha', type=float, default=0.2)
    parser.add_argument('--soft-beta', type=float, default=0.1)
    parser.add_argument('--soft-min-weight', type=float, default=0.8)
    parser.add_argument('--soft-max-weight', type=float, default=1.2)
    parser.add_argument('--vis-per-organ', type=int, default=30)
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

    data_name = getattr(getattr(cfg, 'data_test', cfg.data), 'name', getattr(cfg.data, 'name', 'data'))
    output_dir = args.output_dir or os.path.join(cfg.logdir, f'region_verified_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    variants = ['raw', 'component_only', 'region_verified', 'soft_region_verified']
    pixel_store = {
        variant: defaultdict(lambda: {'labels': [], 'scores': []})
        for variant in variants
    }
    records = []
    proposal_rows = []
    vis_counts = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(trainer.test_loader, start=1):
            if args.max_batches and batch_idx > args.max_batches:
                break
            trainer.set_input(batch)
            trainer.forward()

            imgs = trainer.imgs
            gt_masks = trainer.imgs_mask
            if gt_masks.ndim == 3:
                gt_masks = gt_masks.unsqueeze(1)
            raw_map = trainer.anomaly_map.detach()
            if raw_map.ndim == 4:
                raw_map = raw_map.squeeze(1)
            raw_map_4d = raw_map.unsqueeze(1)
            original_score = _score_direction(trainer.image_score.detach(), args.score_direction)

            foreground = _foreground_mask(imgs, args.foreground_threshold)
            foreground = _resize_bool(foreground, raw_map.shape[-2:])
            interior = _erode(foreground, args.foreground_erode_iters)
            edge = foreground & ~interior
            gt_on_map = _resize_bool(gt_masks > 0.5, raw_map.shape[-2:])

            raw_image_score = _topk_image_score(raw_map_4d, foreground, args.map_topk_ratio)
            component_maps = []
            verified_maps = []
            soft_maps = []

            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), imgs.shape[0])
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), imgs.shape[0])

            for idx in range(imgs.shape[0]):
                raw_np = raw_map[idx].detach().cpu().numpy().astype(np.float32)
                fg_np = foreground[idx, 0].detach().cpu().numpy().astype(bool)
                edge_np = edge[idx, 0].detach().cpu().numpy().astype(bool)
                fg_values = raw_np[fg_np]
                if fg_values.size == 0:
                    threshold = float(np.percentile(raw_np, args.component_percentile))
                else:
                    threshold = float(np.percentile(fg_values, args.component_percentile))
                binary = (raw_np >= threshold) & fg_np
                components = _connected_components(binary)
                proposals = []
                for comp_idx, coords in enumerate(components):
                    if len(coords) < int(args.min_component_pixels):
                        continue
                    comp_mask = _component_mask(coords, raw_np.shape)
                    stats = _component_stats(comp_mask, raw_np, fg_np, edge_np)
                    if stats is None:
                        continue
                    comp_tensor = torch.from_numpy(comp_mask).to(device=imgs.device).view(1, 1, *raw_np.shape)
                    masked_img = _apply_deletion(imgs[idx:idx + 1], comp_tensor, args.fill_value, args.blur_kernel)
                    _, masked_score = _forward_model(trainer, masked_img, cls_names=[organs[idx]])
                    masked_score = _score_direction(masked_score.detach(), args.score_direction)
                    drop = float((original_score[idx] - masked_score[0]).detach().cpu())
                    verify_score, shape_score, area_penalty, drop_score = _proposal_score(stats, drop, args)
                    proposal = {
                        **stats,
                        'component_index': comp_idx,
                        'mask': comp_mask,
                        'drop': drop,
                        'drop_score': drop_score,
                        'shape_score': shape_score,
                        'area_penalty': area_penalty,
                        'verify_score': verify_score,
                    }
                    proposals.append(proposal)

                proposals = sorted(proposals, key=lambda item: item['verify_score'], reverse=True)[:int(args.max_proposals)]
                kept = proposals[:int(args.keep_top_components)]
                comp_map, verified_map, soft_map = _make_region_maps(raw_np, kept, raw_np.shape, args)
                component_maps.append(torch.from_numpy(comp_map).to(device=imgs.device))
                verified_maps.append(torch.from_numpy(verified_map).to(device=imgs.device))
                soft_maps.append(torch.from_numpy(soft_map).to(device=imgs.device))

                for rank, proposal in enumerate(proposals):
                    row = {k: v for k, v in proposal.items() if k != 'mask'}
                    row.update({
                        'organ': organs[idx],
                        'img_path': img_paths[idx],
                        'label': labels[idx],
                        'proposal_rank': rank,
                        'kept': int(rank < int(args.keep_top_components)),
                    })
                    proposal_rows.append(row)

            component_map = torch.stack(component_maps, dim=0)
            verified_map = torch.stack(verified_maps, dim=0)
            soft_map = torch.stack(soft_maps, dim=0)
            component_image_score = _topk_image_score(component_map.unsqueeze(1), foreground, args.map_topk_ratio)
            verified_image_score = _topk_image_score(verified_map.unsqueeze(1), foreground, args.map_topk_ratio)
            soft_image_score = _topk_image_score(soft_map.unsqueeze(1), foreground, args.map_topk_ratio)
            map_by_variant = {
                'raw': raw_map,
                'component_only': component_map,
                'region_verified': verified_map,
                'soft_region_verified': soft_map,
            }
            image_score_by_variant = {
                'raw': raw_image_score,
                'component_only': component_image_score,
                'region_verified': verified_image_score,
                'soft_region_verified': soft_image_score,
            }

            for variant, score_map in map_by_variant.items():
                for idx, organ in enumerate(organs):
                    pixel_store[variant][organ]['labels'].append(gt_on_map[idx, 0].detach().cpu().numpy().reshape(-1).astype(np.uint8))
                    pixel_store[variant][organ]['scores'].append(score_map[idx].detach().cpu().numpy().reshape(-1).astype(np.float32))
                    pixel_store[variant]['Avg']['labels'].append(gt_on_map[idx, 0].detach().cpu().numpy().reshape(-1).astype(np.uint8))
                    pixel_store[variant]['Avg']['scores'].append(score_map[idx].detach().cpu().numpy().reshape(-1).astype(np.float32))

            for idx, organ in enumerate(organs):
                records.append({
                    'organ': organ,
                    'img_path': img_paths[idx],
                    'mask_path': mask_paths[idx],
                    'label': labels[idx],
                    'gt_pixels': int(gt_on_map[idx, 0].sum().detach().cpu()),
                    'model_image_score': float(original_score[idx].detach().cpu()),
                    'raw_image_score': float(image_score_by_variant['raw'][idx].detach().cpu()),
                    'component_only_image_score': float(image_score_by_variant['component_only'][idx].detach().cpu()),
                    'region_verified_image_score': float(image_score_by_variant['region_verified'][idx].detach().cpu()),
                    'soft_region_verified_image_score': float(image_score_by_variant['soft_region_verified'][idx].detach().cpu()),
                    'num_proposals': int(sum(1 for row in proposal_rows if row['img_path'] == img_paths[idx])),
                })
                if args.vis_per_organ > 0 and vis_counts[organ] < args.vis_per_organ:
                    name = os.path.splitext(os.path.basename(img_paths[idx]))[0] or f'batch{batch_idx}_item{idx}'
                    _save_panel(
                        os.path.join(output_dir, 'debug_vis', organ, f'{vis_counts[organ]:03d}_{name}_label{labels[idx]}.png'),
                        imgs[idx],
                        gt_masks[idx],
                        raw_map[idx].detach().cpu().numpy(),
                        component_map[idx].detach().cpu().numpy(),
                        verified_map[idx].detach().cpu().numpy(),
                        soft_map[idx].detach().cpu().numpy(),
                    )
                    vis_counts[organ] += 1

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} batches, records={len(records)}, proposals={len(proposal_rows)}')

    if not records:
        raise RuntimeError('No records were processed.')

    record_fields = [
        'organ', 'img_path', 'mask_path', 'label', 'gt_pixels',
        'model_image_score', 'raw_image_score',
        'component_only_image_score', 'region_verified_image_score',
        'soft_region_verified_image_score',
        'num_proposals',
    ]
    proposal_fields = [
        'organ', 'img_path', 'label', 'proposal_rank', 'kept',
        'component_index', 'area', 'area_ratio',
        'bbox_y1', 'bbox_x1', 'bbox_y2', 'bbox_x2',
        'compactness', 'edge_overlap', 'foreground_overlap',
        'mean_score', 'max_score',
        'drop', 'drop_score', 'shape_score', 'area_penalty', 'verify_score',
        'soft_norm_score', 'soft_weight',
    ]
    _write_csv(os.path.join(output_dir, 'region_verified_records.csv'), records, record_fields)
    _write_csv(os.path.join(output_dir, 'region_verified_proposals.csv'), proposal_rows, proposal_fields)
    metrics = _summarize(records, pixel_store)
    metric_fields = [
        'variant', 'organ', 'n',
        'image_AUROC', 'image_AP', 'image_F1',
        'pixel_AUROC', 'pixel_AP', 'pixel_F1_max',
    ]
    _write_csv(os.path.join(output_dir, 'region_verified_metrics.csv'), metrics, metric_fields)

    print(f'Output: {output_dir}')
    print(f'Records: {os.path.join(output_dir, "region_verified_records.csv")}')
    print(f'Proposals: {os.path.join(output_dir, "region_verified_proposals.csv")}')
    print(f'Metrics: {os.path.join(output_dir, "region_verified_metrics.csv")}')
    for row in metrics:
        if row['organ'] == 'Avg':
            print(
                f"{row['variant']}: image_AUROC={row['image_AUROC']:.4f} "
                f"pixel_AUROC={row['pixel_AUROC']:.4f} "
                f"pixel_AP={row['pixel_AP']:.4f} F1={row['pixel_F1_max']:.4f}"
            )


if __name__ == '__main__':
    main()
