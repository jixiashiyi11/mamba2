import argparse
import copy
import csv
import math
import os
import random
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from configs import get_cfg
from trainer import get_trainer
from util.net import init_training
from util.util import init_checkpoint, run_pre


def _parse_ratios(text):
    ratios = []
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        ratio = float(item)
        if ratio <= 0 or ratio > 1:
            raise ValueError(f'Invalid ratio={ratio}; expected 0 < ratio <= 1.')
        ratios.append(ratio)
    if not ratios:
        raise ValueError('At least one top-k ratio is required.')
    return ratios


def _parse_csv_items(text):
    return [item.strip() for item in str(text).split(',') if item.strip()]


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
    if args.disable_debug_eval:
        cfg.debug_eval = False
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
    return imgs_01.max(dim=1, keepdim=True).values > threshold


def _erode(mask, iters):
    if iters <= 0:
        return mask
    x = mask.float()
    for _ in range(iters):
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return x > 0.5


def _resize_like(mask, target_hw, mode='nearest'):
    if mask.shape[-2:] == target_hw:
        return mask
    return F.interpolate(mask.float(), size=target_hw, mode=mode) > 0.5


def _topk_mask(scores, candidate_mask, ratio):
    if scores.ndim == 3:
        scores = scores.unsqueeze(1)
    batch_size, _, height, width = scores.shape
    masks = torch.zeros((batch_size, 1, height, width), device=scores.device, dtype=torch.bool)
    flat_scores = scores.flatten(1)
    flat_candidates = candidate_mask.flatten(1)
    for idx in range(batch_size):
        candidate_idx = torch.nonzero(flat_candidates[idx], as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            candidate_idx = torch.arange(flat_scores.shape[1], device=scores.device)
        k = max(1, int(math.ceil(float(candidate_idx.numel()) * ratio)))
        k = min(k, int(candidate_idx.numel()))
        selected_local = torch.topk(flat_scores[idx, candidate_idx], k=k, largest=True).indices
        selected = candidate_idx[selected_local]
        masks[idx].view(-1)[selected] = True
    return masks


def _random_mask(candidate_mask, ratio, generator):
    batch_size, _, height, width = candidate_mask.shape
    masks = torch.zeros((batch_size, 1, height, width), device=candidate_mask.device, dtype=torch.bool)
    flat_candidates = candidate_mask.flatten(1)
    for idx in range(batch_size):
        candidate_idx = torch.nonzero(flat_candidates[idx], as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            candidate_idx = torch.arange(flat_candidates.shape[1], device=candidate_mask.device)
        k = max(1, int(math.ceil(float(candidate_idx.numel()) * ratio)))
        k = min(k, int(candidate_idx.numel()))
        perm = torch.randperm(candidate_idx.numel(), device=candidate_mask.device, generator=generator)[:k]
        selected = candidate_idx[perm]
        masks[idx].view(-1)[selected] = True
    return masks


def _odd_kernel(value):
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def _apply_deletion(imgs, delete_mask, fill_value, blur_kernel=21):
    mask = _resize_like(delete_mask, imgs.shape[-2:]).to(dtype=torch.bool)
    if fill_value in ('zero', 'mean'):
        # Normalized zero corresponds to ImageNet mean in raw image space.
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


def _score_after_deletion(trainer, imgs, delete_mask, fill_value, blur_kernel):
    masked_imgs = _apply_deletion(imgs, delete_mask, fill_value, blur_kernel)
    _, score = _forward_model(trainer, masked_imgs)
    return score.detach()


def _as_float_list(tensor):
    return [float(x) for x in tensor.detach().cpu().view(-1)]


def _apply_score_direction(score, direction):
    direction = str(direction).lower()
    if direction in ('original', 'orig', 'model'):
        return score
    if direction in ('reverse', 'reversed', 'neg'):
        return -score
    raise ValueError(f'Unsupported score direction={direction}')


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_mean(values):
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return float('nan')
    return float(sum(values) / len(values))


def _summarize(rows, ratios):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['fill_value'], row['score_direction'], row['organ'], row['label'])].append(row)
        grouped[(row['fill_value'], row['score_direction'], 'Avg', row['label'])].append(row)
    summary = []
    for (fill_value, score_direction, organ, label), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][3], item[0][2]),
    ):
        out = {
            'fill_value': fill_value,
            'score_direction': score_direction,
            'organ': organ,
            'label': label,
            'n': len(group),
            'orig_score_mean': _safe_mean([r['original_score'] for r in group]),
        }
        for ratio in ratios:
            suffix = f'{int(round(ratio * 100))}'
            pred_key = f'pred_top{suffix}_drop'
            rand_key = f'random_fg_top{suffix}_drop'
            edge_key = f'edge_top{suffix}_drop'
            out[f'{pred_key}_mean'] = _safe_mean([r[pred_key] for r in group])
            out[f'{rand_key}_mean'] = _safe_mean([r[rand_key] for r in group])
            out[f'{edge_key}_mean'] = _safe_mean([r[edge_key] for r in group])
            out[f'pred_minus_random_top{suffix}_mean'] = _safe_mean([r[f'pred_minus_random_top{suffix}'] for r in group])
        out['gt_drop_mean'] = _safe_mean([r['gt_drop'] for r in group])
        summary.append(out)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Eval-only deletion consistency: does predicted heatmap support image anomaly score?'
    )
    parser.add_argument('-c', '--cfg', required=True, help='Config path, e.g. configs/mambaad/a_arcc_e2_busi_net_test.py')
    parser.add_argument('--resume-dir', default='', help='Optional run folder name under runs/.')
    parser.add_argument('--checkpoint', default='', help='Optional checkpoint filename/path, e.g. net.pth.')
    parser.add_argument('--output-dir', default='', help='Output directory. Default: <logdir>/deletion_consistency_<data_name>.')
    parser.add_argument('--top-ratios', default='0.01,0.05', help='Comma-separated foreground top-k ratios.')
    parser.add_argument('--foreground-threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--foreground-erode-iters', type=int, default=3)
    parser.add_argument(
        '--fill-values',
        default='black,mean,blur,local_mean',
        help='Comma-separated deletion fills: black,mean,blur,local_mean. `zero` is an alias of mean.',
    )
    parser.add_argument(
        '--score-directions',
        default='original,reverse',
        help='Comma-separated score directions: original,reverse. reverse multiplies image_score by -1.',
    )
    parser.add_argument('--blur-kernel', type=int, default=21)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--disable-debug-eval', action='store_true')
    args = parser.parse_args()

    ratios = _parse_ratios(args.top_ratios)
    fill_values = _parse_csv_items(args.fill_values)
    score_directions = _parse_csv_items(args.score_directions)
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
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(cfg.logdir, f'deletion_consistency_{data_name}')
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    generator = torch.Generator(device=f'cuda:{cfg.local_rank}')
    generator.manual_seed(args.seed + cfg.local_rank)

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
            anomaly_map = trainer.anomaly_map
            if anomaly_map.ndim == 3:
                anomaly_map_4d = anomaly_map.unsqueeze(1)
            else:
                anomaly_map_4d = anomaly_map
            original_score = trainer.image_score.detach()

            map_hw = anomaly_map_4d.shape[-2:]
            foreground = _foreground_mask(imgs, args.foreground_threshold)
            foreground = _resize_like(foreground, map_hw)
            eroded = _erode(foreground, args.foreground_erode_iters)
            edge = foreground & ~eroded
            gt_on_map = _resize_like(gt_masks > 0.5, map_hw)

            masks_by_name = {}
            for ratio in ratios:
                suffix = f'{int(round(ratio * 100))}'
                masks_by_name[f'pred_top{suffix}'] = _topk_mask(anomaly_map_4d, foreground, ratio)
                masks_by_name[f'random_fg_top{suffix}'] = _random_mask(foreground, ratio, generator)
                edge_candidate = edge
                if edge_candidate.flatten(1).sum(dim=1).min().item() == 0:
                    edge_candidate = foreground
                masks_by_name[f'edge_top{suffix}'] = _topk_mask(anomaly_map_4d, edge_candidate, ratio)
            masks_by_name['gt'] = gt_on_map

            scores_after_by_fill = {}
            for fill_value in fill_values:
                scores_after = {}
                for name, delete_mask in masks_by_name.items():
                    scores_after[name] = _score_after_deletion(
                        trainer,
                        imgs,
                        delete_mask,
                        fill_value,
                        args.blur_kernel,
                    )
                scores_after_by_fill[fill_value] = scores_after

            batch_size = imgs.shape[0]
            img_paths = _normalize_paths(getattr(trainer, 'img_path', None), batch_size)
            mask_paths = _normalize_paths(getattr(trainer, 'mask_path', None), batch_size)
            labels = [int(x) for x in trainer.anomaly.detach().cpu().view(-1)]
            organs = [str(x) for x in trainer.cls_name]
            gt_pixels = gt_on_map.flatten(1).sum(dim=1).detach().cpu().numpy().astype(int).tolist()

            for fill_value, scores_after in scores_after_by_fill.items():
                for score_direction in score_directions:
                    directed_original = _apply_score_direction(original_score, score_direction)
                    orig_list = _as_float_list(directed_original)
                    score_lists = {
                        name: _as_float_list(_apply_score_direction(score, score_direction))
                        for name, score in scores_after.items()
                    }
                    for idx in range(batch_size):
                        row = {
                            'fill_value': fill_value,
                            'score_direction': score_direction,
                            'organ': organs[idx],
                            'img_path': img_paths[idx],
                            'mask_path': mask_paths[idx],
                            'label': labels[idx],
                            'gt_pixels': gt_pixels[idx],
                            'original_score': orig_list[idx],
                        }
                        for ratio in ratios:
                            suffix = f'{int(round(ratio * 100))}'
                            pred_name = f'pred_top{suffix}'
                            rand_name = f'random_fg_top{suffix}'
                            edge_name = f'edge_top{suffix}'
                            row[f'{pred_name}_score'] = score_lists[pred_name][idx]
                            row[f'{rand_name}_score'] = score_lists[rand_name][idx]
                            row[f'{edge_name}_score'] = score_lists[edge_name][idx]
                            row[f'{pred_name}_drop'] = orig_list[idx] - score_lists[pred_name][idx]
                            row[f'{rand_name}_drop'] = orig_list[idx] - score_lists[rand_name][idx]
                            row[f'{edge_name}_drop'] = orig_list[idx] - score_lists[edge_name][idx]
                            row[f'pred_minus_random_top{suffix}'] = row[f'{pred_name}_drop'] - row[f'{rand_name}_drop']
                        row['gt_score'] = score_lists['gt'][idx]
                        row['gt_drop'] = orig_list[idx] - score_lists['gt'][idx] if gt_pixels[idx] > 0 else float('nan')
                        rows.append(row)

            if batch_idx % 20 == 0:
                print(f'Processed {batch_idx} batches, {len(rows)} samples')

    if not rows:
        raise RuntimeError('No samples were processed.')

    ratio_suffixes = [f'{int(round(r * 100))}' for r in ratios]
    fieldnames = [
        'fill_value',
        'score_direction',
        'organ',
        'img_path',
        'mask_path',
        'label',
        'gt_pixels',
        'original_score',
    ]
    for suffix in ratio_suffixes:
        fieldnames.extend([
            f'pred_top{suffix}_score',
            f'random_fg_top{suffix}_score',
            f'edge_top{suffix}_score',
            f'pred_top{suffix}_drop',
            f'random_fg_top{suffix}_drop',
            f'edge_top{suffix}_drop',
            f'pred_minus_random_top{suffix}',
        ])
    fieldnames.extend(['gt_score', 'gt_drop'])

    records_path = os.path.join(output_dir, 'deletion_consistency_records.csv')
    _write_csv(records_path, rows, fieldnames)

    summary = _summarize(rows, ratios)
    summary_fields = ['fill_value', 'score_direction', 'organ', 'label', 'n', 'orig_score_mean']
    for suffix in ratio_suffixes:
        summary_fields.extend([
            f'pred_top{suffix}_drop_mean',
            f'random_fg_top{suffix}_drop_mean',
            f'edge_top{suffix}_drop_mean',
            f'pred_minus_random_top{suffix}_mean',
        ])
    summary_fields.append('gt_drop_mean')
    summary_path = os.path.join(output_dir, 'deletion_consistency_summary.csv')
    _write_csv(summary_path, summary, summary_fields)

    print(f'Output: {output_dir}')
    print(f'Records: {records_path}')
    print(f'Summary: {summary_path}')
    for row in summary:
        if str(row['label']) == '1':
            msg = f"{row['fill_value']} {row['score_direction']} {row['organ']} label=1 n={row['n']}"
            for suffix in ratio_suffixes:
                msg += (
                    f" pred_top{suffix}_drop={row[f'pred_top{suffix}_drop_mean']:.6f}"
                    f" random_top{suffix}_drop={row[f'random_fg_top{suffix}_drop_mean']:.6f}"
                    f" margin={row[f'pred_minus_random_top{suffix}_mean']:.6f}"
                )
            msg += f" gt_drop={row['gt_drop_mean']:.6f}"
            print(msg)


if __name__ == '__main__':
    main()
