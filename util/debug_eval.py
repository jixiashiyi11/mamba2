import csv
import json
import math
import os
import random
from collections import defaultdict

import matplotlib.cm as cm
import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from util.util import log_msg


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _safe_float(value):
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return ''
    return value


def _safe_metric(fn, y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return np.nan


def _best_f1(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    finite = f1_scores[np.isfinite(f1_scores)]
    return float(finite.max()) if finite.size > 0 else np.nan


def _normalize_map(anomaly_map):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    min_v = float(np.min(anomaly_map))
    max_v = float(np.max(anomaly_map))
    if max_v <= min_v:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return (anomaly_map - min_v) / (max_v - min_v)


def _normalize_map_with_bounds(anomaly_map, low, high):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    low = float(low)
    high = float(high)
    if high <= low:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return np.clip((anomaly_map - low) / (high - low), 0.0, 1.0)


IMAGE_SCORE_AGGREGATIONS = [
    ('max', None),
    ('top1%', 0.01),
    ('top5%', 0.05),
    ('top10%', 0.10),
    ('mean', None),
]


def compute_foreground_masks_from_images(imgs, threshold=5.0 / 255.0):
    if torch.is_tensor(imgs):
        imgs_np = imgs.detach().cpu().float().numpy()
    else:
        imgs_np = np.asarray(imgs, dtype=np.float32)
    if imgs_np.ndim != 4:
        raise ValueError(f'expected NCHW images, got shape {imgs_np.shape}')
    imgs_np = imgs_np.transpose(0, 2, 3, 1)
    imgs_np = imgs_np * IMAGENET_STD[None, None, None, :] + IMAGENET_MEAN[None, None, None, :]
    imgs_np = np.clip(imgs_np, 0.0, 1.0)
    return (imgs_np.max(axis=-1) > float(threshold)).astype(np.uint8)


def _score_topk_mean(anomaly_maps, ratio):
    flat = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    k = max(1, int(flat.shape[1] * ratio))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def _score_model_default(anomaly_maps, ratio):
    flat = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    if ratio is None:
        return flat.max(axis=1)
    return _score_topk_mean(anomaly_maps, ratio)


def compute_image_score_variants(anomaly_maps):
    flat = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    max_score = flat.max(axis=1)
    mean_score = flat.mean(axis=1)
    return {
        'max': max_score,
        'top1%': _score_topk_mean(anomaly_maps, 0.01),
        'top5%': _score_topk_mean(anomaly_maps, 0.05),
        'top10%': _score_topk_mean(anomaly_maps, 0.10),
        'mean': mean_score,
    }


def _score_one_map(values, ratio):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.nan
    if ratio == 'mean':
        return float(values.mean())
    if ratio is None:
        return float(values.max())
    k = max(1, int(values.size * float(ratio)))
    return float(np.partition(values, values.size - k)[-k:].mean())


def compute_masked_image_scores(anomaly_maps, foreground_masks, ratio):
    scores = []
    for amap, fg_mask in zip(anomaly_maps, foreground_masks):
        fg = np.asarray(fg_mask).astype(bool)
        values = np.asarray(amap, dtype=np.float32)[fg]
        if values.size == 0:
            values = np.asarray(amap, dtype=np.float32).reshape(-1)
        scores.append(_score_one_map(values, ratio))
    return np.asarray(scores, dtype=np.float32)


def compute_masked_image_score_variants(anomaly_maps, foreground_masks):
    variants = {}
    for name, ratio in IMAGE_SCORE_AGGREGATIONS:
        if name == 'mean':
            scores = []
            for amap, fg_mask in zip(anomaly_maps, foreground_masks):
                fg = np.asarray(fg_mask).astype(bool)
                values = np.asarray(amap, dtype=np.float32)[fg]
                if values.size == 0:
                    values = np.asarray(amap, dtype=np.float32).reshape(-1)
                scores.append(float(values.mean()))
            variants[name] = np.asarray(scores, dtype=np.float32)
        else:
            variants[name] = compute_masked_image_scores(anomaly_maps, foreground_masks, ratio)
    return variants


def mask_anomaly_maps_to_foreground(anomaly_maps, foreground_masks, background_value='min'):
    masked_maps = np.asarray(anomaly_maps, dtype=np.float32).copy()
    foreground_masks = np.asarray(foreground_masks).astype(bool)
    for idx in range(masked_maps.shape[0]):
        amap = masked_maps[idx]
        fg = foreground_masks[idx]
        if background_value == 'zero':
            fill_value = 0.0
        elif fg.any():
            fill_value = float(amap[fg].min())
        else:
            fill_value = float(amap.min())
        amap[~fg] = fill_value
    return masked_maps


def _topk_background_fraction(anomaly_maps, foreground_masks, ratio):
    region_fractions = _topk_region_fractions(anomaly_maps, foreground_masks, None, ratio)
    return region_fractions['background']


def _topk_region_fractions(anomaly_maps, foreground_masks, eroded_foreground_masks, ratio):
    flat_maps = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    flat_fg = foreground_masks.reshape(foreground_masks.shape[0], -1).astype(bool)
    if eroded_foreground_masks is None:
        flat_inner = flat_fg
    else:
        flat_inner = eroded_foreground_masks.reshape(eroded_foreground_masks.shape[0], -1).astype(bool)
    flat_edge = flat_fg & ~flat_inner
    regions = {
        'background': ~flat_fg,
        'foreground_edge': flat_edge,
        'foreground_interior': flat_inner,
    }
    out = {name: [] for name in regions}
    for scores, region_masks in zip(flat_maps, zip(*[regions[name] for name in regions])):
        if ratio is None:
            k = 1
        else:
            k = max(1, int(scores.size * float(ratio)))
        top_idx = np.argpartition(scores, scores.size - k)[-k:]
        for name, region_mask in zip(regions, region_masks):
            out[name].append(float(region_mask[top_idx].mean()))
    return {name: np.asarray(values, dtype=np.float32) for name, values in out.items()}


def erode_foreground_masks(foreground_masks, iterations=3):
    foreground_masks = np.asarray(foreground_masks).astype(bool)
    if int(iterations) <= 0:
        return foreground_masks.astype(np.uint8)
    eroded = np.zeros_like(foreground_masks, dtype=bool)
    for idx, fg in enumerate(foreground_masks):
        if not fg.any():
            continue
        item = binary_erosion(fg, iterations=int(iterations))
        if not item.any():
            item = fg
        eroded[idx] = item
    return eroded.astype(np.uint8)


def compute_region_masked_scores(anomaly_maps, region_masks, ratio):
    fractions = []
    for amap, region_mask in zip(anomaly_maps, region_masks):
        valid = np.asarray(region_mask).astype(bool)
        values = np.asarray(amap, dtype=np.float32)[valid]
        if values.size == 0:
            values = np.asarray(amap, dtype=np.float32).reshape(-1)
        fractions.append(_score_one_map(values, ratio))
    return np.asarray(fractions, dtype=np.float32)


def _resize_maps_to_masks(anomaly_maps, masks):
    if anomaly_maps.shape[-2:] == masks.shape[-2:]:
        return anomaly_maps, False
    maps_t = torch.from_numpy(anomaly_maps).float().unsqueeze(1)
    maps_t = F.interpolate(maps_t, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    return maps_t.squeeze(1).numpy(), True


def _resize_binary_masks_to_shape(masks, shape):
    masks = np.asarray(masks)
    if masks.shape[-2:] == tuple(shape):
        return masks.astype(np.uint8), False
    masks_t = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)
    masks_t = F.interpolate(masks_t, size=shape, mode='nearest')
    return (masks_t.squeeze(1).numpy() > 0.5).astype(np.uint8), True


class DebugEvalHelper:
    def __init__(self, cfg, logger=None, rank=0, master=True):
        self.cfg = cfg
        self.logger = logger
        self.rank = rank
        self.master = master
        self.enabled = bool(getattr(cfg, 'debug_eval', False))
        self.vis_per_organ = int(getattr(cfg, 'debug_eval_vis_per_organ', 20))
        self.seed = int(getattr(cfg, 'debug_eval_seed', 42))
        self.max_records_preview = int(getattr(cfg, 'debug_eval_preview_rows', 8))
        self.foreground_enabled = bool(getattr(cfg, 'debug_eval_foreground_mask', True))
        self.foreground_threshold = float(getattr(cfg, 'debug_eval_foreground_threshold', 5.0 / 255.0))
        self.foreground_background_value = str(getattr(cfg, 'debug_eval_foreground_background_value', 'min'))
        self.foreground_erode_iters = int(getattr(cfg, 'debug_eval_foreground_erode_iters', 3))
        score_modes = getattr(
            cfg,
            'debug_eval_score_modes',
            ['model_top1', 'fg_top1', 'fg_eroded_top1', 'fg_top5', 'fg_mean'],
        )
        if isinstance(score_modes, str):
            score_modes = [item.strip() for item in score_modes.split(',') if item.strip()]
        self.score_modes = list(score_modes)
        self.vis_norm = str(getattr(cfg, 'debug_eval_vis_norm', 'both')).lower()
        self.vis_percentile_low = float(getattr(cfg, 'debug_eval_vis_percentile_low', 1.0))
        self.vis_percentile_high = float(getattr(cfg, 'debug_eval_vis_percentile_high', 99.0))
        base_dir = getattr(cfg, 'logdir', None) or getattr(cfg.trainer, 'checkpoint', 'runs')
        self.out_dir = os.path.join(base_dir, 'debug_eval')
        self.vis_dir = os.path.join(base_dir, 'debug_vis')
        self._vis_percentile_bounds = None
        self._rng = random.Random(self.seed + self.rank)
        self._seen = defaultdict(int)
        self._samples = defaultdict(list)

    def add_vis_batch(self, imgs, masks, anomaly_maps, image_scores, cls_names, labels, img_paths, foreground_masks=None):
        if not self.enabled or self.vis_per_organ <= 0:
            return
        imgs_np = imgs.detach().cpu().float().numpy()
        masks_np = masks.detach().cpu().numpy()
        maps_np = anomaly_maps.detach().cpu().float().numpy()
        scores_np = image_scores.detach().cpu().float().numpy().reshape(-1)
        labels_np = labels.detach().cpu().numpy().astype(int).reshape(-1)
        if masks_np.ndim == 4:
            masks_np = np.squeeze(masks_np, axis=1)
        if maps_np.ndim == 4:
            maps_np = np.squeeze(maps_np, axis=1)
        if foreground_masks is None and self.foreground_enabled:
            foreground_masks = compute_foreground_masks_from_images(imgs, self.foreground_threshold)
        if foreground_masks is not None:
            foreground_masks = np.asarray(foreground_masks).astype(np.uint8)
            if foreground_masks.ndim == 4 and foreground_masks.shape[1] == 1:
                foreground_masks = np.squeeze(foreground_masks, axis=1)
        paths = self._normalize_paths(img_paths, len(labels_np))

        for idx in range(len(labels_np)):
            organ = str(cls_names[idx])
            label_key = 'abnormal' if int(labels_np[idx]) == 1 else 'normal'
            key = (organ, label_key)
            self._seen[key] += 1
            sample = {
                'img': imgs_np[idx].copy(),
                'mask': masks_np[idx].copy(),
                'amap': maps_np[idx].copy(),
                'score': float(scores_np[idx]),
                'label': int(labels_np[idx]),
                'organ': organ,
                'path': paths[idx],
            }
            if foreground_masks is not None:
                sample['foreground_mask'] = foreground_masks[idx].copy()
            bucket = self._samples[key]
            per_label_cap = max(1, self.vis_per_organ // 2)
            if len(bucket) < per_label_cap:
                bucket.append(sample)
            else:
                replace_idx = self._rng.randint(0, self._seen[key] - 1)
                if replace_idx < per_label_cap:
                    bucket[replace_idx] = sample

    def save_visualizations(self):
        if not self.enabled:
            return
        for (organ, _), samples in self._samples.items():
            organ_dir = os.path.join(self.vis_dir, self._safe_name(organ))
            os.makedirs(organ_dir, exist_ok=True)
            for sample_idx, sample in enumerate(samples):
                self._save_sample_visualization(organ_dir, sample, sample_idx)

    def write_and_summarize(self, results, evaluator):
        if not self.enabled or not self.master:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        records, shape_summary = self._build_records(results)
        records_path = os.path.join(self.out_dir, 'debug_eval_records.csv')
        self._write_csv(records_path, records)

        score_rows, score_summary = self._build_score_distribution(results)
        score_path = os.path.join(self.out_dir, 'score_distribution.csv')
        self._write_csv(score_path, score_rows)
        hist_rows = self._build_score_histogram(score_rows)
        hist_path = os.path.join(self.out_dir, 'score_histogram.csv')
        self._write_csv(hist_path, hist_rows)

        score_sweep_metrics = self._score_sweep_metrics(results, evaluator)
        sweep_path = os.path.join(self.out_dir, 'score_sweep_metrics.csv')
        self._write_csv(sweep_path, self._flatten_score_sweep_metrics(score_sweep_metrics))
        foreground_rows, foreground_summary = self._build_foreground_diagnostic(results, evaluator)
        foreground_path = os.path.join(self.out_dir, 'foreground_mask_diagnostic.csv')
        self._write_csv(foreground_path, foreground_rows)
        foreground_score_rows, foreground_score_summary = self._build_foreground_score_sweep(results)
        foreground_score_path = os.path.join(self.out_dir, 'foreground_score_sweep_metrics.csv')
        self._write_csv(foreground_score_path, foreground_score_rows)
        false_positive_rows, false_positive_summary = self._build_false_positive_region_diagnostic(results)
        false_positive_path = os.path.join(self.out_dir, 'false_positive_region_diagnostic.csv')
        self._write_csv(false_positive_path, false_positive_rows)
        self._vis_percentile_bounds = self._percentile_bounds_from_results(results)
        self.save_visualizations()
        self._log_debug_summary(
            records_path,
            score_path,
            hist_path,
            sweep_path,
            foreground_path,
            foreground_score_path,
            false_positive_path,
            records,
            shape_summary,
            score_sweep_metrics,
            score_summary,
            foreground_summary,
            foreground_score_summary,
            false_positive_summary,
            evaluator,
        )

    def _build_records(self, results):
        masks = self._squeeze_maps(results['imgs_masks']).astype(int)
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        img_paths = self._normalize_paths(results.get('img_paths'), len(labels))
        mask_paths = self._normalize_paths(results.get('mask_paths'), len(labels), fill_value='')
        meta_lookup = self._load_meta_lookup()
        if 'image_scores' in results:
            image_scores = results['image_scores'].reshape(-1)
        else:
            image_scores = maps.reshape(maps.shape[0], -1).max(axis=1)

        input_shapes = results.get('model_input_shapes')
        map_shapes = results.get('anomaly_map_shapes')
        mask_shapes = results.get('gt_mask_shapes')
        result_raw_positive = results.get('raw_positive_pixels', None)
        if result_raw_positive is not None:
            result_raw_positive = result_raw_positive.reshape(-1)
        final_map_shape = maps.shape[-2:]
        final_mask_shape = masks.shape[-2:]
        shape_mismatch = tuple(final_map_shape) != tuple(final_mask_shape)
        foreground_masks = self._foreground_masks_for_maps(results, maps)
        current_topk = self._current_image_score_topk_ratio()
        if foreground_masks is not None:
            eroded_foreground_masks = erode_foreground_masks(foreground_masks, self.foreground_erode_iters)
            topk_regions = _topk_region_fractions(maps, foreground_masks, eroded_foreground_masks, current_topk)
        else:
            eroded_foreground_masks = None
            topk_regions = None
        adapter_debug = {}
        for key in [
            'adapter_feature_delta_l2',
            'adapter_feature_delta_abs',
            'adapter_raw_l2',
            'adapter_refined_l2',
        ]:
            if key in results:
                adapter_debug[key] = np.asarray(results[key]).reshape(-1)

        records = []
        for idx in range(len(labels)):
            mask = masks[idx]
            amap = maps[idx]
            image_path = img_paths[idx]
            mask_path = mask_paths[idx] or meta_lookup.get(image_path, {}).get('mask_path', '')
            mask_info = self._inspect_raw_mask(mask_path)
            if mask_info['raw_positive_pixels'] == '' and result_raw_positive is not None:
                mask_info['raw_positive_pixels'] = int(result_raw_positive[idx])
            original_shape = self._read_original_shape(image_path)
            foreground_info = {}
            if foreground_masks is not None:
                fg = foreground_masks[idx].astype(bool)
                fg_eroded = eroded_foreground_masks[idx].astype(bool)
                bg = ~fg
                edge = fg & ~fg_eroded
                fg_values = amap[fg]
                eroded_values = amap[fg_eroded]
                edge_values = amap[edge]
                bg_values = amap[bg]
                foreground_info = {
                    'foreground_pixels': int(fg.sum()),
                    'foreground_ratio': _safe_float(fg.mean()),
                    'foreground_eroded_pixels': int(fg_eroded.sum()),
                    'foreground_eroded_ratio': _safe_float(fg_eroded.mean()),
                    'topk_background_fraction': _safe_float(topk_regions['background'][idx]) if topk_regions is not None else '',
                    'topk_foreground_edge_fraction': _safe_float(topk_regions['foreground_edge'][idx]) if topk_regions is not None else '',
                    'topk_foreground_interior_fraction': _safe_float(topk_regions['foreground_interior'][idx]) if topk_regions is not None else '',
                    'foreground_anomaly_map_max': _safe_float(fg_values.max()) if fg_values.size else '',
                    'foreground_anomaly_map_mean': _safe_float(fg_values.mean()) if fg_values.size else '',
                    'foreground_eroded_anomaly_map_max': _safe_float(eroded_values.max()) if eroded_values.size else '',
                    'foreground_eroded_anomaly_map_mean': _safe_float(eroded_values.mean()) if eroded_values.size else '',
                    'foreground_edge_anomaly_map_max': _safe_float(edge_values.max()) if edge_values.size else '',
                    'foreground_edge_anomaly_map_mean': _safe_float(edge_values.mean()) if edge_values.size else '',
                    'background_anomaly_map_max': _safe_float(bg_values.max()) if bg_values.size else '',
                    'background_anomaly_map_mean': _safe_float(bg_values.mean()) if bg_values.size else '',
                }
            record = {
                'image_path': image_path,
                'mask_path': mask_path,
                'class_name': cls_names[idx],
                'organ_name': cls_names[idx],
                'image_label': int(labels[idx]),
                'mask_sum': int(mask.sum()),
                'final_mask_sum': int(mask.sum()),
                'mask_file_exists': int(mask_info['exists']),
                'raw_mask_sum': mask_info['raw_mask_sum'],
                'raw_positive_pixels': mask_info['raw_positive_pixels'],
                'mask_status': self._mask_status(int(labels[idx]), int(mask.sum()), mask_path, mask_info),
                'has_mask': int(mask.sum() > 0),
                'anomaly_score': _safe_float(image_scores[idx]),
                'anomaly_map_min': _safe_float(amap.min()),
                'anomaly_map_max': _safe_float(amap.max()),
                'anomaly_map_mean': _safe_float(amap.mean()),
                'original_image_shape': self._shape_to_str(original_shape),
                'model_input_shape': self._shape_to_str(input_shapes[idx] if input_shapes is not None else None),
                'anomaly_map_shape_before_resize': self._shape_to_str(map_shapes[idx] if map_shapes is not None else amap.shape),
                'anomaly_map_shape_after_resize': self._shape_to_str(amap.shape),
                'gt_mask_shape': self._shape_to_str(mask_shapes[idx] if mask_shapes is not None else mask.shape),
                'final_metric_anomaly_map_shape': self._shape_to_str(final_map_shape),
                'final_metric_gt_mask_shape': self._shape_to_str(final_mask_shape),
                'final_metric_shape_match': int(not shape_mismatch),
                'eval_adapter_mode': str(getattr(self.cfg, 'eval_adapter_mode', 'trained')),
                'adapter_feature_delta_l2': _safe_float(adapter_debug['adapter_feature_delta_l2'][idx]) if 'adapter_feature_delta_l2' in adapter_debug else '',
                'adapter_feature_delta_abs': _safe_float(adapter_debug['adapter_feature_delta_abs'][idx]) if 'adapter_feature_delta_abs' in adapter_debug else '',
                'adapter_raw_l2': _safe_float(adapter_debug['adapter_raw_l2'][idx]) if 'adapter_raw_l2' in adapter_debug else '',
                'adapter_refined_l2': _safe_float(adapter_debug['adapter_refined_l2'][idx]) if 'adapter_refined_l2' in adapter_debug else '',
                **foreground_info,
            }
            records.append(record)

        summary = {
            'shape_match': not shape_mismatch,
            'final_map_shape': tuple(final_map_shape),
            'final_mask_shape': tuple(final_mask_shape),
            'resized_for_debug_metrics': shape_mismatch,
        }
        return records, summary

    def _foreground_masks_for_maps(self, results, maps):
        if not self.foreground_enabled or 'foreground_masks' not in results:
            return None
        foreground_masks = self._squeeze_maps(results['foreground_masks']).astype(np.uint8)
        foreground_masks, _ = _resize_binary_masks_to_shape(foreground_masks, maps.shape[-2:])
        return foreground_masks

    def _foreground_masks_for_masks(self, results, masks):
        if not self.foreground_enabled or 'foreground_masks' not in results:
            return None
        foreground_masks = self._squeeze_maps(results['foreground_masks']).astype(np.uint8)
        foreground_masks, _ = _resize_binary_masks_to_shape(foreground_masks, masks.shape[-2:])
        return foreground_masks

    def _build_score_distribution(self, results):
        masks = self._squeeze_maps(results['imgs_masks']).astype(int)
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        img_paths = self._normalize_paths(results.get('img_paths'), len(labels))
        rows = []

        summary = {}
        for direction_key, direction_name, direction_maps in self._directed_maps(maps):
            variants = compute_image_score_variants(direction_maps)
            flat_direction = direction_maps.reshape(direction_maps.shape[0], -1)
            for idx in range(len(labels)):
                amap = direction_maps[idx]
                for agg_name, agg_ratio in IMAGE_SCORE_AGGREGATIONS:
                    image_score = variants[agg_name][idx]
                    rows.append({
                        'score_direction': direction_key,
                        'score_direction_formula': direction_name,
                        'aggregation': agg_name,
                        'image_path': img_paths[idx],
                        'label': int(labels[idx]),
                        'image_label': int(labels[idx]),
                        'class_name': cls_names[idx],
                        'organ': cls_names[idx],
                        'image_score': _safe_float(image_score),
                        'anomaly_map_max': _safe_float(amap.max()),
                        'anomaly_map_mean': _safe_float(amap.mean()),
                        'topk_score': _safe_float(image_score) if agg_ratio is not None else '',
                        'mask_sum': int(masks[idx].sum()),
                    })

            direction_summary = {}
            for organ in sorted(set(cls_names.tolist())):
                idxes = cls_names == organ
                organ_summary = {}
                for agg_name, _ in IMAGE_SCORE_AGGREGATIONS:
                    scores = variants[agg_name][idxes]
                    organ_labels = labels[idxes]
                    normal = scores[organ_labels == 0]
                    abnormal = scores[organ_labels == 1]
                    organ_summary[agg_name] = self._distribution_stats(normal, abnormal)
                direction_summary[organ] = organ_summary
            summary[direction_key] = direction_summary
        return rows, summary

    def _build_score_histogram(self, score_rows, bins=30):
        if not score_rows:
            return []
        hist_rows = []
        groups = sorted(set((row['score_direction'], row['aggregation'], row['class_name']) for row in score_rows))
        for direction, aggregation, organ in groups:
            organ_rows = [
                row for row in score_rows
                if row['score_direction'] == direction
                and row['aggregation'] == aggregation
                and row['class_name'] == organ
            ]
            values = np.array([float(row['image_score']) for row in organ_rows], dtype=np.float32)
            if values.size == 0:
                continue
            min_v = float(values.min())
            max_v = float(values.max())
            if max_v <= min_v:
                edges = np.linspace(min_v - 0.5, max_v + 0.5, bins + 1)
            else:
                edges = np.linspace(min_v, max_v, bins + 1)
            for label_value, label_name in [(0, 'normal'), (1, 'abnormal')]:
                label_values = np.array(
                    [float(row['image_score']) for row in organ_rows if int(row['label']) == label_value],
                    dtype=np.float32,
                )
                counts, _ = np.histogram(label_values, bins=edges)
                for bin_idx, count in enumerate(counts):
                    hist_rows.append({
                        'score_direction': direction,
                        'aggregation': aggregation,
                        'class_name': organ,
                        'label': label_name,
                        'bin_left': float(edges[bin_idx]),
                        'bin_right': float(edges[bin_idx + 1]),
                        'count': int(count),
                    })
        return hist_rows

    def _directed_maps(self, maps):
        return [
            ('old', 'sim(normal)-sim(abnormal)', maps),
            ('reverse', 'sim(abnormal)-sim(normal)', -maps),
        ]

    def _score_sweep_metrics(self, results, evaluator):
        masks = self._squeeze_maps(results['imgs_masks']).astype(int)
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        raw_positive_pixels = results.get('raw_positive_pixels', None)
        if raw_positive_pixels is not None:
            raw_positive_pixels = raw_positive_pixels.reshape(-1)
        else:
            raw_positive_pixels = masks.reshape(masks.shape[0], -1).sum(axis=1)
        maps, _ = _resize_maps_to_masks(maps, masks)
        rows_by_combo = {}
        for direction_key, direction_name, direction_maps in self._directed_maps(maps):
            variants = compute_image_score_variants(direction_maps)
            pixel_by_organ = {}
            for organ in sorted(set(cls_names.tolist())):
                idxes = cls_names == organ
                organ_labels = labels[idxes]
                organ_raw_positive = raw_positive_pixels[idxes]
                pixel_keep = np.ones(organ_labels.shape[0], dtype=np.bool_)
                if getattr(evaluator, 'skip_tiny_mask_for_pixel', False):
                    pixel_keep = ~(
                        (organ_labels == 1)
                        & (organ_raw_positive > 0)
                        & (organ_raw_positive <= getattr(evaluator, 'tiny_mask_pixel_threshold', 10))
                    )
                pixel_by_organ[organ] = _safe_metric(
                    roc_auc_score,
                    masks[idxes][pixel_keep].ravel(),
                    direction_maps[idxes][pixel_keep].ravel(),
                )

            for agg_name, _ in IMAGE_SCORE_AGGREGATIONS:
                combo_rows = []
                for organ in sorted(set(cls_names.tolist())):
                    idxes = cls_names == organ
                    image_metrics = self._image_metrics(labels[idxes], variants[agg_name][idxes])
                    metrics = {
                        'image_AUROC': image_metrics['sp_AUROC'],
                        'pixel_AUROC': pixel_by_organ[organ],
                    }
                    combo_rows.append((organ, metrics))
                combo_rows.append(('Avg', self._average_metric_dict([metrics for _, metrics in combo_rows])))
                rows_by_combo[(direction_key, direction_name, agg_name)] = combo_rows
        return rows_by_combo

    def _build_foreground_diagnostic(self, results, evaluator):
        masks = self._squeeze_maps(results['imgs_masks']).astype(int)
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        if 'image_scores' in results:
            original_image_scores = results['image_scores'].reshape(-1).astype(np.float32)
        else:
            original_image_scores = maps.reshape(maps.shape[0], -1).max(axis=1)

        foreground_maps = self._foreground_masks_for_maps(results, maps)
        if foreground_maps is None:
            return [], {}

        current_ratio = self._current_image_score_topk_ratio()
        eroded_foreground_maps = erode_foreground_masks(foreground_maps, self.foreground_erode_iters)
        masked_image_scores = compute_masked_image_scores(maps, foreground_maps, current_ratio)
        eroded_image_scores = compute_region_masked_scores(maps, eroded_foreground_maps, current_ratio)
        topk_regions = _topk_region_fractions(maps, foreground_maps, eroded_foreground_maps, current_ratio)
        masked_maps = mask_anomaly_maps_to_foreground(maps, foreground_maps, self.foreground_background_value)
        eroded_masked_maps = mask_anomaly_maps_to_foreground(maps, eroded_foreground_maps, self.foreground_background_value)

        maps_for_pixel, _ = _resize_maps_to_masks(maps, masks)
        masked_maps_for_pixel, _ = _resize_maps_to_masks(masked_maps, masks)
        eroded_masked_maps_for_pixel, _ = _resize_maps_to_masks(eroded_masked_maps, masks)
        foreground_for_pixel = self._foreground_masks_for_masks(results, masks)
        eroded_foreground_for_pixel = erode_foreground_masks(foreground_for_pixel, self.foreground_erode_iters)
        rows = []
        metric_dicts = []
        for organ in sorted(set(cls_names.tolist())):
            idxes = cls_names == organ
            original_image_auroc = _safe_metric(roc_auc_score, labels[idxes], original_image_scores[idxes])
            foreground_image_auroc = _safe_metric(roc_auc_score, labels[idxes], masked_image_scores[idxes])
            eroded_image_auroc = _safe_metric(roc_auc_score, labels[idxes], eroded_image_scores[idxes])
            full_pixel_auroc = _safe_metric(roc_auc_score, masks[idxes].ravel(), maps_for_pixel[idxes].ravel())
            foreground_pixel_auroc = self._foreground_pixel_auroc(
                masks[idxes],
                masked_maps_for_pixel[idxes],
                foreground_for_pixel[idxes],
            )
            eroded_foreground_pixel_auroc = self._foreground_pixel_auroc(
                masks[idxes],
                eroded_masked_maps_for_pixel[idxes],
                eroded_foreground_for_pixel[idxes],
            )
            metrics = {
                'original_image_AUROC': original_image_auroc,
                'foreground_masked_image_AUROC': foreground_image_auroc,
                'image_AUROC_delta': foreground_image_auroc - original_image_auroc
                if np.isfinite(original_image_auroc) and np.isfinite(foreground_image_auroc)
                else np.nan,
                'foreground_eroded_image_AUROC': eroded_image_auroc,
                'foreground_eroded_image_AUROC_delta': eroded_image_auroc - original_image_auroc
                if np.isfinite(original_image_auroc) and np.isfinite(eroded_image_auroc)
                else np.nan,
                'full_image_pixel_AUROC': full_pixel_auroc,
                'foreground_only_pixel_AUROC': foreground_pixel_auroc,
                'foreground_eroded_pixel_AUROC': eroded_foreground_pixel_auroc,
                'mean_foreground_ratio': float(foreground_maps[idxes].reshape(idxes.sum(), -1).mean()),
                'mean_foreground_eroded_ratio': float(eroded_foreground_maps[idxes].reshape(idxes.sum(), -1).mean()),
                'mean_topk_background_fraction': float(topk_regions['background'][idxes].mean()),
                'mean_topk_foreground_edge_fraction': float(topk_regions['foreground_edge'][idxes].mean()),
                'mean_topk_foreground_interior_fraction': float(topk_regions['foreground_interior'][idxes].mean()),
            }
            metric_dicts.append(metrics)
            rows.append({
                'class_name': organ,
                'image_score': self._current_image_score_description(current_ratio),
                'foreground_threshold': self.foreground_threshold,
                'background_fill': self.foreground_background_value,
                **{key: _safe_float(value) for key, value in metrics.items()},
            })
        avg_metrics = self._average_metric_dict(metric_dicts)
        rows.append({
            'class_name': 'Avg',
            'image_score': self._current_image_score_description(current_ratio),
            'foreground_threshold': self.foreground_threshold,
            'background_fill': self.foreground_background_value,
            **{key: _safe_float(value) for key, value in avg_metrics.items()},
        })
        summary = {
            'rows': rows,
            'avg': avg_metrics,
            'has_foreground_masks': True,
        }
        return rows, summary

    def _build_foreground_score_sweep(self, results):
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        image_scores = results['image_scores'].reshape(-1).astype(np.float32) if 'image_scores' in results else None
        foreground_maps = self._foreground_masks_for_maps(results, maps)
        if foreground_maps is None:
            return [], {}
        eroded_foreground_maps = erode_foreground_masks(foreground_maps, self.foreground_erode_iters)
        score_variants = self._foreground_score_variants(maps, image_scores, foreground_maps, eroded_foreground_maps)
        rows = []
        summary_rows = []
        for mode in self.score_modes:
            if mode not in score_variants:
                continue
            mode_scores = score_variants[mode]
            mode_rows = []
            for organ in sorted(set(cls_names.tolist())):
                idxes = cls_names == organ
                image_metrics = self._image_metrics(labels[idxes], mode_scores[idxes])
                metrics = {
                    'image_AUROC': image_metrics['sp_AUROC'],
                    'image_AP': image_metrics['sp_AP'],
                    'image_F1': image_metrics['sp_F1'],
                    'normal_mean': float(mode_scores[idxes][labels[idxes] == 0].mean()) if np.any(labels[idxes] == 0) else np.nan,
                    'abnormal_mean': float(mode_scores[idxes][labels[idxes] == 1].mean()) if np.any(labels[idxes] == 1) else np.nan,
                }
                mode_rows.append(metrics)
                row = {
                    'score_mode': mode,
                    'score_description': self._score_mode_description(mode),
                    'class_name': organ,
                    **{key: _safe_float(value) for key, value in metrics.items()},
                }
                rows.append(row)
                summary_rows.append(row)
            avg_metrics = self._average_metric_dict(mode_rows)
            rows.append({
                'score_mode': mode,
                'score_description': self._score_mode_description(mode),
                'class_name': 'Avg',
                **{key: _safe_float(value) for key, value in avg_metrics.items()},
            })
        summary = {
            'rows': rows,
            'avg_rows': [row for row in rows if row['class_name'] == 'Avg'],
            'has_foreground_score_sweep': True,
        }
        return rows, summary

    def _build_false_positive_region_diagnostic(self, results):
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        img_paths = self._normalize_paths(results.get('img_paths'), len(labels))
        foreground_maps = self._foreground_masks_for_maps(results, maps)
        if foreground_maps is None:
            return [], {}
        eroded_foreground_maps = erode_foreground_masks(foreground_maps, self.foreground_erode_iters)
        current_ratio = self._current_image_score_topk_ratio()
        topk_regions = _topk_region_fractions(maps, foreground_maps, eroded_foreground_maps, current_ratio)
        image_scores = results['image_scores'].reshape(-1).astype(np.float32) if 'image_scores' in results else _score_model_default(maps, current_ratio)
        rows = []
        for idx in range(len(labels)):
            rows.append({
                'row_type': 'sample',
                'image_path': img_paths[idx],
                'class_name': cls_names[idx],
                'image_label': int(labels[idx]),
                'label_name': 'abnormal' if int(labels[idx]) == 1 else 'normal',
                'n_samples': 1,
                'score_mode': 'model_top1',
                'score': _safe_float(image_scores[idx]),
                'topk_ratio': _safe_float(current_ratio if current_ratio is not None else 0.0),
                'topk_background_fraction': _safe_float(topk_regions['background'][idx]),
                'topk_foreground_edge_fraction': _safe_float(topk_regions['foreground_edge'][idx]),
                'topk_foreground_interior_fraction': _safe_float(topk_regions['foreground_interior'][idx]),
                'foreground_ratio': _safe_float(foreground_maps[idx].mean()),
                'foreground_eroded_ratio': _safe_float(eroded_foreground_maps[idx].mean()),
            })

        for organ in sorted(set(cls_names.tolist())):
            organ_idx = cls_names == organ
            for label_value, label_name in [(0, 'normal'), (1, 'abnormal')]:
                idxes = organ_idx & (labels == label_value)
                if not idxes.any():
                    continue
                rows.append({
                    'row_type': f'{label_name}_summary',
                    'image_path': '',
                    'class_name': organ,
                    'image_label': label_value,
                    'label_name': label_name,
                    'n_samples': int(idxes.sum()),
                    'score_mode': 'model_top1',
                    'score': _safe_float(image_scores[idxes].mean()),
                    'topk_ratio': _safe_float(current_ratio if current_ratio is not None else 0.0),
                    'topk_background_fraction': _safe_float(topk_regions['background'][idxes].mean()),
                    'topk_foreground_edge_fraction': _safe_float(topk_regions['foreground_edge'][idxes].mean()),
                    'topk_foreground_interior_fraction': _safe_float(topk_regions['foreground_interior'][idxes].mean()),
                    'foreground_ratio': _safe_float(foreground_maps[idxes].mean()),
                    'foreground_eroded_ratio': _safe_float(eroded_foreground_maps[idxes].mean()),
                })
        summary = {
            'rows': rows,
            'normal_summary_rows': [row for row in rows if row['row_type'] == 'normal_summary'],
            'has_false_positive_regions': True,
        }
        return rows, summary

    def _foreground_score_variants(self, maps, image_scores, foreground_maps, eroded_foreground_maps):
        current_ratio = self._current_image_score_topk_ratio()
        variants = {
            'model_top1': image_scores if image_scores is not None else _score_model_default(maps, current_ratio),
            'fg_top1': compute_region_masked_scores(maps, foreground_maps, 0.01),
            'fg_top5': compute_region_masked_scores(maps, foreground_maps, 0.05),
            'fg_top10': compute_region_masked_scores(maps, foreground_maps, 0.10),
            'fg_mean': compute_region_masked_scores(maps, foreground_maps, 'mean'),
            'fg_eroded_top1': compute_region_masked_scores(maps, eroded_foreground_maps, 0.01),
        }
        return variants

    def _score_mode_description(self, mode):
        descriptions = {
            'model_top1': 'model image_score, whole image top 1% mean',
            'fg_top1': 'foreground-only top 1% mean',
            'fg_eroded_top1': f'eroded foreground top 1% mean, erosion={self.foreground_erode_iters}',
            'fg_top5': 'foreground-only top 5% mean',
            'fg_top10': 'foreground-only top 10% mean',
            'fg_mean': 'foreground-only mean',
        }
        return descriptions.get(mode, mode)

    def _foreground_pixel_auroc(self, masks, maps, foreground_masks):
        y_true, y_score = [], []
        for gt, amap, fg in zip(masks, maps, foreground_masks):
            valid = np.asarray(fg).astype(bool)
            if not valid.any():
                valid = np.ones_like(gt, dtype=bool)
            y_true.append(gt[valid].reshape(-1))
            y_score.append(amap[valid].reshape(-1))
        if not y_true:
            return np.nan
        return _safe_metric(roc_auc_score, np.concatenate(y_true), np.concatenate(y_score))

    def _pixel_metrics(self, masks, maps, metric_names, evaluator):
        out = {}
        maps_norm = _normalize_map(maps)
        gt = masks.astype(bool)
        for metric in metric_names:
            if metric.startswith('mAUROC_px'):
                out[metric] = _safe_metric(roc_auc_score, masks.ravel(), maps.ravel())
            elif metric.startswith('mAP_px'):
                out[metric] = _safe_metric(average_precision_score, masks.ravel(), maps.ravel())
            elif metric.startswith('mF1_max_px'):
                out[metric] = self._best_pixel_overlap(gt, maps_norm, 'f1')
            elif metric.startswith('mAUPRO_px'):
                try:
                    out[metric] = float(evaluator.cal_pro_score(masks, maps, max_step=evaluator.max_step_aupro, mp=False))
                except Exception:
                    out[metric] = np.nan
        return out

    def _best_pixel_overlap(self, gt, maps_norm, mode):
        scores = []
        for threshold in np.arange(0.0, 1.0 + 1e-3, 0.05):
            pr = maps_norm > threshold
            intersect = np.logical_and(gt, pr).sum()
            pred = pr.sum()
            label = gt.sum()
            if mode == 'f1':
                precision = intersect / (pred + 1e-8)
                recall = intersect / (label + 1e-8)
                scores.append(2 * precision * recall / (precision + recall + 1e-8))
        return float(np.max(scores)) if scores else np.nan

    def _image_metrics(self, labels, scores):
        return {
            'sp_AUROC': _safe_metric(roc_auc_score, labels, scores),
            'sp_AP': _safe_metric(average_precision_score, labels, scores),
            'sp_F1': _best_f1(labels, scores),
        }

    def _distribution_stats(self, normal, abnormal):
        stats = {
            'normal_mean': float(np.mean(normal)) if normal.size else np.nan,
            'normal_std': float(np.std(normal)) if normal.size else np.nan,
            'abnormal_mean': float(np.mean(abnormal)) if abnormal.size else np.nan,
            'abnormal_std': float(np.std(abnormal)) if abnormal.size else np.nan,
            'overlap_frac': np.nan,
            'pairwise_abnormal_le_normal': np.nan,
        }
        if normal.size and abnormal.size:
            low = max(float(np.min(normal)), float(np.min(abnormal)))
            high = min(float(np.max(normal)), float(np.max(abnormal)))
            in_overlap = 0
            if high >= low:
                in_overlap = int(((normal >= low) & (normal <= high)).sum() + ((abnormal >= low) & (abnormal <= high)).sum())
            stats['overlap_frac'] = float(in_overlap / (normal.size + abnormal.size))
            stats['pairwise_abnormal_le_normal'] = float((abnormal[:, None] <= normal[None, :]).mean())
        return stats

    def _flatten_score_sweep_metrics(self, score_sweep_metrics):
        rows = []
        for (direction_key, direction_name, agg_name), organ_rows in score_sweep_metrics.items():
            for organ, metrics in organ_rows:
                rows.append({
                    'score_direction': direction_key,
                    'score_direction_formula': direction_name,
                    'aggregation': agg_name,
                    'class_name': organ,
                    'image_AUROC': _safe_float(metrics.get('image_AUROC', np.nan)),
                    'pixel_AUROC': _safe_float(metrics.get('pixel_AUROC', np.nan)),
                })
        return rows

    def _log_debug_summary(
        self,
        records_path,
        score_path,
        hist_path,
        sweep_path,
        foreground_path,
        foreground_score_path,
        false_positive_path,
        records,
        shape_summary,
        score_sweep_metrics,
        score_summary,
        foreground_summary,
        foreground_score_summary,
        false_positive_summary,
        evaluator,
    ):
        label1_mask0 = [r for r in records if int(r['image_label']) == 1 and int(r['mask_sum']) == 0]
        label0_mask1 = [r for r in records if int(r['image_label']) == 0 and int(r['mask_sum']) > 0]
        tiny_threshold = getattr(evaluator, 'tiny_mask_pixel_threshold', 10)
        skip_tiny_enabled = getattr(evaluator, 'skip_tiny_mask_for_pixel', False)
        tiny_pixel_skips = [
            r for r in records
            if skip_tiny_enabled
            and int(r['image_label']) == 1
            and str(r.get('raw_positive_pixels', '')) != ''
            and 0 < int(r['raw_positive_pixels']) <= tiny_threshold
        ]
        best_direction, best_agg, _ = self._best_score_combo(score_sweep_metrics)
        best_rows = self._combo_rows(score_sweep_metrics, best_direction, best_agg)
        worst_organ = self._worst_organ(best_rows) if best_rows else 'n/a'
        resize_issue = not shape_summary['shape_match']

        log_msg(
            self.logger,
            f'==> DebugEval files: {records_path} ; {score_path} ; {hist_path} ; {sweep_path} ; '
            f'{foreground_path} ; {foreground_score_path} ; {false_positive_path} ; vis_dir={self.vis_dir}'
        )
        log_msg(
            self.logger,
            '==> DebugEval label/mask mismatch: '
            f'image_label=1 & mask_sum=0: {len(label1_mask0)}, '
            f'image_label=0 & mask_sum>0: {len(label0_mask1)}'
        )
        if label1_mask0[:self.max_records_preview]:
            log_msg(self.logger, '==> DebugEval examples label=1 mask=0: ' + ', '.join(r['image_path'] for r in label1_mask0[:self.max_records_preview]))
            mismatch_rows = [
                {
                    'organ': r['organ_name'],
                    'image_path': r['image_path'],
                    'mask_path': r['mask_path'],
                    'image_label': r['image_label'],
                    'final_mask_sum': r['final_mask_sum'],
                    'mask_file_exists': r['mask_file_exists'],
                    'raw_mask_sum': r['raw_mask_sum'],
                    'raw_positive_pixels': r['raw_positive_pixels'],
                    'mask_status': r['mask_status'],
                }
                for r in label1_mask0[:self.max_records_preview]
            ]
            log_msg(
                self.logger,
                '==> DebugEval label=1 mask_sum=0 details\n'
                + tabulate.tabulate(mismatch_rows, headers='keys', tablefmt='pipe')
            )
        if label0_mask1[:self.max_records_preview]:
            log_msg(self.logger, '==> DebugEval examples label=0 mask>0: ' + ', '.join(r['image_path'] for r in label0_mask1[:self.max_records_preview]))
        if skip_tiny_enabled:
            log_msg(
                self.logger,
                f'==> DebugEval tiny-mask pixel skip: {len(tiny_pixel_skips)} samples '
                f'(raw_positive_pixels <= {tiny_threshold}); image metrics keep all samples'
            )
            if tiny_pixel_skips[:self.max_records_preview]:
                tiny_rows = [
                    {
                        'organ': r['organ_name'],
                        'image_path': r['image_path'],
                        'mask_path': r['mask_path'],
                        'image_label': r['image_label'],
                        'raw_positive_pixels': r['raw_positive_pixels'],
                        'final_mask_sum': r['final_mask_sum'],
                        'mask_status': r['mask_status'],
                    }
                    for r in tiny_pixel_skips[:self.max_records_preview]
                ]
                log_msg(
                    self.logger,
                    '==> DebugEval tiny-mask skipped pixel samples\n'
                    + tabulate.tabulate(tiny_rows, headers='keys', tablefmt='pipe')
                )

        log_msg(
            self.logger,
            '==> DebugEval shape check: '
            f'map={shape_summary["final_map_shape"]}, mask={shape_summary["final_mask_shape"]}, '
            f'match={shape_summary["shape_match"]}'
        )
        self._log_score_sweep_table(score_sweep_metrics)
        self._log_score_distribution(score_summary)
        self._log_foreground_diagnostic(foreground_summary)
        self._log_foreground_score_sweep(foreground_score_summary)
        self._log_false_positive_regions(false_positive_summary)
        recommendation = self._recommendation(label1_mask0, label0_mask1, resize_issue, score_sweep_metrics)
        current_topk = self._current_image_score_topk_ratio()
        foreground_avg = foreground_summary.get('avg', {}) if foreground_summary else {}
        topk_bg = foreground_avg.get('mean_topk_background_fraction', np.nan)
        fg_delta = foreground_avg.get('image_AUROC_delta', np.nan)
        summary = [
            ['label/mask inconsistent samples', f'label1_mask0={len(label1_mask0)}, label0_mask1={len(label0_mask1)}'],
            ['map/mask shape consistent', str(shape_summary['shape_match'])],
            ['current eval image_score', self._current_image_score_description(current_topk)],
            ['foreground top-k background fraction', f'{topk_bg:.4f}' if np.isfinite(topk_bg) else 'n/a'],
            ['foreground-masked image AUROC delta', f'{fg_delta * 100:.3f}' if np.isfinite(fg_delta) else 'n/a'],
            ['better score direction', best_direction or 'n/a'],
            ['best image score aggregation', best_agg or 'n/a'],
            ['weakest organ by image AUROC', worst_organ],
            ['resize/crop/mask alignment issue', 'possible' if resize_issue else 'not detected by shape check'],
            ['tiny masks skipped for pixel metrics', f'{len(tiny_pixel_skips)} (enabled={skip_tiny_enabled}, threshold={tiny_threshold})'],
            ['suggested next step', recommendation],
        ]
        log_msg(self.logger, '==> DebugEval Summary\n' + tabulate.tabulate(summary, headers=['check', 'result'], tablefmt='pipe'))

    def _log_foreground_diagnostic(self, foreground_summary):
        rows = foreground_summary.get('rows', []) if foreground_summary else []
        if not rows:
            log_msg(self.logger, '==> ForegroundMaskDiagnostic skipped: no foreground masks collected')
            return
        display_rows = []
        for row in rows:
            display_rows.append({
                'Name': row['class_name'],
                'orig_img_AUROC': row['original_image_AUROC'] * 100 if row['original_image_AUROC'] != '' else np.nan,
                'fg_img_AUROC': row['foreground_masked_image_AUROC'] * 100 if row['foreground_masked_image_AUROC'] != '' else np.nan,
                'delta': row['image_AUROC_delta'] * 100 if row['image_AUROC_delta'] != '' else np.nan,
                'fg_eroded_img_AUROC': row['foreground_eroded_image_AUROC'] * 100 if row.get('foreground_eroded_image_AUROC', '') != '' else np.nan,
                'eroded_delta': row['foreground_eroded_image_AUROC_delta'] * 100 if row.get('foreground_eroded_image_AUROC_delta', '') != '' else np.nan,
                'full_px_AUROC': row['full_image_pixel_AUROC'] * 100 if row['full_image_pixel_AUROC'] != '' else np.nan,
                'fg_px_AUROC': row['foreground_only_pixel_AUROC'] * 100 if row['foreground_only_pixel_AUROC'] != '' else np.nan,
                'fg_eroded_px_AUROC': row['foreground_eroded_pixel_AUROC'] * 100 if row.get('foreground_eroded_pixel_AUROC', '') != '' else np.nan,
                'fg_ratio': row['mean_foreground_ratio'],
                'fg_eroded_ratio': row.get('mean_foreground_eroded_ratio', np.nan),
                'topk_bg_frac': row['mean_topk_background_fraction'],
                'topk_edge_frac': row.get('mean_topk_foreground_edge_fraction', np.nan),
                'topk_inner_frac': row.get('mean_topk_foreground_interior_fraction', np.nan),
            })
        log_msg(
            self.logger,
            '==> ForegroundMaskDiagnostic (original vs foreground-masked)\n'
            + tabulate.tabulate(display_rows, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center')
        )

    def _log_foreground_score_sweep(self, foreground_score_summary):
        rows = foreground_score_summary.get('avg_rows', []) if foreground_score_summary else []
        if not rows:
            log_msg(self.logger, '==> ForegroundScoreSweep skipped: no foreground masks collected')
            return
        display_rows = []
        for row in rows:
            display_rows.append({
                'score_mode': row['score_mode'],
                'Name': row['class_name'],
                'image_AUROC': row['image_AUROC'] * 100 if row['image_AUROC'] != '' else np.nan,
                'normal_mean': row['normal_mean'],
                'abnormal_mean': row['abnormal_mean'],
            })
        log_msg(
            self.logger,
            '==> ForegroundScoreSweep (Avg rows)\n'
            + tabulate.tabulate(display_rows, headers='keys', tablefmt='pipe', floatfmt='.5f', numalign='center')
        )

    def _log_false_positive_regions(self, false_positive_summary):
        rows = false_positive_summary.get('normal_summary_rows', []) if false_positive_summary else []
        if not rows:
            log_msg(self.logger, '==> FalsePositiveRegionDiagnostic skipped: no foreground masks collected')
            return
        display_rows = []
        for row in rows:
            display_rows.append({
                'Name': row['class_name'],
                'n_normal': row['n_samples'],
                'score_mean': row['score'],
                'topk_bg_frac': row['topk_background_fraction'],
                'topk_edge_frac': row['topk_foreground_edge_fraction'],
                'topk_inner_frac': row['topk_foreground_interior_fraction'],
            })
        log_msg(
            self.logger,
            '==> FalsePositiveRegionDiagnostic (normal samples, model top-k)\n'
            + tabulate.tabulate(display_rows, headers='keys', tablefmt='pipe', floatfmt='.5f', numalign='center')
        )

    def _log_score_sweep_table(self, score_sweep_metrics):
        rows = []
        for (direction_key, direction_name, agg_name), organ_rows in score_sweep_metrics.items():
            for organ, metrics in organ_rows:
                rows.append({
                    'direction': direction_key,
                    'formula': direction_name,
                    'aggregation': agg_name,
                    'Name': organ,
                    'image_AUROC': metrics['image_AUROC'] * 100 if np.isfinite(metrics['image_AUROC']) else np.nan,
                    'pixel_AUROC': metrics['pixel_AUROC'] * 100 if np.isfinite(metrics['pixel_AUROC']) else np.nan,
                })
        log_msg(
            self.logger,
            '==> DebugEval image-score sweep (direction x aggregation)\n'
            + tabulate.tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center')
        )

    def _log_score_distribution(self, score_summary):
        rows = []
        for direction, organs in score_summary.items():
            for organ, variants in organs.items():
                for name, stats in variants.items():
                    rows.append({
                        'direction': direction,
                        'organ': organ,
                        'aggregation': name,
                        'normal_mean': stats['normal_mean'],
                        'normal_std': stats['normal_std'],
                        'abnormal_mean': stats['abnormal_mean'],
                        'abnormal_std': stats['abnormal_std'],
                        'overlap_frac': stats['overlap_frac'],
                        'P(abn<=norm)': stats['pairwise_abnormal_le_normal'],
                    })
        log_msg(
            self.logger,
            '==> DebugEval score distribution\n'
            + tabulate.tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.5f', numalign='center')
        )

    def _best_score_combo(self, score_sweep_metrics):
        best_direction, best_agg, best_score = None, None, -np.inf
        for (direction_key, _, agg_name), rows in score_sweep_metrics.items():
            avg_metrics = dict(rows[-1][1])
            score = avg_metrics.get('image_AUROC', np.nan)
            if np.isfinite(score) and score > best_score:
                best_direction, best_agg, best_score = direction_key, agg_name, score
        return best_direction, best_agg, best_score

    def _combo_rows(self, score_sweep_metrics, direction, aggregation):
        for (direction_key, _, agg_name), rows in score_sweep_metrics.items():
            if direction_key == direction and agg_name == aggregation:
                return rows
        return []

    def _combo_avg(self, score_sweep_metrics, direction, aggregation):
        rows = self._combo_rows(score_sweep_metrics, direction, aggregation)
        if not rows:
            return {}
        return dict(rows[-1][1])

    def _worst_organ(self, rows):
        organ_rows = [(organ, metrics) for organ, metrics in rows if organ != 'Avg']
        finite = [(organ, metrics['image_AUROC']) for organ, metrics in organ_rows if np.isfinite(metrics.get('image_AUROC', np.nan))]
        if not finite:
            return 'n/a'
        return min(finite, key=lambda item: item[1])[0]

    def _recommendation(self, label1_mask0, label0_mask1, resize_issue, score_sweep_metrics):
        if label1_mask0 or label0_mask1:
            return 'fix labels/masks before changing model'
        if resize_issue:
            return 'fix evaluation resize/shape alignment'
        best_direction, best_agg, _ = self._best_score_combo(score_sweep_metrics)
        old_avg = self._combo_avg(score_sweep_metrics, 'old', best_agg).get('image_AUROC', np.nan)
        reverse_avg = self._combo_avg(score_sweep_metrics, 'reverse', best_agg).get('image_AUROC', np.nan)
        if best_direction == 'reverse' and np.isfinite(reverse_avg) and (not np.isfinite(old_avg) or reverse_avg > old_avg + 0.05):
            return 'fix score direction in evaluation'
        current_agg = self._current_image_score_aggregation()
        if best_agg and current_agg and best_agg != current_agg:
            return f'fix image score aggregation first ({best_agg})'
        return 'evaluation looks consistent; investigate model/feature signal next'

    def _current_image_score_topk_ratio(self):
        model_cfg = getattr(self.cfg, 'model', None)
        kwargs = getattr(model_cfg, 'kwargs', {}) if model_cfg is not None else {}
        if not isinstance(kwargs, dict):
            return 0.01
        return kwargs.get('image_score_topk_ratio', 0.01)

    def _current_image_score_aggregation(self):
        ratio = self._current_image_score_topk_ratio()
        if ratio is None:
            return 'max'
        ratio = float(ratio)
        if abs(ratio - 0.01) < 1e-9:
            return 'top1%'
        if abs(ratio - 0.05) < 1e-9:
            return 'top5%'
        if abs(ratio - 0.10) < 1e-9:
            return 'top10%'
        return f'top{ratio * 100:g}%'

    def _current_image_score_description(self, ratio):
        if ratio is None:
            return 'old direction, max over anomaly_map'
        return f'old direction, top {float(ratio) * 100:g}% mean over anomaly_map'

    def _percentile_bounds_from_results(self, results):
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        low = float(np.percentile(maps, self.vis_percentile_low))
        high = float(np.percentile(maps, self.vis_percentile_high))
        if high <= low:
            low = float(maps.min())
            high = float(maps.max())
        return low, high

    def _heatmap(self, anomaly_map, norm_mode='minmax'):
        if norm_mode == 'percentile':
            bounds = self._vis_percentile_bounds
            if bounds is None:
                bounds = (
                    float(np.percentile(anomaly_map, self.vis_percentile_low)),
                    float(np.percentile(anomaly_map, self.vis_percentile_high)),
                )
            norm = _normalize_map_with_bounds(anomaly_map, bounds[0], bounds[1])
        else:
            norm = _normalize_map(anomaly_map)
        return (cm.jet(norm)[..., :3] * 255).astype(np.uint8)

    def _save_sample_visualization(self, out_dir, sample, sample_idx):
        img = self._denormalize_image(sample['img'])
        mask = np.asarray(sample['mask'])
        amap = np.asarray(sample['amap'], dtype=np.float32)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if amap.shape != mask.shape:
            amap, _ = _resize_maps_to_masks(amap[None, ...], mask[None, ...])
            amap = amap[0]
        foreground_mask = sample.get('foreground_mask')
        if foreground_mask is not None:
            foreground_mask = np.asarray(foreground_mask).astype(np.uint8)
            if foreground_mask.shape != amap.shape:
                foreground_mask, _ = _resize_binary_masks_to_shape(foreground_mask[None, ...], amap.shape)
                foreground_mask = foreground_mask[0]
        heat = self._heatmap(amap, 'minmax')
        mask_img = (mask > 0).astype(np.uint8) * 255
        overlay_heat = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
        boundary = self._mask_boundary(mask > 0)
        overlay_boundary = img.copy()
        overlay_boundary[boundary] = np.array([255, 40, 40], dtype=np.uint8)

        panels = [
            img,
            np.repeat(mask_img[..., None], 3, axis=2),
            heat,
            overlay_heat,
            overlay_boundary,
        ]
        if foreground_mask is not None:
            masked_map = mask_anomaly_maps_to_foreground(
                amap[None, ...],
                foreground_mask[None, ...],
                self.foreground_background_value,
            )[0]
            masked_heat = self._heatmap(masked_map, 'minmax')
            foreground_img = (foreground_mask > 0).astype(np.uint8) * 255
            masked_overlay = (0.55 * img + 0.45 * masked_heat).clip(0, 255).astype(np.uint8)
            panels.extend([
                np.repeat(foreground_img[..., None], 3, axis=2),
                masked_heat,
                masked_overlay,
            ])
            if self.vis_norm in ('both', 'percentile'):
                percentile_heat = self._heatmap(amap, 'percentile')
                percentile_overlay = (0.55 * img + 0.45 * percentile_heat).clip(0, 255).astype(np.uint8)
                eroded_mask = erode_foreground_masks(
                    foreground_mask[None, ...],
                    self.foreground_erode_iters,
                )[0]
                eroded_map = mask_anomaly_maps_to_foreground(
                    amap[None, ...],
                    eroded_mask[None, ...],
                    self.foreground_background_value,
                )[0]
                eroded_heat = self._heatmap(eroded_map, 'percentile')
                eroded_overlay = (0.55 * img + 0.45 * eroded_heat).clip(0, 255).astype(np.uint8)
                eroded_img = (eroded_mask > 0).astype(np.uint8) * 255
                panels.extend([
                    percentile_heat,
                    percentile_overlay,
                    np.repeat(eroded_img[..., None], 3, axis=2),
                    eroded_heat,
                    eroded_overlay,
                ])
        elif self.vis_norm in ('both', 'percentile'):
            percentile_heat = self._heatmap(amap, 'percentile')
            percentile_overlay = (0.55 * img + 0.45 * percentile_heat).clip(0, 255).astype(np.uint8)
            panels.extend([percentile_heat, percentile_overlay])

        panel = np.concatenate(panels, axis=1)

        mask_sum = int(mask.sum())
        base = (
            f'rank{self.rank}_{sample_idx:03d}_'
            f'{self._safe_name(sample["organ"])}_label{sample["label"]}_'
            f'mask{mask_sum}_score{sample["score"]:.6f}.png'
        )
        Image.fromarray(panel).save(os.path.join(out_dir, base))

    def _denormalize_image(self, img_chw):
        img = np.asarray(img_chw, dtype=np.float32).transpose(1, 2, 0)
        img = img * IMAGENET_STD[None, None, :] + IMAGENET_MEAN[None, None, :]
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    def _mask_boundary(self, mask):
        if mask.sum() == 0:
            return np.zeros_like(mask, dtype=bool)
        return np.logical_xor(mask, binary_erosion(mask))

    def _write_csv(self, path, rows):
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _average_metric_dict(self, metric_dicts):
        if not metric_dicts:
            return {}
        keys = metric_dicts[0].keys()
        avg = {}
        for key in keys:
            vals = np.array([metrics[key] for metrics in metric_dicts], dtype=float)
            avg[key] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        return avg

    def _normalize_paths(self, img_paths, length, fill_value=None):
        fill_value = 'unknown_{}' if fill_value is None else str(fill_value)
        if img_paths is None:
            return [fill_value.format(idx) if '{}' in fill_value else fill_value for idx in range(length)]
        if isinstance(img_paths, np.ndarray):
            paths = img_paths.reshape(-1).tolist()
        elif isinstance(img_paths, (list, tuple)):
            paths = list(img_paths)
        else:
            paths = [img_paths]
        paths = [str(path) for path in paths]
        if len(paths) < length:
            paths.extend([fill_value.format(idx) if '{}' in fill_value else fill_value for idx in range(len(paths), length)])
        return paths[:length]

    def _load_meta_lookup(self):
        lookup = {}
        for root in self._candidate_data_roots():
            meta_path = os.path.join(root, 'meta.json')
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
            except Exception:
                continue
            for split_items in meta.values():
                if not isinstance(split_items, dict):
                    continue
                for items in split_items.values():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        img_path = str(item.get('img_path', ''))
                        if img_path:
                            lookup[img_path] = item
                            lookup[img_path.replace('\\', '/')] = item
                            lookup[img_path.replace('/', '\\')] = item
        return lookup

    def _candidate_data_roots(self):
        roots = []
        data_test = getattr(self.cfg, 'data_test', None)
        if data_test is not None and getattr(data_test, 'root', None):
            roots.append(data_test.root)
        if getattr(self.cfg.data, 'root', None):
            roots.append(self.cfg.data.root)
        return [str(root) for root in roots]

    def _inspect_raw_mask(self, mask_path):
        if not mask_path:
            return {'exists': False, 'raw_mask_sum': '', 'raw_positive_pixels': ''}
        resolved = self._resolve_mask_path(mask_path)
        if resolved is None:
            return {'exists': False, 'raw_mask_sum': '', 'raw_positive_pixels': ''}
        try:
            with Image.open(resolved) as img:
                arr = np.asarray(img.convert('L'))
            return {'exists': True, 'raw_mask_sum': int(arr.sum()), 'raw_positive_pixels': int((arr > 0).sum())}
        except Exception:
            return {'exists': True, 'raw_mask_sum': '', 'raw_positive_pixels': ''}

    def _mask_status(self, image_label, mask_sum, mask_path, mask_info):
        if image_label == 0 and mask_sum > 0:
            return 'normal_has_mask_after_transform'
        if image_label == 1 and mask_sum > 0:
            return 'ok'
        if image_label == 1 and mask_sum == 0:
            if not mask_path:
                return 'missing_mask_path'
            if not mask_info['exists']:
                return 'mask_file_not_found'
            raw_positive_pixels = mask_info['raw_positive_pixels']
            if raw_positive_pixels == '':
                return 'mask_file_unreadable'
            if int(raw_positive_pixels) == 0:
                return 'raw_mask_all_black'
            return 'raw_mask_nonzero_but_empty_after_transform'
        return 'ok'

    def _resolve_mask_path(self, mask_path):
        if not mask_path:
            return None
        candidates = [mask_path]
        for root in self._candidate_data_roots():
            candidates.append(os.path.join(root, mask_path))
        for candidate in candidates:
            candidate = str(candidate).replace('/', os.sep)
            if os.path.exists(candidate):
                return candidate
        return None

    def _read_original_shape(self, image_path):
        resolved = self._resolve_image_path(image_path)
        if resolved is None:
            return None
        try:
            with Image.open(resolved) as img:
                width, height = img.size
            return (height, width)
        except Exception:
            return None

    def _resolve_image_path(self, image_path):
        if not image_path or str(image_path).startswith('unknown_'):
            return None
        candidates = [image_path]
        data_root = getattr(self.cfg.data, 'root', None)
        data_test_root = getattr(getattr(self.cfg, 'data_test', None), 'root', None)
        for root in [data_test_root, data_root]:
            if root:
                candidates.append(os.path.join(root, image_path))
        for candidate in candidates:
            candidate = str(candidate).replace('/', os.sep)
            if os.path.exists(candidate):
                return candidate
        return None

    def _squeeze_maps(self, value):
        arr = np.asarray(value)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr

    def _shape_to_str(self, shape):
        if shape is None:
            return ''
        arr = np.asarray(shape).astype(int).reshape(-1)
        return 'x'.join(str(int(x)) for x in arr)

    def _safe_name(self, value):
        return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
