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


def _score_topk_mean(anomaly_maps, ratio):
    flat = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    k = max(1, int(flat.shape[1] * ratio))
    part = np.partition(flat, flat.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


def compute_image_score_variants(anomaly_maps):
    flat = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    max_score = flat.max(axis=1)
    mean_score = flat.mean(axis=1)
    return {
        'max': max_score,
        'top-0.5%': _score_topk_mean(anomaly_maps, 0.005),
        'top-1%': _score_topk_mean(anomaly_maps, 0.01),
        'top-2%': _score_topk_mean(anomaly_maps, 0.02),
        'top-5%': _score_topk_mean(anomaly_maps, 0.05),
        'mean': mean_score,
        '0.5max+0.5mean': 0.5 * max_score + 0.5 * mean_score,
    }


def _resize_maps_to_masks(anomaly_maps, masks):
    if anomaly_maps.shape[-2:] == masks.shape[-2:]:
        return anomaly_maps, False
    maps_t = torch.from_numpy(anomaly_maps).float().unsqueeze(1)
    maps_t = F.interpolate(maps_t, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    return maps_t.squeeze(1).numpy(), True


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
        base_dir = getattr(cfg, 'logdir', None) or getattr(cfg.trainer, 'checkpoint', 'runs')
        self.out_dir = os.path.join(base_dir, 'debug_eval')
        self.vis_dir = os.path.join(base_dir, 'debug_vis')
        self._rng = random.Random(self.seed + self.rank)
        self._seen = defaultdict(int)
        self._samples = defaultdict(list)

    def add_vis_batch(self, imgs, masks, anomaly_maps, image_scores, cls_names, labels, img_paths):
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

        direction_metrics = self._direction_metrics(results, evaluator)
        aggregation_metrics = self._aggregation_metrics(results)
        self._log_debug_summary(
            records_path,
            score_path,
            hist_path,
            records,
            shape_summary,
            direction_metrics,
            aggregation_metrics,
            score_summary,
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
            }
            records.append(record)

        summary = {
            'shape_match': not shape_mismatch,
            'final_map_shape': tuple(final_map_shape),
            'final_mask_shape': tuple(final_mask_shape),
            'resized_for_debug_metrics': shape_mismatch,
        }
        return records, summary

    def _build_score_distribution(self, results):
        masks = self._squeeze_maps(results['imgs_masks']).astype(int)
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        img_paths = self._normalize_paths(results.get('img_paths'), len(labels))
        variants = compute_image_score_variants(maps)
        rows = []
        for idx in range(len(labels)):
            rows.append({
                'organ': cls_names[idx],
                'image_path': img_paths[idx],
                'image_label': int(labels[idx]),
                'image_score_max': _safe_float(variants['max'][idx]),
                'image_score_top1': _safe_float(variants['top-1%'][idx]),
                'image_score_mean': _safe_float(variants['mean'][idx]),
                'mask_sum': int(masks[idx].sum()),
            })

        summary = {}
        for organ in sorted(set(cls_names.tolist())):
            idxes = cls_names == organ
            organ_summary = {}
            for name in ['max', 'top-1%', 'mean']:
                scores = variants[name][idxes]
                organ_labels = labels[idxes]
                normal = scores[organ_labels == 0]
                abnormal = scores[organ_labels == 1]
                organ_summary[name] = self._distribution_stats(normal, abnormal)
            summary[organ] = organ_summary
        return rows, summary

    def _build_score_histogram(self, score_rows, bins=30):
        if not score_rows:
            return []
        hist_rows = []
        organs = sorted(set(row['organ'] for row in score_rows))
        score_keys = ['image_score_max', 'image_score_top1', 'image_score_mean']
        for organ in organs:
            organ_rows = [row for row in score_rows if row['organ'] == organ]
            for score_key in score_keys:
                values = np.array([float(row[score_key]) for row in organ_rows], dtype=np.float32)
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
                        [float(row[score_key]) for row in organ_rows if int(row['image_label']) == label_value],
                        dtype=np.float32,
                    )
                    counts, _ = np.histogram(label_values, bins=edges)
                    for bin_idx, count in enumerate(counts):
                        hist_rows.append({
                            'organ': organ,
                            'score': score_key,
                            'label': label_name,
                            'bin_left': float(edges[bin_idx]),
                            'bin_right': float(edges[bin_idx + 1]),
                            'count': int(count),
                        })
        return hist_rows

    def _direction_metrics(self, results, evaluator):
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
        metric_names = [
            metric for metric in evaluator.metrics
            if metric.startswith(('mAUROC_sp', 'mAP_sp', 'mF1_max_sp', 'mAUROC_px', 'mAP_px', 'mF1_max_px', 'mAUPRO_px'))
        ]
        metric_names = metric_names or ['mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max', 'mAUROC_px', 'mAP_px', 'mF1_max_px']

        output = {}
        for direction, direction_maps in [('normal', maps), ('reversed', -maps)]:
            direction_results = []
            debug_results = {
                'imgs_masks': masks,
                'anomaly_maps': direction_maps,
                'cls_names': cls_names,
                'anomalys': labels,
            }
            for organ in sorted(set(cls_names.tolist())):
                idxes = cls_names == organ
                organ_raw_positive = raw_positive_pixels[idxes]
                organ_labels = labels[idxes]
                pixel_keep = np.ones(organ_labels.shape[0], dtype=np.bool_)
                if getattr(evaluator, 'skip_tiny_mask_for_pixel', False):
                    pixel_keep = ~(
                        (organ_labels == 1)
                        & (organ_raw_positive > 0)
                        & (organ_raw_positive <= getattr(evaluator, 'tiny_mask_pixel_threshold', 10))
                    )
                organ_metrics = self._pixel_metrics(masks[idxes][pixel_keep], direction_maps[idxes][pixel_keep], metric_names, evaluator)
                organ_metrics.update(self._image_metrics(labels[idxes], direction_maps[idxes].reshape(idxes.sum(), -1).max(axis=1)))
                direction_results.append((organ, organ_metrics))
            avg = self._average_metric_dict([metrics for _, metrics in direction_results])
            direction_results.append(('Avg', avg))
            output[direction] = direction_results
        return output

    def _aggregation_metrics(self, results):
        maps = self._squeeze_maps(results['anomaly_maps']).astype(np.float32)
        labels = results['anomalys'].astype(int).reshape(-1)
        cls_names = results['cls_names'].astype(str)
        variants = compute_image_score_variants(maps)
        rows_by_variant = {}
        for variant_name, scores in variants.items():
            rows = []
            for organ in sorted(set(cls_names.tolist())):
                idxes = cls_names == organ
                metrics = self._image_metrics(labels[idxes], scores[idxes])
                rows.append((organ, metrics))
            rows.append(('Avg', self._average_metric_dict([metrics for _, metrics in rows])))
            rows_by_variant[variant_name] = rows
        return rows_by_variant

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

    def _log_debug_summary(self, records_path, score_path, hist_path, records, shape_summary, direction_metrics, aggregation_metrics, score_summary, evaluator):
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
        best_direction = self._best_direction(direction_metrics)
        best_agg = self._best_aggregation(aggregation_metrics)
        worst_organ = self._worst_organ(aggregation_metrics.get(best_agg, [])) if best_agg else 'n/a'
        resize_issue = not shape_summary['shape_match']

        log_msg(self.logger, f'==> DebugEval files: {records_path} ; {score_path} ; {hist_path} ; vis_dir={self.vis_dir}')
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
        self._log_direction_table(direction_metrics)
        self._log_aggregation_tables(aggregation_metrics)
        self._log_score_distribution(score_summary)
        recommendation = self._recommendation(label1_mask0, label0_mask1, resize_issue, direction_metrics, aggregation_metrics)
        summary = [
            ['label/mask inconsistent samples', f'label1_mask0={len(label1_mask0)}, label0_mask1={len(label0_mask1)}'],
            ['map/mask shape consistent', str(shape_summary['shape_match'])],
            ['better score direction', best_direction],
            ['best image score aggregation', best_agg or 'n/a'],
            ['weakest organ by image AUROC', worst_organ],
            ['resize/crop/mask alignment issue', 'possible' if resize_issue else 'not detected by shape check'],
            ['tiny masks skipped for pixel metrics', f'{len(tiny_pixel_skips)} (enabled={skip_tiny_enabled}, threshold={tiny_threshold})'],
            ['suggested next step', recommendation],
        ]
        log_msg(self.logger, '==> DebugEval Summary\n' + tabulate.tabulate(summary, headers=['check', 'result'], tablefmt='pipe'))

    def _log_direction_table(self, direction_metrics):
        for direction, rows in direction_metrics.items():
            table = {'Name': []}
            for organ, metrics in rows:
                table['Name'].append(organ)
                for name, value in metrics.items():
                    table.setdefault(name, []).append(value * 100 if np.isfinite(value) else np.nan)
            log_msg(
                self.logger,
                f'==> DebugEval {direction} direction metrics\n'
                + tabulate.tabulate(table, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center')
            )

    def _log_aggregation_tables(self, aggregation_metrics):
        rows = []
        for variant_name, organ_rows in aggregation_metrics.items():
            for organ, metrics in organ_rows:
                rows.append({
                    'score': variant_name,
                    'Name': organ,
                    'sp_AUROC': metrics['sp_AUROC'] * 100 if np.isfinite(metrics['sp_AUROC']) else np.nan,
                    'sp_AP': metrics['sp_AP'] * 100 if np.isfinite(metrics['sp_AP']) else np.nan,
                    'sp_F1': metrics['sp_F1'] * 100 if np.isfinite(metrics['sp_F1']) else np.nan,
                })
        log_msg(
            self.logger,
            '==> DebugEval image score aggregation metrics\n'
            + tabulate.tabulate(rows, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center')
        )

    def _log_score_distribution(self, score_summary):
        rows = []
        for organ, variants in score_summary.items():
            for name, stats in variants.items():
                rows.append({
                    'organ': organ,
                    'score': name,
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

    def _best_direction(self, direction_metrics):
        scores = {}
        for direction, rows in direction_metrics.items():
            avg = dict(rows[-1][1])
            scores[direction] = avg.get('mAUROC_px', np.nan) + avg.get('sp_AUROC', np.nan)
        if all(not np.isfinite(v) for v in scores.values()):
            return 'n/a'
        return max(scores, key=lambda key: scores[key] if np.isfinite(scores[key]) else -np.inf)

    def _best_aggregation(self, aggregation_metrics):
        best_name, best_score = None, -np.inf
        for variant_name, rows in aggregation_metrics.items():
            avg_metrics = dict(rows[-1][1])
            score = avg_metrics.get('sp_AUROC', np.nan)
            if np.isfinite(score) and score > best_score:
                best_name, best_score = variant_name, score
        return best_name

    def _worst_organ(self, rows):
        organ_rows = [(organ, metrics) for organ, metrics in rows if organ != 'Avg']
        finite = [(organ, metrics['sp_AUROC']) for organ, metrics in organ_rows if np.isfinite(metrics.get('sp_AUROC', np.nan))]
        if not finite:
            return 'n/a'
        return min(finite, key=lambda item: item[1])[0]

    def _recommendation(self, label1_mask0, label0_mask1, resize_issue, direction_metrics, aggregation_metrics):
        if label1_mask0 or label0_mask1:
            return 'fix labels/masks before changing model'
        if resize_issue:
            return 'fix evaluation resize/shape alignment'
        best_direction = self._best_direction(direction_metrics)
        if best_direction == 'reversed':
            return 'fix score direction in evaluation'
        best_agg = self._best_aggregation(aggregation_metrics)
        if best_agg and best_agg != 'top-1%':
            return f'fix image score aggregation first ({best_agg})'
        return 'evaluation looks consistent; investigate model/feature signal next'

    def _save_sample_visualization(self, out_dir, sample, sample_idx):
        img = self._denormalize_image(sample['img'])
        mask = np.asarray(sample['mask'])
        amap = np.asarray(sample['amap'], dtype=np.float32)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if amap.shape != mask.shape:
            amap, _ = _resize_maps_to_masks(amap[None, ...], mask[None, ...])
            amap = amap[0]
        amap_norm = _normalize_map(amap)
        heat = (cm.jet(amap_norm)[..., :3] * 255).astype(np.uint8)
        mask_img = (mask > 0).astype(np.uint8) * 255
        overlay_heat = (0.55 * img + 0.45 * heat).clip(0, 255).astype(np.uint8)
        boundary = self._mask_boundary(mask > 0)
        overlay_boundary = img.copy()
        overlay_boundary[boundary] = np.array([255, 40, 40], dtype=np.uint8)

        panel = np.concatenate([
            img,
            np.repeat(mask_img[..., None], 3, axis=2),
            heat,
            overlay_heat,
            overlay_boundary,
        ], axis=1)

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
