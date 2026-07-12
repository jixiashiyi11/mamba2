import argparse
import csv
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

if 'adeval' not in sys.modules:
    adeval_stub = types.ModuleType('adeval')

    class EvalAccumulatorCuda:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('adeval is not installed. This script runs Evaluator with use_adeval=False.')

    adeval_stub.EvalAccumulatorCuda = EvalAccumulatorCuda
    sys.modules['adeval'] = adeval_stub

from util.metric import Evaluator


METRICS = [
    'mAUROC_sp_max',
    'mAUROC_px',
    'mAP_px',
    'mF1_max_px',
    'mAUPRO_px',
]


def _squeeze_maps(array):
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _resize_maps_to_masks(maps, masks):
    if maps.shape[-2:] == masks.shape[-2:]:
        return maps
    maps_t = torch.from_numpy(maps).float().unsqueeze(1)
    maps_t = F.interpolate(maps_t, size=masks.shape[-2:], mode='bilinear', align_corners=False)
    return maps_t.squeeze(1).numpy()


def _resize_masks_to_masks(foreground_masks, masks):
    foreground_masks = _squeeze_maps(foreground_masks).astype(np.float32)
    if foreground_masks.shape[-2:] == masks.shape[-2:]:
        return foreground_masks.astype(bool)
    fg_t = torch.from_numpy(foreground_masks).float().unsqueeze(1)
    fg_t = F.interpolate(fg_t, size=masks.shape[-2:], mode='nearest')
    return fg_t.squeeze(1).numpy() > 0.5


def _safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def _foreground_pixel_auroc(masks, maps, foreground_masks):
    y_true = []
    y_score = []
    for gt, amap, fg in zip(masks, maps, foreground_masks):
        valid = np.asarray(fg).astype(bool)
        if not valid.any():
            valid = np.ones_like(gt, dtype=bool)
        y_true.append(gt[valid].reshape(-1))
        y_score.append(amap[valid].reshape(-1))
    if not y_true:
        return np.nan
    return _safe_roc_auc(np.concatenate(y_true), np.concatenate(y_score))


def _fmt(value):
    if value is None or not np.isfinite(value):
        return 'nan'
    return f'{value * 100:.2f}'


def _mean_finite(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def _print_table(rows):
    headers = [
        'class',
        'Image AUROC',
        'Full Pixel AUROC',
        'Foreground Pixel AUROC',
        'AUPRO',
        'Pixel AP',
        'F1-max',
    ]
    widths = [max(len(str(row.get(header, ''))) for row in rows + [dict(zip(headers, headers))]) for header in headers]
    print(' | '.join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print(' | '.join('-' * width for width in widths))
    for row in rows:
        print(' | '.join(str(row.get(header, '')).ljust(widths[idx]) for idx, header in enumerate(headers)))


def evaluate(args):
    data = np.load(args.input, allow_pickle=True)
    results = {
        'imgs_masks': data['imgs_masks'].astype(np.uint8),
        'anomaly_maps': data['anomaly_maps'].astype(np.float32),
        'image_scores': data['image_scores'].astype(np.float32),
        'cls_names': data['cls_names'].astype(str),
        'anomalys': data['anomalys'].astype(int),
    }
    for optional_key in ['img_paths', 'mask_paths', 'raw_positive_pixels', 'foreground_masks']:
        if optional_key in data:
            results[optional_key] = data[optional_key]

    evaluator = Evaluator(metrics=METRICS, max_step_aupro=args.max_step_aupro)
    cls_names = list(data['class_names'].astype(str)) if 'class_names' in data else sorted(set(results['cls_names'].tolist()))
    masks = _squeeze_maps(results['imgs_masks']).astype(np.uint8)
    maps = _resize_maps_to_masks(_squeeze_maps(results['anomaly_maps']).astype(np.float32), masks)
    if 'foreground_masks' in results:
        foreground_masks = _resize_masks_to_masks(results['foreground_masks'], masks)
    else:
        foreground_masks = np.ones_like(masks, dtype=bool)

    rows = []
    raw_rows = []
    for cls_name in cls_names:
        idxes = results['cls_names'] == cls_name
        if not idxes.any():
            continue
        metric_results = evaluator.run(results, cls_name, logger=None)
        fg_auroc = _foreground_pixel_auroc(masks[idxes], maps[idxes], foreground_masks[idxes])
        raw_row = {
            'class': cls_name,
            'Image AUROC': metric_results.get('mAUROC_sp_max', np.nan),
            'Full Pixel AUROC': metric_results.get('mAUROC_px', np.nan),
            'Foreground Pixel AUROC': fg_auroc,
            'AUPRO': metric_results.get('mAUPRO_px', np.nan),
            'Pixel AP': metric_results.get('mAP_px', np.nan),
            'F1-max': metric_results.get('mF1_max_px', np.nan),
        }
        raw_rows.append(raw_row)
        rows.append({key: (_fmt(value) if key != 'class' else value) for key, value in raw_row.items()})

    avg = {'class': 'Avg'}
    for key in ['Image AUROC', 'Full Pixel AUROC', 'Foreground Pixel AUROC', 'AUPRO', 'Pixel AP', 'F1-max']:
        avg[key] = _mean_finite([row[key] for row in raw_rows])
    rows.append({key: (_fmt(value) if key != 'class' else value) for key, value in avg.items()})
    raw_rows.append(avg)

    _print_table(rows)
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            for row in raw_rows:
                writer.writerow(row)
        print(f'Saved CSV to {csv_path}')


def parse_args():
    parser = argparse.ArgumentParser('Evaluate exported AnomalyCLIP .npz with project metrics')
    parser.add_argument('--input', required=True, help='Exported .npz from tools/export_anomalyclip_medical.py.')
    parser.add_argument('--csv', default='', help='Optional CSV output path.')
    parser.add_argument('--max_step_aupro', type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
