import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _squeeze_maps(array):
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _normalize(array):
    array = np.asarray(array, dtype=np.float32)
    min_v = float(np.min(array))
    max_v = float(np.max(array))
    if max_v <= min_v:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_v) / (max_v - min_v)


def _safe_name(path, index):
    stem = Path(str(path)).stem or f'image_{index:05d}'
    parent = Path(str(path)).parent.name
    name = f'{parent}_{stem}' if parent else stem
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in ['-', '_', '.'] else '_')
    return ''.join(keep)


def _read_rgb(path):
    image = Image.open(path).convert('RGB')
    return np.array(image)


def _resize_map(amap, shape):
    height, width = shape[:2]
    return cv2.resize(amap.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask, shape):
    height, width = shape[:2]
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def _heatmap(amap_norm):
    heat = (np.clip(amap_norm, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _draw_mask_contour(image_rgb, mask, color=(0, 255, 255)):
    out = image_rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness=2)
    return out


def _save_rgb(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8)).save(path)


def visualize(args):
    data = np.load(args.input, allow_pickle=True)
    maps = _squeeze_maps(data['anomaly_maps']).astype(np.float32)
    masks = _squeeze_maps(data['imgs_masks']).astype(np.uint8) if 'imgs_masks' in data else None
    img_paths = data['img_paths'].astype(str)
    cls_names = data['cls_names'].astype(str)
    labels = data['anomalys'].astype(int)
    image_scores = data['image_scores'].astype(float).reshape(-1) if 'image_scores' in data else maps.reshape(maps.shape[0], -1).max(axis=1)

    class_filter = {item.strip() for item in args.class_names.split(',') if item.strip()}
    indices = list(range(len(img_paths)))
    if class_filter:
        indices = [idx for idx in indices if cls_names[idx] in class_filter]
    if args.only_abnormal:
        indices = [idx for idx in indices if labels[idx] == 1]
    if args.sort_by_score:
        indices = sorted(indices, key=lambda idx: float(image_scores[idx]), reverse=True)
    if args.max_images > 0:
        indices = indices[:args.max_images]

    out_dir = Path(args.output_dir)
    rows = []
    for rank, idx in enumerate(indices):
        image_path = img_paths[idx]
        image = _read_rgb(image_path)
        amap = _resize_map(maps[idx], image.shape)
        amap_norm = _normalize(amap)
        heat = _heatmap(amap_norm)
        overlay = (args.alpha * heat + (1.0 - args.alpha) * image).astype(np.uint8)

        mask_sum = 0
        if masks is not None:
            mask = _resize_mask(masks[idx], image.shape) > 0
            mask_sum = int(mask.sum())
            overlay = _draw_mask_contour(overlay, mask.astype(np.uint8), color=(255, 255, 255))
            image_with_mask = _draw_mask_contour(image, mask.astype(np.uint8), color=(255, 255, 0))
        else:
            image_with_mask = image

        cls_dir = out_dir / cls_names[idx]
        base = f'{rank:05d}_{int(labels[idx])}_{float(image_scores[idx]):.4f}_{_safe_name(image_path, idx)}'
        overlay_path = cls_dir / f'{base}_overlay.png'
        heat_path = cls_dir / f'{base}_heatmap.png'
        image_path_out = cls_dir / f'{base}_image_mask.png'
        _save_rgb(overlay_path, overlay)
        _save_rgb(heat_path, heat)
        _save_rgb(image_path_out, image_with_mask)
        rows.append({
            'rank': rank,
            'index': idx,
            'class_name': cls_names[idx],
            'label': int(labels[idx]),
            'image_score': float(image_scores[idx]),
            'mask_sum': mask_sum,
            'source_image_path': image_path,
            'overlay_path': str(overlay_path),
            'heatmap_path': str(heat_path),
            'image_mask_path': str(image_path_out),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            'rank', 'index', 'class_name', 'label', 'image_score', 'mask_sum',
            'source_image_path', 'overlay_path', 'heatmap_path', 'image_mask_path',
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} visualizations to {out_dir}')
    print(f'Manifest: {manifest_path}')


def parse_args():
    parser = argparse.ArgumentParser('Visualize AnomalyCLIP exported anomaly maps')
    parser.add_argument('--input', required=True, help='Exported .npz from tools/export_anomalyclip_medical.py.')
    parser.add_argument('--output_dir', default='runs/anomalyclip_medical/visualizations')
    parser.add_argument('--class_names', default='', help='Optional comma-separated class filter.')
    parser.add_argument('--max_images', type=int, default=100, help='Use <=0 for all images.')
    parser.add_argument('--only_abnormal', action='store_true')
    parser.add_argument('--sort_by_score', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.45, help='Heatmap opacity in overlay.')
    return parser.parse_args()


if __name__ == '__main__':
    visualize(parse_args())
