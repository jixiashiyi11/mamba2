import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')


def _is_image(path):
    return path.suffix.lower() in IMG_EXTENSIONS


def _load_binary_mask(path):
    mask = Image.open(path).convert('L')
    return np.asarray(mask, dtype=np.uint8) > 0


def _component_label(mask):
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components = []
    label_id = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            label_id += 1
            stack = [(y, x)]
            labels[y, x] = label_id
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label_id
                        stack.append((ny, nx))
            components.append(np.asarray(coords, dtype=np.int32))
    return labels, components


def _bbox_from_coords(coords):
    ys = coords[:, 0]
    xs = coords[:, 1]
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _perimeter(mask):
    padded = np.pad(mask.astype(np.uint8), 1, mode='constant')
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    edge = center & ((up == 0) | (down == 0) | (left == 0) | (right == 0))
    return int(edge.sum())


def _eccentricity(coords):
    if coords.shape[0] < 3:
        return 0.0
    xy = coords.astype(np.float64)
    xy -= xy.mean(axis=0, keepdims=True)
    cov = xy.T @ xy / max(coords.shape[0], 1)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-8)
    return float(np.sqrt(np.clip(1.0 - eigvals[0] / eigvals[1], 0.0, 1.0)))


def _relative_position(lesion_coords, organ_mask):
    if organ_mask is None or not organ_mask.any():
        return None
    organ_coords = np.column_stack(np.where(organ_mask))
    y0, y1, x0, x1 = _bbox_from_coords(organ_coords)
    center_yx = lesion_coords.mean(axis=0)
    return {
        'relative_y': float((center_yx[0] - y0) / max(y1 - y0, 1)),
        'relative_x': float((center_yx[1] - x0) / max(x1 - x0, 1)),
    }


def _mask_stats(mask, mask_path, dataset_name, organ='', organ_mask=None, min_component_area=4):
    height, width = mask.shape
    labels, components = _component_label(mask)
    component_items = [
        (label_id, coords)
        for label_id, coords in enumerate(components, start=1)
        if coords.shape[0] >= min_component_area
    ]
    if not component_items:
        return None

    filtered = np.zeros_like(mask, dtype=bool)
    for label_id, _ in component_items:
        filtered[labels == label_id] = True
    all_coords = np.column_stack(np.where(filtered))
    y0, y1, x0, x1 = _bbox_from_coords(all_coords)
    area = int(filtered.sum())
    perimeter = _perimeter(filtered)
    bbox_h = max(y1 - y0, 1)
    bbox_w = max(x1 - x0, 1)
    compactness = float((perimeter * perimeter) / (4.0 * np.pi * max(area, 1)))
    row = {
        'dataset': dataset_name,
        'organ': organ,
        'mask_path': str(mask_path),
        'height': int(height),
        'width': int(width),
        'lesion_area': area,
        'area_ratio': float(area / max(height * width, 1)),
        'bbox_height': int(bbox_h),
        'bbox_width': int(bbox_w),
        'aspect_ratio': float(bbox_w / bbox_h),
        'eccentricity': _eccentricity(all_coords),
        'component_count': int(len(component_items)),
        'compactness': compactness,
        'perimeter_area_ratio': float(perimeter / max(area, 1)),
        'boundary_irregularity': float(perimeter / max(2.0 * np.sqrt(np.pi * max(area, 1)), 1e-6)),
    }
    rel = _relative_position(all_coords, organ_mask)
    if rel is not None:
        row.update(rel)
    return row


def _collect_from_meta(root):
    root = Path(root)
    meta_path = root / 'meta.json'
    if not meta_path.exists():
        raise RuntimeError(f'meta.json not found: {meta_path}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    samples = []
    for split_name, split in meta.items():
        for organ, items in split.items():
            for sample in items:
                if int(sample.get('anomaly', 0)) != 1:
                    continue
                mask_path = sample.get('mask_path', '')
                if not mask_path:
                    continue
                samples.append({
                    'mask_path': root / mask_path,
                    'organ': sample.get('cls_name', organ),
                    'organ_mask_path': root / sample['organ_mask_path'] if sample.get('organ_mask_path') else None,
                    'split': split_name,
                })
    return samples


def _collect_from_globs(mask_globs, organ=''):
    samples = []
    for pattern in mask_globs:
        for path in sorted(Path(match) for match in glob.glob(pattern, recursive=True)):
            if path.is_file() and _is_image(path):
                samples.append({'mask_path': path, 'organ': organ, 'organ_mask_path': None, 'split': ''})
    return samples


def _percentiles(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {f'p{q}': float(np.percentile(arr, q)) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}


def _summarize(rows):
    numeric_keys = [
        'area_ratio',
        'aspect_ratio',
        'eccentricity',
        'component_count',
        'compactness',
        'perimeter_area_ratio',
        'boundary_irregularity',
        'relative_y',
        'relative_x',
    ]
    summary = {'all': {'n': len(rows)}}
    for key in numeric_keys:
        values = [row[key] for row in rows if key in row]
        if values:
            summary['all'][key] = _percentiles(values)
    by_organ = defaultdict(list)
    for row in rows:
        by_organ[row.get('organ', '')].append(row)
    summary['by_organ'] = {}
    for organ, items in sorted(by_organ.items()):
        item_summary = {'n': len(items)}
        for key in numeric_keys:
            values = [row[key] for row in items if key in row]
            if values:
                item_summary[key] = _percentiles(values)
        summary['by_organ'][organ or 'unknown'] = item_summary
    return summary


def _distribution_columns(rows):
    keys = [
        'area_ratio',
        'aspect_ratio',
        'eccentricity',
        'component_count',
        'compactness',
        'perimeter_area_ratio',
        'boundary_irregularity',
        'relative_y',
        'relative_x',
    ]
    columns = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        if values:
            columns[key] = values
    return columns


def main():
    parser = argparse.ArgumentParser(description='Build scalar lesion morphology priors from auxiliary public lesion masks only.')
    parser.add_argument('--public-root', action='append', default=[], help='ADer-style public dataset root containing meta.json.')
    parser.add_argument('--mask-glob', action='append', default=[], help='Glob for public lesion mask files.')
    parser.add_argument('--dataset-name', default='public_auxiliary_lesion_masks')
    parser.add_argument('--organ', default='', help='Organ name used for --mask-glob inputs.')
    parser.add_argument('--output', default='assets/morphology_prior.json')
    parser.add_argument('--npz-output', default='')
    parser.add_argument('--min-mask-pixels', type=int, default=10)
    parser.add_argument('--min-component-area', type=int, default=4)
    parser.add_argument('--allow-target-test-masks', action='store_true', help='Safety override for analysis only. Do not use this for training priors.')
    args = parser.parse_args()

    if not args.allow_target_test_masks:
        print('[safety] Use only auxiliary public lesion masks. Do not pass target test-set anomaly masks.')

    samples = []
    for root in args.public_root:
        samples.extend(_collect_from_meta(root))
    if args.mask_glob:
        samples.extend(_collect_from_globs(args.mask_glob, organ=args.organ))
    if not samples:
        raise RuntimeError('No lesion masks found. Provide --public-root or --mask-glob.')

    rows = []
    skipped_empty = 0
    for sample in samples:
        mask_path = Path(sample['mask_path'])
        if not mask_path.exists():
            continue
        mask = _load_binary_mask(mask_path)
        if int(mask.sum()) < args.min_mask_pixels:
            skipped_empty += 1
            continue
        organ_mask = None
        organ_mask_path = sample.get('organ_mask_path')
        if organ_mask_path and Path(organ_mask_path).exists():
            organ_mask = _load_binary_mask(organ_mask_path)
            if organ_mask.shape != mask.shape:
                organ_mask = np.asarray(
                    Image.fromarray(organ_mask.astype(np.uint8) * 255).resize(mask.shape[::-1], Image.NEAREST),
                    dtype=np.uint8,
                ) > 0
        row = _mask_stats(mask, mask_path, args.dataset_name, sample.get('organ', ''), organ_mask, args.min_component_area)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError('No valid lesion masks remained after filtering.')

    prior = {
        'version': 1,
        'source_policy': 'auxiliary_public_lesion_masks_only',
        'dataset_name': args.dataset_name,
        'n_masks': len(rows),
        'skipped_empty_or_tiny': skipped_empty,
        'features': [
            'area_ratio',
            'aspect_ratio',
            'eccentricity',
            'component_count',
            'compactness',
            'perimeter_area_ratio',
            'boundary_irregularity',
            'relative_y',
            'relative_x',
        ],
        'summary': _summarize(rows),
        'samples': _distribution_columns(rows),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(prior, f, indent=2)
    print(f'Wrote morphology prior: {output}')
    print(f'Valid masks: {len(rows)}')
    print(f'Skipped empty/tiny masks: {skipped_empty}')

    if args.npz_output:
        npz_output = Path(args.npz_output)
        npz_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_output, **{key: np.asarray(value) for key, value in prior['samples'].items()})
        print(f'Wrote NPZ prior: {npz_output}')


if __name__ == '__main__':
    main()
