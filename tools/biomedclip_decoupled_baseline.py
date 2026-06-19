import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / 'tools'
for path in [PROJECT_ROOT, TOOLS_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from biomedclip_patch_text_baseline import (  # noqa: E402
    MedicalTestDataset,
    aggregate_score,
    build_prompts,
    collate_test,
    create_output_dir,
    encode_prompts,
    extract_patch_tokens,
    load_cfg,
    load_open_clip_model,
    resize_mask_to_grid,
    write_csv,
    write_json,
)


def safe_metric(fn, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(fn(y_true, y_score))


def f1_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.nanmax(f1))


def erode_binary(mask, iterations):
    mask = np.asarray(mask).astype(bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(mask, 1, mode='constant', constant_values=False)
        eroded = np.ones_like(mask, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        mask = eroded
    return mask


def foreground_mask_to_grid(image, grid_shape, threshold):
    image = image.convert('RGB').resize((grid_shape[1], grid_shape[0]))
    arr = np.asarray(image).astype(np.float32)
    return arr.max(axis=-1) > float(threshold)


def mask_to_low(score_map, keep_mask):
    score_map = np.asarray(score_map, dtype=np.float32)
    keep = np.asarray(keep_mask).astype(bool)
    if keep.any():
        low = float(score_map[keep].min()) - 1e-6
    else:
        low = float(score_map.min()) - 1e-6
    return np.where(keep, score_map, low).astype(np.float32)


def topk_mask(score_map, top_percent):
    flat = np.asarray(score_map, dtype=np.float32).reshape(-1)
    ratio = float(top_percent) / 100.0
    k = max(1, int(np.ceil(flat.size * ratio)))
    if k >= flat.size:
        selected = np.ones_like(flat, dtype=bool)
    else:
        indices = np.argpartition(flat, flat.size - k)[flat.size - k:]
        selected = np.zeros_like(flat, dtype=bool)
        selected[indices] = True
    return selected.reshape(score_map.shape)


def compute_topk_row(base, score_map, lesion_mask, foreground_mask, erode_iters, top_percent):
    score_map = np.asarray(score_map, dtype=np.float32)
    lesion = np.asarray(lesion_mask).astype(bool)
    foreground = np.asarray(foreground_mask).astype(bool)
    interior = erode_binary(foreground, erode_iters)
    edge = foreground & ~interior
    background = ~foreground
    selected = topk_mask(score_map, top_percent)
    selected_count = max(1, int(selected.sum()))
    lesion_count = int(lesion.sum())

    max_y, max_x = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    hit_count = int((selected & lesion).sum())

    row = dict(base)
    row.update({
        'top_percent': float(top_percent),
        'grid_h': int(score_map.shape[0]),
        'grid_w': int(score_map.shape[1]),
        'lesion_grid_pixels': lesion_count,
        'foreground_grid_pixels': int(foreground.sum()),
        'foreground_edge_grid_pixels': int(edge.sum()),
        'foreground_interior_grid_pixels': int(interior.sum()),
        'topk_pixels': int(selected.sum()),
        'max_y': int(max_y),
        'max_x': int(max_x),
        'max_point_in_lesion': int(bool(lesion[max_y, max_x])) if lesion_count > 0 else '',
        'max_point_in_background': int(bool(background[max_y, max_x])),
        'max_point_in_foreground_edge': int(bool(edge[max_y, max_x])),
        'max_point_in_foreground_interior': int(bool(interior[max_y, max_x])),
        'topk_lesion_fraction': hit_count / selected_count,
        'topk_background_fraction': int((selected & background).sum()) / selected_count,
        'topk_foreground_edge_fraction': int((selected & edge).sum()) / selected_count,
        'topk_foreground_interior_fraction': int((selected & interior).sum()) / selected_count,
        'lesion_coverage': (hit_count / lesion_count) if lesion_count > 0 else '',
    })
    return row


def mean_float(rows, key):
    values = []
    for row in rows:
        value = row.get(key, '')
        if value == '' or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else ''


def summarize_topk(records):
    groups = defaultdict(list)
    for row in records:
        label_group = 'abnormal' if int(row['label']) == 1 else 'normal'
        groups[(row['map_name'], row['class_name'], label_group, row['top_percent'])].append(row)

    fields = [
        'map_name',
        'class_name',
        'label_group',
        'top_percent',
        'n',
        'max_point_in_lesion_rate',
        'topk_lesion_fraction_mean',
        'lesion_coverage_mean',
        'topk_background_fraction_mean',
        'topk_foreground_edge_fraction_mean',
        'topk_foreground_interior_fraction_mean',
        'max_point_in_background_rate',
        'max_point_in_foreground_edge_rate',
        'max_point_in_foreground_interior_rate',
    ]
    rows = []
    for (map_name, class_name, label_group, top_percent), items in sorted(groups.items()):
        rows.append({
            'map_name': map_name,
            'class_name': class_name,
            'label_group': label_group,
            'top_percent': top_percent,
            'n': len(items),
            'max_point_in_lesion_rate': mean_float(items, 'max_point_in_lesion'),
            'topk_lesion_fraction_mean': mean_float(items, 'topk_lesion_fraction'),
            'lesion_coverage_mean': mean_float(items, 'lesion_coverage'),
            'topk_background_fraction_mean': mean_float(items, 'topk_background_fraction'),
            'topk_foreground_edge_fraction_mean': mean_float(items, 'topk_foreground_edge_fraction'),
            'topk_foreground_interior_fraction_mean': mean_float(items, 'topk_foreground_interior_fraction'),
            'max_point_in_background_rate': mean_float(items, 'max_point_in_background'),
            'max_point_in_foreground_edge_rate': mean_float(items, 'max_point_in_foreground_edge'),
            'max_point_in_foreground_interior_rate': mean_float(items, 'max_point_in_foreground_interior'),
        })

    avg_groups = defaultdict(list)
    for row in rows:
        avg_groups[(row['map_name'], row['label_group'], row['top_percent'])].append(row)
    for (map_name, label_group, top_percent), items in sorted(avg_groups.items()):
        rows.append({
            'map_name': map_name,
            'class_name': 'Avg',
            'label_group': label_group,
            'top_percent': top_percent,
            'n': sum(int(item['n']) for item in items),
            'max_point_in_lesion_rate': mean_float(items, 'max_point_in_lesion_rate'),
            'topk_lesion_fraction_mean': mean_float(items, 'topk_lesion_fraction_mean'),
            'lesion_coverage_mean': mean_float(items, 'lesion_coverage_mean'),
            'topk_background_fraction_mean': mean_float(items, 'topk_background_fraction_mean'),
            'topk_foreground_edge_fraction_mean': mean_float(items, 'topk_foreground_edge_fraction_mean'),
            'topk_foreground_interior_fraction_mean': mean_float(items, 'topk_foreground_interior_fraction_mean'),
            'max_point_in_background_rate': mean_float(items, 'max_point_in_background_rate'),
            'max_point_in_foreground_edge_rate': mean_float(items, 'max_point_in_foreground_edge_rate'),
            'max_point_in_foreground_interior_rate': mean_float(items, 'max_point_in_foreground_interior_rate'),
        })
    return rows, fields


def summarize_image(records, score_key):
    rows = []
    for cls_name in sorted({row['class_name'] for row in records}):
        sub = [row for row in records if row['class_name'] == cls_name]
        labels = np.array([row['label'] for row in sub], dtype=np.int64)
        scores = np.array([row[score_key] for row in sub], dtype=np.float64)
        rows.append({
            'score_name': score_key,
            'class_name': cls_name,
            'n': len(sub),
            'n_normal': int((labels == 0).sum()),
            'n_abnormal': int((labels == 1).sum()),
            'image_AUROC': safe_metric(roc_auc_score, labels, scores),
            'image_AP': safe_metric(average_precision_score, labels, scores),
            'image_F1': f1_max(labels, scores),
            'normal_mean': float(scores[labels == 0].mean()) if np.any(labels == 0) else float('nan'),
            'abnormal_mean': float(scores[labels == 1].mean()) if np.any(labels == 1) else float('nan'),
        })
    avg = {'score_name': score_key, 'class_name': 'Avg', 'n': sum(row['n'] for row in rows), 'n_normal': '', 'n_abnormal': ''}
    for key in ['image_AUROC', 'image_AP', 'image_F1', 'normal_mean', 'abnormal_mean']:
        vals = [row[key] for row in rows if np.isfinite(row[key])]
        avg[key] = float(np.mean(vals)) if vals else float('nan')
    rows.append(avg)
    return rows


def summarize_pixel(pixel_store):
    rows = []
    for map_name, cls_dict in pixel_store.items():
        per_class = []
        for cls_name in sorted(cls_dict.keys()):
            labels = np.concatenate(cls_dict[cls_name]['labels']).astype(np.uint8)
            scores = np.concatenate(cls_dict[cls_name]['scores']).astype(np.float32)
            value = safe_metric(roc_auc_score, labels, scores)
            per_class.append(value)
            rows.append({
                'map_name': map_name,
                'class_name': cls_name,
                'pixel_AUROC_patchgrid': value,
                'n_pixels': int(labels.size),
                'positive_pixels': int(labels.sum()),
            })
        valid = [value for value in per_class if np.isfinite(value)]
        rows.append({
            'map_name': map_name,
            'class_name': 'Avg',
            'pixel_AUROC_patchgrid': float(np.mean(valid)) if valid else float('nan'),
            'n_pixels': '',
            'positive_pixels': '',
        })
    return rows


def parse_percents(value):
    return [float(part.strip()) for part in value.split(',') if part.strip()]


def colorize_fixed(values, lo, hi):
    values = np.asarray(values, dtype=np.float32)
    if hi <= lo:
        x = np.zeros_like(values, dtype=np.float32)
    else:
        x = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay(image_np, heat_np, alpha=0.45):
    image_np = image_np.astype(np.float32)
    heat_np = heat_np.astype(np.float32)
    return np.clip((1.0 - alpha) * image_np + alpha * heat_np, 0, 255).astype(np.uint8)


def save_panel(path, item, vis_ranges):
    image = item['image'].convert('RGB')
    mask = item['mask'].convert('L')
    width, height = image.size
    image_np = np.asarray(image)
    mask_img = Image.fromarray((np.asarray(mask.resize((width, height), Image.NEAREST)) > 0).astype(np.uint8) * 255)

    panels = [image, mask_img.convert('RGB')]
    for map_name in ['localization_map_raw', 'localization_map_foreground', 'localization_map_eroded_foreground']:
        lo, hi = vis_ranges[map_name]
        heat = Image.fromarray(colorize_fixed(item[map_name], lo, hi)).resize((width, height), Image.BILINEAR)
        panels.append(Image.fromarray(overlay(image_np, np.asarray(heat))))

    canvas = Image.new('RGB', (width * len(panels), height), 'white')
    for idx, panel in enumerate(panels):
        canvas.paste(panel.convert('RGB'), (idx * width, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Decoupled BiomedCLIP baseline: global text score for image-level, reversed patch text map for localization.'
    )
    parser.add_argument('-c', '--cfg', default='configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1.py')
    parser.add_argument('--model-name', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--class-names', default='', help='Comma-separated test classes. Default: all.')
    parser.add_argument('--top-percents', default='1,5,10')
    parser.add_argument('--foreground-threshold', type=float, default=8.0)
    parser.add_argument('--foreground-erode-iters', type=int, default=1)
    parser.add_argument('--vis-per-organ', type=int, default=30)
    parser.add_argument('--vis-percentile-low', type=float, default=1.0)
    parser.add_argument('--vis-percentile-high', type=float, default=99.0)
    parser.add_argument('--output-dir', default='', help='Default: runs/biomedclip_decoupled_<timestamp>.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_cfg(args.cfg)
    data_cfg = getattr(cfg, 'data_test', cfg.data)
    class_names = [name.strip() for name in args.class_names.split(',') if name.strip()]
    top_percents = parse_percents(args.top_percents)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, tokenizer, preprocess = load_open_clip_model(args.model_name, device)
    dataset = MedicalTestDataset(data_cfg.root, getattr(data_cfg, 'meta', 'meta.json'), preprocess, class_names)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_test,
    )
    classes = sorted({sample['cls_name'] for sample in dataset.samples})
    prompts_by_class = build_prompts(cfg, classes)
    text_features = encode_prompts(model, tokenizer, prompts_by_class, device)

    out_dir = create_output_dir('biomedclip_decoupled', args.output_dir)
    aggregations = ['max', 'top1%', 'top5%', 'top10%', 'mean']
    map_names = ['localization_map_raw', 'localization_map_foreground', 'localization_map_eroded_foreground']
    records = []
    topk_records = []
    pixel_store = {
        map_name: {cls_name: {'labels': [], 'scores': []} for cls_name in classes}
        for map_name in map_names
    }
    vis_counts = {cls_name: 0 for cls_name in classes}
    vis_items = []
    vis_values = {map_name: [] for map_name in map_names}

    with torch.no_grad():
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            image_features = F.normalize(model.encode_image(imgs), p=2, dim=-1)
            patch_tokens, grid_shape = extract_patch_tokens(model, imgs, project=True)
            if patch_tokens.shape[-1] != next(iter(text_features.values()))['normal'].shape[-1]:
                raise RuntimeError(
                    f'Patch token dim {patch_tokens.shape[-1]} does not match text dim '
                    f"{next(iter(text_features.values()))['normal'].shape[-1]}."
                )

            for idx, cls_name in enumerate(batch['cls_name']):
                cls_text = text_features[cls_name]
                global_sim_normal = float((image_features[idx] * cls_text['normal']).sum().detach().cpu())
                global_sim_abnormal = float((image_features[idx] * cls_text['abnormal']).sum().detach().cpu())
                global_abnormal_minus_normal = global_sim_abnormal - global_sim_normal

                sim_normal = (patch_tokens[idx] * cls_text['normal']).sum(dim=-1)
                sim_abnormal = (patch_tokens[idx] * cls_text['abnormal']).sum(dim=-1)
                # Deliberately reversed for localization, based on the top-k diagnostic.
                raw_map = (sim_normal - sim_abnormal).reshape(grid_shape).detach().cpu().numpy()

                lesion_mask = resize_mask_to_grid(batch['raw_mask'][idx], grid_shape)
                foreground = foreground_mask_to_grid(
                    batch['raw_image'][idx],
                    grid_shape,
                    threshold=args.foreground_threshold,
                )
                eroded_foreground = erode_binary(foreground, args.foreground_erode_iters)
                maps = {
                    'localization_map_raw': raw_map,
                    'localization_map_foreground': mask_to_low(raw_map, foreground),
                    'localization_map_eroded_foreground': mask_to_low(raw_map, eroded_foreground),
                }

                label = int(batch['label'][idx].item())
                row = {
                    'image_path': batch['img_path'][idx],
                    'mask_path': batch['mask_path'][idx],
                    'class_name': cls_name,
                    'label': label,
                    'grid_h': grid_shape[0],
                    'grid_w': grid_shape[1],
                    'mask_grid_sum': int(lesion_mask.sum()),
                    'foreground_grid_pixels': int(foreground.sum()),
                    'eroded_foreground_grid_pixels': int(eroded_foreground.sum()),
                    'global_sim_normal': global_sim_normal,
                    'global_sim_abnormal': global_sim_abnormal,
                    'global_abnormal_minus_normal': global_abnormal_minus_normal,
                    'global_normal_minus_abnormal': -global_abnormal_minus_normal,
                }

                for map_name, score_map in maps.items():
                    pixel_store[map_name][cls_name]['labels'].append(lesion_mask.reshape(-1))
                    pixel_store[map_name][cls_name]['scores'].append(score_map.reshape(-1))
                    row[f'{map_name}_min'] = float(np.min(score_map))
                    row[f'{map_name}_max'] = float(np.max(score_map))
                    row[f'{map_name}_mean'] = float(np.mean(score_map))
                    for agg in aggregations:
                        row[f'{map_name}_{agg}'] = aggregate_score(score_map, agg)
                    base = {
                        'map_name': map_name,
                        'image_path': batch['img_path'][idx],
                        'mask_path': batch['mask_path'][idx],
                        'class_name': cls_name,
                        'label': label,
                    }
                    for top_percent in top_percents:
                        topk_records.append(
                            compute_topk_row(
                                base,
                                score_map,
                                lesion_mask,
                                foreground,
                                args.foreground_erode_iters,
                                top_percent,
                            )
                        )

                records.append(row)
                for map_name, score_map in maps.items():
                    vis_values[map_name].append(score_map.reshape(-1))
                if args.vis_per_organ > 0 and vis_counts[cls_name] < args.vis_per_organ:
                    vis_items.append({
                        'class_name': cls_name,
                        'stem': Path(batch['img_path'][idx]).stem,
                        'label': label,
                        'image': batch['raw_image'][idx],
                        'mask': batch['raw_mask'][idx],
                        **maps,
                    })
                    vis_counts[cls_name] += 1

    write_csv(out_dir / 'records.csv', records, list(records[0].keys()) if records else [])
    image_rows = summarize_image(records, 'global_abnormal_minus_normal')
    for map_name in map_names:
        for agg in ['top1%', 'top5%', 'top10%', 'mean']:
            image_rows.extend(summarize_image(records, f'{map_name}_{agg}'))
    image_fields = [
        'score_name', 'class_name', 'n', 'n_normal', 'n_abnormal',
        'image_AUROC', 'image_AP', 'image_F1', 'normal_mean', 'abnormal_mean',
    ]
    write_csv(out_dir / 'image_metrics.csv', image_rows, image_fields)

    pixel_rows = summarize_pixel(pixel_store)
    write_csv(out_dir / 'pixel_metrics.csv', pixel_rows, ['map_name', 'class_name', 'pixel_AUROC_patchgrid', 'n_pixels', 'positive_pixels'])

    topk_fields = [
        'map_name',
        'image_path',
        'mask_path',
        'class_name',
        'label',
        'top_percent',
        'grid_h',
        'grid_w',
        'lesion_grid_pixels',
        'foreground_grid_pixels',
        'foreground_edge_grid_pixels',
        'foreground_interior_grid_pixels',
        'topk_pixels',
        'max_y',
        'max_x',
        'max_point_in_lesion',
        'max_point_in_background',
        'max_point_in_foreground_edge',
        'max_point_in_foreground_interior',
        'topk_lesion_fraction',
        'topk_background_fraction',
        'topk_foreground_edge_fraction',
        'topk_foreground_interior_fraction',
        'lesion_coverage',
    ]
    write_csv(out_dir / 'localization_topk_records.csv', topk_records, topk_fields)
    topk_summary_rows, topk_summary_fields = summarize_topk(topk_records)
    write_csv(out_dir / 'localization_topk_summary.csv', topk_summary_rows, topk_summary_fields)

    vis_ranges = {}
    for map_name, values in vis_values.items():
        flat = np.concatenate(values).astype(np.float32) if values else np.array([0.0], dtype=np.float32)
        lo, hi = np.percentile(flat, [args.vis_percentile_low, args.vis_percentile_high])
        vis_ranges[map_name] = (float(lo), float(hi))
    for idx, item in enumerate(vis_items):
        name = f"{idx:03d}_{item['stem']}_label{item['label']}.png"
        save_panel(out_dir / 'debug_vis' / item['class_name'] / name, item, vis_ranges)

    write_json(out_dir / 'run_info.json', {
        **vars(args),
        'n_test': len(dataset),
        'classes': classes,
        'image_score': 'global_abnormal_minus_normal',
        'localization_map': 'patch_normal_minus_abnormal',
        'vis_ranges': vis_ranges,
    })

    print(f'Wrote decoupled BiomedCLIP baseline to: {out_dir}')
    for row in image_rows:
        if row['class_name'] == 'Avg' and row['score_name'] == 'global_abnormal_minus_normal':
            print(f"image global_abnormal_minus_normal Avg AUROC={row['image_AUROC']:.4f}")
    for row in pixel_rows:
        if row['class_name'] == 'Avg':
            print(f"{row['map_name']} Avg patch-grid pixel AUROC={row['pixel_AUROC_patchgrid']:.4f}")
    for row in topk_summary_rows:
        if row['class_name'] == 'Avg' and row['label_group'] == 'abnormal' and float(row['top_percent']) == top_percents[0]:
            print(
                f"{row['map_name']} top{row['top_percent']}%: "
                f"max_hit={row['max_point_in_lesion_rate']}, "
                f"topk_lesion={row['topk_lesion_fraction_mean']}, "
                f"background={row['topk_background_fraction_mean']}"
            )


if __name__ == '__main__':
    main()
