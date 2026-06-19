import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / 'tools'
for path in [PROJECT_ROOT, TOOLS_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from biomedclip_patch_text_baseline import (  # noqa: E402
    MedicalTestDataset,
    MedicalTrainNormalDataset,
    build_prompts,
    collate_test,
    collate_train,
    create_output_dir,
    encode_prompts,
    extract_patch_tokens,
    load_cfg,
    load_open_clip_model,
    resize_mask_to_grid,
    write_csv,
    write_json,
)
from biomedclip_patch_memory_baseline import (  # noqa: E402
    build_memory_bank,
    nearest_memory_scores,
)


def _to_float(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


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
    return {key: _to_float(value) for key, value in row.items()}


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


def summarize_records(records):
    groups = defaultdict(list)
    for row in records:
        label_group = 'abnormal' if int(row['label']) == 1 else 'normal'
        keys = [
            row['method'],
            row['map_name'],
            row['class_name'],
            label_group,
            row['top_percent'],
        ]
        groups[tuple(keys)].append(row)

    fields = [
        'method',
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
    for (method, map_name, class_name, label_group, top_percent), items in sorted(groups.items()):
        row = {
            'method': method,
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
        }
        rows.append(row)

    # Add per-method/map averages across organs for easier reading.
    avg_groups = defaultdict(list)
    for row in rows:
        if row['class_name'] == 'Avg':
            continue
        avg_groups[(row['method'], row['map_name'], row['label_group'], row['top_percent'])].append(row)
    for (method, map_name, label_group, top_percent), items in sorted(avg_groups.items()):
        rows.append({
            'method': method,
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


def parse_percents(value):
    return [float(part.strip()) for part in value.split(',') if part.strip()]


def make_text_maps(patch_tokens, cls_name, text_features, grid_shape):
    cls_text = text_features[cls_name]
    sim_normal = (patch_tokens * cls_text['normal']).sum(dim=-1)
    sim_abnormal = (patch_tokens * cls_text['abnormal']).sum(dim=-1)
    return {
        'abnormal_minus_normal': (sim_abnormal - sim_normal).reshape(grid_shape).detach().cpu().numpy(),
        'normal_minus_abnormal': (sim_normal - sim_abnormal).reshape(grid_shape).detach().cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Top-k localization diagnostic for BiomedCLIP patch text/memory heatmaps.'
    )
    parser.add_argument('-c', '--cfg', default='configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1.py')
    parser.add_argument('--method', choices=['patch_text', 'patch_memory', 'both'], default='both')
    parser.add_argument('--model-name', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--class-names', default='', help='Comma-separated test classes. Default: all.')
    parser.add_argument('--train-class-names', default='', help='Comma-separated train normal classes for memory. Default: cfg.data_train cls_names.')
    parser.add_argument('--top-percents', default='1,5,10')
    parser.add_argument('--foreground-threshold', type=float, default=8.0)
    parser.add_argument('--foreground-erode-iters', type=int, default=1)
    parser.add_argument('--max-bank-patches', type=int, default=20000)
    parser.add_argument('--distance-chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='', help='Default: runs/localization_topk_diagnostic_<timestamp>.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_cfg(args.cfg)
    test_data_cfg = getattr(cfg, 'data_test', cfg.data)
    train_data_cfg = getattr(cfg, 'data_train', cfg.data)
    test_class_names = [name.strip() for name in args.class_names.split(',') if name.strip()]
    train_class_names = [name.strip() for name in args.train_class_names.split(',') if name.strip()]
    if not train_class_names:
        train_class_names = list(getattr(train_data_cfg, 'cls_names', []) or [])

    top_percents = parse_percents(args.top_percents)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, tokenizer, preprocess = load_open_clip_model(args.model_name, device)

    test_dataset = MedicalTestDataset(
        test_data_cfg.root,
        getattr(test_data_cfg, 'meta', 'meta.json'),
        preprocess,
        test_class_names,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_test,
    )
    classes = sorted({sample['cls_name'] for sample in test_dataset.samples})

    text_features = None
    if args.method in ['patch_text', 'both']:
        prompts_by_class = build_prompts(cfg, classes)
        text_features = encode_prompts(model, tokenizer, prompts_by_class, device)

    memory_bank = None
    train_grid_shape = None
    if args.method in ['patch_memory', 'both']:
        train_dataset = MedicalTrainNormalDataset(
            train_data_cfg.root,
            getattr(train_data_cfg, 'meta', 'meta.json'),
            preprocess,
            train_class_names,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate_train,
        )
        print('Building memory bank for localization diagnostic...')
        memory_bank, train_grid_shape = build_memory_bank(
            model,
            train_loader,
            device,
            max_bank_patches=args.max_bank_patches,
            seed=args.seed,
        )
        memory_bank = memory_bank.to(device)
        print(f'Memory bank patches: {memory_bank.shape[0]} dim={memory_bank.shape[1]} grid={train_grid_shape}')

    out_dir = create_output_dir('localization_topk_diagnostic', args.output_dir)
    records = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['img'].to(device, non_blocking=True)
            patch_tokens, grid_shape = extract_patch_tokens(model, imgs, project=True)

            memory_maps = None
            if memory_bank is not None:
                memory_scores = nearest_memory_scores(patch_tokens, memory_bank, args.distance_chunk_size)
                memory_maps = memory_scores.reshape(
                    memory_scores.shape[0], grid_shape[0], grid_shape[1]
                ).detach().cpu().numpy()

            for idx, cls_name in enumerate(batch['cls_name']):
                label = int(batch['label'][idx].item())
                lesion_mask = resize_mask_to_grid(batch['raw_mask'][idx], grid_shape)
                foreground = foreground_mask_to_grid(
                    batch['raw_image'][idx],
                    grid_shape,
                    threshold=args.foreground_threshold,
                )
                base = {
                    'image_path': batch['img_path'][idx],
                    'mask_path': batch['mask_path'][idx],
                    'class_name': cls_name,
                    'label': label,
                }

                maps = {}
                if text_features is not None:
                    for name, score_map in make_text_maps(patch_tokens[idx], cls_name, text_features, grid_shape).items():
                        maps[('patch_text', name)] = score_map
                if memory_maps is not None:
                    maps[('patch_memory', 'memory_distance')] = memory_maps[idx]

                for (method, map_name), score_map in maps.items():
                    method_base = dict(base)
                    method_base.update({
                        'method': method,
                        'map_name': map_name,
                        'map_min': float(np.min(score_map)),
                        'map_max': float(np.max(score_map)),
                        'map_mean': float(np.mean(score_map)),
                    })
                    for top_percent in top_percents:
                        records.append(
                            compute_topk_row(
                                method_base,
                                score_map,
                                lesion_mask,
                                foreground,
                                args.foreground_erode_iters,
                                top_percent,
                            )
                        )

    record_fields = [
        'method',
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
        'map_min',
        'map_max',
        'map_mean',
    ]
    write_csv(out_dir / 'localization_topk_records.csv', records, record_fields)
    summary_rows, summary_fields = summarize_records(records)
    write_csv(out_dir / 'localization_topk_summary.csv', summary_rows, summary_fields)
    run_info = dict(vars(args))
    run_info.update({
        'n_test': len(test_dataset),
        'classes': classes,
        'train_classes': train_class_names if args.method in ['patch_memory', 'both'] else [],
        'memory_bank_patches': int(memory_bank.shape[0]) if memory_bank is not None else None,
        'memory_bank_dim': int(memory_bank.shape[1]) if memory_bank is not None else None,
        'train_grid_shape': list(train_grid_shape) if train_grid_shape is not None else None,
    })
    write_json(out_dir / 'run_info.json', run_info)

    print(f'Wrote localization diagnostic to: {out_dir}')
    for row in summary_rows:
        if row['class_name'] == 'Avg' and row['label_group'] == 'abnormal' and float(row['top_percent']) == top_percents[0]:
            print(
                f"{row['method']} / {row['map_name']} top{row['top_percent']}%: "
                f"max_hit={row['max_point_in_lesion_rate']}, "
                f"topk_lesion={row['topk_lesion_fraction_mean']}, "
                f"coverage={row['lesion_coverage_mean']}, "
                f"bg={row['topk_background_fraction_mean']}, "
                f"edge={row['topk_foreground_edge_fraction_mean']}"
            )


if __name__ == '__main__':
    main()
