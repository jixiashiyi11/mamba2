import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader

from biomedclip_patch_text_baseline import (
    MedicalTestDataset,
    MedicalTrainNormalDataset,
    aggregate_score,
    collate_test,
    collate_train,
    create_output_dir,
    extract_patch_tokens,
    f1_max,
    load_cfg,
    load_open_clip_model,
    resize_mask_to_grid,
    safe_metric,
    save_debug_panel,
    write_csv,
    write_json,
)


def summarize_image_metrics(records, score_key):
    rows = []
    classes = sorted({record['class_name'] for record in records})
    for cls_name in classes:
        sub = [record for record in records if record['class_name'] == cls_name]
        labels = np.array([record['label'] for record in sub], dtype=np.int64)
        scores = np.array([record[score_key] for record in sub], dtype=np.float64)
        rows.append({
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
    metric_keys = ['image_AUROC', 'image_AP', 'image_F1', 'normal_mean', 'abnormal_mean']
    avg = {'class_name': 'Avg', 'n': sum(row['n'] for row in rows), 'n_normal': '', 'n_abnormal': ''}
    for key in metric_keys:
        values = [row[key] for row in rows if np.isfinite(row[key])]
        avg[key] = float(np.mean(values)) if values else float('nan')
    rows.append(avg)
    return rows


def summarize_pixel_metrics(pixel_store):
    rows = {}
    for cls_name, values in pixel_store.items():
        labels = np.concatenate(values['labels']).astype(np.uint8)
        scores = np.concatenate(values['scores']).astype(np.float32)
        rows[cls_name] = safe_metric(roc_auc_score, labels, scores)
    valid = [value for value in rows.values() if np.isfinite(value)]
    rows['Avg'] = float(np.mean(valid)) if valid else float('nan')
    return rows


def build_memory_bank(model, loader, device, max_bank_patches, seed):
    all_patches = []
    grid_shape = None
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            patch_tokens, batch_grid_shape = extract_patch_tokens(model, imgs, project=True)
            grid_shape = batch_grid_shape
            all_patches.append(patch_tokens.reshape(-1, patch_tokens.shape[-1]).cpu())
    bank = torch.cat(all_patches, dim=0)
    if max_bank_patches > 0 and bank.shape[0] > max_bank_patches:
        generator = torch.Generator()
        generator.manual_seed(seed)
        index = torch.randperm(bank.shape[0], generator=generator)[:max_bank_patches]
        bank = bank.index_select(0, index)
    bank = torch.nn.functional.normalize(bank, p=2, dim=-1)
    return bank, grid_shape


def nearest_memory_scores(patch_tokens, memory_bank, chunk_size):
    flat = patch_tokens.reshape(-1, patch_tokens.shape[-1])
    scores = []
    for start in range(0, flat.shape[0], chunk_size):
        chunk = flat[start:start + chunk_size]
        sim = chunk @ memory_bank.T
        nearest_sim = sim.max(dim=1).values
        scores.append(1.0 - nearest_sim)
    scores = torch.cat(scores, dim=0)
    return scores.reshape(patch_tokens.shape[0], patch_tokens.shape[1])


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP patch-token normal memory-bank dense baseline.')
    parser.add_argument('-c', '--cfg', default='configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1.py')
    parser.add_argument('--model-name', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--class-names', default='', help='Comma-separated test classes. Default: all.')
    parser.add_argument('--train-class-names', default='', help='Comma-separated train normal classes. Default: cfg.data cls_names.')
    parser.add_argument('--max-bank-patches', type=int, default=20000)
    parser.add_argument('--distance-chunk-size', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='', help='Default: runs/biomedclip_patch_memory_<timestamp>.')
    parser.add_argument('--vis-per-organ', type=int, default=20)
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, _, preprocess = load_open_clip_model(args.model_name, device)

    train_dataset = MedicalTrainNormalDataset(
        train_data_cfg.root,
        getattr(train_data_cfg, 'meta', 'meta.json'),
        preprocess,
        train_class_names,
    )
    test_dataset = MedicalTestDataset(
        test_data_cfg.root,
        getattr(test_data_cfg, 'meta', 'meta.json'),
        preprocess,
        test_class_names,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_train,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_test,
    )

    out_dir = create_output_dir('biomedclip_patch_memory', args.output_dir)
    print('Building memory bank...')
    memory_bank, train_grid_shape = build_memory_bank(
        model,
        train_loader,
        device,
        max_bank_patches=args.max_bank_patches,
        seed=args.seed,
    )
    memory_bank = memory_bank.to(device)
    print(f'Memory bank patches: {memory_bank.shape[0]} dim={memory_bank.shape[1]} grid={train_grid_shape}')

    classes = sorted({sample['cls_name'] for sample in test_dataset.samples})
    aggregations = ['max', 'top1%', 'top5%', 'top10%', 'mean']
    records = []
    pixel_store = {cls_name: {'labels': [], 'scores': []} for cls_name in classes}
    vis_counts = {cls_name: 0 for cls_name in classes}

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['img'].to(device, non_blocking=True)
            patch_tokens, grid_shape = extract_patch_tokens(model, imgs, project=True)
            patch_scores = nearest_memory_scores(patch_tokens, memory_bank, args.distance_chunk_size)
            patch_scores_np = patch_scores.reshape(patch_scores.shape[0], grid_shape[0], grid_shape[1]).cpu().numpy()

            for idx, cls_name in enumerate(batch['cls_name']):
                score_map = patch_scores_np[idx]
                mask_grid = resize_mask_to_grid(batch['raw_mask'][idx], grid_shape)
                pixel_store[cls_name]['labels'].append(mask_grid.reshape(-1))
                pixel_store[cls_name]['scores'].append(score_map.reshape(-1))

                row = {
                    'image_path': batch['img_path'][idx],
                    'mask_path': batch['mask_path'][idx],
                    'class_name': cls_name,
                    'label': int(batch['label'][idx].item()),
                    'grid_h': grid_shape[0],
                    'grid_w': grid_shape[1],
                    'mask_grid_sum': int(mask_grid.sum()),
                    'map_min': float(score_map.min()),
                    'map_max': float(score_map.max()),
                    'map_mean': float(score_map.mean()),
                }
                for agg in aggregations:
                    row[agg] = aggregate_score(score_map, agg)
                records.append(row)

                if args.vis_per_organ > 0 and vis_counts[cls_name] < args.vis_per_organ:
                    name = Path(batch['img_path'][idx]).stem
                    label = int(batch['label'][idx].item())
                    save_debug_panel(
                        out_dir / 'debug_vis' / cls_name / f'{vis_counts[cls_name]:03d}_{name}_label{label}.png',
                        batch['raw_image'][idx],
                        batch['raw_mask'][idx],
                        score_map,
                    )
                    vis_counts[cls_name] += 1

    write_csv(out_dir / 'records.csv', records, list(records[0].keys()) if records else [])

    pixel_metrics = summarize_pixel_metrics(pixel_store)
    metric_rows = []
    for agg in aggregations:
        for row in summarize_image_metrics(records, agg):
            row = dict(row)
            row['aggregation'] = agg
            row['pixel_AUROC_patchgrid'] = pixel_metrics.get(row['class_name'], float('nan'))
            metric_rows.append(row)

    metric_fields = [
        'aggregation', 'class_name', 'n', 'n_normal', 'n_abnormal',
        'image_AUROC', 'image_AP', 'image_F1', 'pixel_AUROC_patchgrid',
        'normal_mean', 'abnormal_mean',
    ]
    write_csv(out_dir / 'metrics.csv', metric_rows, metric_fields)
    run_info = dict(vars(args))
    run_info.update({
        'n_train': len(train_dataset),
        'n_test': len(test_dataset),
        'test_classes': classes,
        'train_classes': train_class_names,
        'memory_bank_patches': int(memory_bank.shape[0]),
        'memory_bank_dim': int(memory_bank.shape[1]),
        'train_grid_shape': list(train_grid_shape) if train_grid_shape is not None else None,
    })
    write_json(out_dir / 'run_info.json', run_info)

    print(f'Wrote BiomedCLIP patch memory baseline to: {out_dir}')
    for row in metric_rows:
        if row['class_name'] == 'Avg' and row['aggregation'] == 'top1%':
            print(
                f"memory top1%: image_AUROC={row['image_AUROC']:.4f}, "
                f"pixel_AUROC_patchgrid={row['pixel_AUROC_patchgrid']:.4f}"
            )


if __name__ == '__main__':
    main()
