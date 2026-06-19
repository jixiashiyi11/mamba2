import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def cfg_path_to_module(cfg_path):
    return cfg_path.split('.')[0].replace('/', '.')


def load_cfg(cfg_path):
    module = importlib.import_module(cfg_path_to_module(cfg_path))
    return module.cfg()


def resolve_prompt(prompt_map, cls_name):
    key = str(cls_name).lower()
    if key in prompt_map:
        return prompt_map[key]
    if '__shared__' in prompt_map:
        value = prompt_map['__shared__']
        if '{class_name}' in value:
            return value.format(class_name=key)
        if '{cls_name}' in value:
            return value.format(cls_name=key)
        return value
    raise KeyError(f'No prompt found for class `{cls_name}`. Available keys: {sorted(prompt_map.keys())}.')


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


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(data, f, indent=2)


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return (values - lo) / (hi - lo)


def colorize_heatmap(values):
    x = normalize_map(values)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(image_np, heatmap_np, alpha=0.45):
    image_np = image_np.astype(np.float32)
    heatmap_np = heatmap_np.astype(np.float32)
    return np.clip((1.0 - alpha) * image_np + alpha * heatmap_np, 0, 255).astype(np.uint8)


def save_debug_panel(path, image, mask, patch_map, title=''):
    image = image.convert('RGB')
    width, height = image.size
    heat = Image.fromarray(colorize_heatmap(patch_map)).resize((width, height), Image.BILINEAR)
    overlay = Image.fromarray(overlay_heatmap(np.asarray(image), np.asarray(heat)))
    mask_img = Image.fromarray((np.asarray(mask.resize((width, height), Image.NEAREST)) > 0).astype(np.uint8) * 255)
    panel = Image.new('RGB', (width * 4, height + 24), 'white')
    panel.paste(image, (0, 24))
    panel.paste(mask_img.convert('RGB'), (width, 24))
    panel.paste(heat.convert('RGB'), (width * 2, 24))
    panel.paste(overlay.convert('RGB'), (width * 3, 24))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


class MedicalTestDataset(Dataset):
    def __init__(self, root, meta_name, preprocess, class_names=None):
        self.root = Path(root)
        self.preprocess = preprocess
        meta_path = self.root / meta_name
        with meta_path.open('r') as f:
            meta = json.load(f)
        split_meta = meta.get('test', {})
        classes = [str(name) for name in class_names] if class_names else sorted(split_meta.keys())
        self.samples = []
        for cls_name in classes:
            for sample in split_meta.get(cls_name, []):
                item = dict(sample)
                item['cls_name'] = cls_name
                self.samples.append(item)
        if not self.samples:
            raise RuntimeError(f'No test samples found in {meta_path} for classes={classes}.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['img_path']
        image = Image.open(self.root / img_path).convert('RGB')
        image_tensor = self.preprocess(image)
        mask_path = sample.get('mask_path', '')
        if mask_path:
            full_mask_path = self.root / mask_path
            if full_mask_path.exists():
                mask = Image.open(full_mask_path).convert('L')
            else:
                mask = Image.new('L', image.size, 0)
        else:
            mask = Image.new('L', image.size, 0)
        return {
            'img': image_tensor,
            'label': int(sample.get('anomaly', 0)),
            'cls_name': sample['cls_name'],
            'img_path': img_path,
            'mask_path': mask_path,
            'raw_image': image,
            'raw_mask': mask,
        }


class MedicalTrainNormalDataset(Dataset):
    def __init__(self, root, meta_name, preprocess, class_names=None):
        self.root = Path(root)
        self.preprocess = preprocess
        self.samples = self._load_samples(meta_name, class_names)
        if not self.samples:
            raise RuntimeError(f'No train normal samples found under {self.root}.')

    def _load_samples(self, meta_name, class_names):
        meta_path = self.root / meta_name
        samples = []
        if meta_path.exists():
            with meta_path.open('r') as f:
                meta = json.load(f)
            split_meta = meta.get('train', {})
            classes = [str(name) for name in class_names] if class_names else sorted(split_meta.keys())
            for cls_name in classes:
                for item in split_meta.get(cls_name, []):
                    if int(item.get('anomaly', 0)) != 0:
                        continue
                    sample = dict(item)
                    sample['cls_name'] = cls_name
                    samples.append(sample)
            if samples:
                return samples

        split_root = self.root / 'train'
        classes = [str(name) for name in class_names] if class_names else []
        if not classes and split_root.exists():
            classes = sorted([p.name for p in split_root.iterdir() if p.is_dir()])
        for cls_name in classes:
            cls_root = split_root / cls_name
            if not cls_root.exists():
                continue
            for path in sorted(cls_root.rglob('*')):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                samples.append({
                    'img_path': str(path.relative_to(self.root)).replace(os.sep, '/'),
                    'cls_name': cls_name,
                    'anomaly': 0,
                    'mask_path': '',
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(self.root / sample['img_path']).convert('RGB')
        return {
            'img': self.preprocess(image),
            'cls_name': sample['cls_name'],
            'img_path': sample['img_path'],
        }


def collate_test(batch):
    return {
        'img': torch.stack([item['img'] for item in batch], dim=0),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'cls_name': [item['cls_name'] for item in batch],
        'img_path': [item['img_path'] for item in batch],
        'mask_path': [item['mask_path'] for item in batch],
        'raw_image': [item['raw_image'] for item in batch],
        'raw_mask': [item['raw_mask'] for item in batch],
    }


def collate_train(batch):
    return {
        'img': torch.stack([item['img'] for item in batch], dim=0),
        'cls_name': [item['cls_name'] for item in batch],
        'img_path': [item['img_path'] for item in batch],
    }


def _pick_feature_tensor(features):
    if isinstance(features, dict):
        for key in ['x_norm_patchtokens', 'x_norm', 'tokens', 'last_hidden_state']:
            if key in features:
                return features[key]
        return next(reversed(features.values()))
    if isinstance(features, (list, tuple)):
        return features[-1]
    return features


def _tokens_from_feature_tensor(features, visual):
    features = _pick_feature_tensor(features)
    if features.ndim == 4:
        bsz, channels, height, width = features.shape
        return features.permute(0, 2, 3, 1).reshape(bsz, height * width, channels), (height, width)
    if features.ndim != 3:
        raise RuntimeError(f'Expected patch features with 3 or 4 dims, got shape {tuple(features.shape)}.')

    num_tokens = features.shape[1]
    trunk = getattr(visual, 'trunk', visual)
    prefix_candidates = [
        int(getattr(trunk, 'num_prefix_tokens', 0) or 0),
        int(getattr(trunk, 'num_tokens', 0) or 0),
        1,
        0,
    ]
    for prefix in prefix_candidates:
        patch_count = num_tokens - prefix
        side = int(round(math.sqrt(patch_count)))
        if patch_count > 0 and side * side == patch_count:
            return features[:, prefix:, :], (side, side)
    side = int(math.floor(math.sqrt(num_tokens)))
    patch_count = side * side
    if patch_count <= 0:
        raise RuntimeError(f'Cannot infer patch grid from {num_tokens} tokens.')
    return features[:, -patch_count:, :], (side, side)


def _get_projection_module(model):
    visual = getattr(model, 'visual', None)
    candidates = []
    if visual is not None:
        head = getattr(visual, 'head', None)
        if head is not None:
            candidates.extend([getattr(head, 'proj', None), getattr(head, 'fc', None)])
        candidates.extend([getattr(visual, 'proj', None), getattr(visual, 'projection', None)])
        trunk = getattr(visual, 'trunk', None)
        if trunk is not None:
            candidates.extend([getattr(trunk, 'proj', None), getattr(trunk, 'head', None)])
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def _apply_projection(tokens, projection):
    if projection is None:
        return tokens
    if isinstance(projection, torch.nn.Identity):
        return tokens
    if isinstance(projection, torch.nn.Linear):
        flat = tokens.reshape(-1, tokens.shape[-1])
        return projection(flat).reshape(tokens.shape[0], tokens.shape[1], -1)
    if isinstance(projection, torch.nn.Parameter):
        return tokens @ projection
    if torch.is_tensor(projection):
        return tokens @ projection
    return tokens


def extract_patch_tokens(model, imgs, project=True):
    visual = model.visual
    trunk = getattr(visual, 'trunk', visual)
    if hasattr(trunk, 'forward_features'):
        features = trunk.forward_features(imgs)
    else:
        features = trunk(imgs)
    tokens, grid_shape = _tokens_from_feature_tensor(features, visual)
    if project:
        tokens = _apply_projection(tokens, _get_projection_module(model))
    return F.normalize(tokens, p=2, dim=-1), grid_shape


def encode_prompts(model, tokenizer, prompts_by_class, device):
    text_features = {}
    with torch.no_grad():
        for cls_name, prompt_pair in prompts_by_class.items():
            tokens = tokenizer([prompt_pair['normal'], prompt_pair['abnormal']]).to(device)
            features = F.normalize(model.encode_text(tokens), p=2, dim=-1)
            text_features[cls_name] = {
                'normal': features[0],
                'abnormal': features[1],
            }
    return text_features


def aggregate_score(values, aggregation):
    flat = values.reshape(-1)
    if aggregation == 'max':
        return float(np.max(flat))
    if aggregation == 'mean':
        return float(np.mean(flat))
    if aggregation.startswith('top'):
        ratio = float(aggregation[3:].rstrip('%')) / 100.0
        k = max(1, int(flat.size * ratio))
        return float(np.sort(flat)[-k:].mean())
    raise ValueError(f'Unknown aggregation: {aggregation}')


def resize_mask_to_grid(mask, grid_shape):
    return (np.asarray(mask.resize((grid_shape[1], grid_shape[0]), Image.NEAREST)) > 0).astype(np.uint8)


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


def build_prompts(cfg, classes):
    prompt_normal = {str(k).lower(): v for k, v in cfg.prompt_normal.items()}
    prompt_abnormal = {str(k).lower(): v for k, v in cfg.prompt_abnormal.items()}
    return {
        cls_name: {
            'normal': resolve_prompt(prompt_normal, cls_name),
            'abnormal': resolve_prompt(prompt_abnormal, cls_name),
        }
        for cls_name in classes
    }


def create_output_dir(prefix, output_dir=''):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = Path(output_dir or f'runs/{prefix}_{timestamp}')
    if out_dir.exists():
        suffix = datetime.now().strftime('%f')
        out_dir = out_dir.parent / f'{out_dir.name}_{suffix}'
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def load_open_clip_model(model_name, device):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError('This baseline requires `open_clip_torch` to be installed.') from exc
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, tokenizer, preprocess


def main():
    parser = argparse.ArgumentParser(description='BiomedCLIP patch-token text-similarity dense baseline.')
    parser.add_argument('-c', '--cfg', default='configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1.py')
    parser.add_argument('--model-name', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--class-names', default='', help='Comma-separated test classes. Default: all.')
    parser.add_argument('--output-dir', default='', help='Default: runs/biomedclip_patch_text_<timestamp>.')
    parser.add_argument('--vis-per-organ', type=int, default=20)
    parser.add_argument('--vis-direction', default='abnormal_minus_normal', choices=['abnormal_minus_normal', 'normal_minus_abnormal'])
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    data_cfg = getattr(cfg, 'data_test', cfg.data)
    class_names = [name.strip() for name in args.class_names.split(',') if name.strip()]
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

    out_dir = create_output_dir('biomedclip_patch_text', args.output_dir)
    aggregations = ['max', 'top1%', 'top5%', 'top10%', 'mean']
    records = []
    pixel_store = {
        direction: {cls_name: {'labels': [], 'scores': []} for cls_name in classes}
        for direction in ['abnormal_minus_normal', 'normal_minus_abnormal']
    }
    vis_counts = {cls_name: 0 for cls_name in classes}

    with torch.no_grad():
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            patch_tokens, grid_shape = extract_patch_tokens(model, imgs, project=True)
            if patch_tokens.shape[-1] != next(iter(text_features.values()))['normal'].shape[-1]:
                raise RuntimeError(
                    f'Patch token dim {patch_tokens.shape[-1]} does not match text dim '
                    f"{next(iter(text_features.values()))['normal'].shape[-1]}. Projection could not be inferred."
                )
            for idx, cls_name in enumerate(batch['cls_name']):
                cls_text = text_features[cls_name]
                sim_normal = (patch_tokens[idx] * cls_text['normal']).sum(dim=-1)
                sim_abnormal = (patch_tokens[idx] * cls_text['abnormal']).sum(dim=-1)
                maps = {
                    'abnormal_minus_normal': (sim_abnormal - sim_normal).reshape(grid_shape).detach().cpu().numpy(),
                    'normal_minus_abnormal': (sim_normal - sim_abnormal).reshape(grid_shape).detach().cpu().numpy(),
                }
                mask_grid = resize_mask_to_grid(batch['raw_mask'][idx], grid_shape)
                for direction, score_map in maps.items():
                    pixel_store[direction][cls_name]['labels'].append(mask_grid.reshape(-1))
                    pixel_store[direction][cls_name]['scores'].append(score_map.reshape(-1))

                row = {
                    'image_path': batch['img_path'][idx],
                    'mask_path': batch['mask_path'][idx],
                    'class_name': cls_name,
                    'label': int(batch['label'][idx].item()),
                    'grid_h': grid_shape[0],
                    'grid_w': grid_shape[1],
                    'mask_grid_sum': int(mask_grid.sum()),
                }
                for direction, score_map in maps.items():
                    row[f'{direction}_map_min'] = float(score_map.min())
                    row[f'{direction}_map_max'] = float(score_map.max())
                    row[f'{direction}_map_mean'] = float(score_map.mean())
                    for agg in aggregations:
                        row[f'{direction}_{agg}'] = aggregate_score(score_map, agg)
                records.append(row)

                if args.vis_per_organ > 0 and vis_counts[cls_name] < args.vis_per_organ:
                    score_map = maps[args.vis_direction]
                    name = Path(batch['img_path'][idx]).stem
                    label = int(batch['label'][idx].item())
                    save_debug_panel(
                        out_dir / 'debug_vis' / cls_name / f'{vis_counts[cls_name]:03d}_{name}_label{label}.png',
                        batch['raw_image'][idx],
                        batch['raw_mask'][idx],
                        score_map,
                    )
                    vis_counts[cls_name] += 1

    record_fields = list(records[0].keys()) if records else []
    write_csv(out_dir / 'records.csv', records, record_fields)

    metric_rows = []
    for direction in ['abnormal_minus_normal', 'normal_minus_abnormal']:
        pixel_metrics = summarize_pixel_metrics(pixel_store[direction])
        for agg in aggregations:
            score_key = f'{direction}_{agg}'
            for row in summarize_image_metrics(records, score_key):
                row = dict(row)
                row['score_direction'] = direction
                row['aggregation'] = agg
                row['pixel_AUROC_patchgrid'] = pixel_metrics.get(row['class_name'], float('nan'))
                metric_rows.append(row)
    metric_fields = [
        'score_direction', 'aggregation', 'class_name', 'n', 'n_normal', 'n_abnormal',
        'image_AUROC', 'image_AP', 'image_F1', 'pixel_AUROC_patchgrid',
        'normal_mean', 'abnormal_mean',
    ]
    write_csv(out_dir / 'metrics.csv', metric_rows, metric_fields)
    write_csv(
        out_dir / 'prompts.csv',
        [{'class_name': k, 'normal_prompt': v['normal'], 'abnormal_prompt': v['abnormal']} for k, v in prompts_by_class.items()],
        ['class_name', 'normal_prompt', 'abnormal_prompt'],
    )
    run_info = dict(vars(args))
    run_info.update({'n_test': len(dataset), 'classes': classes})
    write_json(out_dir / 'run_info.json', run_info)

    print(f'Wrote BiomedCLIP patch text baseline to: {out_dir}')
    for row in metric_rows:
        if row['class_name'] == 'Avg' and row['aggregation'] == 'top1%':
            print(
                f"{row['score_direction']} top1%: "
                f"image_AUROC={row['image_AUROC']:.4f}, "
                f"pixel_AUROC_patchgrid={row['pixel_AUROC_patchgrid']:.4f}"
            )


if __name__ == '__main__':
    main()
