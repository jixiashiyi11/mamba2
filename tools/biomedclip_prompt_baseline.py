import argparse
import csv
import importlib
import json
import os
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


def _cfg_path_to_module(cfg_path):
    return cfg_path.split('.')[0].replace('/', '.')


def _load_cfg(cfg_path):
    module = importlib.import_module(_cfg_path_to_module(cfg_path))
    return module.cfg()


def _resolve_prompt(prompt_map, cls_name):
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


def _safe_metric(fn, y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(fn(y_true, y_score))


def _f1_max(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float('nan')
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(np.nanmax(f1))


class MedicalMetaImageDataset(Dataset):
    def __init__(self, root, meta_name, preprocess, class_names=None):
        self.root = Path(root)
        self.preprocess = preprocess
        meta_path = self.root / meta_name
        with meta_path.open('r') as f:
            meta = json.load(f)
        split_meta = meta.get('test', {})
        if class_names:
            classes = [str(name) for name in class_names]
        else:
            classes = sorted(split_meta.keys())
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
        return {
            'img': image_tensor,
            'cls_name': sample['cls_name'],
            'label': int(sample.get('anomaly', 0)),
            'img_path': img_path,
            'mask_path': sample.get('mask_path', ''),
        }


def _collate(batch):
    return {
        'img': torch.stack([item['img'] for item in batch], dim=0),
        'cls_name': [item['cls_name'] for item in batch],
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'img_path': [item['img_path'] for item in batch],
        'mask_path': [item['mask_path'] for item in batch],
    }


def _encode_prompts(model, tokenizer, prompts_by_class, device):
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


def _summarize(records, score_key):
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
            'image_AUROC': _safe_metric(roc_auc_score, labels, scores),
            'image_AP': _safe_metric(average_precision_score, labels, scores),
            'image_F1': _f1_max(labels, scores),
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


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Frozen BiomedCLIP prompt-only image-level baseline.')
    parser.add_argument(
        '-c',
        '--cfg',
        default='configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1.py',
        help='Config used only for test data path and prompts.',
    )
    parser.add_argument('--model-name', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--class-names', default='', help='Comma-separated classes. Default: all classes in test meta.')
    parser.add_argument('--output-dir', default='', help='Default: runs/biomedclip_prompt_baseline_<timestamp>.')
    args = parser.parse_args()

    cfg = _load_cfg(args.cfg)
    data_cfg = getattr(cfg, 'data_test', cfg.data)
    class_names = [name.strip() for name in args.class_names.split(',') if name.strip()]

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError('This baseline requires `open_clip_torch` to be installed.') from exc

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device).eval()

    dataset = MedicalMetaImageDataset(
        root=data_cfg.root,
        meta_name=getattr(data_cfg, 'meta', 'meta.json'),
        preprocess=preprocess,
        class_names=class_names,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=_collate,
    )

    prompt_normal = {str(k).lower(): v for k, v in cfg.prompt_normal.items()}
    prompt_abnormal = {str(k).lower(): v for k, v in cfg.prompt_abnormal.items()}
    classes = sorted({sample['cls_name'] for sample in dataset.samples})
    prompts_by_class = {
        cls_name: {
            'normal': _resolve_prompt(prompt_normal, cls_name),
            'abnormal': _resolve_prompt(prompt_abnormal, cls_name),
        }
        for cls_name in classes
    }
    text_features = _encode_prompts(model, tokenizer, prompts_by_class, device)

    records = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['img'].to(device, non_blocking=True)
            image_features = F.normalize(model.encode_image(imgs), p=2, dim=-1)
            for idx, cls_name in enumerate(batch['cls_name']):
                cls_features = text_features[cls_name]
                sim_normal = torch.sum(image_features[idx] * cls_features['normal']).item()
                sim_abnormal = torch.sum(image_features[idx] * cls_features['abnormal']).item()
                records.append({
                    'image_path': batch['img_path'][idx],
                    'mask_path': batch['mask_path'][idx],
                    'class_name': cls_name,
                    'label': int(batch['label'][idx].item()),
                    'sim_normal': sim_normal,
                    'sim_abnormal': sim_abnormal,
                    'score_abnormal_minus_normal': sim_abnormal - sim_normal,
                    'score_normal_minus_abnormal': sim_normal - sim_abnormal,
                })

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.output_dir or f'runs/biomedclip_prompt_baseline_{timestamp}')
    out_dir.mkdir(parents=True, exist_ok=True)

    record_fields = [
        'image_path', 'mask_path', 'class_name', 'label',
        'sim_normal', 'sim_abnormal',
        'score_abnormal_minus_normal', 'score_normal_minus_abnormal',
    ]
    _write_csv(out_dir / 'records.csv', records, record_fields)

    metric_fields = [
        'score_direction', 'class_name', 'n', 'n_normal', 'n_abnormal',
        'image_AUROC', 'image_AP', 'image_F1', 'normal_mean', 'abnormal_mean',
    ]
    metric_rows = []
    for score_direction, score_key in [
        ('abnormal_minus_normal', 'score_abnormal_minus_normal'),
        ('normal_minus_abnormal', 'score_normal_minus_abnormal'),
    ]:
        for row in _summarize(records, score_key):
            row = dict(row)
            row['score_direction'] = score_direction
            metric_rows.append(row)
    _write_csv(out_dir / 'metrics.csv', metric_rows, metric_fields)

    prompt_rows = []
    for cls_name, prompts in prompts_by_class.items():
        prompt_rows.append({
            'class_name': cls_name,
            'normal_prompt': prompts['normal'],
            'abnormal_prompt': prompts['abnormal'],
        })
    _write_csv(out_dir / 'prompts.csv', prompt_rows, ['class_name', 'normal_prompt', 'abnormal_prompt'])

    print(f'Wrote BiomedCLIP prompt baseline to: {out_dir}')
    for row in metric_rows:
        if row['class_name'] == 'Avg':
            print(
                f"{row['score_direction']}: "
                f"AUROC={row['image_AUROC']:.4f}, AP={row['image_AP']:.4f}, F1={row['image_F1']:.4f}"
            )


if __name__ == '__main__':
    main()
