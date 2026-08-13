import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINCLIP_ROOT = REPO_ROOT / 'third_party' / 'WinClip-master'
WINCLIP_ROOT = Path(os.environ.get('WINCLIP_ROOT', DEFAULT_WINCLIP_ROOT))
sys.path.insert(0, str(WINCLIP_ROOT))

if not (WINCLIP_ROOT / 'WinCLIP').exists():
    raise ModuleNotFoundError(
        f'Cannot find WinCLIP under {WINCLIP_ROOT}. '
        'Set WINCLIP_ROOT=/path/to/WinClip-master or place it at third_party/WinClip-master.'
    )

from WinCLIP import WinClipAD  # noqa: E402


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_meta_samples(root, split, class_names):
    meta_path = Path(root) / 'meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    split_meta = meta[split]
    names = class_names if class_names else list(split_meta.keys())
    samples = []
    for cls_name in names:
        samples.extend(split_meta.get(cls_name, []))
    if not samples:
        raise RuntimeError(f'No samples found in {meta_path} split={split} classes={names}')
    return names, samples


def _resize_map(array, size, mode):
    tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False if mode == 'bilinear' else None)
    return tensor.squeeze(0).squeeze(0).numpy()


def _load_mask(root, sample, size):
    anomaly = int(sample.get('anomaly', 0))
    mask_path = sample.get('mask_path', '')
    abs_mask_path = Path(root) / mask_path
    if anomaly == 0 or not mask_path or not abs_mask_path.exists() or abs_mask_path.is_dir():
        return np.zeros((size, size), dtype=np.uint8), 0
    mask = np.array(Image.open(abs_mask_path).convert('L')) > 0
    raw_positive_pixels = int(mask.sum())
    mask = _resize_map(mask.astype(np.float32), size=size, mode='nearest') > 0.5
    return mask.astype(np.uint8), raw_positive_pixels


def _foreground_mask(image, size, threshold):
    image_np = np.array(image)
    foreground = image_np.max(axis=2) > threshold * 255.0
    foreground = _resize_map(foreground.astype(np.float32), size=size, mode='nearest') > 0.5
    return foreground.astype(np.uint8)


class MedicalMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, class_names, resolution, foreground_threshold):
        self.root = Path(root)
        self.resolution = int(resolution)
        self.foreground_threshold = float(foreground_threshold)
        self.class_names, self.samples = _load_meta_samples(self.root, split, class_names)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(self.root / sample['img_path']).convert('RGB')
        mask, raw_positive_pixels = _load_mask(self.root, sample, self.resolution)
        foreground = _foreground_mask(image, self.resolution, self.foreground_threshold)
        return {
            'image': image,
            'img_mask': mask,
            'foreground_mask': foreground,
            'cls_name': sample['cls_name'],
            'anomaly': int(sample.get('anomaly', 0)),
            'img_path': str(self.root / sample['img_path']),
            'mask_path': str(self.root / sample.get('mask_path', '')) if sample.get('mask_path', '') else '',
            'raw_positive_pixels': raw_positive_pixels,
        }


def _collate(batch):
    return {
        'image': [item['image'] for item in batch],
        'img_mask': np.stack([item['img_mask'] for item in batch], axis=0),
        'foreground_mask': np.stack([item['foreground_mask'] for item in batch], axis=0),
        'cls_name': [item['cls_name'] for item in batch],
        'anomaly': np.asarray([item['anomaly'] for item in batch], dtype=np.int64),
        'img_path': [item['img_path'] for item in batch],
        'mask_path': [item['mask_path'] for item in batch],
        'raw_positive_pixels': np.asarray([item['raw_positive_pixels'] for item in batch], dtype=np.int64),
    }


def _category_prompt_name(cls_name):
    return {
        'brain': 'brain medical image',
        'liver': 'liver medical image',
        'retinal': 'retinal fundus image',
    }.get(cls_name, f'{cls_name} medical image')


def export(args):
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    class_names = [item.strip() for item in args.class_names.split(',') if item.strip()]
    dataset = MedicalMetaDataset(
        args.data_path,
        split=args.split,
        class_names=class_names,
        resolution=args.resolution,
        foreground_threshold=args.foreground_threshold,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
        collate_fn=_collate,
    )

    model = WinClipAD(
        out_size_h=args.resolution,
        out_size_w=args.resolution,
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        scales=tuple(args.scales),
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize,
    ).to(device)
    model.eval_mode()

    out = {
        'imgs_masks': [],
        'foreground_masks': [],
        'anomaly_maps': [],
        'image_scores': [],
        'cls_names': [],
        'anomalys': [],
        'img_paths': [],
        'mask_paths': [],
        'raw_positive_pixels': [],
    }
    active_text_gallery = None
    for items in tqdm(dataloader, desc='WinCLIP export'):
        images = torch.stack([model.transform(image) for image in items['image']], dim=0).to(device)
        batch_maps = []

        with torch.no_grad():
            # If a batch crosses class boundaries, evaluate each image with its own text gallery.
            for image, cls_name in zip(images, items['cls_name']):
                if cls_name != active_text_gallery:
                    model.build_text_feature_gallery(_category_prompt_name(cls_name))
                    active_text_gallery = cls_name
                maps = model(image.unsqueeze(0))
                batch_maps.append(maps[0])

        anomaly_maps = np.stack(batch_maps, axis=0).astype(np.float32)
        image_scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(axis=1)

        out['imgs_masks'].append(items['img_mask'].astype(np.uint8))
        out['foreground_masks'].append(items['foreground_mask'].astype(np.uint8))
        out['anomaly_maps'].append(anomaly_maps)
        out['image_scores'].append(image_scores.astype(np.float32))
        out['cls_names'].append(np.asarray(items['cls_name']))
        out['anomalys'].append(items['anomaly'].astype(np.int64))
        out['img_paths'].extend(items['img_path'])
        out['mask_paths'].extend(items['mask_path'])
        out['raw_positive_pixels'].append(items['raw_positive_pixels'].astype(np.int64))

    save = {
        'class_names': np.asarray(dataset.class_names),
        'img_paths': np.asarray(out['img_paths']),
        'mask_paths': np.asarray(out['mask_paths']),
    }
    for key in ['imgs_masks', 'foreground_masks', 'anomaly_maps', 'image_scores', 'cls_names', 'anomalys', 'raw_positive_pixels']:
        save[key] = np.concatenate(out[key], axis=0)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **save)
    print(f'Saved {len(dataset)} samples to {output_path}')


def parse_args():
    parser = argparse.ArgumentParser('Export WinCLIP scores/maps on ADer meta.json medical data.')
    parser.add_argument('--data_path', default='data/medical')
    parser.add_argument('--split', default='test')
    parser.add_argument('--class_names', default='', help='Comma-separated classes. Empty means all classes from meta.json split.')
    parser.add_argument('--output', default='outputs/compare/winclip/winclip_medical_zeroshot.npz')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--img_resize', type=int, default=240)
    parser.add_argument('--img_cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--foreground_threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--scales', nargs='+', type=int, default=(2, 3))
    parser.add_argument('--backbone', default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', default='laion400m_e32')
    return parser.parse_args()


if __name__ == '__main__':
    os.environ['CURL_CA_BUNDLE'] = ''
    export(parse_args())
