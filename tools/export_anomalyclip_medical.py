import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANOMALYCLIP_ROOT = REPO_ROOT / 'third_party' / 'AnomalyCLIP'
ANOMALYCLIP_ROOT = Path(os.environ.get('ANOMALYCLIP_ROOT', DEFAULT_ANOMALYCLIP_ROOT))
sys.path.insert(0, str(ANOMALYCLIP_ROOT))

if not (ANOMALYCLIP_ROOT / 'AnomalyCLIP_lib').exists():
    raise ModuleNotFoundError(
        f'Cannot find AnomalyCLIP_lib under {ANOMALYCLIP_ROOT}. '
        'Set ANOMALYCLIP_ROOT=/path/to/AnomalyCLIP or place official AnomalyCLIP at third_party/AnomalyCLIP.'
    )

import AnomalyCLIP_lib  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
from utils import get_transform  # noqa: E402


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
    if class_names:
        names = class_names
    else:
        names = list(split_meta.keys())
    samples = []
    for cls_name in names:
        samples.extend(split_meta.get(cls_name, []))
    if not samples:
        raise RuntimeError(f'No samples found in {meta_path} split={split} classes={names}')
    return names, samples


class MedicalMetaDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, target_transform, split='test', class_names=None, foreground_threshold=5.0 / 255.0):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.foreground_threshold = float(foreground_threshold)
        self.obj_list, self.samples = _load_meta_samples(self.root, split, class_names or [])
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.obj_list)}

    def __len__(self):
        return len(self.samples)

    def _zero_mask(self, size):
        mask = Image.new('L', size, 0)
        return self.target_transform(mask) if self.target_transform is not None else mask

    def _load_mask(self, sample, size):
        anomaly = int(sample.get('anomaly', 0))
        mask_path = sample.get('mask_path', '')
        abs_mask_path = self.root / mask_path
        if anomaly == 0 or not mask_path or not abs_mask_path.exists() or abs_mask_path.is_dir():
            return self._zero_mask(size), 0
        mask_np = np.array(Image.open(abs_mask_path).convert('L')) > 0
        raw_positive_pixels = int(mask_np.sum())
        mask = Image.fromarray(mask_np.astype(np.uint8) * 255, mode='L')
        return self.target_transform(mask) if self.target_transform is not None else mask, raw_positive_pixels

    def _foreground_mask(self, image):
        image_np = np.array(image)
        if image_np.ndim == 2:
            foreground = image_np > self.foreground_threshold * 255.0
        else:
            foreground = image_np.max(axis=2) > self.foreground_threshold * 255.0
        foreground = Image.fromarray(foreground.astype(np.uint8) * 255, mode='L')
        foreground = self.target_transform(foreground) if self.target_transform is not None else foreground
        if torch.is_tensor(foreground):
            foreground = (foreground > 0.5).to(torch.uint8)
        return foreground

    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = sample['img_path']
        cls_name = sample['cls_name']
        anomaly = int(sample.get('anomaly', 0))
        image = Image.open(self.root / img_path).convert('RGB')
        mask, raw_positive_pixels = self._load_mask(sample, image.size)
        foreground = self._foreground_mask(image)
        image_tensor = self.transform(image) if self.transform is not None else image
        return {
            'img': image_tensor,
            'img_mask': mask,
            'foreground_mask': foreground,
            'cls_name': cls_name,
            'cls_id': self.class_name_to_id[cls_name],
            'anomaly': torch.tensor(anomaly, dtype=torch.long),
            'img_path': str(self.root / img_path),
            'mask_path': str(self.root / sample.get('mask_path', '')) if sample.get('mask_path', '') else '',
            'raw_positive_pixels': torch.tensor(raw_positive_pixels, dtype=torch.long),
        }


def _resolve_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        def epoch_number(candidate):
            try:
                return int(candidate.stem.split('_')[-1])
            except ValueError:
                return -1

        candidates = sorted(path.glob('epoch_*.pth'), key=epoch_number)
        if not candidates:
            raise FileNotFoundError(f'No epoch_*.pth checkpoints found under {path}')
        return candidates[-1]
    return path


def export(args):
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    preprocess, target_transform = get_transform(args)
    class_names = [item.strip() for item in args.class_names.split(',') if item.strip()]
    dataset = MedicalMetaDataset(
        args.data_path,
        preprocess,
        target_transform,
        split=args.split,
        class_names=class_names,
        foreground_threshold=args.foreground_threshold,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
    )

    anomalyclip_parameters = {
        'Prompt_length': args.n_ctx,
        'learnabel_text_embedding_depth': args.depth,
        'learnabel_text_embedding_length': args.t_n_ctx,
    }
    clip_cache_dir = Path(args.clip_cache_dir)
    clip_cache_dir.mkdir(parents=True, exist_ok=True)
    model, _ = AnomalyCLIP_lib.load(
        'ViT-L/14@336px',
        device=device,
        design_details=anomalyclip_parameters,
        download_root=str(clip_cache_dir),
    )
    model.eval()

    prompt_learner = AnomalyCLIP_PromptLearner(model.to('cpu'), anomalyclip_parameters)
    checkpoint_path = _resolve_checkpoint(args.checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    prompt_learner.load_state_dict(checkpoint['prompt_learner'])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    with torch.no_grad():
        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
    min_feature_idx = min(args.feature_map_layer)
    for items in tqdm(dataloader, desc='AnomalyCLIP export'):
        image = items['img'].to(device)
        gt_mask = (items['img_mask'] > 0.5).to(torch.uint8)
        foreground_mask = (items['foreground_mask'] > 0.5).to(torch.uint8)

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=args.dpam_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / args.temperature).softmax(-1)
            image_scores = text_probs[:, 0, 1]

            anomaly_map_list = []
            for feature_idx, patch_feature in enumerate(patch_features):
                if feature_idx < min_feature_idx:
                    continue
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)
            if not anomaly_map_list:
                raise RuntimeError('No anomaly maps were produced. Check --features_list and --feature_map_layer.')
            anomaly_maps = torch.stack(anomaly_map_list).sum(dim=0).detach().cpu().numpy()
            if args.sigma > 0:
                anomaly_maps = np.stack([gaussian_filter(amap, sigma=args.sigma) for amap in anomaly_maps])

        out['imgs_masks'].append(gt_mask.cpu().numpy().astype(np.uint8))
        out['foreground_masks'].append(foreground_mask.cpu().numpy().astype(np.uint8))
        out['anomaly_maps'].append(anomaly_maps.astype(np.float32))
        out['image_scores'].append(image_scores.detach().cpu().numpy().astype(np.float32))
        out['cls_names'].extend(list(items['cls_name']))
        out['anomalys'].append(items['anomaly'].cpu().numpy().astype(np.int64))
        out['img_paths'].extend(list(items['img_path']))
        out['mask_paths'].extend(list(items['mask_path']))
        out['raw_positive_pixels'].append(items['raw_positive_pixels'].cpu().numpy().astype(np.int64))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        imgs_masks=np.concatenate(out['imgs_masks'], axis=0),
        foreground_masks=np.concatenate(out['foreground_masks'], axis=0),
        anomaly_maps=np.concatenate(out['anomaly_maps'], axis=0),
        image_scores=np.concatenate(out['image_scores'], axis=0),
        cls_names=np.array(out['cls_names']),
        anomalys=np.concatenate(out['anomalys'], axis=0),
        img_paths=np.array(out['img_paths']),
        mask_paths=np.array(out['mask_paths']),
        raw_positive_pixels=np.concatenate(out['raw_positive_pixels'], axis=0),
        class_names=np.array(dataset.obj_list),
        checkpoint_path=str(checkpoint_path),
        image_size=args.image_size,
        sigma=args.sigma,
    )
    print(f'Saved {len(dataset)} samples to {output_path}')


def parse_args():
    parser = argparse.ArgumentParser('Export AnomalyCLIP scores/maps on a meta.json medical dataset')
    parser.add_argument('--data_path', required=True, help='Dataset root containing meta.json.')
    parser.add_argument('--checkpoint_path', required=True, help='AnomalyCLIP checkpoint file or directory.')
    parser.add_argument('--output', required=True, help='Output .npz path.')
    parser.add_argument('--clip_cache_dir', default=str(DEFAULT_ANOMALYCLIP_ROOT / '.cache' / 'clip'))
    parser.add_argument('--split', default='test')
    parser.add_argument('--class_names', default='', help='Comma-separated classes. Default: all classes in split.')
    parser.add_argument('--features_list', type=int, nargs='+', default=[24])
    parser.add_argument('--feature_map_layer', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--image_size', type=int, default=518)
    parser.add_argument('--depth', type=int, default=9)
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--t_n_ctx', type=int, default=4)
    parser.add_argument('--dpam_layer', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--sigma', type=float, default=4.0)
    parser.add_argument('--foreground_threshold', type=float, default=5.0 / 255.0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    export(parse_args())
