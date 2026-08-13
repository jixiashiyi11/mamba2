import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from util.data import get_img_loader


IMG_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp')


def _is_image(path):
    return str(path).lower().endswith(IMG_EXTENSIONS)


def _build_transform(transform_cfg):
    if transform_cfg is None:
        return None
    transform_list = []
    for item in transform_cfg:
        kwargs = dict(item)
        transform_type = kwargs.pop('type')
        transform_cls = getattr(transforms, transform_type)
        transform_list.append(transform_cls(**kwargs))
    return transforms.Compose(transform_list)


def _zero_mask(size, target_transform):
    mask = Image.new('L', size, 0)
    if target_transform is not None:
        return target_transform(mask)
    return transforms.ToTensor()(mask)


def _binarize_mask(mask):
    """Convert indexed, grayscale, or 0/255 masks to one binary 0/255 mask."""
    return mask.point(lambda value: 255 if value > 0 else 0, mode='L')


def _meta_image_paths(split_meta):
    """Return normalized image paths used by one meta.json split."""
    return {
        str(item.get('img_path', '')).replace('\\', '/').strip()
        for items in split_meta.values()
        for item in items
        if str(item.get('img_path', '')).strip()
    }


def _assert_disjoint_train_test(meta, meta_path):
    train_paths = _meta_image_paths(meta.get('train', {}))
    test_paths = _meta_image_paths(meta.get('test', {}))
    overlap = sorted(train_paths & test_paths)
    if not overlap:
        return

    examples = ', '.join(overlap[:5])
    raise RuntimeError(
        f'Data leakage detected in {meta_path}: {len(overlap)} image path(s) '
        f'appear in both train and test. Examples: {examples}'
    )


class FolderNormalADDataset(Dataset):
    """Normal-only AD dataset for layouts like root/train/good/*.png."""

    def __init__(self, cfg_data, train=True):
        self.cfg_data = cfg_data
        self.train = train
        self.root = Path(cfg_data.root)
        self.split = 'train' if train else getattr(cfg_data, 'test_split', 'test')
        self.cls_names = list(getattr(cfg_data, 'cls_names', []) or ['good'])
        self.loader = get_img_loader(getattr(cfg_data, 'loader_type', 'pil'))
        self.transform = _build_transform(
            getattr(cfg_data, 'train_transforms' if train else 'test_transforms', None)
        )
        self.target_transform = _build_transform(getattr(cfg_data, 'target_transforms', None))
        self.samples = self._collect_samples()
        self.length = len(self.samples)
        if self.length == 0:
            raise RuntimeError(
                f'No images found for classes {self.cls_names} under {self.root / self.split}. '
                'Expected a normal-only layout such as root/train/good/*.png.'
            )

    def _collect_samples(self):
        split_root = self.root / self.split
        if not split_root.exists() and not self.train:
            split_root = self.root / 'train'

        samples = []
        for cls_name in self.cls_names:
            cls_dir = split_root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(p for p in cls_dir.rglob('*') if p.is_file() and _is_image(p)):
                rel_path = os.path.relpath(img_path, self.root).replace(os.sep, '/')
                samples.append(
                    {
                        'img_path': rel_path,
                        'mask_path': '',
                        'cls_name': cls_name,
                        'specie_name': 'good',
                        'anomaly': 0,
                    }
                )
        return samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_abs = self.root / sample['img_path']
        img = self.loader(str(img_abs))
        mask = _zero_mask(img.size, self.target_transform)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'img': img,
            'img_mask': mask,
            'cls_name': sample['cls_name'],
            'anomaly': torch.tensor(sample['anomaly'], dtype=torch.long),
            'img_path': sample['img_path'],
            'mask_path': sample['mask_path'],
        }


class MetaADDataset(Dataset):
    """Small compatibility loader for ADer-style meta.json datasets."""

    def __init__(self, cfg_data, train=True):
        self.cfg_data = cfg_data
        self.train = train
        self.root = Path(cfg_data.root)
        self.split = getattr(cfg_data, 'train_split', 'train') if train else getattr(cfg_data, 'test_split', 'test')
        self.loader = get_img_loader(getattr(cfg_data, 'loader_type', 'pil'))
        self.mask_loader = get_img_loader(getattr(cfg_data, 'loader_type_target', 'pil_L'))
        self.transform = _build_transform(
            getattr(cfg_data, 'train_transforms' if train else 'test_transforms', None)
        )
        self.target_transform = _build_transform(getattr(cfg_data, 'target_transforms', None))
        self.samples = self._load_samples()
        self.cls_names = list(getattr(cfg_data, 'cls_names', []) or sorted({s['cls_name'] for s in self.samples}))
        self.length = len(self.samples)
        if self.length == 0:
            raise RuntimeError(f'No {self.split} samples found under {self.root}.')

    def _load_samples(self):
        meta_path = self.root / getattr(self.cfg_data, 'meta', 'meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if bool(getattr(self.cfg_data, 'enforce_disjoint_train_test', False)):
                _assert_disjoint_train_test(meta, meta_path)
            split_meta = meta.get(self.split, {})
            cls_names = list(getattr(self.cfg_data, 'cls_names', []) or split_meta.keys())
            samples = []
            for cls_name in cls_names:
                samples.extend(split_meta.get(cls_name, []))
            return samples
        if bool(getattr(self.cfg_data, 'require_meta', False)):
            raise FileNotFoundError(
                f'Required dataset metadata does not exist: {meta_path}. '
                'Refusing to fall back to directory scanning for this experiment.'
            )
        return self._scan_mvtec_style()

    def _scan_mvtec_style(self):
        split_root = self.root
        cls_names = list(getattr(self.cfg_data, 'cls_names', []) or [
            p.name for p in sorted(split_root.iterdir()) if p.is_dir()
        ])
        samples = []
        for cls_name in cls_names:
            cls_split_dir = self.root / cls_name / self.split
            if not cls_split_dir.exists():
                continue
            for specie_dir in sorted(p for p in cls_split_dir.iterdir() if p.is_dir()):
                specie_name = specie_dir.name
                train_with_masks = bool(getattr(self.cfg_data, 'train_with_anomaly_masks', False))
                anomaly = 0 if (self.train and not train_with_masks) or specie_name == 'good' else 1
                for img_path in sorted(p for p in specie_dir.rglob('*') if p.is_file() and _is_image(p)):
                    mask_path = ''
                    if anomaly:
                        mask_name = f'{img_path.stem}_mask.png'
                        mask_path = str(Path(cls_name) / 'ground_truth' / specie_name / mask_name).replace(os.sep, '/')
                    samples.append(
                        {
                            'img_path': os.path.relpath(img_path, self.root).replace(os.sep, '/'),
                            'mask_path': mask_path,
                            'cls_name': cls_name,
                            'specie_name': specie_name,
                            'anomaly': anomaly,
                        }
                    )
        return samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self.loader(str(self.root / sample['img_path']))
        mask_path = sample.get('mask_path', '')
        if mask_path and (self.root / mask_path).exists():
            mask = self.mask_loader(str(self.root / mask_path))
            # VisA masks use non-zero class indices rather than only value 255.
            # Binarize before ToTensor(), otherwise the later >0.5 threshold can
            # erase every low-valued anomalous pixel. This is a no-op for MVTec's
            # existing 0/255 masks.
            mask = _binarize_mask(mask)
            mask = self.target_transform(mask) if self.target_transform is not None else transforms.ToTensor()(mask)
        else:
            mask = _zero_mask(img.size, self.target_transform)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'img': img,
            'img_mask': mask,
            'cls_name': sample['cls_name'],
            'anomaly': torch.tensor(int(sample.get('anomaly', 0)), dtype=torch.long),
            'img_path': sample['img_path'],
            'mask_path': mask_path,
        }


def _make_dataset(cfg_data, train):
    data_type = getattr(cfg_data, 'type', 'DefaultAD')
    if data_type == 'FolderNormalAD':
        return FolderNormalADDataset(cfg_data, train=train)
    if data_type in ['DefaultAD', 'RealIAD', 'RealIADO']:
        return MetaADDataset(cfg_data, train=train)
    raise NotImplementedError(f'Unsupported dataset type: {data_type}')


def _make_loader(dataset, cfg, train):
    if train:
        batch_size = cfg.trainer.data.batch_size_per_gpu
        drop_last = cfg.trainer.data.drop_last
    else:
        batch_size = cfg.trainer.data.batch_size_per_gpu_test
        drop_last = False

    sampler = DistributedSampler(dataset, shuffle=train) if cfg.dist else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=cfg.trainer.data.num_workers_per_gpu,
        pin_memory=cfg.trainer.data.pin_memory,
        drop_last=drop_last,
        persistent_workers=cfg.trainer.data.persistent_workers,
    )


def get_loader(cfg):
    if bool(getattr(cfg, 'clip_ad_cross_domain', False)):
        if not hasattr(cfg, 'data_train') or not hasattr(cfg, 'data_test'):
            raise ValueError(
                'clip_ad_cross_domain=True requires both cfg.data_train and cfg.data_test.'
            )
        train_dataset = _make_dataset(cfg.data_train, train=True)
        test_dataset = _make_dataset(cfg.data_test, train=False)
    else:
        train_dataset = _make_dataset(cfg.data, train=True)
        test_dataset = _make_dataset(cfg.data, train=False)
    train_loader = _make_loader(train_dataset, cfg, train=True)
    test_loader = _make_loader(test_dataset, cfg, train=False)
    cfg.trainer.iter_full = cfg.trainer.epoch_full * len(train_loader)
    return train_loader, test_loader
