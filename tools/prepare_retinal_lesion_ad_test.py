import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


LESION_TYPES = ('EX', 'HE', 'MA', 'SE')


def _read_mask(path, size):
    if not path.exists():
        return np.zeros((size, size), dtype=np.uint8)
    mask = Image.open(path).convert('L').resize((size, size), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def _save_image(src_path, dst_path, size):
    image = Image.open(src_path).convert('RGB').resize((size, size), Image.BILINEAR)
    image.save(dst_path)


def _save_mask(mask, dst_path):
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode='L').save(dst_path)


def _image_paths(raw_root):
    image_root = Path(raw_root) / 'lesion_test' / 'image'
    paths = sorted(image_root.glob('*.jpg'))
    if not paths:
        raise RuntimeError(f'No retinal jpg images found under {image_root}')
    return paths


def _merged_mask(raw_root, image_path, size):
    mask_root = Path(raw_root) / 'lesion_test' / 'mask'
    merged = np.zeros((size, size), dtype=np.uint8)
    for lesion_type in LESION_TYPES:
        mask_path = mask_root / lesion_type / f'{image_path.stem}.tif'
        merged = np.maximum(merged, _read_mask(mask_path, size))
    return merged


def main():
    parser = argparse.ArgumentParser(
        description='Convert retinal lesion segmentation data into ADer-style retinal anomaly test set.'
    )
    parser.add_argument('--raw-root', default='data/raw/retinal_benchmark/sibins_dr_lesion_seg')
    parser.add_argument('--output-root', default='data/medical_standard_retinal_lesion')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--min-mask-pixels', type=int, default=10)
    parser.add_argument('--max-bad', type=int, default=100000)
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.clear and output_root.exists():
        shutil.rmtree(output_root)

    bad_dir = output_root / 'retinal' / 'test' / 'bad'
    mask_dir = output_root / 'retinal' / 'ground_truth' / 'bad'
    for directory in [bad_dir, mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    meta = {'train': {'retinal': []}, 'test': {'retinal': []}}
    saved_bad = 0
    skipped_empty = 0

    for image_path in _image_paths(args.raw_root):
        if saved_bad >= args.max_bad:
            break
        mask = _merged_mask(args.raw_root, image_path, args.size)
        mask_pixels = int(mask.sum())
        if mask_pixels < args.min_mask_pixels:
            skipped_empty += 1
            continue

        stem = image_path.stem
        img_rel = f'retinal/test/bad/{stem}.png'
        mask_rel = f'retinal/ground_truth/bad/{stem}_mask.png'
        _save_image(image_path, output_root / img_rel, args.size)
        _save_mask(mask, output_root / mask_rel)
        meta['test']['retinal'].append({
            'img_path': img_rel,
            'mask_path': mask_rel,
            'cls_name': 'retinal',
            'specie_name': 'bad',
            'anomaly': 1,
        })
        saved_bad += 1

    if meta['test']['retinal']:
        dummy = dict(meta['test']['retinal'][0])
        dummy['mask_path'] = ''
        dummy['anomaly'] = 0
        meta['train']['retinal'] = [dummy]

    meta_path = output_root / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Output: {output_root}')
    print(f'Meta: {meta_path}')
    print(f'Saved bad images: {saved_bad}')
    print(f'Skipped empty masks: {skipped_empty}')
    print('Note: this dataset has only abnormal images with masks, so use it for pixel metrics/frequency analysis.')


if __name__ == '__main__':
    main()
