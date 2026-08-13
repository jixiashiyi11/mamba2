import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTENSIONS = ('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')
CATEGORIES = ('normal', 'benign', 'malignant')


def _is_image(path):
    return path.suffix.lower() in IMG_EXTENSIONS


def _category_from_path(path):
    lowered_parts = [part.lower() for part in path.parts]
    for category in CATEGORIES:
        if category in lowered_parts:
            return category
    stem = path.stem.lower()
    for category in CATEGORIES:
        if stem.startswith(category):
            return category
    return None


def _is_mask(path):
    return bool(re.search(r'(^|[_\s-])mask($|[_\s-]|\d)', path.stem.lower()))


def _image_key(path):
    stem = path.stem
    stem = re.sub(r'[_\s-]*mask[_\s-]*\d*$', '', stem, flags=re.IGNORECASE)
    stem = re.sub(r'[_\s-]*mask$', '', stem, flags=re.IGNORECASE)
    return stem


def _collect_busi_files(raw_root):
    raw_root = Path(raw_root)
    images = {}
    masks = {}
    for path in sorted(p for p in raw_root.rglob('*') if p.is_file() and _is_image(p)):
        category = _category_from_path(path)
        if category is None:
            continue
        key = (category, _image_key(path))
        if _is_mask(path):
            masks.setdefault(key, []).append(path)
        else:
            images[key] = path
    if not images:
        raise RuntimeError(f'No BUSI images found under {raw_root}')
    return images, masks


def _save_image(src_path, dst_path, size):
    image = Image.open(src_path).convert('RGB').resize((size, size), Image.BILINEAR)
    image.save(dst_path)


def _merged_mask(mask_paths, size):
    merged = np.zeros((size, size), dtype=np.uint8)
    for mask_path in mask_paths:
        mask = Image.open(mask_path).convert('L').resize((size, size), Image.NEAREST)
        merged = np.maximum(merged, (np.asarray(mask) > 0).astype(np.uint8))
    return merged


def _save_mask(mask, dst_path):
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode='L').save(dst_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert BUSI breast ultrasound images into ADer-style breast anomaly test set.'
    )
    parser.add_argument('--raw-root', default='data/raw/busi')
    parser.add_argument('--output-root', default='data/medical_standard_busi')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--min-mask-pixels', type=int, default=10)
    parser.add_argument('--max-good', type=int, default=100000)
    parser.add_argument('--max-bad', type=int, default=100000)
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.clear and output_root.exists():
        shutil.rmtree(output_root)

    good_dir = output_root / 'breast' / 'test' / 'good'
    bad_dir = output_root / 'breast' / 'test' / 'bad'
    mask_dir = output_root / 'breast' / 'ground_truth' / 'bad'
    for directory in [good_dir, bad_dir, mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    images, masks = _collect_busi_files(args.raw_root)
    meta = {'train': {'breast': []}, 'test': {'breast': []}}
    saved_good = 0
    saved_bad = 0
    skipped_bad_without_mask = 0
    skipped_small_mask = 0

    for (category, key), image_path in sorted(images.items(), key=lambda item: (item[0][0], item[0][1])):
        safe_key = re.sub(r'[^A-Za-z0-9_.-]+', '_', key).strip('_')
        if category == 'normal':
            if saved_good >= args.max_good:
                continue
            img_rel = f'breast/test/good/{safe_key}.png'
            _save_image(image_path, output_root / img_rel, args.size)
            sample = {
                'img_path': img_rel,
                'mask_path': '',
                'cls_name': 'breast',
                'specie_name': 'good',
                'anomaly': 0,
            }
            meta['test']['breast'].append(sample)
            if not meta['train']['breast']:
                meta['train']['breast'].append(dict(sample))
            saved_good += 1
            continue

        if saved_bad >= args.max_bad:
            continue
        mask_paths = masks.get((category, key), [])
        if not mask_paths:
            skipped_bad_without_mask += 1
            continue
        mask = _merged_mask(mask_paths, args.size)
        mask_pixels = int(mask.sum())
        if mask_pixels < args.min_mask_pixels:
            skipped_small_mask += 1
            continue

        stem = f'{category}_{safe_key}'
        img_rel = f'breast/test/bad/{stem}.png'
        mask_rel = f'breast/ground_truth/bad/{stem}_mask.png'
        _save_image(image_path, output_root / img_rel, args.size)
        _save_mask(mask, output_root / mask_rel)
        meta['test']['breast'].append({
            'img_path': img_rel,
            'mask_path': mask_rel,
            'cls_name': 'breast',
            'specie_name': category,
            'anomaly': 1,
        })
        saved_bad += 1

    if not meta['train']['breast'] and meta['test']['breast']:
        dummy = dict(meta['test']['breast'][0])
        dummy['mask_path'] = ''
        dummy['anomaly'] = 0
        dummy['specie_name'] = 'good'
        meta['train']['breast'] = [dummy]

    meta_path = output_root / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Output: {output_root}')
    print(f'Meta: {meta_path}')
    print(f'Saved good images: {saved_good}')
    print(f'Saved bad images: {saved_bad}')
    print(f'Skipped bad images without mask: {skipped_bad_without_mask}')
    print(f'Skipped bad images with tiny mask: {skipped_small_mask}')


if __name__ == '__main__':
    main()
