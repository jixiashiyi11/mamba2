import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _load_nifti(path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError('NIfTI conversion requires nibabel: pip install nibabel') from exc
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def _image_label_pairs(task_root):
    image_root = Path(task_root) / 'imagesTr'
    label_root = Path(task_root) / 'labelsTr'
    pairs = []
    for image_path in sorted(image_root.glob('*.nii.gz')):
        if image_path.name.startswith('._'):
            continue
        label_path = label_root / image_path.name
        if label_path.exists() and not label_path.name.startswith('._'):
            pairs.append((image_path, label_path))
    if not pairs:
        raise RuntimeError(f'No image/label pairs found under {task_root}')
    return pairs


def _window_ct(slice_2d, window_low=-150.0, window_high=250.0):
    arr = np.asarray(slice_2d, dtype=np.float32)
    arr = np.clip((arr - window_low) / (window_high - window_low), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _save_image_slice(image_volume, slice_idx, dst_path, size):
    arr = _window_ct(image_volume[:, :, slice_idx])
    Image.fromarray(arr, mode='L').resize((size, size), Image.BILINEAR).save(dst_path)


def _save_mask_slice(label_volume, slice_idx, dst_path, size):
    mask = (label_volume[:, :, slice_idx] == 2).astype(np.uint8) * 255
    Image.fromarray(mask, mode='L').resize((size, size), Image.NEAREST).save(dst_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert MSD Task03 Liver into ADer-style liver anomaly test set.'
    )
    parser.add_argument('--task-root', default='data/raw/msd/Task03_Liver')
    parser.add_argument('--output-root', default='data/medical_standard_msd_liver')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--min-liver-pixels', type=int, default=500)
    parser.add_argument('--min-tumor-pixels', type=int, default=20)
    parser.add_argument('--max-good', type=int, default=1200)
    parser.add_argument('--max-bad', type=int, default=1200)
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.clear and output_root.exists():
        shutil.rmtree(output_root)

    good_dir = output_root / 'liver' / 'test' / 'good'
    bad_dir = output_root / 'liver' / 'test' / 'bad'
    mask_dir = output_root / 'liver' / 'ground_truth' / 'bad'
    for directory in [good_dir, bad_dir, mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    meta = {'train': {'liver': []}, 'test': {'liver': []}}
    saved_good = 0
    saved_bad = 0

    for image_path, label_path in _image_label_pairs(args.task_root):
        image = _load_nifti(image_path)
        label = _load_nifti(label_path)
        if image.shape != label.shape:
            print(f'[skip] shape mismatch: {image_path.name} image={image.shape} label={label.shape}')
            continue

        case_id = image_path.name.replace('.nii.gz', '')
        for slice_idx in range(image.shape[2]):
            seg = label[:, :, slice_idx]
            liver_pixels = int((seg == 1).sum())
            tumor_pixels = int((seg == 2).sum())
            if liver_pixels < args.min_liver_pixels and tumor_pixels < args.min_tumor_pixels:
                continue

            stem = f'{case_id}_slice{slice_idx:03d}'
            if tumor_pixels >= args.min_tumor_pixels and saved_bad < args.max_bad:
                img_rel = f'liver/test/bad/{stem}.png'
                mask_rel = f'liver/ground_truth/bad/{stem}_mask.png'
                _save_image_slice(image, slice_idx, output_root / img_rel, args.size)
                _save_mask_slice(label, slice_idx, output_root / mask_rel, args.size)
                meta['test']['liver'].append({
                    'img_path': img_rel,
                    'mask_path': mask_rel,
                    'cls_name': 'liver',
                    'specie_name': 'bad',
                    'anomaly': 1,
                })
                saved_bad += 1
            elif tumor_pixels == 0 and liver_pixels >= args.min_liver_pixels and saved_good < args.max_good:
                img_rel = f'liver/test/good/{stem}.png'
                _save_image_slice(image, slice_idx, output_root / img_rel, args.size)
                meta['test']['liver'].append({
                    'img_path': img_rel,
                    'mask_path': '',
                    'cls_name': 'liver',
                    'specie_name': 'good',
                    'anomaly': 0,
                })
                saved_good += 1

            if saved_good >= args.max_good and saved_bad >= args.max_bad:
                break
        if saved_good >= args.max_good and saved_bad >= args.max_bad:
            break

    meta_path = output_root / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Output: {output_root}')
    print(f'Meta: {meta_path}')
    print(f'Saved good slices: {saved_good}')
    print(f'Saved bad slices: {saved_bad}')


if __name__ == '__main__':
    main()
