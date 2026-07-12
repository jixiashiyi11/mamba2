import argparse
import csv
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
    arr = np.clip((arr - window_low) / (window_high - window_low), 0, 1)
    return (arr * 255).astype(np.uint8)


def _save_slice(image_volume, slice_idx, dst_path, size):
    arr = _window_ct(image_volume[:, :, slice_idx])
    img = Image.fromarray(arr, mode='L').resize((size, size), Image.BILINEAR)
    img.save(dst_path)


def main():
    parser = argparse.ArgumentParser(
        description='Extract normal liver CT slices from MSD Task03 Liver for normal-only FolderNormalAD training.'
    )
    parser.add_argument('--task-root', default='data/raw/msd/Task03_Liver')
    parser.add_argument(
        '--output',
        default='data/medical_aux_multisource_normal/train/good/msd_liver',
    )
    parser.add_argument('--max-cases', type=int, default=0, help='0 means use all matched cases.')
    parser.add_argument('--max-slices', type=int, default=3000)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--min-liver-pixels', type=int, default=500)
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    output = Path(args.output)
    if args.clear and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    pairs = _image_label_pairs(args.task_root)
    if args.max_cases and len(pairs) > args.max_cases:
        pairs = pairs[:args.max_cases]

    rows = []
    saved = 0
    for image_path, label_path in pairs:
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
            if liver_pixels < args.min_liver_pixels or tumor_pixels > 0:
                continue
            dst = output / f'{case_id}_slice{slice_idx:03d}.png'
            _save_slice(image, slice_idx, dst, args.size)
            rows.append({
                'case_id': case_id,
                'image_path': str(image_path),
                'label_path': str(label_path),
                'slice_idx': slice_idx,
                'liver_pixels': liver_pixels,
                'tumor_pixels': tumor_pixels,
                'dst_path': str(dst),
            })
            saved += 1
            if args.max_slices and saved >= args.max_slices:
                break
        if args.max_slices and saved >= args.max_slices:
            break

    manifest = Path(args.output).parents[2] / 'msd_liver_normal_manifest.csv'
    with open(manifest, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'case_id',
                'image_path',
                'label_path',
                'slice_idx',
                'liver_pixels',
                'tumor_pixels',
                'dst_path',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'Matched cases: {len(pairs)}')
    print(f'Saved normal liver slices: {saved}')
    print(f'Output: {output}')
    print(f'Manifest: {manifest}')


if __name__ == '__main__':
    main()
