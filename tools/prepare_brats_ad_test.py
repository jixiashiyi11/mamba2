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


def _strip_nii_suffix(path):
    name = path.name
    for suffix in ['.nii.gz', '.nii']:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def _case_id_from_seg(seg_path):
    stem = _strip_nii_suffix(seg_path)
    for suffix in ['_seg', '-seg', '.seg']:
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem.replace('seg', '').rstrip('_-.')


def _find_cases(root, modality):
    root = Path(root)
    seg_paths = sorted(
        p for p in root.rglob('*')
        if p.is_file()
        and not p.name.startswith('._')
        and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))
        and '_seg' in p.name.lower()
    )
    cases = []
    for seg_path in seg_paths:
        case_id = _case_id_from_seg(seg_path)
        candidates = []
        for ext in ['.nii.gz', '.nii']:
            candidates.extend([
                seg_path.with_name(f'{case_id}_{modality}{ext}'),
                seg_path.with_name(f'{case_id}-{modality}{ext}'),
            ])
        image_path = next((p for p in candidates if p.exists()), None)
        if image_path is None:
            image_matches = [
                p for p in seg_path.parent.iterdir()
                if p.is_file()
                and not p.name.startswith('._')
                and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))
                and modality.lower() in p.name.lower()
                and '_seg' not in p.name.lower()
            ]
            image_path = sorted(image_matches)[0] if image_matches else None
        if image_path is not None:
            cases.append((case_id, image_path, seg_path))
    if not cases:
        raise RuntimeError(f'No BraTS cases found under {root} for modality={modality}')
    return cases


def _normalize_mri_slice(slice_2d, low_percentile=1.0, high_percentile=99.0):
    arr = np.asarray(slice_2d, dtype=np.float32)
    nonzero = arr[np.abs(arr) > 1e-6]
    if nonzero.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    low = float(np.percentile(nonzero, low_percentile))
    high = float(np.percentile(nonzero, high_percentile))
    if high <= low:
        high = low + 1.0
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _save_image_slice(image_volume, slice_idx, dst_path, size):
    arr = _normalize_mri_slice(image_volume[:, :, slice_idx])
    Image.fromarray(arr, mode='L').resize((size, size), Image.BILINEAR).save(dst_path)


def _save_mask_slice(seg_volume, slice_idx, dst_path, size):
    mask = (seg_volume[:, :, slice_idx] > 0).astype(np.uint8) * 255
    Image.fromarray(mask, mode='L').resize((size, size), Image.NEAREST).save(dst_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert BraTS into ADer-style brain anomaly test set.'
    )
    parser.add_argument('--raw-root', default='data/raw/brats')
    parser.add_argument('--output-root', default='data/medical_standard_brats')
    parser.add_argument('--modality', default='flair', choices=['flair', 't1', 't1ce', 't2'])
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--min-brain-pixels', type=int, default=500)
    parser.add_argument('--min-tumor-pixels', type=int, default=20)
    parser.add_argument('--max-good', type=int, default=1200)
    parser.add_argument('--max-bad', type=int, default=1200)
    parser.add_argument('--clear', action='store_true')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.clear and output_root.exists():
        shutil.rmtree(output_root)

    good_dir = output_root / 'brain' / 'test' / 'good'
    bad_dir = output_root / 'brain' / 'test' / 'bad'
    mask_dir = output_root / 'brain' / 'ground_truth' / 'bad'
    for directory in [good_dir, bad_dir, mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    meta = {'train': {'brain': []}, 'test': {'brain': []}}
    saved_good = 0
    saved_bad = 0

    for case_id, image_path, seg_path in _find_cases(args.raw_root, args.modality):
        image = _load_nifti(image_path)
        seg = _load_nifti(seg_path)
        if image.shape != seg.shape:
            print(f'[skip] shape mismatch: {case_id} image={image.shape} seg={seg.shape}')
            continue

        for slice_idx in range(image.shape[2]):
            img_slice = image[:, :, slice_idx]
            seg_slice = seg[:, :, slice_idx]
            brain_pixels = int((np.abs(img_slice) > 1e-6).sum())
            tumor_pixels = int((seg_slice > 0).sum())
            if brain_pixels < args.min_brain_pixels:
                continue

            stem = f'{case_id}_slice{slice_idx:03d}'
            if tumor_pixels >= args.min_tumor_pixels and saved_bad < args.max_bad:
                img_rel = f'brain/test/bad/{stem}.png'
                mask_rel = f'brain/ground_truth/bad/{stem}_mask.png'
                _save_image_slice(image, slice_idx, output_root / img_rel, args.size)
                _save_mask_slice(seg, slice_idx, output_root / mask_rel, args.size)
                meta['test']['brain'].append({
                    'img_path': img_rel,
                    'mask_path': mask_rel,
                    'cls_name': 'brain',
                    'specie_name': 'bad',
                    'anomaly': 1,
                })
                saved_bad += 1
            elif tumor_pixels == 0 and saved_good < args.max_good:
                img_rel = f'brain/test/good/{stem}.png'
                _save_image_slice(image, slice_idx, output_root / img_rel, args.size)
                meta['test']['brain'].append({
                    'img_path': img_rel,
                    'mask_path': '',
                    'cls_name': 'brain',
                    'specie_name': 'good',
                    'anomaly': 0,
                })
                saved_good += 1

            if saved_good >= args.max_good and saved_bad >= args.max_bad:
                break
        if saved_good >= args.max_good and saved_bad >= args.max_bad:
            break

    if meta['test']['brain']:
        dummy = dict(next(s for s in meta['test']['brain'] if int(s.get('anomaly', 0)) == 0))
        dummy['mask_path'] = ''
        dummy['anomaly'] = 0
        meta['train']['brain'] = [dummy]

    meta_path = output_root / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Output: {output_root}')
    print(f'Meta: {meta_path}')
    print(f'Saved good slices: {saved_good}')
    print(f'Saved bad slices: {saved_bad}')
    print(f'Modality: {args.modality}')


if __name__ == '__main__':
    main()
