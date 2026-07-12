import argparse
import csv
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


VOLUME_SUFFIXES = ('.nii', '.nii.gz', '.mgz', '.mgh', '.img')


def _lower_map(columns):
    return {str(col).strip().lower(): col for col in columns}


def _read_metadata(path):
    path = Path(path)
    if path.suffix.lower() in ['.xlsx', '.xls']:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError('Reading Excel metadata requires pandas/openpyxl.') from exc
        return pd.read_excel(path).to_dict('records')
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _find_column(rows, candidates):
    if not rows:
        return None
    columns = _lower_map(rows[0].keys())
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    for lower_name, original in columns.items():
        if any(candidate.lower() in lower_name for candidate in candidates):
            return original
    return None


def _as_float(value):
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normal_subject_ids(metadata_path):
    rows = _read_metadata(metadata_path)
    if not rows:
        raise RuntimeError(f'No metadata rows found in {metadata_path}')

    id_col = _find_column(rows, ['ID', 'Subject', 'OASISID', 'OASIS_ID', 'Subject ID'])
    cdr_col = _find_column(rows, ['CDR'])
    group_col = _find_column(rows, ['Group', 'Diagnosis', 'Dx', 'Label'])
    if id_col is None:
        raise RuntimeError('Could not find subject ID column in metadata.')
    if cdr_col is None and group_col is None:
        raise RuntimeError('Could not find CDR or diagnosis/group column in metadata.')

    subject_ids = set()
    for row in rows:
        sid = str(row.get(id_col, '')).strip()
        if not sid:
            continue
        is_normal = False
        if cdr_col is not None:
            cdr = _as_float(row.get(cdr_col))
            is_normal = cdr == 0.0
        if not is_normal and group_col is not None:
            group = str(row.get(group_col, '')).strip().lower()
            is_normal = any(token in group for token in ['nondemented', 'normal', 'control', 'healthy'])
        if is_normal:
            subject_ids.add(sid)
    if not subject_ids:
        raise RuntimeError('No normal subjects selected from metadata.')
    return subject_ids


def _canonical_subject_id(text):
    match = re.search(r'(OAS\d+[_-]\d+[_-]MR\d+|OAS\d+[_-]\d+)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace('-', '_')
    return None


def _volume_suffix(path):
    name = path.name.lower()
    if name.endswith('.nii.gz'):
        return '.nii.gz'
    return path.suffix.lower()


def _volume_priority(path):
    text = str(path).lower()
    if 'processed/mprage/t88_111' in text and 'masked_gfc.img' in text and 'fseg' not in text:
        return 0
    if 'processed/mprage/t88_111' in text and text.endswith('_gfc.img') and 'masked' not in text:
        return 1
    if 'processed/mprage/subj_111' in text and text.endswith('.img'):
        return 2
    if '/raw/' in text and text.endswith('.img'):
        return 3
    return 4


def _collect_volumes(raw_root, subject_ids, use_gif_preview=False):
    subject_ids_norm = {sid.replace('-', '_') for sid in subject_ids}
    volumes = []
    for path in Path(raw_root).rglob('*'):
        if not path.is_file():
            continue
        if use_gif_preview:
            if path.suffix.lower() != '.gif' or '_sag_' not in path.name.lower():
                continue
        elif _volume_suffix(path) not in VOLUME_SUFFIXES:
            continue
        sid = _canonical_subject_id(str(path))
        if sid is None:
            continue
        sid_short = '_'.join(sid.split('_')[:2])
        if sid in subject_ids_norm or sid_short in subject_ids_norm:
            volumes.append((sid, path))
    volumes.sort(key=lambda item: (_volume_priority(item[1]), str(item[1])))

    first_per_subject = {}
    for sid, path in volumes:
        sid_short = '_'.join(sid.split('_')[:2])
        first_per_subject.setdefault(sid_short, path)
    return sorted(first_per_subject.items())


def _save_gif_preview(path, dst_path, size):
    img = Image.open(path).convert('L')
    img = img.resize((size, size), Image.BILINEAR)
    img.save(dst_path)
    return True


def _load_volume(path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError('MRI volume conversion requires nibabel: pip install nibabel') from exc
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f'Unsupported volume shape {data.shape} for {path}')
    return data


def _normalize_slice(slice_2d, percentile_low=1.0, percentile_high=99.5):
    arr = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    arr = np.where(finite, arr, 0)
    foreground = arr > np.percentile(arr, 5)
    values = arr[foreground] if foreground.any() else arr.reshape(-1)
    lo = np.percentile(values, percentile_low)
    hi = np.percentile(values, percentile_high)
    if hi <= lo:
        return None
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def _slice_indices(volume, axis, slices_per_volume):
    moved = np.moveaxis(volume, axis, 0)
    nonzero = np.abs(moved) > np.percentile(np.abs(moved), 70)
    valid = np.where(nonzero.reshape(nonzero.shape[0], -1).mean(axis=1) > 0.01)[0]
    if len(valid) == 0:
        center = moved.shape[0] // 2
        span = max(1, slices_per_volume // 2)
        return list(range(max(0, center - span), min(moved.shape[0], center + span + 1)))[:slices_per_volume]
    lo, hi = int(valid.min()), int(valid.max())
    if slices_per_volume == 1:
        return [(lo + hi) // 2]
    return np.linspace(lo, hi, slices_per_volume + 2, dtype=int)[1:-1].tolist()


def _save_slice(volume, axis, idx, dst_path, size):
    moved = np.moveaxis(volume, axis, 0)
    arr = _normalize_slice(moved[idx])
    if arr is None:
        return False
    img = Image.fromarray(arr, mode='L')
    img = img.resize((size, size), Image.BILINEAR)
    img.save(dst_path)
    return True


def main():
    parser = argparse.ArgumentParser(description='Prepare normal OASIS MRI slices for FolderNormalAD training.')
    parser.add_argument('--metadata', required=True, help='OASIS metadata xlsx/csv with CDR or diagnosis column.')
    parser.add_argument('--raw-root', required=True, help='Directory containing extracted OASIS MRI volumes.')
    parser.add_argument(
        '--output',
        default='data/medical_aux_multisource_normal/train/good/oasis_brain',
        help='Output image directory.',
    )
    parser.add_argument('--max-subjects', type=int, default=300)
    parser.add_argument('--slices-per-volume', type=int, default=5)
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument(
        '--use-gif-preview',
        action='store_true',
        help='Use OASIS sagittal GIF previews instead of loading Analyze volumes. This is a fallback.',
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output = Path(args.output)
    if args.clear and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    subject_ids = _normal_subject_ids(args.metadata)
    volumes = _collect_volumes(args.raw_root, subject_ids, use_gif_preview=args.use_gif_preview)
    if not volumes:
        raise RuntimeError('No MRI volumes matched the normal metadata subjects.')
    if len(volumes) > args.max_subjects:
        volumes = rng.sample(volumes, args.max_subjects)

    rows = []
    saved = 0
    for subject_id, volume_path in volumes:
        try:
            if args.use_gif_preview:
                dst = output / f'{subject_id}_preview.png'
                if _save_gif_preview(volume_path, dst, args.size):
                    rows.append({
                        'subject_id': subject_id,
                        'volume_path': str(volume_path),
                        'slice_idx': -1,
                        'dst_path': str(dst),
                    })
                    saved += 1
                continue

            volume = _load_volume(volume_path)
            indices = _slice_indices(volume, args.axis, args.slices_per_volume)
            for slice_idx in indices:
                dst = output / f'{subject_id}_axis{args.axis}_slice{slice_idx:03d}.png'
                if _save_slice(volume, args.axis, slice_idx, dst, args.size):
                    rows.append({
                        'subject_id': subject_id,
                        'volume_path': str(volume_path),
                        'slice_idx': slice_idx,
                        'dst_path': str(dst),
                    })
                    saved += 1
        except Exception as exc:
            print(f'[skip] {volume_path}: {exc}')

    manifest = output.parent.parent.parent / 'oasis_brain_manifest.csv'
    with open(manifest, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject_id', 'volume_path', 'slice_idx', 'dst_path'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'Selected normal subjects from metadata: {len(subject_ids)}')
    print(f'Matched volumes: {len(volumes)}')
    print(f'Saved slices: {saved}')
    print(f'Output: {output}')
    print(f'Manifest: {manifest}')


if __name__ == '__main__':
    main()
