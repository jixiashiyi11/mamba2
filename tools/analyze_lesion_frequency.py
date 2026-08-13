import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def _load_meta_samples(root):
    root = Path(root)
    meta_path = root / 'meta.json'
    if not meta_path.exists():
        raise RuntimeError(f'meta.json not found: {meta_path}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    samples = []
    for cls_name, cls_samples in meta.get('test', {}).items():
        for sample in cls_samples:
            if int(sample.get('anomaly', 0)) != 1:
                continue
            mask_path = sample.get('mask_path', '')
            if not mask_path:
                continue
            samples.append((root, sample))
    return samples


def _load_gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def _load_mask(path, shape):
    mask = Image.open(path).convert('L')
    if mask.size != (shape[1], shape[0]):
        mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def _bbox_from_mask(mask, margin_ratio):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = y1 - y0
    w = x1 - x0
    margin = int(round(max(h, w) * margin_ratio))
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mask.shape[0], y1 + margin)
    x1 = min(mask.shape[1], x1 + margin)
    return y0, y1, x0, x1


def _hann2d(h, w):
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _radial_frequency_grid(h, w):
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    radius = np.sqrt(xx * xx + yy * yy)
    max_radius = float(radius.max()) if radius.size else 1.0
    return radius / max(max_radius, 1e-8)


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or b.size < 2:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom <= 1e-12:
        return np.nan
    return float(np.sum(a * b) / denom)


def _band_energy(signal, low_cut, mid_cut):
    h, w = signal.shape
    windowed = signal * _hann2d(h, w)
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    energy = np.abs(spectrum) ** 2
    radius = _radial_frequency_grid(h, w)

    low_mask = radius <= low_cut
    mid_mask = (radius > low_cut) & (radius <= mid_cut)
    high_mask = radius > mid_cut
    total = float(energy.sum()) + 1e-12
    return {
        'low_energy_ratio': float(energy[low_mask].sum() / total),
        'mid_energy_ratio': float(energy[mid_mask].sum() / total),
        'high_energy_ratio': float(energy[high_mask].sum() / total),
    }


def _highpass_retention(crop, lesion_mask, cutoffs):
    h, w = crop.shape
    centered = crop - float(crop.mean())
    window = _hann2d(h, w)
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    radius = _radial_frequency_grid(h, w)
    lesion_values = centered[lesion_mask]
    lesion_energy = float(np.sum(lesion_values * lesion_values)) + 1e-12
    rows = []
    for cutoff in cutoffs:
        hp = spectrum.copy()
        hp[radius < cutoff] = 0
        recon = np.real(np.fft.ifft2(np.fft.ifftshift(hp)))
        recon_values = recon[lesion_mask]
        rows.append({
            'cutoff': cutoff,
            'pearson': _pearson(lesion_values, recon_values),
            'energy_retention': float(np.sum(recon_values * recon_values) / lesion_energy),
        })
    return rows


def _safe_sample_id(sample):
    stem = Path(sample.get('img_path', 'sample')).stem
    return stem.replace(',', '_')


def _analyze_sample(root, sample, args, cutoffs):
    image = _load_gray(root / sample['img_path'])
    mask = _load_mask(root / sample['mask_path'], image.shape)
    bbox = _bbox_from_mask(mask, args.margin_ratio)
    if bbox is None:
        return None
    y0, y1, x0, x1 = bbox
    crop = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    if crop.shape[0] < args.min_crop_size or crop.shape[1] < args.min_crop_size:
        return None

    lesion_signal = (crop - float(crop[crop_mask].mean())) * crop_mask.astype(np.float32)
    band = _band_energy(lesion_signal, args.low_cut, args.mid_cut)
    retention = _highpass_retention(crop, crop_mask, cutoffs)
    base = {
        'dataset_root': str(root),
        'organ': sample.get('cls_name', ''),
        'sample_id': _safe_sample_id(sample),
        'img_path': sample.get('img_path', ''),
        'mask_path': sample.get('mask_path', ''),
        'crop_h': crop.shape[0],
        'crop_w': crop.shape[1],
        'lesion_pixels': int(crop_mask.sum()),
    }
    base.update(band)
    return base, retention


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean(values):
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else np.nan


def _summarize_band(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['organ']].append(row)
    summary = []
    for organ, items in sorted(grouped.items()):
        summary.append({
            'organ': organ,
            'n': len(items),
            'low_energy_ratio': _mean([float(x['low_energy_ratio']) for x in items]),
            'mid_energy_ratio': _mean([float(x['mid_energy_ratio']) for x in items]),
            'high_energy_ratio': _mean([float(x['high_energy_ratio']) for x in items]),
        })
    return summary


def _summarize_retention(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['organ'], float(row['cutoff']))].append(row)
    summary = []
    for (organ, cutoff), items in sorted(grouped.items()):
        summary.append({
            'organ': organ,
            'cutoff': cutoff,
            'n': len(items),
            'pearson': _mean([float(x['pearson']) for x in items]),
            'energy_retention': _mean([float(x['energy_retention']) for x in items]),
        })
    return summary


def _plot_outputs(output_dir, band_summary, retention_summary):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('[warn] matplotlib not installed; skip plots')
        return

    if band_summary:
        organs = [x['organ'] for x in band_summary]
        low = [x['low_energy_ratio'] for x in band_summary]
        mid = [x['mid_energy_ratio'] for x in band_summary]
        high = [x['high_energy_ratio'] for x in band_summary]
        x = np.arange(len(organs))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, low, label='low')
        ax.bar(x, mid, bottom=low, label='mid')
        ax.bar(x, high, bottom=np.asarray(low) + np.asarray(mid), label='high')
        ax.set_xticks(x)
        ax.set_xticklabels(organs)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Lesion frequency energy ratio')
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / 'organ_band_energy_bar.png', dpi=200)
        plt.close(fig)

    if retention_summary:
        by_organ = defaultdict(list)
        for row in retention_summary:
            by_organ[row['organ']].append(row)
        fig, ax = plt.subplots(figsize=(7, 4))
        for organ, items in sorted(by_organ.items()):
            items = sorted(items, key=lambda x: x['cutoff'])
            ax.plot(
                [x['cutoff'] for x in items],
                [x['pearson'] for x in items],
                marker='o',
                label=organ,
            )
        ax.set_xlabel('High-pass cutoff ratio')
        ax.set_ylabel('Masked lesion Pearson retention')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / 'organ_highpass_curve.png', dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze lesion frequency distribution from ADer-style masked medical test sets.'
    )
    parser.add_argument(
        '--roots',
        nargs='+',
        required=True,
        help='Dataset roots containing meta.json, e.g. data/medical_standard_msd_liver data/medical_standard_brats',
    )
    parser.add_argument('--output-dir', default='outputs/lesion_frequency_analysis')
    parser.add_argument('--margin-ratio', type=float, default=0.5)
    parser.add_argument('--min-crop-size', type=int, default=16)
    parser.add_argument('--low-cut', type=float, default=0.15)
    parser.add_argument('--mid-cut', type=float, default=0.35)
    parser.add_argument('--cutoffs', default='0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50')
    parser.add_argument('--max-samples-per-organ', type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs = [float(x) for x in args.cutoffs.split(',') if x.strip()]

    samples_by_organ = defaultdict(list)
    for root in args.roots:
        for root_path, sample in _load_meta_samples(root):
            samples_by_organ[sample.get('cls_name', '')].append((root_path, sample))

    band_rows = []
    retention_rows = []
    for organ, samples in sorted(samples_by_organ.items()):
        for root, sample in samples[:args.max_samples_per_organ]:
            result = _analyze_sample(root, sample, args, cutoffs)
            if result is None:
                continue
            band, retention = result
            band_rows.append(band)
            for item in retention:
                row = dict(band)
                row.update(item)
                retention_rows.append(row)

    band_fields = [
        'dataset_root', 'organ', 'sample_id', 'img_path', 'mask_path',
        'crop_h', 'crop_w', 'lesion_pixels',
        'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio',
    ]
    retention_fields = band_fields + ['cutoff', 'pearson', 'energy_retention']
    band_summary = _summarize_band(band_rows)
    retention_summary = _summarize_retention(retention_rows)

    _write_csv(output_dir / 'frequency_band_energy.csv', band_rows, band_fields)
    _write_csv(output_dir / 'highpass_retention_curve.csv', retention_rows, retention_fields)
    _write_csv(output_dir / 'frequency_band_energy_summary.csv', band_summary, [
        'organ', 'n', 'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio',
    ])
    _write_csv(output_dir / 'highpass_retention_curve_summary.csv', retention_summary, [
        'organ', 'cutoff', 'n', 'pearson', 'energy_retention',
    ])
    _plot_outputs(output_dir, band_summary, retention_summary)

    print(f'Output: {output_dir}')
    print(f'Band rows: {len(band_rows)}')
    print(f'High-pass rows: {len(retention_rows)}')
    print('Summary:')
    for row in band_summary:
        print(
            f"{row['organ']}: n={row['n']} "
            f"low={row['low_energy_ratio']:.4f} "
            f"mid={row['mid_energy_ratio']:.4f} "
            f"high={row['high_energy_ratio']:.4f}"
        )


if __name__ == '__main__':
    main()
