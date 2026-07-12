import argparse
import csv
import hashlib
import os
import random
import shutil
from pathlib import Path


IMG_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def is_image(path):
    return path.suffix.lower() in IMG_EXTENSIONS


def parse_source(value):
    parts = value.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            '--source must be formatted as name:path:count, for example oct:data/oct2017/train/NORMAL:5000'
        )
    name, path, count = parts
    name = name.strip()
    path = Path(path.strip())
    try:
        count = int(count)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'Invalid sample count in --source {value!r}.') from exc
    if not name:
        raise argparse.ArgumentTypeError('Source name must not be empty.')
    if count <= 0:
        raise argparse.ArgumentTypeError('Source count must be positive.')
    return name, path, count


def collect_images(root):
    if not root.exists():
        raise FileNotFoundError(f'Source directory does not exist: {root}')
    return sorted(path for path in root.rglob('*') if path.is_file() and is_image(path))


def safe_link_name(source_root, image_path):
    rel = image_path.relative_to(source_root)
    digest = hashlib.sha1(str(rel).encode('utf-8')).hexdigest()[:10]
    stem = '_'.join(rel.with_suffix('').parts)
    return f'{stem}_{digest}{image_path.suffix.lower()}'


def place_file(src, dst, mode):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    elif mode == 'symlink':
        os.symlink(os.path.relpath(src, dst.parent), dst)
    else:
        raise ValueError(f'Unsupported mode={mode}')


def main():
    parser = argparse.ArgumentParser(
        description='Build a balanced normal-only multi-source training folder for FolderNormalAD.'
    )
    parser.add_argument(
        '--output',
        default='data/medical_aux_multisource_normal',
        help='Output dataset root. Images are written under output/train/good/<source_name>/.',
    )
    parser.add_argument(
        '--source',
        action='append',
        required=True,
        type=parse_source,
        help='Normal image source as name:path:count. Repeat for multiple sources.',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=['symlink', 'copy', 'hardlink'], default='symlink')
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Delete output/train/good before rebuilding.',
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = Path(args.output)
    good_root = output_root / 'train' / 'good'
    if args.clear and good_root.exists():
        shutil.rmtree(good_root)
    good_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    summary_rows = []
    for source_name, source_root, sample_count in args.source:
        images = collect_images(source_root)
        if not images:
            raise RuntimeError(f'No images found under {source_root}')
        chosen = images if len(images) <= sample_count else rng.sample(images, sample_count)
        dst_dir = good_root / source_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for image_path in chosen:
            dst_path = dst_dir / safe_link_name(source_root, image_path)
            place_file(image_path.resolve(), dst_path, args.mode)
            manifest_rows.append({
                'source': source_name,
                'src_path': str(image_path),
                'dst_path': str(dst_path),
                'mode': args.mode,
            })
        summary_rows.append({
            'source': source_name,
            'source_root': str(source_root),
            'available_images': len(images),
            'selected_images': len(chosen),
        })

    with open(output_root / 'manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'src_path', 'dst_path', 'mode'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(output_root / 'summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'source_root', 'available_images', 'selected_images'])
        writer.writeheader()
        writer.writerows(summary_rows)

    total = sum(row['selected_images'] for row in summary_rows)
    print(f'Wrote {total} normal images to {good_root}')
    for row in summary_rows:
        print(
            f"{row['source']}: selected={row['selected_images']} "
            f"available={row['available_images']} root={row['source_root']}"
        )


if __name__ == '__main__':
    main()

