import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def normalize_map(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)


def load_image(data_root, rel_path, size):
    if not rel_path:
        return np.ones((size[0], size[1], 3), dtype=np.float32)
    path = Path(data_root) / str(rel_path)
    if not path.exists():
        return np.ones((size[0], size[1], 3), dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def resize_map(x, size):
    image = Image.fromarray(normalize_map(x))
    image = image.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def colorize_jet(x):
    x = normalize_map(x)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def to_uint8(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def add_title(image, title, height=28):
    image = Image.fromarray(to_uint8(image))
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), str(title), fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def text_panel(lines, size):
    width, height = size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 8
    for line in lines:
        draw.text((8, y), str(line), fill=(0, 0, 0), font=ImageFont.load_default())
        y += 14
    return canvas


def hist_panel(values, size, title):
    width, height = size
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    hist, _ = np.histogram(values, bins=48)
    hist = hist.astype(np.float32) / max(float(hist.max()), 1.0)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(0, 0, 0), font=ImageFont.load_default())
    left, top, right, bottom = 8, 28, width - 8, height - 8
    bar_w = max(1, (right - left) // len(hist))
    for i, value in enumerate(hist):
        x0 = left + i * bar_w
        x1 = min(right, x0 + bar_w)
        y0 = bottom - int(value * (bottom - top))
        draw.rectangle([x0, y0, x1, bottom], fill=(210, 70, 55))
    draw.rectangle([left, top, right, bottom], outline=(0, 0, 0))
    return canvas


def make_grid(panels, cols):
    widths = [panel.width for panel in panels]
    heights = [panel.height for panel in panels]
    cell_w, cell_h = max(widths), max(heights)
    rows = int(np.ceil(len(panels) / float(cols)))
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        canvas.paste(panel, (col * cell_w, row * cell_h))
    return canvas


def score_array(data, key, fallback):
    if key in data.files:
        return np.asarray(data[key]).reshape(-1)
    return fallback


def optional_array(data, key, fallback=None):
    if key in data.files:
        return np.asarray(data[key])
    return fallback


def _ranked_indices(indices, image_scores, red_ratio, count):
    if len(indices) == 0:
        return []
    score_order = indices[np.argsort(-image_scores[indices])]
    red_order = indices[np.argsort(-red_ratio[indices])]
    low_score_order = indices[np.argsort(image_scores[indices])]
    selected = []
    for group in (score_order, red_order, low_score_order):
        for idx in group:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= count:
                return selected
    return selected[:count]


def pick_indices(labels, final_maps, image_scores, count, label_filter=None):
    labels = np.asarray(labels).reshape(-1).astype(int)
    final_maps = np.asarray(final_maps)
    image_scores = np.asarray(image_scores).reshape(-1)
    flat = np.stack([normalize_map(x).reshape(-1) for x in final_maps], axis=0)
    red_ratio = (flat > 0.6).mean(axis=1)

    if label_filter is not None:
        candidates = np.where(labels == int(label_filter))[0]
        return _ranked_indices(candidates, image_scores, red_ratio, count)

    per_label = max(1, count // 2)
    selected = []
    groups = [
        _ranked_indices(np.where(labels == 0)[0], image_scores, red_ratio, per_label),
        _ranked_indices(np.where(labels == 1)[0], image_scores, red_ratio, count - per_label),
        np.argsort(-red_ratio)[:count],
        np.argsort(image_scores)[:count],
    ]
    for group in groups:
        for idx in group:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= count:
                return selected
    return selected[:count]


def map_stats(x):
    x = np.asarray(x, dtype=np.float32)
    xn = normalize_map(x)
    flat = x.reshape(-1)
    k1 = max(1, int(flat.size * 0.01))
    k5 = max(1, int(flat.size * 0.05))
    return {
        "raw_min": float(flat.min()),
        "raw_max": float(flat.max()),
        "raw_mean": float(flat.mean()),
        "raw_std": float(flat.std()),
        "top1_mean": float(np.partition(flat, -k1)[-k1:].mean()),
        "top5_mean": float(np.partition(flat, -k5)[-k5:].mean()),
        "red_ratio_06": float((xn > 0.6).mean()),
        "red_ratio_08": float((xn > 0.8).mean()),
    }


def draw_sample(data, idx, data_root, out_dir):
    final_map = np.asarray(data["anomaly_maps"])[idx]
    raw_map = optional_array(data, "raw_anomaly_maps", np.asarray(data["anomaly_maps"]))[idx]
    mask = np.asarray(data["imgs_masks"])[idx]
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    img_paths = optional_array(data, "img_paths", np.array([""] * len(data["anomaly_maps"])))
    image = load_image(data_root, img_paths[idx], final_map.shape)
    raw_norm = normalize_map(raw_map)
    final_norm = normalize_map(final_map)
    mask_norm = normalize_map(mask)
    overlay = np.clip(0.55 * image + 0.45 * colorize_jet(final_norm), 0.0, 1.0)

    optional_maps = []
    if "arcc_cal_maps" in data.files:
        optional_maps.append(("G_cal", resize_map(np.asarray(data["arcc_cal_maps"])[idx], final_map.shape)))
    if "mamba_prior_maps" in data.files:
        optional_maps.append(("G_mamba", resize_map(np.asarray(data["mamba_prior_maps"])[idx], final_map.shape)))

    cols = 5 + len(optional_maps)
    cls_name = str(np.asarray(data["cls_names"])[idx])
    label = int(np.asarray(data["anomalys"])[idx])
    score = float(np.asarray(data["image_scores"]).reshape(-1)[idx])
    score_max = float(score_array(data, "image_scores_max", data["image_scores"])[idx])
    score_top1 = float(score_array(data, "image_scores_top1", data["image_scores"])[idx])
    score_top5 = float(score_array(data, "image_scores_top5", data["image_scores"])[idx])
    stats = map_stats(final_map)

    panels = [
        ("image", image, None),
        ("mask", mask_norm, "gray"),
        ("A_raw", colorize_jet(raw_norm), None),
        ("A_final", colorize_jet(final_norm), None),
        ("overlay", overlay, None),
    ]
    for name, value in optional_maps:
        panels.append((name, colorize_jet(value), None))

    title_lines = [
        f"idx={idx} cls={cls_name} label={label}",
        f"score={score:.3f} max={score_max:.3f}",
        f"top1={score_top1:.3f} top5={score_top5:.3f}",
        f"red>0.6={stats['red_ratio_06']:.2f}",
        f"red>0.8={stats['red_ratio_08']:.2f}",
    ]
    panel_images = [add_title(value, title) for title, value, _ in panels]
    panel_size = (final_map.shape[1], final_map.shape[0] + 28)
    panel_images.extend(
        [
            hist_panel(final_map, panel_size, "A_final raw hist"),
            hist_panel(final_norm, panel_size, "A_final norm hist"),
            add_title((final_norm > 0.6).astype(np.float32), "red area > 0.6"),
            add_title((final_norm > 0.8).astype(np.float32), "red area > 0.8"),
            text_panel(title_lines + [f"{key}: {value:.4f}" for key, value in stats.items()], panel_size),
        ]
    )

    grid = make_grid(panel_images, cols=cols)
    out_path = out_dir / f"sample_{idx:04d}_label{label}_score{score:.3f}.png"
    grid.save(out_path)
    return out_path, stats


def main():
    parser = argparse.ArgumentParser(description="Visualize CLIP AD heatmaps from saved NPZ outputs.")
    parser.add_argument("--npz", required=True, help="Path to clip_ad_*_outputs.npz.")
    parser.add_argument("--data-root", default="data/mvtec", help="Dataset root used to resolve img_paths.")
    parser.add_argument("--out-dir", default="", help="Output directory for PNG files.")
    parser.add_argument("--count", type=int, default=8, help="Number of samples to visualize.")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="Only visualize one label.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent / "heatmap_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    labels = np.asarray(data["anomalys"]).reshape(-1)
    final_maps = np.asarray(data["anomaly_maps"])
    image_scores = np.asarray(data["image_scores"]).reshape(-1)
    indices = pick_indices(labels, final_maps, image_scores, args.count, label_filter=args.label)

    csv_path = out_dir / "heatmap_diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        writer = None
        for idx in indices:
            out_path, stats = draw_sample(data, idx, args.data_root, out_dir)
            row = {
                "idx": idx,
                "png": str(out_path),
                "cls_name": str(np.asarray(data["cls_names"])[idx]),
                "label": int(labels[idx]),
                "image_score": float(image_scores[idx]),
                "image_score_max": float(score_array(data, "image_scores_max", image_scores)[idx]),
                "image_score_top1": float(score_array(data, "image_scores_top1", image_scores)[idx]),
                "image_score_top5": float(score_array(data, "image_scores_top5", image_scores)[idx]),
                **stats,
            }
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            print(out_path)
    print(f"Wrote diagnostics: {csv_path}")


if __name__ == "__main__":
    main()
