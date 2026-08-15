#!/usr/bin/env python3
"""Offline ARCC ablation using one saved CLIP-AD outputs.npz.

This script does not retrain or mutate a checkpoint. It compares:

1. raw: the anomaly map before ARCC;
2. bidirectional: the currently saved ARCC output;
3. suppress_only: min(raw, bidirectional), which keeps only logit decreases.

Image scores are rebuilt as S_global + beta * max(A_variant). S_global is
recovered from the saved bidirectional image score, so all variants use the
same global CLIP score and differ only in their anomaly map.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from tabulate import tabulate


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.metric import Evaluator


METRIC_NAMES = (
    "image_AUROC",
    "image_AP",
    "image_F1",
    "AUPRO",
    "pixel_AUROC",
    "pixel_AP",
    "pixel_F1",
)


def _maps(array, name):
    values = np.asarray(array)
    if values.ndim == 4 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 3:
        raise ValueError(f"{name} must have shape [N,H,W] or [N,1,H,W], got {values.shape}.")
    return values.astype(np.float32, copy=False)


def _best_f1(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    denom = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    return float(np.nanmax(f1))


def _pixel_f1(masks, maps, eps=1e-8):
    normalized = (maps - maps.min()) / (maps.max() - maps.min() + eps)
    target = masks.astype(bool)
    scores = []
    for threshold in np.arange(0.0, 1.0 + 1e-3, 0.05):
        prediction = normalized > threshold
        intersection = np.logical_and(target, prediction).sum()
        predicted = prediction.sum()
        positive = target.sum()
        precision = intersection / (predicted + eps)
        recall = intersection / (positive + eps)
        scores.append(2.0 * precision * recall / (precision + recall + eps))
    return float(max(scores))


def _class_metrics(labels, masks, maps, image_scores, class_name, max_step_aupro):
    if np.unique(labels).size < 2:
        raise ValueError(f"Class {class_name} does not contain both normal and anomalous images.")
    pixel_labels = masks.reshape(-1)
    pixel_scores = maps.reshape(-1)
    return {
        "image_AUROC": roc_auc_score(labels, image_scores) * 100.0,
        "image_AP": average_precision_score(labels, image_scores) * 100.0,
        "image_F1": _best_f1(labels, image_scores) * 100.0,
        "AUPRO": Evaluator.cal_pro_score(
            masks,
            maps,
            max_step=max_step_aupro,
            mp=False,
            cls_name=class_name,
        ) * 100.0,
        "pixel_AUROC": roc_auc_score(pixel_labels, pixel_scores) * 100.0,
        "pixel_AP": average_precision_score(pixel_labels, pixel_scores) * 100.0,
        "pixel_F1": _pixel_f1(masks, maps) * 100.0,
    }


def _evaluate_variant(class_names, labels, masks, maps, image_scores, max_step_aupro):
    rows = []
    for class_name in sorted(np.unique(class_names).tolist()):
        selected = class_names == class_name
        metrics = _class_metrics(
            labels[selected],
            masks[selected],
            maps[selected],
            image_scores[selected],
            class_name,
            max_step_aupro,
        )
        rows.append({"class": class_name, **metrics})
    average = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRIC_NAMES
    }
    return rows, average


def _region_diagnostics(labels, masks, raw_maps, final_maps, tolerance):
    delta = final_maps - raw_maps
    amplified = delta > tolerance
    suppressed = delta < -tolerance
    positive_gain = np.maximum(delta, 0.0)
    negative_gain = np.maximum(-delta, 0.0)

    normal_images = labels == 0
    abnormal_images = labels != 0
    mask_pixels = masks > 0.5
    regions = {
        "normal_all": np.broadcast_to(normal_images[:, None, None], masks.shape),
        "mask_in": abnormal_images[:, None, None] & mask_pixels,
        "mask_out": abnormal_images[:, None, None] & ~mask_pixels,
    }
    rows = []
    for name, selected in regions.items():
        count = int(selected.sum())
        if count == 0:
            continue
        rows.append(
            {
                "region": name,
                "pixels": count,
                "amplified_pct": float(amplified[selected].mean() * 100.0),
                "suppressed_pct": float(suppressed[selected].mean() * 100.0),
                "positive_gain": float(positive_gain[selected].mean()),
                "negative_gain": float(negative_gain[selected].mean()),
            }
        )

    normal_raw_max = raw_maps[normal_images].reshape(normal_images.sum(), -1).max(axis=1)
    normal_final_max = final_maps[normal_images].reshape(normal_images.sum(), -1).max(axis=1)
    normal_max_delta = normal_final_max - normal_raw_max

    raw_normal = raw_maps[normal_images].reshape(normal_images.sum(), -1)
    delta_normal = delta[normal_images].reshape(normal_images.sum(), -1)
    topk = max(1, int(raw_normal.shape[1] * 0.01))
    top_indices = np.argpartition(raw_normal, raw_normal.shape[1] - topk, axis=1)[:, -topk:]
    top_delta = np.take_along_axis(delta_normal, top_indices, axis=1)
    summary = {
        "normal_image_max_gain_mean": float(normal_max_delta.mean()),
        "normal_image_max_amplified_pct": float((normal_max_delta > tolerance).mean() * 100.0),
        "normal_raw_top1_amplified_pct": float((top_delta > tolerance).mean() * 100.0),
        "normal_raw_top1_gain_mean": float(np.maximum(top_delta, 0.0).mean()),
    }
    return rows, summary


def _write_csv(path, variant_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "class", *METRIC_NAMES]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant, rows in variant_rows.items():
            for row in rows:
                writer.writerow({"variant": variant, **row})


def main():
    parser = argparse.ArgumentParser(description="Verify whether ARCC amplification reinforces false positives.")
    parser.add_argument("--npz", required=True, help="Path to the run's show_test/outputs.npz.")
    parser.add_argument(
        "--topk-beta",
        type=float,
        default=0.5,
        help="Model beta in image_score = S_global + beta * max(A); current config uses 0.5.",
    )
    parser.add_argument(
        "--max-step-aupro",
        type=int,
        default=100,
        help="AUPRO threshold steps; CLIP-AD configs use 100.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--csv", default="", help="Optional per-class CSV output path.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    required = {"imgs_masks", "anomaly_maps", "raw_anomaly_maps", "cls_names", "anomalys"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"{npz_path} is missing required arrays: {missing}")

    masks = _maps(data["imgs_masks"], "imgs_masks")
    final_maps = _maps(data["anomaly_maps"], "anomaly_maps")
    raw_maps = _maps(data["raw_anomaly_maps"], "raw_anomaly_maps")
    if raw_maps.shape != final_maps.shape or masks.shape != final_maps.shape:
        raise ValueError(
            f"Map shapes differ: masks={masks.shape}, raw={raw_maps.shape}, final={final_maps.shape}."
        )
    labels = np.asarray(data["anomalys"]).reshape(-1).astype(np.int64)
    class_names = np.asarray(data["cls_names"]).reshape(-1).astype(str)
    if labels.size != final_maps.shape[0] or class_names.size != final_maps.shape[0]:
        raise ValueError("Image labels/class names do not match the number of anomaly maps.")

    score_key = "image_scores_max" if "image_scores_max" in data.files else "image_scores"
    saved_final_scores = np.asarray(data[score_key]).reshape(-1).astype(np.float64)
    final_max = final_maps.reshape(final_maps.shape[0], -1).max(axis=1)
    global_scores = saved_final_scores - args.topk_beta * final_max

    variants = {
        "raw": raw_maps,
        "bidirectional": final_maps,
        "suppress_only": np.minimum(raw_maps, final_maps),
    }
    variant_rows = {}
    averages = {}
    for variant, maps in variants.items():
        map_max = maps.reshape(maps.shape[0], -1).max(axis=1)
        image_scores = global_scores + args.topk_beta * map_max
        rows, average = _evaluate_variant(
            class_names,
            labels,
            masks,
            maps,
            image_scores,
            args.max_step_aupro,
        )
        variant_rows[variant] = rows
        averages[variant] = average

    reconstructed = global_scores + args.topk_beta * final_max
    reconstruction_error = float(np.max(np.abs(reconstructed - saved_final_scores)))
    print(f"NPZ: {npz_path}")
    print(f"images: {final_maps.shape[0]}, map shape: {final_maps.shape[1:]}")
    print(f"image-score reconstruction max error: {reconstruction_error:.3e}")

    average_table = []
    for variant in variants:
        average_table.append(
            [variant, *[f"{averages[variant][metric]:.3f}" for metric in METRIC_NAMES]]
        )
    print("\n==> Average metrics")
    print(tabulate(average_table, headers=["Variant", *METRIC_NAMES], tablefmt="github"))

    delta_table = []
    for reference in ("raw", "bidirectional"):
        delta_table.append(
            [
                f"suppress_only - {reference}",
                *[
                    f"{averages['suppress_only'][metric] - averages[reference][metric]:+.3f}"
                    for metric in METRIC_NAMES
                ],
            ]
        )
    print("\n==> Suppress-only deltas")
    print(tabulate(delta_table, headers=["Comparison", *METRIC_NAMES], tablefmt="github"))

    region_rows, region_summary = _region_diagnostics(
        labels,
        masks,
        raw_maps,
        final_maps,
        args.tolerance,
    )
    print("\n==> Bidirectional ARCC change regions")
    print(
        tabulate(
            [
                [
                    row["region"],
                    row["pixels"],
                    f"{row['amplified_pct']:.3f}",
                    f"{row['suppressed_pct']:.3f}",
                    f"{row['positive_gain']:.6f}",
                    f"{row['negative_gain']:.6f}",
                ]
                for row in region_rows
            ],
            headers=["Region", "Pixels", "Amplified %", "Suppressed %", "Mean +gain", "Mean -gain"],
            tablefmt="github",
        )
    )
    print("\n==> Normal false-positive amplification diagnostics")
    for key, value in region_summary.items():
        print(f"{key}: {value:.6f}")

    if args.csv:
        csv_path = Path(args.csv)
        _write_csv(csv_path, variant_rows)
        print(f"\nSaved per-class metrics: {csv_path}")


if __name__ == "__main__":
    main()
