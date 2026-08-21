#!/usr/bin/env python3
"""Diagnose image-score signals and their target-set fusion upper bound."""

import argparse
from itertools import product

import numpy as np
import tabulate
from sklearn.metrics import average_precision_score, roc_auc_score


RAW_KEYS = ("raw_scores_max", "raw_scores_top1", "raw_scores_top5")
MAMBA_KEYS = ("mamba_scores_max", "mamba_scores_top1", "mamba_scores_top5")


def mean_class_metrics(class_names, labels, scores):
    aurocs, aps = [], []
    for class_name in sorted(set(class_names.tolist())):
        selected = class_names == class_name
        class_labels = labels[selected]
        if np.unique(class_labels).size < 2:
            continue
        class_scores = scores[selected]
        aurocs.append(roc_auc_score(class_labels, class_scores) * 100.0)
        aps.append(average_precision_score(class_labels, class_scores) * 100.0)
    return float(np.mean(aurocs)), float(np.mean(aps))


def vector(data, key):
    return np.asarray(data[key], dtype=np.float64).reshape(-1)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate exported V4 image evidence without loading full-resolution maps."
    )
    parser.add_argument("--npz", required=True)
    parser.add_argument(
        "--weight-grid",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
    )
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    required = {
        "cls_names", "anomalys", "image_scores", "global_scores",
        *RAW_KEYS, *MAMBA_KEYS,
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Missing exported score arrays: {missing}")

    class_names = np.asarray(data["cls_names"]).reshape(-1).astype(str)
    labels = np.asarray(data["anomalys"]).reshape(-1).astype(np.int64)
    signals = {
        "trained_fusion": vector(data, "image_scores"),
        "global": vector(data, "global_scores"),
        **{key: vector(data, key) for key in RAW_KEYS + MAMBA_KEYS},
    }

    rows = []
    for name, scores in signals.items():
        auroc, ap = mean_class_metrics(class_names, labels, scores)
        rows.append([name, f"{auroc:.3f}", f"{ap:.3f}"])
    print("==> Individual and trained image-score signals")
    print(tabulate.tabulate(rows, headers=["Signal", "mAUROC_sp", "mAP_sp"], tablefmt="github"))

    # This deliberately uses target labels and is therefore diagnostic only.
    # It answers whether the exported signals contain sufficient ranking
    # information; its best weights must never be reported as a valid test result.
    global_scores = signals["global"]
    candidates = []
    for raw_key, mamba_key in product(RAW_KEYS, MAMBA_KEYS):
        for raw_weight, mamba_weight in product(args.weight_grid, repeat=2):
            scores = (
                global_scores
                + raw_weight * signals[raw_key]
                + mamba_weight * signals[mamba_key]
            )
            auroc, ap = mean_class_metrics(class_names, labels, scores)
            candidates.append((auroc, ap, raw_key, raw_weight, mamba_key, mamba_weight))
    candidates.sort(reverse=True)
    sweep_rows = [
        [raw_key, raw_weight, mamba_key, mamba_weight, f"{auroc:.3f}", f"{ap:.3f}"]
        for auroc, ap, raw_key, raw_weight, mamba_key, mamba_weight in candidates[:10]
    ]
    print("\n==> Diagnostic target-label sweep (upper bound only; not a valid test selection)")
    print(
        tabulate.tabulate(
            sweep_rows,
            headers=["Raw", "w_raw", "Mamba", "w_mamba", "mAUROC_sp", "mAP_sp"],
            tablefmt="github",
        )
    )


if __name__ == "__main__":
    main()
