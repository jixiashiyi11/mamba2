#!/usr/bin/env python3
"""Build the V44 supervised VisA source metadata, including visa_other.

The current VisA metadata has the 35 oracle-diagnostic samples removed from its
test split and stored under ``data/visa_other``.  V44 reverses the cross-domain
direction, so every labeled VisA image is source training data.  This builder
merges the official VisA train split, the remaining test split, and all moved
``visa_other`` samples into one unique supervised train split.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--meta", type=Path, default=Path("data/visa_meta.json"))
    parser.add_argument(
        "--other-manifest",
        type=Path,
        default=Path("data/visa_other/image_auc_outlier_manifest_v43.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visa_meta_supervised_v44.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def flatten(split: dict[str, list[dict]]) -> list[dict]:
    return [entry for entries in split.values() for entry in entries]


def prefixed_other_entry(entry: dict) -> dict:
    output = copy.deepcopy(entry)
    output["img_path"] = f"visa_other/{entry['img_path']}"
    if int(entry["anomaly"]) == 1:
        mask_path = entry.get("mask_path", "")
        if not mask_path:
            raise ValueError(f"Moved anomaly has no mask: {entry['img_path']}")
        output["mask_path"] = f"visa_other/{mask_path}"
    return output


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    meta = load_json(args.meta)
    manifest = load_json(args.other_manifest)
    official_train = flatten(meta["train"])
    remaining_test = flatten(meta["test"])
    moved_entries = manifest.get("selected_entries", [])
    if len(moved_entries) != 35:
        raise RuntimeError(f"Expected 35 visa_other images, got {len(moved_entries)}")

    moved_train = [prefixed_other_entry(entry) for entry in moved_entries]
    merged = official_train + remaining_test + moved_train
    paths = [entry["img_path"] for entry in merged]
    duplicates = sorted(path for path, count in Counter(paths).items() if count != 1)
    if duplicates:
        raise RuntimeError(f"Duplicate V44 training paths: {duplicates[:10]}")

    missing_images = [
        entry["img_path"]
        for entry in merged
        if not (args.data_root / entry["img_path"]).is_file()
    ]
    missing_masks = [
        entry.get("mask_path", "")
        for entry in merged
        if int(entry["anomaly"]) == 1
        and not (args.data_root / entry.get("mask_path", "")).is_file()
    ]
    if missing_images or missing_masks:
        raise FileNotFoundError(
            f"Missing V44 files: images={missing_images[:5]}, masks={missing_masks[:5]}"
        )

    classes = list(meta["train"].keys())
    train_by_class = {name: [] for name in classes}
    for entry in merged:
        train_by_class[entry["cls_name"]].append(entry)
    output = copy.deepcopy(meta)
    output["train"] = train_by_class
    output["test"] = {name: [] for name in classes}
    output["v44_source_description"] = (
        "VisA official train + complete labeled test, including 35 visa_other images"
    )

    counts = Counter(int(entry["anomaly"]) for entry in merged)
    other_counts = Counter(int(entry["anomaly"]) for entry in moved_train)
    if len(official_train) != 8659 or len(remaining_test) != 2127:
        raise RuntimeError(
            f"Unexpected input counts: train={len(official_train)}, test={len(remaining_test)}"
        )
    if len(merged) != 10821 or counts != Counter({0: 9621, 1: 1200}):
        raise RuntimeError(f"Unexpected merged counts: total={len(merged)}, labels={counts}")
    if other_counts != Counter({0: 25, 1: 10}):
        raise RuntimeError(f"Unexpected visa_other labels: {other_counts}")

    atomic_write_json(args.output, output)
    print(f"OUTPUT={args.output}")
    print(f"OFFICIAL_TRAIN={len(official_train)}")
    print(f"REMAINING_TEST_ADDED={len(remaining_test)}")
    print(f"VISA_OTHER_ADDED={len(moved_train)}")
    print(f"VISA_OTHER_NORMAL={other_counts[0]}")
    print(f"VISA_OTHER_ANOMALY={other_counts[1]}")
    print(f"V44_TRAIN_TOTAL={len(merged)}")
    print(f"V44_TRAIN_NORMAL={counts[0]}")
    print(f"V44_TRAIN_ANOMALY={counts[1]}")
    print("MISSING_IMAGES=0")
    print("MISSING_MASKS=0")


if __name__ == "__main__":
    main()
