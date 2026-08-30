#!/usr/bin/env python3
"""Build the V45 joint BTAD/MPDD/BMAD test metadata.

V45 trains on the labeled VisA source split prepared for V44.  This tool only
builds the target-domain test metadata.  It preserves the original per-product
classes for BTAD and MPDD so their reported dataset scores can be macro-averaged
in the same way as standard anomaly-detection benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath


DATASET_SPECS = {
    "BTAD": {
        "root_prefix": "BTech_Dataset_transformed",
        "class_prefix": "BTAD_",
        "expected": 741,
    },
    "MPDD": {
        "root_prefix": "MPDD",
        "class_prefix": "MPDD_",
        "expected": 458,
    },
    "Brain": {
        "root_prefix": "MedAD/Brain_AD",
        "class_map": {"Brain": "Brain_MRI"},
        "expected": 3715,
    },
    "Liver": {
        "root_prefix": "MedAD/Liver_AD",
        "class_map": {"Liver": "Liver_CT"},
        "expected": 1493,
    },
    "Retina": {
        "root_prefix": "MedAD/Retina_RESC_AD",
        "class_map": {"Retina": "Retina_OCT"},
        "expected": 1805,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path("third_party/AA-CLIP-main/dataset/metadata"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visa_to_benchmarks_v45.json"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def prefixed_path(prefix: str, path: str) -> str:
    return str(PurePosixPath(prefix) / PurePosixPath(path))


def target_class_name(dataset_name: str, source_name: str) -> str:
    spec = DATASET_SPECS[dataset_name]
    if "class_map" in spec:
        return spec["class_map"][source_name]
    return f"{spec['class_prefix']}{source_name}"


def defect_name(image_path: str, anomaly: int) -> str:
    if not anomaly:
        return "good"
    parts = PurePosixPath(image_path).parts
    if "test" in parts:
        test_index = parts.index("test")
        if test_index + 1 < len(parts) - 1:
            return parts[test_index + 1]
    return "anomaly"


def load_dataset_entries(metadata_root: Path, dataset_name: str) -> list[dict]:
    path = metadata_root / dataset_name / "full-shot.jsonl"
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    expected = DATASET_SPECS[dataset_name]["expected"]
    if len(entries) != expected:
        raise RuntimeError(
            f"Unexpected {dataset_name} metadata count: {len(entries)} != {expected}"
        )
    return entries


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    test_by_class: dict[str, list[dict]] = {}
    dataset_counts = Counter()
    label_counts = Counter()

    for dataset_name, spec in DATASET_SPECS.items():
        for source in load_dataset_entries(args.metadata_root, dataset_name):
            anomaly = int(float(source["label"]) > 0)
            cls_name = target_class_name(dataset_name, source["class_name"])
            image_path = prefixed_path(spec["root_prefix"], source["image_path"])
            mask_path = ""
            if anomaly:
                source_mask = source.get("mask_path", "")
                if not source_mask:
                    raise ValueError(
                        f"Anomalous {dataset_name} sample has no mask: {source['image_path']}"
                    )
                mask_path = prefixed_path(spec["root_prefix"], source_mask)

            entry = {
                "img_path": image_path,
                "mask_path": mask_path,
                "cls_name": cls_name,
                "specie_name": defect_name(source["image_path"], anomaly),
                "anomaly": anomaly,
                "source_dataset": dataset_name,
            }
            test_by_class.setdefault(cls_name, []).append(entry)
            dataset_counts[dataset_name] += 1
            label_counts[(dataset_name, anomaly)] += 1

    all_entries = [entry for entries in test_by_class.values() for entry in entries]
    duplicate_images = [
        path
        for path, count in Counter(entry["img_path"] for entry in all_entries).items()
        if count != 1
    ]
    if duplicate_images:
        raise RuntimeError(f"Duplicate target images: {duplicate_images[:10]}")

    missing_images = [
        entry["img_path"]
        for entry in all_entries
        if not (args.data_root / entry["img_path"]).is_file()
    ]
    missing_masks = [
        entry["mask_path"]
        for entry in all_entries
        if entry["anomaly"] and not (args.data_root / entry["mask_path"]).is_file()
    ]
    if missing_images or missing_masks:
        raise FileNotFoundError(
            "Missing V45 target files: "
            f"images={missing_images[:5]}, masks={missing_masks[:5]}"
        )

    class_names = list(test_by_class)
    output = {
        "train": {name: [] for name in class_names},
        "test": test_by_class,
        "v45_source": "VisA supervised metadata from V44",
        "v45_targets": list(DATASET_SPECS),
        "v45_target_count": len(all_entries),
    }
    atomic_write_json(args.output, output)

    print(f"OUTPUT={args.output}")
    print(f"TARGET_CLASSES={len(class_names)}")
    print(f"TARGET_TOTAL={len(all_entries)}")
    for dataset_name in DATASET_SPECS:
        print(
            f"{dataset_name}: total={dataset_counts[dataset_name]} "
            f"normal={label_counts[(dataset_name, 0)]} "
            f"abnormal={label_counts[(dataset_name, 1)]}"
        )
    print("MISSING_IMAGES=0")
    print("MISSING_MASKS=0")


if __name__ == "__main__":
    main()
