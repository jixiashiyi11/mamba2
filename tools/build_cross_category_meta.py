import argparse
import json
from copy import deepcopy
from pathlib import Path


def _split_names(values):
    names = []
    for value in values or []:
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    return names


def _is_abnormal(item):
    return int(item.get("anomaly", 0)) == 1 or bool(item.get("mask_path", ""))


def _class_names(meta):
    return sorted(set(meta.get("train", {}).keys()) | set(meta.get("test", {}).keys()))


def _image_paths(split_meta):
    return {
        str(item.get("img_path", "")).replace("\\", "/").strip()
        for items in split_meta.values()
        for item in items
        if str(item.get("img_path", "")).strip()
    }


def build_cross_category_meta(input_path, output_path, target_classes, source_classes=None):
    with open(input_path, "r") as f:
        meta = json.load(f)

    all_classes = _class_names(meta)
    target_classes = list(dict.fromkeys(target_classes))
    missing_targets = sorted(set(target_classes) - set(all_classes))
    if missing_targets:
        raise ValueError(f"Unknown target classes: {missing_targets}. Available: {all_classes}")

    if source_classes:
        source_classes = list(dict.fromkeys(source_classes))
        missing_sources = sorted(set(source_classes) - set(all_classes))
        if missing_sources:
            raise ValueError(f"Unknown source classes: {missing_sources}. Available: {all_classes}")
    else:
        source_classes = [cls_name for cls_name in all_classes if cls_name not in target_classes]

    overlap = sorted(set(source_classes) & set(target_classes))
    if overlap:
        raise ValueError(f"Source and target classes overlap: {overlap}")

    train_meta = {}
    source_normal = 0
    source_abnormal = 0
    for cls_name in source_classes:
        train_items = [deepcopy(item) for item in meta.get("train", {}).get(cls_name, [])]
        source_normal += len(train_items)

        for item in meta.get("test", {}).get(cls_name, []):
            if not _is_abnormal(item):
                continue
            supervised_item = deepcopy(item)
            supervised_item["anomaly"] = 1
            train_items.append(supervised_item)
            source_abnormal += 1

        if train_items:
            train_meta[cls_name] = train_items

    test_meta = {
        cls_name: deepcopy(meta.get("test", {}).get(cls_name, []))
        for cls_name in target_classes
    }

    path_overlap = sorted(_image_paths(train_meta) & _image_paths(test_meta))
    if path_overlap:
        raise RuntimeError(
            f"Cross-category split is not disjoint: {len(path_overlap)} image path(s) "
            f"appear in both train and test. Examples: {path_overlap[:5]}"
        )

    out = deepcopy(meta)
    out["train"] = train_meta
    out["test"] = test_meta
    out["_cross_category"] = {
        "source_classes": source_classes,
        "target_classes": target_classes,
        "source_normal_train_samples": source_normal,
        "source_abnormal_mask_samples": source_abnormal,
        "target_test_samples": sum(len(items) for items in test_meta.values()),
        "train_test_path_overlap": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    train_count = sum(len(items) for items in train_meta.values())
    train_abnormal = sum(_is_abnormal(item) for items in train_meta.values() for item in items)
    test_count = sum(len(items) for items in test_meta.values())
    test_abnormal = sum(_is_abnormal(item) for items in test_meta.values() for item in items)

    print(f"wrote: {output_path}")
    print(f"source classes: {', '.join(source_classes)}")
    print(f"target classes: {', '.join(target_classes)}")
    print(f"train samples: {train_count}")
    print(f"train abnormal/mask samples: {train_abnormal}")
    print(f"test samples: {test_count}")
    print(f"test abnormal/mask samples: {test_abnormal}")
    print("train/test image path overlap: 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--input", default="meta.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--target-classes", nargs="+", required=True)
    parser.add_argument("--source-classes", nargs="*", default=None)
    args = parser.parse_args()

    target_classes = _split_names(args.target_classes)
    source_classes = _split_names(args.source_classes)
    if not target_classes:
        raise ValueError("At least one target class is required.")

    output_name = args.output
    if not output_name:
        target_tag = "_".join(target_classes)
        output_name = f"meta_cross_category_{target_tag}.json"

    root = Path(args.root)
    build_cross_category_meta(
        root / args.input,
        root / output_name,
        target_classes=target_classes,
        source_classes=source_classes,
    )


if __name__ == "__main__":
    main()
