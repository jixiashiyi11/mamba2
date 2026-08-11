import argparse
import json
from copy import deepcopy
from pathlib import Path


def _is_abnormal(item):
    return int(item.get("anomaly", 0)) == 1 or bool(item.get("mask_path", ""))


def build_supervised_meta(input_path, output_path):
    with open(input_path, "r") as f:
        meta = json.load(f)

    train_meta = deepcopy(meta.get("train", {}))
    test_meta = meta.get("test", {})

    added = 0
    for cls_name, items in test_meta.items():
        target_items = train_meta.setdefault(cls_name, [])
        for item in items:
            if not _is_abnormal(item):
                continue
            supervised_item = deepcopy(item)
            supervised_item["anomaly"] = 1
            target_items.append(supervised_item)
            added += 1

    out = deepcopy(meta)
    out["train"] = train_meta
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    train_count = sum(len(items) for items in train_meta.values())
    train_abnormal = sum(_is_abnormal(item) for items in train_meta.values() for item in items)
    print(f"wrote: {output_path}")
    print(f"train samples: {train_count}")
    print(f"train abnormal/mask samples: {train_abnormal}")
    print(f"added abnormal samples from test: {added}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/mvtec")
    parser.add_argument("--input", default="meta.json")
    parser.add_argument("--output", default="meta_supervised.json")
    args = parser.parse_args()

    root = Path(args.root)
    build_supervised_meta(root / args.input, root / args.output)


if __name__ == "__main__":
    main()
