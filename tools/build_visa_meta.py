import argparse
import csv
import json
from collections import Counter
from pathlib import Path


VISA_CLASSES = [
    "pcb1", "pcb2", "pcb3", "pcb4",
    "macaroni1", "macaroni2", "capsules", "candle",
    "cashew", "chewinggum", "fryum", "pipe_fryum",
]


def _clean_relative_path(value):
    value = str(value or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def _column_index(header, candidates, fallback):
    normalized = {str(name).strip().lower(): idx for idx, name in enumerate(header)}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if fallback < len(header):
        return fallback
    raise ValueError(
        f"Cannot find any of columns {candidates} in VisA CSV header: {header}"
    )


def build_visa_meta(root, split_csv, output):
    root = Path(root)
    split_csv = Path(split_csv)
    output = Path(output)

    if not split_csv.is_absolute():
        split_csv = root / split_csv
    if not output.is_absolute():
        output = root / output
    if not split_csv.is_file():
        raise FileNotFoundError(
            f"VisA split CSV does not exist: {split_csv}. "
            "Expected the official file split_csv/1cls.csv under the VisA root."
        )

    meta = {
        "train": {name: [] for name in VISA_CLASSES},
        "test": {name: [] for name in VISA_CLASSES},
    }
    counts = Counter()
    missing_files = []
    seen_images = set()

    with split_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"VisA split CSV is empty: {split_csv}") from exc

        object_idx = _column_index(header, ("object", "class", "category"), 0)
        split_idx = _column_index(header, ("split", "set", "phase"), 1)
        label_idx = _column_index(header, ("label",), 2)
        image_idx = _column_index(header, ("image", "image_path", "img_path"), 3)
        mask_idx = _column_index(header, ("mask", "mask_path"), 4)

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not str(value).strip() for value in row):
                continue
            required_idx = max(object_idx, split_idx, label_idx, image_idx, mask_idx)
            if len(row) <= required_idx:
                raise ValueError(
                    f"Malformed VisA CSV row {line_number}: expected at least "
                    f"{required_idx + 1} columns, got {len(row)}"
                )

            cls_name = str(row[object_idx]).strip()
            split = str(row[split_idx]).strip().lower()
            label = str(row[label_idx]).strip().lower()
            image_path = _clean_relative_path(row[image_idx])
            mask_path = _clean_relative_path(row[mask_idx])

            if cls_name not in VISA_CLASSES:
                raise ValueError(
                    f"Unknown VisA class `{cls_name}` at CSV row {line_number}."
                )
            if split not in meta:
                raise ValueError(
                    f"Unsupported split `{split}` at CSV row {line_number}; "
                    "expected train or test."
                )
            is_abnormal = label not in {"normal", "good", "0"}
            if not image_path:
                raise ValueError(f"Empty image path at CSV row {line_number}.")
            if image_path in seen_images:
                raise ValueError(
                    f"Duplicate image path in VisA CSV at row {line_number}: {image_path}"
                )
            seen_images.add(image_path)

            image_exists = (root / image_path).is_file()
            mask_exists = bool(mask_path) and (root / mask_path).is_file()
            if not image_exists:
                missing_files.append(f"row {line_number}: image {image_path}")
            if split == "test" and is_abnormal and not mask_exists:
                missing_files.append(
                    f"row {line_number}: anomaly mask {mask_path or '<empty>'}"
                )

            item = {
                "img_path": image_path,
                "mask_path": mask_path if is_abnormal else "",
                "cls_name": cls_name,
                "specie_name": "anomaly" if is_abnormal else "good",
                "anomaly": int(is_abnormal),
            }
            meta[split][cls_name].append(item)
            counts[(split, "abnormal" if is_abnormal else "normal")] += 1

    if missing_files:
        examples = "\n  ".join(missing_files[:20])
        raise FileNotFoundError(
            f"Found {len(missing_files)} missing VisA image/mask file(s). "
            f"First examples:\n  {examples}\n"
            "The --root argument must point to the directory that contains "
            "both split_csv/ and the paths listed in 1cls.csv."
        )

    empty_test_classes = [name for name in VISA_CLASSES if not meta["test"][name]]
    if empty_test_classes:
        raise ValueError(
            "VisA CSV has no test samples for required classes: "
            + ", ".join(empty_test_classes)
        )
    for cls_name in VISA_CLASSES:
        class_labels = {item["anomaly"] for item in meta["test"][cls_name]}
        if class_labels != {0, 1}:
            raise ValueError(
                f"VisA test class `{cls_name}` must contain both normal and anomaly "
                f"samples, found labels {sorted(class_labels)}."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")

    print(f"wrote: {output}")
    print(
        "train normal={train_normal}, train abnormal={train_abnormal}, "
        "test normal={test_normal}, test abnormal={test_abnormal}".format(
            train_normal=counts[("train", "normal")],
            train_abnormal=counts[("train", "abnormal")],
            test_normal=counts[("test", "normal")],
            test_abnormal=counts[("test", "abnormal")],
        )
    )
    print(f"test classes={len(VISA_CLASSES)}")


def main():
    parser = argparse.ArgumentParser(
        description="Build an ADer-style meta.json from the official VisA 1cls split."
    )
    parser.add_argument(
        "--root",
        default="data",
        help="VisA root containing split_csv/1cls.csv and the CSV image paths.",
    )
    parser.add_argument(
        "--split-csv",
        default="split_csv/1cls.csv",
        help="CSV path, absolute or relative to --root.",
    )
    parser.add_argument(
        "--output",
        default="visa_meta.json",
        help="Output path, absolute or relative to --root.",
    )
    args = parser.parse_args()
    build_visa_meta(args.root, args.split_csv, args.output)


if __name__ == "__main__":
    main()
