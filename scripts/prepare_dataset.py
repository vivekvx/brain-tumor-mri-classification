"""Create source-image-disjoint, class-stratified MRI dataset splits.

Usage:
    python3 scripts/prepare_dataset.py --source /path/to/dataset.zip

The Roboflow export names variants like ``Tr-gl_0019_jpg.rf.<hash>.jpg``.
Everything before ``_jpg.rf.`` is treated as one source image and assigned
to exactly one output split. This cannot establish patient-level independence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


CLASSES = ("glioma", "meningioma", "no_tumor", "pituitary")
SPLITS = ("train", "valid", "test")
CSV_LABELS = ("Glioma", "Meningioma", "No Tumor", "Pituitary")
SOURCE_PATTERN = re.compile(r"^(.+)_jpg\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)
SEED = 42


def targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    result = {split: int(raw[split]) for split in SPLITS}
    remaining = total - sum(result.values())
    order = sorted(SPLITS, key=lambda split: (raw[split] - result[split], -SPLITS.index(split)), reverse=True)
    for split in order[:remaining]:
        result[split] += 1
    return result


def read_archive(archive: ZipFile) -> tuple[list[dict[str, str]], dict[str, Counter]]:
    records: list[dict[str, str]] = []
    original: dict[str, Counter] = defaultdict(Counter)
    archive_names = set(archive.namelist())
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".jpg"):
            continue
        parts = PurePosixPath(info.filename).parts
        if len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in CLASSES:
            raise ValueError(f"Unexpected image path: {info.filename}")
        match = SOURCE_PATTERN.fullmatch(parts[2])
        if not match:
            raise ValueError(f"Cannot identify source image: {info.filename}")
        records.append({
            "archive_path": info.filename,
            "original_split": parts[0],
            "class": parts[1],
            "source_id": match.group(1),
            "filename": parts[2],
        })
        original[parts[0]][parts[1]] += 1

    if not records:
        raise ValueError("No JPEG images found")

    for split in SPLITS:
        csv_path = f"{split}/_classes.csv"
        if csv_path not in archive_names:
            raise ValueError(f"Missing {csv_path}")
        rows = list(csv.DictReader(io.StringIO(archive.read(csv_path).decode("utf-8-sig")), skipinitialspace=True))
        expected = {r["filename"]: r["class"] for r in records if r["original_split"] == split}
        seen: set[str] = set()
        for row in rows:
            filename = row["filename"]
            if filename in seen or filename not in expected:
                raise ValueError(f"Duplicate or unknown CSV filename: {split}/{filename}")
            seen.add(filename)
            labels = [CLASSES[i] for i, name in enumerate(CSV_LABELS) if row[name].strip() == "1"]
            if labels != [expected[filename]]:
                raise ValueError(f"CSV label disagrees with folder: {split}/{filename}")
        if seen != set(expected):
            raise ValueError(f"CSV is missing images in {split}")
    return records, original


def assign_groups(records: list[dict[str, str]], ratios: dict[str, float]) -> tuple[dict[tuple[str, str], str], dict[str, dict[str, int]]]:
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        groups[record["class"]][record["source_id"]].append(record)

    assignments: dict[tuple[str, str], str] = {}
    desired: dict[str, dict[str, int]] = {}
    for class_index, class_name in enumerate(CLASSES):
        class_groups = list(groups[class_name].items())
        random.Random(SEED + class_index).shuffle(class_groups)
        desired[class_name] = targets(sum(len(items) for _, items in class_groups), ratios)
        placed = Counter()
        for source_id, items in class_groups:
            # Fill the largest remaining image quota. Shuffling first makes
            # selection independent of the archive's existing split order.
            split = max(SPLITS, key=lambda s: (desired[class_name][s] - placed[s], -SPLITS.index(s)))
            assignments[(class_name, source_id)] = split
            placed[split] += len(items)
    return assignments, desired


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", *CSV_LABELS))
        for record in sorted(records, key=lambda item: item["filename"]):
            writer.writerow((record["filename"], *(int(record["class"] == name) for name in CLASSES)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original Roboflow ZIP")
    parser.add_argument("--output", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--report", type=Path, default=Path("reports/dataset_split.md"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Remove or choose another output path to rebuild.")

    with ZipFile(args.source) as archive:
        records, original = read_archive(archive)
        original_totals = Counter({split: sum(original[split].values()) for split in SPLITS})
        ratios = {split: original_totals[split] / len(records) for split in SPLITS}
        assignments, desired = assign_groups(records, ratios)
        for record in records:
            record["new_split"] = assignments[(record["class"], record["source_id"])]

        args.output.mkdir(parents=True)
        try:
            for split in SPLITS:
                split_records = [r for r in records if r["new_split"] == split]
                for class_name in CLASSES:
                    (args.output / split / class_name).mkdir(parents=True, exist_ok=True)
                for record in split_records:
                    content = archive.read(record["archive_path"])
                    record["sha256"] = hashlib.sha256(content).hexdigest()
                    (args.output / split / record["class"] / record["filename"]).write_bytes(content)
                write_csv(args.output / split / "_classes.csv", split_records)

            with (args.output / "manifest.csv").open("w", newline="") as handle:
                fields = ("archive_path", "original_split", "new_split", "class", "source_id", "filename", "sha256")
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(sorted(records, key=lambda r: r["archive_path"]))

            actual = {split: Counter(r["class"] for r in records if r["new_split"] == split) for split in SPLITS}
            group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
            for record in records:
                group_splits[(record["class"], record["source_id"])].add(record["new_split"])
            if any(len(splits) != 1 for splits in group_splits.values()):
                raise AssertionError("Source-image leakage remains")
            if sum(sum(actual[s].values()) for s in SPLITS) != len(records):
                raise AssertionError("Image count changed")

            summary = {
                "source_archive": args.source.name,
                "seed": SEED,
                "group_key": "class + filename before _jpg.rf.",
                "original_counts": {s: dict(original[s]) for s in SPLITS},
                "target_counts": desired,
                "actual_counts": {s: dict(actual[s]) for s in SPLITS},
                "source_groups": len(group_splits),
                "moved_images": sum(r["original_split"] != r["new_split"] for r in records),
                "cross_split_source_groups": 0,
            }
            (args.output / "split_report.json").write_text(json.dumps(summary, indent=2) + "\n")
            args.report.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "# Brain MRI dataset split report",
                "",
                f"Source archive: `{args.source.name}`. Random seed: `{SEED}`.",
                f"Grouping key: `{summary['group_key']}`. All {len(records)} images were retained in {len(group_splits)} source-image groups.",
                "",
                "| Class | Train | Validation | Test |",
                "|---|---:|---:|---:|",
            ]
            for class_name in CLASSES:
                lines.append(f"| {class_name} | {actual['train'][class_name]} | {actual['valid'][class_name]} | {actual['test'][class_name]} |")
            lines += [
                f"| **Total** | **{sum(actual['train'].values())}** | **{sum(actual['valid'].values())}** | **{sum(actual['test'].values())}** |",
                "",
                f"Images moved from the supplied split: **{summary['moved_images']}**.",
                "Source-image groups crossing output splits: **0**.",
                "",
                "The output is in `data/cleaned/`; `manifest.csv` records each image's original and new split plus SHA-256.",
                "These names identify source images, not patients. Patient-level independence cannot be confirmed from this archive.",
                "The test split should be reserved for final evaluation; model selection should use validation only.",
                "",
            ]
            args.report.write_text("\n".join(lines))
            print(json.dumps(summary, indent=2))
        except Exception:
            # Keep the partial output for inspection instead of silently deleting it.
            raise


if __name__ == "__main__":
    main()
