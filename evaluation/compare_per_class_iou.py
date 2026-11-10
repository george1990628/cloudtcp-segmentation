#!/usr/bin/env python3
"""Prepare validation references and compare CloudTCP per-class IoU CSVs.

Python 3.9+; standard library only. This utility does not train or evaluate a
model. Save a reference at evaluation time; matching references cannot prove
that independently supplied metric files were actually produced on that data.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile


CLASSES = {0: "background", 1: "curb", 2: "lane_marking",
           3: "crosswalk", 4: "median_boundary"}
FOREGROUND = tuple(CLASSES[i] for i in range(1, 5))
REF_FIELDS = ("chip", "split", "image", "mask", "image_sha256",
              "mask_sha256", "label_status")


def csv_rows(text, required, source):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError(f"{source}: duplicate CSV column names")
    missing = set(required) - set(fields)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: no data rows")
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"{source}: malformed CSV row")
    return rows


def load_metrics(path):
    rows = csv_rows(Path(path).read_text(encoding="utf-8-sig"),
                    ("class", "IoU"), path)
    result = {}
    for row in rows:
        name = row["class"].strip()
        if name not in CLASSES.values():
            raise ValueError(f"{path}: unknown class {name!r}; use the original class names")
        if name in result:
            raise ValueError(f"{path}: duplicate class {name!r}")
        try:
            value = float(row["IoU"])
        except ValueError as exc:
            raise ValueError(f"{path}: {name} has no numeric IoU; check class support") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{path}: {name} IoU must be finite and in [0, 1], not a percentage")
        result[name] = value
    missing = set(FOREGROUND) - set(result)
    if missing:
        raise ValueError(f"{path}: missing foreground classes {sorted(missing)}")
    return result


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts
            or "\\" in name or ":" in name or str(path) == "."):
        raise ValueError(f"Invalid dataset-relative path: {name!r}")
    return path


def make_reference(dataset):
    dataset = Path(dataset)
    archive = None
    if dataset.is_file() and zipfile.is_zipfile(dataset):
        archive = zipfile.ZipFile(dataset)
        names = [n for n in archive.namelist()
                 if not n.endswith("/") and "__MACOSX" not in PurePosixPath(n).parts]
        if len(names) != len(set(names)):
            archive.close()
            raise ValueError("ZIP contains duplicate member names")
        roots = [str(PurePosixPath(n).parent) for n in names
                 if PurePosixPath(n).name == "manifest.csv"
                 and str(PurePosixPath(n).parent / "pipeline_settings.json") in names]
        if len(roots) != 1:
            archive.close()
            raise ValueError("ZIP must have exactly one manifest/settings dataset root")
        root = PurePosixPath(roots[0])

        def read_file(name):
            member = str(root / safe_relative(name))
            if member not in names:
                raise ValueError(f"Missing dataset file: {member}")
            return archive.read(member)
    elif dataset.is_dir():
        root = dataset.resolve()

        def read_file(name):
            candidate = (root / safe_relative(name)).resolve()
            if root not in candidate.parents or not candidate.is_file():
                raise ValueError(f"Missing or outside-dataset file: {name}")
            return candidate.read_bytes()
    else:
        raise ValueError("Dataset must be a ZIP or the folder containing manifest.csv")

    try:
        settings = json.loads(read_file("pipeline_settings.json"))
        expected = {str(k): v for k, v in CLASSES.items()}
        if settings.get("classes") != expected or settings.get("ignore_label") != 255:
            raise ValueError("Expected the original class IDs 0–4 and ignore_label=255")
        rows = csv_rows(read_file("manifest.csv").decode("utf-8-sig"),
                        ("chip", "split", "image", "mask"), "manifest.csv")
        seen, references = set(), []
        for row in rows:
            chip = row["chip"].strip()
            if not chip or chip in seen:
                raise ValueError(f"Missing or duplicate chip ID: {chip!r}")
            seen.add(chip)
            if row["split"] not in ("train", "val"):
                raise ValueError(f"{chip}: split must be train or val")
            if row["split"] == "val":
                references.append({
                    "chip": chip, "split": "val", "image": row["image"],
                    "mask": row["mask"],
                    "image_sha256": hashlib.sha256(read_file(row["image"])).hexdigest(),
                    "mask_sha256": hashlib.sha256(read_file(row["mask"])).hexdigest(),
                    "label_status": row.get("label_status", "unknown"),
                })
        if not references:
            raise ValueError("No validation chips in manifest.csv")
        return sorted(references, key=lambda r: r["chip"])
    finally:
        if archive is not None:
            archive.close()


def load_reference(path):
    rows = csv_rows(Path(path).read_text(encoding="utf-8-sig"),
                    ("chip", "split", "image_sha256", "mask_sha256"), path)
    result = {}
    for row in rows:
        chip = row["chip"].strip()
        if not chip or chip in result:
            raise ValueError(f"{path}: missing or duplicate chip ID {chip!r}")
        if row["split"] != "val":
            raise ValueError(f"{path}: {chip} is not a validation chip")
        hashes = tuple(row[field].strip().lower()
                       for field in ("image_sha256", "mask_sha256"))
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise ValueError(f"{path}: {chip} has an invalid SHA-256 hash")
        result[chip] = hashes
    return result


def check_references(before, after):
    left, right = load_reference(before), load_reference(after)
    if set(left) != set(right):
        raise ValueError("Validation chip IDs differ. Evaluate both checkpoints on one common set.")
    changed = sorted(chip for chip in left if left[chip] != right[chip])
    if changed:
        raise ValueError("Validation image or mask bytes changed for " + ", ".join(changed)
                         + ". Re-evaluate BOTH checkpoints on the same images and masks.")
    return len(left)


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an earlier report or an input file accidentally.
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reference = commands.add_parser("reference", help="Hash the current validation images/masks")
    reference.add_argument("--dataset", required=True)
    reference.add_argument("--output", required=True)
    rank = commands.add_parser("rank", help="Rank the four foreground classes by increasing IoU")
    rank.add_argument("metrics", help="Existing per_class_metrics.csv")
    rank.add_argument("--output", required=True)
    compare = commands.add_parser("compare", help="Compare IoU on a matching validation reference")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--before-eval-reference", required=True)
    compare.add_argument("--after-eval-reference", required=True)
    compare.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "reference":
            rows = make_reference(args.dataset)
            write_csv(args.output, REF_FIELDS, rows)
            print(f"Recorded {len(rows)} validation chips. Save this file with the matching evaluation.")
        elif args.command == "rank":
            metrics = load_metrics(args.metrics)
            ordered = sorted(FOREGROUND, key=lambda name: (metrics[name], name))
            rows = [{"rank": i, "class": name, "IoU": f"{metrics[name]:.8f}"}
                    for i, name in enumerate(ordered, start=1)]
            write_csv(args.output, ("rank", "class", "IoU"), rows)
            print("Lowest IoU first: " + ", ".join(f"{n}={metrics[n]:.4f}" for n in ordered))
        else:
            before, after = load_metrics(args.before), load_metrics(args.after)
            n_chips = check_references(args.before_eval_reference, args.after_eval_reference)
            rows = []
            for name in FOREGROUND:
                delta = after[name] - before[name]
                rows.append({"class": name, "before_IoU": f"{before[name]:.8f}",
                             "after_IoU": f"{after[name]:.8f}", "delta_IoU": f"{delta:.8f}",
                             "delta_percentage_points": f"{100 * delta:.6f}"})
            mean_before = sum(before[n] for n in FOREGROUND) / len(FOREGROUND)
            mean_after = sum(after[n] for n in FOREGROUND) / len(FOREGROUND)
            rows.append({"class": "foreground_mIoU", "before_IoU": f"{mean_before:.8f}",
                         "after_IoU": f"{mean_after:.8f}",
                         "delta_IoU": f"{mean_after - mean_before:.8f}",
                         "delta_percentage_points": f"{100 * (mean_after - mean_before):.6f}"})
            write_csv(args.output, tuple(rows[0]), rows)
            print(f"Compared 4 foreground classes; {n_chips} validation image/mask pairs match.")
            print(f"Foreground mIoU: {mean_before:.4f} -> {mean_after:.4f} "
                  f"({100 * (mean_after - mean_before):+.2f} percentage points)")
            print("Hashes check supplied references only; verify metric/checkpoint provenance separately.")
    except (ValueError, OSError, KeyError, csv.Error, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
