"""Dataset reader and metrics copied from the existing CloudTCP baseline.
Source: CloudTCP_SegFormer.py, SHA-256
b983ec207318713f0d06e41dc3722cc0e149a04611801f35c3340bdf891b5fe8.
Only imports were reduced; the selected function bodies are unchanged.
No model or training is performed by this module.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import zipfile
import numpy as np
import rasterio
IGNORE = 255

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def extract_dataset(zip_path, workspace):
    """Unpack into a content-addressed local folder; keep Drive I/O out of training."""
    dest = Path(workspace) / ("dataset_" + sha256(zip_path)[:12])
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for item in archive.infolist():
            if "__MACOSX" in Path(item.filename).parts:
                continue
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError("ZIP contains an unsafe path.")
            archive.extract(item, dest)
    roots = [p.parent for p in dest.rglob("pipeline_settings.json")
             if (p.parent / "manifest.csv").exists()]
    if len(roots) != 1:
        raise ValueError("Expected one dataset folder containing manifest.csv and pipeline_settings.json.")
    return roots[0]

def _inside(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f"Missing/invalid dataset path: {name}")
    return path

def read_pair(row, bands):
    with rasterio.open(row["image_path"]) as src:
        if len(bands) != 3 or min(bands) < 1 or max(bands) > src.count:
            raise ValueError("This RGB baseline needs three valid, 1-based image band indices.")
        if any(src.dtypes[b - 1] != "uint8" for b in bands):
            raise ValueError("Expected uint8 imagery; specify scaling before using another dtype.")
        image = src.read(bands).transpose(1, 2, 0)
        valid = np.all(src.read_masks(bands) > 0, axis=0)
        profile = src.profile.copy()
        bounds = tuple(src.bounds)
        source_bands = src.count
    with rasterio.open(row["mask_path"]) as src:
        if src.count != 1 or not np.issubdtype(np.dtype(src.dtypes[0]), np.integer):
            raise ValueError(f"Expected a single-band integer mask: {row['chip']}")
        if (src.height, src.width) != image.shape[:2] or src.crs != profile["crs"]:
            raise ValueError(f"Image/mask size or CRS mismatch: {row['chip']}")
        if not src.transform.almost_equals(profile["transform"]):
            raise ValueError(f"Image/mask transform mismatch: {row['chip']}")
        if src.nodata is not None and src.nodata != IGNORE:
            raise ValueError("Mask nodata must be 255 or unset. Class 0 is annotated background.")
        labels = src.read(1).astype(np.int64)
    invalid_ids = set(np.unique(labels).tolist()) - {0, 1, 2, 3, 4, IGNORE}
    if invalid_ids:
        raise ValueError(f"Unexpected mask IDs {invalid_ids}: {row['chip']}")
    labels[~valid] = IGNORE
    if not np.any(labels != IGNORE):
        raise ValueError(f"No supervised pixels in {row['chip']}")
    return dict(image=image, mask=labels, valid=valid, profile=profile,
                bounds=bounds, source_bands=source_bands, row=row)

def inspect_dataset(root, input_bands=(1, 2, 3)):
    root = Path(root)
    settings = json.loads((root / "pipeline_settings.json").read_text())
    classes = {int(k): v for k, v in settings["classes"].items()}
    if set(classes) != set(range(5)) or settings.get("ignore_label") != IGNORE:
        raise ValueError("Expected background=0, foreground classes=1..4, ignore=255.")
    names = [classes[i] for i in range(5)]
    with (root / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"chip", "split", "image", "mask"}.issubset(rows[0]):
        raise ValueError("Manifest must include chip, split, image and mask columns.")
    samples, seen, digest = [], set(), hashlib.sha256()
    digest.update((root / "manifest.csv").read_bytes())
    digest.update((root / "pipeline_settings.json").read_bytes())
    for row in rows:
        if row["chip"] in seen or row["split"] not in {"train", "val"}:
            raise ValueError("Duplicate chip name or invalid split.")
        seen.add(row["chip"])
        row["image_path"] = _inside(root, row["image"])
        row["mask_path"] = _inside(root, row["mask"])
        for key in ("image_path", "mask_path"):
            digest.update(sha256(row[key]).encode())
        samples.append(read_pair(row, list(input_bands)))
    splits = {split: [s for s in samples if s["row"]["split"] == split]
              for split in ("train", "val")}
    if not all(splits.values()):
        raise ValueError("Both train and validation need image/mask pairs.")
    if len({s["image"].shape for s in samples}) != 1:
        raise ValueError("This baseline expects equal chip dimensions; it never silently resizes labels.")
    group_overlap = sorted({s["row"].get("group") for s in splits["train"]}
                           & {s["row"].get("group") for s in splits["val"]} - {None, ""})
    overlap = []
    for a in splits["train"]:
        for b in splits["val"]:
            if a["profile"]["crs"] != b["profile"]["crs"]:
                raise ValueError("Mixed CRSs require a common-CRS leakage check first.")
            aa, bb = a["bounds"], b["bounds"]
            dx = min(aa[2], bb[2]) - max(aa[0], bb[0])
            dy = min(aa[3], bb[3]) - max(aa[1], bb[1])
            if dx > 1e-4 and dy > 1e-4:
                overlap.append([a["row"]["chip"], b["row"]["chip"]])
    if group_overlap or overlap:
        raise ValueError(f"Split leakage: shared groups={group_overlap}, overlapping chips={overlap}")
    counts = {split: np.zeros(5, dtype=np.int64) for split in splits}
    table = []
    for split, ss in splits.items():
        for sample in ss:
            y = sample["mask"]
            counts[split] += np.bincount(y[y != IGNORE], minlength=5)
    if np.any(counts["train"][1:] == 0):
        raise ValueError("At least one foreground class has no training labels.")
    for cls, name in enumerate(names):
        row = {"id": cls, "class": name}
        for split, ss in splits.items():
            row[f"{split}_pixels"] = int(counts[split][cls])
            row[f"{split}_percent"] = float(100 * counts[split][cls] / counts[split].sum())
            row[f"{split}_chips"] = sum(bool(np.any(s["mask"] == cls)) for s in ss)
        table.append(row)
    audit = dict(dataset_fingerprint=digest.hexdigest(),
                 n_chips={s: len(v) for s, v in splits.items()},
                 class_names=names, input_bands=list(input_bands),
                 shape=list(samples[0]["image"].shape), source_bands=samples[0]["source_bands"],
                 label_status=dict(Counter(r.get("label_status", "unknown") for r in rows)),
                 crs=str(samples[0]["profile"]["crs"]),
                 exported_pixel_spacing=list(samples[0]["profile"]["transform"])[0:6:4],
                 class_counts={k: v.tolist() for k, v in counts.items()},
                 ignored_pixels=sum(int((s["mask"] == IGNORE).sum()) for s in samples),
                 group_overlap=group_overlap, overlapping_train_val_pairs=overlap,
                 class_report=table,
                 caveat="Label agreement with provided annotations; auto_draft is not verified ground truth.")
    # The supplied UTM raster uses metres. Pixel sampling is not verified native sensor GSD.
    audit["exported_pixel_spacing"] = [abs(samples[0]["profile"]["transform"].a),
                                       abs(samples[0]["profile"]["transform"].e)]
    return dict(root=root, settings=settings, samples=samples, splits=splits,
                counts=counts, names=names, audit=audit)

def metrics_from_confusion(matrix, names):
    a = matrix.astype(np.float64)
    diagonal = np.diag(a)
    union = a.sum(0) + a.sum(1) - diagonal
    total = a.sum(0) + a.sum(1)
    iou = [float(diagonal[i] / union[i]) if union[i] else None for i in range(5)]
    f1 = [float(2 * diagonal[i] / total[i]) if total[i] else None for i in range(5)]
    fg = [x for x in iou[1:] if x is not None]
    return dict(foreground_miou=float(np.mean(fg)) if fg else None,
                iou=dict(zip(names, iou)), f1=dict(zip(names, f1)),
                pixel_accuracy=float(diagonal.sum() / a.sum()) if a.sum() else None,
                confusion_matrix=matrix.tolist(), confusion_axes="rows=annotation, columns=prediction")
