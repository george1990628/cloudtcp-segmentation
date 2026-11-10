"""CloudTCP GEO-W3-3: SegFormer-B0 baseline for dataset_split.zip.

The accompanying Colab notebook embeds this entire module; no separate .py
upload is necessary there. For local use, install the notebook's dependencies:
python CloudTCP_SegFormer.py --dataset dataset_split --output runs/b0 --epochs 20

Labels in the supplied dataset are auto_draft. Metrics measure agreement with
those drafts, not independently verified real-world accuracy.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random
import shutil
import time
import zipfile
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (SegformerConfig, SegformerForSemanticSegmentation,
                          SegformerImageProcessor)

BASE_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
BASE_REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE = np.array([[35, 35, 35], [245, 160, 45], [245, 225, 55],
                    [50, 195, 240], [195, 90, 220]], dtype=np.uint8)
IGNORE = 255
DEFAULTS = dict(epochs=20, batch_size=8, seed=42, encoder_lr=1e-4,
                decoder_lr=5e-4, weight_decay=0.01, dice_weight=0.5,
                input_bands=[1, 2, 3], brightness=[0.9, 1.1], contrast=[0.9, 1.1],
                device="auto", threads=4, resume=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


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


def colorize(labels):
    out = PALETTE[np.minimum(labels, 4)].copy()
    out[labels == IGNORE] = (130, 130, 130)
    return out


def show_samples(bundle, limit=4, output=None):
    import matplotlib.pyplot as plt
    ss = sorted(bundle["splits"]["val"],
                key=lambda s: (-int(np.any(s["mask"] == 4)), s["row"]["chip"]))[:limit]
    fig, axes = plt.subplots(len(ss), 3, figsize=(12, 3.7 * len(ss)), squeeze=False)
    for ax, sample in zip(axes, ss):
        rgb, y = sample["image"], sample["mask"]
        overlay = rgb.copy()
        visible = (y > 0) & (y != IGNORE)
        overlay[visible] = (0.5 * rgb[visible] + 0.5 * colorize(y)[visible]).astype(np.uint8)
        for a, im, title in zip(ax, [rgb, colorize(y), overlay],
                               [sample["row"]["chip"] + " RGB", "Draft annotation", "Draft overlay"]):
            a.imshow(im, interpolation="nearest"); a.set_title(title); a.axis("off")
    fig.suptitle("Input check — bands 1–3 used as RGB; labels require review", fontsize=14)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.show()
    plt.close(fig)


class RoadDataset(Dataset):
    def __init__(self, samples, config, augment=False):
        self.samples, self.config, self.augment, self.epoch = samples, config, augment, 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        image = s["image"].astype(np.float32) / 255.0
        labels = s["mask"].copy()
        if self.augment:
            rng = np.random.default_rng([self.config["seed"], self.epoch, index])
            k = int(rng.integers(4))
            image, labels = np.rot90(image, k), np.rot90(labels, k)
            if rng.random() < 0.5:
                image, labels = image[:, ::-1], labels[:, ::-1]
            image = image * rng.uniform(*self.config["brightness"])
            mean = image.mean((0, 1), keepdims=True)
            image = np.clip((image - mean) * rng.uniform(*self.config["contrast"]) + mean, 0, 1)
        image = ((image - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image)), torch.from_numpy(np.ascontiguousarray(labels))


def loss_and_logits(logits, target, weights, dice_weight=0.5):
    logits = F.interpolate(logits.float(), size=target.shape[-2:], mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE)
    valid = target != IGNORE
    truth = F.one_hot(target.masked_fill(~valid, 0), 5).permute(0, 3, 1, 2).float()
    truth = truth * valid[:, None]
    probability = logits.softmax(1) * valid[:, None]
    intersection = (truth * probability).sum((0, 2, 3))
    denominator = (truth + probability).sum((0, 2, 3))
    present = truth.sum((0, 2, 3)) > 0
    present[0] = False
    dice = 1 - (2 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = dice[present].mean() if present.any() else ce * 0
    return ce + dice_weight * dice_loss, logits


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


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, weights, names, dice_weight):
    model.eval()
    confusion = np.zeros((5, 5), dtype=np.int64)
    total_loss, n = 0.0, 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        with amp_context(device):
            raw = model(pixel_values=image).logits
        loss, logits = loss_and_logits(raw, target, weights, dice_weight)
        pred = logits.argmax(1)
        valid = target != IGNORE
        encoded = (target[valid] * 5 + pred[valid]).cpu()
        confusion += torch.bincount(encoded, minlength=25).reshape(5, 5).numpy()
        total_loss += float(loss) * len(image)
        n += len(image)
    return {"loss": total_loss / n, **metrics_from_confusion(confusion, names)}


def copy_atomic(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".uploading")
    shutil.copy2(source, temp)
    temp.replace(destination)


def checkpoint_atomic(state, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def train_baseline(bundle, output, config=None, backup=None):
    """Train or resume. Backup points at a mounted Drive run directory in Colab."""
    cfg = {**DEFAULTS, **(config or {})}
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if list(cfg["input_bands"]) != bundle["audit"]["input_bands"]:
        raise ValueError("Re-run the dataset audit after changing input bands.")
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if cfg["device"] == "auto" else cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. In Colab select Runtime > Change runtime type > GPU.")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    backup = Path(backup) if backup else None
    if backup:
        backup.mkdir(parents=True, exist_ok=True)
    local_state = output / "last_checkpoint.pt"
    remote_state = backup / local_state.name if backup else None
    existing = local_state if local_state.exists() else remote_state
    state = None
    if existing is not None and existing.exists():
        if not cfg["resume"]:
            raise ValueError("This run already has a checkpoint. Use resume=True or choose a new RUN_NAME.")
        copy_atomic(existing, local_state)
        # Only tensor/primitive state is stored; no arbitrary objects are unpickled.
        state = torch.load(local_state, map_location="cpu", weights_only=True)
    immutable = {k: v for k, v in cfg.items() if k not in {"epochs", "device", "threads", "resume"}}
    signature = {"dataset_fingerprint": bundle["audit"]["dataset_fingerprint"],
                 "class_names": bundle["names"], "training_options": immutable,
                 "base_model": BASE_MODEL, "base_revision": BASE_REVISION}
    if state and state["signature"] != signature:
        raise ValueError("Dataset or training settings changed. Choose a new RUN_NAME instead of resuming this run.")
    if state:
        model = SegformerForSemanticSegmentation(SegformerConfig.from_dict(state["model_config"]))
        model.load_state_dict(state["model"])
    else:
        # The original 150-class classifier is intentionally replaced by five outputs.
        model = SegformerForSemanticSegmentation.from_pretrained(
            BASE_MODEL, revision=BASE_REVISION, num_labels=5,
            id2label=dict(enumerate(bundle["names"])),
            label2id={name: i for i, name in enumerate(bundle["names"])},
            semantic_loss_ignore_index=IGNORE, ignore_mismatched_sizes=True)
    model.config.cloudtcp_input_bands = list(cfg["input_bands"])
    model.config.cloudtcp_label_status = bundle["audit"]["label_status"]
    model.config.cloudtcp_pipeline_test_only = False
    model.config.cloudtcp_evaluation_caveat = bundle["audit"]["caveat"]
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.segformer.parameters(), "lr": cfg["encoder_lr"]},
        {"params": model.decode_head.parameters(), "lr": cfg["decoder_lr"]}],
        weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_set = RoadDataset(bundle["splits"]["train"], cfg, augment=True)
    val_set = RoadDataset(bundle["splits"]["val"], cfg)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    counts = bundle["counts"]["train"]
    weights = np.clip(np.sqrt(counts.sum() / (5 * np.maximum(counts, 1))), 0.25, 10)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    history, best_state, best_metrics, best_epoch, best_score, first_epoch = [], None, None, 0, -1.0, 1
    if state:
        optimizer.load_state_dict(state["optimizer"])
        if state["scaler"] and device.type == "cuda":
            scaler.load_state_dict(state["scaler"])
        history, best_state = state["history"], state["best_model"]
        best_metrics, best_epoch, best_score = state["best_metrics"], state["best_epoch"], state["best_score"]
        first_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resuming after epoch {state['epoch']}; target is {cfg['epochs']} total epochs.", flush=True)
        del state
    run_info = dict(config=cfg, device=str(device), parameters=sum(p.numel() for p in model.parameters()),
                    signature=signature, class_weights=weights.cpu().tolist(),
                    loss="weighted cross-entropy + 0.5 foreground Dice" if cfg["dice_weight"] == 0.5
                    else f"weighted cross-entropy + {cfg['dice_weight']} foreground Dice",
                    selection="highest validation foreground mIoU against supplied labels",
                    label_status=bundle["audit"]["label_status"],
                    band_note="Bands 1–3 treated as RGB following the pipeline convention; band 4 unused. TIFFs have no band descriptions.",
                    native_GSD="unknown; exported sampling is approximately 0.6 m",
                    versions={"torch": str(torch.__version__), "numpy": str(np.__version__)})
    write_json(output / "training_config.json", run_info)
    write_json(output / "data_audit.json", bundle["audit"])
    pd.DataFrame(bundle["audit"]["class_report"]).to_csv(output / "class_distribution.csv", index=False)
    print(f"Device: {device}; parameters: {run_info['parameters']:,}; all layers trainable.", flush=True)
    print("Validation scores measure agreement with the supplied draft labels.", flush=True)
    for epoch in range(first_epoch, cfg["epochs"] + 1):
        started = time.monotonic()
        train_set.epoch = epoch
        # Each epoch has a reproducible order, also when restarting from a saved epoch.
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                            generator=generator, num_workers=0, pin_memory=device.type == "cuda")
        model.train(); total_loss, n = 0.0, 0
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                raw = model(pixel_values=image).logits
            loss, _ = loss_and_logits(raw, target, weights, cfg["dice_weight"])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss. Inspect masks, input scaling and learning rates.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            total_loss += float(loss.detach()) * len(image); n += len(image)
        val = evaluate(model, val_loader, device, weights, bundle["names"], cfg["dice_weight"])
        score = val["foreground_miou"]
        if score is None:
            raise RuntimeError("No evaluable foreground classes in validation.")
        if score > best_score:
            best_score, best_epoch, best_metrics = score, epoch, val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = dict(epoch=epoch, train_loss=total_loss / n, val_loss=val["loss"],
                      foreground_miou=score, epoch_seconds=round(time.monotonic() - started, 2))
        history.append(record)
        snapshot = dict(epoch=epoch, model=model.state_dict(), model_config=model.config.to_dict(),
                        optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        best_model=best_state, best_metrics=best_metrics, best_score=best_score,
                        best_epoch=best_epoch, history=history, signature=signature,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        checkpoint_atomic(snapshot, local_state)
        del snapshot
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        write_json(output / "history.json", history)
        if backup:
            copy_atomic(local_state, remote_state)
            for name in ["history.csv", "history.json", "training_config.json", "data_audit.json"]:
                copy_atomic(output / name, backup / name)
        print(json.dumps(record) + f" | saved; best epoch={best_epoch}", flush=True)
    if best_state is None:
        raise RuntimeError("No trained state is available.")
    # Export the best validation model, while last_checkpoint.pt retains the latest optimizer state.
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = output / "best_model"
    model.save_pretrained(checkpoint, safe_serialization=True)
    SegformerImageProcessor(do_resize=False, do_reduce_labels=False).save_pretrained(checkpoint)
    write_json(checkpoint / "training_provenance.json", {**run_info, "best_epoch": best_epoch,
               "completed_epochs": history[-1]["epoch"], "metrics": best_metrics,
               "warning": bundle["audit"]["caveat"]})
    if "LICENSE_TEXT" in globals():
        (checkpoint / "NVIDIA_SEGFORMER_LICENSE.txt").write_text(LICENSE_TEXT)
    write_json(output / "metrics.json", {"best_epoch": best_epoch,
               "completed_epochs": history[-1]["epoch"], "label_status": bundle["audit"]["label_status"],
               "warning": bundle["audit"]["caveat"], **best_metrics})
    if backup:
        for path in checkpoint.rglob("*"):
            if path.is_file(): copy_atomic(path, backup / path.relative_to(output))
        copy_atomic(output / "metrics.json", backup / "metrics.json")
    return dict(model=model, device=device, output=output, backup=backup,
                config=cfg, history=history, metrics=best_metrics, best_epoch=best_epoch,
                weights=weights, val_loader=val_loader, resumed_from_epoch=first_epoch - 1)


@torch.no_grad()
def predict_geotiff(image_path, model, output_path, input_bands=(1, 2, 3), device=None):
    """Export IDs 0..4 on the original pixel grid; 255 marks invalid image pixels."""
    device = torch.device(device) if device else next(model.parameters()).device
    model.eval()
    with rasterio.open(image_path) as src:
        rgb = src.read(list(input_bands)).transpose(1, 2, 0)
        if rgb.dtype != np.uint8:
            raise ValueError("Prediction input must be uint8 with the training band order.")
        valid = np.all(src.read_masks(list(input_bands)) > 0, axis=0)
        profile = src.profile.copy()
    x = ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
    with amp_context(device):
        raw = model(pixel_values=torch.from_numpy(x.copy())[None].to(device)).logits
    pred = F.interpolate(raw.float(), size=rgb.shape[:2], mode="bilinear",
                         align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
    pred[~valid] = IGNORE
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=IGNORE, compress="deflate")
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred, 1)
        dst.update_tags(class_names=json.dumps(model.config.id2label),
                        label_status=json.dumps(getattr(model.config, "cloudtcp_label_status", {})),
                        evaluation_caveat="Baseline predictions; annotations include unreviewed auto_draft labels.")
    return pred


def export_results(bundle, run):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    output, backup = run["output"], run["backup"]
    names, model = bundle["names"], run["model"]
    predictions = {}
    per_chip = []
    for sample in bundle["splits"]["val"]:
        chip = sample["row"]["chip"]
        dest = output / "predictions" / (chip + ".tif")
        pred = predict_geotiff(sample["row"]["image_path"], model, dest,
                               run["config"]["input_bands"], run["device"])
        predictions[chip] = pred
        valid = sample["mask"] != IGNORE
        matrix = np.bincount((sample["mask"][valid] * 5 + pred[valid]).ravel(),
                             minlength=25).reshape(5, 5)
        score = metrics_from_confusion(matrix, names)
        per_chip.append({"chip": chip, "label_status": sample["row"].get("label_status"),
                         "foreground_miou": score["foreground_miou"], **score["iou"]})
    pd.DataFrame(per_chip).to_csv(output / "per_chip_metrics.csv", index=False)
    metrics_table = pd.DataFrame([{"class": n, "IoU": run["metrics"]["iou"][n],
                                  "F1": run["metrics"]["f1"][n]} for n in names])
    metrics_table.to_csv(output / "per_class_metrics.csv", index=False)
    h = pd.DataFrame(run["history"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h.epoch, h.train_loss, label="Train"); axes[0].plot(h.epoch, h.val_loss, label="Validation")
    axes[0].set_title("Weighted CE + foreground Dice"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(h.epoch, h.foreground_miou, color="#237a66")
    axes[1].set_title("Validation agreement with draft labels"); axes[1].set_ylabel("Foreground mIoU"); axes[1].set_ylim(0, 1)
    for ax in axes: ax.set_xlabel("Epoch"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output / "learning_curves.png", dpi=150); plt.close(fig)
    # Deterministic sample choice includes median examples, where available.
    examples = sorted(bundle["splits"]["val"],
                      key=lambda s: (-int(np.any(s["mask"] == 4)), s["row"]["chip"]))[:4]
    fig, axes = plt.subplots(len(examples), 4, figsize=(15, 3.4 * len(examples)), squeeze=False)
    for ax, sample in zip(axes, examples):
        chip = sample["row"]["chip"]; rgb, target = sample["image"], sample["mask"]
        pred = predictions[chip]
        overlay = rgb.copy(); selected = (pred > 0) & (pred != IGNORE)
        overlay[selected] = (0.5 * rgb[selected] + 0.5 * colorize(pred)[selected]).astype(np.uint8)
        for a, im, title in zip(ax, [rgb, colorize(target), colorize(pred), overlay],
                               [chip + " RGB", "Draft annotation", "Prediction", "Prediction overlay"]):
            a.imshow(im, interpolation="nearest"); a.set_title(title, fontsize=11); a.axis("off")
    handles = [Patch(color=PALETTE[i] / 255, label=names[i]) for i in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10)
    fig.suptitle("CloudTCP baseline — validation labels are auto_draft, not verified ground truth", fontsize=14)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    fig.savefig(output / "qualitative_results.png", dpi=130); plt.close(fig)
    note = f"""# GEO-W3-3 — SegFormer-B0 baseline

Trained on the supplied real image/mask pairs. Labels are **auto_draft**; scores measure agreement with those drafts.

- Dataset: {len(bundle['splits']['train'])} train and {len(bundle['splits']['val'])} validation chips. The supplied group split is preserved; the audit found no train/val footprint overlap.
- Input: 256 × 256 uint8, bands {run['config']['input_bands']} treated as RGB following the pipeline convention. Band 4 is unused; TIFF band descriptions are absent. No image or mask resizing is applied.
- Classes: {', '.join(names)}. Background is retained as class 0, and 255 is ignored.
- Initialization: {BASE_MODEL}, pinned revision {BASE_REVISION}. A new five-output head is used and all layers are fine-tuned.
- Setup: {run['config']['epochs']} requested epochs; {h.epoch.max()} completed; batch size {run['config']['batch_size']}; seed {run['config']['seed']}; device {run['device']}; AdamW encoder/decoder learning rates {run['config']['encoder_lr']}/{run['config']['decoder_lr']}.
- Best validation epoch: {run['best_epoch']}; foreground mIoU against the draft labels: {run['metrics']['foreground_miou']:.4f}. Per-class IoU/F1 and qualitative examples are included.
- Quality limits: exported sampling is approximately 0.6 m per pixel; native sensor GSD and capture date are not recorded here. Thin paint and curb edges can be poorly resolved. Most pixels are background. Auto-generated curb/crosswalk/median labels require visual review, especially in validation. These scores are not independent real-world accuracy.
- Checkpoints: best_model/ is loadable with Hugging Face; last_checkpoint.pt additionally holds the current model, best model, optimizer, mixed-precision scaler, epoch and RNG state for resuming.
- License: the NVIDIA model card links research/evaluation-only terms. Full upstream license is included with best_model/. This is an evaluation baseline.

Next: review validation labels against the RGB imagery, inspect missed thin features, and test on additional independent regions. Do not infer road geometry hidden by trees or shadows from these predictions alone.
"""
    (output / "TRAINING_NOTE.md").write_text(note)
    archive_path = output.parent / (output.name + "_results.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name not in {"last_checkpoint.pt"} and not path.name.endswith((".tmp", ".uploading")):
                archive.write(path, path.relative_to(output))
    if backup:
        for path in output.rglob("*"):
            if path.is_file() and path.name != "last_checkpoint.pt":
                copy_atomic(path, backup / path.relative_to(output))
        copy_atomic(archive_path, backup / archive_path.name)
    return archive_path, metrics_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backup")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = {**DEFAULTS, "epochs": args.epochs, "batch_size": args.batch_size,
              "device": args.device, "resume": not args.no_resume}
    data = inspect_dataset(args.dataset, config["input_bands"])
    run = train_baseline(data, args.output, config, args.backup)
    archive, table = export_results(data, run)
    print(table.to_string(index=False))
    print("Results:", archive)


LICENSE_TEXT = 'NVIDIA Source Code License for SegFormer\n\n1. Definitions\n\n“Licensor” means any person or entity that distributes its Work.\n\n“Software” means the original work of authorship made available under this License.\n\n“Work” means the Software and any additions to or derivative works of the Software that are made available under\nthis License.\n\nThe terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided under\nU.S. copyright law; provided, however, that for the purposes of this License, derivative works shall not include\nworks that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.\n\nWorks, including the Software, are “made available” under this License by including in or with the Work either\n(a) a copyright notice referencing the applicability of this License to the Work, or (b) a copy of this License.\n\n2. License Grant\n\n2.1 Copyright Grant. Subject to the terms and conditions of this License, each Licensor grants to you a perpetual,\nworldwide, non-exclusive, royalty-free, copyright license to reproduce, prepare derivative works of, publicly\ndisplay, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.\n\n3. Limitations\n\n3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this License, (b) you\ninclude a complete copy of this License with your distribution, and (c) you retain without modification any\ncopyright, patent, trademark, or attribution notices that are present in the Work.\n\n3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and\ndistribution of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use\nlimitation in Section 3.3 applies to your derivative works, and (b) you identify the specific derivative works\nthat are subject to Your Terms. Notwithstanding Your Terms, this License (including the redistribution\nrequirements in Section 3.1) will continue to apply to the Work itself.\n\n3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use\nnon-commercially. Notwithstanding the foregoing, NVIDIA and its affiliates may use the Work and any derivative\nworks commercially. As used herein, “non-commercially” means for research or evaluation purposes only.\n\n3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim,\ncross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then\nyour rights under this License from such Licensor (including the grant in Section 2.1) will terminate immediately.\n\n3.5 Trademarks. This License does not grant any rights to use any Licensor’s or its affiliates’ names, logos,\nor trademarks, except as necessary to reproduce the notices described in this License.\n\n3.6 Termination. If you violate any term of this License, then your rights under this License (including the\ngrant in Section 2.1) will terminate immediately.\n\n4. Disclaimer of Warranty.\n\nTHE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING\nWARRANTIES OR CONDITIONS OF M ERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU\nBEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE.\n\n5. Limitation of Liability.\n\nEXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING\nNEGLIGENCE), CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,\nINDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR\nINABILITY TO USE THE WORK (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR\nDATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER COMM ERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN\nADVISED OF THE POSSIBILITY OF SUCH DAMAGES.\n'

if __name__ == "__main__":
    main()
