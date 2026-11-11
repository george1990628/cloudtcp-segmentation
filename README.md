# CloudTCP: road-feature segmentation

**Status: in progress; baseline training completed, evaluation and label review ongoing.**

SegFormer-B0 baseline and evaluation tools for segmenting curbs, lane markings, crosswalks, and median boundaries in aerial imagery. Developed as part of CloudTCP at Open Project at Berkeley.

The project works with a small geospatial dataset of 92 image/mask pairs. A 40-epoch baseline was completed with 73 training and 19 validation pairs. Labels are provisional `auto_draft` annotations: evaluation measures agreement with those labels, not independently verified real-world detection accuracy. This repository publishes the implementation; the dataset, trained weights, and exact baseline output files are not bundled.

## Implementation

- Five outputs: background and four road-feature classes; ignore label 255 is excluded from loss and metrics.
- Class-weighted cross-entropy plus Dice loss.
- Matching image/mask flips and right-angle rotations, with image-only brightness/contrast augmentation.
- Checks for label IDs, shape, CRS, transforms, duplicate chips, shared geographic groups, and train/validation footprint overlap.
- Dataset fingerprints and resumable checkpoints to catch accidental changes between runs.
- Per-class IoU/F1, foreground mIoU, loss plots, prediction overlays, and georeferenced prediction TIFF exports.

## Run in Colab

Open [`notebooks/01_train_segformer.ipynb`](notebooks/01_train_segformer.ipynb), select a GPU runtime, and follow the cells. Supply your authorized `dataset_split.zip`. The notebook includes the complete implementation and can save checkpoints to your own Google Drive.

The supplied training default is **20 epochs**, batch size 8. The historical project baseline used **40 epochs**; set the notebook's configuration explicitly when reproducing that schedule. Use a different run name when changing settings.

## Run locally

Use a Python environment with a PyTorch build suitable for your hardware, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python CloudTCP_SegFormer.py --dataset /path/to/dataset_split --output runs/b0 --epochs 40
```

The CLI expects an **extracted dataset directory**, not the ZIP. The Colab notebook handles ZIP extraction. Review [the expected dataset format](docs/dataset_format.md).

## Evaluate an existing run

[`notebooks/02_record_baseline.ipynb`](notebooks/02_record_baseline.ipynb) scores prediction TIFFs already saved by the training run. Supply the original dataset archive and saved run archive. It calculates exact per-class metrics, preserves validation-file hashes, and records any difference from the original training log.

The same scoring utilities are available in [`evaluation/`](evaluation/). Comparing the baseline against itself is a workflow check. Comparisons between models must use matching validation images, masks, class IDs, and scoring rules. If labels are revised, evaluate both models against the same revised references.

## Model and attribution

Uses the Hugging Face Transformers SegFormer implementation and the pretrained `nvidia/segformer-b0-finetuned-ade-512-512` checkpoint, pinned in the source to revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`.

The project contribution is data validation, training orchestration, geospatial export, and evaluation. SegFormer is an upstream model architecture. See [third-party notices](THIRD_PARTY_NOTICES.md); no blanket license is assigned to data, model weights, or team-owned work.

