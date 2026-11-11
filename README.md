# CloudTCP: road-feature segmentation

**In progress**

I'm fine-tuning SegFormer-B0 to identify curbs, lane markings, crosswalks, and median boundaries in aerial images. This is part of my work on CloudTCP at Open Project at Berkeley.

I completed a 40-epoch baseline using 92 image/mask pairs: 73 for training and 19 for validation. The labels are still drafts (`auto_draft`), so the scores measure how closely the model matches those labels. Label review and further evaluation are ongoing.

## What's included

- Training with class-weighted cross-entropy and Dice loss.
- Checks for image/mask alignment, class IDs, duplicate files, and overlap between training and validation areas.
- Per-class IoU and F1 scores, foreground mIoU, loss curves, and prediction overlays.
- Checkpoints for resuming training and GeoTIFF prediction masks that keep the original map coordinates.

The model has five outputs: background and four road-feature classes. Label 255 is ignored in the loss and metrics. Images and masks receive the same flips and right-angle rotations. Brightness and contrast changes apply only to the images.

## Run in Colab

Open [the training notebook](notebooks/01_train_segformer.ipynb), select a GPU runtime, and provide `dataset_split.zip`. The notebook includes the training code and can save checkpoints to Google Drive.

The default is **20 epochs** with a batch size of 8. Set it to **40 epochs** to match the run described above. Use a new run name when changing the settings.

## Run locally

Install a PyTorch build that works with your hardware, then run:

```bash
python -m pip install -r requirements.txt
python CloudTCP_SegFormer.py --dataset /path/to/dataset_split --output runs/b0 --epochs 40
```

Extract the dataset ZIP before using the command-line script. See [the dataset format](docs/dataset_format.md) for the expected files and labels.

## Evaluate a saved run

[The evaluation notebook](notebooks/02_record_baseline.ipynb) calculates per-class scores from the saved prediction TIFFs. Provide the original dataset ZIP and the run's results ZIP. It also keeps the validation-file hashes and records differences from the training log.

The scripts are in [evaluation/](evaluation/). When comparing models, use the same validation images, masks, class IDs, and scoring method. If the labels change, evaluate both models against the updated labels. Comparing a run with itself is just a check that the comparison code works.

## Model and data

I use the Transformers implementation of SegFormer and the pretrained `nvidia/segformer-b0-finetuned-ade-512-512` model at revision `489d5cd81a0b59fab9b7ea758d3548ebe99677da`.

The dataset, trained weights, and original run outputs are not included here. See [the model and license notes](THIRD_PARTY_NOTICES.md) for the NVIDIA terms.
