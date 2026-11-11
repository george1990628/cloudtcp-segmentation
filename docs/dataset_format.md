# Dataset format

Start with an extracted dataset folder containing `pipeline_settings.json`, `manifest.csv`, and the image/mask TIFF files listed in the manifest. The dataset itself is not included in this repo.

## Classes

`pipeline_settings.json` must define class IDs 0–4 and set `ignore_label` to 255. Class 0 is background. Use the file's class names to identify the four foreground classes and their order.

## Manifest

The required columns are `chip`, `split`, `image`, and `mask`. Split names must be `train` or `val`, and image/mask paths must stay inside the dataset folder. If a geographic `group` column is present, the code checks that groups do not appear in both splits.

## Images and masks

The baseline expects chips of the same size. It reads bands 1–3 as RGB and keeps the original image and mask dimensions. Each pair must line up spatially. Prediction TIFFs keep the source map coordinates.

The original split contains 73 training and 19 validation chips. Most pixels are background, and the labels are still drafts. Per-class IoU and F1 are more useful than pixel accuracy alone for understanding where the model struggles.
