# Dataset contract

Supply an extracted dataset directory containing `pipeline_settings.json`, `manifest.csv`, and the image/mask TIFF files referenced by the manifest. The dataset is not redistributed in this repository.

`pipeline_settings.json` must define `classes` with IDs 0 through 4 and `ignore_label` equal to 255. Class 0 is background. Read the four foreground names from the dataset metadata rather than guessing their numeric order.

The manifest requires `chip`, `split`, `image`, and `mask` columns. Supported split names are `train` and `val`; referenced paths must stay inside the dataset directory. Optional geographic `group` information is checked for overlap between splits.

The baseline uses equal-sized chips, reads input bands 1–3 as RGB, and keeps native image/mask dimensions. Images and masks must agree spatially. The model exports predictions with the source georeferencing.

The original baseline has 73 training and 19 validation chips. Foreground imbalance and provisional annotations are central limitations; pixel accuracy alone is not a sufficient measure of quality.
