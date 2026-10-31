<h1 align="center">Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models</h1>

<p align="center">
Morteza Mahdiani, Catherine Landry, Jasper JF van den Bosch, Frédéric Gosselin, and Ian Charest
</p>

<p align="center">
  <a href="https://openreview.net/pdf?id=XB3iY2q9Vb"><img alt="Paper" src="https://img.shields.io/badge/Paper-OpenReview-b31b1b?logo=adobeacrobatreader&logoColor=white"></a>
  <a href="https://openreview.net/pdf?id=XB3iY2q9Vb"><img alt="NeurIPS 2026 Spotlight" src="https://img.shields.io/badge/NeurIPS-2026_Spotlight-blue"></a>
  <a href="https://github.com/morteza-mahdiani/imagined-scene-vividness"><img alt="Code" src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white"></a>
  <a href="https://morteza-mahdiani.github.io/imagined-scene-vividness/"><img alt="Project page" src="https://img.shields.io/badge/Project-page-2f6f62"></a>
</p>

<p align="center">
  <img src="video/spotlight.gif" width="720" alt="Modeling the Vividness of Imagined Natural Scenes">
</p>

## Abstract

The visual features associated with vivid mental images remain poorly understood. We collected more than 229,000 vividness judgments for the 73,000 images in the Natural Scenes Dataset in a large online experiment with 1,991 participants. We then trained lightweight MLP readouts on frozen vision-model representations to predict average imagery-vividness ratings from image content. The best individual model reached *r* = 0.376 [0.355, 0.396], approximately 95% of the estimated image-level reliability ceiling (*r* ≈ 0.395). A principal-component analysis of predictions from 14 models identified a shared image-property component that explained 72.3% of cross-model prediction variance and correlated with the measured target at *r* = 0.378. The best tested predictor of variation remaining after this component reached *r* = 0.077, approximately 39% of the corresponding residual-target ceiling (*r* ≈ 0.200). Position, distractor, low-level, caption-derived, coarse-category, memorability, and participant-level analyses constrain the interpretation of this performance. Average vividness ratings are strongly predictable from image content relative to the reliability of the measured image-level target. However, much of the predictable signal is shared across model families and appears to reflect general image properties, while reliable variation remains beyond this shared component. The results therefore establish a robust association between visual content and average vividness ratings, without implying that the models explain the subjective experience of mental imagery; the observer-specific processes that contribute to this experience remain open.

This repository trains the frozen-encoder probe and recomputes the shared component, the model correlations, the split-half bound, and the variance each model adds over control features.

## Models

Pearson *r* on the shared test images. Intervals are 95% bootstrap intervals from 10,000 resamples.

| Rank | Model | Pearson *r* | 95% interval |
| --- | --- | ---: | --- |
| 1 | SigLIP2-B/16 | 0.376 | 0.355, 0.396 |
| 2 | SigLIP-B/16 | 0.368 | 0.347, 0.388 |
| 3 | CLIP-B/16 | 0.356 | 0.335, 0.376 |
| 4 | MAE-B/16 | 0.341 | 0.320, 0.362 |
| 5 | DINOv3-B | 0.339 | 0.318, 0.360 |
| 6 | ResNet-50 | 0.330 | 0.309, 0.351 |
| 7 | DINOv3-L | 0.323 | 0.302, 0.344 |
| 8 | ResNet-18 | 0.321 | 0.300, 0.342 |
| 9 | Qwen2-VL 2B | 0.320 | 0.299, 0.341 |
| 10 | VGG-16 | 0.316 | 0.295, 0.338 |
| 11 | ViT-B/16 | 0.316 | 0.294, 0.338 |
| 12 | DINOv2-B/14 | 0.309 | 0.288, 0.330 |
| 13 | ConvNeXt-T | 0.301 | 0.280, 0.323 |
| 14 | Handcrafted features | 0.094 | 0.071, 0.118 |

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trials are one rating per row. The default columns are `participant`, `image`, and `rating`. The rating column is the within-participant z-scored rating, and the image-level target is its mean within each image. `--zscore` standardizes raw ratings within participant before that average.

```bash
python -m vividness target --trials trials.csv --out results/target.csv
```

Predictions are one row per image. `image_id` identifies the row, `target` is the image-level rating, and each remaining column is one model, read from left to right:

```text
image_id,target,model_a,model_b
```

```bash
python -m vividness component --predictions predictions.csv --out results/component
python -m vividness bootstrap --predictions predictions.csv --out results/bootstrap
python -m vividness ceiling --trials trials.csv --out results/ceiling.json
```

`component` writes the variance explained by each component, the correlation of each component with the target, the model loadings, and the mean and median correlation among predictions before and after the target is linearly removed. The sign of the first component is chosen so that its correlation with the target is positive. It also writes how much variance each model adds beyond that component. `bootstrap` resamples images with replacement, using the same images for every model, and writes Pearson *r* and Spearman *ρ* with a 95% interval, plus the paired difference in Pearson *r*. `ceiling` splits participants into two halves, correlates the two per-image means, and projects that correlation to the full sample with the Spearman–Brown formula. The interval is the 5th to 95th percentile across splits. The square root of the projected value is the bound on the correlation an image-level prediction can reach.

Control features are a second table with the same image id and one column per feature. An optional category column adds the within-category correlation and the correlation after category means are removed.

```text
image_id,feature_a,feature_b,category
```

```bash
python -m vividness controls \
  --predictions predictions.csv \
  --controls controls.csv \
  --category-column category \
  --out results/controls
```

The defaults are 10,000 resamples for the correlations, 1,000 for the other intervals, 50 participant splits, and seed 42. Column names can be set with `--id-column`, `--target-column`, `--participant-column`, `--image-column`, and `--rating-column`.

## Train

The probe is a two-layer MLP with 256 hidden units and a GELU activation. The encoder stays frozen. Training uses AdamW with learning rate 0.001 and weight decay 0.01, mean squared error, and an 80/10/10 image split with seed 42. The defaults are 3 epochs and a patience of 2.

```bash
pip install -r requirements-train.txt
python -m vividness train --list-models
python -m vividness train \
  --table images.csv \
  --model siglip2_b16 \
  --out results/siglip2
```

`images.csv` has `image_id`, `target`, and `image`. `image` is a file path, or a name under `--image-root`. A `--splits` file can supply `train_idx`, `val_idx`, and `test_idx` as image ids. The run writes `head_best.pt`, `test_metrics.json`, and `test_predictions.csv`. Qwen2-VL and the handcrafted features are reported in the table above and are not part of this trainer.

[Paper](https://openreview.net/pdf?id=XB3iY2q9Vb)

## Citation

```bibtex
@inproceedings{mahdiani2026modeling,
  title={Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models},
  author={Mahdiani, Morteza and Landry, Catherine and van den Bosch, Jasper JF and Gosselin, Fr{\'e}d{\'e}ric and Charest, Ian},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/pdf?id=XB3iY2q9Vb}
}
```

`CITATION.cff` carries the same record. The analysis code is by Morteza Mahdiani (m72.morteza@gmail.com).
