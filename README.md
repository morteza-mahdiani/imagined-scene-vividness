<p align="center">
  <img src="video/spotlight.gif" width="720" alt="Modeling the Vividness of Imagined Natural Scenes">
</p>

# Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models

**NeurIPS 2026 Spotlight**

Morteza Mahdiani, Catherine Landry, Jasper JF van den Bosch, Frédéric Gosselin, and Ian Charest

The paper measures how vividly people can imagine a natural scene, and how much of that image-level pattern is shared by current vision models. Frozen encoders with a lightweight probe predict the image-level target. The leading component of those predictions is the model-common signal. This repository recomputes that result from the study tables: the shared component, the model correlations, the split-half bound, and the variance each model adds over control features.

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trials are one rating per row. The default columns are `participant`, `image`, and `rating`. The image-level target is the mean of ratings after they are z-scored within participant.

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

`component` writes the variance explained by each component, the correlation of each component with the target, the model loadings, and the mean and median correlation among predictions before and after the target is linearly removed. The sign of the first component is chosen so that its correlation with the target is positive. It also writes how much variance each model adds beyond that component. `bootstrap` resamples images with replacement, using the same images for every model, and writes Pearson *r* and Spearman *ρ* with a 95% interval, plus the paired difference in Pearson *r*. `ceiling` splits participants into two halves, correlates the two per-image means, and projects that correlation to the full sample with the Spearman–Brown formula. The square root of the projected value is the bound on the correlation an image-level prediction can reach.

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

`CITATION.cff` carries the same record. The analysis code is by Morteza Mahdiani.
