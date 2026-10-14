<p align="center">
  <img src="video/spotlight.gif" width="720" alt="Modeling the Vividness of Imagined Natural Scenes">
</p>

# Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models

**NeurIPS 2026 Spotlight**

Morteza Mahdiani, Catherine Landry, Ian Charest, Frédéric Gosselin, and Jasper JF van den Bosch

The paper measures how vividly people can imagine a natural scene, and how much of that image-level pattern is shared by current vision models. Frozen encoders with a lightweight probe predict the image-level target. The leading component of those predictions is the model-common signal. This repository computes that component, the bootstrap intervals on the model–target correlations, and the participant split-half ceiling.

## Run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Predictions are a CSV. `image_id` identifies the row, `target` is the image-level rating, and each remaining column is one model:

```text
image_id,target,model_a,model_b
```

```bash
python -m vividness component --predictions predictions.csv --out results/component
python -m vividness bootstrap --predictions predictions.csv --out results/bootstrap
```

The first command writes the variance explained by each component, the correlation of the first component with the target, the model loadings, and the mean correlation among predictions before and after the target is linearly removed. The sign of the first component is chosen so that correlation is positive. The second command resamples images with replacement and reports a 95% interval for each correlation and for each paired difference.

Trials are a CSV with one rating per row. The default columns are `participant`, `image`, and `rating`.

```bash
python -m vividness ceiling --trials trials.csv --out results/ceiling.json
```

Participants are split into two halves. The correlation of the two per-image means is projected to the full sample with the Spearman–Brown formula.

Column names can be set with `--id-column`, `--target-column`, `--participant-column`, `--image-column`, and `--rating-column`.

Paper PDF: `PLACEHOLDER_PDF_URL`
OpenReview: `PLACEHOLDER_OPENREVIEW_URL`
DOI: `PLACEHOLDER_DOI`

## Citation

```bibtex
@inproceedings{mahdiani2026imaginedvividness,
  title     = {Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models},
  author    = {Mahdiani, Morteza and Landry, Catherine and Charest, Ian and Gosselin, Fr{\'e}d{\'e}ric and van den Bosch, Jasper JF},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {NeurIPS 2026 Spotlight. PDF: PLACEHOLDER_PDF_URL. OpenReview: PLACEHOLDER_OPENREVIEW_URL. DOI: PLACEHOLDER_DOI.}
}
```

`CITATION.cff` carries the same record. The analysis code is by Morteza Mahdiani.
