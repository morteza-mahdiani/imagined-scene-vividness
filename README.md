# imagined-scene-vividness

Companion code for the NeurIPS 2026 Spotlight paper:

**Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models**

Morteza Mahdiani, Catherine Landry, Ian Charest, Frédéric Gosselin, and Jasper JF van den Bosch.

That author order is the order given by the first author in a public announcement of the Spotlight on 2 October 2026. No camera-ready PDF is in the research project this release was prepared from, so affiliations, proceedings pages, and a PDF link are not filled in here.

## Method

The paper studies how much of the vividness with which people imagine a natural scene is predictable from the image. Ratings from a cued-imagery task are aggregated to an image-level target: the mean, across participants, of within-participant z-scored clarity ratings. Frozen vision encoders are each paired with a lightweight probe trained to predict that target. The probes' predictions are then compared with one another.

The comparison implemented here is the post-hoc shared-component analysis. Prediction tables are aligned on common image indices. Each model's correlation with the target is reported with Pearson *r* and Spearman *ρ*. The standardized prediction matrix is decomposed by principal component analysis. The sign of the leading component is arbitrary, so it is flipped when needed to make its correlation with the target non-negative. Each prediction is then linearly regressed on the target; the correlations among those residuals measure agreement that remains after the target has been removed. Optional bootstrap intervals resample images with replacement.

## What this repository supports

This repository runs that comparison on a small synthetic input, and on any directory that uses the same file layout.

Each model is a subdirectory containing `test_predictions.txt` with three whitespace-separated columns and optional `#` comments:

```text
synthetic_idx target prediction
```

The same loader also accepts `test_predictions.csv` with columns `h5_idx`, `target`, and `pred` (or `score`). Targets must agree across files on every shared index. At least three models are required.

The synthetic input has 400 rows and four columns generated from one latent factor plus independent noise. Three columns observe that factor with decreasing weight. The fourth is independent noise. The generator and its coefficients are in `example/run_shared_component.py` (`write_synthetic`). Running with `--expect-synthetic` checks that the strong, mid, and weak columns stay in that order, that the noise column stays near zero correlation, and that the first component explains more variance than the second.

Verified by running that command in this environment: the script completes and writes `example/output/summary.json` and `example/output/residual_agreement.csv`. Those outputs describe the synthetic input only.

## What this repository does not support

It does not train probes or encoders, read Natural Scenes Dataset images, or include participant-level ratings. It does not compute the noise ceiling, memorability controls, observer-level models, or the other analyses in the paper. The numbers it prints are not the coefficients reported in the paper, and this release is not a reproduction of those results. A full rerun was not verified here.

Paper PDF: `PLACEHOLDER_PDF_URL`
OpenReview: `PLACEHOLDER_OPENREVIEW_URL`
DOI: `PLACEHOLDER_DOI`

No license is included. Ownership of the code and of the video still needs to be confirmed before one is added.

## Example

Python 3.11 and NumPy are sufficient. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python example/run_shared_component.py \
  --predictions-dir example/synthetic_runs \
  --out-dir example/output \
  --expect-synthetic
```

To regenerate the synthetic tables before the analysis:

```bash
python example/run_shared_component.py \
  --write-synthetic \
  --predictions-dir example/synthetic_runs \
  --out-dir example/output \
  --expect-synthetic
```

`--n-boot` defaults to 1000 image resamples. The seed for the bootstrap and for `--write-synthetic` defaults to 0.

## Code credit

The procedure is adapted from `scripts/pca_residual_analysis.py` in the research codebase used for the paper. That repository's commits are authored by Morteza Mahdiani. The other paper authors are not credited here as code contributors, because those commits do not list them. This public example was prepared as a separate, reduced script so the release could run without the private rating tables or the rest of the training stack. Its git history starts with this release; the research repository's history was not imported, because that history contains participant-level data.

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

Machine-readable citation metadata is in `CITATION.cff`.

## Video

`video/vividness_15s.mp4` is the outreach clip supplied with the research project. It is 1080×1080, H.264, 30 fps, and 43.01 seconds long; the filename does not match the duration. The file also contains a silent AAC audio track. The closing card reads “Modeling the Vividness of Imagined Natural Scenes” and “NeurIPS 2026”. A hosted URL for the clip is `PLACEHOLDER_VIDEO_URL`.
