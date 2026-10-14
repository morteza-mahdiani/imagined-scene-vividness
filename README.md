<p align="center">
  <video src="https://github.com/morteza-mahdiani/imagined-scene-vividness/raw/main/video/vividness_15s.mp4" width="640" controls playsinline></video>
</p>

# Modeling the Vividness of Imagined Natural Scenes Reveals a Model-Common Image-Level Component in Vision Models

**NeurIPS 2026 Spotlight**

Morteza Mahdiani, Catherine Landry, Ian Charest, Frédéric Gosselin, and Jasper JF van den Bosch

People differ in how vividly they can imagine a natural scene. This paper measures that vividness for images from the Natural Scenes Dataset and asks how much of the image-level pattern is captured by current vision models. Frozen encoders, each with a lightweight probe, predict the per-image mean of within-participant z-scored ratings. Their predictions share one leading component. That component tracks the reliable image-level signal, and the models still agree with each other after the human target is removed.

The Spotlight clip is [`video/vividness_15s.mp4`](video/vividness_15s.mp4).

## Reproduce

Each model is a directory containing `test_predictions.txt` (`index`, `target`, `prediction`). Install the dependencies, then point the scripts at a directory of those runs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/pca_residual_analysis.py --runs-dir RUNS --out-dir results/pca
python scripts/bootstrap_significance.py --runs-dir RUNS --out-dir results/bootstrap
python scripts/cross_model_agreement.py --runs-dir RUNS --out-dir results/agreement
python scripts/aggregate_results.py --runs-dir RUNS
python scripts/controls_regression.py --controls-npz CONTROLS.npz --runs-dir RUNS --out-dir results/controls
python scripts/noise_ceiling.py --clean-csv TRIALS.csv --out results/noise_ceiling.json
python scripts/strengthen_paper_analysis.py --runs-dir RUNS --noise-ceiling-json results/noise_ceiling.json --out-dir results/strengthen
```

`predictions/synthetic` is a small simulated set in the same layout. It runs the shared-component and bootstrap scripts without the study files:

```bash
python scripts/pca_residual_analysis.py \
  --runs-dir predictions/synthetic --out-dir results/synthetic_pca \
  --include model_ --n-boot 200

python scripts/bootstrap_significance.py \
  --runs-dir predictions/synthetic --out-dir results/synthetic_bootstrap \
  --include model_ --n-boot 200
```

The Natural Scenes Dataset images, participant ratings, and the paper's prediction tables are not in this repository. `TRIALS.csv` is the trial table (`participation`, `stim1_name`, `rating_zscore`, `include`). `CONTROLS.npz` is the interpretable image-control matrix. `RUNS` also supplies `test_metrics.json` for the summary table and the layer-wise probe outputs used by `strengthen_paper_analysis.py`.

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

`CITATION.cff` carries the same record. The analysis scripts are by Morteza Mahdiani.
