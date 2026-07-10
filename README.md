# Generalized cell phenotyping for spatial proteomics with language-informed vision models

> **License notice.** This project is distributed under a *Modified Apache
> License, Version 2.0* with non-commercial / academic-only carve-outs (see
> the [LICENSE](LICENSE) file for the full text). For any other use,
> including commercial use, contact `vanvalenlab@gmail.com`.

This repository reproduces the publication figures (Figures 1–3 and
Supplementary Figures 1–9) from Wang et al., *Generalized cell phenotyping for
spatial proteomics with language-informed vision models*. It contains a set of
per-theme notebooks covering classification, latent-space visualization, marker
positivity, self-supervised pretraining, FOV exemplars, dataset statistics, and
few-shot adaptation, plus a legacy notebook for the external-dataset predictions
that appeared in earlier preprint versions but are not in the current draft. See
[`notebooks/README.md`](notebooks/README.md) for the full notebook list, the
shared `dct_figures` helper package, the data-download step, and how to run each
notebook.

## About DeepCell Types

DeepCell Types is a language-informed vision model for cell phenotyping in
spatial proteomics that generalizes across datasets with different marker
panels and imaging platforms *without per-dataset retraining*. It combines
three components:

- a **visual encoder** (a shared per-channel CNN plus a spatial-context CNN over
  the self-mask, neighbor-mask, and distance transform) that embeds each
  marker's 32×32 cell patch;
- a **language encoder** that embeds each marker/cell-type name — wrapped in a
  short biology-context prompt — with OpenAI's frozen `text-embedding-3-large`,
  giving the model semantic knowledge of what each channel means; and
- a **channel-wise transformer** that fuses the per-marker image and text
  embeddings with self-attention (no positional encoding, so it is invariant to
  panel size and order) and a learnable `[CLS]` token summarizing the cell.

Additional training strategies improve cross-platform generalization: a focal
cross-entropy cell-type head, a FiLM-conditioned marker-positivity head, a
mean-intensity `[CLS]` residual that reintroduces the quantitative gating cue,
and a gradient-reversal modality head that pushes the encoder to unlearn
platform-specific features.

## Expanded TissueNet

The model is trained on **Expanded TissueNet**, a curated dataset assembled from
the literature and the NIH HuBMAP consortium through a human-in-the-loop
labeling framework (DeepCell Label). The labeled subset used for all experiments
comprises:

- ≈ **8.7 million** labeled cells across **2,153** labeled fields of view;
- **7** imaging platforms (IMC, CODEX, MIBI, IBEX, MACSima/MICS, Cell DIVE, and
  CycIF), with most data from the first three;
- **51** specific cell types organized into **8** lineages, and 17 tissue types
  in the labeled subset (21 across the broader archive);
- **278** standardized protein markers (≈ 28.5 markers per FOV).

## Headline results

On the canonical 129-FOV held-out test split (486,705 cells; a file-level FOV
split so no test cell is ever seen in training):

| Task | DeepCell Types | Baselines |
| --- | --- | --- |
| Cell-type classification (macro F1) | **84.4%** | XGBoost 82.2% (78.3% untuned), MAPS 77.2%, CellSighter 73.7% |
| Marker positivity on Expanded TissueNet (macro F1) | **85.6%** | Nimbus 37.8% |
| Marker positivity on Pan-Multiplex Gold Standard (macro F1) | 39.1% | Nimbus **66.8%** |

Each marker-positivity model is strongest on its own native distribution (the
Pan-Multiplex Gold Standard is in-distribution for Nimbus and out-of-distribution
for DeepCell Types). Masked-marker self-supervised pretraining yields modest
additional gains (cell-type +0.8, marker-positivity +3.9 F1 points). See the
paper's Materials and Methods for the full evaluation protocol (macro F1 primary
metric, hierarchical-collapsed scoring, minimum-support floor, per-marker
thresholds).

## Reproducing the figures

Each notebook reads derived inputs from `data/` (and, for a few, the released
zarr archives) and renders its panels with embedded outputs. Install
`requirements.txt` and run, e.g.:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/classification.ipynb
```

Full instructions — including the S3 download of derived inputs, the
`expanded-tissuenet.zarr` archive, environment variables, and a self-test that
checks the reproduced headline numbers against the published values — are in
[`notebooks/README.md`](notebooks/README.md).

## Data availability

The result figures only contain a subset of Expanded TissueNet that includes all
data sourced from public datasets. It is available at
https://vanvalenlab.github.io/deepcell-types/. The remaining datasets were made
available to our lab before their publication. These are available upon
reasonable request and will be made publicly available upon publication of the
corresponding manuscripts. Source code for model inference is available at
https://github.com/vanvalenlab/deepcell-types.

## Citation
```
@article{deepcelltypes,
  title={Generalized cell phenotyping for spatial proteomics with language-informed vision models},
  author={Wang, Xuefei and Dilip, Rohit and Iqbal, Ahamed Raffey and Bussi, Yuval and Brown, Caitlin and Pradhan, Elora and Jain, Yashvardhan and Yu, Kevin and Li, Shenyi and Abt, Martin and B{\"o}rner, Katy and Keren, Leeat and Yue, Yisong and Barnowski, Ross and Van Valen, David},
  journal={bioRxiv},
  pages={2024--11},
  year={2026},
  doi={10.1101/2024.11.02.621624},
  url={https://www.biorxiv.org/content/10.1101/2024.11.02.621624v4},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

Modified Apache License, Version 2.0, with non-commercial / academic-only
carve-outs. See [LICENSE](LICENSE) for the full text. For commercial or other
use, contact `vanvalenlab@gmail.com`.
