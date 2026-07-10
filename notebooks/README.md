# Figure-reproduction notebooks

Per-theme notebooks that reproduce the figures in Wang et al. (2026). Each
notebook is self-contained: it reads derived inputs from `../data/` (and, for
a few, the released zarr archives), then renders the panels with embedded
outputs.

## Notebooks

| Notebook | What it produces |
| --- | --- |
| `classification.ipynb` | Cell-type classification results: confusion matrix, per-cell-type / per-modality / per-tissue F1, and a cross-method benchmark (our model vs. XGBoost / MAPS / static baselines). |
| `latent.ipynb` | Latent-space visualization (NCA / t-SNE of [CLS]-token embeddings, colored by cell type and by imaging modality). |
| `marker_positivity.ipynb` | Marker-positivity benchmarks: the headline comparison vs Nimbus, a per-marker F1 waterfall, FiLM decision curves, and the learned-threshold histogram. |
| `ssl.ipynb` | Self-supervised pretraining ablations: From Scratch vs Pretrain + Fine-tune metric bars and the masked-marker pretext-task diagnostic. |
| `fov_exemplars.ipynb` | Field-of-view exemplar montages rendered from the expanded-TissueNet zarr. |
| `data_statistics.ipynb` | Dataset composition statistics (counts by cell type / modality / tissue, channel-intensity stats, FOV split sizes), computed from the zarr archive. |
| `few_shot.ipynb` | Few-shot adaptation results on the Keren held-out dataset. |
| `unannotated.ipynb` | **Legacy** — cell-type predictions on unannotated external datasets: XGBoost / CellSighter / MAPS / DeepCell Types spatial maps + predicted composition, rendered from saved per-FOV plot bundles. These were Supplementary Figures 10–12 in earlier preprint versions; the current draft removes them (the out-of-distribution / new-dataset story is now the `preproc-adapt` Supplementary Methods section), so this notebook is retained for reference only and is not a current publication figure. |

## Helper package: `dct_figures`

Shared styling, color maps, and scoring helpers imported by every notebook:

- `paths.py` — resolves data + archive locations (see env vars below) and a
  `need()` guard that raises an actionable error on missing inputs.
- `style.py` — matplotlib/seaborn styling for consistent publication panels.
- `colors.py` — cell-type and imaging-modality color maps.
- `scoring.py` — the hierarchical-evaluation helpers: the
  `CELL_TYPE_HIERARCHY` and ordered class list (`CT2IDX`),
  `adjust_conf_mat_hierarchy`, `prf_from_cm` (per-class
  precision/recall/F1 from a confusion matrix), and `hier_conf_mat` /
  `score_csv` / `score_many` for scoring prediction CSVs. Pure numpy /
  pandas. Reproduces the headline cell-type numbers exactly as the
  DeepCell Types training code.

## Data

Two small config files are committed in the repo and arrive with a clone:
[`../data/lineage_mapping.yaml`](../data/lineage_mapping.yaml) (cell-type →
lineage map used by several notebooks) and
[`../data/required_datasets.yaml`](../data/required_datasets.yaml) (the
download manifest below). **Everything else under `data/` is gitignored** and
must be downloaded as described next.

### Derived inputs (S3)

The small/medium derived inputs (CSV / NPZ / JSON) the notebooks read are
**not** in git. Download them from the public S3 bucket
`deepcelltypes-2024-wang-et-al` into `data/`, preserving the relative paths.
The full list with md5 checksums is in
[`../data/required_datasets.yaml`](../data/required_datasets.yaml); they land
under `data/output/`, `data/output/tuning/`, `data/output/few_shot/keren/`,
`data/figures_data/`, and `data/splits/`.

### Zarr archive (separate large release)

Some notebooks read raw imagery / cell crops or per-cell metadata directly
from the released `expanded-tissuenet.zarr` archive (~2.4 TB), which is
**not** part of the S3 manifest above:

- `fov_exemplars.ipynb` — FOV exemplar montages (raw images + masks).
- `data_statistics.ipynb` — dataset composition (per-cell metadata scan).

Point the notebooks at it with the env vars below.

### Environment variables

| Var | Purpose |
| --- | --- |
| `DCT_DATA_ROOT` | Override the data root (defaults to `<repo>/data`). |
| `DATA_DIR` | Path to the `expanded-tissuenet.zarr` archive. Honored **only** if it points at a real zarr archive (a `zarr.json` is present); otherwise it falls back to the local default. This guards against a globally-exported `DATA_DIR` silently misdirecting the archive-reading notebooks. |

## Running a notebook

Execute in place with embedded outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

Install `requirements.txt` (which includes zarr v3 and Jupyter) and run with
any Python 3 kernel.

### Sanity-check your data download

After downloading the derived inputs, verify they reproduce the paper's
headline cell-type numbers (macro / weighted accuracy and F1) with the
bundled self-test:

```bash
python -m notebooks.dct_figures._selftest
```

It scores `data/output/deepcelltypes_test_prediction.csv` and checks the four
headline metrics against the published values within ±0.1; it prints
`SELFTEST PASS` on success.
