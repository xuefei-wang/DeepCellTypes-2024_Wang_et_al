# Figure-reproduction notebooks

Per-theme notebooks that reproduce the figures in Wang et al. (2024). Each
notebook is self-contained: it reads derived inputs from `../data/` (and, for
a few, the released zarr archives), then renders the panels with embedded
outputs.

## Notebooks

| Notebook | What it produces |
| --- | --- |
| `classification.ipynb` | Cell-type classification results: confusion matrix, per-cell-type / per-modality / per-tissue F1, and a guarded zero-shot comparison section (our model vs. XGBoost / MAPS / static baselines). |
| `calibration.ipynb` | Abstention / calibration analysis, including an abstention exemplar rendered from the zarr archive. |
| `latent.ipynb` | Latent-space visualization (NCA / t-SNE of CLS-token embeddings, colored by cell type and by imaging modality). |
| `marker_positivity.ipynb` | Marker-positivity benchmarks on our test data and on the gold-standard ("nimbus") data, with FOV exemplars rendered from the gold zarr. |
| `ssl.ipynb` | Self-supervised pretraining ablations (pretext-task metrics, SSL vs. supervised, XGBoost tuning history). |
| `fov_exemplars.ipynb` | Field-of-view exemplar montages rendered from the expanded-TissueNet zarr. |
| `data_statistics.ipynb` | Dataset composition statistics (counts by cell type / modality / tissue, channel-intensity stats, FOV split sizes), computed from the zarr archive. |
| `few_shot.ipynb` | Few-shot adaptation results on the Keren held-out dataset. |

## Helper package: `dct_figures`

Shared, torch-free helper package imported by every notebook:

- `paths.py` — resolves data + archive locations (see env vars below) and a
  `need()` guard that raises an actionable error on missing inputs.
- `style.py` — matplotlib/seaborn styling for consistent publication panels.
- `colors.py` — cell-type and imaging-modality color maps.
- `scoring.py` — a **torch-free vendoring** of the `deepcell_types`
  hierarchical-evaluation + IQR-fence abstention primitives
  (`CELL_TYPE_HIERARCHY`, `adjust_conf_mat_hierarchy`, `compute_iqr_fence`,
  and the ordered class list). Pure numpy / pandas / (optional) zarr — no
  `torch` and no `deepcell_types` import required. Reproduces the headline
  cell-type numbers exactly as the workspace reference implementation.

## Data

### Derived inputs (S3)

The small/medium derived inputs (CSV / NPZ / JSON) live under `../data/`,
which is **gitignored**. Download them from the public S3 bucket
`deepcelltypes-2024-wang-et-al` into `data/`, preserving the relative paths.
The full list with md5 checksums is in
[`../data/required_datasets.yaml`](../data/required_datasets.yaml); they land
under `data/output/`, `data/output/tuning/`, `data/output/few_shot/keren/`,
`data/embeddings/`, `data/figures_data/`, `data/splits/`, and
`data/gold_standard/`.

### Zarr archives (separate large release)

Some notebooks read raw imagery / cell crops directly from the released zarr
archives, which are **not** part of the S3 manifest above:

- `calibration.ipynb` — abstention exemplar (expanded-TissueNet zarr).
- `marker_positivity.ipynb` — FOV exemplars (gold zarr).
- `fov_exemplars.ipynb` — FOV exemplar montages (expanded-TissueNet zarr).
- `data_statistics.ipynb` — dataset composition (expanded-TissueNet zarr).

In addition, **all scoring** that applies the per-`(tissue, modality)` IQR
fence grouping needs the zarr to recover the per-cell tissue/modality
metadata used for grouping.

The archives are `expanded-tissuenet.zarr` (~2.4 TB) and
`gold_standard.zarr`. Point the notebooks at them with the env vars below.

### Environment variables

| Var | Purpose |
| --- | --- |
| `DCT_DATA_ROOT` | Override the data root (defaults to `<repo>/data`). |
| `DATA_DIR` | Path to the expanded-TissueNet zarr. Honored **only** if it points at a real zarr archive (a `zarr.json` is present); otherwise it falls back to the local default. This guards against a globally-exported `DATA_DIR` silently misdirecting the archive-reading notebooks. |
| `GOLD_ZARR` | Path to the gold-standard zarr. |

## Running a notebook

Execute in place with embedded outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb
```

The kernel must have the `requirements.txt` dependencies **plus zarr v3**
installed. The registered `dct` Jupyter kernel is the research-workspace venv
and already satisfies these.
