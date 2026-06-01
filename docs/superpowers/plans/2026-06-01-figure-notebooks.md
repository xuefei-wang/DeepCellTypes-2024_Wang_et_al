# Per-theme Figure Notebooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace this repo's two notebooks with 8 per-theme notebooks that reproduce the research workspace's v10 paper figures with embedded outputs, driven by derived data copied into `data/` plus the local zarr archives.

**Architecture:** A shared `notebooks/dct_figures/` helper package (vendored, torch-free style/colors/scoring) feeds 8 self-contained theme notebooks. Each notebook lifts figure-building code from the corresponding `analysis/plot_*.py` script in the workspace, repoints inputs to `data/`, replaces `save_fig(...)` with inline display, and is executed end-to-end so outputs embed.

**Tech Stack:** Python, Jupyter (nbconvert), numpy/pandas/matplotlib/seaborn/scikit-learn, zarr, pyyaml, s3fs.

**Source workspace (read-only):** `/data/xwang3/Projects/deepcell-types-research-workspace` (referred to below as `$WS`).
**This repo:** `/data/xwang3/Projects/DeepCellTypes-2024_Wang_et_al` (referred to as `$REPO`), branch `new-figures`.
**Local archives:** `/data/xwang3/expanded-tissuenet.zarr`, `/data/xwang3/gold_standard.zarr`.

**Conversion recipe (applies to every notebook task):**
1. Read the cited `$WS/analysis/plot_*.py` source. Locate the panel's figure-building function (the code path leading to its `save_fig(fig, ..., "<panel_name>")` call).
2. Lift that code into a notebook code cell: drop `@click.command`/`@click.option` decorators and CLI arg plumbing; bind former CLI args to literals (defaults from the source).
3. Replace `save_fig(fig, out_dir, name, ...)` with leaving `fig` as the last expression / `plt.show()` so the figure embeds. Drop provenance-sidecar calls.
4. Repoint every input path through `dct_figures.paths` (so it reads from `$REPO/data/` or the zarr env var).
5. Shared style/colors/scoring import from `dct_figures`; per-panel plotting stays inline.
6. Each panel gets: a markdown cell (caption from `$WS/figures/figures.yaml`) + the code cell + embedded output.

**Verification per notebook:** execute headless and assert zero errors and embedded outputs:
```bash
cd $REPO && PYTHONPYCACHEPREFIX=$HOME/.cache/pycache \
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 \
  notebooks/<name>.ipynb
```
Then confirm at least one image/output is embedded:
```bash
python3 -c "import json,sys; nb=json.load(open('notebooks/<name>.ipynb')); n=sum(1 for c in nb['cells'] if c['cell_type']=='code' for o in c.get('outputs',[]) if o.get('output_type') in ('display_data','execute_result') and 'image/png' in o.get('data',{})); print('embedded images:',n); sys.exit(0 if n>0 else 1)"
```

---

## Task 0: Scaffolding — dirs, requirements, data copy, paths helper

**Files:**
- Create: `$REPO/notebooks/dct_figures/__init__.py`
- Create: `$REPO/notebooks/dct_figures/paths.py`
- Modify: `$REPO/requirements.txt`
- Create (data, gitignored): `$REPO/data/...` (see copy list)

- [ ] **Step 1: Create the helper package dir + `__init__.py`**

```bash
mkdir -p $REPO/notebooks/dct_figures
printf '"""Shared figure helpers vendored from the deepcell-types research workspace."""\n' > $REPO/notebooks/dct_figures/__init__.py
```

- [ ] **Step 2: Populate `requirements.txt`**

Write `$REPO/requirements.txt`:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
pyyaml
zarr>=3
s3fs
tifffile
scikit-image
```

- [ ] **Step 3: Copy derived data into `data/`**

The declared `inputs:` for in-scope panels (verified present in `$WS`). Copy with `cp -L` to dereference the checkpoint symlinks:
```bash
cd $WS
mkdir -p $REPO/data/output $REPO/data/output/tuning $REPO/data/output/few_shot/keren \
         $REPO/data/embeddings $REPO/data/splits $REPO/data/config $REPO/data/figures_data
# prediction CSVs + baselines (cp -L resolves symlinks into $CHECKPOINT_DIR)
for f in final_frozen_cls_test_prediction.csv final_finetuned_test_prediction.csv \
         final_xgb_plain_test_prediction.csv baseline_xgb_tuned_test_prediction.csv \
         baseline_maps_test_prediction.csv baseline_cellsighter_test_prediction.csv \
         final_frozen_cls_test_mp_preds.csv final_frozen_cls_gold_preds.csv \
         final_frozen_cls_attn_prediction.csv ssl_ablation.csv ; do
  cp -L "output/$f" "$REPO/data/output/$f"; done
# json/npz derived
for f in final_frozen_cls_mp_thresholds.json final_frozen_cls_mp_all_datasets.json \
         final_finetuned_mp_all_datasets.json iqr_sweep_frozen_cls.json \
         ssl_pretext_metrics.json ; do cp -L "output/$f" "$REPO/data/output/$f"; done
for f in cls_token_embedding_final_frozen_cls.npz cls_token_meta_final_frozen_cls.npz \
         final_frozen_cls_attn_mp_artifacts.npz ; do cp -L "output/$f" "$REPO/data/output/$f"; done
cp -L output/tuning/xgb_tune_v10_0_history.csv $REPO/data/output/tuning/
cp -L output/few_shot/keren/metrics.csv $REPO/data/output/few_shot/keren/
# embeddings, splits, config, cached channel stats, gold-standard groundtruth
cp -L embeddings/svd_512.npz $REPO/data/embeddings/
cp -L splits/fov_split_v10.json splits/fov_split_v10_test.json splits/fov_split_v10_valsubset.json $REPO/data/splits/
cp -L config/combined_celltypes.yaml $REPO/data/config/
cp -L figures/data_stats/_data/channel_intensity_stats.npz $REPO/data/figures_data/
mkdir -p $REPO/data/gold_standard/gold_standard_labelled
cp -L data/gold_standard/gold_standard_labelled/gold_standard_groundtruth.csv $REPO/data/gold_standard/gold_standard_labelled/
```
Note: `mp_summary_comparison` / `mp_gold_*` panels read additional MP files; while building `marker_positivity.ipynb` (Task 6), inspect `$WS/analysis/plot_mp_analysis.py` default paths and copy any further referenced files into `$REPO/data/output/` then.

- [ ] **Step 4: Write `paths.py`**

Write `$REPO/notebooks/dct_figures/paths.py`:
```python
"""Resolve data + archive locations for the figure notebooks.

DATA_ROOT defaults to ``<repo>/data`` (where Task 0 copied derived inputs).
The zarr archives default to the local research-box paths; public users
override via the DATA_DIR / GOLD_ZARR env vars after downloading the
release archive.
"""
from __future__ import annotations

import os
from pathlib import Path

# notebooks/dct_figures/paths.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DCT_DATA_ROOT", REPO_ROOT / "data"))

OUTPUT = DATA_ROOT / "output"
EMBEDDINGS = DATA_ROOT / "embeddings"
SPLITS = DATA_ROOT / "splits"
CONFIG = DATA_ROOT / "config"
FIGURES_DATA = DATA_ROOT / "figures_data"
GOLD_GT = DATA_ROOT / "gold_standard" / "gold_standard_labelled" / "gold_standard_groundtruth.csv"

EXPANDED_TISSUENET_ZARR = Path(
    os.environ.get("DATA_DIR", "/data/xwang3/expanded-tissuenet.zarr")
)
GOLD_ZARR = Path(os.environ.get("GOLD_ZARR", "/data/xwang3/gold_standard.zarr"))


def need(p: Path) -> Path:
    """Return ``p`` or raise an actionable error if it is missing."""
    if not Path(p).exists():
        raise FileNotFoundError(
            f"Required input not found: {p}\n"
            "Run the data-copy step (see notebooks/README) or set DCT_DATA_ROOT / "
            "DATA_DIR / GOLD_ZARR."
        )
    return Path(p)
```

- [ ] **Step 5: Smoke-test the package imports**

Run: `cd $REPO && python3 -c "from notebooks.dct_figures import paths; print(paths.OUTPUT, paths.OUTPUT.exists())"`
Expected: prints the data/output path and `True`.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add notebooks/dct_figures/__init__.py notebooks/dct_figures/paths.py requirements.txt
git commit -m "scaffold: dct_figures package, paths helper, requirements"
```
(`data/` is gitignored; not committed.)

---

## Task 1: Helper package — style, colors, scoring (vendored, torch-free)

**Files:**
- Create: `$REPO/notebooks/dct_figures/style.py`
- Create: `$REPO/notebooks/dct_figures/colors.py`
- Create: `$REPO/notebooks/dct_figures/scoring.py`
- Test: `$REPO/notebooks/dct_figures/_selftest.py`

- [ ] **Step 1: Vendor `colors.py`**

Copy `$WS/analysis/celltype_colors.py` to `$REPO/notebooks/dct_figures/colors.py` verbatim (it imports only `colorsys`/stdlib + matplotlib — confirm with `grep -E "^import|^from" $WS/analysis/celltype_colors.py`; it must not import `deepcell_types`/`torch`). Add a top-of-file provenance comment: `# Vendored from deepcell-types-research-workspace analysis/celltype_colors.py`.

- [ ] **Step 2: Vendor `style.py`**

Copy `$WS/analysis/style.py` to `$REPO/notebooks/dct_figures/style.py`. Then edit:
- Change `from analysis.celltype_colors import LINEAGE_ANCHORS as LINEAGE_COLORS` to `from .colors import LINEAGE_ANCHORS as LINEAGE_COLORS`.
- Delete `save_fig`, `_write_provenance_sidecar`, and any `subprocess`/`git` provenance code (notebooks display inline; no sidecars).
- Keep `apply_style`, `style_ax`, `use_log_count_axis`, `RCPARAMS`, palettes, font-size constants, `setup_perf_bar_axes`, `method_bar_color`, `is_baseline_method`, `add_value_labels`, `panel_label`, `add_subtitle`, `is_narrow_figure`, colormaps.
- Add provenance comment at top.

- [ ] **Step 3: Vendor `scoring.py` (the headline-number primitives)**

Inspect the upstream sources for the symbols imported by `$WS/analysis/_score_csv.py`, `hierarchical_accuracy.py`, `ct_abstention_iqr.py`:
```bash
python3 -c "import deepcell_types.training.config as c, deepcell_types.training.utils as u, deepcell_types.training.abstention as a, inspect; print(inspect.getsourcefile(c)); print(inspect.getsourcefile(u)); print(inspect.getsourcefile(a))"
```
Read `CELL_TYPE_HIERARCHY`, `adjust_conf_mat_hierarchy`, `compute_iqr_fence`, and the `TissueNetConfig` fields actually used by `_score_csv.py`. Copy these **pure-Python** definitions into `$REPO/notebooks/dct_figures/scoring.py` (provenance comment per symbol). Also port the reusable scoring functions from `$WS/analysis/_score_csv.py` (e.g. the hierarchical-collapse + IQR-abstention + macro/weighted F1 computation that `plot_experiment_results.py`, `macro_f1_by_slice.py`, `plot_calibration_and_abstention.py` rely on) — adapted to import from `.scoring`/`.colors` instead of `deepcell_types`/`analysis`.

- [ ] **Step 4: Torch-free check**

Run: `cd $REPO && python3 -c "import sys; import notebooks.dct_figures.scoring, notebooks.dct_figures.style, notebooks.dct_figures.colors; assert 'torch' not in sys.modules, 'torch leaked in'; print('torch-free OK')"`
Expected: `torch-free OK`. If `torch` leaks in, a vendored function pulled a torch dependency — replace that function's body with a numpy/stdlib equivalent (the abstention/hierarchy math is array-level), or as a documented fallback add `deepcell-types @ git+https://github.com/vanvalenlab/deepcell-types` to `requirements.txt` and import from it. Prefer the numpy rewrite.

- [ ] **Step 5: Self-test against the headline number**

Write `$REPO/notebooks/dct_figures/_selftest.py` that loads `data/output/final_frozen_cls_test_prediction.csv`, applies the vendored hierarchical-collapse + IQR-fence (k=0.2) scoring, and prints macro_acc / macro_F1 / weighted_acc / weighted_F1.
Run: `cd $REPO && python3 -m notebooks.dct_figures._selftest`
Expected (from workspace README, tolerance ±0.1): macro_acc≈91.72, macro_F1≈83.84, weighted_acc≈95.34, weighted_F1≈95.40. A material mismatch means the vendored scoring diverged — reconcile against `$WS/analysis/_score_csv.py` before proceeding.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add notebooks/dct_figures/style.py notebooks/dct_figures/colors.py notebooks/dct_figures/scoring.py notebooks/dct_figures/_selftest.py
git commit -m "feat: vendor torch-free style/colors/scoring helpers; verify headline F1"
```

---

## Tasks 2–9: Theme notebooks

Each task follows the **Conversion recipe** (top of plan) and the **Verification per notebook** commands. After verification passes, commit with `git add notebooks/<name>.ipynb && git commit -m "feat: add <name> reproduction notebook"`. Tasks 2–9 are independent of each other (each touches only its own `.ipynb` + reads shared read-only `data/`/`dct_figures`); they depend only on Tasks 0–1.

### Task 2: `classification.ipynb`
**Source scripts:** `plot_experiment_results.py`, `plot_model_analysis.py`, `plot_macro_f1_test.py`, `macro_f1_by_slice.py`, `plot_per_class_pr.py`, `plot_lineage_confusion.py`. **Plus** the zero-shot baseline comparison folded from the original `notebooks/results.ipynb` (cells covering DCT vs XGBoost vs MAPS and static-vs-dynamic — preserve it as a "Zero-shot comparison" section reading from `data/` CSVs; if the zero-shot CSVs were S3-only, port the cells verbatim and note the data dependency in markdown).
**Panels:** class_support_distribution, confusion_matrix, confidence_distribution, top_confusion_pairs, per_modality_f1, dann_head, macro_f1_test, macro_f1_by_modality, macro_f1_by_tissue, macro_f1_by_lineage, per_class_pr_scatter, confusion_matrix_lineage, (optional) xgb_tune_v10_0_progress.
- [ ] Build cells per recipe · [ ] Execute headless · [ ] Confirm embedded images · [ ] Commit

### Task 3: `calibration.ipynb`
**Source scripts:** `plot_calibration_and_abstention.py`, `plot_iqr_sweep.py`, `plot_abstention_exemplar.py`.
**Panels:** calibration_reliability, calibration_confidence_by_correctness, iqr_coverage_macro_f1_pareto, iqr_per_class_abstention_rate, iqr_per_group_fence, calibration_reliability_by_lineage, iqr_pareto_macro_f1, iqr_pareto_weighted_f1, fov_abstention_exemplar (reads `EXPANDED_TISSUENET_ZARR`).
- [ ] Build cells · [ ] Execute headless · [ ] Confirm embedded images · [ ] Commit

### Task 4: `latent.ipynb`
**Source script:** `plot_tsne.py`. Inputs `data/output/cls_token_embedding_final_frozen_cls.npz` + `cls_token_meta_final_frozen_cls.npz`.
**Panels:** latent_celltype_final_frozen_cls_nca_tsne, latent_modality_final_frozen_cls_nca_tsne, latent_tissue_final_frozen_cls_nca_tsne.
- [ ] Build cells · [ ] Execute headless · [ ] Confirm embedded images · [ ] Commit

### Task 5: `ssl.ipynb`
**Source scripts:** `plot_finetune_vs_frozen.py`, `plot_ssl_ablation.py`, `plot_ssl_pretext.py`.
**Panels:** per_class_pft_vs_frozen, per_class_delta_pft_vs_frozen, ssl_pretrain_schematic, ssl_finetune_schematic, ssl_metric_bars, ssl_pretext_task.
- [ ] Build cells · [ ] Execute headless · [ ] Confirm embedded images · [ ] Commit

### Task 6: `marker_positivity.ipynb`
**Source scripts:** `plot_mp_analysis.py`, `plot_film_mp_real_curves.py`, `plot_mp_fov_exemplar.py` (reads `GOLD_ZARR`). While building, copy any extra default-path inputs `plot_mp_analysis.py` reads into `data/output/` (Task 0 Step 3 note) and commit them implicitly via the data dir (gitignored).
**Panels:** mp_summary_comparison, mp_gold_per_dataset, mp_per_marker_waterfall, mp_gold_paired_bars, mp_gold_delta_bars, film_mp_decision_cd45, film_mp_decision_cd206, film_mp_decision_sma, film_mp_threshold_histogram, mp_fov_exemplar_cd3_mibi_breast, mp_fov_exemplar_sma_mibi_decidua, mp_fov_exemplar_pdl1_vectra_colon.
- [ ] Build cells · [ ] Execute headless · [ ] Confirm embedded images · [ ] Commit

### Task 7: `fov_exemplars.ipynb` (reads local zarr; slow)
**Source scripts:** `real_fov_examples.py`, `plot_workflow_comparison_real.py`, `plot_attention_single_cell.py`.
**Panels:** fov_examples, fov_traditional_workflow, fov_dct_prediction, fov_ground_truth, attention_single_cell_examples. Add a markdown header noting these read the zarr archive and are slow; public users set `DATA_DIR`.
- [ ] Build cells · [ ] Execute headless (timeout 3600) · [ ] Confirm embedded images · [ ] Commit

### Task 8: `data_statistics.ipynb` (supersedes original; reads local zarr; slow)
**Source scripts:** `plot_dataset_stats.py` (panels 01–12), `plot_channel_stats_extra.py` (panels 13–17). Reads `EXPANDED_TISSUENET_ZARR` + `data/splits/fov_split_v10*.json` + `data/embeddings/svd_512.npz` + `data/figures_data/channel_intensity_stats.npz`. Preserve the original `data_statistics.ipynb`'s simple count-based panels as an intro section if they don't duplicate a zarr panel.
**Panels:** 01_celltype_abundance … 17_marker_celltype_auc (the runnable subset; panels needing per-cell zarr scans are fine — zarr is local).
- [ ] Build cells · [ ] Execute headless (timeout 3600) · [ ] Confirm embedded images · [ ] Commit

### Task 9: `few_shot.ipynb`
**Source script:** `plot_fewshot_curve.py`, input `data/output/few_shot/keren/metrics.csv`. Add markdown noting the scoring step (`score_fewshot.py`, GPU) is upstream; this notebook plots the saved metrics.
**Panels:** fewshot_efficiency_keren.
- [ ] Build cells · [ ] Execute headless · [ ] Confirm embedded images · [ ] Commit

---

## Task 10: Supersede originals, data manifest, README

**Files:**
- Delete: `$REPO/notebooks/results.ipynb`, `$REPO/notebooks/data_statistics.ipynb` (old) — *only after* the new `data_statistics.ipynb` (Task 8) and `classification.ipynb` (Task 2, with zero-shot folded in) are verified. Use `git rm`; history retains them.
- Create: `$REPO/notebooks/README.md`
- Create/Modify: `$REPO/data/required_datasets.yaml`
- Modify: `$REPO/README.md`

- [ ] **Step 1: Confirm coverage before deleting originals**

Verify the new notebooks cover everything published in the originals (basic data stats → `data_statistics.ipynb`; confusion/F1/t-SNE/MP/zero-shot → `classification.ipynb`/`latent.ipynb`/`marker_positivity.ipynb`). List any original panel not reproduced; if found, add it to the relevant new notebook before deleting.

- [ ] **Step 2: Replace old `data_statistics.ipynb`**

The new Task 8 notebook already lives at `notebooks/data_statistics.ipynb`. Ensure it was created as a new file (not appended to the old). Then `git rm notebooks/results.ipynb`.

- [ ] **Step 3: Write `notebooks/README.md`**

Document: the per-theme notebook list with one-line descriptions; the `dct_figures` helper package; the data-copy step (the Task 0 Step 3 commands, generalized) and `required_datasets.yaml`; the `DATA_DIR`/`GOLD_ZARR`/`DCT_DATA_ROOT` env vars; which notebooks need the zarr archive.

- [ ] **Step 4: Update `data/required_datasets.yaml`**

Extend the existing manifest pattern (keys = filenames under the public S3 bucket) to list the newly required derived files, so the existing S3-download cell pattern can fetch them. Compute md5s with `md5sum data/output/*`.

- [ ] **Step 5: Update top-level `README.md`**

Update the "notebooks to reproduce the figures" description to reflect the per-theme set; keep the citation block.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add notebooks/README.md data/required_datasets.yaml README.md
git rm notebooks/results.ipynb
git commit -m "docs: per-theme notebook README + data manifest; retire old results notebook"
```

---

## Task 11: Final review

- [ ] **Step 1: Re-execute all notebooks clean**

Run the headless execute + embedded-image check (Verification commands) for every notebook in `notebooks/*.ipynb`. All must exit 0 with ≥1 embedded image.

- [ ] **Step 2: Dispatch code review**

Use `feature-dev:code-reviewer` (or the `superpowers:requesting-code-review` skill) over the diff on `new-figures`: check the vendored scoring matches upstream semantics, no `deepcell_types`/`torch` import leaks, every in-scope `figures.yaml` panel has a cell, paths resolve via `dct_figures.paths`, and no large derived data was accidentally committed (confirm `git status` shows `data/` untracked).

- [ ] **Step 3: Address review findings, re-verify, final commit**

Fix any blockers/highs surfaced, re-run Step 1 for affected notebooks, commit.

---

## Self-review (plan vs. spec)

- **Spec coverage:** helper package (Task 1) ✓; 8 theme notebooks (Tasks 2–9) ✓; data copy into `data/` (Task 0) ✓; supersede + preserve zero-shot/data-stats (Tasks 2, 8, 10) ✓; vendor scoring + torch-free check (Task 1 Steps 3–4) ✓; headline-number verification (Task 1 Step 5, Task 11) ✓; zarr panels (Tasks 3, 6, 7, 8) ✓; requirements + manifest + README (Tasks 0, 10) ✓; out-of-scope GPU/hand-drawn panels excluded ✓.
- **Placeholder scan:** panel code is lifted from cited existing scripts (not net-new), so per-panel code is referenced by exact source path rather than duplicated — intentional given the code already exists in `$WS`. Net-new code (paths.py, the conversion transforms, verification) is given in full.
- **Consistency:** path symbols (`OUTPUT`, `EMBEDDINGS`, `SPLITS`, `CONFIG`, `FIGURES_DATA`, `GOLD_GT`, `EXPANDED_TISSUENET_ZARR`, `GOLD_ZARR`, `need`) defined once in Task 0 Step 4 and referenced consistently.
