"""Extract self-contained plot bundles for the SuppFig 10-12 unannotated panels.

Run ONCE in the workspace venv (needs zarr + the workspace ``analysis`` package
and ``deepcell_types`` on PYTHONPATH):

    cd /data/xwang3/Projects/deepcell-types-research-workspace
    PYTHONPATH=.:/path/to/this-figures-repo \
      .venv/bin/python -m notebooks.dct_figures.unannotated_bundle

For each of the 6 paper FOVs it reuses the workspace producer's DATA half
(representative-crop selection, marker composite, per-method cell coloring from
analysis/unannotated_combined.py + analysis/unannotated_viz.py) and SAVES the
resulting arrays instead of plotting:

    data/output/unannotated/<fov>.npz   -- sub_mask, comp RGB, per-method RGB maps
    data/output/unannotated/<fov>.json  -- metadata + composition fractions
    data/output/unannotated/_display.json -- fov -> display name

so notebooks/unannotated.ipynb renders fully offline (numpy/matplotlib only, no
zarr / no model).

Columns per FOV panel: markers (tinted composite) + XGBoost + CellSighter + MAPS + Ours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# workspace producer helpers -- single source of truth for crop + coloring
from analysis.unannotated_combined import _representative_crop
from analysis.unannotated_viz import FALLBACK_COLOR, _marker_composite, _color_cells
from analysis.celltype_colors import celltype_color_mapping
from deepcell_types.training.config import TissueNetConfig

ZARR = "/data/xwang3/expanded-tissuenet.zarr"
# Public-repo output dir, derived from this file's location so it survives a repo rename.
PUB = Path(__file__).resolve().parents[2] / "data/output/unannotated"
PRED_DIR = Path("output/unannotated")
CROP = 1024

# The four prediction columns of the paper SuppFig 10-12 panels (plus the
# markers composite, added in the notebook). NO adapted column.
METHODS = ["xgb", "cellsighter", "maps", "ours"]

# (fov key in zarr, display name from the paper caption -- key == display here)
FOVS = [
    ("liu_gi_mibi_Slide21Stain1_Point13_R8C1", "liu_gi_mibi_Slide21Stain1_Point13_R8C1"),
    ("HBM236_WBFT_443", "HBM236_WBFT_443"),
    ("SP08_8_CR_001", "SP08_8_CR_001"),
    ("liu_reproductive_mibi_Slide21Stain1_Point11_R6C7", "liu_reproductive_mibi_Slide21Stain1_Point11_R6C7"),
    ("HBM564_DSPG_945", "HBM564_DSPG_945"),
    ("HBM222_WQKC_382", "HBM222_WQKC_382"),
]


def _load_cmaps(fov):
    """cell_index -> pred_label per method (only methods with a CSV)."""
    cmaps = {}
    for m in METHODS:
        csv = PRED_DIR / f"{fov}__{m}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            cmaps[m] = dict(zip(df["cell_index"].astype(int), df["pred_label"].astype(str)))
    return cmaps


def extract_one(zf, fov, display, color_map):
    grp = zf[fov]
    mask = grp["preprocessed/mask"][:]
    raw = grp["preprocessed/raw"][:]
    channel_names = list(grp["preprocessed"].attrs.get("channel_names", []))
    attrs = dict(grp.attrs)
    tissue = attrs.get("tissue", "?")
    modality = str(attrs.get("modality", "?")).upper()
    cents_attr = grp["preprocessed"].attrs.get("centroids", {}) or {}
    centroids = {int(k): (float(v[0]), float(v[1])) for k, v in cents_attr.items()}

    cmaps = _load_cmaps(fov)
    # crop reference = OURS predictions (fall back to any available method)
    ref = cmaps.get("ours") or (next(iter(cmaps.values())) if cmaps else {})

    # --- representative crop (mirror unannotated_combined.render_fov, data half) ---
    side = CROP
    r0, c0, _, _ = _representative_crop(mask, centroids, ref, side)
    H, W = mask.shape
    r0 = max(0, min(r0, max(0, H - side)))
    c0 = max(0, min(c0, max(0, W - side)))
    sub_mask = mask[r0:r0 + side, c0:c0 + side]
    raw_crop = raw[:, r0:r0 + side, c0:c0 + side]
    if sub_mask.shape[0] < side or sub_mask.shape[1] < side:
        pm = np.zeros((side, side), dtype=sub_mask.dtype)
        pm[:sub_mask.shape[0], :sub_mask.shape[1]] = sub_mask
        sub_mask = pm
        pr = np.zeros((raw.shape[0], side, side), dtype=raw_crop.dtype)
        pr[:, :raw_crop.shape[1], :raw_crop.shape[2]] = raw_crop
        raw_crop = pr

    comp, used_markers = _marker_composite(raw_crop, channel_names, color_map)
    crop_ids = [int(i) for i in np.unique(sub_mask) if i]

    arrs = {
        "sub_mask": sub_mask.astype(np.int32),
        "comp": np.clip(comp * 255, 0, 255).astype(np.uint8),
    }
    fracs, present_all = {}, set()
    for m in METHODS:
        if m not in cmaps:
            continue
        rgb, present = _color_cells(sub_mask, cmaps[m], color_map)
        present_all |= present
        arrs[f"rgb_{m}"] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        labs = [cmaps[m][i] for i in crop_ids if i in cmaps[m]]
        vc = pd.Series(labs).value_counts(normalize=True) if labs else pd.Series(dtype=float)
        fracs[m] = {str(k): float(v) for k, v in vc.items()}

    methods_present = [m for m in METHODS if f"rgb_{m}" in arrs]
    meta = {
        "fov": fov,
        "display": display,
        "tissue": tissue,
        "modality": modality,
        "n_channels": len(channel_names),
        "side": side,
        "fracs": fracs,
        "used_markers": {str(k): str(v) for k, v in used_markers.items()},
        "present": sorted(present_all),
        # per-cell-type hex so the offline notebook colors bars/legends exactly as
        # the producer's _color() does (Unknown handled separately in the notebook).
        "colors": {ct: str(color_map.get(ct, FALLBACK_COLOR)) for ct in present_all},
        "methods_present": methods_present,
    }
    return arrs, meta


def main():
    PUB.mkdir(parents=True, exist_ok=True)
    cfg = TissueNetConfig(ZARR)
    color_map = celltype_color_mapping(cfg)
    zf = zarr.open_group(ZARR, mode="r")
    disp = {}
    for fov, display in FOVS:
        arrs, meta = extract_one(zf, fov, display, color_map)
        np.savez_compressed(PUB / f"{fov}.npz", **arrs)
        (PUB / f"{fov}.json").write_text(json.dumps(meta, indent=1))
        disp[fov] = display
        print("wrote", fov, "methods:", meta["methods_present"],
              {k: tuple(v.shape) for k, v in arrs.items()})
    (PUB / "_display.json").write_text(json.dumps(disp, indent=1))
    print("wrote _display.json with", len(disp), "FOVs")


if __name__ == "__main__":
    sys.exit(main())
