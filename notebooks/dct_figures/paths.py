"""Resolve data + archive locations for the figure notebooks.

DATA_ROOT defaults to ``<repo>/data`` (where the derived inputs are copied).
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
