"""Resolve data + archive locations for the figure notebooks.

DATA_ROOT defaults to ``<repo>/data`` (where the derived inputs are copied).
The ``expanded-tissuenet.zarr`` archive defaults to the local default path;
public users override via the DATA_DIR env var after downloading the release
archive.
"""
from __future__ import annotations

import os
from pathlib import Path

# notebooks/dct_figures/paths.py -> repo root is two parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("DCT_DATA_ROOT", REPO_ROOT / "data"))

OUTPUT = DATA_ROOT / "output"
SPLITS = DATA_ROOT / "splits"
FIGURES_DATA = DATA_ROOT / "figures_data"


def _resolve_zarr(env_var: str, default: str) -> Path:
    """Honor ``$env_var`` only when it points at a real zarr archive.

    A globally-set ``DATA_DIR`` (this research box exports ``DATA_DIR=/data``)
    that is *not* a zarr archive would otherwise silently misdirect the
    archive-reading notebooks. We require a ``zarr.json`` at the path and
    reject the legacy "outer wrapper" layout (a same-named subdirectory that
    holds the real archive). Falls back to the known-good local ``default``.
    """
    cand = os.environ.get(env_var)
    if cand:
        c = Path(cand)
        if (c / "zarr.json").is_file() and not (c / c.name).is_dir():
            return c
    return Path(default)


EXPANDED_TISSUENET_ZARR = _resolve_zarr("DATA_DIR", str(DATA_ROOT / "expanded-tissuenet.zarr"))


def need(p: Path) -> Path:
    """Return ``p`` or raise an actionable error if it is missing."""
    if not Path(p).exists():
        raise FileNotFoundError(
            f"Required input not found: {p}\n"
            "Run the data-copy step (see notebooks/README) or set DCT_DATA_ROOT / "
            "DATA_DIR."
        )
    return Path(p)
