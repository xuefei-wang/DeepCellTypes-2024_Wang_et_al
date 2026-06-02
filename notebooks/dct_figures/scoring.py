"""Scoring helpers for the DeepCell Types reproduction notebooks.

Pure numpy / pandas. Reproduces the headline cell-type numbers exactly as the
DeepCell Types training code does:

    load prediction CSV -> per-cell argmax over the 51 CT columns ->
    drop cells flagged ``abstained`` in the released predictions ->
    confusion matrix -> ``adjust_conf_mat_hierarchy(CELL_TYPE_HIERARCHY)`` ->
    has-support macro / weighted accuracy + F1.

The model's abstention decision is precomputed and shipped as a boolean
``abstained`` column in the released prediction CSV, so scoring here is a
straight read-and-filter — no thresholds or per-group fences are recomputed.

The scoring pulls two primitives from the DeepCell Types training code:
    - ``CELL_TYPE_HIERARCHY`` (child predictions count as correct for parents)
    - ``adjust_conf_mat_hierarchy``
plus the ordered class list of 51 cell types. All are reproduced below; each
is pure-Python (numpy / dict).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Vendored from deepcell_types.training.hierarchy.CELL_TYPE_HIERARCHY
# (re-exported as deepcell_types.training.config.CELL_TYPE_HIERARCHY).
# Predictions of a child type count as correct when the GT label is the parent.
# ---------------------------------------------------------------------------
CELL_TYPE_HIERARCHY = {
    "Tcell": ["CD4T", "CD8T", "Treg", "NKT"],
    "Stromal": ["Fibroblast", "Pericyte"],
}


# ---------------------------------------------------------------------------
# Ordered class list: the 51 standardized cell types in their canonical
# training-index order (matching the DeepCell Types training code). The
# integer values are an arbitrary-but-fixed bijection; only the name->index
# mapping consistency matters for the confusion-matrix / hierarchy math.
# ---------------------------------------------------------------------------
CT2IDX = {
    "AlphaCell": 0, "Astrocyte": 1, "Basophil": 2, "Bcell": 3, "BetaCell": 4,
    "BloodVesselEndothelial": 5, "CD4T": 6, "CD8T": 7, "Club": 8,
    "CollectingDuct": 9, "Dendritic": 10, "EVT": 11, "Endocrine": 12,
    "Endothelial": 13, "Enterocyte": 14, "Eosinophil": 15, "Epithelial": 16,
    "Erythrocyte": 17, "Fibroblast": 18, "Foveolar": 19, "Glial": 20,
    "Goblet": 21, "HSEC": 22, "Hepatocyte": 23, "ICC": 24, "Keratocyte": 25,
    "Langerhans": 26, "LittoralCell": 27, "LymphaticEndothelial": 28,
    "Macrophage": 29, "Mast": 30, "Melanocyte": 31, "Mesangial": 32,
    "Microglial": 33, "Monocyte": 34, "NK": 35, "NKT": 36, "Neuron": 37,
    "Neutrophil": 38, "Paneth": 39, "Pericyte": 40, "Photoreceptor": 41,
    "Plasma": 42, "Podocyte": 43, "SmoothMuscle": 44, "Stellate": 45,
    "Stromal": 46, "Tcell": 47, "Telocyte": 48, "Treg": 49, "Tumor": 50,
}


# ---------------------------------------------------------------------------
# Vendored from deepcell_types.training.metrics.adjust_conf_mat_hierarchy
# (re-exported as deepcell_types.training.utils.adjust_conf_mat_hierarchy).
# Pure numpy — operates on an (N, N) confusion matrix + ct2idx dict.
# ---------------------------------------------------------------------------
def adjust_conf_mat_hierarchy(conf_mat, hierarchy, ct2idx):
    """Adjust confusion matrix so child predictions count as correct for parents.

    For each parent in ``hierarchy``, moves parent->child counts onto the
    parent diagonal.

    Args:
        conf_mat: (N, N) confusion matrix (rows=true, cols=predicted).
        hierarchy: dict parent-name -> list of child-names.
        ct2idx: dict cell-type-name -> index in ``conf_mat``.
    """
    adjusted = conf_mat.copy()
    for parent, children in hierarchy.items():
        if parent not in ct2idx:
            continue
        parent_idx = ct2idx[parent]
        for child in children:
            if child not in ct2idx:
                continue
            child_idx = ct2idx[child]
            adjusted[parent_idx, parent_idx] += adjusted[parent_idx, child_idx]
            adjusted[parent_idx, child_idx] = 0
    return adjusted


# ---------------------------------------------------------------------------
# Confusion matrix over the kept (non-abstained) cells.
# ---------------------------------------------------------------------------
def hier_conf_mat(csv_path, ct2idx):
    """Hierarchy-adjusted confusion matrix + (n_total, n_kept).

    Loads the prediction CSV, drops any cells flagged in the ``abstained``
    column (the model's abstention decision is precomputed in the released
    predictions), builds the raw confusion matrix over ``ct2idx``, then applies
    :func:`adjust_conf_mat_hierarchy`. CSVs without an ``abstained`` column
    (e.g. the baselines, which use raw argmax) keep every cell.
    """
    df = pd.read_csv(csv_path)
    ct_columns = [c for c in df.columns if c in ct2idx]
    n_total = len(df)
    if "abstained" in df.columns:
        df = df.loc[~df["abstained"].astype(bool)].reset_index(drop=True)
    n_kept = len(df)
    probs = df[ct_columns].values
    pred_names = np.array(ct_columns)[probs.argmax(axis=1)]
    true_names = df["cell_type_actual"].values
    n = len(ct2idx)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(true_names, pred_names):
        if t in ct2idx and p in ct2idx:
            cm[ct2idx[t], ct2idx[p]] += 1
    return adjust_conf_mat_hierarchy(cm, CELL_TYPE_HIERARCHY, ct2idx), n_total, n_kept


# ---------------------------------------------------------------------------
# Per-class precision / recall / F1 from a confusion matrix. Single source of
# truth for the metric definition (zero-filled on divide-by-zero), reused by
# score_csv and importable by the notebooks instead of re-deriving it.
# ---------------------------------------------------------------------------
def prf_from_cm(cm):
    """Per-class (precision, recall, f1, support, has_support) from ``cm``.

    Args:
        cm: (N, N) confusion matrix (rows=true, cols=predicted).

    Returns five arrays aligned to the matrix index order: ``precision``,
    ``recall``, ``f1`` (floats; 0.0 where the denominator is 0), ``support``
    (per-class true count), and ``has_support`` (``support > 0``).
    """
    cm = np.asarray(cm)
    support = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0).astype(float)
    tp = np.diag(cm).astype(float)
    recall = np.zeros_like(tp)
    precision = np.zeros_like(tp)
    np.divide(tp, support, out=recall, where=support > 0)
    np.divide(tp, pred_sum, out=precision, where=pred_sum > 0)
    f1 = np.zeros_like(tp)
    denom = recall + precision
    np.divide(2 * recall * precision, denom, out=f1, where=denom > 0)
    return precision, recall, f1, support, support > 0


# ---------------------------------------------------------------------------
# Reproduces score_csv from the DeepCell Types training code
# ---------------------------------------------------------------------------
def score_csv(csv_path, ct2idx=None):
    """Hier-adjusted macro/weighted accuracy + F1 (all in %).

    Args:
        csv_path: prediction CSV (one row per cell; 51 CT-probability columns
            named after the keys of ``ct2idx``, a ``cell_type_actual`` column,
            and an optional boolean ``abstained`` column flagging cells the
            model declines to call).
        ct2idx: name->index mapping; defaults to :data:`CT2IDX`.

    Returns dict with macro_acc, macro_f1, weighted_acc, weighted_f1, n_cells,
    n_kept, coverage, n_classes_with_support, and per-class breakdowns.
    """
    if ct2idx is None:
        ct2idx = CT2IDX
    cm, n_cells, n_kept = hier_conf_mat(csv_path, ct2idx)
    precision, recall, f1, support, has_support = prf_from_cm(cm)

    macro_acc = recall[has_support].mean() * 100.0
    macro_f1 = f1[has_support].mean() * 100.0
    total = support[has_support].sum()
    if total > 0:
        weighted_acc = ((recall[has_support] * support[has_support]).sum()
                        / total) * 100.0
        weighted_f1 = ((f1[has_support] * support[has_support]).sum()
                       / total) * 100.0
    else:
        weighted_acc = weighted_f1 = float("nan")

    idx2ct = {v: k for k, v in ct2idx.items()}
    per_class_acc = {idx2ct[i]: float(recall[i] * 100.0)
                     for i in range(len(ct2idx)) if has_support[i]}
    per_class_f1 = {idx2ct[i]: float(f1[i] * 100.0)
                    for i in range(len(ct2idx)) if has_support[i]}
    per_class_support = {idx2ct[i]: int(support[i])
                         for i in range(len(ct2idx)) if has_support[i]}

    coverage = float(n_kept) / float(n_cells) if n_cells > 0 else 1.0
    return {
        "csv_path": str(csv_path),
        "n_cells": int(n_cells),
        "n_kept": int(n_kept),
        "coverage": coverage,
        "n_classes_with_support": int(has_support.sum()),
        "macro_acc": float(macro_acc),
        "macro_f1": float(macro_f1),
        "weighted_acc": float(weighted_acc),
        "weighted_f1": float(weighted_f1),
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
    }


def score_many(paths_by_label, ct2idx=None):
    """paths_by_label: dict label -> csv_path. Returns dict label -> score dict."""
    if ct2idx is None:
        ct2idx = CT2IDX
    out = {}
    for label, p in paths_by_label.items():
        if not Path(p).exists():
            print(f"  [score_csv] WARN: {p} not found, skipping {label}")
            continue
        out[label] = score_csv(p, ct2idx)
    return out
