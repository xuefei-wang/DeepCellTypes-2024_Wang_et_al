# Figure colors for the DeepCell Types reproduction notebooks.
"""Color mapping for the 51 cell types.

The archive's ``root.attrs["color_mapping"]`` was ad-hoc and not
color-blind-friendly. This module rebuilds the mapping deterministically
from the lineage assignments so that:

  * Each LINEAGE gets a distinct hue chosen for color-vision-deficiency
    safety (palette anchors fail to collide under all three of deutan,
    protan, tritan simulation when tested against CIEDE2000 ΔE > 5).
  * Each CELL TYPE within a lineage gets a unique (hue, lightness)
    coordinate so adjacent types are perceptually separable.
  * Within-lineage ORDERING follows biological similarity (T-cell
    subtypes contiguous, granulocytes contiguous, GI-luminal epithelial
    contiguous, etc.) so confusion-matrix blocks read as block-diagonal
    when biologically-related types confuse each other.

The anchor + ordering choices come from a three-agent review pass:
CB-safety analysis (Viénot/Brettel dichromacy simulation + CIEDE2000),
perceptual distinguishability (CIEDE2000 over the 51-color set), and
published convention (Schürch 2020 CRC CODEX, Goltsev 2018 CODEX,
HuBMAP intestine CODEX 2023). See git history for the review reports.

Use ``celltype_color_mapping(config)`` to obtain the dict — pass the
config so the function reads the same lineage_mapping the
archive declares.
"""

from __future__ import annotations

import colorsys
from collections import defaultdict
from typing import Dict, List


# Per-lineage hue anchor. Choices below resolve the 3 cross-lineage
# collision groups that earlier Okabe-Ito anchors produced:
#   * Stromal #E69F00 (orange) collapsed onto Lymphocyte #D55E00
#     (vermillion) under deutan/protan — both go to ~yellow-brown when
#     the L-M channel is lost. Replaced with olive #999933 (still warm
#     but desaturated, no red-green dependence).
#   * Nerve #56B4E9 (sky blue) collapsed onto Endothelial #0072B2 (blue)
#     under all 3 CVDs (same hue, only lightness gap). Replaced with
#     purple #7A4FB8 — orthogonal hue, conventional for nervous tissue
#     (Allen Brain Atlas).
#   * Tumor #1A1A1A (near-black) disappeared as background/ink in print.
#     Replaced with crimson #B30000 — saturated alarm color, still
#     distinct from lymphocyte vermillion (different L and slight hue
#     shift) and from myeloid reddish-purple.
LINEAGE_ANCHORS: Dict[str, str] = {
    "Lymphocyte":  "#D55E00",   # vermillion (Okabe-Ito)
    "Myeloid":     "#CC79A7",   # reddish purple (Okabe-Ito)
    "Epithelial":  "#009E73",   # bluish green (Okabe-Ito)
    "Endothelial": "#0072B2",   # blue (Okabe-Ito)
    "Stromal":     "#999933",   # olive — CB-safe alternative to orange
    "Nerve":       "#7A4FB8",   # purple — CB-safe alternative to sky-blue
    "Tumor":       "#B30000",   # crimson alarm
    "Other":       "#777777",   # neutral gray
}

# Per-lineage lightness range (HLS L). Tightened upper bounds so the
# lightest shade in a populous lineage doesn't wash out to white-on-cyan
# (which previously made Goblet/Paneth/Hepatocyte indistinguishable under
# tritan and visually identical to readers with normal vision).
LIGHTNESS_RANGE: Dict[str, tuple] = {
    "Lymphocyte":  (0.30, 0.58),
    "Myeloid":     (0.28, 0.62),
    "Epithelial":  (0.30, 0.55),   # capped to keep ramp in green not cyan
    "Endothelial": (0.32, 0.62),
    "Stromal":     (0.30, 0.55),   # olive bleaches at L > 0.6
    "Nerve":       (0.35, 0.62),
    "Tumor":       (0.32, 0.32),
    "Other":       (0.40, 0.60),
}

# Per-lineage hue spread and bias. The ramp covers hue range
# [h0 + bias·dh - dh, h0 + bias·dh + dh] where dh = HUE_DELTA and
# bias = HUE_BIAS (-1 = ramp stays *below* the anchor, +1 = stays
# *above*, 0 = symmetric). Without bias, Epithelial's ramp from
# bluish-green (h0=0.41) drifts +dh toward cyan at the top, which
# washes out under any high-L combination. Setting Epithelial bias = -1
# keeps the ramp in [0.37, 0.41] — pure green territory. Endothelial
# bias = +1 keeps blues from drifting toward Nerve's purple.
HUE_DELTA: Dict[str, float] = {
    "Lymphocyte":  0.05,
    "Myeloid":     0.09,           # widened: 10 types, prior 0.06 too tight
    "Epithelial":  0.05,           # range now biased fully into green
    "Endothelial": 0.04,           # biased fully into blue (not purple)
    "Stromal":     0.04,
    "Nerve":       0.03,
    "Tumor":       0.0,
    "Other":       0.03,
}

HUE_BIAS: Dict[str, int] = {
    "Lymphocyte":  0,              # symmetric: dark-red → orange (no yellow tail)
    "Myeloid":     0,
    "Epithelial":  -1,             # ramp toward warm-green, AWAY from cyan
    "Endothelial": +1,             # ramp toward true-blue, AWAY from cyan
    "Stromal":     0,              # symmetric around olive
    "Nerve":       0,
    "Tumor":       0,
    "Other":       0,
}


# Within-lineage biological ordering. Cell types are assigned (hue,
# lightness) coordinates in this order — so adjacent shades are
# biologically related (e.g. CD4T next to CD8T, Neutrophil next to
# Eosinophil), which makes the on-paper color blocks read as
# functional groupings. Any cell type not listed for its lineage falls
# back to alphabetical at the end of that lineage's order.
BIOLOGICAL_ORDER: Dict[str, List[str]] = {
    "Lymphocyte": [
        "CD4T", "CD8T", "Treg", "Tcell",     # T-cell subtypes contiguous
        "NKT", "NK",                          # NK / NKT bridge
        "Bcell", "Plasma",                    # B-lineage at the end
    ],
    "Myeloid": [
        "Neutrophil", "Eosinophil", "Basophil", "Mast",   # granulocytes
        "Monocyte", "Macrophage", "Microglial",            # mononuclear phagocytes
        "Dendritic", "Langerhans",                          # DC family
        "Erythrocyte",                                       # RBC catch-all last
    ],
    "Epithelial": [
        "AlphaCell", "BetaCell", "Endocrine",              # endocrine (pancreas)
        "Enterocyte", "Goblet", "Paneth", "Foveolar", "Club",  # GI luminal
        "Hepatocyte", "Podocyte", "CollectingDuct",         # parenchymal
        "Epithelial",                                        # generic last
    ],
    "Stromal": [
        "Fibroblast", "SmoothMuscle", "Pericyte",
        "Mesangial", "Stellate", "Telocyte",
        "Keratocyte", "ICC",
        "Stromal",                                           # generic last
    ],
    "Endothelial": [
        "BloodVesselEndothelial", "Endothelial",
        "HSEC", "LittoralCell",
        "LymphaticEndothelial",
    ],
    "Nerve": [
        "Neuron", "Astrocyte", "Glial", "Photoreceptor",
    ],
}


def _hex_to_hls(hex_color: str):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255),
                                        int(b * 255))


def _ordered_cts(lineage: str, cts: List[str]) -> List[str]:
    """Sort cts by BIOLOGICAL_ORDER[lineage]; types not listed fall
    through to alphabetical at the end. Stable so re-runs produce the
    same mapping."""
    order = {name: i for i, name in enumerate(BIOLOGICAL_ORDER.get(lineage, []))}
    return sorted(cts, key=lambda ct: (order.get(ct, len(order)), ct))


def _shades_for_lineage(lineage: str, n: int):
    """Return ``n`` distinguishable hex shades of the lineage anchor.

    Each shade is a (hue, lightness) pair: hue spread linearly across
    [h0 - HUE_DELTA, h0 + HUE_DELTA] and lightness spread across
    LIGHTNESS_RANGE[lineage]. The two axes vary together, giving every
    cell type a unique perceptual coordinate while keeping the lineage
    identity legible.
    """
    anchor = LINEAGE_ANCHORS.get(lineage, "#A0A8B4")
    h0, _, s = _hex_to_hls(anchor)
    s = max(s, 0.45)   # bump pastel anchors to keep saturation legible
    # Cap saturation at 0.85 so pure-hue + high-L combinations don't
    # bleach to washed-out near-whites (e.g. green+L=0.55 → near-cyan).
    s = min(s, 0.85)
    if n == 1:
        return [anchor]
    lo, hi = LIGHTNESS_RANGE.get(lineage, (0.3, 0.75))
    dh = HUE_DELTA.get(lineage, 0.0)
    bias = HUE_BIAS.get(lineage, 0)
    # Hue range: [h0 + bias·dh - dh, h0 + bias·dh + dh].
    # bias=+1 → [h0,        h0 + 2·dh]
    # bias= 0 → [h0 - dh,   h0 + dh]
    # bias=-1 → [h0 - 2·dh, h0]
    h_lo = h0 + (bias - 1) * dh
    h_hi = h0 + (bias + 1) * dh
    return [_hls_to_hex(
        (h_lo + (h_hi - h_lo) * i / (n - 1)) % 1.0,
        lo + (hi - lo) * i / (n - 1),
        s,
    ) for i in range(n)]


def celltype_color_mapping(config) -> Dict[str, str]:
    """Build the 51-entry cell-type -> hex color dict.

    Cell types are grouped by lineage (per ``config.lineage_mapping``),
    ordered within each group by BIOLOGICAL_ORDER, then assigned shades
    along the lineage's (hue, lightness) ramp.
    """
    lineage_map = config.lineage_mapping
    groups = defaultdict(list)
    for ct, lin in lineage_map.items():
        groups[lin].append(ct)

    mapping: Dict[str, str] = {}
    for lin, cts in groups.items():
        cts_ordered = _ordered_cts(lin, cts)
        shades = _shades_for_lineage(lin, len(cts_ordered))
        for ct, color in zip(cts_ordered, shades):
            mapping[ct] = color
    return mapping
