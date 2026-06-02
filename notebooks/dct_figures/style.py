# Figure styling for the DeepCell Types reproduction notebooks.
"""Shared visualization style for the figure notebooks.

Single source of truth for color palette, rcParams, colormaps, layout
constants, and helpers used across every plotting notebook.

Style follows a consistent publication figure style:

- Column widths
    - single column : 3.4 in (8.6 cm)
    - 1.5 column    : 4.5 in (11.4 cm)  (used for moderately-wide single panels)
    - full / double : 7.3 in (18.3 cm)
- Sans-serif font family (Helvetica / Arial / DejaVu Sans fallback)
- Axis text 7.5–8 pt (6 pt minimum), tick text 6.5 pt, panel labels 9 pt bold
- Hairline axes (0.5 pt), data line weights 0.6–1.5 pt
- Vector PDF + 600 dpi PNG outputs
- Colorblind-safe palette anchored on the project theme color #565D8B

Cell-count axes are ALWAYS log-scaled (see ``use_log_count_axis``) so the
six orders of magnitude spanned by the archive are legible without
clipping rare classes.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Color palette (theme color anchored on #565D8B, Okabe-Ito accents)
# ---------------------------------------------------------------------------

THEME = "#565D8B"          # primary -- replaces the previous PAL["blue"]
THEME_DARK = "#3D4366"     # darker shade for emphasis / outlines
THEME_LIGHT = "#A8AEC6"    # lighter shade for secondary fills

PAL = {
    # Theme aliases (use these in new code)
    "theme": THEME,
    "theme_dark": THEME_DARK,
    "theme_light": THEME_LIGHT,
    # Backwards-compatible name (old scripts still reference PAL["blue"])
    "blue": THEME,
    # Accents (Okabe-Ito where possible -- colorblind safe)
    "red": "#D55E00",       # Okabe-Ito vermillion (safe vs green)
    "teal": "#4DBBD5",
    "green": "#009E73",     # Okabe-Ito bluish green (safe vs red)
    "salmon": "#F39B7F",
    "lavender": "#8491B4",
    "mint": "#91D1C2",
    "yellow": "#E8C547",
    "brown": "#B09C85",
    "dark": "#2E3440",
    "gray": "#7B8794",
    "light_gray": "#D8DEE9",
}

# Lineage palette -- single source of truth lives in analysis.celltype_colors
# (LINEAGE_ANCHORS); imported here so every figure -- bar charts, heatmaps,
# scatter, cell maps -- shares the same colorblind-tested anchors and the
# per-celltype shade ramps stay flush with their parent lineage swatch.
from .colors import LINEAGE_ANCHORS as LINEAGE_COLORS  # noqa: E402

LINEAGE_ORDER = [
    "Lymphocyte", "Myeloid", "Epithelial", "Endothelial",
    "Stromal", "Tumor", "Nerve", "Other",
]

# Stacked / categorical sequences (theme color first)
STACKED_COLORS = [
    THEME, PAL["red"], PAL["teal"], PAL["green"],
    PAL["salmon"], PAL["lavender"], PAL["yellow"],
]


# ---------------------------------------------------------------------------
# Custom colormaps (theme-anchored)
# ---------------------------------------------------------------------------

CMAP_BLUES = LinearSegmentedColormap.from_list(
    "theme_blues",
    ["#FFFFFF", "#E2E4EE", "#A8AEC6", "#7177A2", "#565D8B", "#3D4366"],
)
CMAP_WARM = LinearSegmentedColormap.from_list(
    "custom_warm", ["#FFFFFF", "#F8E8D6", "#E9B685", "#C16C2E", "#6E3A1A"]
)


# ---------------------------------------------------------------------------
# Layout constants (publication figure style)
# ---------------------------------------------------------------------------

# Widths in inches
COL_1 = 3.4          # single column      (86 mm)
COL_1_5 = 4.5        # 1.5 column         (114 mm)
COL_2 = 7.3          # full / 2 column    (183 mm)

# Conservative height ceiling (Sci Adv max printable height is ~9.5 in)
MAX_HEIGHT = 9.5

# Bar plot widths -- use these everywhere for consistency
BAR_WIDTH = 0.78         # single series
BAR_WIDTH_GROUP = 0.38   # two-series grouped (offset by +/- BAR_WIDTH_GROUP/2)
BAR_EDGE_LW = 0.4

# ---------------------------------------------------------------------------
# Performance-bar layout (single source of truth for bar plots that compare
# model variants or report aggregate model metrics).
#
# Why this exists: `ax.bar(..., width=BAR_WIDTH)` is in DATA units, so the
# rendered bar thickness depends on figsize, n_bars, ylabel padding, etc.
# Identical `BAR_WIDTH=0.78` calls produce visually different bars across
# panels. Past attempts patched individual panels and re-drifted as soon as
# a new panel was added. `setup_perf_bar_axes` fixes the axes width in
# INCHES and computes the data-unit bar width that lands on
# `PERF_BAR_INCHES`, so the bar thickness is the same across every
# panel that calls it (regardless of figsize / n_bars).
# ---------------------------------------------------------------------------

PERF_BAR_INCHES = 0.20          # target visual single-bar width (inches)
PERF_BAR_GROUP_INCHES = 0.10    # target visual within-group bar width
PERF_BAR_PITCH_INCHES = 0.32    # target bar-to-bar pitch (bar + gap) when
                                # `setup_perf_bar_axes(..., equal_pitch=True)`
                                # auto-resizes the figure width.

# Axes-margin budget (inches) used by setup_perf_bar_axes. Tuned so the
# 8 pt y-label + 6.5 pt tick labels + 9 pt title + (optional) subtitle fit
# without overflow at both COL_1_5 (4.5") and COL_2 (7.3") widths.
PERF_AXES_MARGINS = dict(left_in=0.85, right_in=0.32,
                         top_in=0.80, bottom_in=0.90)

# Method names recognised as baselines for bar coloring. Match is
# case-insensitive substring on the label (newlines are stripped). DCT
# variants ("DeepCell Types", "From Scratch", "Pretrain+FT", "Ours") fall
# through and receive the theme color.
BASELINE_METHODS = ("XGBoost", "XGB", "MAPS", "CellSighter", "Nimbus")


def is_baseline_method(label: str) -> bool:
    """True iff `label` names a baseline method (case-insensitive substring)."""
    if label is None:
        return False
    lab = str(label).replace("\n", " ").strip().lower()
    return any(b.lower() in lab for b in BASELINE_METHODS)


def method_bar_color(label, *, dct_color=None, baseline_color=None):
    """Color rule for per-method bar charts: baselines are grey, DCT is theme."""
    if is_baseline_method(label):
        return baseline_color or PAL["gray"]
    return dct_color or THEME


def setup_perf_bar_axes(fig, ax, n_groups, *,
                        target_bar_inches=PERF_BAR_INCHES,
                        target_pitch_inches=None,
                        pad_units=0.6, margins=None):
    """Pin axes margins + xlim so each bar renders at a consistent visual width.

    Call this in place of `tight_layout()` for any "model performance" bar
    chart. After this returns, DO NOT call `fig.tight_layout()` — it would
    override the pinned margins and reintroduce the drift this helper exists
    to prevent.

    The returned `bar_w` is the per-bar width (data units) that renders as
    `target_bar_inches` regardless of figsize or n_groups. Grouped charts
    (two-series paired comparisons, etc.) should position each within-group
    bar at `x + (k - (n_per_group-1)/2) * bar_w` so every individual bar
    has the same physical width as a bar in a single-series chart — not
    half of one.

    Two modes:
      * default — keeps the figure's existing width (callers supplied
        `figsize=(COL_1_5, ...)`); only the bar width is pinned. With this
        mode the inter-bar gap shrinks/grows with n_groups.
      * `target_pitch_inches=...` — the figure width is auto-resized so each
        data-unit step (= bar-to-bar pitch) renders at exactly
        `target_pitch_inches`. Combined with `target_bar_inches`, every
        panel that calls with the same arguments has identical bar width
        AND identical inter-bar spacing, regardless of n_groups. Use this
        for panels meant to be visually comparable (e.g. all per-slice
        macro-F1 boards).

    Args:
        fig, ax: matplotlib figure/axes.
        n_groups: number of bar groups along x (one per category).
        target_bar_inches: target visual per-bar width.
        target_pitch_inches: if given, axes width = data_range × this; figure
            width is resized to keep margins constant.
        pad_units: half-width of edge padding in data units.
        margins: per-call override of the global `PERF_AXES_MARGINS`.

    Returns:
        (x, bar_w): x positions (np.arange(n_groups)) and the data-unit width
        that renders as `target_bar_inches` inches.
    """
    import numpy as _np
    m = {**PERF_AXES_MARGINS, **(margins or {})}
    data_range = (n_groups - 1) + 2 * pad_units
    if target_pitch_inches is not None:
        axes_w_in = data_range * target_pitch_inches
        _, fh = fig.get_size_inches()
        fig.set_size_inches(axes_w_in + m["left_in"] + m["right_in"], fh,
                            forward=True)
    fw, fh = fig.get_size_inches()
    fig.subplots_adjust(
        left=m["left_in"] / fw,
        right=1 - m["right_in"] / fw,
        top=1 - m["top_in"] / fh,
        bottom=m["bottom_in"] / fh,
    )
    axes_w_in = fw - m["left_in"] - m["right_in"]
    x = _np.arange(n_groups)
    ax.set_xlim(-pad_units, n_groups - 1 + pad_units)
    bar_w = target_bar_inches * data_range / axes_w_in
    return x, bar_w

# Line/marker widths
LINE_LW = 1.1
LINE_LW_THIN = 0.6
MARKER_SIZE = 4.0

# ---------------------------------------------------------------------------
# Font roles (Science Advances figure spec)
#
# Sci-Adv asks for sans-serif (Helvetica/Arial), 5–9 pt across the figure.
# These named constants are the canonical sizes; reach for them instead of
# typing literal fontsize=10. Anything > 10 pt is non-compliant for the
# print column.
#
#   FS_AXIS_TITLE   panel-internal axes title          9
#   FS_AXIS_LABEL   x/y axis labels                    8
#   FS_BODY         legend, value labels, annotations  7
#   FS_TICK         tick labels                        6.5
#   FS_SMALL        dense per-row/per-column labels    6
#   FS_TINY         very dense (e.g. per-CT yticks)    5
#
#   FS_PANEL_LABEL  bold A/B/C/D corner panel marker  10  (still inside spec)
#   FS_SUPTITLE     figure-wide suptitle               9  (uses FS_AXIS_TITLE)
# ---------------------------------------------------------------------------
FS_AXIS_TITLE = 9.0
FS_AXIS_LABEL = 8.0
FS_BODY = 7.0
FS_TICK = 6.5
FS_SMALL = 6.0
FS_TINY = 5.0
FS_PANEL_LABEL = 10.0
FS_SUPTITLE = 9.0


# ---------------------------------------------------------------------------
# Global rcParams (Science Advances compliant)
# ---------------------------------------------------------------------------

RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.titleweight": "normal",
    "axes.titlepad": 6.0,
    "axes.labelsize": 8.0,
    "axes.labelweight": "normal",
    "axes.labelpad": 4.0,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#666666",
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 7.0,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "figure.dpi": 150,
    "figure.facecolor": "none",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "axes.facecolor": "none",
    "text.color": "#2E3440",
    "axes.labelcolor": "#2E3440",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": LINE_LW,
    "lines.markersize": MARKER_SIZE,
    "patch.linewidth": BAR_EDGE_LW,
    "pdf.fonttype": 42,         # embed real fonts (TrueType), no Type-3
    "ps.fonttype": 42,
}


def apply_style():
    """Apply the shared rcParams to matplotlib."""
    plt.rcParams.update(RCPARAMS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def style_ax(ax, grid_axis="y"):
    """Apply consistent spine + grid styling to an axes.

    Args:
        ax: matplotlib Axes object.
        grid_axis: "y" (default), "x", "both", or None.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.45)
    ax.spines["bottom"].set_alpha(0.45)
    if grid_axis == "y":
        ax.yaxis.grid(True, alpha=0.18, linewidth=0.4, zorder=0)
    elif grid_axis == "x":
        ax.xaxis.grid(True, alpha=0.18, linewidth=0.4, zorder=0)
    elif grid_axis == "both":
        ax.yaxis.grid(True, alpha=0.18, linewidth=0.4, zorder=0)
        ax.xaxis.grid(True, alpha=0.18, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def use_log_count_axis(ax, axis="y", units="cells"):
    """Force a log scale on the given axis with SI-style tick labels.

    Use this on every plot whose axis represents a count (cells, FOVs,
    datasets, markers) so the scale is consistent across the figure set.
    """
    if axis == "y":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_si_fmt))
    else:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_si_fmt))


def _si_fmt(v, _pos=None):
    """Format integer counts as 10, 1K, 10K, 1M (no decimals < 1K)."""
    if v <= 0:
        return "0"
    if v >= 1_000_000:
        return f"{v / 1_000_000:g}M"
    if v >= 1_000:
        return f"{v / 1_000:g}K"
    return f"{int(v)}"


si_fmt = _si_fmt  # public alias; _si_fmt retained for backward compat


def is_narrow_figure(fig, threshold=5.0):
    """True when the figure is narrower than ``threshold`` inches.

    Used by composite-aware plot functions to drop non-essential decorations
    (bar value labels, oversized legends) that overflow at narrow widths.
    The default 5.0 in matches the boundary between Sci-Adv 1.5-column
    (4.5 in) and 2-column (7.3 in), so panels rendered for half-row slots
    (~3 in) and 1.5-column standalone (4.5 in) count as narrow.
    """
    return fig.get_size_inches()[0] < threshold


def add_subtitle(ax, text):
    """Add an italic subtitle below the axes title.

    Title and subtitle are anchored with fixed point offsets (not
    axes-relative fractions), so they never overlap regardless of axes
    height — short axes used to put the y=1.02 subtitle on top of the
    title because 2% of a 1-inch axes is only ~1.4 pt.
    """
    from matplotlib.transforms import ScaledTranslation
    fig = ax.figure
    # Push the main title up so the subtitle has its own line below it.
    title = ax.get_title()
    if title:
        ax.set_title(title, pad=plt.rcParams["axes.titlepad"] + 10)
    # Place subtitle 4 pt above the axes top, in fixed points.
    offset = ScaledTranslation(0, 4 / 72, fig.dpi_scale_trans)
    ax.text(0.5, 1.0, text,
            transform=ax.transAxes + offset,
            ha="center", va="bottom",
            fontsize=7.0, color=PAL["gray"], style="italic")


def add_value_labels(ax, bars, fmt="{:.1f}", fontsize=6.0, offset=1.5,
                     color=None):
    """Add value labels above bar chart bars."""
    color = color or PAL["dark"]
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize,
                color=color,
            )


def panel_label(ax, letter, x=-0.10, y=1.02):
    """Add a Sci-Adv panel label (uppercase, bold, 9 pt) at top-left."""
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=9.0, fontweight="bold",
        color=PAL["dark"],
    )
