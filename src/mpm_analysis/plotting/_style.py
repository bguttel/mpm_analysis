"""
_style.py
---------
Shared plotting style constants and helpers.

All plotting functions in this package should import their constants from here.
"""

from __future__ import annotations

import numpy as np

_HAS_ZENO_PLOTTING = False
_PFS = None
_TTFC = None


def _mm_to_inch(mm: float) -> float:
    return mm / 25.4


# ---------------------------------------------------------------------------
# Figure dimensions
# ---------------------------------------------------------------------------


def mm_to_inch(mm: float) -> float:
    return _mm_to_inch(mm)


if _HAS_ZENO_PLOTTING:
    ONE_COLUMN_WIDTH = float(_PFS.ONE_COLUMN_FIGURE_WIDTH)
    TWO_COLUMN_WIDTH = float(_PFS.TWO_COLUMNS_FIGURE_WIDTH)
    DPI = int(_PFS.DPI)
    LABELS_FONT_SIZE = int(_PFS.LABELS_FONT_SIZE)
    MARKER_SIZE = float(_PFS.MARKER_SIZE)
    LINE_WIDTH = float(_PFS.LINE_WIDTH)
    AXES_ARROW_WIDTH = float(_PFS.AXES_ARROW_WIDTH)
    LAMBDA_OBS_COLOR = str(_PFS.LAMBDA_OBS_COLOR)
    SURVIVAL_PROBABILITY_LABEL = str(_PFS.SURVIVAL_PROBABILITY_LABEL)
else:
    ONE_COLUMN_WIDTH = mm_to_inch(88)  # 88 mm — Nature one-column
    TWO_COLUMN_WIDTH = mm_to_inch(180)  # 180 mm — Nature two-column
    DPI = 600
    LABELS_FONT_SIZE = 10
    MARKER_SIZE = 1.5
    LINE_WIDTH = 1.5
    AXES_ARROW_WIDTH = 0.03
    LAMBDA_OBS_COLOR = "#d02500"
    SURVIVAL_PROBABILITY_LABEL = r"$P_{|1\rangle}^{\mathrm{ens}}$"

# Fit line style (shared across all panels)
FIT_LINEWIDTH = LINE_WIDTH * 0.5
FIT_LINESTYLE = (0, (7, 2))  # long dash, short gap

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

if _HAS_ZENO_PLOTTING:
    PURPLE_MAGENTA = str(_TTFC.PURPLE_MAGENTA)
    TEAL = str(_TTFC.TEAL)
    MUSTARD = str(_TTFC.MUSTARD)
    AZURE_BLUE = str(_TTFC.AZURE_BLUE)
    LAVENDER_BLUE = str(_TTFC.LAVENDER_BLUE)
else:
    PURPLE_MAGENTA = "#A13D97"
    TEAL = "#1ABC9C"
    MUSTARD = "#D4AC0D"
    AZURE_BLUE = "#3498DB"
    LAVENDER_BLUE = "#8787de"

# Default linecut colors (for panel A)
LINECUT_COLORS = [PURPLE_MAGENTA, TEAL, MUSTARD]

# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------


def make_ax_lines_wider(
    ax, *, labelsize: float = LABELS_FONT_SIZE, linewidth: float = LINE_WIDTH
) -> None:
    """Apply consistent tick and spine widths to a matplotlib Axes."""
    ax.tick_params(axis="both", which="both", labelsize=labelsize, width=linewidth)
    for sp in ax.spines.values():
        sp.set_linewidth(linewidth)


def panel_letter_text(letter: str) -> str:
    """The panel letter as text for the *active* renderer.

    Under ``text.usetex`` a bold sans letter needs LaTeX markup — matplotlib's
    ``fontweight="bold"`` is ignored by the LaTeX text pipeline.
    """
    import matplotlib.pyplot as plt

    if plt.rcParams.get("text.usetex", False):
        return rf"\textsf{{\textbf{{{letter}}}}}"
    return letter


def add_panel_letter(
    ax,
    letter: str,
    *,
    fontsize: float | None = None,
    x: float = 0.06,
    y: float = 1.0,
    dx_mm: float = -4.0,
    dy_mm: float = 4.0,
    offset_mm: bool = True,
) -> None:
    """Place a bold panel letter near an axes' top-left corner.

    Two placements, both in axes coordinates for ``x``/``y``:

    * ``offset_mm=True`` (default) — the main-text convention: additionally
      nudged from the corner in *millimetres* via a ScaledTranslation, so it
      sits identically regardless of axes size or dpi.
    * ``offset_mm=False`` — plain axes-fraction placement, used by the
      appendix figures (``x=-0.2, y=1.05``: clear of the y-axis label).

    Either way the letter renders through :func:`panel_letter_text`, so the
    typography follows the active renderer.
    """
    transform = ax.transAxes
    if offset_mm:
        from matplotlib.transforms import ScaledTranslation

        transform = transform + ScaledTranslation(
            xt=mm_to_inch(dx_mm),
            yt=mm_to_inch(dy_mm),
            scale_trans=ax.figure.dpi_scale_trans,
        )
    ax.text(
        x=x,
        y=y,
        s=panel_letter_text(letter),
        transform=transform,
        fontsize=LABELS_FONT_SIZE if fontsize is None else fontsize,
        fontweight="bold",
        ha="left",
        va="top",
    )


def latex_available() -> bool:
    """Whether a LaTeX toolchain usable by matplotlib's usetex is installed.

    ``text.usetex`` needs ``latex`` and ``dvipng`` on PATH; without them every
    draw raises.  Cached so the check runs once per process.
    """
    global _LATEX_OK
    if _LATEX_OK is None:
        from shutil import which

        _LATEX_OK = bool(which("latex") and which("dvipng"))
    return _LATEX_OK


_LATEX_OK: bool | None = None


def apply_paper_style(fig=None, axes=None, *, usetex: bool | None = None) -> None:
    """Apply paper-ready rc defaults to the current figure.

    Typography follows the published convention of
    ``ZenoPlotting/base.py`` — LaTeX text rendering with the Computer Modern
    math fontset on a sans-serif family — so figures produced here match the
    paper's other panels.  ``usetex=None`` (default) enables it whenever a
    LaTeX toolchain is installed and silently falls back to mathtext with the
    same fontset otherwise, which keeps headless machines working.

    Call once before plotting, or pass ``fig``/``axes`` directly.
    """
    import matplotlib.pyplot as plt

    if usetex is None:
        usetex = latex_available()

    plt.rcParams.update(
        {
            # --- typography: match ZenoPlotting/base.py ---
            "text.usetex": bool(usetex),
            "font.family": "sans-serif",
            "mathtext.fontset": "cm",
            "font.size": LABELS_FONT_SIZE,
            # --- line/marker weights ---
            "axes.linewidth": LINE_WIDTH,
            "xtick.major.width": LINE_WIDTH,
            "ytick.major.width": LINE_WIDTH,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "savefig.dpi": DPI,
            "figure.dpi": 100,
        }
    )

    if axes is not None:
        axes_list = [axes] if hasattr(axes, "tick_params") else list(axes)
        for ax in axes_list:
            make_ax_lines_wider(ax)
