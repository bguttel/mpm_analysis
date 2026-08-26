"""
third_transition_main.py
------------------------
Main-text figure for the third transition.

Panel A: P_1 probability linecuts at 3 selected λ values.
Panel B: decay rate (left axis) + oscillation frequency (right axis) vs λ.
Partially uses the stale naming convention of survival for P_1 in the code. Figures all follow the new naming.

"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def make_third_transition_figure(
    panel_b_result,
    panel_a_records,
    *,
    fit_dir: Path | None = None,
    target_lambdas: np.ndarray | None = None,
    lambda_max_b: float | None = 1.55,
    save_dir: Path | None = None,
    figure_stem: str = "third_transition_main_text_figure",
    yticks_freq: list | None = None,
) -> "plt.Figure":
    """Build the two-panel main-text figure.

    Parameters
    ----------
    panel_b_result:
        ``ZenoAnalysisResult`` for panel B (decay / frequency vs λ).
    panel_a_records:
        ``list[ObservableRecord]`` for panel A (p1 linecuts).
    fit_dir:
        If given, overlay fit curves in panel A from this directory.
    target_lambdas:
        Lambda values to select for panel A linecuts.
        Default: ``[0.4, 1.1, 1.5]``.
    lambda_max_b:
        Upper λ cutoff for panel B.
    save_dir:
        If given, save SVG + PDF to this directory.
    figure_stem:
        Filename stem (no extension).
    yticks_freq:
        Y-tick positions for the right frequency axis in panel B.

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import ScaledTranslation

    from mpm_analysis.plotting._style import (
        ONE_COLUMN_WIDTH, LINECUT_COLORS, LABELS_FONT_SIZE, LINE_WIDTH,
        LAMBDA_OBS_COLOR, mm_to_inch, apply_paper_style,
    )
    from mpm_analysis.plotting.survival_curves import plot_survival_curves_panel
    from mpm_analysis.plotting.pole_spectrum import plot_decay_and_freq_panel

    if target_lambdas is None:
        target_lambdas = np.array([0.4, 1.1, 1.5])

    apply_paper_style()

    fig_w = float(ONE_COLUMN_WIDTH)
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_w * 1.0),
        layout="constrained",
    )

    # ---- Panel A ----
    _, plotted_lambdas = plot_survival_curves_panel(
        panel_a_records,
        ax=ax_a,
        target_lambdas=target_lambdas,
        colors=LINECUT_COLORS,
        fit_dir=fit_dir,
        downsample=3,
    )

    # ---- Panel B ----
    ax_b_decay, ax_b_freq = plot_decay_and_freq_panel(
        panel_b_result,
        ax=ax_b,
        lambda_max=lambda_max_b,
        pole_selection="second_smallest_decay",
        downsample=3,
        show_sqrt_fit=True,
        show_critical_line=True,
        yticks_freq=yticks_freq or [0.0, 0.4, 0.8],
    )

    # ---- Lambda arrows below panel B ----
    _add_lambda_arrows(ax_b, plotted_lambdas, LINECUT_COLORS, LINE_WIDTH)

    # ---- Panel labels ----
    _add_panel_letter(ax_a, "A", mm_to_inch, LABELS_FONT_SIZE)
    _add_panel_letter(ax_b, "B", mm_to_inch, LABELS_FONT_SIZE)

    # ---- Save ----
    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_lambda_arrows(ax, lambdas: list[float], colors: list, arrow_lw: float) -> None:
    """Add coloured upward arrows below the x-axis at each plotted lambda."""
    ymin, ymax = ax.get_ylim()
    y_arrow = ymin - 0.04 * (ymax - ymin)
    for lam, color in zip(lambdas, colors):
        ax.annotate(
            "",
            xy=(lam, ymin),
            xytext=(lam, y_arrow),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=arrow_lw),
            zorder=20,
            clip_on=False,
        )


def _add_panel_letter(ax, letter: str, mm_to_inch_fn, fontsize: int) -> None:
    from matplotlib.transforms import ScaledTranslation
    offset = ScaledTranslation(
        xt=mm_to_inch_fn(-4.0),
        yt=mm_to_inch_fn(4.0),
        scale_trans=ax.figure.dpi_scale_trans,
    )
    ax.text(
        x=0.06, y=1.0, s=letter,
        transform=ax.transAxes + offset,
        fontsize=fontsize,
        fontweight="bold",
        ha="left", va="top", usetex=False,
    )
