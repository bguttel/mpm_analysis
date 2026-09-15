"""
appendix_bootstrap.py
---------------------
Appendix figure: full pole spectrum (decay rates) + imaginary splitting +
real splitting for a single transition.

Replaces ``ZenoPlotting/Noams_code/first_transition_plotting.py::plot_paper_style_bootstrap()``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def draw_full_spectrum_panel(
    ax,
    lambdas: np.ndarray,
    r_med: np.ndarray,
    r_lower: np.ndarray,
    r_upper: np.ndarray,
    *,
    real_axis_scale: str = "linear",
    real_axis_ticks: list | None = None,
    err_kws: dict | None = None,
) -> None:
    """Draw the full-pole-spectrum panel: |Re(e_n)| / Ω_S strands vs λ.

    The shared panel A of the paper bootstrap figures (this module's 3-panel
    sqrt layout and ``first_transition_main``'s 2-panel discriminant layout).
    Decay rates arrive already normalized by Ω_S; the critical-λ line, the
    panel tag and the x-label are the caller's business.
    """
    import matplotlib.ticker as ticker

    from mpm_analysis.plotting._style import (
        LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH, LINECUT_COLORS,
    )

    colors = LINECUT_COLORS + ["gray"]
    if err_kws is None:
        err_kws = dict(
            elinewidth=LINE_WIDTH * 0.6,
            capsize=MARKER_SIZE * 1.2,
            capthick=LINE_WIDTH * 0.6,
            zorder=10,
        )

    for k in range(r_med.shape[1]):
        c = colors[k % len(colors)]
        ax.errorbar(
            lambdas, np.abs(r_med[:, k]),
            yerr=[r_lower[:, k], r_upper[:, k]],
            color=c, marker="o",
            markersize=MARKER_SIZE,
            linestyle="none",
            label=rf"$\mathrm{{Re}}(e_{{{k+1}}})$",
            **err_kws,
        )
    ax.set_yscale(real_axis_scale)
    ax.set_ylabel(
        r"$\mathrm{Re}(e_n)\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE, labelpad=0,
    )
    ax.legend(loc="upper left", fontsize=LABELS_FONT_SIZE)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    if real_axis_scale == "log":
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=2))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    elif real_axis_ticks is not None:
        ax.set_yticks(real_axis_ticks)
    ax.tick_params(
        axis="both", which="major",
        width=LINE_WIDTH, labelsize=LABELS_FONT_SIZE,
    )


def plot_appendix_bootstrap(
    result,
    *,
    parameters: dict | None = None,
    lambda_max: float = 1.5,
    fit_validity_limit: float | None = None,
    real_axis_scale: str = "linear",
    real_axis_ticks: list | None = None,
    real_split_axis_ticks: list | None = None,
    save_dir: Path | None = None,
    figure_stem: str = "appendix_bootstrap",
) -> "plt.Figure":
    """Build the 3-panel appendix bootstrap figure.

    Panel A (top): all decay rates vs λ (full pole spectrum).
    Panel B (mid): imaginary splitting Im(Δe₁₂) / Ω_S vs λ with sqrt fit.
    Panel C (bot): real splitting Re(Δe₁₂) / Ω_S vs λ with fit.

    Parameters
    ----------
    result:
        ``ZenoAnalysisResult`` (must have ``poles.has_splitting`` True).
    parameters:
        Override ``result.parameters`` for display (optional).
    lambda_max:
        Upper λ cutoff for all panels.
    fit_validity_limit:
        Upper λ cutoff for the fit curve overlay.  Default: ``lambda_max``.
    real_axis_scale:
        ``"linear"`` or ``"log"`` for panel A.
    real_axis_ticks:
        Manual y-tick positions for panel A.
    real_split_axis_ticks:
        Manual y-tick positions for panel C.
    save_dir:
        If given, save SVG + PDF here.
    figure_stem:
        Filename stem.

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from mpm_analysis.plotting._style import (
        TWO_COLUMN_WIDTH, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH,
        LAMBDA_OBS_COLOR, add_panel_letter, apply_paper_style,
        make_ax_lines_wider,
    )

    if fit_validity_limit is None:
        fit_validity_limit = lambda_max

    omega_s = result.omega_s

    apply_paper_style()

    # --- Data prep ---
    lambdas = result.lambda_values
    mask = lambdas <= lambda_max
    lambdas = lambdas[mask]
    poles = result.poles[mask]

    # Panel A: full decay rates
    r_med   = poles.decay_rates / omega_s
    r_lower = poles.decay_rates_lower / omega_s
    r_upper = poles.decay_rates_upper / omega_s

    # Panels B, C: splitting
    if not poles.has_splitting:
        raise ValueError(
            "result.poles.has_splitting is False — re-run the analysis to compute splittings."
        )

    imag_split = (poles.imag_splitting / omega_s).flatten()
    imag_err   = ((poles.imag_splitting_upper + poles.imag_splitting_lower) / 2.0 / omega_s).flatten()

    real_split = (poles.real_splitting / omega_s).flatten()
    real_err   = ((poles.real_splitting_upper + poles.real_splitting_lower) / 2.0 / omega_s).flatten()

    # Fit curves (if available)
    fit = result.critical_point_sqrt_fit
    lam_fit = imag_fit = real_fit = None
    lc = None
    if fit is not None and fit.lambda_plot is not None:
        fv_mask = fit.lambda_plot <= fit_validity_limit
        lam_fit  = fit.lambda_plot[fv_mask]
        imag_fit = fit.imag_fit_curve[fv_mask] if fit.imag_fit_curve is not None else None
        real_fit = fit.real_fit_curve[fv_mask] if fit.real_fit_curve is not None else None
        lc = fit.lambda_c

    # --- Figure layout ---
    fig_w = float(TWO_COLUMN_WIDTH) * 0.67
    fig = plt.figure(figsize=(fig_w, fig_w * 1.2))
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax_real_full  = fig.add_subplot(gs[0])
    ax_imag       = fig.add_subplot(gs[1])
    ax_real_split = fig.add_subplot(gs[2], sharex=ax_imag)

    err_kws = dict(
        elinewidth=LINE_WIDTH * 0.6,
        capsize=MARKER_SIZE * 1.2,
        capthick=LINE_WIDTH * 0.6,
        zorder=10,
    )

    def _add_crit_line(ax):
        if lc is not None:
            ax.axvline(
                lc, color=LAMBDA_OBS_COLOR, linestyle="--",
                linewidth=LINE_WIDTH, alpha=0.8, zorder=0,
            )
        make_ax_lines_wider(ax)

    # --- Panel A: full decay rates (shared with the discriminant layout) ---
    draw_full_spectrum_panel(
        ax_real_full, lambdas, r_med, r_lower, r_upper,
        real_axis_scale=real_axis_scale, real_axis_ticks=real_axis_ticks,
        err_kws=err_kws,
    )
    ax_real_full.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE)
    _add_crit_line(ax_real_full)

    add_panel_letter(
        ax_real_full, "a",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel B: imaginary splitting ---
    ax_imag.errorbar(
        lambdas, imag_split, yerr=imag_err,
        color="steelblue", marker="o",
        linestyle="none", markersize=MARKER_SIZE,
        **err_kws,
    )
    if lam_fit is not None and imag_fit is not None:
        ax_imag.plot(lam_fit, imag_fit, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax_imag.set_ylabel(
        r"$\mathrm{Im}(\Delta e_{12})\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE,
    )
    plt.setp(ax_imag.get_xticklabels(), visible=False)
    ax_imag.set_yticks([0.0, 0.4, 0.8])
    _add_crit_line(ax_imag)
    add_panel_letter(
        ax_imag, "b",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel C: real splitting ---
    ax_real_split.errorbar(
        lambdas, real_split, yerr=real_err,
        color="steelblue", marker="o",
        linestyle="none", markersize=MARKER_SIZE,
        **err_kws,
    )
    if lam_fit is not None and real_fit is not None:
        ax_real_split.plot(lam_fit, real_fit, color="k", linestyle="--", linewidth=LINE_WIDTH)
    ax_real_split.set_ylabel(
        r"$\mathrm{Re}(\Delta e_{12})\,/\,\Omega_\mathrm{S}$",
        fontsize=LABELS_FONT_SIZE,
    )
    ax_real_split.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE)
    if real_split_axis_ticks is not None:
        ax_real_split.set_yticks(real_split_axis_ticks)
    _add_crit_line(ax_real_split)
    add_panel_letter(
        ax_real_split, "c",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # x-axis tick formatting
    for _ax in [ax_real_split, ax_real_full]:
        _ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)

    return fig
