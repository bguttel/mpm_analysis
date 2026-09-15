"""
first_transition_main.py
------------------------
First-transition paper figure (appendix style), built from a live
``ZenoAnalysisResult`` — toggled directly from ``run_files/run.py``
(``RUN_FIRST_TRANSITION_FIGURE``), which produces all the inputs.

Same style as ``ZenoPlotting/Noams_code/first_transition_plotting.py``
(constants from :mod:`mpm_analysis.plotting._style`).  The ``layout``
argument picks the variant:

* ``"sqrt"`` — the 3-panel layout: A full pole spectrum; B/C imaginary and
  real splitting of the first pair with the square-root fit.  Delegates to
  :func:`~mpm_analysis.plotting.paper_figures.appendix_bootstrap.plot_appendix_bootstrap`
  (needs ``result.critical_point_sqrt_fit`` — set by the run's
  critical-point task).
* ``"discriminant"`` — 2 panels: the same panel A, and one panel where the
  pair discriminant ``Δ12(λ) = c1² − 4c0`` with its straight-line fit
  through the zero crossing (the EP diagnostic) replaces panels B and C.
  The vertical marker is the fitted λ_EP; when the result carries raw
  bootstrap samples the Δ points get their 16/84 band as error bars.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_first_transition_figure(
    result,
    *,
    layout: str = "sqrt",
    lambda_max: float = 1.5,
    fit_validity_limit: float | None = None,
    real_axis_scale: str = "log",
    real_axis_ticks: list | None = None,
    real_split_axis_ticks: list | None = None,
    discriminant_half_width: float = 0.15,
    save_dir: Path | None = None,
    figure_stem: str = "first_transition_main",
) -> "plt.Figure":
    """Build the first-transition paper figure in either layout.

    Parameters
    ----------
    result:
        ``ZenoAnalysisResult`` from the run's MPM + bootstrap pipeline.
    layout:
        ``"sqrt"`` (3 panels, splittings + sqrt fit) or ``"discriminant"``
        (2 panels, the Δ12 line fit replaces panels B and C).
    lambda_max:
        Upper λ cutoff for all panels.
    fit_validity_limit:
        Upper λ cutoff of the sqrt-fit overlay (``"sqrt"`` layout only).
    real_axis_scale, real_axis_ticks:
        Panel A y-scale / manual ticks (``"log"`` is the first-transition
        paper choice).
    real_split_axis_ticks:
        Panel C y-ticks (``"sqrt"`` layout only).
    discriminant_half_width:
        Half-window in λ of the straight-line fit through the Δ12 crossing
        (``"discriminant"`` layout only).
    save_dir, figure_stem:
        When given, save SVG + PDF there.

    Returns
    -------
    plt.Figure
    """
    if layout == "sqrt":
        from mpm_analysis.plotting.paper_figures.appendix_bootstrap import (
            plot_appendix_bootstrap,
        )
        return plot_appendix_bootstrap(
            result,
            lambda_max=lambda_max,
            fit_validity_limit=fit_validity_limit,
            real_axis_scale=real_axis_scale,
            real_axis_ticks=real_axis_ticks,
            real_split_axis_ticks=real_split_axis_ticks,
            save_dir=save_dir,
            figure_stem=figure_stem,
        )
    if layout != "discriminant":
        raise ValueError(
            f"layout must be 'sqrt' or 'discriminant', got {layout!r}."
        )

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from mpm_analysis.analysis.ksi_crossing import (
        discriminant_crossing_samples,
        discriminant_from_params,
        linfit_zero_crossing,
    )
    from mpm_analysis.analysis.matrix_pencil import pack_pole_params
    from mpm_analysis.plotting._style import (
        TWO_COLUMN_WIDTH, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH,
        LAMBDA_OBS_COLOR, add_panel_letter, apply_paper_style,
        make_ax_lines_wider,
    )
    from mpm_analysis.plotting.paper_figures.appendix_bootstrap import (
        draw_full_spectrum_panel,
    )

    omega_s = result.omega_s
    apply_paper_style()

    # --- Data prep ---
    lambdas = result.lambda_values
    mask = lambdas <= lambda_max
    lambdas = lambdas[mask]
    poles = result.poles[mask]

    r_med   = poles.decay_rates / omega_s
    r_lower = poles.decay_rates_lower / omega_s
    r_upper = poles.decay_rates_upper / omega_s

    # Discriminant of the first quadratic pair from the central poles,
    # normalized by Ω_S² (poles enter Δ quadratically).
    central = -poles.decay_rates + 1j * poles.frequencies
    order = central.shape[1]
    delta = np.array([
        discriminant_from_params(pack_pole_params(p), order) for p in central
    ], dtype=float) / omega_s**2
    lam_ep, dfit = linfit_zero_crossing(
        lambdas, delta, half_width=discriminant_half_width, return_fit=True,
    )

    # 16/84 bootstrap band on Δ, when raw samples were kept
    yerr = None
    raw = result.poles.raw_samples
    if raw is not None and raw.param_samples is not None:
        _, d_med, d_lo, d_hi = discriminant_crossing_samples(
            raw, lambda_max=lambda_max, half_width=discriminant_half_width,
        )
        if d_med is not None and len(d_med) == len(lambdas):
            d_lo = np.asarray(d_lo, dtype=float) / omega_s**2
            d_hi = np.asarray(d_hi, dtype=float) / omega_s**2
            yerr = [np.clip(delta - d_lo, 0.0, None),
                    np.clip(d_hi - delta, 0.0, None)]

    # --- Figure layout: A (spectrum) over B (discriminant) ---
    fig_w = float(TWO_COLUMN_WIDTH) * 0.67
    fig = plt.figure(figsize=(fig_w, fig_w * 1.1))
    fig.set_layout_engine("constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
    ax_real_full = fig.add_subplot(gs[0])
    ax_delta     = fig.add_subplot(gs[1], sharex=ax_real_full)

    err_kws = dict(
        elinewidth=LINE_WIDTH * 0.6,
        capsize=MARKER_SIZE * 1.2,
        capthick=LINE_WIDTH * 0.6,
        zorder=10,
    )

    def _add_ep_line(ax):
        if np.isfinite(lam_ep):
            ax.axvline(
                lam_ep, color=LAMBDA_OBS_COLOR, linestyle="--",
                linewidth=LINE_WIDTH, alpha=0.8, zorder=0,
            )
        make_ax_lines_wider(ax)

    # --- Panel A: full decay rates (shared with the sqrt layout) ---
    draw_full_spectrum_panel(
        ax_real_full, lambdas, r_med, r_lower, r_upper,
        real_axis_scale=real_axis_scale, real_axis_ticks=real_axis_ticks,
        err_kws=err_kws,
    )
    plt.setp(ax_real_full.get_xticklabels(), visible=False)
    _add_ep_line(ax_real_full)
    add_panel_letter(
        ax_real_full, "a",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )

    # --- Panel B: Δ12(λ) with the straight-line fit through the crossing ---
    ax_delta.errorbar(
        lambdas, delta, yerr=yerr,
        color="steelblue", marker="o",
        linestyle="none", markersize=MARKER_SIZE,
        **err_kws,
    )
    if dfit is not None and dfit.get("coef") is not None:
        lam_w = lambdas[dfit["window"]]
        lam_line = np.linspace(lam_w.min(), lam_w.max(), 50)
        ax_delta.plot(
            lam_line, np.polyval(dfit["coef"], lam_line),
            color="k", linestyle="--", linewidth=LINE_WIDTH,
        )
    ax_delta.axhline(0.0, color="0.5", linewidth=LINE_WIDTH * 0.5, zorder=0)
    ax_delta.set_ylabel(
        r"$\Delta_{12}\,/\,\Omega_\mathrm{S}^2$",
        fontsize=LABELS_FONT_SIZE,
    )
    ax_delta.set_xlabel(r"$\lambda$", fontsize=LABELS_FONT_SIZE)
    _add_ep_line(ax_delta)
    add_panel_letter(
        ax_delta, "b",
        x=-0.2, y=1.05, offset_mm=False,
        fontsize=LABELS_FONT_SIZE * 1.4,
    )
    ax_delta.set_yticks([-2.0, 0.0, 1.0])

    for _ax in (ax_real_full, ax_delta):
        _ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)

    return fig
