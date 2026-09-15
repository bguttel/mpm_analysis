"""
ksi_plot.py
-----------
Plotting helpers for the two transitions read off the pole strands:

* ``plot_ksi_curve``          — ξ(λ) = (2·Re(e3) − Re(e2))/(Re(e2) − Re(e3)).
  Its zero is the **state-freezing point** λ(ξ=0), *not* the exceptional point.
* ``plot_discriminant_curve`` — Δ(λ) = c1² − 4c0 of the pole pair.  Its zero is
  the **exceptional point** λ_EP.

Both use the paper style of the sqrt-fit figure: measured points with error
bars, the fit as a black dashed line, the transition as a vertical dashed line
with its bootstrap band.
"""
from __future__ import annotations

import numpy as np


def _paper_axes(ax, figsize=(4.4, 3.2)):
    """Create/return an axes with the paper style applied, plus its kwargs."""
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import (
        LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH, apply_paper_style,
    )

    apply_paper_style()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    err_kws = dict(
        elinewidth=LINE_WIDTH * 0.6,
        capsize=MARKER_SIZE * 1.2,
        capthick=LINE_WIDTH * 0.6,
        zorder=10,
    )
    return ax, err_kws, LABELS_FONT_SIZE, MARKER_SIZE, LINE_WIDTH


def _asym_yerr(y, lower, upper):
    """(2, n) asymmetric error bars from 16/84 curves around the plotted y."""
    if lower is None or upper is None:
        return None
    y = np.asarray(y, dtype=float)
    lo = np.clip(y - np.asarray(lower, dtype=float), 0, None)
    hi = np.clip(np.asarray(upper, dtype=float) - y, 0, None)
    return np.vstack([lo, hi])


def plot_ksi_curve(
    ksi_result,
    *,
    ax=None,
    color: str = "steelblue",
    crossing_color: str | None = None,
    show_zero_line: bool = True,
    show_band: bool = False,
    show_crossing_quantiles: bool = True,
    title: str | None = None,
) -> "plt.Axes":
    """Plot ξ(λ) with error bars and the state-freezing crossing λ(ξ=0).

    Parameters
    ----------
    ksi_result:
        A ``KsiCrossingResult`` from ``KsiCrossingPipeline.run()``.
    ax:
        Target axes.  Created if ``None``.
    color:
        Colour of the measured points.
    crossing_color:
        Colour of the vertical crossing line (default: the paper's λ colour).
    show_zero_line:
        Draw the horizontal ξ = 0 line.
    show_band:
        Also shade the 16/84 band (off by default — the error bars carry it).
    show_crossing_quantiles:
        Mark the bootstrap crossing median + 16/84 span when available.
    title:
        Axes title; a default naming the transition is used when ``None``.

    Returns
    -------
    plt.Axes
    """
    from mpm_analysis.plotting._style import LAMBDA_OBS_COLOR

    ax, err_kws, fs, ms, lw = _paper_axes(ax)
    crossing_color = crossing_color or LAMBDA_OBS_COLOR

    lam = np.asarray(ksi_result.lambda_values, dtype=float)
    ksi = np.asarray(ksi_result.ksi_values, dtype=float)
    yerr = _asym_yerr(ksi, ksi_result.ksi_lower, ksi_result.ksi_upper)

    if show_band and ksi_result.ksi_lower is not None:
        ax.fill_between(lam, ksi_result.ksi_lower, ksi_result.ksi_upper,
                        color=color, alpha=0.2, linewidth=0)
    ax.errorbar(lam, ksi, yerr=yerr, color=color, marker="o", linestyle="none",
                markersize=ms, label=r"$\xi(\lambda)$", **err_kws)

    if show_zero_line:
        ax.axhline(0, color="k", ls="--", lw=lw * 0.6, alpha=0.7)

    quantiles = ksi_result.crossing_quantiles
    if show_crossing_quantiles and quantiles is not None:
        q16, q50, q84 = quantiles
        ax.axvspan(q16, q84, color=crossing_color, alpha=0.15, linewidth=0, zorder=0)
        ax.axvline(q50, color=crossing_color, ls="--", lw=lw, alpha=0.8, zorder=0,
                   label=(rf"$\lambda_{{\xi=0}} = {q50:.3f}"
                          rf"^{{+{q84 - q50:.3f}}}_{{-{q50 - q16:.3f}}}$"))
    else:
        for lc in ksi_result.zero_crossings:
            ax.axvline(lc, color=crossing_color, ls="--", lw=lw, alpha=0.8, zorder=0,
                       label=rf"$\lambda_{{\xi=0}} = {lc:.3f}$")

    ax.set_xlabel(r"$\lambda$", fontsize=fs)
    ax.set_ylabel(r"$\xi = (2e_3 - \mathrm{Re}\,e_2)\,/\,(\mathrm{Re}\,e_2 - e_3)$",
                  fontsize=fs)
    ax.set_title(title or "State freezing: zero of "
                 rf"$\xi$ (poles {ksi_result.pole_idx1}, {ksi_result.pole_idx2})",
                 fontsize=fs)
    ax.legend(fontsize=fs * 0.8)
    ax.grid(False)
    return ax


def plot_discriminant_curve(
    lambdas,
    delta,
    *,
    fit=None,
    lambda_ep=None,
    first_crossing=None,
    crossing_quantiles=None,
    delta_lower=None,
    delta_upper=None,
    ax=None,
    color: str = "steelblue",
    crossing_color: str | None = None,
    title: str | None = None,
) -> "plt.Axes":
    """Plot the EP-pair discriminant with the straight line fitted through zero.

    ``Delta(lambda) = c1**2 - 4*c0`` is negative while the pair is complex and
    positive once it has split into two real poles, and it is linear in λ near
    a square-root exceptional point — so the fitted zero of that line is the EP.

    Parameters
    ----------
    lambdas, delta:
        The measured curve (normally from the central pole set).
    fit:
        The ``dict`` returned by
        :func:`mpm_analysis.analysis.ksi_crossing.linfit_zero_crossing` with
        ``return_fit=True``; its window and fitted line are drawn.
    lambda_ep, first_crossing:
        Line-fit zero (dashed vertical) and the raw first sign change (dotted).
    crossing_quantiles:
        ``(16, 50, 84)`` percentiles of the per-realization line fits, shaded.
    delta_lower, delta_upper:
        16/84 percentiles of the per-realization Delta curves — drawn as error
        bars on the points.
    ax, color, crossing_color, title:
        Cosmetics; ``ax`` is created when ``None``.

    Returns
    -------
    plt.Axes
    """
    from mpm_analysis.plotting._style import LAMBDA_OBS_COLOR

    ax, err_kws, fs, ms, lw = _paper_axes(ax)
    crossing_color = crossing_color or LAMBDA_OBS_COLOR

    lam = np.asarray(lambdas, dtype=float)
    d = np.asarray(delta, dtype=float)
    yerr = _asym_yerr(d, delta_lower, delta_upper)

    ax.axhline(0, color="k", ls="--", lw=lw * 0.6, alpha=0.7)
    ax.errorbar(lam, d, yerr=yerr, color=color, marker="o", linestyle="none",
                markersize=ms, label=r"$\Delta = c_1^2 - 4c_0$", **err_kws)

    if fit is not None and fit.get("coef") is not None:
        w = fit["window"]
        xs = np.linspace(lam[w].min(), lam[w].max(), 50)
        ax.plot(xs, np.polyval(fit["coef"], xs), color="k", linestyle="--",
                linewidth=lw, zorder=11,
                label=rf"line fit ({int(w.sum())} pts in $\pm${fit['half_width']:.2f})")

    if crossing_quantiles is not None:
        q16, _, q84 = crossing_quantiles
        ax.axvspan(q16, q84, color=crossing_color, alpha=0.15, linewidth=0, zorder=0)
    if lambda_ep is not None and np.isfinite(lambda_ep):
        label = rf"$\lambda_{{EP}} = {lambda_ep:.3f}$"
        if crossing_quantiles is not None:
            q16, q50, q84 = crossing_quantiles
            label = (rf"$\lambda_{{EP}} = {lambda_ep:.3f}"
                     rf"^{{+{q84 - q50:.3f}}}_{{-{q50 - q16:.3f}}}$")
        ax.axvline(lambda_ep, color=crossing_color, ls="--", lw=lw, alpha=0.8,
                   zorder=0, label=label)
    if first_crossing is not None and np.isfinite(first_crossing):
        ax.axvline(first_crossing, color=crossing_color, ls=":", lw=lw * 0.8,
                   alpha=0.6, zorder=0,
                   label=rf"first crossing $= {first_crossing:.3f}$")

    finite = d[np.isfinite(d)]
    if finite.size:
        span = np.percentile(np.abs(finite), 90)
        ax.set_ylim(-1.4 * span, 1.4 * span)
    ax.set_xlabel(r"$\lambda$", fontsize=fs)
    ax.set_ylabel(r"$\Delta(\lambda)$", fontsize=fs)
    ax.set_title(title or "Exceptional point: zero of the pair discriminant",
                 fontsize=fs)
    ax.legend(fontsize=fs * 0.8)
    ax.grid(False)
    return ax
