"""
ksi_plot.py
-----------
Plotting helper for ksi (ξ) crossing results.
"""
from __future__ import annotations

import numpy as np


def plot_ksi_curve(
    ksi_result,
    *,
    ax=None,
    color: str = "steelblue",
    crossing_color: str = "crimson",
    show_zero_line: bool = True,
    markersize: float = 4.0,
    show_band: bool = True,
    show_crossing_quantiles: bool = True,
) -> "plt.Axes":
    """Plot ξ(λ) with zero crossings highlighted.

    Parameters
    ----------
    ksi_result:
        A ``KsiCrossingResult`` from ``KsiCrossingPipeline.run()``.
    ax:
        Target axes.  Created if ``None``.
    color:
        Colour for the ξ curve.
    crossing_color:
        Colour for the vertical lines at zero crossings.
    show_zero_line:
        Whether to draw a horizontal dashed line at ξ = 0.
    markersize:
        Marker size for data points.
    show_band:
        Shade the 16/84 bootstrap band of ξ(λ) when available.
    show_crossing_quantiles:
        Mark the bootstrap crossing median + 16/84 span when available
        (replaces the per-curve crossing lines).

    Returns
    -------
    plt.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    lam = ksi_result.lambda_values
    ksi = ksi_result.ksi_values

    if (
        show_band
        and ksi_result.ksi_lower is not None
        and ksi_result.ksi_upper is not None
    ):
        ax.fill_between(
            lam,
            ksi_result.ksi_lower,
            ksi_result.ksi_upper,
            color=color,
            alpha=0.25,
            linewidth=0,
            label=r"16/84 %",
        )

    ax.plot(
        lam, ksi,
        "o-",
        color=color,
        markersize=markersize,
        linewidth=1.2,
        label=r"$\xi(\lambda)$",
    )

    if show_zero_line:
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)

    quantiles = ksi_result.crossing_quantiles
    if show_crossing_quantiles and quantiles is not None:
        q16, q50, q84 = quantiles
        ax.axvspan(q16, q84, color=crossing_color, alpha=0.15, linewidth=0)
        ax.axvline(
            q50,
            color=crossing_color,
            ls=":",
            lw=1.2,
            alpha=0.9,
            label=(rf"$\lambda_{{EP}} = {q50:.3f}"
                   rf"^{{+{q84 - q50:.3f}}}_{{-{q50 - q16:.3f}}}$"),
        )
    else:
        for lc in ksi_result.zero_crossings:
            ax.axvline(
                lc,
                color=crossing_color,
                ls=":",
                lw=1.2,
                alpha=0.9,
                label=rf"$\lambda_{{EP}} \approx {lc:.3f}$",
            )

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(
        rf"Ksi crossing  (poles {ksi_result.pole_idx1}, {ksi_result.pole_idx2})"
    )
    ax.legend(fontsize=8)
    ax.grid(False)

    return ax
