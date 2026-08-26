"""
click_correlation_plots.py
--------------------------
Figures for the click-probability chain (``analysis/click_correlations.py``
-> MPM poles), covering both estimator modes:

* ``conditional``   — post-selected no-jump curves, P(n=0,t) and the
  first-click probability;
* ``unconditional`` — full-dynamics per-bin click / no-click probability;
* ``partial``       — the interpolation between them (``max_clicks``).

Figure set (what ``save_click_correlation_figures`` writes):

* ``corr_raw_<channel>`` — one figure **per channel**: every λ stacked on
  the same axes, colour-coded by λ with a colourbar (up to ``n_stack``
  evenly-chosen representatives).  Only the fitted part is shown — the
  estimator already truncates each λ at its ``min_alive_frac`` cut, so the
  stored arrays *are* the fitted range.
* ``corr_click_prob`` — ``P(run has >= 1 click)`` vs λ.
* ``corr_poles_<key>`` — MPM pole spectrum vs λ for one fitted observable:
  decay rates (top) and oscillation frequencies (bottom) with bootstrap
  error bars, one colour per pole index.
* (the Hankel-SVD order-calibration figure is saved by ``MPMStep`` itself.)

All figures are written with timestamped, non-clobbering names and
metadata-bearing titles; styling follows ``plotting/_style.py``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless-safe; we only save PNGs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from mpm_analysis.analysis.click_correlations import (
    COND_CLICK, COND_NO_CLICK, ClickCorrelationResult,
    PARTIAL_CLICK, PARTIAL_NO_CLICK, UNCOND_CLICK, UNCOND_NO_CLICK,
)
from mpm_analysis.io.experimental.compressed_click_record import _unique_path
from mpm_analysis.io.file_naming import build_figure_filename
from mpm_analysis.plotting._style import (
    AZURE_BLUE, LINE_WIDTH, MARKER_SIZE, PURPLE_MAGENTA, TEAL,
    TWO_COLUMN_WIDTH, apply_paper_style, make_ax_lines_wider,
)
from mpm_analysis.utils.windows_paths import win_long_path

#: Human-readable name, y-label and y-scale per observable key.
CHANNEL_INFO = {
    COND_NO_CLICK: dict(
        name="conditional (post-selected) no-click probability",
        ylabel=r"$P(n=0,\,t)$",
        logy=True,
    ),
    COND_CLICK: dict(
        name="conditional first-click probability",
        ylabel=r"$P(\mathrm{first\ click\ at\ } t)$",
        logy=True,
    ),
    UNCOND_NO_CLICK: dict(
        name="unconditional no-click probability",
        ylabel=r"$P(\mathrm{no\ click\ at\ } t)$",
        logy=False,
    ),
    UNCOND_CLICK: dict(
        name="unconditional click probability",
        ylabel=r"$P(\mathrm{click\ at\ } t)$",
        logy=False,
    ),
    PARTIAL_NO_CLICK: dict(
        name="partially conditioned no-click probability",
        ylabel=r"$P(\mathrm{no\ click\ at\ } t,\ \leq k\ \mathrm{before})$",
        logy=True,
    ),
    PARTIAL_CLICK: dict(
        name="partially conditioned click probability",
        ylabel=r"$P(\mathrm{click\ at\ } t,\ \leq k\ \mathrm{before})$",
        logy=True,
    ),
}

#: Every mode measures absolute time from each run's own t = 0.
_XLABEL = r"$t$ ($\mu$s)"

_POLE_COLORS = [AZURE_BLUE, PURPLE_MAGENTA, TEAL, "#D4AC0D", "#555555"]


def _suptitle(results: list[ClickCorrelationResult], what: str) -> str:
    r0 = results[0]
    lam = [r.lambda_val for r in results]
    scan = r0.metadata.get("scan_number")
    scan_seg = f"scan {scan}  |  " if scan is not None else ""
    win = r0.window
    return (
        f"{what}  |  {scan_seg}"
        f"{len(results)} $\\lambda$ in [{min(lam):.2f}, {max(lam):.2f}]  |  "
        f"{r0.mode} estimator (max_clicks={r0.max_clicks}), "
        f"min alive frac {win.get('min_alive_frac', float('nan')):.2f}  |  "
        f"dt = {r0.dt_us:.3f} $\\mu$s"
    )


# ---------------------------------------------------------------------------
# Raw data — one channel, all λ stacked with a λ colourbar
# ---------------------------------------------------------------------------

def plot_channel_stacked(
    results: list[ClickCorrelationResult],
    key: str,
    *,
    n_stack: int = 10,
    logy: bool | None = None,
) -> plt.Figure:
    """All (or ``n_stack`` representative) λ of one channel on the same axes.

    Curves are colour-coded by λ (viridis) with a colourbar, error bars are
    thinned (``errorevery``) to stay readable.  Only the fitted range is
    drawn: each λ's stored arrays already end at its ``min_alive_frac`` cut.
    """
    apply_paper_style()
    info = CHANNEL_INFO.get(key, dict(name=key, ylabel=key, logy=False))
    if logy is None:
        logy = info["logy"]

    ordered = sorted(results, key=lambda r: r.lambda_val)
    idx = np.unique(np.linspace(0, len(ordered) - 1,
                                min(n_stack, len(ordered))).astype(int))
    shown = [ordered[i] for i in idx]

    lam_all = [r.lambda_val for r in ordered]
    norm = Normalize(vmin=min(lam_all), vmax=max(lam_all))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(
        figsize=(TWO_COLUMN_WIDTH / 1.4, TWO_COLUMN_WIDTH / 2.2),
    )
    for res in shown:
        y, err = res.signal(key)
        color = cmap(norm(res.lambda_val))
        ax.errorbar(
            res.t_us, y, yerr=err,
            fmt="o-", color=color, ms=MARKER_SIZE * 1.6,
            lw=LINE_WIDTH * 0.8, elinewidth=LINE_WIDTH * 0.4, capsize=1.2,
            errorevery=max(1, len(res.t_us) // 25),
        )
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(info["ylabel"])
    ax.grid(alpha=0.25, lw=LINE_WIDTH * 0.3)
    make_ax_lines_wider(ax)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\lambda$")

    fig.suptitle(_suptitle(ordered, info["name"].capitalize()), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Condition probability r̄ vs λ
# ---------------------------------------------------------------------------

def plot_click_probability(
    results: list[ClickCorrelationResult],
) -> plt.Figure:
    """``r_bar = P(run has >= 1 click)`` vs λ, a per-λ summary number."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(TWO_COLUMN_WIDTH / 2, TWO_COLUMN_WIDTH / 3))
    ordered = sorted(results, key=lambda r: r.lambda_val)
    ax.plot([r.lambda_val for r in ordered], [r.r_bar for r in ordered],
            "o-", color=AZURE_BLUE, ms=MARKER_SIZE * 2.5, lw=LINE_WIDTH)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\bar r = P(\geq 1\ \mathrm{click})$")
    ax.grid(alpha=0.25, lw=LINE_WIDTH * 0.3)
    make_ax_lines_wider(ax)
    fig.suptitle(_suptitle(ordered, "Probability of at least one click"),
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------
# MPM pole spectrum vs λ for one fitted observable
# ---------------------------------------------------------------------------

def plot_correlation_poles(result, *, observable_key: str) -> plt.Figure:
    """Decay rates and frequencies vs λ from a ``ZenoAnalysisResult``.

    One colour per pole index; error bars are the bootstrap half-widths;
    all-NaN poles are skipped.
    """
    apply_paper_style()
    poles = result.poles
    lam = result.lambda_values
    order = poles.order
    info = CHANNEL_INFO.get(observable_key, dict(name=observable_key))

    fig, (ax_d, ax_f) = plt.subplots(
        2, 1, sharex=True,
        figsize=(TWO_COLUMN_WIDTH / 1.4, TWO_COLUMN_WIDTH / 1.7),
    )
    for j in range(order):
        if np.all(~np.isfinite(poles.decay_rates[:, j])):
            continue
        c = _POLE_COLORS[j % len(_POLE_COLORS)]
        ax_d.errorbar(
            lam, poles.decay_rates[:, j],
            yerr=[poles.decay_rates_lower[:, j], poles.decay_rates_upper[:, j]],
            fmt="o-", color=c, ms=MARKER_SIZE * 2, lw=LINE_WIDTH * 0.7,
            elinewidth=LINE_WIDTH * 0.5, capsize=1.5, label=f"pole {j}",
        )
        ax_f.errorbar(
            lam, poles.frequencies[:, j],
            yerr=[poles.frequencies_lower[:, j], poles.frequencies_upper[:, j]],
            fmt="o-", color=c, ms=MARKER_SIZE * 2, lw=LINE_WIDTH * 0.7,
            elinewidth=LINE_WIDTH * 0.5, capsize=1.5,
        )
    ax_d.set_ylabel(r"$-\mathrm{Re}\,s$ ($\mu$s$^{-1}$)")
    ax_f.set_ylabel(r"$\mathrm{Im}\,s$ (rad/$\mu$s)")
    ax_f.set_xlabel(r"$\lambda$")
    ax_d.legend(frameon=False, fontsize=8, ncol=min(order, 3))
    for ax in (ax_d, ax_f):
        ax.grid(alpha=0.25, lw=LINE_WIDTH * 0.3)
        make_ax_lines_wider(ax)
    fig.align_ylabels((ax_d, ax_f))
    fig.suptitle(
        f"MPM poles of the {info['name']}  |  "
        f"{len(lam)} $\\lambda$, order {order}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Save-all entry point
# ---------------------------------------------------------------------------

def save_click_correlation_figures(
    results: list[ClickCorrelationResult],
    mpm_results: dict,
    out_dir: str | Path,
    *,
    source_tag: str,
    scan_number: int | None = None,
    n_stack: int = 10,
) -> list[Path]:
    """Save the full correlation figure set as PNGs; return the saved paths.

    One raw stacked-λ figure per channel of the results' mode, the
    at-least-one-click probability vs λ, and one pole-spectrum figure per
    fitted observable in ``mpm_results`` (``{key: ZenoAnalysisResult}``).
    """
    out_dir = Path(out_dir)
    saved: list[Path] = []

    def _save(fig: plt.Figure, tag: str) -> None:
        name = build_figure_filename(tag, source_tag,
                                     scan_number=scan_number, extension="png")
        path = _unique_path(out_dir / name)
        fig.savefig(win_long_path(path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    for key in results[0].keys:
        _save(plot_channel_stacked(results, key, n_stack=n_stack),
              f"corr_raw_{key}")
    _save(plot_click_probability(results), "corr_click_prob")
    for key, res in mpm_results.items():
        _save(plot_correlation_poles(res, observable_key=key),
              f"corr_poles_{key}")
    return saved
