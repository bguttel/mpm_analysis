"""
plotting.py
-----------
Visualisation functions for k-space parity analysis.

Public API
~~~~~~~~~~
plot_eigenvalues_vs_lambda
    Decay rates and frequencies vs λ at a fixed k.  Mirrors the style
    of the existing ``pole_spectrum.py``.

plot_raw_traces
    Overlay time traces for multiple k values at one λ.  Only real parts
    are overlaid with real parts (or imaginary with imaginary) to avoid
    the phase-mixing artefact that appears when mixing components.

plot_poles_vs_k_slider
    Interactive figure: decay rates + frequencies vs k, with a responsive
    Slider widget to sweep through λ values.

plot_poles_vs_k_grid
    Static panel grid: one subplot per λ value, poles vs k.

plot_poles_vs_k_3d_grid
    Rotatable 3D grid: one subplot per λ, axes (k, Re(pole), Im(pole)).
    Tracks are reordered along k via continuity sort so each colour
    follows the most continuous trajectory in the complex plane.

plot_svd_vs_k_slider
    Interactive SVD diagnostic for calibrating the pole-order threshold:
    singular values + the decision ratio vs k, with a slider over λ.

plot_svd_vs_lambda_grid
    Static panel grid of singular-value magnitudes vs λ — one panel per k.

export_svd_vs_lambda_slider_html
    Standalone HTML: singular-value magnitudes vs λ with a k slider (the
    transpose of ``plot_svd_vs_k_slider``, shareable in any browser).

export_pole_pair_topology_html
    Standalone HTML: the pair gaps Δ_ab(k) = pole_a − pole_b traced in the
    complex plane with k as the parameter, one panel per pair and one λ
    slider per panel.  Reports the normalized winding (one loop = 1) and
    marks the k of any crossing through the origin (exceptional point).

K-space track sort
~~~~~~~~~~~~~~~~~~
Pole columns are reordered at plot time in two stages, so that a colour means
the same physical branch everywhere in the figure set:

1. *along k* — Hungarian assignment against the previous k (primary cost:
   complex-plane distance; tiebreaker: distance to a linear extrapolation in k);
2. *across λ* — Hungarian assignment of each λ slice's whole k-tracks against
   the previous λ's.  Without this the ordering at the first k is whatever
   canonical sort ``KSpaceAnalysis`` happened to produce for that cell, which
   flips from λ to λ and made track colours meaningless between panels.

Both stages are plot-time only — the underlying ``KSpacePolesResult.poles``
array keeps the canonical sort produced by ``KSpaceAnalysis``.  Use
``track_permutation`` to apply the identical reordering to companion arrays
(e.g. ``poles_err``).
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.optimize import linear_sum_assignment

from mpm_analysis.kspace.data_types import KSpacePolesResult, KSpaceRecord


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_POLE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def _pole_color(pole_idx: int) -> str:
    return _POLE_COLORS[pole_idx % len(_POLE_COLORS)]


# ---------------------------------------------------------------------------
# Continuity sort along k
# ---------------------------------------------------------------------------

def sort_poles_along_k(
    poles_lambda: np.ndarray,
    *,
    deriv_weight: float = 0.1,
) -> np.ndarray:
    """Reorder pole columns along k for visual continuity.

    For each k > 0, finds the permutation of the candidate poles that
    minimises::

        cost(prev_slot, candidate) =
            |candidate - prev|  +  deriv_weight * |candidate - predicted|

    where ``predicted = 2 * prev - prev_prev`` is a linear extrapolation
    in k (acts as a derivative tiebreaker).  Solved exactly via Hungarian
    assignment so each candidate is used once.

    NaN-padded slots (``pole_order < max_order``) are passed through:
    cost entries involving NaN are set to a large finite value, so valid
    poles are matched first and NaN slots fall to whatever's left.

    Parameters
    ----------
    poles_lambda : np.ndarray, complex, shape (n_k, max_order)
        Pole values for one λ slice (canonical sort).
    deriv_weight : float
        Weight on the linear-extrapolation tiebreaker term.  Small values
        (~0.1) keep distance as the primary criterion.

    Returns
    -------
    np.ndarray, complex, shape (n_k, max_order)
        Same poles with columns permuted at each k for continuity.
    """
    poles_lambda = np.asarray(poles_lambda, dtype=complex)
    n_k, m = poles_lambda.shape
    if n_k < 2 or m < 2:
        return poles_lambda.copy()

    perm = _sort_perm_along_k(poles_lambda, deriv_weight=deriv_weight)
    return np.take_along_axis(poles_lambda, perm, axis=1)


_BIG_COST = 1e9


def _sort_perm_along_k(
    poles_lambda: np.ndarray,
    *,
    deriv_weight: float = 0.1,
) -> np.ndarray:
    """Column permutation implementing :func:`sort_poles_along_k`.

    Returns ``perm`` of shape ``(n_k, max_order)`` such that
    ``sorted[k, slot] = poles_lambda[k, perm[k, slot]]``.  Working with the
    permutation (rather than the permuted values) lets callers apply the exact
    same reordering to companion arrays — ``poles_err``, amplitudes — so error
    bars never drift away from the track they belong to.
    """
    poles_lambda = np.asarray(poles_lambda, dtype=complex)
    n_k, m = poles_lambda.shape
    perm = np.tile(np.arange(m), (n_k, 1))
    if n_k < 2 or m < 2:
        return perm

    out = np.empty_like(poles_lambda)
    out[0] = poles_lambda[0]

    for k in range(1, n_k):
        prev = out[k - 1]
        pred = 2.0 * out[k - 1] - out[k - 2] if k >= 2 else prev
        cur = poles_lambda[k]

        d_prev = np.abs(prev[:, None] - cur[None, :])
        d_pred = np.abs(pred[:, None] - cur[None, :])
        cost = np.where(np.isnan(d_prev + d_pred), _BIG_COST,
                        d_prev + deriv_weight * d_pred)

        row, col = linear_sum_assignment(cost)
        perm[k, row] = col
        out[k] = cur[perm[k]]

    return perm


def _match_perm_across_lambda(poles_sorted: np.ndarray) -> np.ndarray:
    """Column permutation that keeps track identity stable **across λ**.

    ``sort_poles_along_k`` only enforces continuity along k; its starting order
    at the first k is whatever canonical sort ``KSpaceAnalysis`` produced for
    that cell, and that ordering flips freely from one λ to the next.  Colours
    were therefore only meaningful within a single panel: "track 0" could be the
    long-lived branch at one λ and the fast-decaying one at the next.

    Here each λ slice is Hungarian-matched against the previous one, with cost =
    mean complex-plane distance between the two whole k-tracks.  Returns
    ``lam_perm`` of shape ``(n_lambda, max_order)`` such that
    ``matched[l, :, slot] = poles_sorted[l, :, lam_perm[l, slot]]``.
    """
    poles_sorted = np.asarray(poles_sorted, dtype=complex)
    n_lambda, _, m = poles_sorted.shape
    lam_perm = np.tile(np.arange(m), (n_lambda, 1))
    if n_lambda < 2 or m < 2:
        return lam_perm

    prev = poles_sorted[0]
    for li in range(1, n_lambda):
        cur = poles_sorted[li]
        d = np.abs(prev[:, :, None] - cur[:, None, :])      # (n_k, m, m)
        valid = np.isfinite(d)
        n_valid = valid.sum(axis=0)
        cost = np.where(n_valid > 0,
                        np.where(valid, d, 0.0).sum(axis=0) / np.maximum(n_valid, 1),
                        _BIG_COST)

        _, col = linear_sum_assignment(cost)
        lam_perm[li] = col
        prev = cur[:, col]

    return lam_perm


def track_permutation(
    result: KSpacePolesResult,
    *,
    deriv_weight: float = 0.1,
    match_lambda: bool = True,
) -> np.ndarray:
    """Full plot-time reordering of ``result.poles`` as an index array.

    Composes the along-k continuity sort with the across-λ track match, so that
    ``np.take_along_axis(arr, perm, axis=2)`` reorders *any* ``(n_λ, n_k, M)``
    array onto consistent, colour-stable tracks.
    """
    poles = np.asarray(result.poles, dtype=complex)
    n_lambda, n_k, m = poles.shape

    perm = np.empty((n_lambda, n_k, m), dtype=int)
    for li in range(n_lambda):
        perm[li] = _sort_perm_along_k(poles[li], deriv_weight=deriv_weight)
    if not match_lambda:
        return perm

    sorted_poles = np.take_along_axis(poles, perm, axis=2)
    lam_perm = _match_perm_across_lambda(sorted_poles)
    # compose: slot -> (λ-match) -> (along-k sort) -> canonical column
    return np.take_along_axis(perm, np.broadcast_to(
        lam_perm[:, None, :], (n_lambda, n_k, m)), axis=2)


def _sorted_poles_per_lambda(
    result: KSpacePolesResult,
    *,
    deriv_weight: float = 0.1,
    match_lambda: bool = True,
) -> np.ndarray:
    """``result.poles`` continuity-sorted along k and track-matched across λ."""
    perm = track_permutation(result, deriv_weight=deriv_weight,
                             match_lambda=match_lambda)
    return np.take_along_axis(np.asarray(result.poles, dtype=complex),
                              perm, axis=2)


# ---------------------------------------------------------------------------
# 1. Eigenvalues vs lambda at a fixed k
# ---------------------------------------------------------------------------

def plot_eigenvalues_vs_lambda(
    result: KSpacePolesResult,
    k_idx_or_val,
    *,
    pole_indices: list[int] | None = None,
    axs: tuple | None = None,
    title_suffix: str = "",
    continuity_sort: bool = True,
    deriv_weight: float = 0.1,
) -> plt.Figure:
    """Plot decay rates and frequencies vs λ at a fixed k.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()``.
    k_idx_or_val:
        Integer index or float k value (nearest-neighbour lookup).
    pole_indices:
        Which pole slots to plot.  Default: all (0..max_order-1).
    axs:
        Optional ``(ax_decay, ax_freq)`` tuple.  A new figure is created
        if not supplied.
    title_suffix:
        Appended to the figure suptitle.
    continuity_sort:
        If True (default), plot the same continuity-sorted, λ-matched tracks
        the vs-k figures use, so "track 0 = blue" means the same branch here
        as everywhere else.  The raw canonical sort flips order from λ to λ,
        which made these curves jump between branches.
    deriv_weight:
        Tiebreaker weight for the continuity sort.

    Returns
    -------
    plt.Figure
    """
    k_idx = result._resolve_k(k_idx_or_val)
    k_val = result.k_array[k_idx]

    if continuity_sort:
        perm = track_permutation(result, deriv_weight=deriv_weight)
        poles_k = np.take_along_axis(
            np.asarray(result.poles, dtype=complex), perm, axis=2)[:, k_idx, :]
    else:
        perm = None
        poles_k = result.poles_at_k(k_idx)      # (n_lambda, max_order)
    decay = -poles_k.real
    freq = poles_k.imag
    lam = result.lambda_values

    # Optional run-bootstrap error bars (std of Re / Im across runs).
    derr = ferr = None
    if getattr(result, "poles_err", None) is not None:
        err = np.asarray(result.poles_err)
        if perm is not None:                    # keep bars on their own track
            err = np.take_along_axis(err, perm, axis=2)
        err_k = err[:, k_idx, :]
        derr = np.abs(err_k.real)
        ferr = np.abs(err_k.imag)

    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    if axs is None:
        fig, (ax_d, ax_f) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    else:
        ax_d, ax_f = axs
        fig = ax_d.get_figure()

    # Colours run in reverse across the plotted slots, so these tracks come out
    # the opposite way round from the vs-k figures.
    n_plotted = len(pole_indices)
    for slot, pi in enumerate(pole_indices):
        color = _pole_color(n_plotted - 1 - slot)
        label = f"pole {pi}"
        if derr is not None:
            ax_d.errorbar(lam, decay[:, pi], yerr=derr[:, pi], fmt="o-",
                          color=color, ms=4, lw=1.4, capsize=2, label=label)
            ax_f.errorbar(lam, freq[:, pi], yerr=ferr[:, pi], fmt="o-",
                          color=color, ms=4, lw=1.4, capsize=2, label=label)
        else:
            ax_d.plot(lam, decay[:, pi], "o-", color=color, ms=4, lw=1.4,
                      label=label)
            ax_f.plot(lam, freq[:, pi], "o-", color=color, ms=4, lw=1.4,
                      label=label)

    ax_d.set_ylabel("Decay rate  [1/µs]")
    ax_f.set_ylabel("Frequency  [1/µs]")
    ax_f.set_xlabel("λ")
    for ax in (ax_d, ax_f):
        ax.grid(True, color="0.92", lw=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_d.legend(fontsize=8, frameon=True, framealpha=0.9)
    # Say "k = π" instead of a rounded number when the fixed k *is* the π point.
    k_arr = np.asarray(result.k_array, dtype=float)
    tol = 0.5 * float(np.min(np.diff(k_arr))) if k_arr.size > 1 else 1e-9
    k_str = "π" if abs(k_val - np.pi) <= tol else f"{k_val:.3f} rad"
    fig.suptitle(f"k = {k_str} eigenvalues{title_suffix}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Raw time traces overlaid at one lambda
# ---------------------------------------------------------------------------

def plot_raw_traces(
    records: list[KSpaceRecord],
    lambda_val: float,
    *,
    component: str = "real",
    k_indices: list[int] | None = None,
    ax: plt.Axes | None = None,
    colormap: str = "viridis",
    alpha: float = 0.6,
) -> plt.Axes:
    """Overlay parity time traces for multiple k values at one λ.

    Parameters
    ----------
    records:
        List of ``KSpaceRecord`` objects.
    lambda_val:
        The λ value to plot.  Nearest available is used.
    component:
        ``"real"`` (default) or ``"imag"`` — which component to plot.
        Only like-with-like components are ever overlaid to avoid artefacts.
    k_indices:
        Subset of k indices to plot.  Default: all.
    ax:
        Axes to draw into.  A new figure is created if not supplied.
    colormap:
        Matplotlib colormap name for colouring the k traces.
    alpha:
        Line transparency.

    Returns
    -------
    plt.Axes
    """
    if component not in ("real", "imag"):
        raise ValueError("component must be 'real' or 'imag'.")

    # Find nearest record
    lam_arr = np.array([r.lambda_val for r in records])
    rec = records[int(np.argmin(np.abs(lam_arr - lambda_val)))]

    if k_indices is None:
        k_indices = list(range(rec.n_k))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / max(len(k_indices) - 1, 1)) for i in range(len(k_indices))]

    for color, ki in zip(colors, k_indices):
        sig = rec.signal[ki, :]
        y = sig.real if component == "real" else sig.imag
        ax.plot(rec.t, y, color=color, alpha=alpha, lw=0.8)

    comp_label = "Re" if component == "real" else "Im"
    ax.set_xlabel("t  [µs]")
    ax.set_ylabel(f"{comp_label}(parity)")
    ax.set_title(f"λ = {rec.lambda_val:.3f}  |  {len(k_indices)} k traces  ({comp_label})")

    # Colourbar for k axis
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(rec.k_array[k_indices[0]], rec.k_array[k_indices[-1]]),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="k  [rad]")
    return ax


# ---------------------------------------------------------------------------
# 3. Interactive slider: poles vs k, sweep lambda
# ---------------------------------------------------------------------------

def plot_poles_vs_k_slider(
    result: KSpacePolesResult,
    *,
    pole_indices: list[int] | None = None,
    continuity_sort: bool = True,
    deriv_weight: float = 0.1,
) -> None:
    """Interactive figure: poles vs k with a lambda slider.

    Two stacked axes are shown:
    - Top:    decay rates −Re(pole) vs k at the selected λ
    - Bottom: frequencies Im(pole) vs k at the selected λ

    The Slider widget uses blitting for a responsive update.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()``.
    pole_indices:
        Which pole slots to show.  Default: all.
    continuity_sort:
        If True (default), reorder pole columns along k for each λ via
        Hungarian assignment so each colour follows the most continuous
        track.  Set False to use the canonical sort from the analysis.
    deriv_weight:
        Tiebreaker weight for the continuity sort (see ``sort_poles_along_k``).
    """
    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    fig, (ax_d, ax_f) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={"hspace": 0.05, "bottom": 0.15},
    )

    k = result.k_array
    lam_vals = result.lambda_values
    init_lam_idx = len(lam_vals) // 2

    if continuity_sort:
        poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    else:
        poles_all = result.poles
    decay_all = -poles_all.real
    freq_all = poles_all.imag

    # --- initial draw ---
    decay_lines = []
    freq_lines = []
    for pi in pole_indices:
        color = _pole_color(pi)
        dl, = ax_d.plot(k, decay_all[init_lam_idx, :, pi],
                        "o-", color=color, ms=3, lw=1, label=f"track {pi}")
        fl, = ax_f.plot(k, freq_all[init_lam_idx, :, pi],
                        "o-", color=color, ms=3, lw=1)
        decay_lines.append(dl)
        freq_lines.append(fl)

    ax_d.set_ylabel("Decay rate  −Re(pole)  [1/µs]")
    ax_f.set_ylabel("Frequency  Im(pole)  [1/µs]")
    ax_f.set_xlabel("k  [rad]")
    ax_d.legend(fontsize=8, loc="upper right")

    lam_title = ax_d.set_title(
        f"λ = {lam_vals[init_lam_idx]:.4f}", fontsize=11
    )

    # --- slider ---
    ax_slider = fig.add_axes([0.15, 0.04, 0.70, 0.03])
    slider = Slider(
        ax_slider,
        "λ",
        valmin=lam_vals.min(),
        valmax=lam_vals.max(),
        valinit=lam_vals[init_lam_idx],
        valstep=np.diff(lam_vals).min() if len(lam_vals) > 1 else 1.0,
    )

    def _update(val):
        lam_idx = result.nearest_lambda_idx(slider.val)
        for j, pi in enumerate(pole_indices):
            decay_lines[j].set_ydata(decay_all[lam_idx, :, pi])
            freq_lines[j].set_ydata(freq_all[lam_idx, :, pi])
        lam_title.set_text(f"λ = {lam_vals[lam_idx]:.4f}")
        # Autoscale both axes
        for ax in (ax_d, ax_f):
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    # Keep a reference to the slider so it isn't garbage-collected
    fig._kspace_slider = slider  # noqa: SLF001
    plt.show()


# ---------------------------------------------------------------------------
# 4. Panel grid: poles vs k for selected lambda values
# ---------------------------------------------------------------------------

def _resolve_lambda_indices(
    result: KSpacePolesResult,
    lambda_values: list[float] | int | None,
    *,
    default_n_panels: int = 6,
) -> list[int]:
    """Translate the user-facing ``lambda_values`` arg into a list of indices."""
    if lambda_values is None:
        n_panels = min(default_n_panels, result.n_lambda)
        lam_indices = list(np.round(np.linspace(0, result.n_lambda - 1, n_panels)).astype(int))
    elif isinstance(lambda_values, int):
        # ``that many evenly-spaced λ values`` across the full range — same form
        # as the ``None`` branch above, just with a caller-supplied count.  The
        # old bounds ``linspace(n_lambda-1-n_panels, n_lambda-1, ...)`` dipped to
        # index -1 whenever ``n_panels == n_lambda`` (i.e. as many panels as λ
        # values were requested); numpy read -1 as the last λ, so λ_max was
        # duplicated at the front, one interior λ was dropped by rounding, and
        # ``sorted(set())`` placed λ_max in the top-left panel.
        n_panels = min(lambda_values, result.n_lambda)
        lam_indices = list(np.round(np.linspace(0, result.n_lambda - 1, n_panels)).astype(int))
    else:
        lam_indices = [result.nearest_lambda_idx(lv) for lv in lambda_values]
    return sorted(set(lam_indices))


def plot_poles_vs_k_grid(
    result: KSpacePolesResult,
    *,
    lambda_values: list[float] | int | None = None,
    n_cols: int = 3,
    component: str = "decay",
    pole_indices: list[int] | None = None,
    continuity_sort: bool = True,
    deriv_weight: float = 0.1,
) -> plt.Figure:
    """Static panel grid — one subplot per λ, poles vs k.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()``.
    lambda_values:
        Which λ values to include.
        - ``None``: auto-select up to 6 evenly-spaced λ values.
        - ``int``: select that many evenly-spaced λ values.
        - ``list[float]``: use nearest-neighbour lookup for each value.
    n_cols:
        Number of panel columns.
    component:
        ``"decay"`` (default) — plot −Re(poles).
        ``"frequency"`` — plot Im(poles).
        ``"both"`` — decay on top, frequency on bottom of each panel.
    pole_indices:
        Which pole slots to show.  Default: all.
    continuity_sort:
        If True (default), reorder pole columns along k via Hungarian
        assignment so each colour follows the most continuous track.
    deriv_weight:
        Tiebreaker weight for the continuity sort.

    Returns
    -------
    plt.Figure
    """
    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(result, lambda_values)
    n_panels = len(lam_indices)
    n_rows = (n_panels + n_cols - 1) // n_cols

    rows_per_panel = 2 if component == "both" else 1
    fig, axes_flat = plt.subplots(
        n_rows * rows_per_panel, n_cols,
        figsize=(4 * n_cols, 3 * n_rows * rows_per_panel),
        squeeze=False,
    )

    k = result.k_array
    if continuity_sort:
        poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    else:
        poles_all = result.poles
    decay_all = -poles_all.real
    freq_all = poles_all.imag

    for panel_idx, lam_idx in enumerate(lam_indices):
        row = (panel_idx // n_cols) * rows_per_panel
        col = panel_idx % n_cols

        if component == "both":
            ax_d = axes_flat[row, col]
            ax_f = axes_flat[row + 1, col]
        else:
            ax_d = axes_flat[row, col]
            ax_f = None

        lam_val = result.lambda_values[lam_idx]

        track_label = "track" if continuity_sort else "p"
        for pi in pole_indices:
            color = _pole_color(pi)
            lbl = f"{track_label}{pi}"
            if component in ("decay", "both"):
                ax_d.plot(k, decay_all[lam_idx, :, pi],
                          "-", color=color, lw=1, label=lbl)
            if component == "frequency":
                ax_d.plot(k, freq_all[lam_idx, :, pi],
                          "-", color=color, lw=1, label=lbl)
            if component == "both" and ax_f is not None:
                ax_f.plot(k, freq_all[lam_idx, :, pi],
                          "-", color=color, lw=1)

        ax_d.set_title(f"λ = {lam_val:.3f}", fontsize=9)
        ax_d.set_xlabel("k  [rad]", fontsize=8)
        if component == "decay":
            ax_d.set_ylabel("−Re(pole)  [1/µs]", fontsize=8)
        elif component == "frequency":
            ax_d.set_ylabel("Im(pole)  [1/µs]", fontsize=8)
        else:
            ax_d.set_ylabel("−Re(pole)", fontsize=8)
            if ax_f is not None:
                ax_f.set_ylabel("Im(pole)", fontsize=8)
                ax_f.set_xlabel("k  [rad]", fontsize=8)

        if panel_idx == 0:
            ax_d.legend(fontsize=7)

    # Hide empty panels
    for panel_idx in range(n_panels, n_rows * n_cols):
        row = (panel_idx // n_cols) * rows_per_panel
        col = panel_idx % n_cols
        for r in range(rows_per_panel):
            axes_flat[row + r, col].set_visible(False)

    comp_label = {"decay": "Decay rates", "frequency": "Frequencies", "both": "Poles"}[component]
    sort_label = " (continuity-sorted)" if continuity_sort else ""
    fig.suptitle(f"k-space MPM  |  {comp_label} vs k{sort_label}", fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Rotatable 3D grid: pole tracks (Re, Im) vs k
# ---------------------------------------------------------------------------

def plot_poles_vs_k_3d_grid(
    result: KSpacePolesResult,
    *,
    lambda_values: list[float] | int | None = None,
    n_cols: int = 3,
    pole_indices: list[int] | None = None,
    continuity_sort: bool = True,
    deriv_weight: float = 0.1,
) -> plt.Figure:
    """Rotatable 3D panel grid — one subplot per λ.

    Each subplot is a 3D Axes with::

        x = k        (rad)
        y = Re(pole) (1/µs)
        z = Im(pole) (1/µs)

    Within each panel the pole columns are reordered along k via Hungarian
    assignment so that each colour follows the most continuous trajectory
    in the complex plane (primary cost: complex-plane distance to the
    previous k; tiebreaker: distance to a linear extrapolation in k).

    The figure uses an interactive backend by default — drag with the mouse
    to rotate each subplot independently.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()``.
    lambda_values:
        Which λ values to include.  Same conventions as
        ``plot_poles_vs_k_grid``.
    n_cols:
        Number of panel columns.
    pole_indices:
        Which pole tracks to show.  Default: all.
    continuity_sort:
        If True (default), apply the continuity sort.  If False, plot the
        canonical pole columns from the analysis.
    deriv_weight:
        Tiebreaker weight for the continuity sort.

    Returns
    -------
    plt.Figure
    """
    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(result, lambda_values)
    n_panels = len(lam_indices)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    k = result.k_array
    if continuity_sort:
        poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    else:
        poles_all = result.poles

    for panel_idx, lam_idx in enumerate(lam_indices):
        ax = fig.add_subplot(n_rows, n_cols, panel_idx + 1, projection="3d")

        for pi in pole_indices:
            color = _pole_color(pi)
            track = poles_all[lam_idx, :, pi]
            valid = ~np.isnan(track)
            ax.plot(
                k[valid], track.real[valid], track.imag[valid],
                "o-", color=color, ms=2.5, lw=1, label=f"track {pi}",
            )

        lam_val = result.lambda_values[lam_idx]
        ax.set_title(f"λ = {lam_val:.3f}", fontsize=10)
        ax.set_xlabel("k  [rad]", fontsize=8)
        ax.set_ylabel("Re(pole)", fontsize=8)
        ax.set_zlabel("Im(pole)", fontsize=8)
        ax.tick_params(labelsize=7)
        if panel_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    sort_label = " (continuity-sorted)" if continuity_sort else ""
    fig.suptitle(
        f"k-space poles in complex plane vs k{sort_label}", fontsize=12,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5b. Same 3D grid as a standalone Plotly HTML (shareable, no server)
# ---------------------------------------------------------------------------

_POLE_COLORS_HEX = [
    "#1f77b4",  # tab:blue
    "#ff7f0e",  # tab:orange
    "#2ca02c",  # tab:green
    "#d62728",  # tab:red
    "#9467bd",  # tab:purple
]


# Browsers keep only a limited number of live WebGL contexts per page (Chrome
# / Edge cap at ~16); plotly uses one per 3D subplot, and once the cap is passed
# the oldest contexts are silently dropped so those panels render blank.  We
# therefore paginate the 3D grid so no single HTML page ever holds more than
# this many 3D panels.  Kept safely below 16 to leave room for the browser.
_MAX_3D_PANELS_PER_HTML = 12


def _build_poles_vs_k_3d_figure(
    result: KSpacePolesResult,
    lam_indices: list[int],
    k: np.ndarray,
    poles_all: np.ndarray,
    pole_indices: list[int],
    n_cols: int,
    *,
    continuity_sort: bool,
    part: tuple[int, int] = (1, 1),
):
    """Build one Plotly 3D grid figure for the given ``lam_indices`` panels."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_panels = len(lam_indices)
    n_rows = (n_panels + n_cols - 1) // n_cols

    titles = [f"λ = {result.lambda_values[li]:.3f}" for li in lam_indices]
    # Only allocate a 3D scene for cells that actually hold a panel; trailing
    # cells (when n_panels doesn't fill the last row) get ``None`` so they render
    # as empty space instead of blank, axis-boxed 3D cubes.
    specs = [
        [({"type": "scene"} if r * n_cols + c < n_panels else None)
         for c in range(n_cols)]
        for r in range(n_rows)
    ]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.06,
    )

    for panel_idx, lam_idx in enumerate(lam_indices):
        r = panel_idx // n_cols + 1
        c = panel_idx % n_cols + 1
        for pi in pole_indices:
            color = _POLE_COLORS_HEX[pi % len(_POLE_COLORS_HEX)]
            track = poles_all[lam_idx, :, pi]
            valid = ~np.isnan(track)
            fig.add_trace(
                go.Scatter3d(
                    x=k[valid],
                    y=track.real[valid],
                    z=track.imag[valid],
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                    marker={"color": color, "size": 3},
                    name=f"track {pi}",
                    legendgroup=f"track{pi}",
                    showlegend=(panel_idx == 0),
                    hovertemplate=(
                        "k=%{x:.3f}<br>Re=%{y:.4f}<br>Im=%{z:.4f}"
                        "<extra>track " + str(pi) + "</extra>"
                    ),
                ),
                row=r, col=c,
            )

    # Axis labels — Plotly puts them per-scene
    for i in range(1, n_panels + 1):
        scene_key = "scene" if i == 1 else f"scene{i}"
        fig.update_layout({
            scene_key: {
                "xaxis_title": "k [rad]",
                "yaxis_title": "Re(pole)",
                "zaxis_title": "Im(pole)",
            }
        })

    sort_label = " (continuity-sorted)" if continuity_sort else ""
    part_label = f" — part {part[0]}/{part[1]}" if part[1] > 1 else ""
    fig.update_layout(
        title=f"k-space poles in complex plane vs k{sort_label}{part_label}",
        height=420 * n_rows,
        width=520 * n_cols,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig


def export_poles_vs_k_3d_html(
    result: KSpacePolesResult,
    path: str,
    *,
    lambda_values: list[float] | int | None = None,
    n_cols: int = 3,
    pole_indices: list[int] | None = None,
    continuity_sort: bool = True,
    deriv_weight: float = 0.1,
    include_plotlyjs: str = "cdn",
    max_panels_per_file: int = _MAX_3D_PANELS_PER_HTML,
) -> list[str]:
    """Write the rotatable 3D grid as one or more standalone HTML files.

    Same content as ``plot_poles_vs_k_3d_grid``, but rendered with Plotly so
    each subplot is fully interactive (rotate / zoom / pan / hover) in any
    modern web browser.  Each output is a self-contained ``.html`` file — no
    Python or server required to view it; recipients just open it in a browser.

    Because a browser only keeps ~16 live WebGL contexts per page (one per 3D
    subplot), the λ panels are split across as many files as needed so that no
    page exceeds ``max_panels_per_file`` — otherwise the oldest panels (the top
    row) render blank.  With one part the file is named exactly ``path``; with
    several, a ``_part{i}of{n}`` suffix is inserted before the extension.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()``.
    path:
        Destination path (``.html``).  Parent directories are created.
    lambda_values, n_cols, pole_indices, continuity_sort, deriv_weight:
        Same semantics as ``plot_poles_vs_k_3d_grid``.  All resolved λ panels
        are shown; the count adapts to the scan.
    include_plotlyjs:
        Forwarded to ``plotly.io.write_html``:
        - ``"cdn"`` (default): tiny file (~few KB), needs an internet
          connection on first open to fetch plotly.js from a CDN.
        - ``True``: embed plotly.js (~3 MB), file works fully offline.
        - ``"directory"``: write a ``plotly.min.js`` next to the HTML.
    max_panels_per_file:
        Maximum 3D panels per HTML page (WebGL-context budget).

    Returns
    -------
    list[str]
        Absolute paths to every written HTML file (one per part).
    """
    from pathlib import Path as _Path
    from mpm_analysis.utils.windows_paths import win_long_path
    import os as _os

    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(result, lambda_values)
    if continuity_sort:
        poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    else:
        poles_all = result.poles
    k = result.k_array

    # Split the panels into WebGL-safe chunks (all λ are kept).
    step = max(1, max_panels_per_file)
    chunks = [lam_indices[i:i + step] for i in range(0, len(lam_indices), step)]
    n_parts = len(chunks)

    out = _Path(path)
    _os.makedirs(win_long_path(out.parent), exist_ok=True)

    written: list[str] = []
    for part_i, chunk in enumerate(chunks, start=1):
        fig = _build_poles_vs_k_3d_figure(
            result, chunk, k, poles_all, pole_indices, n_cols,
            continuity_sort=continuity_sort, part=(part_i, n_parts),
        )
        if n_parts == 1:
            part_path = out
        else:
            part_path = out.with_name(
                f"{out.stem}_part{part_i}of{n_parts}{out.suffix}")
        fig.write_html(win_long_path(part_path),
                       include_plotlyjs=include_plotlyjs, full_html=True)
        written.append(str(part_path.resolve()))
    return written


# ---------------------------------------------------------------------------
# 5c. Parametric (Re, Im) plots — eigenvalue tracks vs k for each λ
# ---------------------------------------------------------------------------
#
# Convention from reference figures: horizontal axis = Im(λ), vertical = Re(λ).
# k=0 and k=2π share an "endpoint" marker (orange dot) so the user can read
# off topological periodicity at a glance — if the two markers overlap the
# track is 2π-periodic; if they sit on different tracks, period is 4π
# (the two tracks have swapped over one Brillouin-zone traversal).

_K_ENDPOINT_KW = {                      # k=0 and k=2π
    "marker": "o", "s": 24,
    "facecolor": "#ff7f0e", "edgecolor": "k", "linewidth": 0.35,
    "alpha": 0.82, "zorder": 5,
}
_K_PI_KW = {                            # k=π
    "marker": "s", "s": 22,
    "facecolor": "#808000", "edgecolor": "k", "linewidth": 0.35,
    "alpha": 0.82, "zorder": 5,
}


def _annotate_k_markers_2d(
    ax: plt.Axes,
    track_complex: np.ndarray,
    k_array: np.ndarray,
    *,
    text_color: str,
    label_text: bool = True,
    fontsize: int = 7,
    track_index: int = 0,
) -> None:
    """Drop k=0, k=π, k=2π markers on a 2-D parametric track.

    Labels are placed with fixed display-space offsets scaled by track_index
    so that labels from inner and outer loops don't overlap:
        k=0   →  above the marker
        k=2π  →  below the marker
        k=π   →  to the right of the marker
    The text colour matches the eigenvalue track colour so each label can
    be visually tied back to its curve in the vs-k plots.
    """
    valid = ~np.isnan(track_complex)
    if not valid.any():
        return
    k_valid = k_array[valid]
    z_valid = track_complex[valid]

    i0 = 0
    i_end = len(k_valid) - 1
    i_pi = int(np.argmin(np.abs(k_valid - np.pi)))

    ax.scatter([z_valid[i0].imag], [z_valid[i0].real], **_K_ENDPOINT_KW)
    ax.scatter([z_valid[i_end].imag], [z_valid[i_end].real], **_K_ENDPOINT_KW)
    ax.scatter([z_valid[i_pi].imag], [z_valid[i_pi].real], **_K_PI_KW)

    if not label_text:
        return

    # Scale offsets by track index so labels from multiple tracks don't overlap.
    # Track 0 gets base offset; track 1 gets ~1.8× further out, etc.
    base = 8
    scale = 1.0 + 0.85 * track_index
    d_vert = base * scale
    d_horiz = base * scale
    items = (
        ("k=0",  i0,    (0,        +d_vert),  "center", "bottom"),
        ("k=2π", i_end, (0,        -d_vert),  "center", "top"),
        ("k=π",  i_pi,  (d_horiz,  0),        "left",   "center"),
    )
    for label, idx, offset, ha, va in items:
        ax.annotate(
            label, (z_valid[idx].imag, z_valid[idx].real),
            xytext=offset, textcoords="offset points",
            fontsize=fontsize, color=text_color, fontweight="bold",
            ha=ha, va=va,
            path_effects=[
                path_effects.withStroke(linewidth=1.8, foreground="white"),
            ],
        )


def plot_eigenvalues_parametric_grid(
    result: KSpacePolesResult,
    *,
    lambda_values: list[float] | int | None = None,
    n_cols: int = 3,
    pole_indices: list[int] | None = None,
    deriv_weight: float = 0.1,
    annotate_k: bool = False,
    marker_legend: bool = True,
) -> plt.Figure:
    """Grid of parametric eigenvalue plots — one panel per λ.

    Each panel plots, for every continuity-sorted track, the curve traced
    in the complex plane as k sweeps from 0 to 2π::

        x  =  Im(pole(λ, k))
        y  =  Re(pole(λ, k))

    Markers:
        - orange dot   →  k=0  and  k=2π (overlap iff 2π-periodic)
        - olive square →  k=π

    By default the presentation plot keeps only compact endpoint markers.  Pass
    ``annotate_k=True`` for per-track text labels when inspecting details.
    """
    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(result, lambda_values)
    n_panels = len(lam_indices)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes_flat = plt.subplots(
        n_rows, n_cols,
        figsize=(4.8 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )

    poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    k = result.k_array

    for panel_idx, lam_idx in enumerate(lam_indices):
        row, col = divmod(panel_idx, n_cols)
        ax = axes_flat[row, col]

        for track_enum, pi in enumerate(pole_indices):
            color = _pole_color(pi)
            track = poles_all[lam_idx, :, pi]
            valid = ~np.isnan(track)
            ax.plot(
                track.imag[valid], track.real[valid],
                "-", color=color, lw=1.8, alpha=0.92,
                label=f"track {pi}",
            )
            _annotate_k_markers_2d(
                ax, track, k,
                text_color=color,
                label_text=annotate_k,
                track_index=track_enum,
            )

        lam_val = result.lambda_values[lam_idx]
        ax.set_title(f"lambda = {lam_val:.3f}", fontsize=8, loc="left")
        if row == n_rows - 1:
            ax.set_xlabel("Im(pole)  [1/us]", fontsize=7)
        if col == 0:
            ax.set_ylabel("Re(pole)  [1/us]", fontsize=7)
        ax.axhline(0, color="0.78", lw=0.45)
        ax.axvline(0, color="0.78", lw=0.45)
        ax.tick_params(labelsize=6, width=0.5, length=2.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="0.92", lw=0.45)
        if panel_idx == 0:
            handles = [
                Line2D([0], [0], color=_pole_color(pi), lw=1.8,
                       label=f"track {pi}")
                for pi in pole_indices
            ]
            if marker_legend:
                handles.extend([
                    Line2D([0], [0], marker="o", linestyle="None",
                           markerfacecolor="#ff7f0e", markeredgecolor="k",
                           markeredgewidth=0.35, markersize=4,
                           label="k=0, 2pi"),
                    Line2D([0], [0], marker="s", linestyle="None",
                           markerfacecolor="#808000", markeredgecolor="k",
                           markeredgewidth=0.35, markersize=4,
                           label="k=pi"),
                ])
            ax.legend(handles=handles, fontsize=6, loc="best",
                      frameon=True, framealpha=0.86, borderpad=0.35,
                      handlelength=1.5)

    # Hide empty panels
    for panel_idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes_flat[row, col].set_visible(False)

    fig.suptitle(
        "Parametric eigenvalue tracks  (k: 0 to 2pi)  per lambda", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return fig


def export_eigenvalues_parametric_slider_html(
    result: KSpacePolesResult,
    path: str,
    *,
    pole_indices: list[int] | None = None,
    lambda_values: list[float] | int | None = None,
    deriv_weight: float = 0.1,
    include_plotlyjs: str | bool = "cdn",
) -> str:
    """Standalone 2-D parametric plot with a lambda slider.

    Each slider step shows the complex-plane eigenvalue tracks for one lambda
    as k runs from 0 to 2pi.  Endpoint markers stay small; exact k values are
    available on hover.
    """
    import plotly.graph_objects as go
    from pathlib import Path as _Path

    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(
        result, lambda_values, default_n_panels=result.n_lambda,
    )
    poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    k = result.k_array

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for lam_idx in lam_indices:
        for pi in pole_indices:
            track = poles_all[lam_idx, :, pi]
            valid = np.isfinite(track)
            if valid.any():
                xs.append(track.imag[valid])
                ys.append(track.real[valid])
    if xs:
        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)
        x_pad = 0.08 * max(float(np.ptp(x_all)), 1e-9)
        y_pad = 0.08 * max(float(np.ptp(y_all)), 1e-9)
        x_range = [float(x_all.min() - x_pad), float(x_all.max() + x_pad)]
        y_range = [float(y_all.min() - y_pad), float(y_all.max() + y_pad)]
    else:
        x_range = y_range = None

    fig = go.Figure()
    trace_groups: list[list[int]] = []
    for lam_pos, lam_idx in enumerate(lam_indices):
        lam_val = float(result.lambda_values[lam_idx])
        group_indices: list[int] = []
        visible = lam_pos == 0
        for track_enum, pi in enumerate(pole_indices):
            color = _POLE_COLORS_HEX[pi % len(_POLE_COLORS_HEX)]
            track = poles_all[lam_idx, :, pi]
            valid = np.isfinite(track)
            if not valid.any():
                continue
            k_valid = k[valid]
            z_valid = track[valid]
            i_pi = int(np.argmin(np.abs(k_valid - np.pi)))

            group_indices.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=z_valid.imag,
                y=z_valid.real,
                mode="lines",
                line={"color": color, "width": 3},
                name=f"track {pi}",
                legendgroup=f"track{pi}",
                showlegend=True,
                visible=visible,
                customdata=k_valid,
                hovertemplate=(
                    "k=%{customdata:.3f} rad<br>"
                    "Im=%{x:.4f}<br>Re=%{y:.4f}"
                    f"<extra>lambda={lam_val:.3f}, track {pi}</extra>"
                ),
            ))

            group_indices.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[z_valid.imag[0], z_valid.imag[-1]],
                y=[z_valid.real[0], z_valid.real[-1]],
                mode="markers",
                marker={
                    "color": "#ff7f0e", "size": 7,
                    "line": {"color": "black", "width": 0.6},
                },
                name="k=0, 2pi",
                legendgroup="k_endpoints",
                showlegend=(track_enum == 0),
                visible=visible,
                customdata=["0", "2pi"],
                hovertemplate=(
                    "k=%{customdata}<br>Im=%{x:.4f}<br>Re=%{y:.4f}"
                    f"<extra>lambda={lam_val:.3f}, track {pi}</extra>"
                ),
            ))

            group_indices.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[z_valid.imag[i_pi]],
                y=[z_valid.real[i_pi]],
                mode="markers",
                marker={
                    "color": "#808000", "size": 7, "symbol": "square",
                    "line": {"color": "black", "width": 0.6},
                },
                name="k=pi",
                legendgroup="k_pi",
                showlegend=(track_enum == 0),
                visible=visible,
                customdata=[float(k_valid[i_pi])],
                hovertemplate=(
                    "k=%{customdata:.3f} rad<br>Im=%{x:.4f}<br>Re=%{y:.4f}"
                    f"<extra>lambda={lam_val:.3f}, track {pi}</extra>"
                ),
            ))
        trace_groups.append(group_indices)

    steps = []
    n_traces = len(fig.data)
    for lam_pos, lam_idx in enumerate(lam_indices):
        visible = [False] * n_traces
        for trace_idx in trace_groups[lam_pos]:
            visible[trace_idx] = True
        lam_val = float(result.lambda_values[lam_idx])
        steps.append(dict(
            method="update",
            label=f"{lam_val:.3f}",
            args=[
                {"visible": visible},
                {"title": f"Parametric eigenvalue tracks | lambda = {lam_val:.3f}"},
            ],
        ))

    first_lambda = float(result.lambda_values[lam_indices[0]]) if lam_indices else float("nan")
    fig.update_layout(
        title=f"Parametric eigenvalue tracks | lambda = {first_lambda:.3f}",
        xaxis={
            "title": "Im(pole) [1/us]",
            "range": x_range,
            "zeroline": True,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
        },
        yaxis={
            "title": "Re(pole) [1/us]",
            "range": y_range,
            "zeroline": True,
            "showgrid": True,
            "gridcolor": "rgba(0,0,0,0.08)",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "lambda = ", "font": {"size": 13}},
            pad={"t": 48},
            steps=steps,
        )],
        legend={"font": {"size": 11}, "itemsizing": "constant"},
        font={"size": 12},
        width=1050,
        height=850,
        margin={"l": 70, "r": 30, "t": 70, "b": 105},
        template="plotly_white",
    )

    from mpm_analysis.utils.windows_paths import win_long_path
    import os as _os
    out = _Path(path)
    _os.makedirs(win_long_path(out.parent), exist_ok=True)
    fig.write_html(win_long_path(out), include_plotlyjs=include_plotlyjs, full_html=True)
    return str(out.resolve())


def plot_eigenvalues_parametric_3d(
    result: KSpacePolesResult,
    *,
    pole_indices: list[int] | None = None,
    lambda_values: list[float] | int | None = None,
    deriv_weight: float = 0.1,
    show_endpoint_markers: bool = False,
) -> plt.Figure:
    """Single rotatable 3D plot: all parametric tracks stacked over λ.

        x = Im(pole),  y = Re(pole),  z = λ

    Each (λ, track) is one curve in this 3D space; tracks share a colour
    across λ so a topology change shows up as a colour discontinuity along
    z.

    Endpoint markers (k=0 / k=π / k=2π) are *off* by default in the 3D
    view — too cluttered when stacking many λ slices.  Pass
    ``show_endpoint_markers=True`` to re-enable them.

    Drag the figure to rotate.
    """
    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(
        result, lambda_values, default_n_panels=result.n_lambda,
    )

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    k = result.k_array
    i_pi = int(np.argmin(np.abs(k - np.pi)))

    seen_labels: set[int] = set()
    for lam_idx in lam_indices:
        lam_val = float(result.lambda_values[lam_idx])
        for pi in pole_indices:
            color = _pole_color(pi)
            track = poles_all[lam_idx, :, pi]
            valid = ~np.isnan(track)
            if not valid.any():
                continue
            label = None
            if pi not in seen_labels:
                label = f"track {pi}"
                seen_labels.add(pi)
            zs = np.full(valid.sum(), lam_val)
            ax.plot(
                track.imag[valid], track.real[valid], zs,
                "-", color=color, lw=1.0, alpha=0.85, label=label,
            )
            if show_endpoint_markers:
                # k=0, k=2π: orange dots; k=π: olive square
                ax.scatter(
                    [track.imag[0]], [track.real[0]], [lam_val],
                    color="#ff7f0e", edgecolor="k", s=18, linewidth=0.4,
                )
                ax.scatter(
                    [track.imag[-1]], [track.real[-1]], [lam_val],
                    color="#ff7f0e", edgecolor="k", s=18, linewidth=0.4,
                )
                ax.scatter(
                    [track.imag[i_pi]], [track.real[i_pi]], [lam_val],
                    color="#808000", edgecolor="k", s=14,
                    linewidth=0.4, marker="s",
                )

    ax.set_xlabel("Im(pole)  [1/µs]")
    ax.set_ylabel("Re(pole)  [1/µs]")
    ax.set_zlabel("λ")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Parametric eigenvalue tracks (k: 0→2π) stacked over λ",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def export_eigenvalues_parametric_3d_html(
    result: KSpacePolesResult,
    path: str,
    *,
    pole_indices: list[int] | None = None,
    lambda_values: list[float] | int | None = None,
    deriv_weight: float = 0.1,
    show_endpoint_markers: bool = False,
    include_plotlyjs: str | bool = "cdn",
) -> str:
    """Standalone, rotatable 3D HTML version of ``plot_eigenvalues_parametric_3d``.

    Same axes (x=Im, y=Re, z=λ) and same colour scheme.  The legend
    toggles all curves of the same track together (legendgroup).

    Endpoint markers (k=0 / k=π / k=2π) are *off* by default — too cluttered
    when stacking many λ slices.  Pass ``show_endpoint_markers=True`` to
    re-enable them (orange dots for k=0/2π, olive squares for k=π).
    """
    import plotly.graph_objects as go
    from pathlib import Path as _Path

    if pole_indices is None:
        pole_indices = list(range(result.max_order))

    lam_indices = _resolve_lambda_indices(
        result, lambda_values, default_n_panels=result.n_lambda,
    )

    poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    k = result.k_array
    i_pi = int(np.argmin(np.abs(k - np.pi)))

    fig = go.Figure()
    seen: set[int] = set()
    for lam_idx in lam_indices:
        lam_val = float(result.lambda_values[lam_idx])
        for pi in pole_indices:
            color = _POLE_COLORS_HEX[pi % len(_POLE_COLORS_HEX)]
            track = poles_all[lam_idx, :, pi]
            valid = ~np.isnan(track)
            if not valid.any():
                continue
            zs = np.full(valid.sum(), lam_val)
            fig.add_trace(go.Scatter3d(
                x=track.imag[valid], y=track.real[valid], z=zs,
                mode="lines",
                line={"color": color, "width": 3},
                name=f"track {pi}",
                legendgroup=f"track{pi}",
                showlegend=(pi not in seen),
                hovertemplate=(
                    "Im=%{x:.4f}<br>Re=%{y:.4f}<br>λ=%{z:.3f}"
                    f"<extra>track {pi}</extra>"
                ),
            ))
            seen.add(pi)
            if show_endpoint_markers:
                fig.add_trace(go.Scatter3d(
                    x=[track.imag[0], track.imag[-1]],
                    y=[track.real[0], track.real[-1]],
                    z=[lam_val, lam_val],
                    mode="markers",
                    marker={
                        "color": "#ff7f0e", "size": 4,
                        "line": {"color": "black", "width": 0.5},
                    },
                    name="k=0 / k=2π",
                    legendgroup="k_endpoints",
                    showlegend=("kendp" not in seen),
                    hovertemplate="Im=%{x:.4f}<br>Re=%{y:.4f}<br>λ=%{z:.3f}<extra>k=0 / 2π</extra>",
                ))
                seen.add("kendp")
                fig.add_trace(go.Scatter3d(
                    x=[track.imag[i_pi]], y=[track.real[i_pi]], z=[lam_val],
                    mode="markers",
                    marker={
                        "color": "#808000", "size": 4, "symbol": "square",
                        "line": {"color": "black", "width": 0.5},
                    },
                    name="k=π",
                    legendgroup="k_pi",
                    showlegend=("kpi" not in seen),
                    hovertemplate="Im=%{x:.4f}<br>Re=%{y:.4f}<br>λ=%{z:.3f}<extra>k=π</extra>",
                ))
                seen.add("kpi")

    fig.update_layout(
        title="Parametric eigenvalue tracks (k: 0→2π) stacked over λ",
        scene={
            "xaxis_title": "Im(pole) [1/µs]",
            "yaxis_title": "Re(pole) [1/µs]",
            "zaxis_title": "λ",
        },
        height=800, width=1000,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    from mpm_analysis.utils.windows_paths import win_long_path
    import os as _os
    out = _Path(path)
    _os.makedirs(win_long_path(out.parent), exist_ok=True)
    fig.write_html(win_long_path(out), include_plotlyjs=include_plotlyjs, full_html=True)
    return str(out.resolve())


# ===========================================================================
# 5d. Pair-gap topology — arg(Δ) in the complex plane, k as the parameter
# ===========================================================================
#
# The braid invariant of a non-Hermitian band structure lives in the *gaps*, not
# in the eigenvalues themselves: for a pair of tracks (a, b) the object that
# carries the topology is
#
#     Δ_ab(k) = pole_a(k) − pole_b(k)
#
# traced in the complex plane as k sweeps the Brillouin zone.  Δ_ab can only
# reach the origin at an exceptional point (the two poles coincide), so away from
# EPs the curve is a closed loop around a puncture and its winding number is a
# quantized integer.  Watching that loop open, pinch through the origin, and
# re-close with a different winding *is* the topological transition.
#
# Conventions used here (stated on the figure as well):
#   * axes            x = Re(Δ),  y = Im(Δ)   — the ordinary complex plane;
#   * positive angle  counterclockwise (standard math convention);
#   * normalization   one full CCW loop = +1, i.e. W = (1/2π) ∮ d·arg Δ.
#
# W relates to the paper's pair index as ν_{a;b} = ν^(a,b) + ν^(b,a) = 2·W,
# because Δ_ba = −Δ_ab has the same winding as Δ_ab.  So a curve that visibly
# goes around twice (W = ±2) is ν_{a;b} = ±4.

_PAIR_LINE_COLORS = ["#7f3f98", "#d62728", "#17a2b8", "#8c564b", "#e377c2", "#7f7f7f"]


def gap_winding(z: np.ndarray) -> tuple[np.ndarray, float]:
    """Cumulative and total normalized winding of a sampled complex curve.

    Returns ``(cum, total)`` where ``cum[i]`` is the accumulated turning from the
    first sample to sample ``i`` in units of full loops, and ``total = cum[-1]``.
    No wraparound step is appended, so an open k-grid stays honest instead of
    being silently closed.

    Thin alias for
    :func:`mpm_analysis.analysis.braid_topology.winding_curve`, which is the
    canonical implementation.  Kept under this name so the pair-gap topology
    export below reads self-contained, but deliberately not a second copy of the
    estimator — the two must never be able to disagree.
    """
    from mpm_analysis.analysis.braid_topology import winding_curve

    return winding_curve(z)


def _origin_crossings(
    k: np.ndarray,
    z: np.ndarray,
    *,
    rel_tol: float = 0.06,
) -> list[tuple[float, float]]:
    """Local minima of ``|Δ|`` that come within ``rel_tol`` of the origin.

    The k position is refined by fitting a parabola to ``|Δ|`` over the three
    samples around the minimum, so the reported crossing k is not limited to the
    grid spacing.  Returns ``[(k_cross, |Δ|_min), …]``, most-central first.
    """
    r = np.abs(z)
    if r.size < 3:
        return []
    scale = float(np.median(r))
    if not np.isfinite(scale) or scale <= 0:
        return []
    thr = rel_tol * scale

    out: list[tuple[float, float]] = []
    for i in range(1, r.size - 1):
        if r[i] <= r[i - 1] and r[i] <= r[i + 1] and r[i] < thr:
            y0, y1, y2 = r[i - 1], r[i], r[i + 1]
            denom = y0 - 2.0 * y1 + y2
            shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-300 else 0.0
            shift = float(np.clip(shift, -1.0, 1.0))
            dk = k[i + 1] - k[i] if shift >= 0 else k[i] - k[i - 1]
            out.append((float(k[i] + shift * abs(dk)), float(y1)))
    return sorted(out, key=lambda t: t[1])


def export_pole_pair_topology_html(
    result: KSpacePolesResult,
    path: str,
    *,
    pole_indices: list[int] | None = None,
    lambda_values: list[float] | int | None = None,
    deriv_weight: float = 0.1,
    crossing_rel_tol: float = 0.06,
    include_plotlyjs: str | bool = "cdn",
) -> str:
    """Standalone HTML: pair-gap loops Δ_ab(k) with one λ slider **per panel**.

    One panel per pole pair (three panels for the usual 3-pole analysis:
    0–1, 0–2, 1–2).  Each panel draws the curve ``Δ_ab(k) = pole_a − pole_b`` in
    the complex plane with k as the curve parameter, markers colour-coded by k so
    a doubly-traversed loop is visible as two colour sweeps rather than one.

    Every panel carries its own independent λ slider, so one pair can be parked
    below the transition while another sits above it and the two compared
    side by side.

    Annotations per panel:
      * ``W`` — normalized winding, **one CCW loop = +1** (see module notes);
        ``ν_{a;b} = 2W`` is printed alongside for the paper convention.
      * loop count — |W| rounded, so "2 loops" is read off directly.
      * origin crossings — every local minimum of |Δ| closer than
        ``crossing_rel_tol`` × median|Δ| is marked with a red ✕ and labelled with
        its (parabolically refined) k value.  These are the exceptional points.
      * k endpoints — the first and last k of the grid, so an open Brillouin zone
        (``K_INDEX_RANGE`` trimming k=0 / k=2π) is visible rather than assumed
        closed; W is *not* artificially closed across the gap.

    Parameters
    ----------
    result, pole_indices, lambda_values, deriv_weight:
        As elsewhere in this module; tracks are continuity-sorted along k and
        matched across λ, so a pair label means the same two branches at every λ.
    crossing_rel_tol:
        Relative closeness to the origin that counts as a crossing.
    include_plotlyjs:
        Forwarded to ``plotly.io.write_html``.

    Returns
    -------
    str
        Path to the written HTML file.
    """
    import plotly.graph_objects as go
    from itertools import combinations
    from pathlib import Path as _Path
    from plotly.subplots import make_subplots

    if pole_indices is None:
        pole_indices = list(range(result.max_order))
    pairs = list(combinations(pole_indices, 2))
    if not pairs:
        raise ValueError("Need at least two pole tracks to form a gap pair.")

    lam_indices = _resolve_lambda_indices(
        result, lambda_values, default_n_panels=result.n_lambda,
    )
    poles_all = _sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
    k = np.asarray(result.k_array, dtype=float)

    n_pairs = len(pairs)
    fig = make_subplots(
        rows=1, cols=n_pairs,
        subplot_titles=[f"Δ = pole {a} − pole {b}" for a, b in pairs],
        horizontal_spacing=0.06,
    )

    def _axis_key(col: int, letter: str) -> str:
        """Layout key, e.g. ``xaxis`` / ``xaxis2`` — for relayout updates."""
        return f"{letter}axis" if col == 1 else f"{letter}axis{col}"

    def _axis_ref(col: int, letter: str) -> str:
        """Axis reference, e.g. ``x`` / ``x2`` — for ``scaleanchor``."""
        return letter if col == 1 else f"{letter}{col}"

    panel_trace_idx: list[list[int]] = []      # global trace ids, per panel
    panel_lam_slots: list[list[list[int]]] = []  # positions within the above
    panel_ranges: list[list[tuple]] = []       # (x_range, y_range) per λ

    for j, (a, b) in enumerate(pairs):
        col = j + 1
        line_color = _PAIR_LINE_COLORS[j % len(_PAIR_LINE_COLORS)]
        idx_list: list[int] = []
        lam_slots: list[list[int]] = []
        ranges: list[tuple] = []

        for lam_pos, lam_idx in enumerate(lam_indices):
            lam_val = float(result.lambda_values[lam_idx])
            gap = poles_all[lam_idx, :, a] - poles_all[lam_idx, :, b]
            valid = np.isfinite(gap)
            kv, zv = k[valid], gap[valid]
            visible = lam_pos == 0
            slots: list[int] = []

            if zv.size >= 2:
                cum, total = gap_winding(zv)
                crossings = _origin_crossings(kv, zv, rel_tol=crossing_rel_tol)
                r = np.abs(zv)
                i_min = int(np.argmin(r))
                # square, origin-containing view box for this (pair, λ)
                half = max(float(np.max(np.abs(zv.real))),
                           float(np.max(np.abs(zv.imag))), 1e-12) * 1.15
                x_rng = [-half, half]
                y_rng = [-half, half]
            else:
                cum, total = np.zeros(zv.size), float("nan")
                crossings, r, i_min = [], np.array([]), 0
                x_rng = y_rng = [-1.0, 1.0]
            ranges.append((x_rng, y_rng))

            # --- the loop itself --------------------------------------------
            slots.append(len(idx_list)); idx_list.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=zv.real, y=zv.imag,
                mode="lines+markers",
                line={"color": line_color, "width": 1.6},
                marker={
                    "color": kv, "colorscale": "Viridis", "size": 5,
                    "cmin": float(k.min()), "cmax": float(k.max()),
                    "showscale": (j == n_pairs - 1),
                    "colorbar": {"title": "k [rad]", "thickness": 12,
                                 "len": 0.75, "x": 1.01},
                },
                customdata=np.stack([kv, cum], axis=-1),
                hovertemplate=(
                    "k=%{customdata[0]:.4f} rad<br>"
                    "Re Δ=%{x:.4f}<br>Im Δ=%{y:.4f}<br>"
                    "turns so far=%{customdata[1]:+.3f}"
                    f"<extra>λ={lam_val:.3f} | {a}−{b}</extra>"
                ),
                visible=visible, showlegend=False,
            ), row=1, col=col)

            # --- k endpoints (periodicity check) ----------------------------
            slots.append(len(idx_list)); idx_list.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[zv.real[0], zv.real[-1]] if zv.size else [],
                y=[zv.imag[0], zv.imag[-1]] if zv.size else [],
                mode="markers",
                marker={"color": ["#2ca02c", "#ff7f0e"], "size": 10,
                        "symbol": ["circle", "diamond"],
                        "line": {"color": "black", "width": 0.8}},
                customdata=[kv[0], kv[-1]] if zv.size else [],
                hovertemplate=(
                    "k=%{customdata:.4f} rad<br>Re Δ=%{x:.4f}<br>Im Δ=%{y:.4f}"
                    f"<extra>λ={lam_val:.3f} | endpoint</extra>"
                ),
                visible=visible, showlegend=False,
            ), row=1, col=col)

            # --- origin crossings (exceptional points) ----------------------
            slots.append(len(idx_list)); idx_list.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[0.0] * len(crossings), y=[0.0] * len(crossings),
                mode="markers+text",
                marker={"color": "#d02500", "size": 13, "symbol": "x-thin",
                        "line": {"color": "#d02500", "width": 2.5}},
                text=[f"  k={kc:.3f}" for kc, _ in crossings],
                textposition="middle right",
                textfont={"color": "#d02500", "size": 11},
                customdata=[[kc, rmin] for kc, rmin in crossings],
                hovertemplate=(
                    "origin crossing<br>k=%{customdata[0]:.4f} rad<br>"
                    "|Δ|min=%{customdata[1]:.4g}"
                    f"<extra>λ={lam_val:.3f} | {a}−{b}</extra>"
                ),
                visible=visible, showlegend=False,
            ), row=1, col=col)

            # --- per-panel readout ------------------------------------------
            if zv.size >= 2:
                loops = abs(total)
                sense = "CCW" if total > 0 else ("CW" if total < 0 else "—")
                summary = (
                    f"W = {total:+.3f} loops ({sense})<br>"
                    f"ν<sub>{a};{b}</sub> = 2W = {2 * total:+.3f}<br>"
                    f"≈ {loops:.2f} loop(s)<br>"
                    f"min|Δ| = {r[i_min]:.4g} at k = {kv[i_min]:.3f}"
                )
                if crossings:
                    ks = ", ".join(f"{kc:.3f}" for kc, _ in crossings)
                    summary += f"<br><span style='color:#d02500'>crossing k = {ks}</span>"
            else:
                summary = "no valid poles at this λ"

            slots.append(len(idx_list)); idx_list.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[x_rng[0] + 0.04 * (x_rng[1] - x_rng[0])],
                y=[y_rng[1] - 0.03 * (y_rng[1] - y_rng[0])],
                mode="text", text=[summary], textposition="bottom right",
                textfont={"size": 11, "color": "#222"},
                hoverinfo="skip", visible=visible, showlegend=False,
            ), row=1, col=col)

            lam_slots.append(slots)

        # static origin marker — never toggled by the sliders
        fig.add_trace(go.Scatter(
            x=[0.0], y=[0.0], mode="markers",
            marker={"color": "black", "size": 8, "symbol": "cross-thin",
                    "line": {"color": "black", "width": 1.6}},
            hovertemplate="origin (Δ = 0 ⇒ exceptional point)<extra></extra>",
            showlegend=False,
        ), row=1, col=col)

        panel_trace_idx.append(idx_list)
        panel_lam_slots.append(lam_slots)
        panel_ranges.append(ranges)

    # --- one independent λ slider per panel ---------------------------------
    sliders = []
    for j, (a, b) in enumerate(pairs):
        col = j + 1
        xk, yk = _axis_key(col, "x"), _axis_key(col, "y")
        steps = []
        for lam_pos, lam_idx in enumerate(lam_indices):
            vis = [False] * len(panel_trace_idx[j])
            for slot in panel_lam_slots[j][lam_pos]:
                vis[slot] = True
            x_rng, y_rng = panel_ranges[j][lam_pos]
            steps.append(dict(
                method="update",
                label=f"{float(result.lambda_values[lam_idx]):.3f}",
                args=[
                    {"visible": vis},
                    {f"{xk}.range": x_rng, f"{yk}.range": y_rng},
                    panel_trace_idx[j],
                ],
            ))
        sliders.append(dict(
            active=0,
            steps=steps,
            x=j / n_pairs + 0.02,
            len=1.0 / n_pairs - 0.05,
            xanchor="left",
            y=-0.06,
            yanchor="top",
            pad={"t": 10, "b": 10},
            currentvalue={"prefix": f"λ ({a}−{b}) = ", "font": {"size": 12}},
            font={"size": 9},
        ))

    for j in range(n_pairs):
        col = j + 1
        x_rng, y_rng = panel_ranges[j][0]
        fig.update_xaxes(title_text="Re Δ  [1/µs]", range=x_rng, zeroline=True,
                         zerolinecolor="rgba(0,0,0,0.35)", showgrid=True,
                         gridcolor="rgba(0,0,0,0.07)", row=1, col=col)
        fig.update_yaxes(title_text="Im Δ  [1/µs]" if col == 1 else None,
                         range=y_rng, zeroline=True,
                         zerolinecolor="rgba(0,0,0,0.35)", showgrid=True,
                         gridcolor="rgba(0,0,0,0.07)",
                         scaleanchor=_axis_ref(col, "x"), scaleratio=1,
                         row=1, col=col)

    fig.update_layout(
        title={
            "text": (
                "Pair-gap topology&nbsp;&nbsp;Δ<sub>ab</sub>(k) = pole a − pole b"
                "&nbsp;&nbsp;|&nbsp;&nbsp;k is the curve parameter"
                "<br><sub>angles counterclockwise-positive in the (Re, Im) plane; "
                "normalized so one full loop = 1 &nbsp;(W = 1 → single loop, "
                "W = 2 → two loops). ν<sub>a;b</sub> = 2W. "
                "Marker colour = k; green circle / orange diamond = first / last k "
                "of the grid; red ✕ = crossing through the origin (exceptional "
                "point), labelled with its k.</sub>"
            ),
            "x": 0.5, "xanchor": "center", "font": {"size": 15},
        },
        height=680,
        width=max(520 * n_pairs, 900),
        margin={"l": 70, "r": 90, "t": 120, "b": 150},
        sliders=sliders,
        template="plotly_white",
    )

    from mpm_analysis.utils.windows_paths import win_long_path
    import os as _os
    out = _Path(path)
    _os.makedirs(win_long_path(out.parent), exist_ok=True)
    fig.write_html(win_long_path(out), include_plotlyjs=include_plotlyjs,
                   full_html=True)
    return str(out.resolve())


# ---------------------------------------------------------------------------
# 6. SVD diagnostic with lambda slider — for calibrating pole_order threshold
# ---------------------------------------------------------------------------

def plot_svd_vs_k_slider(
    result: KSpacePolesResult,
    *,
    n_singvals: int = 5,
    threshold: float | None = None,
    min_order: int = 2,
    max_order: int = 3,
) -> None:
    """Interactive SVD diagnostic: singular values + decision ratio vs k.

    Three stacked panels with a λ slider:

    1. Top — Hankel singular values S[0..n_singvals-1] vs k (log scale).
       Each singular value is a separate line.

    2. Middle — gap ratios vs k:
           ratio_23 = S[min_order-1] / S[min_order]    (default: S[1]/S[2])
           ratio_34 = S[min_order]   / S[max_order]    (default: S[2]/S[3])

    3. Bottom — decision ratio  ratio_34 / ratio_23  vs k.
       A horizontal line marks the configured threshold.  Cells that fall
       below the line use ``min_order`` poles; cells above use ``max_order``.
       Move the threshold up/down (manually) to see how the boundary shifts.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()``.  Must contain
        the ``svd_singvals`` array.
    n_singvals:
        How many leading singular values to plot in the top panel.
    threshold:
        Threshold to overlay on the bottom panel.  Defaults to
        ``result.metadata['threshold']`` if present, else 0.5.
    min_order, max_order:
        Pole-order bounds matching the analysis configuration.
        Defaults match the standard ``PoleOrderConfig`` (2 and 3).
    """
    sv = result.svd_singvals  # (n_lambda, n_k, n_sv_stored)
    if sv.ndim != 3 or sv.shape[2] < max_order + 1:
        raise ValueError(
            f"svd_singvals has shape {sv.shape}; need at least {max_order + 1} "
            "singular values stored per cell."
        )

    if threshold is None:
        threshold = float(result.metadata.get("threshold", 0.5))

    k = result.k_array
    lam_vals = result.lambda_values
    init_idx = len(lam_vals) // 2

    fig, (ax_sv, ax_gaps, ax_dec) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True,
        gridspec_kw={"hspace": 0.08, "bottom": 0.12},
    )

    # --- Panel 1: singular values vs k (log scale) ---
    n_show = min(n_singvals, sv.shape[2])
    sv_lines = []
    cmap = plt.get_cmap("viridis")
    for i in range(n_show):
        color = cmap(i / max(n_show - 1, 1))
        line, = ax_sv.semilogy(k, sv[init_idx, :, i], "-", color=color,
                               lw=1, label=f"S[{i}]")
        sv_lines.append(line)
    ax_sv.set_ylabel("Singular values  (log)")
    ax_sv.legend(fontsize=8, ncol=n_show, loc="upper right")
    ax_sv.grid(True, alpha=0.3)

    # --- Panel 2: gap ratios vs k ---
    def _ratios(lam_idx: int) -> tuple[np.ndarray, np.ndarray]:
        s = sv[lam_idx]                                  # (n_k, n_sv_stored)
        denom_23 = np.where(s[:, min_order]    > 1e-12, s[:, min_order],    np.nan)
        denom_34 = np.where(s[:, max_order]    > 1e-12, s[:, max_order],    np.nan)
        r23 = s[:, min_order - 1] / denom_23             # S[1]/S[2]
        r34 = s[:, min_order]     / denom_34             # S[2]/S[3]
        return r23, r34

    r23_init, r34_init = _ratios(init_idx)
    line_r23, = ax_gaps.semilogy(k, r23_init, "-", color="tab:blue",
                                 lw=1.2, label=f"ratio_23 = S[{min_order-1}]/S[{min_order}]")
    line_r34, = ax_gaps.semilogy(k, r34_init, "-", color="tab:red",
                                 lw=1.2, label=f"ratio_34 = S[{min_order}]/S[{max_order}]")
    ax_gaps.set_ylabel("Gap ratios  (log)")
    ax_gaps.legend(fontsize=8, loc="upper right")
    ax_gaps.grid(True, alpha=0.3)

    # --- Panel 3: decision metric vs k ---
    decision_init = r34_init / r23_init
    line_dec, = ax_dec.plot(k, decision_init, "-", color="tab:purple", lw=1.2,
                            label="ratio_34 / ratio_23")
    thresh_line = ax_dec.axhline(
        threshold, color="k", ls="--", lw=1,
        label=f"threshold = {threshold:.2f}",
    )
    ax_dec.set_ylabel("Decision metric")
    ax_dec.set_xlabel("k  [rad]")
    ax_dec.legend(fontsize=8, loc="upper right")
    ax_dec.grid(True, alpha=0.3)
    ax_dec.text(
        0.01, 0.97,
        f"above threshold → order={max_order}\nbelow threshold → order={min_order}",
        transform=ax_dec.transAxes, va="top", ha="left",
        fontsize=8, alpha=0.7,
    )

    title = ax_sv.set_title(f"λ = {lam_vals[init_idx]:.4f}", fontsize=11)

    # --- Slider ---
    ax_slider = fig.add_axes([0.15, 0.04, 0.70, 0.03])
    slider = Slider(
        ax_slider, "λ",
        valmin=lam_vals.min(), valmax=lam_vals.max(),
        valinit=lam_vals[init_idx],
        valstep=np.diff(lam_vals).min() if len(lam_vals) > 1 else 1.0,
    )

    def _update(_val):
        lam_idx = result.nearest_lambda_idx(slider.val)
        for i, line in enumerate(sv_lines):
            line.set_ydata(sv[lam_idx, :, i])
        r23, r34 = _ratios(lam_idx)
        line_r23.set_ydata(r23)
        line_r34.set_ydata(r34)
        line_dec.set_ydata(r34 / r23)
        title.set_text(f"λ = {lam_vals[lam_idx]:.4f}")
        for ax in (ax_sv, ax_gaps, ax_dec):
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    fig._kspace_svd_slider = slider  # noqa: SLF001
    plt.show()


# ===========================================================================
# Static SVD spectrum grid (moved here from plotting/compressed_click_stats)
# ===========================================================================

def plot_svd_spectrum_grid(
    result: KSpacePolesResult,
    *,
    lambda_values=None,
    n_cols: int = 3,
    k_val: float | None = None,
) -> plt.Figure:
    """Static SVD diagnostic: Hankel singular-value spectrum per λ at fixed k.

    For each selected λ, plots normalized singular values ``S_i / S_0`` (log)
    vs index at ``k ≈ k_val`` (default π).  The elbow is what the order-selection
    ratio test reads; vertical guides mark ``min_order`` / ``max_order`` and each
    panel is annotated with the order chosen for that cell.
    """
    from mpm_analysis.plotting._style import (
        TWO_COLUMN_WIDTH, MARKER_SIZE, LINE_WIDTH, AZURE_BLUE, PURPLE_MAGENTA,
        apply_paper_style, make_ax_lines_wider,
    )
    apply_paper_style()
    lam = np.asarray(result.lambda_values)
    k_arr = np.asarray(result.k_array)
    sv = np.asarray(result.svd_singvals)              # (n_lambda, n_k, n_sv)
    k_val = float(np.pi) if k_val is None else float(k_val)
    k_idx = int(np.argmin(np.abs(k_arr - k_val)))
    max_order = result.metadata.get("max_order")
    min_order = result.metadata.get("min_order")

    if lambda_values is None:
        n_show = min(6, len(lam))
        sel = np.linspace(0, len(lam) - 1, n_show).round().astype(int)
    else:
        sel = [int(np.argmin(np.abs(lam - v))) for v in lambda_values]

    n = len(sel)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(TWO_COLUMN_WIDTH, 1.9 * n_rows + 0.6),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, li in zip(axes.ravel(), sel):
        s = sv[li, k_idx, :]
        s = s[np.isfinite(s)]
        if s.size == 0 or s[0] == 0:
            ax.set_visible(False)
            continue
        idx = np.arange(1, s.size + 1)
        ax.semilogy(idx, s / s[0], "o-", color=AZURE_BLUE,
                    markersize=MARKER_SIZE * 2.5, lw=LINE_WIDTH)
        if min_order:
            ax.axvline(min_order, color="0.6", lw=LINE_WIDTH * 0.7, ls=(0, (2, 2)))
        if max_order:
            ax.axvline(max_order, color=PURPLE_MAGENTA, lw=LINE_WIDTH * 0.8,
                       ls=(0, (4, 2)))
        chosen = int(result.pole_order[li, k_idx])
        ax.set_title(f"$\\lambda$ = {lam[li]:.2f}   (order={chosen})", fontsize=8)
        ax.set_xlabel("singular-value index", fontsize=8)
        ax.set_ylabel("$S_i / S_0$", fontsize=8)
        ax.set_xticks(idx)
        make_ax_lines_wider(ax, labelsize=8)
    fig.suptitle(
        f"Hankel SVD spectra at k = {k_arr[k_idx]:.2f} rad  "
        f"(purple = max_order, grey = min_order)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ===========================================================================
# SVD magnitudes vs λ  (static panel grid + HTML slider over k)
# ===========================================================================
#
# These are the *transpose* of ``plot_svd_vs_k_slider`` / the spectrum grid:
# the horizontal axis is λ and the k role becomes either one-panel-per-k
# (static) or a slider (HTML).  Useful for reading how each Hankel singular
# value evolves with λ at a fixed quasi-momentum.

def _resolve_k_indices(
    result: KSpacePolesResult,
    k_values: list[float] | int | None,
    *,
    default_n_panels: int = 6,
) -> list[int]:
    """Translate a user-facing ``k_values`` arg into a sorted list of k indices.

    Mirrors ``_resolve_lambda_indices`` for the k axis:
    - ``None``: up to ``default_n_panels`` evenly-spaced k indices.
    - ``int``: that many evenly-spaced k indices.
    - ``list[float]``: nearest-neighbour lookup for each k value.
    """
    if k_values is None:
        n_panels = min(default_n_panels, result.n_k)
        idx = list(np.round(np.linspace(0, result.n_k - 1, n_panels)).astype(int))
    elif isinstance(k_values, int):
        n_panels = min(k_values, result.n_k)
        idx = list(np.round(np.linspace(0, result.n_k - 1, n_panels)).astype(int))
    else:
        idx = [result.nearest_k_idx(kv) for kv in k_values]
    return sorted(set(idx))


def plot_svd_vs_lambda_grid(
    result: KSpacePolesResult,
    *,
    k_values: list[float] | int | None = None,
    n_cols: int = 3,
    n_singvals: int = 5,
    normalize: bool = False,
) -> plt.Figure:
    """Static panel grid — singular-value magnitudes vs λ, one panel per k.

    The static companion to ``export_svd_vs_lambda_slider_html``.  Each panel
    fixes one k and plots the leading ``n_singvals`` Hankel singular values
    ``S[0..]`` as a function of λ on a log y-axis.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()`` (needs
        ``svd_singvals``).
    k_values:
        Which k values to show as panels.  Same conventions as
        ``plot_poles_vs_k_grid``'s ``lambda_values`` but for the k axis
        (``None`` → up to 6 evenly-spaced k; ``int`` → that many; list →
        nearest-neighbour).
    n_cols:
        Number of panel columns.
    n_singvals:
        How many leading singular values to draw per panel.
    normalize:
        If True, plot ``S_i / S_0`` (spectrum shape, comparable across k);
        default plots the raw singular-value magnitudes.

    Returns
    -------
    plt.Figure
    """
    sv = result.svd_singvals                       # (n_lambda, n_k, n_sv)
    if sv.ndim != 3:
        raise ValueError(f"svd_singvals has shape {sv.shape}; expected 3-D.")

    lam = result.lambda_values
    k = result.k_array
    n_show = min(n_singvals, sv.shape[2])

    k_idx = _resolve_k_indices(result, k_values)
    n_panels = len(k_idx)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False,
    )
    cmap = plt.get_cmap("viridis")

    for panel_idx, ki in enumerate(k_idx):
        ax = axes.ravel()[panel_idx]
        spec = sv[:, ki, :]                         # (n_lambda, n_sv)
        if normalize:
            s0 = np.where(spec[:, [0]] > 1e-300, spec[:, [0]], np.nan)
            spec = spec / s0
        for i in range(n_show):
            color = cmap(i / max(n_show - 1, 1))
            ax.semilogy(lam, spec[:, i], "o-", color=color, ms=3, lw=1,
                        label=f"S[{i}]")
        ax.set_title(f"k = {k[ki]:.3f} rad", fontsize=9)
        ax.set_xlabel("λ", fontsize=8)
        ax.set_ylabel("$S_i / S_0$" if normalize else "singular value", fontsize=8)
        ax.grid(True, alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=7, ncol=n_show)

    for panel_idx in range(n_panels, n_rows * n_cols):
        axes.ravel()[panel_idx].set_visible(False)

    title = "Hankel SVD magnitudes vs λ"
    if normalize:
        title += "  (normalized $S_i/S_0$)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def export_svd_vs_lambda_slider_html(
    result: KSpacePolesResult,
    path: str,
    *,
    n_singvals: int = 5,
    normalize: bool = False,
    include_plotlyjs: str = "cdn",
) -> str:
    """Standalone HTML — singular-value magnitudes vs λ, with a **k slider**.

    The transpose of ``plot_svd_vs_k_slider``: the horizontal axis is λ and
    the slider sweeps k.  Each slider step shows the leading ``n_singvals``
    Hankel singular values ``S[0..]`` as a function of λ at one fixed k, on a
    log y-axis.  The result is a single self-contained ``.html`` file that
    opens in any browser — no Python or server required.

    Parameters
    ----------
    result:
        ``KSpacePolesResult`` from ``KSpaceAnalysis.run()`` (needs
        ``svd_singvals``).
    path:
        Destination path (``.html``).  Parent directories are created.
    n_singvals:
        How many leading singular values to draw per k slice.
    normalize:
        If True, plot ``S_i / S_0`` (spectrum shape); default plots the raw
        singular-value magnitudes.
    include_plotlyjs:
        Forwarded to ``plotly.io.write_html`` (``"cdn"`` for a tiny file,
        ``True`` to embed plotly.js for full offline use).

    Returns
    -------
    str
        Path to the written HTML file.
    """
    import plotly.graph_objects as go
    from pathlib import Path
    from mpm_analysis.utils.windows_paths import win_long_path

    sv = result.svd_singvals                       # (n_lambda, n_k, n_sv)
    if sv.ndim != 3:
        raise ValueError(f"svd_singvals has shape {sv.shape}; expected 3-D.")

    lam = result.lambda_values
    k = result.k_array
    n_show = min(n_singvals, sv.shape[2])

    # viridis colours as rgb strings, matching the matplotlib SVD slider
    cmap = plt.get_cmap("viridis")
    colors = [
        "rgb({:.0f},{:.0f},{:.0f})".format(
            *(np.array(cmap(i / max(n_show - 1, 1))[:3]) * 255)
        )
        for i in range(n_show)
    ]

    fig = go.Figure()
    for ki in range(len(k)):
        spec = sv[:, ki, :]                         # (n_lambda, n_sv)
        if normalize:
            s0 = np.where(spec[:, [0]] > 1e-300, spec[:, [0]], np.nan)
            spec = spec / s0
        for i in range(n_show):
            fig.add_trace(go.Scatter(
                x=lam, y=spec[:, i], mode="lines+markers",
                name=f"S[{i}]",
                line={"color": colors[i], "width": 2},
                marker={"color": colors[i], "size": 4},
                visible=(ki == 0),
                hovertemplate=(
                    f"λ=%{{x:.3f}}<br>S[{i}]=%{{y:.4g}}"
                    f"<extra>k={k[ki]:.3f} rad</extra>"
                ),
            ))

    steps = []
    for ki in range(len(k)):
        vis = [False] * (len(k) * n_show)
        for i in range(n_show):
            vis[ki * n_show + i] = True
        steps.append(dict(method="update", label=f"{k[ki]:.3f}",
                          args=[{"visible": vis}]))

    ylabel = "S_i / S_0" if normalize else "singular value magnitude"
    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "k = ", "suffix": " rad"},
                      steps=steps)],
        xaxis_title="λ", yaxis_title=ylabel, yaxis_type="log",
        title="Hankel SVD magnitudes vs λ   (slider: k)",
    )

    import os as _os
    p = Path(path)
    _os.makedirs(win_long_path(p.parent), exist_ok=True)
    fig.write_html(win_long_path(p), include_plotlyjs=include_plotlyjs)
    return str(p)


# ===========================================================================
# Raw k-space time traces over λ  (static + HTML slider)
# ===========================================================================

def plot_raw_kspace_traces(
    records: list[KSpaceRecord],
    *,
    k_idx_or_val=np.pi,
    component: str = "real",
    ax: plt.Axes | None = None,
    colormap: str = "viridis",
) -> plt.Figure:
    """Raw k-space time traces at a fixed k, one curve per λ (colour-coded).

    ``component``: "real" or "imag" part of the complex P(k,t) signal.
    """
    recs = sorted(records, key=lambda r: r.lambda_val)
    lam = np.array([r.lambda_val for r in recs])
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.get_figure()
    cmap = plt.get_cmap(colormap)
    norm = (lam - lam.min()) / (np.ptp(lam) or 1.0)
    k_disp = None
    for r, c in zip(recs, norm):
        ki = r.nearest_k_idx(float(k_idx_or_val)) if not isinstance(
            k_idx_or_val, (int, np.integer)) else int(k_idx_or_val)
        k_disp = r.k_array[ki]
        sig = r.signal[ki, :]
        y = sig.real if component == "real" else sig.imag
        ax.plot(r.t, y, color=cmap(c), lw=1.0)
    ax.set_xlabel("time  (µs)")
    ax.set_ylabel(f"{component}  P(k, t)")
    ax.set_title(f"Raw k-space traces over λ   |   k ≈ {k_disp:.3f} rad")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lam.min(), lam.max()))
    fig.colorbar(sm, ax=ax, label="λ")
    fig.tight_layout()
    return fig


def export_raw_kspace_traces_html(
    records: list[KSpaceRecord],
    path: str,
    *,
    k_idx_or_val=np.pi,
    include_plotlyjs: str = "cdn",
) -> str:
    """Standalone HTML with a **λ slider** over raw k-space time traces.

    Each slider step shows Re and Im of P(k,t) vs time at a fixed k for one λ.
    """
    import plotly.graph_objects as go
    from pathlib import Path
    from mpm_analysis.utils.windows_paths import win_long_path

    recs = sorted(records, key=lambda r: r.lambda_val)
    fig = go.Figure()
    k_disp = None
    for i, r in enumerate(recs):
        ki = r.nearest_k_idx(float(k_idx_or_val)) if not isinstance(
            k_idx_or_val, (int, np.integer)) else int(k_idx_or_val)
        k_disp = r.k_array[ki]
        sig = r.signal[ki, :]
        fig.add_trace(go.Scatter(x=r.t, y=sig.real, name="Re", visible=(i == 0),
                                 line=dict(color="#3498DB")))
        fig.add_trace(go.Scatter(x=r.t, y=sig.imag, name="Im", visible=(i == 0),
                                 line=dict(color="#A13D97")))
    steps = []
    for i, r in enumerate(recs):
        vis = [False] * (2 * len(recs))
        vis[2 * i] = True
        vis[2 * i + 1] = True
        steps.append(dict(method="update", label=f"{r.lambda_val:.3f}",
                          args=[{"visible": vis}]))
    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "λ = "}, steps=steps)],
        xaxis_title="time (µs)", yaxis_title="P(k, t)",
        title=f"Raw k-space traces vs λ   |   k ≈ {k_disp:.3f} rad",
    )
    import os as _os
    p = Path(path)
    _os.makedirs(win_long_path(p.parent), exist_ok=True)
    fig.write_html(win_long_path(p), include_plotlyjs=include_plotlyjs)
    return str(p)


# ===========================================================================
# Eigenvalues vs λ with a k-slider  (double panel Re/Im, exp vs theory) — HTML
# ===========================================================================

def export_eigenvalues_vs_lambda_kslider_html(
    result: KSpacePolesResult,
    path: str,
    *,
    theory_poles: np.ndarray | None = None,
    k_stride: int = 1,
    continuity_sort: bool = True,
    deriv_weight: float = 0.1,
    include_plotlyjs: str = "cdn",
) -> str:
    """Interactive HTML: decay rate −Re(λ-pole) (top) and frequency Im (bottom)
    vs λ, with a **slider over k**.  Experiment solid, theory dashed (if given).

    This is the tool for locating exceptional points and comparing the transition
    λ (and k) to theory: slide k to π and watch two eigenvalue branches collide
    (decay split opens / frequencies merge).  ``theory_poles`` (shape
    ``(n_λ, n_k, M)``, from ``analysis.braiding.theory_poles_on_grid`` using your
    imperfection parameters) is overlaid for direct comparison.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pathlib import Path
    from mpm_analysis.utils.windows_paths import win_long_path

    lam = result.lambda_values
    k = result.k_array
    order = result.max_order
    k_idx = list(range(0, result.n_k, max(1, k_stride)))
    # Same track colours as every other figure: _POLE_COLORS_HEX on the
    # continuity-sorted, λ-matched tracks.
    colors = _POLE_COLORS_HEX
    poles = (_sorted_poles_per_lambda(result, deriv_weight=deriv_weight)
             if continuity_sort else np.asarray(result.poles, dtype=complex))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("decay rate  −Re(pole)  [1/µs]",
                                        "frequency  Im(pole)  [1/µs]"))
    groups = []                                   # trace indices per k step
    t0 = 0
    for ki in k_idx:
        idxs = []
        dec = -poles[:, ki, :].real
        frq = poles[:, ki, :].imag
        for pj in range(order):
            c = colors[pj % len(colors)]
            fig.add_trace(go.Scatter(x=lam, y=dec[:, pj], mode="lines+markers",
                          line=dict(color=c), name=f"track {pj}", visible=False,
                          legendgroup=f"exp{pj}"), row=1, col=1); idxs.append(t0); t0 += 1
            fig.add_trace(go.Scatter(x=lam, y=frq[:, pj], mode="lines+markers",
                          line=dict(color=c), name=f"track {pj}", visible=False,
                          showlegend=False, legendgroup=f"exp{pj}"), row=2, col=1); idxs.append(t0); t0 += 1
        if theory_poles is not None and ki < theory_poles.shape[1]:
            tdec = -theory_poles[:, ki, :].real
            tfrq = theory_poles[:, ki, :].imag
            for pj in range(theory_poles.shape[2]):
                fig.add_trace(go.Scatter(x=lam, y=tdec[:, pj], mode="lines",
                              line=dict(color="black", dash="dash", width=1),
                              name="theory" if pj == 0 else None,
                              showlegend=(pj == 0), visible=False,
                              legendgroup="theory"), row=1, col=1); idxs.append(t0); t0 += 1
                fig.add_trace(go.Scatter(x=lam, y=tfrq[:, pj], mode="lines",
                              line=dict(color="black", dash="dash", width=1),
                              showlegend=False, visible=False,
                              legendgroup="theory"), row=2, col=1); idxs.append(t0); t0 += 1
        groups.append(idxs)

    for ti in groups[0]:                          # show first k
        fig.data[ti].visible = True
    steps = []
    for j, ki in enumerate(k_idx):
        vis = [False] * len(fig.data)
        for ti in groups[j]:
            vis[ti] = True
        steps.append(dict(method="update", label=f"{k[ki]:.2f}",
                          args=[{"visible": vis},
                                {"title": f"Eigenvalues vs λ   |   k = {k[ki]:.3f} rad"}]))
    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "k = "}, steps=steps)],
        title=f"Eigenvalues vs λ   |   k = {k[k_idx[0]]:.3f} rad",
        height=720,
    )
    fig.update_xaxes(title_text="λ", row=2, col=1)
    import os as _os
    p = Path(path)
    _os.makedirs(win_long_path(p.parent), exist_ok=True)
    fig.write_html(win_long_path(p), include_plotlyjs=include_plotlyjs)
    return str(p)


# ===========================================================================
# Braiding ν vs λ  (experiment, optionally with theory)
# ===========================================================================

def _braid_index_keys(*braidings: dict | None, field: str = "raw_pairs") -> list[str]:
    keys: set[str] = set()
    for b in braidings:
        if b is None:
            continue
        for d in b.get(field, []) or []:
            keys.update(str(k) for k in d)
    return sorted(keys)


def _braid_index_values(braiding: dict, key: str, field: str = "raw_pairs") -> np.ndarray:
    vals = []
    for d in braiding.get(field, []) or []:
        vals.append(float(d.get(key, np.nan)))
    return np.asarray(vals, dtype=float)


def plot_braiding_vs_lambda(
    braiding_exp: dict,
    *,
    braiding_theory: dict | None = None,
    primary_label: str = "experiment",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Braiding characterization vs λ: ν_total (top) and the EP location in k
    (bottom), experiment vs optional theory, with the ν-transition λ marked.

    ``braiding_*`` are dicts from :mod:`mpm_analysis.analysis.braiding`
    (``lambda_values``, ``nu_total``, ``ep_k``, ``links``, ``braid_words``).
    Transition λ (where ν changes) are the exceptional-point crossings.
    """
    from mpm_analysis.analysis.braiding import braiding_transitions
    if ax is not None:                       # single-axes fallback (ν only)
        fig = ax.get_figure(); ax_nu = ax; ax_pair = ax_k = None
    else:
        fig, (ax_nu, ax_pair, ax_k) = plt.subplots(
            3, 1, figsize=(8.5, 8.0), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.15, 1.0]},
        )

    lam = braiding_exp["lambda_values"]
    ax_nu.plot(lam, braiding_exp["nu_total"], "o-", color="#A13D97",
               label=f"total ({primary_label})")
    if braiding_theory is not None:
        ax_nu.plot(braiding_theory["lambda_values"], braiding_theory["nu_total"],
                   "s--", color="0.4", label="total (theory)")
    # mark ν transitions (exceptional points in λ)
    tr = braiding_transitions(braiding_exp)
    for lm in tr["lambda_mid"]:
        ax_nu.axvline(lm, color="#d02500", lw=1.0, ls=(0, (3, 2)), alpha=0.8)
        if ax_pair is not None:
            ax_pair.axvline(lm, color="#d02500", lw=1.0, ls=(0, (3, 2)), alpha=0.8)
        if ax_k is not None:
            ax_k.axvline(lm, color="#d02500", lw=1.0, ls=(0, (3, 2)), alpha=0.8)
    ax_nu.axhline(0, color="0.7", lw=0.6)
    ax_nu.set_ylabel(r"braid index  $\nu_{\mathrm{total}}$")
    ax_nu.set_title("Braiding vs λ  (red dashed = ν transition / exceptional point)")
    ax_nu.legend(fontsize=8)
    ax_nu.grid(True, alpha=0.25)

    if ax_pair is not None:
        keys = _braid_index_keys(braiding_exp, braiding_theory, field="raw_pairs")
        cmap = plt.get_cmap("tab10")
        for i, key in enumerate(keys):
            color = cmap(i % 10)
            vals = _braid_index_values(braiding_exp, key, field="raw_pairs")
            if np.any(np.isfinite(vals)):
                ax_pair.plot(lam, vals, "o-", color=color, lw=1.3, ms=4,
                             label=f"{key} ({primary_label})")
            if braiding_theory is not None:
                tvals = _braid_index_values(braiding_theory, key, field="raw_pairs")
                tlam = braiding_theory["lambda_values"]
                if np.any(np.isfinite(tvals)):
                    ax_pair.plot(tlam, tvals, "s--", color=color, lw=1.1, ms=3.5,
                                 alpha=0.75, label=f"{key} (theory)")
        ax_pair.axhline(0, color="0.7", lw=0.6)
        ax_pair.set_ylabel("pair indices")
        ax_pair.grid(True, alpha=0.25)
        ax_pair.legend(fontsize=7, ncol=2, loc="best")

    if ax_k is not None:
        ax_k.plot(lam, braiding_exp.get("ep_k", np.full_like(lam, np.nan)), "o-",
                  color="#A13D97", label=primary_label)
        if braiding_theory is not None and "ep_k" in braiding_theory:
            ax_k.plot(braiding_theory["lambda_values"], braiding_theory["ep_k"],
                      "s--", color="0.4", label="theory")
        ax_k.axhline(np.pi, color="0.7", lw=0.6, ls=":")
        ax_k.set_ylabel("EP location  k  (rad)")
        ax_k.set_xlabel("λ")
        ax_k.grid(True, alpha=0.25)
        ax_k.legend(fontsize=8)
    else:
        ax_nu.set_xlabel("λ")
    fig.tight_layout()
    return fig


# ===========================================================================
# Unified figure saver
# ===========================================================================

def save_kspace_figures(
    result: KSpacePolesResult,
    out_dir,
    *,
    records: list[KSpaceRecord] | None = None,
    source_tag: str = "kspace",
    scan_number: int | None = None,
    grid_n_panels: int = 16,
    grid_n_cols: int = 4,
    raw_component: str = "real",
    include_3d: bool = True,
    include_html: bool = True,
    include_svd: bool = True,
    include_braid_topology: bool = True,
    theory_poles: np.ndarray | None = None,
    braiding: dict | None = None,
    braiding_theory: dict | None = None,
    braiding_label: str = "experiment",
    dpi: int = 180,
):
    """Build and save the full k-space figure set (timestamped, non-clobbering).

    Produces: eigenvalues-vs-λ at k≈π (with theory overlay + error bars when
    available), poles-vs-k decay/freq grids, parametric grid, SVD spectrum, raw
    k-space traces (static + HTML slider, if ``records``), 3-D grid + HTML,
    parametric 3-D + HTML, a braiding-ν panel (if ``braiding``), and the 4-panel
    braid topology figure + λ-slider HTML (``include_braid_topology``).  Returns
    the list of written paths.  Long-path-safe.
    """
    from pathlib import Path as _Path
    from mpm_analysis.io.file_naming import build_figure_filename
    from mpm_analysis.io.experimental.compressed_click_record import _unique_path
    from mpm_analysis.utils.windows_paths import win_long_path

    out_dir = _Path(out_dir)
    import os as _os
    _os.makedirs(win_long_path(out_dir), exist_ok=True)
    saved = []

    def _save(fig, tag):
        stem = build_figure_filename(tag, source_tag, scan_number=scan_number,
                                     extension="png")
        path = _unique_path(out_dir / stem)
        fig.savefig(win_long_path(path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[figure] saved {path.name}")
        saved.append(path)

    def _save_html(html_fn, tag):
        stem = build_figure_filename(tag, source_tag, scan_number=scan_number,
                                     extension="html")
        path = _unique_path(out_dir / stem)
        # html_fn may return a single path or a list of paths (3D grids are
        # paginated across several files to stay within the browser's WebGL
        # limit).  Record whatever was actually written.
        result_paths = html_fn(str(path))
        if isinstance(result_paths, (list, tuple)):
            for p in result_paths:
                p = _Path(p)
                print(f"[figure] saved {p.name}")
                saved.append(p)
        else:
            print(f"[figure] saved {path.name}")
            saved.append(path)

    k_pi = result.nearest_k_idx(np.pi)

    # k-slider double panel (decay + freq) vs λ, exp vs theory — the EP-locator
    if include_html:
        _save_html(lambda p: export_eigenvalues_vs_lambda_kslider_html(
            result, p, theory_poles=theory_poles),
            "kspace_eigenvalues_vs_lambda_kslider")

    # eigenvalues vs λ (+ theory overlay)
    fig_eig = plot_eigenvalues_vs_lambda(result, k_pi)
    if theory_poles is not None:
        ax_d, ax_f = fig_eig.axes[0], fig_eig.axes[1]
        tp = theory_poles[:, k_pi, :]
        for j in range(tp.shape[1]):
            lbl = "theory" if j == 0 else None
            ax_d.plot(result.lambda_values, -tp[:, j].real, "--", color="0.35",
                      lw=1.0, alpha=0.8, label=lbl)
            ax_f.plot(result.lambda_values, tp[:, j].imag, "--", color="0.35",
                      lw=1.0, alpha=0.8)
        ax_d.legend(fontsize=8)
    _save(fig_eig, "kspace_eigenvalues_vs_lambda")

    _save(plot_poles_vs_k_grid(result, lambda_values=grid_n_panels,
                               n_cols=grid_n_cols, component="decay"),
          "kspace_poles_vs_k_decay")
    _save(plot_poles_vs_k_grid(result, lambda_values=grid_n_panels,
                               n_cols=grid_n_cols, component="frequency"),
          "kspace_poles_vs_k_frequency")
    _save(plot_eigenvalues_parametric_grid(result, lambda_values=grid_n_panels,
                                           n_cols=grid_n_cols),
          "kspace_eigenvalues_parametric_grid")
    if include_html:
        _save_html(lambda p: export_eigenvalues_parametric_slider_html(result, p),
                   "kspace_eigenvalues_parametric_lambdaslider")
        _save_html(lambda p: export_pole_pair_topology_html(result, p),
                   "kspace_pair_gap_topology")
    if include_svd:
        _save(plot_svd_spectrum_grid(result), "kspace_svd_spectrum")
        _save(plot_svd_vs_lambda_grid(result, k_values=grid_n_panels,
                                      n_cols=grid_n_cols),
              "kspace_svd_vs_lambda_grid")
        if include_html:
            _save_html(lambda p: export_svd_vs_lambda_slider_html(result, p),
                       "kspace_svd_vs_lambda_slider")

    if records is not None:
        _save(plot_raw_kspace_traces(records, component=raw_component),
              "kspace_raw_traces")
        if include_html:
            _save_html(lambda p: export_raw_kspace_traces_html(records, p),
                       "kspace_raw_traces_slider")

    if include_3d:
        _save(plot_poles_vs_k_3d_grid(result, lambda_values=grid_n_panels,
                                      n_cols=grid_n_cols), "kspace_poles_vs_k_3d")
        _save(plot_eigenvalues_parametric_3d(result), "kspace_parametric_3d")
        if include_html:
            _save_html(lambda p: export_poles_vs_k_3d_html(
                result, p, lambda_values=grid_n_panels, n_cols=grid_n_cols),
                "kspace_poles_vs_k_3d_slider")
            _save_html(lambda p: export_eigenvalues_parametric_3d_html(result, p),
                       "kspace_parametric_3d_slider")

    if braiding is not None:
        _save(plot_braiding_vs_lambda(braiding, braiding_theory=braiding_theory,
                                      primary_label=braiding_label),
              "kspace_braiding_vs_lambda")

    # Paper-convention braid topology: the loop is closed and unrolled to the
    # full 4π/6π period, so the indices are quantized and the braid can actually
    # be seen closing.  Runs independently of the ``braiding`` report above.
    if include_braid_topology:
        from mpm_analysis.analysis.braid_topology import analyze_braid_scan
        from mpm_analysis.plotting.paper_figures.braid_topology_main import (
            make_braid_topology_figure,
        )
        try:
            scan = analyze_braid_scan(result)
        except Exception as exc:                       # never lose the rest
            print(f"[figure] braid topology skipped: {exc}")
        else:
            _save(make_braid_topology_figure(scan,
                                             scan.most_interesting_lambda()),
                  "kspace_braid_topology")
            if include_html:
                from mpm_analysis.kspace.braid_html import (
                    export_braid_topology_html,
                )
                _save_html(
                    lambda p: export_braid_topology_html(result, p, scan=scan),
                    "kspace_braid_topology_slider")

    return saved
