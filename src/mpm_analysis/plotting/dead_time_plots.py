"""
dead_time_plots.py
------------------
Figures for the detector dead-time compression sweep
(:class:`mpm_analysis.pipeline.dead_time_pipeline.DeadTimeSweepPipeline`):

* :func:`plot_compression_traces`         — raw vs compressed time-axis data
  with the direct fit for representative ``c`` (static).
* :func:`plot_traces_compression_slider`  — the same rows with a slider over
  ``c`` (one ``c`` at a time, so nothing overlaps).
* :func:`plot_poles_at_compression`       — the standard two-panel pole
  spectrum (all decay rates / all frequencies vs λ) at one ``c`` — the same
  figure ``run.py`` shows after an MPM run — with the theory eigenvalues
  overlaid when ``theory_params`` is given.
* :func:`plot_poles_compression_slider`   — the same two panels with a slider
  over ``c``.
* :func:`plot_transition_vs_compression`  — ``Delta(λ)`` for representative
  ``c`` and the exceptional point vs ``c``, with the theory prediction.

Theory overlay
~~~~~~~~~~~~~~
``theory_params = dict(tau_B=…, T1=…, T2=…[, Omega=…, k=0.0])`` selects the
4×4 Liouvillian of :mod:`mpm_analysis.simulation.liouvillian_spectrum` at
``k = 0`` (the ensemble-average generator).  Compressing away a fraction ``c``
of the bright dwell time maps to ``tau_B_eff = (1 - c) * tau_B``
(:func:`~mpm_analysis.simulation.liouvillian_spectrum.tau_B_effective`), so
the theory line for the transition is
:func:`~mpm_analysis.simulation.liouvillian_spectrum.theory_transition` at
that dwell time.  ``Omega`` defaults to the sweep's ``omega_DG``.

The sweep result stores poles only; amplitudes and fit curves are re-solved
here from the records (a linear least-squares given the poles), so the
traces figures need the same records the sweep was run on.
"""
from __future__ import annotations

import numpy as np

from mpm_analysis.analysis.dead_time import (
    DEFAULT_CLICK_KEY, compress_records, resample_uniform,
)

# Okabe-Ito: colour-blind safe and high contrast on white; one colour per c.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
RAW_COLOR = "0.35"
THEORY_COLOR = "0.15"
FONT = 12
LEGEND_FONT = 10


def _colors(n: int) -> list[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _representative(sweep, c_values):
    """``(indices, coefficients)`` of the swept ``c`` closest to ``c_values``."""
    idx = sorted({sweep.index_of(c) for c in c_values})
    return idx, [float(sweep.compress_coeffs[i]) for i in idx]


def _robust_ylim(a: np.ndarray, k: float = 3.0):
    """Axis limits from the bulk of ``a`` — the data range clipped to
    ``[Q1 - k*IQR, Q3 + k*IQR]`` — so a few runaway poles at large λ cannot
    squash every other point onto the axis."""
    finite = a[np.isfinite(a)]
    if not finite.size:
        return None
    q1, q3 = np.percentile(finite, [25, 75])
    iqr = (q3 - q1) or (abs(q3) or 1.0)
    lo = max(finite.min(), q1 - k * iqr)
    hi = min(finite.max(), q3 + k * iqr)
    pad = 0.08 * ((hi - lo) or 1.0)
    return lo - pad, hi + pad


def _fit_curve(t, y, poles, n_dense: int = 400):
    """Dense modal reconstruction through ``(t, y)`` with the given poles."""
    from mpm_analysis.analysis.matrix_pencil import (
        reconstruct_signal, solve_linear_amplitudes,
    )
    poles = np.asarray(poles, dtype=complex)
    if not np.all(np.isfinite(poles)):
        return None, None
    t_dense = np.linspace(float(t[0]), float(t[-1]), n_dense)
    amps = solve_linear_amplitudes(t, y, poles)
    return t_dense, reconstruct_signal(t_dense, poles, amps)


def _stack(by_key, keys):
    """``(t, Y)`` of one λ as the (joint) fit saw it: ``(N,)`` or ``(k, N)``."""
    t = by_key[keys[0]].t
    if len(keys) == 1:
        return t, by_key[keys[0]].signal
    return t, np.vstack([by_key[k].signal for k in keys])


def _row(y, keys, key):
    """Pick the observable's row out of a (k, N) reconstruction."""
    return y if np.ndim(y) == 1 else y[keys.index(key)]


def _target_lambda_indices(sweep, target_lambdas):
    lambdas = sweep.lambda_values
    if target_lambdas is None:
        target_lambdas = lambdas[np.round(np.linspace(0, len(lambdas) - 1, 3)).astype(int)]
    return [int(np.argmin(np.abs(lambdas - l))) for l in target_lambdas]


def _slice_at_c(records, sweep, ic, il, key, *, click_key, show_fallback):
    """Everything one traces panel needs for λ index ``il`` at ``c`` index ``ic``.

    Returns ``dict(t, y, fit=(t_dense, y_dense) | None, regrid=(t_u, y_u,
    t_f, y_f) | None)`` — the remapped points, the direct-fit curve and,
    optionally, the interpolated points with the regrid + MPM curve.
    """
    keys = list(sweep.parameters["observable_keys"])
    lam = float(sweep.lambda_values[il])
    c = float(sweep.compress_coeffs[ic])
    by_key = {r.observable_key: r for r in compress_records(
        records, c, click_key=click_key, keys=keys,
        remap=sweep.parameters.get("remap", "pointwise"), on_fold="truncate")
        if r.lambda_val == lam}
    rec = by_key[key]
    t_d, y_d = _fit_curve(*_stack(by_key, keys), sweep.direct.poles[ic, il])
    out = dict(t=rec.t, y=rec.signal,
               fit=None if t_d is None else (t_d, _row(y_d, keys, key)),
               regrid=None)
    dt = sweep.parameters.get("regrid_dt")
    if show_fallback and sweep.fallback is not None and dt:
        regrid = {k: resample_uniform(r, dt) for k, r in by_key.items()}
        t_f, y_f = _fit_curve(*_stack(regrid, keys), sweep.fallback.poles[ic, il])
        if t_f is not None:
            out["regrid"] = (regrid[key].t, regrid[key].signal, t_f, _row(y_f, keys, key))
    return out


def _map_label(sweep) -> str:
    return (r"$t' = t\,[1 - c\,p_{\rm click}(t)]$"
            if sweep.parameters.get("remap", "pointwise") == "pointwise"
            else r"$t' = t - c\int_0^t p_{\rm click}$")


def _traces_axes(records, sweep, lam_idx, key, click_key):
    """Figure with one (data | remap) row per λ; raw data and p_click drawn.

    Returns ``(fig, axes, raw_records)``.
    """
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import apply_paper_style, make_ax_lines_wider

    apply_paper_style()
    fig, axes = plt.subplots(
        len(lam_idx), 2, figsize=(11.0, 3.2 * len(lam_idx)),
        gridspec_kw=dict(width_ratios=(2.2, 1.0)), layout="constrained",
        squeeze=False,
    )
    clicks = {r.lambda_val: r for r in records if r.observable_key == click_key}
    raws = []
    for il, (ax, ax_map) in zip(lam_idx, axes):
        lam = float(sweep.lambda_values[il])
        raw = next(r for r in records if r.observable_key == key and r.lambda_val == lam)
        raws.append(raw)
        ax.plot(raw.t, raw.signal, "o", color=RAW_COLOR, ms=3, zorder=5,
                label=r"raw data ($c=0$)")
        ax_map.plot(raw.t, raw.t, ":", color=RAW_COLOR, lw=1.0, label=r"$t' = t$")
        ax_p = ax_map.twinx()
        ax_p.plot(clicks[lam].t, clicks[lam].signal, "-", color="0.6", lw=1.2)
        ax_p.set_ylabel(r"$p_{\rm click}$", fontsize=FONT, color="0.4")
        ax_p.tick_params(axis="y", colors="0.4", labelsize=FONT - 1)
        ax_p.set_ylim(0, 1)
        ax.set_ylabel(f"{key}   " + rf"$\lambda={lam:.2f}$", fontsize=FONT)
        ax.set_xlabel(r"time $t'$ ($\mu$s)", fontsize=FONT)
        ax_map.set_xlabel(r"$t$ ($\mu$s)", fontsize=FONT)
        ax_map.set_ylabel(_map_label(sweep), fontsize=FONT)
        for a in (ax, ax_map):
            make_ax_lines_wider(a, labelsize=FONT - 1)
    return fig, axes, raws


# ---------------------------------------------------------------------------
# 1. traces + fits — static (representative c) and slider
# ---------------------------------------------------------------------------

def plot_compression_traces(
    records,
    sweep,
    *,
    c_values=(0.0, 0.5, 1.0),
    target_lambdas=None,
    click_key: str = DEFAULT_CLICK_KEY,
    show_fallback: bool = False,
    title: str | None = None,
):
    """Raw vs compressed traces with the direct fit, one row per λ.

    Left column: the measured points on the raw axis (grey) and on the
    compressed axis for each ``c`` in ``c_values`` (one colour per ``c``)
    with the direct fit as a solid line; ``show_fallback`` adds the regrid +
    MPM fit (dashed) through the interpolated points (crosses).  Right
    column: the axis remap ``t'(t)`` of that λ and its click probability.
    For many ``c`` this gets crowded — use
    :func:`plot_traces_compression_slider` instead.

    Parameters
    ----------
    records:
        The records the sweep was run on (observables + click probability).
    sweep:
        ``DeadTimeSweepResult``.
    c_values:
        Representative coefficients (each snapped to the nearest swept ``c``).
    target_lambdas:
        λ values to show; ``None`` → three evenly spaced over the sweep axis.
    """
    key = list(sweep.parameters["observable_keys"])[0]
    lam_idx = _target_lambda_indices(sweep, target_lambdas)
    c_idx, cs = _representative(sweep, c_values)
    colors = _colors(len(cs))
    fig, axes, raws = _traces_axes(records, sweep, lam_idx, key, click_key)

    for il, raw, (ax, ax_map) in zip(lam_idx, raws, axes):
        for ic, c, col in zip(c_idx, cs, colors):
            s = _slice_at_c(records, sweep, ic, il, key,
                            click_key=click_key, show_fallback=show_fallback)
            if c > 0:
                ax.plot(s["t"], s["y"], "o", color=col, ms=3, alpha=0.75, zorder=6)
            if s["fit"] is not None:
                ax.plot(*s["fit"], "-", color=col, lw=1.8, zorder=8,
                        label=rf"$c={c:.2f}$ direct fit")
            if s["regrid"] is not None:
                t_u, y_u, t_f, y_f = s["regrid"]
                ax.plot(t_u, y_u, "x", color=col, ms=4, alpha=0.5, zorder=6)
                ax.plot(t_f, y_f, "--", color=col, lw=1.4, alpha=0.9, zorder=7,
                        label=rf"$c={c:.2f}$ regrid + MPM")
            # A truncated (folded) trace keeps only the head of the raw axis.
            ax_map.plot(raw.t[:s["t"].size], s["t"], "-", color=col, lw=2.0,
                        label=rf"$c={c:.2f}$")
        ax_map.legend(fontsize=LEGEND_FONT, loc="upper left")
    axes[0, 0].legend(fontsize=LEGEND_FONT, loc="upper right")
    fig.suptitle(title or "Dead-time compression: data and fits", fontsize=FONT + 2)
    return fig


def plot_traces_compression_slider(
    records,
    sweep,
    *,
    target_lambdas=None,
    click_key: str = DEFAULT_CLICK_KEY,
    show_fallback: bool = False,
):
    """Traces figure with a slider over ``c``: one ``c`` at a time per row.

    Each row shows the raw data (grey, fixed), the data on the compressed
    axis and its direct fit at the slider's ``c`` (coloured), and the remap
    ``t'(t)`` on the right.  Keep a reference to the returned ``Slider``.
    """
    import matplotlib

    key = list(sweep.parameters["observable_keys"])[0]
    lam_idx = _target_lambda_indices(sweep, target_lambdas)
    cs = sweep.compress_coeffs
    col = PALETTE[1]
    fig, axes, raws = _traces_axes(records, sweep, lam_idx, key, click_key)
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.92))  # room for the slider

    # Precompute every (row, c) slice once; the slider only swaps line data.
    slices = [[_slice_at_c(records, sweep, ic, il, key, click_key=click_key,
                           show_fallback=show_fallback)
               for ic in range(len(cs))] for il in lam_idx]
    artists = []
    for (ax, ax_map), raw in zip(axes, raws):
        pts, = ax.plot([], [], "o", color=col, ms=3, alpha=0.75, zorder=6,
                       label="data on $t'$")
        fit, = ax.plot([], [], "-", color=col, lw=2.0, zorder=8, label="direct fit")
        reg_pts, = ax.plot([], [], "x", color=PALETTE[0], ms=4, alpha=0.5, zorder=6)
        reg_fit, = ax.plot([], [], "--", color=PALETTE[0], lw=1.4, zorder=7,
                           label="regrid + MPM" if show_fallback else None)
        mp, = ax_map.plot([], [], "-", color=col, lw=2.0, label="$t'(t)$")
        ax.set_xlim(raw.t[0] - 0.5, raw.t[-1] + 0.5)
        ax_map.set_ylim(raw.t[0] - 0.5, raw.t[-1] + 0.5)
        ax_map.legend(fontsize=LEGEND_FONT, loc="upper left")
        artists.append((pts, fit, reg_pts, reg_fit, mp))
    axes[0, 0].legend(fontsize=LEGEND_FONT, loc="upper right")

    def draw(ic: int) -> None:
        c = float(cs[ic])
        for row, raw, (pts, fit, reg_pts, reg_fit, mp) in zip(slices, raws, artists):
            s = row[ic]
            pts.set_data(s["t"], s["y"])
            fit.set_data(*(s["fit"] if s["fit"] is not None else ([], [])))
            if s["regrid"] is not None:
                t_u, y_u, t_f, y_f = s["regrid"]
                reg_pts.set_data(t_u, y_u)
                reg_fit.set_data(t_f, y_f)
            else:
                reg_pts.set_data([], [])
                reg_fit.set_data([], [])
            mp.set_data(raw.t[:s["t"].size], s["t"])
        fig.suptitle(rf"SLIDER  —  dead-time compression at $c = {c:.2f}$",
                     fontsize=FONT + 2)
        fig.canvas.draw_idle()

    slider = _slider_axis(fig, cs, [0.2, 0.025, 0.6, 0.03])
    slider.on_changed(lambda v: draw(sweep.index_of(v)))
    draw(0)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Dead-time sweep: SLIDER over c (traces)")
    _announce_slider(matplotlib.get_backend(), "traces")
    return slider


def _slider_axis(fig, cs, rect):
    """A ``Slider`` over the swept coefficients (needs at least two of them)."""
    from matplotlib.widgets import Slider

    if len(cs) < 2:
        raise ValueError("a slider over c needs at least two swept coefficients; "
                         f"this sweep has {list(cs)}.")
    ax_s = fig.add_axes(rect)
    slider = Slider(ax_s, "compress coeff $c$", float(cs.min()), float(cs.max()),
                    valinit=float(cs[0]), valstep=cs)
    slider.label.set_fontsize(FONT)
    slider.valtext.set_fontsize(FONT)
    return slider


def _announce_slider(backend: str, what: str) -> None:
    """Say where the slider is, or that this backend cannot host one."""
    b = backend.lower()
    static = "inline" in b or b in ("agg", "pdf", "pgf", "ps", "svg", "template", "cairo")
    if static:
        print(f"[dead_time slider/{what}] matplotlib backend '{backend}' is "
              "non-interactive: the slider figure is a static image. Run from a "
              "terminal (python run.py) or switch to a GUI backend, e.g. "
              "%matplotlib qt / tk in an interactive window.")
    else:
        print(f"[dead_time slider/{what}] window 'Dead-time sweep: SLIDER over c "
              f"({what})' — drag the bar at the bottom to step through c.")


# ---------------------------------------------------------------------------
# theory overlay helpers
# ---------------------------------------------------------------------------

def _theory(sweep, theory_params: dict | None):
    """Normalised theory parameters (``Omega`` defaults to the sweep's) or ``None``."""
    if theory_params is None:
        return None
    missing = [k for k in ("tau_B", "T1", "T2") if k not in theory_params]
    if missing:
        raise KeyError(f"theory_params is missing {missing}; need tau_B, T1, T2 "
                       "(Omega optional — defaults to the sweep's omega_DG).")
    return {"Omega": sweep.omega_s, "k": 0.0, **theory_params}


def _theory_spectrum(sweep, c: float, th: dict, lambda_max=None):
    """Theory eigenvalues ``(λ_grid, eigs (n_λ, 4), tau_B_eff)`` at ``c``."""
    from mpm_analysis.simulation.liouvillian_spectrum import (
        LiouvillianSpectrum, tau_B_effective,
    )
    lam = sweep.lambda_values
    hi = float(lam.max()) if lambda_max is None else min(float(lam.max()), lambda_max)
    spec = LiouvillianSpectrum(
        Omega=th["Omega"], T1=th["T1"], T2=th["T2"],
        tau_B=float(tau_B_effective(c, th["tau_B"])), k=th["k"],
        lambda_min=float(lam.min()), lambda_max=hi, n_lambda=400,
    )
    return spec.lambda_vals, spec.compute(), spec.tau_B


def _theory_ep(c, th: dict) -> float:
    from mpm_analysis.simulation.liouvillian_spectrum import theory_transition_vs_compression
    return float(theory_transition_vs_compression(
        [c], tau_B=th["tau_B"], T1=th["T1"], T2=th["T2"], Omega=th["Omega"], k=th["k"])[0])


# ---------------------------------------------------------------------------
# 2. pole spectrum at one c, and the slider over c
# ---------------------------------------------------------------------------

def plot_poles_at_compression(sweep, c: float, *, method: str = "direct",
                              lambda_max: float | None = None, axes=None,
                              theory_params: dict | None = None):
    """Standard two-panel pole spectrum (all poles per panel) at one ``c``.

    Reuses :func:`mpm_analysis.plotting.exploratory.plot_spectrum_panels`,
    so the figure matches the one ``run.py`` shows after an MPM run.
    Bootstrap error bars appear when the sweep ran one.  With
    ``theory_params`` the 4×4 Liouvillian eigenvalues at
    ``tau_B_eff = (1 - c) tau_B`` are overlaid as thin dark lines and both
    transitions — the data's ``λ_EP`` at this ``c`` and the theory's — are
    marked as vertical lines.

    Returns
    -------
    (fig, axes)
    """
    from mpm_analysis.plotting.exploratory import plot_spectrum_panels

    ic = sweep.index_of(c)
    res = sweep.to_analysis_result(ic, method, lambda_max=lambda_max)
    c_used = float(sweep.compress_coeffs[ic])
    label = {"direct": "direct fit on $t'$", "regrid_mpm": "regrid + MPM"}.get(method, method)
    fig, axes = plot_spectrum_panels(
        res, axes=axes,
        title=(("SLIDER  —  " if axes is not None else "")
               + rf"Poles at compress coeff $c = {c_used:.2f}$  ({label}, "
               rf"{sweep.parameters.get('remap', 'pointwise')} remap)"),
    )
    w = res.omega_s
    for ax, a in zip(axes, (res.poles.decay_rates, res.poles.frequencies)):
        lim = _robust_ylim(a / w)
        if lim is not None:
            ax.set_ylim(*lim)

    track = sweep.track(method)
    lam_ep = float(track.lambda_ep[ic])
    if np.isfinite(lam_ep):
        for ax in axes:
            ax.axvline(lam_ep, color=PALETTE[1], ls=":", lw=1.6,
                       label=rf"data $\lambda_{{EP}} = {lam_ep:.2f}$")

    th = _theory(sweep, theory_params)
    if th is not None:
        lams, eigs, tb = _theory_spectrum(sweep, c_used, th, lambda_max)
        for j in range(eigs.shape[1]):
            axes[0].plot(lams, -eigs[:, j].real / w, "-", color=THEORY_COLOR, lw=1.2,
                         alpha=0.7, zorder=3,
                         label=(rf"theory ($\tau_B^{{\rm eff}} = {tb:.2f}\,\mu$s)"
                                if j == 0 else None))
            axes[1].plot(lams, eigs[:, j].imag / w, "-", color=THEORY_COLOR, lw=1.2,
                         alpha=0.7, zorder=3)
        ep_th = _theory_ep(c_used, th)
        if np.isfinite(ep_th):
            for ax in axes:
                ax.axvline(ep_th, color=THEORY_COLOR, ls="--", lw=1.4,
                           label=rf"theory $\lambda_{{EP}} = {ep_th:.2f}$")
    for ax in axes:
        ax.legend(fontsize=LEGEND_FONT)
    return fig, axes


def plot_poles_compression_slider(sweep, *, method: str = "direct",
                                  lambda_max: float | None = None,
                                  theory_params: dict | None = None):
    """The two spectrum panels with a slider over the compression coefficient.

    Axis limits are fixed over the whole sweep so the poles are seen to
    move rather than the axes.  Keep a reference to the returned ``Slider``
    (matplotlib holds only a weak one) or it stops responding.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    tr = sweep.track(method)
    lam = sweep.lambda_values
    m = np.ones(len(lam), dtype=bool) if lambda_max is None else lam <= lambda_max
    w = sweep.omega_s
    ylim_dec = _robust_ylim(-tr.poles[:, m, :].real / w) or (0.0, 1.0)
    ylim_frq = _robust_ylim(tr.poles[:, m, :].imag / w) or (-1.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.subplots_adjust(bottom=0.2, top=0.88)
    cs = sweep.compress_coeffs

    def draw(ic: int) -> None:
        for ax in axes:
            ax.clear()
        plot_poles_at_compression(sweep, cs[ic], method=method, lambda_max=lambda_max,
                                  axes=axes, theory_params=theory_params)
        axes[0].set_ylim(*ylim_dec)
        axes[1].set_ylim(*ylim_frq)
        fig.canvas.draw_idle()

    slider = _slider_axis(fig, cs, [0.2, 0.06, 0.6, 0.035])
    slider.on_changed(lambda v: draw(sweep.index_of(v)))
    draw(0)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Dead-time sweep: SLIDER over c (poles)")
    _announce_slider(matplotlib.get_backend(), "poles")
    return slider


# ---------------------------------------------------------------------------
# 3. discriminant and EP vs c, with the theory prediction
# ---------------------------------------------------------------------------

def plot_transition_vs_compression(sweep, *, c_values=(0.0, 0.25, 0.5, 0.75, 1.0),
                                   lambda_max: float | None = None,
                                   theory_params: dict | None = None,
                                   references: dict | None = None,
                                   title: str | None = None):
    """``Delta(λ)`` for representative ``c`` (left) and ``λ_EP`` vs ``c`` (right).

    Right panel: the EP as the discriminant line-fit zero of the direct fit
    (filled circles; 16/84 bootstrap error bars when available, and the
    ``c = 0`` interval shaded across the panel as the reference band), of
    the regrid + MPM fallback (open circles), and — with ``theory_params`` —
    the theory transition at ``tau_B_eff = (1 - c) tau_B`` as a dark dashed
    line over the swept ``c`` range.  ``references`` (``{label: other
    DeadTimeSweepResult}``) overlays the direct-route ``λ_EP(c)`` of other
    sweeps, e.g. the experiment next to the synthetic generator.
    """
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import apply_paper_style, make_ax_lines_wider

    apply_paper_style()
    lam = sweep.lambda_values
    m = np.ones(len(lam), dtype=bool) if lambda_max is None else lam <= lambda_max
    cs = sweep.compress_coeffs
    d, fb = sweep.direct, sweep.fallback
    c_idx, c_rep = _representative(sweep, c_values)
    colors = _colors(len(c_rep))

    fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(12.0, 4.4), layout="constrained",
                                     gridspec_kw=dict(width_ratios=(1.2, 1.0)))
    ax_d.axhline(0, color="k", ls="--", lw=1.0, alpha=0.7)
    for ic, c, col in zip(c_idx, c_rep, colors):
        ax_d.plot(lam[m], d.delta[ic][m], "-o", color=col, lw=1.8, ms=3.5,
                  label=rf"$c={c:.2f}$")
        if np.isfinite(d.lambda_ep[ic]):
            ax_d.axvline(d.lambda_ep[ic], color=col, ls=":", lw=1.4, alpha=0.9)
    # Scale from the curves that still cross zero; a fully compressed curve
    # far below zero would otherwise squash the ones that carry the EP.
    scale_idx = [i for i in c_idx if np.isfinite(d.lambda_ep[i])] or c_idx
    finite = d.delta[scale_idx][:, m][np.isfinite(d.delta[scale_idx][:, m])]
    if finite.size:
        span = np.percentile(np.abs(finite), 90)
        ax_d.set_ylim(-1.4 * span, 1.4 * span)
    ax_d.set_xlabel(r"$\lambda$", fontsize=FONT)
    ax_d.set_ylabel(r"$\Delta = c_1^2 - 4c_0$  (direct fit)", fontsize=FONT)
    ax_d.legend(fontsize=LEGEND_FONT, title="dotted: $\\lambda_{EP}$",
                title_fontsize=LEGEND_FONT)

    ep_color = PALETTE[1]
    yerr = None
    if d.lambda_ep_quantiles is not None:
        q = d.lambda_ep_quantiles
        yerr = np.vstack([np.clip(d.lambda_ep - q[:, 0], 0, None),
                          np.clip(q[:, 2] - d.lambda_ep, 0, None)])
        if np.all(np.isfinite(q[0])):
            ax_t.axhspan(q[0, 0], q[0, 2], color=ep_color, alpha=0.15, lw=0,
                         label=r"$c=0$ bootstrap 16/84")
    ax_t.errorbar(cs, d.lambda_ep, yerr=yerr, fmt="o-", color=ep_color, ms=6,
                  lw=1.8, capsize=3, label=r"$\lambda_{EP}$ direct fit on $t'$")
    if fb is not None:
        ax_t.plot(cs, fb.lambda_ep, "o--", mfc="none", color=PALETTE[0], ms=6,
                  lw=1.6, label=r"$\lambda_{EP}$ regrid + MPM")

    shown = [d.lambda_ep] + ([fb.lambda_ep] if fb is not None else [])
    for i, (name, ref) in enumerate((references or {}).items()):
        col = PALETTE[2 + i % (len(PALETTE) - 2)]
        ax_t.plot(ref.compress_coeffs, ref.direct.lambda_ep, "s-", color=col, ms=6, lw=1.6,
                  label=rf"$\lambda_{{EP}}$ {name}")
        shown.append(ref.direct.lambda_ep)
    th = _theory(sweep, theory_params)
    if th is not None:
        from mpm_analysis.simulation.liouvillian_spectrum import theory_transition_vs_compression
        c_th = np.linspace(float(cs.min()), float(cs.max()), 41)
        ep_th = theory_transition_vs_compression(
            c_th, tau_B=th["tau_B"], T1=th["T1"], T2=th["T2"], Omega=th["Omega"], k=th["k"])
        ax_t.plot(c_th, ep_th, "--", color=THEORY_COLOR, lw=2.0, zorder=4,
                  label=(r"theory: 4$\times$4 Liouvillian, "
                         rf"$\tau_B^{{\rm eff}} = (1-c)\,{th['tau_B']:g}\,\mu$s"))
        shown.append(ep_th)

    # Scale to the transitions themselves, but never beyond the fitted λ
    # window: a line-fit zero outside it is an extrapolation, not a
    # transition (the CSV keeps the value).
    eps = np.concatenate(shown)
    eps = eps[np.isfinite(eps)]
    if eps.size:
        lo = max(float(lam[m].min()), eps.min() - 0.1)
        hi = min(float(lam[m].max()), eps.max() + 0.1)
        ax_t.set_ylim(lo, hi)
    ax_t.set_xlabel(r"compress coeff $c = \tau_B^{\rm comp} / \tau_B$", fontsize=FONT)
    ax_t.set_ylabel(r"$\lambda_{EP}$", fontsize=FONT)
    ax_t.legend(fontsize=LEGEND_FONT)
    for ax in (ax_d, ax_t):
        make_ax_lines_wider(ax, labelsize=FONT - 1)
        ax.grid(True, alpha=0.3, lw=0.5)
    fig.suptitle(title or "Exceptional point vs dead-time compression", fontsize=FONT + 2)
    return fig


# ---------------------------------------------------------------------------
# 4. multi-panel grids (static contact sheets over c)
# ---------------------------------------------------------------------------

def _c_indices(sweep, c_values, max_panels: int = 6) -> list[int]:
    """Indices of the swept ``c`` to show in a grid.

    ``None`` → up to ``max_panels`` evenly spaced over the sweep; an ``int``
    → that many evenly spaced; a list → the nearest swept value of each.
    """
    n = len(sweep.compress_coeffs)
    if c_values is None or isinstance(c_values, int):
        k = min(n, max_panels if c_values is None else int(c_values))
        return sorted({int(i) for i in np.round(np.linspace(0, n - 1, k))})
    return sorted({sweep.index_of(c) for c in c_values})


def plot_traces_panel_grid(records, sweep, *, c_values=None, target_lambdas=None,
                           click_key: str = DEFAULT_CLICK_KEY, show_fallback: bool = False,
                           max_panels: int = 6, title: str | None = None):
    """Static grid: one row per λ, one column per ``c`` — the slider's frames
    side by side.  Each cell: raw data (grey), data on ``t'`` and the direct
    fit (colour); rows share the y range, every cell the raw time range.
    """
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting._style import apply_paper_style, make_ax_lines_wider

    apply_paper_style()
    key = list(sweep.parameters["observable_keys"])[0]
    lam_idx = _target_lambda_indices(sweep, target_lambdas)
    c_idx = _c_indices(sweep, c_values, max_panels)
    col = PALETTE[1]
    fig, axes = plt.subplots(len(lam_idx), len(c_idx),
                             figsize=(3.2 * len(c_idx) + 0.8, 2.6 * len(lam_idx) + 0.6),
                             sharex=True, sharey="row", layout="constrained", squeeze=False)
    for r, il in enumerate(lam_idx):
        lam = float(sweep.lambda_values[il])
        raw = next(x for x in records if x.observable_key == key and x.lambda_val == lam)
        for k, ic in enumerate(c_idx):
            ax = axes[r, k]
            s = _slice_at_c(records, sweep, ic, il, key, click_key=click_key,
                            show_fallback=show_fallback)
            ax.plot(raw.t, raw.signal, "o", color=RAW_COLOR, ms=2.2, zorder=5)
            if sweep.compress_coeffs[ic] > 0:
                ax.plot(s["t"], s["y"], "o", color=col, ms=2.2, alpha=0.75, zorder=6)
            if s["fit"] is not None:
                ax.plot(*s["fit"], "-", color=col, lw=1.6, zorder=8)
            if s["regrid"] is not None:
                ax.plot(s["regrid"][2], s["regrid"][3], "--", color=PALETTE[0], lw=1.2, zorder=7)
            if r == 0:
                ax.set_title(rf"$c = {sweep.compress_coeffs[ic]:.2f}$", fontsize=FONT)
            if k == 0:
                ax.set_ylabel(f"{key}   " + rf"$\lambda={lam:.2f}$", fontsize=FONT)
            if r == len(lam_idx) - 1:
                ax.set_xlabel(r"$t'$ ($\mu$s)", fontsize=FONT)
            make_ax_lines_wider(ax, labelsize=FONT - 2)
    fig.suptitle(title or ("Dead-time compression: data on $t'$ (colour) vs raw (grey), "
                           "direct fit"), fontsize=FONT + 1)
    return fig


def plot_poles_panel_grid(sweep, *, c_values=None, method: str = "direct",
                          lambda_max: float | None = None, theory_params: dict | None = None,
                          max_panels: int = 6, title: str | None = None):
    """Static grid: one row per ``c`` with the two standard spectrum panels
    (decay rates | frequencies), theory and both EPs overlaid — the poles
    slider's frames stacked.  Axis limits are shared across rows.
    """
    import matplotlib.pyplot as plt

    c_idx = _c_indices(sweep, c_values, max_panels)
    tr = sweep.track(method)
    lam = sweep.lambda_values
    m = np.ones(len(lam), dtype=bool) if lambda_max is None else lam <= lambda_max
    w = sweep.omega_s
    ylim_dec = _robust_ylim(-tr.poles[:, m, :].real / w) or (0.0, 1.0)
    ylim_frq = _robust_ylim(tr.poles[:, m, :].imag / w) or (-1.0, 1.0)

    fig, axes = plt.subplots(len(c_idx), 2, figsize=(14, 3.6 * len(c_idx) + 0.6),
                             sharex=True, layout="constrained", squeeze=False)
    for row, ic in zip(axes, c_idx):
        c = float(sweep.compress_coeffs[ic])
        plot_poles_at_compression(sweep, c, method=method, lambda_max=lambda_max,
                                  axes=row, theory_params=theory_params)
        row[0].set_title(rf"$c = {c:.2f}$ — decay rates", fontsize=FONT)
        row[1].set_title(rf"$c = {c:.2f}$ — frequencies", fontsize=FONT)
        row[0].set_ylim(*ylim_dec)
        row[1].set_ylim(*ylim_frq)
    label = {"direct": "direct fit on $t'$", "regrid_mpm": "regrid + MPM"}.get(method, method)
    fig.suptitle(title or f"Poles vs dead-time compression ({label})", fontsize=FONT + 1)
    return fig


# ---------------------------------------------------------------------------
# 5. standalone HTML sliders (Plotly, no server needed)
# ---------------------------------------------------------------------------

def _write_html(fig, path, include_plotlyjs) -> str:
    """Write a Plotly figure as a self-contained HTML file; returns its path."""
    import os
    from pathlib import Path

    from mpm_analysis.utils.windows_paths import win_long_path

    out = Path(path)
    os.makedirs(win_long_path(out.parent), exist_ok=True)
    fig.write_html(win_long_path(out), include_plotlyjs=include_plotlyjs, full_html=True)
    return str(out.resolve())


def _c_slider_steps(fig, groups: list[list[int]], cs, title_fmt: str) -> list[dict]:
    """Plotly slider steps: step ``i`` shows the always-on traces plus ``groups[i]``."""
    n = len(fig.data)
    per_step = set().union(*groups) if groups else set()
    always = [i for i in range(n) if i not in per_step]
    steps = []
    for i, c in enumerate(cs):
        visible = [False] * n
        for j in always + groups[i]:
            visible[j] = True
        steps.append(dict(method="update", label=f"{c:.2f}",
                          args=[{"visible": visible}, {"title": title_fmt.format(c=c)}]))
    return steps


def _slider_layout(steps, title: str, **kwargs) -> dict:
    return dict(
        title=title,
        sliders=[dict(active=0, currentvalue={"prefix": "compress coeff c = ",
                                              "font": {"size": 13}},
                      pad={"t": 40}, steps=steps)],
        template="plotly_white", font={"size": 12},
        legend={"font": {"size": 11}, "itemsizing": "constant"},
        **kwargs,
    )


def export_traces_slider_html(records, sweep, path, *, target_lambdas=None,
                              click_key: str = DEFAULT_CLICK_KEY,
                              show_fallback: bool = False,
                              include_plotlyjs: str | bool = "cdn") -> str:
    """Standalone HTML version of :func:`plot_traces_compression_slider`.

    One row per λ: data + direct fit (left), the remap ``t'(t)`` with
    ``p_click`` (right); the slider steps through the swept ``c``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    key = list(sweep.parameters["observable_keys"])[0]
    lam_idx = _target_lambda_indices(sweep, target_lambdas)
    cs = sweep.compress_coeffs
    clicks = {r.lambda_val: r for r in records if r.observable_key == click_key}
    n_rows = len(lam_idx)
    fig = make_subplots(rows=n_rows, cols=2, column_widths=[0.66, 0.34],
                        specs=[[{}, {"secondary_y": True}] for _ in range(n_rows)],
                        horizontal_spacing=0.08, vertical_spacing=0.1)
    groups: list[list[int]] = [[] for _ in cs]
    for r, il in enumerate(lam_idx, start=1):
        lam = float(sweep.lambda_values[il])
        raw = next(x for x in records if x.observable_key == key and x.lambda_val == lam)
        first = r == 1
        fig.add_trace(go.Scatter(x=raw.t, y=raw.signal, mode="markers", name="raw data (c = 0)",
                                 marker={"color": "#595959", "size": 5},
                                 legendgroup="raw", showlegend=first), row=r, col=1)
        fig.add_trace(go.Scatter(x=raw.t, y=raw.t, mode="lines", name="t' = t",
                                 line={"color": "#595959", "dash": "dot", "width": 1},
                                 legendgroup="id", showlegend=first), row=r, col=2)
        fig.add_trace(go.Scatter(x=clicks[lam].t, y=clicks[lam].signal, mode="lines",
                                 name="p_click", line={"color": "#999999", "width": 1.2},
                                 legendgroup="p", showlegend=first),
                      row=r, col=2, secondary_y=True)
        for ic, c in enumerate(cs):
            s = _slice_at_c(records, sweep, ic, il, key, click_key=click_key,
                            show_fallback=show_fallback)
            vis = ic == 0
            groups[ic].append(len(fig.data))
            fig.add_trace(go.Scatter(x=s["t"], y=s["y"], mode="markers", name="data on t'",
                                     marker={"color": PALETTE[1], "size": 5, "opacity": 0.75},
                                     legendgroup="data", showlegend=first, visible=vis),
                          row=r, col=1)
            t_f, y_f = s["fit"] if s["fit"] is not None else ([], [])
            groups[ic].append(len(fig.data))
            fig.add_trace(go.Scatter(x=t_f, y=y_f, mode="lines", name="direct fit",
                                     line={"color": PALETTE[1], "width": 2.5},
                                     legendgroup="fit", showlegend=first, visible=vis),
                          row=r, col=1)
            if show_fallback:
                t_u, y_u = (s["regrid"][2], s["regrid"][3]) if s["regrid"] else ([], [])
                groups[ic].append(len(fig.data))
                fig.add_trace(go.Scatter(x=t_u, y=y_u, mode="lines", name="regrid + MPM",
                                         line={"color": PALETTE[0], "width": 1.8, "dash": "dash"},
                                         legendgroup="reg", showlegend=first, visible=vis),
                              row=r, col=1)
            groups[ic].append(len(fig.data))
            fig.add_trace(go.Scatter(x=raw.t[:s["t"].size], y=s["t"], mode="lines", name="t'(t)",
                                     line={"color": PALETTE[1], "width": 2.5},
                                     legendgroup="map", showlegend=first, visible=vis),
                          row=r, col=2)
        fig.update_yaxes(title_text=f"{key}  λ = {lam:.2f}", row=r, col=1)
        fig.update_yaxes(title_text="t' (µs)", row=r, col=2, secondary_y=False)
        fig.update_yaxes(title_text="p_click", range=[0, 1], row=r, col=2, secondary_y=True)
        fig.update_xaxes(range=[float(raw.t[0]) - 0.5, float(raw.t[-1]) + 0.5], row=r, col=1)
    fig.update_xaxes(title_text="time t' (µs)", row=n_rows, col=1)
    fig.update_xaxes(title_text="t (µs)", row=n_rows, col=2)
    title = "Dead-time compression: data and direct fit | c = {c:.2f}"
    fig.update_layout(**_slider_layout(_c_slider_steps(fig, groups, cs, title),
                                       title.format(c=cs[0]),
                                       width=1150, height=330 * n_rows + 140,
                                       margin={"l": 70, "r": 40, "t": 70, "b": 110}))
    return _write_html(fig, path, include_plotlyjs)


def export_poles_slider_html(sweep, path, *, method: str = "direct",
                             lambda_max: float | None = None,
                             theory_params: dict | None = None,
                             include_plotlyjs: str | bool = "cdn") -> str:
    """Standalone HTML version of :func:`plot_poles_compression_slider`:
    decay rates (left) and frequencies (right) of every pole vs λ, bootstrap
    error bars when present, theory eigenvalues and both EPs, slider over ``c``.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    tr = sweep.track(method)
    cs = sweep.compress_coeffs
    lam = sweep.lambda_values
    m = np.ones(len(lam), dtype=bool) if lambda_max is None else lam <= lambda_max
    w = sweep.omega_s
    ylim_dec = _robust_ylim(-tr.poles[:, m, :].real / w) or (0.0, 1.0)
    ylim_frq = _robust_ylim(tr.poles[:, m, :].imag / w) or (-1.0, 1.0)
    th = _theory(sweep, theory_params)
    pole_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Decay rates", "Frequencies"),
                        horizontal_spacing=0.08)
    groups: list[list[int]] = [[] for _ in cs]

    def _err(lo, hi, ic, k):
        if lo is None:
            return None
        return {"type": "data", "symmetric": False,
                "array": hi[ic][m, k] / w, "arrayminus": lo[ic][m, k] / w}

    for ic, c in enumerate(cs):
        vis = ic == 0
        first = ic == 0
        P = tr.poles[ic][m]
        for k in range(P.shape[1]):
            panels = ((-P[:, k].real / w, tr.decay_lower, tr.decay_upper),
                      (P[:, k].imag / w, tr.freq_lower, tr.freq_upper))
            for col_i, (y, lo, hi) in enumerate(panels, start=1):
                groups[ic].append(len(fig.data))
                fig.add_trace(go.Scatter(
                    x=lam[m], y=y, mode="markers+lines", name=f"Pole {k + 1}",
                    marker={"color": pole_colors[k % 4], "size": 5},
                    line={"color": pole_colors[k % 4], "width": 1, "dash": "dash"},
                    error_y=_err(lo, hi, ic, k), legendgroup=f"pole{k}",
                    showlegend=first and col_i == 1, visible=vis), row=1, col=col_i)
        lam_ep = float(tr.lambda_ep[ic])
        marks = [(lam_ep, PALETTE[1], "dot", "data λ_EP")]
        if th is not None:
            lams_th, eigs, tb = _theory_spectrum(sweep, c, th, lambda_max)
            for j in range(eigs.shape[1]):
                for col_i, y in enumerate((-eigs[:, j].real / w, eigs[:, j].imag / w), start=1):
                    groups[ic].append(len(fig.data))
                    fig.add_trace(go.Scatter(
                        x=lams_th, y=y, mode="lines", name=f"theory (τ_B,eff = {tb:.2f} µs)",
                        line={"color": "#262626", "width": 1.2}, opacity=0.7,
                        legendgroup="theory", showlegend=first and j == 0 and col_i == 1,
                        visible=vis), row=1, col=col_i)
            marks.append((_theory_ep(c, th), "#262626", "dash", "theory λ_EP"))
        for x, color, dash, name in marks:
            if not np.isfinite(x):
                continue
            for col_i, (lo, hi) in enumerate((ylim_dec, ylim_frq), start=1):
                groups[ic].append(len(fig.data))
                fig.add_trace(go.Scatter(
                    x=[x, x], y=[lo, hi], mode="lines", name=f"{name} = {x:.2f}",
                    line={"color": color, "width": 1.6, "dash": dash},
                    legendgroup=name, showlegend=col_i == 1, visible=vis), row=1, col=col_i)
    fig.update_xaxes(title_text="λ", row=1, col=1)
    fig.update_xaxes(title_text="λ", row=1, col=2)
    fig.update_yaxes(title_text="decay rate / Ω_S", range=list(ylim_dec), row=1, col=1)
    fig.update_yaxes(title_text="frequency / Ω_S", range=list(ylim_frq), row=1, col=2)
    label = {"direct": "direct fit on t'", "regrid_mpm": "regrid + MPM"}.get(method, method)
    title = f"Poles vs dead-time compression ({label}) | c = {{c:.2f}}"
    fig.update_layout(**_slider_layout(_c_slider_steps(fig, groups, cs, title),
                                       title.format(c=cs[0]),
                                       width=1250, height=620,
                                       margin={"l": 70, "r": 40, "t": 80, "b": 110}))
    return _write_html(fig, path, include_plotlyjs)
