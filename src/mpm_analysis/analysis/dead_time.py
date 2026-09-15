"""
dead_time.py
------------
Detector dead-time compression of the time axis.

While the system sits in the bright state |B> the fast readout registers a
click and the dynamics of interest is frozen — the detector is "dead".  If
``p_click(t)`` is the ensemble probability of occupying |B> at time ``t``, the
time the system actually spent evolving up to ``t`` is shorter than ``t`` by
the accumulated dead time.  A one-parameter test of whether this dead time
shifts the extracted transition is to remap ONLY the time axis and refit the
same modal form on the remapped axis: ``c = 0`` is the raw data, ``c = 1``
full compression.  The measured y-data are never touched.  Two remaps:

pointwise  (default)
    ``t' = t * (1 - c * p_click(t))`` — the click probability at ``t``
    taken as the dead-time fraction of the whole interval.  Its slope is
    ``1 - c (p + t dp/dt)``, so a *jump* ``dp`` between neighbouring bins
    ``k`` and ``k+1`` folds the axis once ``c (p + k dp) > 1``: late bins of a
    noisy ``p_click`` (few pooled shots) fold first.  :func:`compress_records`
    can truncate a trace at its first fold instead of failing.

cumulative
    ``t' = t - c * int_0^t p_click(s) ds`` (trapezoid) — the accumulated dead
    time.  Slope ``1 - c p(t) > 0`` whenever ``p < 1``: never folds.

The remapped axis is non-uniform, so the matrix pencil cannot be applied to
it directly.  Two routes are provided downstream:

* **direct** — the variable-projection least-squares fit
  (:func:`mpm_analysis.analysis.matrix_pencil.refine_poles`) works on any
  time axis; it is seeded from the raw-data MPM poles and swept in ``c``.
* **fallback** — :func:`resample_uniform` interpolates each remapped trace
  back onto a uniform grid, on which the ordinary MPM runs.

This module holds only the pure array/record transformations; the sweep
itself lives in :mod:`mpm_analysis.pipeline.dead_time_pipeline`.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord


DEFAULT_CLICK_KEY = "click_probability"
REMAPS = ("pointwise", "cumulative")
ON_FOLD = ("raise", "truncate")


def compress_time_axis(
    t: np.ndarray, p_click: np.ndarray, c: float, *, remap: str = "pointwise",
) -> np.ndarray:
    """Remap ``t`` by the click probability (see the module docstring).

    Parameters
    ----------
    t, p_click:
        Time axis and the click (|B> occupation) probability on it, same
        shape.
    c:
        Compression coefficient in ``[0, 1]``: ``0`` leaves ``t`` unchanged,
        ``1`` removes the full dead time.
    remap:
        ``"pointwise"`` → ``t (1 - c p)``;  ``"cumulative"`` → ``t - c ∫p``.

    Returns
    -------
    np.ndarray
        The remapped axis.  Not checked for monotonicity — see
        :func:`first_fold`.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p_click, dtype=float)
    if t.shape != p.shape:
        raise ValueError(
            f"compress_time_axis: t {t.shape} and p_click {p.shape} differ in shape."
        )
    if not 0.0 <= float(c) <= 1.0:
        raise ValueError(f"compress_time_axis: c must be in [0, 1], got {c!r}.")
    if remap == "pointwise":
        return t * (1.0 - float(c) * p)
    if remap == "cumulative":
        dead = np.concatenate(([0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))))
        return t - float(c) * dead
    raise ValueError(f"compress_time_axis: unknown remap '{remap}'; expected {REMAPS}.")


def first_fold(t_new: np.ndarray) -> int | None:
    """Index of the first sample that does not increase the axis, or ``None``.

    ``t_new[:k]`` is strictly increasing for the returned ``k``.
    """
    bad = np.flatnonzero(np.diff(np.asarray(t_new, dtype=float)) <= 0)
    return int(bad[0]) + 1 if bad.size else None


def _truncate_stats(stats: dict, n: int, n_full: int) -> dict:
    """Cut every time-resolved array of the bootstrap statistics to ``n`` points.

    Any array whose last axis has the trace length is time-resolved (the
    loaders slice exactly these by the ``t_max`` mask); scalars and other
    shapes pass through.
    """
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[-1] == n_full:
            out[k] = v[..., :n]
        else:
            out[k] = v
    return out


def compress_records(
    records: list[ObservableRecord],
    c: float,
    *,
    click_key: str = DEFAULT_CLICK_KEY,
    keys: list[str] | None = None,
    remap: str = "pointwise",
    on_fold: str = "raise",
) -> list[ObservableRecord]:
    """Remap the time axis of every observable record by its λ's click probability.

    For each λ the record with ``observable_key == click_key`` supplies
    ``p_click(t)``; every other record of that λ (or only those in ``keys``)
    gets ``t' = compress_time_axis(t, p_click, c, remap=remap)``.  Signals,
    sigmas and metadata are carried over untouched; ``metadata`` gains
    ``compress_coeff`` and ``remap``.  The click records themselves are not
    returned.

    Parameters
    ----------
    on_fold:
        What to do when the remapped axis stops increasing (pointwise remap
        of a noisy ``p_click``): ``"raise"`` fails; ``"truncate"`` keeps the
        strictly increasing head of the trace — time axis, signal, sigma
        AND the time-resolved bootstrap statistics — and records the cut in
        ``metadata["fold_index"]`` (the number of points kept).

    Raises
    ------
    KeyError
        If a λ has no click-probability record.
    ValueError
        If the click record's time axis differs from the observable's, or
        the axis folds with ``on_fold="raise"``.
    """
    if on_fold not in ON_FOLD:
        raise ValueError(f"compress_records: unknown on_fold '{on_fold}'; expected {ON_FOLD}.")
    clicks = {r.lambda_val: r for r in records if r.observable_key == click_key}
    if not clicks:
        raise KeyError(
            f"compress_records: no record with observable_key='{click_key}'. "
            "Load it alongside the observables (EnsembleAverageLoader registers "
            f"'{DEFAULT_CLICK_KEY}' <- data.click_probability)."
        )
    out: list[ObservableRecord] = []
    for rec in records:
        if rec.observable_key == click_key:
            continue
        if keys is not None and rec.observable_key not in keys:
            continue
        click = clicks.get(rec.lambda_val)
        if click is None:
            raise KeyError(
                f"compress_records: lambda={rec.lambda_val:.4f} has no "
                f"'{click_key}' record."
            )
        if not np.array_equal(click.t, rec.t):
            raise ValueError(
                f"compress_records: lambda={rec.lambda_val:.4f}: the "
                f"'{click_key}' time axis differs from '{rec.observable_key}'."
            )
        t_new = compress_time_axis(rec.t, click.signal, c, remap=remap)
        meta = {**rec.metadata, "compress_coeff": float(c), "remap": remap}
        k = first_fold(t_new)
        if k is None:
            out.append(replace(rec, t=t_new, metadata=meta))
            continue
        if on_fold == "raise":
            raise ValueError(
                f"compress_records: lambda={rec.lambda_val:.4f}: the {remap} "
                f"remap folds at sample {k} for c={c:g} "
                f"(p_click={click.signal[k - 1]:.3f} -> {click.signal[k]:.3f}); "
                "use on_fold='truncate' or remap='cumulative'."
            )
        n_full = rec.t.size
        stats = meta.get("bootstrap_stats")
        if stats is not None:
            meta["bootstrap_stats"] = _truncate_stats(stats, k, n_full)
        meta["fold_index"] = k
        out.append(replace(
            rec, t=t_new[:k], signal=rec.signal[:k],
            sigma=None if rec.sigma is None else rec.sigma[:k], metadata=meta,
        ))
    return out


def resample_uniform(record: ObservableRecord, dt: float) -> ObservableRecord:
    """Linearly interpolate a record onto a uniform grid of spacing ``dt``.

    The grid starts at the record's first time and stops at the last one it
    can reach, so the ordinary (uniform-axis) matrix pencil can be run on the
    result.  ``sigma`` is interpolated the same way when present.
    """
    t = record.t
    if t.size < 2:
        raise ValueError("resample_uniform: need at least two time points.")
    n = int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1
    t_u = t[0] + dt * np.arange(n)
    return replace(
        record,
        t=t_u,
        signal=np.interp(t_u, t, record.signal),
        sigma=None if record.sigma is None else np.interp(t_u, t, record.sigma),
        metadata={**record.metadata, "regrid_dt": float(dt)},
    )
