"""
click_correlations.py
---------------------
Click / no-click probabilities vs time, for **every** click-record source.

Both estimator modes measure time from each run's **own t = 0** — there is no
anchoring and no lag axis.  They differ only in how many clicks are tolerated
between ``t = 0`` and the time being reported, which is a single parameter
(``max_clicks``) and the axis along which partial conditioning extends:

``mode="conditional"``   (``max_clicks = 0``)
    Post-selected, no-jump dynamics.  The run must be silent in **every** bin
    up to and including ``t``::

        no_click[t] = P(no click in bins 0 … t)        = P(n = 0, t)
        click[t]    = P(the first click is exactly at t)

    ``no_click`` is exactly the ``n = 0`` row of the counting statistics, i.e.
    the survival probability of the no-jump generator, so a matrix-pencil fit
    of it returns the **post-selected (non-Hermitian) eigenvalues**.

``mode="unconditional"`` (``max_clicks = None``)
    Full dynamics.  Only the bin at ``t`` matters; whatever happened in
    between is irrelevant::

        no_click[t] = P(no click at t)
        click[t]    = P(a click at t)      ( = 1 - no_click[t] )

    A fit of these returns the **full Liouvillian** eigenvalues.

``mode="partial"``       (``max_clicks = k``)
    The interpolation between the two: at most ``k`` clicks are allowed
    strictly before ``t``.  ``k = 0`` reproduces ``conditional`` and
    ``k = None`` reproduces ``unconditional``.

Unified definition
~~~~~~~~~~~~~~~~~~
With ``c(t)`` the number of clicks strictly before ``t`` and ``j(t) = 1`` when
the run clicks at ``t``, every mode reports, over the runs that have **not
terminated** by ``t`` (the common denominator)::

    no_click[t] = P( j(t) = 0  and  c(t) <= max_clicks )
    click[t]    = P( j(t) = 1  and  c(t) <= max_clicks )

The condition ``c(t) <= k`` is equivalent to ``t <= (k+1)-th click time``,
which is what :func:`_click_counts` actually evaluates — so a single kernel
covers the whole family and new conditioning policies only have to supply a
``max_clicks``.

Clicks and terminations
~~~~~~~~~~~~~~~~~~~~~~~
What counts as a "click" is whatever the source's loader defines — the
compressed builder registers the **rising** edge (first 1 of a bright run),
the JSON / simulation loaders the **falling** edge (1→0).  The estimator is
agnostic.

A run contributes to the denominator at ``t`` while its record still reaches
``t``.  Only the compressed source has records of differing length (the
compression removes a different amount of bright dwell per shot), so
``min_alive_frac`` — which truncates the output once the surviving fraction
drops below it — is only ever active there.

Error bars come from a group bootstrap: compressed records and experimental
``individual_runs`` folders resample their real runs; other dense records
resample contiguous pseudo-groups of shots.

Sources
~~~~~~~
* :class:`~mpm_analysis.io.experimental.compressed_click_record.CompressedClickRecord`
  — sparse per-shot click times + per-shot compressed lengths
  (:func:`compute_click_correlations`).
* :class:`~mpm_analysis.io.experimental.click_record_npz.ClickRecordData`
  — dense binary ``(N, T)`` click matrices, experimental or simulated
  (:func:`compute_click_correlations_dense`).
* Streamed per-run click matrices from ``individual_runs`` folders
  (:func:`compute_click_correlations_runs`).

:func:`compute_all_click_correlations` dispatches on the record type, and
:func:`to_observable_records` converts results into ``ObservableRecord``
objects so the standard ``MPMStep -> BootstrapStep`` chain fits poles to the
time axis exactly as it does for any other observable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.io.experimental.click_record_npz import ClickRecordData
from mpm_analysis.io.experimental.compressed_click_record import (
    CompressedClickRecord,
)

#: Observable keys, one complementary pair per mode.
COND_CLICK = "cond_click"
COND_NO_CLICK = "cond_no_click"
UNCOND_CLICK = "uncond_click"
UNCOND_NO_CLICK = "uncond_no_click"
PARTIAL_CLICK = "partial_click"
PARTIAL_NO_CLICK = "partial_no_click"

#: ``mode -> (click key, no-click key)``.
MODE_KEYS = {
    "conditional": (COND_CLICK, COND_NO_CLICK),
    "unconditional": (UNCOND_CLICK, UNCOND_NO_CLICK),
    "partial": (PARTIAL_CLICK, PARTIAL_NO_CLICK),
}

#: ``mode -> max_clicks`` for the two fixed modes.  ``"partial"`` takes its
#: value from the caller, which is the hook for future conditioning policies.
MODE_MAX_CLICKS = {"conditional": 0, "unconditional": None}

DEFAULT_MODE = "conditional"

#: Default alive-shot cut: keep times while at least this fraction of the
#: t = 0 runs has still not terminated (only bites for the compressed source).
DEFAULT_MIN_ALIVE_FRAC = 0.2

#: Names used before the conditional/unconditional convention was fixed.
#: They are refused rather than mapped: ``"no_click"`` used to mean what is
#: now ``"conditional"``, so silently accepting it would flip the meaning.
_RETIRED_MODES = {
    "no_click": "conditional",
    "first_click": None,
    "unconditioned": "unconditional",
    "conditioned": "conditional",
}


@dataclass
class ClickCorrelationResult:
    """Click / no-click probabilities vs time for one λ slice.

    Attributes
    ----------
    lambda_val, rate_khz, dt_us : scalars echoed from the record
        (``rate_khz`` is ``None`` for sources without a drive-rate notion).
    t_us : (n_bins,) the record's own time axis in µs, truncated to the
        retained window.  No lag, no anchoring.
    click, click_err : ``P(click at t and <= max_clicks before t)`` and its
        group-bootstrap std error.
    no_click, no_click_err : ``P(no click at t and <= max_clicks before t)``
        and its error.  For ``conditional`` this is ``P(n = 0, t)``.
    r_bar : ``P(run has >= 1 click)`` over the whole record — a summary
        number, not part of either curve.
    mode : ``"conditional"`` | ``"unconditional"`` | ``"partial"``.
    max_clicks : clicks tolerated strictly before ``t`` (``None`` = unlimited).
    window : bookkeeping — mode, ``max_clicks``, ``min_alive_frac``, retained
        bin count, run counts, alive counts at the first/last retained bin.
    metadata : free-form per-record context (``omega_DG``, ``scan_number``,
        ``source_file``, …) propagated to the ``ObservableRecord`` stage.
    """

    lambda_val: float
    rate_khz: int | None
    dt_us: float
    t_us: np.ndarray
    click: np.ndarray
    click_err: np.ndarray
    no_click: np.ndarray
    no_click_err: np.ndarray
    r_bar: float
    mode: str = DEFAULT_MODE
    max_clicks: int | None = 0
    window: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def keys(self) -> tuple[str, str]:
        """This result's observable keys: ``(click, no_click)``."""
        return MODE_KEYS[self.mode]

    def as_dict(self) -> dict:
        return dict(
            lambda_val=self.lambda_val, rate_khz=self.rate_khz,
            dt_us=self.dt_us, t_us=self.t_us,
            click=self.click, click_err=self.click_err,
            no_click=self.no_click, no_click_err=self.no_click_err,
            r_bar=self.r_bar, mode=self.mode, max_clicks=self.max_clicks,
            window=self.window,
        )

    def signal(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """``(values, errors)`` for an observable key of this result's mode."""
        click_key, no_click_key = self.keys
        if key == click_key:
            return self.click, self.click_err
        if key == no_click_key:
            return self.no_click, self.no_click_err
        raise KeyError(
            f"Unknown key {key!r} for mode {self.mode!r}; valid: {self.keys}."
        )


# ---------------------------------------------------------------------------
# Mode / parameter resolution
# ---------------------------------------------------------------------------

def _resolve_mode(mode: str, max_clicks: int | None) -> tuple[str, int | None]:
    """Validate ``mode`` and resolve the effective ``max_clicks``.

    ``conditional`` and ``unconditional`` fix ``max_clicks`` (0 and ``None``);
    ``partial`` takes it from the caller.  This is the single place a new
    conditioning policy has to plug into.
    """
    if mode in _RETIRED_MODES:
        replacement = _RETIRED_MODES[mode]
        hint = (f"use mode={replacement!r}" if replacement else
                "the first-click-anchored estimator has been removed")
        raise ValueError(
            f"Mode {mode!r} no longer exists ({hint}).  The current modes are "
            f"{sorted(MODE_KEYS)}; note that the meaning of the names changed, "
            "so this is refused rather than silently remapped."
        )
    if mode not in MODE_KEYS:
        raise ValueError(f"Unknown mode {mode!r}; valid: {sorted(MODE_KEYS)}.")

    if mode == "partial":
        if max_clicks is None:
            raise ValueError(
                "mode='partial' needs max_clicks (an int >= 0): the number of "
                "clicks tolerated strictly before t.  Use mode='conditional' "
                "for 0 and mode='unconditional' for unlimited."
            )
        max_clicks = int(max_clicks)
        if max_clicks < 0:
            raise ValueError(f"max_clicks must be >= 0, got {max_clicks}.")
        return mode, max_clicks

    if max_clicks is not None and max_clicks != MODE_MAX_CLICKS[mode]:
        raise ValueError(
            f"mode={mode!r} fixes max_clicks={MODE_MAX_CLICKS[mode]!r}; pass "
            "mode='partial' to choose it yourself."
        )
    return mode, MODE_MAX_CLICKS[mode]


def _resolve_width(max_time_us: float | None, t: np.ndarray) -> int:
    """Number of leading bins to evaluate, before the alive cut."""
    width = len(t)
    if max_time_us is not None:
        width = min(width, int(np.searchsorted(t, max_time_us, side="right")))
    return max(1, width)


# ---------------------------------------------------------------------------
# Core counting kernel (shared by every source and every mode)
# ---------------------------------------------------------------------------

def _tail_counts(values: np.ndarray, width: int, n: int) -> np.ndarray:
    """``#{values > j}`` for ``j = 0 … width-1``, as float.

    Values at or beyond ``width`` are clipped onto ``width`` so they never
    enter the subtracted prefix — the count stays exact for every ``j``.
    """
    hist = np.bincount(np.minimum(values, width), minlength=width + 1)
    return (n - np.cumsum(hist)[:width]).astype(np.float64)


def _click_counts(
    click_times: np.ndarray,
    shot_offsets: np.ndarray,
    lengths: np.ndarray,
    width: int,
    max_clicks: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Click / no-click counts vs time for one group of runs.

    Parameters
    ----------
    click_times : flat int array of click bin indices, grouped by run and
        time-ordered within each run.
    shot_offsets : (n_runs+1,) — run ``i``'s clicks are
        ``click_times[shot_offsets[i]:shot_offsets[i+1]]``.
    lengths : (n_runs,) recorded length in bins; run ``i`` still contributes
        at bin ``j`` while ``j < lengths[i]``.
    width : number of bins evaluated (``j = 0 … width-1``).
    max_clicks : clicks tolerated strictly before ``j`` (``None`` = unlimited).

    Returns
    -------
    n_click : (width,)     runs clicking at ``j`` with ``<= max_clicks`` before.
    n_no_click : (width,)  runs silent at ``j`` with ``<= max_clicks`` before.
    alive : (width,)       runs not yet terminated at ``j`` (the denominator).
    n_clicked : float      runs with at least one click anywhere.
    n_runs : int           runs in this group.

    Notes
    -----
    ``c(j) <= max_clicks`` is equivalent to ``j <= g``, where ``g`` is the
    ``(max_clicks + 1)``-th click time of the run (``+inf`` when it has fewer
    clicks).  So "still eligible at ``j``" is ``j < min(length, g + 1)`` —
    the same tail-count form as the alive count, which is why the whole
    family costs one ``bincount`` each.
    """
    click_times = np.asarray(click_times)
    shot_offsets = np.asarray(shot_offsets)
    lengths = np.asarray(lengths)

    counts = np.diff(shot_offsets)
    n_runs = len(counts)
    n_clicked = float((counts > 0).sum())

    alive = _tail_counts(lengths, width, n_runs)
    if n_runs == 0:
        z = np.zeros(width, dtype=np.float64)
        return z, z.copy(), alive, 0.0, 0

    if max_clicks is None:
        kept = click_times                      # every click is eligible
        eligible_until = lengths                # only termination ends a run
    else:
        # keep each run's first (max_clicks + 1) clicks
        starts = np.repeat(shot_offsets[:-1], counts)
        rank = np.arange(len(click_times)) - starts
        kept = click_times[rank <= max_clicks]
        # g = the (max_clicks+1)-th click time, or "never"
        has_g = counts > max_clicks
        g = np.full(n_runs, width, dtype=np.int64)
        if np.any(has_g):
            g[has_g] = click_times[shot_offsets[:-1][has_g] + max_clicks]
        eligible_until = np.minimum(lengths, g + 1)

    n_click = np.bincount(np.minimum(kept, width), minlength=width + 1)[:width]
    n_click = n_click.astype(np.float64)
    eligible = _tail_counts(eligible_until, width, n_runs)
    n_no_click = eligible - n_click

    return n_click, n_no_click, alive, n_clicked, n_runs


def _clicks_to_sparse(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense boolean ``(N, T)`` click matrix -> (click_times, shot_offsets)."""
    rows, cols = np.nonzero(B)                # row-major: shot then time order
    counts = np.bincount(rows, minlength=B.shape[0])
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return cols, offsets


def _estimate(
    click_counts: np.ndarray,
    no_click_counts: np.ndarray,
    alive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts -> probabilities over the runs that have not terminated."""
    click = np.full(click_counts.shape, np.nan, dtype=np.float64)
    no_click = np.full(no_click_counts.shape, np.nan, dtype=np.float64)
    valid = alive > 0
    click[valid] = click_counts[valid] / alive[valid]
    no_click[valid] = no_click_counts[valid] / alive[valid]
    return click, no_click


def _bootstrap_errors(
    click_g: np.ndarray,        # (n_groups, width)
    no_click_g: np.ndarray,     # (n_groups, width)
    alive_g: np.ndarray,        # (n_groups, width)
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-bootstrap std errors: resample groups, sum counts, re-estimate."""
    n_groups, width = click_g.shape
    c_err = np.zeros(width)
    n_err = np.zeros(width)
    if n_bootstrap and n_groups > 1:
        rng = np.random.default_rng(seed)
        c_draws = np.empty((n_bootstrap, width))
        n_draws = np.empty((n_bootstrap, width))
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_groups, size=n_groups)
            cb, nb = _estimate(
                click_g[idx].sum(axis=0), no_click_g[idx].sum(axis=0),
                alive_g[idx].sum(axis=0),
            )
            c_draws[b] = cb
            n_draws[b] = nb
        c_err = np.nanstd(c_draws, axis=0, ddof=1)
        n_err = np.nanstd(n_draws, axis=0, ddof=1)
    return c_err, n_err


def _finalize(
    click_g: np.ndarray,
    no_click_g: np.ndarray,
    alive_g: np.ndarray,
    n_clicked: float,
    n_runs: float,
    *,
    lambda_val: float,
    rate_khz: int | None,
    t_axis: np.ndarray,
    mode: str,
    max_clicks: int | None,
    min_alive_frac: float,
    n_bootstrap: int,
    seed: int,
    metadata: dict,
    extra_window: dict,
) -> ClickCorrelationResult:
    """Alive-cut, estimate, bootstrap and package the per-group counts."""
    alive = alive_g.sum(axis=0)
    if alive[0] <= 0:
        raise ValueError(f"λ={lambda_val:.3f}: no runs at t = 0.")

    # ``alive`` is monotone non-increasing, so the kept bins are a prefix.
    width = max(1, int((alive >= min_alive_frac * alive[0]).sum()))
    click_g = click_g[:, :width]
    no_click_g = no_click_g[:, :width]
    alive_g = alive_g[:, :width]
    t_axis = np.asarray(t_axis, dtype=float)[:width]

    click, no_click = _estimate(
        click_g.sum(axis=0), no_click_g.sum(axis=0), alive_g.sum(axis=0),
    )
    click_err, no_click_err = _bootstrap_errors(
        click_g, no_click_g, alive_g, n_bootstrap, seed,
    )

    dt_us = float(t_axis[1] - t_axis[0]) if len(t_axis) > 1 else float("nan")
    window = dict(
        estimator=mode,
        max_clicks=max_clicks,
        min_alive_frac=float(min_alive_frac),
        n_bins=int(width),
        n_runs=int(n_runs),
        n_clicked=int(n_clicked),
        alive_first=float(alive[0]),
        alive_last=float(alive[width - 1]),
        **extra_window,
    )
    return ClickCorrelationResult(
        lambda_val=float(lambda_val), rate_khz=rate_khz, dt_us=dt_us,
        t_us=t_axis,
        click=click, click_err=click_err,
        no_click=no_click, no_click_err=no_click_err,
        r_bar=float(n_clicked / n_runs) if n_runs else np.nan,
        mode=mode, max_clicks=max_clicks,
        window=window, metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Compressed source — sparse click times, per-shot compressed lengths
# ---------------------------------------------------------------------------

def compute_click_correlations(
    rec: CompressedClickRecord,
    *,
    mode: str = DEFAULT_MODE,
    max_clicks: int | None = None,
    max_time_us: float | None = None,
    min_alive_frac: float = DEFAULT_MIN_ALIVE_FRAC,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> ClickCorrelationResult:
    """Click / no-click probabilities for one compressed record.

    Each run of the record is one bootstrap resampling unit.  Because the
    compressed lengths differ per shot, this is the source where
    ``min_alive_frac`` actually truncates the output.
    """
    mode, max_clicks = _resolve_mode(mode, max_clicks)
    width0 = _resolve_width(max_time_us, rec.t)
    rso = rec.run_shot_offsets
    so = rec.shot_offsets
    ct = rec.click_times
    lengths = rec.compressed_lengths

    click_g = np.zeros((rec.n_runs, width0))
    no_click_g = np.zeros((rec.n_runs, width0))
    alive_g = np.zeros((rec.n_runs, width0))
    n_clicked = 0.0
    n_runs = 0
    for r in range(rec.n_runs):
        s0, s1 = int(rso[r]), int(rso[r + 1])
        c0 = int(so[s0])
        click_g[r], no_click_g[r], alive_g[r], nc, ns = _click_counts(
            ct[c0:int(so[s1])], so[s0:s1 + 1] - c0, lengths[s0:s1],
            width0, max_clicks,
        )
        n_clicked += nc
        n_runs += ns

    metadata = dict(rec.metadata)
    metadata.setdefault("source_file", rec.source_file)
    if np.isfinite(rec.omega_dg):
        metadata.setdefault("omega_DG", float(rec.omega_dg))

    return _finalize(
        click_g, no_click_g, alive_g, n_clicked, n_runs,
        lambda_val=rec.lambda_val, rate_khz=rec.rate_khz,
        t_axis=np.asarray(rec.t, dtype=float),
        mode=mode, max_clicks=max_clicks, min_alive_frac=min_alive_frac,
        n_bootstrap=n_bootstrap, seed=seed,
        metadata=metadata, extra_window=dict(n_groups=rec.n_runs),
    )


# ---------------------------------------------------------------------------
# Dense source — (N, T) binary click matrices (experiment JSON+NPZ, simulation)
# ---------------------------------------------------------------------------

def compute_click_correlations_dense(
    data: ClickRecordData,
    *,
    mode: str = DEFAULT_MODE,
    max_clicks: int | None = None,
    max_time_us: float | None = None,
    min_alive_frac: float = DEFAULT_MIN_ALIVE_FRAC,
    n_bootstrap: int = 200,
    seed: int = 0,
    n_groups: int = 25,
) -> ClickCorrelationResult:
    """Click / no-click probabilities for one dense record.

    Same estimator as :func:`compute_click_correlations`.  Every shot spans
    the full record here, so the denominator is constant and
    ``min_alive_frac`` never truncates.  ``n_groups`` contiguous shot groups
    act as the bootstrap resampling unit.
    """
    if data.trajectories is None or data.trajectories.size == 0:
        raise ValueError(
            f"Record λ={data.lambda_val:.3f} has no trajectories — "
            "load the source with trajectories enabled."
        )
    mode, max_clicks = _resolve_mode(mode, max_clicks)
    t = np.asarray(data.t, dtype=float)
    width0 = _resolve_width(max_time_us, t)
    n_bins = data.trajectories.shape[1]

    n_shots_total = data.trajectories.shape[0]
    n_groups = max(1, min(n_groups, n_shots_total))
    bounds = np.linspace(0, n_shots_total, n_groups + 1).astype(int)

    click_g = np.zeros((n_groups, width0))
    no_click_g = np.zeros((n_groups, width0))
    alive_g = np.zeros((n_groups, width0))
    n_clicked = 0.0
    n_runs = 0
    for g in range(n_groups):
        B = data.trajectories[bounds[g]:bounds[g + 1]] > 0
        ct, so = _clicks_to_sparse(B)
        lengths = np.full(B.shape[0], n_bins)
        click_g[g], no_click_g[g], alive_g[g], nc, ns = _click_counts(
            ct, so, lengths, width0, max_clicks,
        )
        n_clicked += nc
        n_runs += ns

    metadata = {k: v for k, v in data.metadata.items()
                if np.isscalar(v) or isinstance(v, str)}
    metadata.setdefault("source_file", data.source_file)

    return _finalize(
        click_g, no_click_g, alive_g, n_clicked, n_runs,
        lambda_val=data.lambda_val, rate_khz=None, t_axis=t,
        mode=mode, max_clicks=max_clicks, min_alive_frac=min_alive_frac,
        n_bootstrap=n_bootstrap, seed=seed,
        metadata=metadata, extra_window=dict(n_groups=n_groups),
    )


# ---------------------------------------------------------------------------
# Streamed run groups — one λ split over per-run click matrices
# ---------------------------------------------------------------------------

def compute_click_correlations_runs(
    runs,
    t: np.ndarray,
    *,
    lambda_val: float,
    rate_khz: int | None = None,
    metadata: dict | None = None,
    mode: str = DEFAULT_MODE,
    max_clicks: int | None = None,
    max_time_us: float | None = None,
    min_alive_frac: float = DEFAULT_MIN_ALIVE_FRAC,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> ClickCorrelationResult:
    """Click / no-click probabilities for one λ whose shots arrive per run.

    ``runs`` is any iterable (a generator works — matrices are consumed one at
    a time and never held together) of dense binary ``(N_r, T)`` click
    matrices sharing the time axis ``t``.  Each run is one bootstrap
    resampling unit, exactly like the runs of a compressed record.
    """
    mode, max_clicks = _resolve_mode(mode, max_clicks)
    t = np.asarray(t, dtype=float)
    width0 = _resolve_width(max_time_us, t)

    click_rows, no_click_rows, alive_rows = [], [], []
    n_clicked = 0.0
    n_runs = 0
    for clicks in runs:
        B = np.asarray(clicks) > 0
        ct, so = _clicks_to_sparse(B)
        lengths = np.full(B.shape[0], B.shape[1])
        c, nc_arr, al, nc, ns = _click_counts(
            ct, so, lengths, width0, max_clicks)
        click_rows.append(c)
        no_click_rows.append(nc_arr)
        alive_rows.append(al)
        n_clicked += nc
        n_runs += ns
    if not click_rows:
        raise ValueError(f"λ={lambda_val:.3f}: no runs supplied.")

    return _finalize(
        np.vstack(click_rows), np.vstack(no_click_rows), np.vstack(alive_rows),
        n_clicked, n_runs,
        lambda_val=lambda_val, rate_khz=rate_khz, t_axis=t,
        mode=mode, max_clicks=max_clicks, min_alive_frac=min_alive_frac,
        n_bootstrap=n_bootstrap, seed=seed,
        metadata=dict(metadata or {}),
        extra_window=dict(n_groups=len(click_rows)),
    )


# ---------------------------------------------------------------------------
# Dispatch + ObservableRecord integration
# ---------------------------------------------------------------------------

def compute_all_click_correlations(
    records: list,
    **kwargs,
) -> list[ClickCorrelationResult]:
    """Compute click statistics for every record, sorted by λ.

    Dispatches per record: :class:`CompressedClickRecord` goes through the
    sparse estimator, :class:`ClickRecordData` (dense trajectories) through
    the dense one.  ``kwargs`` are forwarded (``n_groups`` only applies to
    dense records).
    """
    out: list[ClickCorrelationResult] = []
    for rec in records:
        if isinstance(rec, CompressedClickRecord):
            kw = {k: v for k, v in kwargs.items() if k != "n_groups"}
            out.append(compute_click_correlations(rec, **kw))
        elif isinstance(rec, ClickRecordData):
            out.append(compute_click_correlations_dense(rec, **kwargs))
        else:
            raise TypeError(
                f"Unsupported record type {type(rec).__name__}; expected "
                "CompressedClickRecord or ClickRecordData."
            )
    out.sort(key=lambda r: r.lambda_val)
    return out


def to_observable_records(
    results: list[ClickCorrelationResult],
    *,
    observables: tuple[str, ...] | None = None,
    skip_first: int = 0,
) -> list[ObservableRecord]:
    """Convert results into MPM-ready ``ObservableRecord`` objects.

    Each result yields one record per observable key of its mode (or the
    explicit ``observables`` subset), with the record's own time axis as
    ``t`` — so ``MPMStep`` fits complex exponentials over it exactly as it
    fits any other observable.

    ``skip_first`` drops that many leading time points, for records whose
    first bins carry an initialisation transient.
    """
    sl = slice(skip_first, None)
    out: list[ObservableRecord] = []
    for res in results:
        for key in (res.keys if observables is None else observables):
            values, errors = res.signal(key)
            meta = dict(res.metadata)
            meta.update(
                signal_err=np.asarray(errors)[sl],
                r_bar=res.r_bar,
                rate_khz=res.rate_khz,
                mode=res.mode,
                max_clicks=res.max_clicks,
                window=dict(res.window),
            )
            out.append(ObservableRecord(
                lambda_val=res.lambda_val,
                t=res.t_us[sl],
                signal=np.asarray(values)[sl],
                observable_key=key,
                source_file=str(res.metadata.get("source_file", "")),
                metadata=meta,
            ))
    return out
