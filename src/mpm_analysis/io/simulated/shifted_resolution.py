"""
shifted_resolution.py
---------------------
Reconstruct fine-resolution counting statistics from **shifted repetitions** of a
coarse-readout click experiment.

The problem
~~~~~~~~~~~
A readout of length ``dt`` only reports "≥1 click somewhere in this window", so the
click count ``n(t)`` can never exceed one per coarse bin.  Repeating the whole
experiment ``n_shifts`` times with the readout grid delayed by ``j·δ``
(``n_shifts·δ = dt``) gives ``n_shifts`` overlapping window tilings of the same
timeline, which jointly pin down the click probability of each individual δ bin.

Symbols
~~~~~~~
======================  ====================================================
``dt``                  coarse readout interval
``δ = dt / n_shifts``   fine step
``j = 0 … n_shifts-1``  which shifted repetition
``m = 0 … n_coarse-1``  which readout within one repetition
``f``                   fine-bin index; bin ``f`` is ``[f·δ, (f+1)·δ)``
``q_f``                 click probability of fine bin ``f`` — what we solve for
``p[j, m]``             measured click probability of readout ``m`` of shift ``j``
======================  ====================================================

The one modelling assumption is that the fine bins are **independent** coin flips.
Everything below is exact given that.

Step 1 — window coverage
~~~~~~~~~~~~~~~~~~~~~~~~
Shift ``j`` evolves freely for ``j`` fine steps (the hardware cannot perform a
sub-``dt`` readout), then takes full-length readouts back to back.  So readout
``m`` of shift ``j`` covers the ``n_shifts`` fine bins::

    [ j + n_shifts*m ,  j + n_shifts*m + n_shifts )

Across all ``(j, m)`` every fine index is the left edge of exactly one window —
no gaps, no duplicates.

Step 2 — "no click" probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A readout reports "no click" only if *all* of its fine bins were empty::

    1 - p[j, m] = ∏ over that window's bins of (1 - q_f)

Define the running no-click probability on the fine grid::

    N(F) = P(no click anywhere in [0, F·δ)) = ∏_{f < F} (1 - q_f),   N(0) = 1

Because shift ``j``'s readouts tile ``[j·δ, (j + n_shifts·M)·δ)`` end to end::

    N(j + n_shifts*M) = N(j) · ∏_{m<M} (1 - p[j, m])                    (*)

That is the whole reconstruction: a running product of measured numbers.

Step 3 — the ``n_shifts`` starting values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``(*)`` needs one starting value ``N(j)`` per chain — the no-click probability over
the first ``j`` fine bins.  Those live entirely inside the first coarse bin, which
no shifted readout resolves, so assume a flat click rate there::

    N(j) = (1 - p[0, 0]) ** (j / n_shifts)      for j = 0 … n_shifts-1

This is exactly why the **first coarse timestep is dropped** from the output: it is
the one column that rests on that assumption.  Later columns use only measured
values — ``N(j)`` is a constant factor on chain ``j``, and every coarse bin's total
contains one full turn through all the chains, so the assumption cancels.

Step 4 — per-bin probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``q_f = 1 - N(f+1) / N(f)``.  Statistical noise can push this negative; see
:func:`fine_click_probs` for the normalization-preserving guard.

Step 5 — P(n, t) on the coarse grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``n(T)`` is the number of successes among independent coins ``q_0 … q_{F-1}``,
built by folding in one coin at a time and snapshotting at each coarse boundary.
See :func:`poisson_binomial_pnt`.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "no_click_curve",
    "fine_click_probs",
    "poisson_binomial_pnt",
    "reconstruct",
]


def no_click_curve(p: np.ndarray) -> np.ndarray:
    """Running no-click probability ``N(F)`` on the fine grid (Steps 2–3).

    Parameters
    ----------
    p : np.ndarray, shape (n_shifts, n_coarse)
        ``p[j, m]`` — measured click probability of readout ``m`` of shift ``j``.

    Returns
    -------
    np.ndarray, shape (n_shifts * n_coarse + 1,)
        ``N(F)`` for ``F = 0 … n_shifts*n_coarse``.  ``N[0] == 1``.

    Notes
    -----
    Computed in log space: the chains are products of up to ``n_coarse`` factors,
    and the tail can underflow float64 badly enough to lose all precision if the
    products are formed directly.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"p must be 2-D (n_shifts, n_coarse); got shape {p.shape}")
    n_shifts, n_coarse = p.shape
    if not np.all((p >= 0.0) & (p < 1.0)):
        raise ValueError(
            "p must lie in [0, 1); a readout with p == 1 makes N(F) = 0 and the "
            "reconstruction undefined beyond that point."
        )

    # log(1 - p) via log1p: p is often small and 1-p rounds badly.
    log1m_p = np.log1p(-p)                                  # (n_shifts, n_coarse)

    # Step 3: chain j starts at log N(j) = (j / n_shifts) * log(1 - p[0, 0]).
    j_idx = np.arange(n_shifts, dtype=float)
    log_start = (j_idx / n_shifts) * log1m_p[0, 0]          # (n_shifts,)

    # Step 2: log N(j + n_shifts*M) = log N(j) + sum_{m<M} log(1 - p[j, m]).
    # Column M of `chain` is log N(j + n_shifts*M), for M = 0 … n_coarse.
    chain = np.empty((n_shifts, n_coarse + 1), dtype=float)
    chain[:, 0] = log_start
    chain[:, 1:] = log_start[:, None] + np.cumsum(log1m_p, axis=1)

    # Interleave the chains back onto the fine grid: fine index j + n_shifts*M
    # lives at chain[j, M].  Dropping the last n_shifts-1 entries keeps only
    # F <= n_shifts*n_coarse, which is all the coarse boundaries need.
    log_N = chain.T.ravel()[: n_shifts * n_coarse + 1]
    return np.exp(log_N)


def fine_click_probs(p: np.ndarray, *, renormalize: bool = True) -> np.ndarray:
    """Per-fine-bin click probabilities ``q_f`` (Step 4).

    Parameters
    ----------
    p : np.ndarray, shape (n_shifts, n_coarse)
        Measured per-readout click probabilities.
    renormalize : bool
        Apply the normalization guard.  Noise can make ``N(f+1) > N(f)``, i.e.
        ``q_f < 0``.  Clipping alone would inflate every coarse bin's total click
        probability, so each coarse bin is rescaled afterwards to put its
        ``∏(1 - q_f)`` back on the directly measured ``1 - p[0, M]``.  Nothing is
        created or destroyed at coarse resolution.  Set ``False`` to inspect the
        raw (possibly negative) values.

    Returns
    -------
    np.ndarray, shape (n_shifts * n_coarse,)
        ``q_f`` for ``f = 0 … n_shifts*n_coarse - 1``.
    """
    p = np.asarray(p, dtype=float)
    n_shifts, n_coarse = p.shape

    N = no_click_curve(p)
    ratio = N[1:] / N[:-1]
    q = 1.0 - ratio

    if not renormalize:
        return q

    q = np.clip(q, 0.0, 1.0 - 1e-15)

    # Rescale each coarse bin's log(1 - q) so the bin product matches its
    # measured no-click probability exactly.  Bin M covers fine bins
    # [n_shifts*M, n_shifts*(M+1)) and shift 0's readout M measures exactly that
    # span, so 1 - p[0, M] is the direct measurement of the bin product.
    log1m_q = np.log1p(-q).reshape(n_coarse, n_shifts)
    got = log1m_q.sum(axis=1)                      # (n_coarse,)
    want = np.log1p(-p[0, :])                      # (n_coarse,)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(got < 0.0, want / got, 1.0)
    log1m_q *= scale[:, None]
    return (-np.expm1(log1m_q)).ravel()


def poisson_binomial_pnt(
    q: np.ndarray,
    *,
    n_shifts: int,
    drop_first: bool = True,
) -> np.ndarray:
    """P(n, t) on the coarse grid from independent fine-bin coins (Step 5).

    Folds one coin at a time into the count distribution,
    ``P[n] <- P[n]*(1-q) + P[n-1]*q``, snapshotting at every coarse boundary.
    This is "sum over all realizations giving n clicks", done by convolution
    instead of enumeration.

    Parameters
    ----------
    q : np.ndarray, shape (n_fine,)
        Per-fine-bin click probabilities.  ``n_fine`` must be a whole number of
        coarse bins.
    n_shifts : int
        Fine bins per coarse bin.
    drop_first : bool
        Drop the first coarse column, which rests on the flat-rate assumption of
        Step 3.

    Returns
    -------
    np.ndarray, shape (n_fine + 1, n_coarse_out), float64
        ``P(n, t)``.  Column ``M`` is the distribution of the click count
        accumulated **through** coarse bin ``M`` — the same convention as
        ``generate_click_record_raw.py``, where ``pnt[:, t]`` is the count through
        window ``t`` while ``t_values[t]`` labels that window's start.
    """
    q = np.asarray(q, dtype=float)
    n_fine = q.size
    if n_fine % n_shifts:
        raise ValueError(
            f"q has {n_fine} fine bins, not a whole number of coarse bins of "
            f"{n_shifts}."
        )
    n_coarse = n_fine // n_shifts

    pnt = np.zeros((n_fine + 1, n_coarse), dtype=float)
    dist = np.zeros(n_fine + 1, dtype=float)
    dist[0] = 1.0

    n_hi = 0                     # highest n with nonzero mass so far
    for f, qf in enumerate(q):
        n_hi += 1
        # numpy evaluates the right-hand side into a temporary first, so this is
        # a safe in-place fold; P[0] must be updated after, since it feeds P[1].
        dist[1 : n_hi + 1] = dist[1 : n_hi + 1] * (1.0 - qf) + dist[0:n_hi] * qf
        dist[0] *= (1.0 - qf)
        if (f + 1) % n_shifts == 0:
            pnt[:, (f + 1) // n_shifts - 1] = dist

    return pnt[:, 1:] if drop_first else pnt


def reconstruct(
    clicks: np.ndarray,
    n_trials_per_shift: int,
    *,
    k_values: np.ndarray,
    drop_first: bool = True,
) -> dict:
    """Full Steps 2–6 for one lambda.

    Parameters
    ----------
    clicks : np.ndarray, shape (n_shifts, n_coarse)
        Click counts — ``clicks[j, m]`` summed over trials.
    n_trials_per_shift : int
        Trials behind each entry of ``clicks``.
    k_values : np.ndarray, shape (n_k,)
        k grid (rad) for the Fourier transform.
    drop_first : bool
        Drop the first coarse column (Step 3's assumption).

    Returns
    -------
    dict with keys ``q`` (n_fine,), ``pnt`` (n_fine+1, n_coarse_out) float32, and
    ``pkt`` (n_k, n_coarse_out) complex128.
    """
    clicks = np.asarray(clicks)
    n_shifts = clicks.shape[0]
    p = clicks / float(n_trials_per_shift)

    q = fine_click_probs(p)
    pnt = poisson_binomial_pnt(q, n_shifts=n_shifts, drop_first=drop_first)

    # Step 6: P(k, t) = sum_n e^{ikn} P(n, t).  complex128 — with ~n_fine bins
    # |P(k,t)| near k=pi gets small enough that complex64 flushes it to zero.
    n_arr = np.arange(pnt.shape[0], dtype=float)
    exp_matrix = np.exp(1j * np.outer(np.asarray(k_values, dtype=float), n_arr))
    pkt = exp_matrix @ pnt.astype(complex)

    return {"q": q, "pnt": pnt.astype(np.float32), "pkt": pkt}
