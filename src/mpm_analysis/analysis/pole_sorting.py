"""
pole_sorting.py
---------------
Single canonical pole-sorting routine shared by every component that
produces or consumes complex MPM poles.

Rule (matches ``object_oriented_poles.py`` reference implementation):
    Sort by real part ascending.  Ties (e.g. degenerate EP poles) are
    broken by imaginary part ascending.

Why one canonical function?
    Without a shared routine, each module (MPMStep, bootstrap, simulation,
    RRHA) may sort differently, causing index-swap artifacts when comparing
    poles across lambda or between clean/noisy runs.  Everything goes through
    this function so the ordering is guaranteed to be consistent.
"""
from __future__ import annotations

import numpy as np


def sort_poles_canonical(poles: np.ndarray) -> np.ndarray:
    """Sort complex poles by real part ascending (ties broken by imag ascending).

    Parameters
    ----------
    poles:
        1-D complex array of length ``order``.

    Returns
    -------
    np.ndarray
        Same values, re-ordered so ``poles.real`` is non-decreasing.

    Examples
    --------
    >>> sort_poles_canonical(np.array([-1+2j, -3+0j, -1-2j]))
    array([-3.+0.j, -1.-2.j, -1.+2.j])
    """
    poles = np.asarray(poles, dtype=complex)
    # Primary key: real part ascending; secondary: imaginary part ascending
    tol = 1e-2
    binned_real = np.round(poles.real / tol) * tol
    idx = np.lexsort((poles.imag, binned_real))
    return poles[idx]


def match_poles_hungarian(
    poles: np.ndarray,
    reference: np.ndarray,
    *,
    return_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Permute ``poles`` to best match ``reference`` (Hungarian assignment).

    Minimises the total complex distance ``sum_i |poles[perm[i]] - reference[i]|``
    via ``scipy.optimize.linear_sum_assignment``.  Used to keep pole indices
    meaning the *same branch* across lambda and across bootstrap draws, where
    an independent canonical re-sort would swap labels near the EP.

    NaN poles (padding from ``enforce_pole_count``) are assigned last via a
    large finite cost, so they end up matched to whatever reference slots are
    left over.

    Parameters
    ----------
    poles:
        1-D complex array to be re-ordered.
    reference:
        1-D complex array of the same length defining the target order.
    return_indices:
        If ``True``, also return the permutation ``idx`` such that the result
        equals ``poles[idx]`` (useful to permute amplitudes alongside).

    Returns
    -------
    np.ndarray
        ``poles`` permuted so element ``i`` is the best match of ``reference[i]``
        (with ``(poles_matched, idx)`` when ``return_indices=True``).
    """
    from scipy.optimize import linear_sum_assignment

    poles = np.asarray(poles, dtype=complex)
    reference = np.asarray(reference, dtype=complex)
    if len(poles) != len(reference):
        poles = enforce_pole_count(poles, len(reference))

    cost = np.abs(poles[None, :] - reference[:, None])  # (ref, pole)
    # NaN in either side → large finite cost so the assignment stays solvable
    finite = cost[np.isfinite(cost)]
    big = (finite.max() * 1e3 + 1.0) if finite.size else 1.0
    cost = np.where(np.isfinite(cost), cost, big)
    _, col = linear_sum_assignment(cost)
    if return_indices:
        return poles[col], col
    return poles[col]


def track_branches_across_lambda(
    poles_list: list[np.ndarray],
    *,
    return_indices: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
    """Order poles consistently across a lambda scan (branch tracking).

    The first slice with finite poles is put in canonical order
    (``sort_poles_canonical``); every subsequent slice is Hungarian-matched to
    the previous finite slice, so index ``i`` follows one continuous branch
    through the scan (all-NaN slices are passed through and skipped as
    reference).

    Parameters
    ----------
    poles_list:
        One 1-D complex pole array per lambda, sorted by lambda.
    return_indices:
        If ``True``, also return the per-slice permutations (result ==
        ``poles_list[i][idx[i]]``), for permuting amplitudes alongside.

    Returns
    -------
    list[np.ndarray]
        Same arrays, re-ordered per slice (with the permutation list when
        ``return_indices=True``).
    """
    tracked: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    reference: np.ndarray | None = None
    for poles in poles_list:
        poles = np.asarray(poles, dtype=complex)
        if not np.any(np.isfinite(poles)):
            tracked.append(poles)
            indices.append(np.arange(len(poles)))
            continue
        if reference is None:
            tol = 1e-2
            binned_real = np.round(poles.real / tol) * tol
            idx = np.lexsort((poles.imag, binned_real))
            aligned = poles[idx]
        else:
            aligned, idx = match_poles_hungarian(poles, reference, return_indices=True)
        tracked.append(aligned)
        indices.append(idx)
        reference = aligned
    if return_indices:
        return tracked, indices
    return tracked


def enforce_pole_count(poles: np.ndarray, order: int) -> np.ndarray:
    """Guarantee that ``poles`` has exactly ``order`` elements.

    If ``len(poles) == order``, returns as-is.
    If shorter, pads with ``NaN + NaN*j`` on the right.
    If longer, truncates to the first ``order`` elements.

    This should only be needed as a defensive guard; ``matrix_pencil_method``
    should always return exactly ``order`` poles.

    Parameters
    ----------
    poles:
        1-D complex array.
    order:
        Required number of poles.

    Returns
    -------
    np.ndarray
        Complex array of length ``order``.
    """
    poles = np.asarray(poles, dtype=complex)
    n = len(poles)
    if n == order:
        return poles
    if n > order:
        return poles[:order]
    # Pad with NaN
    out = np.full(order, complex(float("nan"), float("nan")), dtype=complex)
    out[:n] = poles
    return out
