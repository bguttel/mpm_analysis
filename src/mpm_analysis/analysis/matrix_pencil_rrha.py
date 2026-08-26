"""
matrix_pencil_rrha.py
---------------------
Reduced-Rank Hankel Approximation (RRHA) for the Matrix Pencil Method.

This is a literal port of ``get_mpm_poles_from_rrha_matrix`` and its helpers
from ``object_oriented_poles.py`` (lines 1849–1993), refactored as stateless
module-level functions.

Algorithm
~~~~~~~~~
1. Build the Hankel data matrix from the signal  (L = N//3).
2. Iterate RRHA: alternating SVD reduction → Hankel projection, stopping when
   the log-ratio of consecutive singular values exceeds a noise-floor threshold.
3. Run the standard Matrix Pencil on the rank-reduced Hankel matrix to extract
   poles.

The iteration enforces that the approximated matrix has near-rank ``order``
before the pencil decomposition, which suppresses noise in the sub-dominant
singular values and typically improves pole recovery in noisy conditions.

Usage
-----
    from mpm_analysis.analysis.matrix_pencil_rrha import (
        mpm_rrha, get_mpm_poles_from_rrha_matrix
    )

    poles, singular_values, L = mpm_rrha(t, y, order=3)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd, hankel


# ---------------------------------------------------------------------------
# Low-level building blocks (ported literally from object_oriented_poles.py)
# ---------------------------------------------------------------------------

def build_hankel_rrha(data: np.ndarray, L: int) -> np.ndarray:
    """Build a Hankel matrix using ``scipy.linalg.hankel`` (matches old code).

    Parameters
    ----------
    data:
        1-D signal array of length N.
    L:
        Pencil parameter.

    Returns
    -------
    np.ndarray
        Hankel matrix of shape (N - L, L + 1).  Constructed as
        ``hankel(data[:N-L], data[N-L-1:])``, which matches the original
        ``build_hankel`` in ``object_oriented_poles.py``.
    """
    N = len(data)
    return hankel(data[: N - L], data[N - L - 1 :])


def svd_reduction(Y: np.ndarray, order: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Truncate matrix to rank ``order`` via SVD.

    Parameters
    ----------
    Y:
        Input matrix.
    order:
        Target rank.

    Returns
    -------
    (Y_reduced, S_first_2order)
        ``Y_reduced`` is the rank-``order`` approximation of ``Y``.
        ``S_first_2order`` are the first ``2*order`` singular values (used by
        the stop criterion and for diagnostics).
    """
    U, S, Vh = svd(Y, full_matrices=False)
    U_red = U[:, :order]
    S_red = np.diag(S[:order])
    Vh_red = Vh[:order, :]
    Y_reduced = U_red @ S_red @ Vh_red
    return Y_reduced, S[: 2 * order]


def hankel_approximation(M: np.ndarray) -> np.ndarray:
    """Project a matrix to the nearest Hankel matrix.

    Each entry in the output is the mean of the anti-diagonal of ``M`` that
    passes through that entry (anti-diagonal index = row + col).

    Parameters
    ----------
    M:
        Input matrix.

    Returns
    -------
    np.ndarray
        Hankel-structured approximation of ``M``, same shape.
    """
    rows, cols = M.shape
    H = np.zeros_like(M, dtype=M.dtype)

    dmax = rows + cols - 1
    sums = np.zeros(
        dmax,
        dtype=np.complex128 if np.iscomplexobj(M) else np.float64,
    )
    counts = np.zeros(dmax, dtype=np.int64)

    for i in range(rows):
        for j in range(cols):
            k = i + j
            sums[k] += M[i, j]
            counts[k] += 1

    means = sums / counts

    for i in range(rows):
        for j in range(cols):
            H[i, j] = means[i + j]

    return H


def rrha_stop_criterion(A: np.ndarray, order: int, rtol: float = 1e-2) -> bool:
    """Return ``True`` when the RRHA iteration has converged.

    The criterion checks whether the log-ratio of the ``order``-th to the
    ``(order-1)``-th singular value exceeds a noise-floor proxy derived from
    the median of singular values just above the signal band.  When this ratio
    is large, the matrix is close to exact rank ``order``.

    Parameters
    ----------
    A:
        Current RRHA-iterated matrix.
    order:
        Target rank.
    rtol:
        Tolerance (not directly used in the ratio test; kept for API
        compatibility with the original).

    Returns
    -------
    bool
        ``True`` → converged; iteration may stop.
    """
    s = svd(A, compute_uv=False)
    if len(s) <= order:
        return True  # nothing to test — already at/below rank
    tail_med = float(np.median(s[order : 2 * order + 1]))
    if tail_med <= 0:
        return True
    rtol_drop = np.log(tail_med)
    return float(np.log(s[order] / (s[order - 1] + 1e-300))) > rtol_drop


def get_approximated_reduced_hankel(
    matrix: np.ndarray,
    order: int = 3,
    max_iter: int = 5,
    rtol: float = 1e-1,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reduce a Hankel matrix to rank ``order``.

    Alternates SVD truncation and Hankel re-projection until the stop
    criterion fires or ``max_iter`` iterations are exhausted.

    Parameters
    ----------
    matrix:
        Initial Hankel matrix (output of ``build_hankel_rrha``).
    order:
        Target rank.
    max_iter:
        Maximum number of RRHA iterations.  Raises ``RuntimeError`` if not
        converged within this many iterations.
    rtol:
        Passed to ``rrha_stop_criterion``.

    Returns
    -------
    (rrha_matrix, S_first_2order)
        ``rrha_matrix`` is the rank-reduced Hankel approximation.
        ``S_first_2order`` are the singular values from the last SVD step.

    Raises
    ------
    RuntimeError
        If the iteration does not converge within ``max_iter`` steps.
    """
    rrha_matrix = matrix.copy()
    S_out = None

    for _ in range(max_iter):
        reduced, S_out = svd_reduction(rrha_matrix, order=order)
        rrha_matrix = hankel_approximation(reduced)
        if rrha_stop_criterion(rrha_matrix, order=order, rtol=rtol):
            return rrha_matrix, S_out

    raise RuntimeError(
        f"RRHA did not converge within {max_iter} iterations "
        f"(order={order}, matrix shape={matrix.shape}).  "
        "Increase max_iter or check signal quality."
    )


def get_mpm_poles_from_hankel_rrha(
    Y: np.ndarray,
    dt: float,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract poles from a (RRHA-reduced) Hankel matrix.

    Implements the Matrix Pencil on the SVD of ``Y``, using ``pinv`` for the
    pencil inversion (matches ``object_oriented_poles.py``).

    Parameters
    ----------
    Y:
        Hankel matrix (may be rank-reduced via RRHA).
    dt:
        Time step in µs.
    order:
        Number of poles to extract.

    Returns
    -------
    (poles, singular_values)
        ``poles``: complex continuous-time poles, shape ``(order,)``.
        ``singular_values``: all singular values of ``Y``.
    """
    from scipy.linalg import pinv, eigvals

    U, S, Vh = svd(Y, full_matrices=False)
    V = Vh.T
    V_prime = V[:, :order]
    V1 = V_prime[:-1, :]
    V2 = V_prime[1:, :]
    V1_inv = pinv(V1)
    z_poles = eigvals(V1_inv @ V2)
    # Filter out near-zero discrete poles (numerical noise)
    z_poles = z_poles[np.abs(z_poles) > 1e-9]
    s_poles = np.log(z_poles.astype(complex)) / dt
    return s_poles, S


# ---------------------------------------------------------------------------
# Public entry point (mirrors the old get_mpm_poles_from_rrha_matrix)
# ---------------------------------------------------------------------------

def mpm_rrha(
    t: np.ndarray,
    y: np.ndarray,
    order: int = 3,
    *,
    L: int | None = None,
    max_iter: int = 5,
    rtol: float = 1e-1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract poles using the RRHA-enhanced Matrix Pencil Method.

    This is a drop-in replacement for ``matrix_pencil_method``, with the same
    return signature ``(poles, singular_values, L_used)``.

    Parameters
    ----------
    t:
        Time axis (uniformly spaced, µs).
    y:
        Signal values, same length as ``t``.
    order:
        Number of poles to extract.
    L:
        Pencil parameter.  Default: ``N // 3`` (matches old code).
    max_iter:
        Maximum RRHA iterations.  Raises ``RuntimeError`` on non-convergence.
    rtol:
        RRHA stop criterion tolerance.

    Returns
    -------
    poles:
        Complex continuous-time poles, shape ``(order,)``.
    singular_values:
        First ``2*order`` singular values from the last RRHA step.
    L_used:
        The pencil parameter actually used.

    Raises
    ------
    RuntimeError
        If RRHA does not converge within ``max_iter`` iterations.
    """
    from mpm_analysis.analysis.pole_sorting import sort_poles_canonical, enforce_pole_count

    N = len(y)
    if L is None:
        L = N // 3
    L = max(order, min(L, N - order - 1))

    dt = float(t[1] - t[0])
    matrix = build_hankel_rrha(y, L)
    rrha_matrix, singular_values = get_approximated_reduced_hankel(
        matrix, order=order, max_iter=max_iter, rtol=rtol
    )
    poles, _ = get_mpm_poles_from_hankel_rrha(rrha_matrix, dt, order=order)
    poles = sort_poles_canonical(enforce_pole_count(poles, order))
    return poles, singular_values, L


# Alias matching the old method name
get_mpm_poles_from_rrha_matrix = mpm_rrha
