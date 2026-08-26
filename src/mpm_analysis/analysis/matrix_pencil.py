"""
matrix_pencil.py
----------------
Matrix Pencil Method (MPM) for extracting complex exponential poles from a
time-domain signal.

"""
from __future__ import annotations

import numpy as np
from scipy.linalg import svd, eig, pinv


# ---------------------------------------------------------------------------
# Pencil parameter selection — spectral gap criterion (matches reference)
# ---------------------------------------------------------------------------

def _optimal_pencil_parameter(y: np.ndarray, order: int) -> int:
    """Return L that maximises S[order-1]/S[order] over L in [N//3, N//2].

    This is the spectral-gap criterion from ``object_oriented_poles.py``
    ``optimize_pencil_parameter``: the gap between the last "signal" singular
    value and the first "noise" singular value is largest at the optimal L.

    ``y`` may be a single signal ``(N,)`` or ``k`` signals ``(k, N)``; in the
    latter case the gap of the joint (stacked) pencil is maximised.
    """
    N = np.atleast_2d(y).shape[1]
    L_min = max(order + 1, N // 3)
    L_max = min(N - order - 1, N // 2)

    best_L = L_min
    best_ratio = -1.0

    for L in range(L_min, L_max + 1):
        H = build_stacked_hankel(y, L)
        try:
            s = np.linalg.svd(H, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        if len(s) <= order:
            continue
        denom = s[order]
        if denom > 1e-12:
            ratio = s[order - 1] / denom
            if ratio > best_ratio:
                best_ratio = ratio
                best_L = L

    return best_L


# ---------------------------------------------------------------------------
# Core MPM
# ---------------------------------------------------------------------------

def build_hankel_matrix(y: np.ndarray, L: int) -> np.ndarray:
    """Build the Hankel data matrix Y from signal y with pencil parameter L.

    Parameters
    ----------
    y:
        1-D signal array of length N.
    L:
        Pencil parameter.  Rule of thumb: L ≈ N/3.

    Returns
    -------
    np.ndarray
        Hankel matrix of shape (N - L, L + 1).
    """
    N = len(y)
    if not 1 <= L <= N - 1:
        raise ValueError(
            f"Pencil parameter L={L} out of range for a signal of length "
            f"{N}: need 1 <= L <= {N - 1}."
        )
    rows = N - L
    cols = L + 1
    dtype = complex if np.iscomplexobj(y) else float
    H = np.zeros((rows, cols), dtype=dtype)
    for c in range(cols):
        H[:, c] = y[c: c + rows]
    return H


def build_stacked_hankel(y: np.ndarray, L: int) -> np.ndarray:
    """Vertically stack the Hankel matrices of one or more signals.

    For ``k`` signals sharing the same poles, each Hankel block factors as
    ``H_j = Z1 R_j Z2`` with a common right Vandermonde factor ``Z2``
    (Sarkar & Pereira 1995, Eq. 30), so the stacked matrix shares the row
    space of ``Z2`` and one SVD yields the common pole set.

    Parameters
    ----------
    y:
        Signal array of shape ``(N,)`` or ``(k, N)``.
    L:
        Pencil parameter.

    Returns
    -------a
    np.ndarray
        Stacked Hankel matrix of shape ``(k * (N - L), L + 1)``.
    """
    return np.vstack([build_hankel_matrix(row, L) for row in np.atleast_2d(y)])


def matrix_pencil_method(
    t: np.ndarray,
    y: np.ndarray,
    order: int,
    *,
    L: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract ``order`` complex poles from signal ``y`` sampled at times ``t``.

    Parameters
    ----------
    t:
        Time axis (uniformly spaced, units: µs).
    y:
        Signal values, shape ``(N,)``, or ``k`` signals sharing the same
        poles, shape ``(k, N)`` — the joint (multi-signal) pencil stacks
        their Hankel blocks and extracts one common pole set.
    order:
        Number of poles to extract.
    L:
        Pencil parameter.  Default: spectral-gap optimum.

    Returns
    -------
    poles:
        Complex poles z_k (1/µs).
        Shape (order,).
    singular_values:
        All singular values of the (stacked) Hankel matrix (useful for
        model-order selection).
    L_used:
        The pencil parameter actually used.
    """

    Y = np.atleast_2d(np.asarray(y))
    N = Y.shape[1]
    # A pencil of `order` needs L >= order rows of V1 and N - L >= order + 1
    # Hankel rows; both fit only when the signal has 2*order + 1 samples.
    if N < 2 * order + 1:
        raise ValueError(
            f"Signal too short for order {order}: {N} samples, need at least "
            f"{2 * order + 1}.  Fit fewer poles or keep more time points."
        )
    if L is None:
        L = _optimal_pencil_parameter(Y, order)
    L = max(order, min(L, N - order - 1))

    H = build_stacked_hankel(Y, L)

    # Truncated SVD
    U, s, Vh = svd(H, full_matrices=False)
   
    V1 = Vh[:order, :].conj().T 

    # Split V1 into V1- and V1+ (remove last / first row)
    V1_minus = V1[:-1, :]
    V1_plus  = V1[1:, :]

    # Solve the pencil  V1+ = V1- Z  via explicit pseudoinverse
    Z = pinv(V1_minus) @ V1_plus
    z_discrete, _ = eig(Z)  # discrete-time poles

    # Convert from discrete to continuous time
    dt = float(t[1] - t[0])  # µs
    poles_continuous = np.log(z_discrete.astype(complex)) / dt

    return poles_continuous, s, L


def solve_linear_amplitudes(
    t: np.ndarray,
    y: np.ndarray,
    poles: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve for the complex amplitudes A_k such that y ≈ Σ_k A_k exp(poles_k t).

    Parameters
    ----------
    t:
        Time axis in µs.
    y:
        Signal values, shape ``(N,)`` or ``(k, N)`` for per-signal amplitudes
        with shared poles.
    poles:
        Complex poles (rad/µs).  Shape (M,).
    weights:
        Optional per-point weights (``1/sigma``), same shape as ``y``.
        ``None`` → unweighted (current behaviour).

    Returns
    -------
    np.ndarray
        Complex amplitudes A_k.  Shape (M,) for 1-D input, (k, M) for 2-D.
    """
    Phi = np.exp(np.outer(t, poles))  # (N, M)
    y = np.asarray(y)
    if y.ndim == 2:
        if weights is not None:
            w = np.asarray(weights, dtype=float)
            # Row-weighted design differs per observable → solve per row.
            amps = [
                np.linalg.lstsq(w_k[:, None] * Phi, w_k * y_k.astype(complex), rcond=None)[0]
                for w_k, y_k in zip(w, y)
            ]
            return np.vstack(amps)
        amplitudes, *_ = np.linalg.lstsq(Phi, y.astype(complex).T, rcond=None)
        return amplitudes.T
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        amplitudes, *_ = np.linalg.lstsq(w[:, None] * Phi, w * y.astype(complex), rcond=None)
        return amplitudes
    amplitudes, *_ = np.linalg.lstsq(Phi, y.astype(complex), rcond=None)
    return amplitudes


def reconstruct_signal(
    t: np.ndarray,
    poles: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    """Reconstruct y(t) ≈ Σ_k A_k exp(poles_k t) (real part taken).

    ``amplitudes`` of shape (M,) → signal (N,); shape (k, M) → signals (k, N).
    """
    Phi = np.exp(np.outer(t, poles))
    amplitudes = np.asarray(amplitudes)
    if amplitudes.ndim == 2:
        return np.real(Phi @ amplitudes.T).T
    return np.real(Phi @ amplitudes)


def pack_pole_params(poles: np.ndarray) -> np.ndarray:
    """Pack complex poles into the real-quadratic parameter vector.

    Each pole *pair* is represented by the real coefficients ``(c1, c0)`` of
    ``s**2 + c1*s + c0 = 0``; leftover poles (odd order) by their real part.
    Layout: ``[r_1..r_R, c1_1, c0_1, ..., c1_P, c0_P]`` with ``P = M // 2``,
    ``R = M % 2`` — total length ``M``.

    Pairing is greedy on ``|s_i - conj(s_j)|`` (zero for true conjugate
    pairs); the representation is exact both for a conjugate pair and for two
    real poles, and the roots move smoothly through the discriminant zero —
    so a fit in these parameters does not assume pre/post critical point.
    """
    poles = np.asarray(poles, dtype=complex)
    remaining = list(range(len(poles)))
    pairs: list[tuple[int, int]] = []
    while len(remaining) >= 2:
        best = None
        for a in range(len(remaining)):
            for b in range(a + 1, len(remaining)):
                i, j = remaining[a], remaining[b]
                cost = abs(poles[i] - np.conj(poles[j]))
                if best is None or cost < best[0]:
                    best = (cost, i, j)
        _, i, j = best
        pairs.append((i, j))
        remaining.remove(i)
        remaining.remove(j)

    reals = [poles[i].real for i in remaining]
    coeffs: list[float] = []
    for i, j in pairs:
        coeffs.append(-(poles[i] + poles[j]).real)   # c1 = -(s_i + s_j)
        coeffs.append((poles[i] * poles[j]).real)    # c0 = s_i * s_j
    return np.array(reals + coeffs, dtype=float)


def unpack_pole_params(p: np.ndarray, order: int) -> np.ndarray:
    """Inverse of ``pack_pole_params``: parameter vector → complex poles.

    Each ``(c1, c0)`` pair yields the roots of ``s**2 + c1*s + c0``:
    a conjugate pair when the discriminant is negative, two real poles
    otherwise — continuous through the transition.
    """
    p = np.asarray(p, dtype=float)
    R = order % 2
    poles = [complex(r) for r in p[:R]]
    for k in range(order // 2):
        c1, c0 = p[R + 2 * k], p[R + 2 * k + 1]
        d = c1 * c1 - 4.0 * c0
        if d < 0:
            root = np.sqrt(-d) / 2.0
            poles.append(complex(-c1 / 2.0, -root))
            poles.append(complex(-c1 / 2.0, root))
        else:
            root = np.sqrt(d) / 2.0
            poles.append(complex(-c1 / 2.0 - root))
            poles.append(complex(-c1 / 2.0 + root))
    return np.array(poles, dtype=complex)


def refine_poles(
    t: np.ndarray,
    y: np.ndarray,
    poles_init: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    conjugate_pairs: bool = False,
    bounds_real: tuple[float, float] | None = None,
    bounds_imag: tuple[float, float] | None = None,
    method: str = "trf",
    return_cov: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonlinear refinement of MPM poles via residual minimisation.

    Variable projection: only the poles are free parameters; the (per-signal)
    complex amplitudes are re-solved by weighted linear least squares at every
    residual evaluation, so poles *and* residues are jointly fitted.

    Parameters
    ----------
    t, y:
        Time axis and signal(s) — ``y`` of shape ``(N,)`` or ``(k, N)`` for a
        joint fit of ``k`` observables with shared poles.
    poles_init:
        Initial poles from ``matrix_pencil_method``.
    weights:
        Optional per-point weights (``1/sigma``), same shape as ``y``.
        Applied to both the amplitude solve and the residual.
    conjugate_pairs:
        If ``True``, constrain the poles to be roots of real polynomials
        (conjugate pairs, or real poles — the fit moves smoothly between the
        two regimes).  Parameters are then ``pack_pole_params(poles_init)``.
        A non-positive ``bounds_real[1]`` is honoured via Routh-Hurwitz
        (``c1 >= -2*hi``, ``c0 >= 0``), which keeps every pole decaying and
        so keeps ``exp(t*Re)`` from overflowing; ``bounds_imag`` has no
        box equivalent here and is ignored.
    bounds_real:
        Optional ``(lower, upper)`` bounds on Re(pole).  Negative → decaying.
    bounds_imag:
        Optional ``(lower, upper)`` bounds on Im(pole).
    method:
        Scipy minimisation method passed to ``scipy.optimize.least_squares``.
    return_cov:
        If ``True``, return ``(poles, p_opt, cov_p)`` where ``p_opt`` is the
        optimal parameter vector and ``cov_p = s^2 * pinv(J^T J)`` its
        covariance with ``s^2 = 2*cost/dof`` (relative-sigma scaling, matching
        ``curve_fit(absolute_sigma=False)``; conservative when ``weights`` are
        genuine ``1/sigma``).  ``pinv`` keeps rank-deficient Jacobians (pole
        with ~zero amplitude, exceptional point) from raising.

    Returns
    -------
    np.ndarray
        Refined complex poles (same length as ``poles_init``), or the
        ``(poles, p_opt, cov_p)`` tuple when ``return_cov=True``.
    """
    from scipy.optimize import least_squares

    M = len(poles_init)
    w = None if weights is None else np.asarray(weights, dtype=float)

    if conjugate_pairs:
        p0 = pack_pole_params(np.asarray(poles_init, dtype=complex))
        unpack = lambda p: unpack_pole_params(p, M)
        lb = np.full(M, -np.inf)
        ub = np.full(M, np.inf)
        # Translate an upper bound on Re(pole) into box constraints on the
        # quadratic coefficients via Routh-Hurwitz: for s^2 + c1*s + c0 both
        # roots satisfy Re <= 0  <=>  c1 >= 0 AND c0 >= 0 (exact for degree 2;
        # c0 < 0 means one root of each sign, c1 < 0 means both positive).
        # Without this the optimiser can wander into growing exponentials,
        # where exp(t*Re) overflows to inf and every downstream lstsq gets a
        # non-finite matrix (LAPACK *LASCLS "illegal value" messages).
        if bounds_real is not None and bounds_real[1] <= 0.0:
            hi = float(bounds_real[1])
            R = M % 2
            ub[:R] = hi              # free real pole(s)
            lb[R::2] = -2.0 * hi     # c1  (= 0 for the usual hi = 0)
            lb[R + 1::2] = 0.0       # c0
            # A growing MPM start would be infeasible for least_squares
            p0 = np.clip(p0, lb, ub)
    else:
        # Pack real and imag parts into a flat parameter vector
        p0 = np.concatenate([poles_init.real, poles_init.imag])
        unpack = lambda p: p[:M] + 1j * p[M:]
        lb = np.full(2 * M, -np.inf)
        ub = np.full(2 * M,  np.inf)
        if bounds_real is not None:
            lb[:M] = bounds_real[0]
            ub[:M] = bounds_real[1]
        if bounds_imag is not None:
            lb[M:] = bounds_imag[0]
            ub[M:] = bounds_imag[1]

    def residuals(p: np.ndarray) -> np.ndarray:
        poles_c = unpack(p)
        y_hat = reconstruct_signal(
            t, poles_c, solve_linear_amplitudes(t, y, poles_c, weights=w)
        )
        r = y - y_hat if w is None else w * (y - y_hat)
        return np.asarray(r).ravel()

    result = least_squares(residuals, p0, bounds=(lb, ub), method=method)
    poles_refined = unpack(result.x)
    if not return_cov:
        return poles_refined

    J = result.jac
    dof = max(J.shape[0] - J.shape[1], 1)
    s_sq = 2.0 * result.cost / dof
    cov_p = s_sq * np.linalg.pinv(J.T @ J)
    return poles_refined, result.x, cov_p


# ---------------------------------------------------------------------------
# Optimal pencil parameter
# ---------------------------------------------------------------------------

def select_pencil_parameter(
    y: np.ndarray,
    order: int,
    *,
    L_range: tuple[int, int] | None = None,
    criterion: str = "condition_number",
) -> int:
    """Select the pencil parameter L that minimises a criterion.

    Parameters
    ----------
    y:
        Signal array of length N.
    order:
        Model order.
    L_range:
        Search range ``(L_min, L_max)``.  Default: ``(order+1, N//2)``.
    criterion:
        ``"condition_number"`` — minimise cond(H), or
        ``"residual"``         — minimise reconstruction residual (slower).

    Returns
    -------
    int
        Optimal L value.
    """
    N = len(y)
    t_dummy = np.arange(N, dtype=float)  # dummy for residual criterion
    L_min, L_max = L_range or (order + 1, N // 2)

    best_L = L_min
    best_score = np.inf

    for L in range(L_min, L_max + 1):
        H = build_hankel_matrix(y, L)
        if criterion == "condition_number":
            s = np.linalg.svd(H, compute_uv=False)
            score = s[0] / (s[order - 1] + 1e-15)
        elif criterion == "residual":
            poles, _, _ = matrix_pencil_method(t_dummy, y, order, L=L)
            amps = solve_linear_amplitudes(t_dummy, y, poles)
            y_hat = reconstruct_signal(t_dummy, poles, amps)
            score = float(np.mean((y - y_hat) ** 2))
        else:
            raise ValueError(f"Unknown criterion '{criterion}'.")
        if score < best_score:
            best_score = score
            best_L = L

    return best_L