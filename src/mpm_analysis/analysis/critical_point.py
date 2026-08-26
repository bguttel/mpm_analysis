"""
critical_point.py
-----------------
Fit the imaginary (and real) pole splitting near the exceptional point to the
square-root model:

    Im(Δe₁₂) ≈ a · √|λ − λ_c| + b · (λ − λ_c)^1.5
    Re(Δe₁₂) ≈ a · √|λ − λ_c| + b · (λ − λ_c)^1.5   (opposite side of EP)

Scipy is imported inside functions (deferred import pattern).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpm_analysis.data_types.analysis_result import SqrtFitResult
from mpm_analysis.data_types.poles import PolesResult


# ---------------------------------------------------------------------------
# Fit model — joint real + imaginary splitting
# ---------------------------------------------------------------------------

def _joint_model(lam: np.ndarray, a: float, b: float, lambda_c: float) -> np.ndarray:
    """Joint model for curve_fit.

    ``lam`` has N points.  Returns 2N values: first N are the real splitting
    prediction (non-zero for λ > λ_c), last N are the imaginary splitting
    prediction (non-zero for λ < λ_c).

    Fits ``y = hstack([real_split_data, imag_split_data])``.
    """
    d = lam - lambda_c
    pos = np.clip(d, 0.0, None)   # λ > λ_c  → real splitting opens
    neg = np.clip(-d, 0.0, None)  # λ < λ_c  → imaginary splitting opens
    real_part = a * np.sqrt(pos) + b * pos ** 1.5
    imag_part = a * np.sqrt(neg) + b * neg ** 1.5
    return np.hstack([real_part, imag_part])


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------

def fit_sqrt_to_eigenvalues(
    poles: PolesResult,
    lambda_values: np.ndarray,
    lambda_c_guess: float,
    *,
    window_left: int = 15,
    window_right: int = 15,
    a_guess: float = 1.0,
    b_guess: float = 0.0,
    fit_real_part: bool = True,
    omega_s: float = 1.0,
    soft_bounds: bool = False,
    lambda_c_bounds: tuple[float, float] = (1.0, 1.5),
) -> SqrtFitResult:
    """Fit the sqrt model jointly to the real and imaginary pole splittings.

    Matches the approach from ``CriticalPointsAnalyzer.fit_sqrt_to_eigenvalues``
    in ``critical_points_analysis.py``:

    - Joint fit: both sides of the EP contribute to the same optimisation.
    - λ_c constrained to ``[lambda_c_guess − 0.1, lambda_c_guess + 0.1]``.
    - RMS symmetrisation for asymmetric uncertainties.
    - ``absolute_sigma=False`` (relative chi²).

    Parameters
    ----------
    poles:
        ``PolesResult`` containing splitting data.
    lambda_values:
        1-D lambda array matching ``poles`` axes.
    lambda_c_guess:
        Initial guess for the critical point λ_c.
    window_left, window_right:
        Number of lambda points to the left/right of ``lambda_c_guess``
        included in the fit window.
    a_guess, b_guess:
        Initial guesses for the amplitude and correction parameters.
    fit_real_part:
        If ``True`` (default), include the real splitting in the joint fit.
    omega_s:
        Normalisation factor (divide data by this before fitting).
    soft_bounds:
        If ``True``, replace the hard ±10 % box with a *soft* guess: only
        λ_c is bounded (to ``lambda_c_bounds``), ``a`` and ``b`` are
        essentially free, and the fit starts at the guess itself.
    lambda_c_bounds:
        ``(lower, upper)`` bounds on λ_c when ``soft_bounds=True``.

    Returns
    -------
    SqrtFitResult
    """
    from scipy.optimize import curve_fit  # deferred

    if not poles.has_splitting:
        raise ValueError(
            "poles.imag_splitting is None — run bootstrap first to compute splittings."
        )

    lambdas = np.asarray(lambda_values, dtype=float)
    eps = 1e-12

    # --- Extract and normalise splitting data ---
    imag_split = (poles.imag_splitting / omega_s).flatten()
    imag_low   = (poles.imag_splitting_lower / omega_s).flatten()
    imag_up    = (poles.imag_splitting_upper / omega_s).flatten()

    real_split = (poles.real_splitting / omega_s).flatten()
    real_low   = (poles.real_splitting_lower / omega_s).flatten()
    real_up    = (poles.real_splitting_upper / omega_s).flatten()

    # Safety: never let half-widths be zero
    imag_low = np.where(imag_low <= 0, eps, imag_low)
    imag_up  = np.where(imag_up  <= 0, eps, imag_up)
    real_low = np.where(real_low <= 0, eps, real_low)
    real_up  = np.where(real_up  <= 0, eps, real_up)

    imag_sig = np.sqrt(0.5 * (imag_low ** 2 + imag_up ** 2))
    real_sig = np.sqrt(0.5 * (real_low ** 2 + real_up ** 2))

    # Cap: sigma must not exceed the median itself
    imag_sig = np.maximum(np.minimum(imag_sig, np.abs(imag_split)), eps)
    real_sig = np.maximum(np.minimum(real_sig, np.abs(real_split)), eps)

    # --- Build window ---
    c_idx = int(np.argmin(np.abs(lambdas - lambda_c_guess)))
    lo = max(0, c_idx - window_left)
    hi = min(len(lambdas), c_idx + window_right + 1)
    mask = np.zeros(len(lambdas), dtype=bool)
    mask[lo:hi] = True

    lam_w      = lambdas[mask]
    is_w       = imag_split[mask]
    rs_w       = real_split[mask]
    is_sig_w   = imag_sig[mask]
    rs_sig_w   = real_sig[mask]

    if len(lam_w) < 4:
        raise ValueError(
            f"Not enough finite points in fit window ({len(lam_w)} < 4). "
            "Try widening window_left / window_right."
        )

    # --- Joint fitting ---
    if fit_real_part:
        y_fit     = np.hstack([rs_w, is_w])
        sigma_fit = np.hstack([rs_sig_w, is_sig_w])
        _model    = _joint_model
    else:
        # Imag-only: model returns only the imag part
        def _model(lam, a, b, lambda_c):
            neg = np.clip(lambda_c - lam, 0.0, None)
            return a * np.sqrt(neg) + b * neg ** 1.5
        y_fit     = is_w
        sigma_fit = is_sig_w

    # Sigma floor: on the closed branch the bootstrapped splitting is
    # identically zero in every draw (conjugate pair), so its half-widths
    # collapse to ~1e-12 and those points would get ~infinite weight —
    # forcing the model to exactly zero there and collapsing a soft-bounds
    # fit to a = 0.  Floor every sigma at the smallest genuinely nonzero
    # uncertainty in the window.
    positive = sigma_fit[sigma_fit > 1e-9]
    if positive.size:
        sigma_fit = np.maximum(sigma_fit, positive.min())

    if soft_bounds:
        # Soft guess: only λ_c is bounded; a and b are essentially free and
        # the optimizer starts at the guess itself.
        lc_lo, lc_hi = (float(lambda_c_bounds[0]), float(lambda_c_bounds[1]))
        p0     = [float(a_guess), float(b_guess),
                  float(np.clip(lambda_c_guess, lc_lo, lc_hi))]
        bounds = ([-1e3, -1e3, lc_lo], [1e3, 1e3, lc_hi])
    else:
        a_lo  = 0.9 * float(a_guess)
        a_hi  = 1.1 * float(a_guess)
        b_lo  = min(0.9 * float(b_guess), 1.1 * float(b_guess))
        b_hi  = max(0.9 * float(b_guess), 1.1 * float(b_guess))
        lc_lo = float(lambda_c_guess - 0.1)
        lc_hi = float(lambda_c_guess + 0.1)

        # Start at the lower end so the optimizer has downward room within the band
        p0     = [a_lo, b_lo, float(lambda_c_guess)]
        bounds = ([a_lo, b_lo, lc_lo], [a_hi, b_hi, lc_hi])

    try:
        popt, pcov = curve_fit(
            _model, lam_w, y_fit,
            p0=p0, sigma=sigma_fit,
            absolute_sigma=False,
            bounds=bounds,
            max_nfev=50_000,
        )
    except Exception as exc:
        raise RuntimeError(f"curve_fit failed: {exc}") from exc

    a_opt, b_opt, lambda_c_opt = popt
    perr = np.sqrt(np.diag(pcov))
    a_err, b_err, lc_err = perr

    # --- Goodness of fit ---
    residuals = y_fit - _model(lam_w, *popt)
    chi2_val  = float(np.sum((residuals / sigma_fit) ** 2))
    dof       = max(len(y_fit) - len(popt), 1)
    chi2_red  = chi2_val / dof
    rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res    = float(np.sum(residuals ** 2))
    ss_tot    = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # --- Smooth curves for plotting ---
    lam_plot     = np.linspace(float(lambdas.min()), float(lambdas.max()), 500)
    d_plot       = lam_plot - lambda_c_opt
    neg_plot     = np.clip(-d_plot, 0.0, None)
    pos_plot     = np.clip(d_plot,  0.0, None)
    imag_fit_curve = a_opt * np.sqrt(neg_plot) + b_opt * neg_plot ** 1.5
    real_fit_curve = a_opt * np.sqrt(pos_plot) + b_opt * pos_plot ** 1.5

    return SqrtFitResult(
        lambda_c=float(lambda_c_opt),
        a=float(a_opt),
        b=float(b_opt),
        lambda_c_err=float(lc_err),
        a_err=float(a_err),
        b_err=float(b_err),
        chi2_reduced=chi2_red,
        rmse=rmse,
        r_squared=r2,
        window_lambda_min=float(lam_w.min()),
        window_lambda_max=float(lam_w.max()),
        window_n_points=int(mask.sum()),
        window_left_points=window_left,
        window_right_points=window_right,
        lambda_data=lam_w,
        imag_split_data=is_w,
        imag_split_err=is_sig_w,
        real_split_data=rs_w,
        real_split_err=rs_sig_w,
        lambda_plot=lam_plot,
        imag_fit_curve=imag_fit_curve,
        real_fit_curve=real_fit_curve,
    )


# ---------------------------------------------------------------------------
# Per-realization sqrt fit (bootstrap transition distribution)
# ---------------------------------------------------------------------------

def fit_sqrt_per_realization(
    samples,
    *,
    lambda_c_guess: float,
    lambda_c_bounds: tuple[float, float] = (1.0, 1.5),
    window_left: int = 50,
    window_right: int = 15,
    omega_s: float = 1.0,
    fit_real_part: bool = True,
) -> np.ndarray:
    """Fit the sqrt model separately to every bootstrap realization.

    Realization ``b`` = draw ``b`` of the pole samples across ALL lambdas
    (a complete synthetic repeat of the scan).  Its splittings are fitted
    with the soft-bounds sqrt model, yielding one ``lambda_c^(b)`` per
    realization; the transition uncertainty is read directly off the
    distribution (e.g. ``np.percentile(lam_c, [16, 50, 84])``) — never by
    propagating per-lambda pole error bars into the fit.

    Parameters
    ----------
    samples:
        A ``PoleSamples`` container (draws branch-matched to the clean
        poles; the EP pair sits at column indices 0 and 1).
    lambda_c_guess, lambda_c_bounds, window_left, window_right:
        Soft-guess fit configuration (see ``fit_sqrt_to_eigenvalues``).
    omega_s:
        Normalisation factor (divide splittings by this before fitting).
    fit_real_part:
        Include the real splitting in the joint fit.

    Returns
    -------
    np.ndarray
        ``lambda_c`` samples, shape (n_boot,); NaN where a realization's fit
        failed or had too few finite points.
    """
    from scipy.optimize import curve_fit  # deferred

    lam = samples.lambda_values
    arr = samples.samples                     # (n_lambda, n_boot, order)
    n_boot = arr.shape[1]
    eps = 1e-12

    # Per-realization splittings between the EP pair (columns 0, 1)
    rs_all = np.abs(arr[:, :, 1].real - arr[:, :, 0].real) / 2.0 / omega_s
    is_all = np.abs(arr[:, :, 1].imag - arr[:, :, 0].imag) / 2.0 / omega_s

    # One shared sigma per lambda from the ensemble spread (fit weights only).
    # Floored at the smallest genuinely nonzero sigma: on the closed branch
    # the splitting is identically zero across draws (conjugate pair) and a
    # ~zero sigma would give those points ~infinite weight (a -> 0 collapse).
    with np.errstate(invalid="ignore"):
        rs_sig = np.nanstd(rs_all, axis=1)
        is_sig = np.nanstd(is_all, axis=1)
    all_sig = np.concatenate([rs_sig, is_sig])
    positive = all_sig[all_sig > 1e-9]
    floor = positive.min() if positive.size else eps
    rs_sig = np.maximum(np.nan_to_num(rs_sig), floor)
    is_sig = np.maximum(np.nan_to_num(is_sig), floor)

    # Window around the guess (same rule as fit_sqrt_to_eigenvalues)
    c_idx = int(np.argmin(np.abs(lam - lambda_c_guess)))
    lo = max(0, c_idx - window_left)
    hi = min(len(lam), c_idx + window_right + 1)
    sl = slice(lo, hi)
    lam_w = lam[sl]

    lc_lo, lc_hi = float(lambda_c_bounds[0]), float(lambda_c_bounds[1])
    p0 = [1.0, 0.0, float(np.clip(lambda_c_guess, lc_lo, lc_hi))]
    bounds = ([-1e3, -1e3, lc_lo], [1e3, 1e3, lc_hi])

    if fit_real_part:
        sigma_fit = np.hstack([rs_sig[sl], is_sig[sl]])
    else:
        def _imag_model(lam_, a, b, lambda_c):
            neg = np.clip(lambda_c - lam_, 0.0, None)
            return a * np.sqrt(neg) + b * neg ** 1.5
        sigma_fit = is_sig[sl]

    lam_c_samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        if fit_real_part:
            y_fit = np.hstack([rs_all[sl, b], is_all[sl, b]])
            model = _joint_model
        else:
            y_fit = is_all[sl, b]
            model = _imag_model
        if not np.all(np.isfinite(y_fit)) or len(lam_w) < 4:
            continue
        try:
            popt, _ = curve_fit(
                model, lam_w, y_fit,
                p0=p0, sigma=sigma_fit,
                absolute_sigma=False, bounds=bounds, max_nfev=20_000,
            )
            lam_c_samples[b] = popt[2]
        except Exception:
            pass
    return lam_c_samples


# ---------------------------------------------------------------------------
# Window calibration
# ---------------------------------------------------------------------------

@dataclass
class WindowCalibrationResult:
    """Result of a window-size sweep."""
    window_lefts: np.ndarray
    window_rights: np.ndarray
    lambda_c_values: np.ndarray   # (n_left × n_right,)
    lambda_c_errs: np.ndarray
    chi2_values: np.ndarray

    @property
    def best_window(self) -> tuple[int, int]:
        """Return (window_left, window_right) with smallest chi2."""
        idx = int(np.argmin(self.chi2_values))
        n_right = len(self.window_rights)
        return int(self.window_lefts[idx // n_right]), int(self.window_rights[idx % n_right])


def calibrate_window(
    poles: PolesResult,
    lambda_values: np.ndarray,
    lambda_c_guess: float,
    *,
    left_range: tuple[int, int] = (20, 100),
    right_range: tuple[int, int] = (5, 30),
    a_guess: float = 1.0,
    b_guess: float = 0.0,
    omega_s: float = 1.0,
) -> WindowCalibrationResult:
    """Sweep window sizes and return the one minimising χ² of the sqrt fit."""
    lefts  = np.arange(left_range[0],  left_range[1]  + 1, 5)
    rights = np.arange(right_range[0], right_range[1] + 1, 5)

    lambda_cs, lambda_c_errs, chi2s = [], [], []

    for wl in lefts:
        for wr in rights:
            try:
                fit = fit_sqrt_to_eigenvalues(
                    poles, lambda_values, lambda_c_guess,
                    window_left=int(wl), window_right=int(wr),
                    a_guess=a_guess, b_guess=b_guess,
                    omega_s=omega_s,
                )
                lambda_cs.append(fit.lambda_c)
                lambda_c_errs.append(fit.lambda_c_err)
                chi2s.append(fit.chi2_reduced)
            except Exception:
                lambda_cs.append(float("nan"))
                lambda_c_errs.append(float("nan"))
                chi2s.append(float("nan"))

    ls_flat = np.repeat(lefts, len(rights))
    rs_flat = np.tile(rights, len(lefts))

    return WindowCalibrationResult(
        window_lefts=ls_flat,
        window_rights=rs_flat,
        lambda_c_values=np.asarray(lambda_cs),
        lambda_c_errs=np.asarray(lambda_c_errs),
        chi2_values=np.asarray(chi2s),
    )
