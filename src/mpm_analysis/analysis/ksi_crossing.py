"""
ksi_crossing.py
---------------
Ksi (ξ) crossing-point detection from sorted MPM poles.

Background
~~~~~~~~~~
Near the exceptional point (EP), three poles are involved.  After canonical
sorting (real part ascending), poles at indices 0, 1, 2 correspond to:
    index 0  →  most negative real part  (fast decay, one EP pole)
    index 1  →  middle real part          (other EP pole)
    index 2  →  least negative real part  (slow / non-EP pole)

The ksi function computes a ratio between the real parts of poles 1 and 2:

    ξ(p₁, p₂) = (2·Re(p₂) − Re(p₁)) / (Re(p₁) − Re(p₂))

This quantity changes sign at the EP: negative on one side (coalescence), zero
at the EP, positive on the other side.  The zero-crossing λ_ξ is an
alternative estimate of the critical point, complementing the sqrt fit.

Pole indexing note
~~~~~~~~~~~~~~~~~~
Old code used ``pole_2`` / ``pole_3`` for indices 1 / 2 (1-indexed), matching
the notation in the literature.  This module uses 0-indexed arrays throughout.

References
~~~~~~~~~~
Literal port of ``get_ksi``, ``ksi_zero_crossings`` logic from
``object_oriented_poles.py`` lines 2053–2060.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def ksi(p1: complex, p2: complex) -> float:
    """Compute the ξ ratio from two sorted complex poles.

    Parameters
    ----------
    p1:
        Pole at index 1 (second-most decaying after canonical sort).
    p2:
        Pole at index 2 (least decaying).

    Returns
    -------
    float
        ξ = (2·Re(p₂) − Re(p₁)) / (Re(p₁) − Re(p₂))

    Notes
    -----
    Sign convention: ξ < 0 means p₁ and p₂ are separated (before EP);
    ξ → ∞ at coalescence; ξ > 0 after EP.  A zero crossing from negative to
    positive marks the EP.
    """
    denom = p1.real - p2.real + 1e-12  # guard against exact equality
    return float((2.0 * p2.real - p1.real) / denom)


# ---------------------------------------------------------------------------
# Array-level helpers
# ---------------------------------------------------------------------------

def compute_ksi_from_real_parts(
    real_parts: np.ndarray,
    *,
    pole_idx1: int = 1,
    pole_idx2: int = 2,
) -> np.ndarray:
    """Compute ξ(λ) from pole real parts.  NaN-tolerant kernel.

    Parameters
    ----------
    real_parts:
        Array of shape (n_lambda, order) with ``Re(pole)`` per λ.
    pole_idx1, pole_idx2:
        Column indices for the two poles (default 1 and 2, matching old code).

    Returns
    -------
    np.ndarray
        ξ values, shape (n_lambda,).  NaN where either pole is missing.
    """
    real_parts = np.asarray(real_parts, dtype=float)
    if real_parts.shape[1] <= pole_idx2:
        raise ValueError(
            f"real_parts has order={real_parts.shape[1]}; pole indices "
            f"{pole_idx1} and {pole_idx2} are out of range."
        )
    p1_re = real_parts[:, pole_idx1]
    p2_re = real_parts[:, pole_idx2]
    denom = p1_re - p2_re + 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        return (2.0 * p2_re - p1_re) / denom


def compute_ksi_curve(
    poles_result,
    *,
    pole_idx1: int = 1,
    pole_idx2: int = 2,
) -> np.ndarray:
    """Compute median ξ(λ) from a ``PolesResult``.

    Parameters
    ----------
    poles_result:
        A ``PolesResult`` with ``decay_rates`` of shape (n_lambda, order).
    pole_idx1, pole_idx2:
        Column indices for the two poles (default 1 and 2, matching old code).

    Returns
    -------
    np.ndarray
        ξ values, shape (n_lambda,).  Entries are NaN where data is missing.
    """
    # decay_rates = -Re(pole); Re(pole) = -decay_rates
    return compute_ksi_from_real_parts(
        -poles_result.decay_rates, pole_idx1=pole_idx1, pole_idx2=pole_idx2
    )


def find_zero_crossings(
    lambdas: np.ndarray,
    ksi_values: np.ndarray,
    *,
    direction: str = "any",
    ignore_above: float | None = None,
) -> list[float]:
    """Find λ values where ξ crosses zero.

    Uses linear interpolation between adjacent sign changes.

    Parameters
    ----------
    lambdas:
        1-D array of λ values.
    ksi_values:
        1-D array of ξ values (same length, may contain NaN).
    direction:
        ``"any"`` (default) detects sign changes in both directions;
        ``"up"`` only negative → positive (the historical behaviour);
        ``"down"`` only positive → negative.
    ignore_above:
        If given, skip segments where both endpoints exceed this magnitude —
        filters spurious "crossings" through ±∞ where the ξ *denominator*
        goes through zero (pole coalescence), which flips sign without ξ
        itself passing through zero.

    Returns
    -------
    list[float]
        Interpolated λ values of zero crossings.  Empty if none found.
    """
    crossings: list[float] = []
    lam = np.asarray(lambdas, dtype=float)
    ksi = np.asarray(ksi_values, dtype=float)

    for i in range(len(ksi) - 1):
        k0, k1 = ksi[i], ksi[i + 1]
        if np.isnan(k0) or np.isnan(k1):
            continue
        up = k0 < 0 < k1
        down = k0 > 0 > k1
        if direction == "up" and not up:
            continue
        if direction == "down" and not down:
            continue
        if not (up or down):
            continue
        if ignore_above is not None and min(abs(k0), abs(k1)) > ignore_above:
            continue  # sign flip through ±inf, not through zero
        # Linear interpolation
        lam_cross = lam[i] + (lam[i + 1] - lam[i]) * (-k0) / (k1 - k0)
        crossings.append(float(lam_cross))
    return crossings


def ksi_crossing_samples(
    samples,
    *,
    pole_idx1: int = 1,
    pole_idx2: int = 2,
    lambda_max: float | None = None,
    direction: str = "any",
    ignore_above: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bootstrap-universe ξ analysis on raw pole samples.

    For each bootstrap draw the full ξ(λ) curve is computed and its first
    zero crossing recorded — the distribution of crossings across draws is
    the bootstrap uncertainty of λ_EP (legacy precedent:
    ``object_oriented_poles.py`` ksi bootstrap).

    Parameters
    ----------
    samples:
        A ``PoleSamples`` container (``samples`` complex, shape
        (n_lambda, n_boot, order); draws are branch-matched to the clean
        poles, so a fixed column index means one branch).
    pole_idx1, pole_idx2:
        Pole indices for the ξ ratio.
    lambda_max:
        Restrict the analysis to λ ≤ lambda_max.  Masking happens here —
        slicing a ``PolesResult`` drops ``raw_samples``.
    direction, ignore_above:
        Passed to ``find_zero_crossings``.

    Returns
    -------
    (crossings, ksi_lo, ksi_hi)
        ``crossings``: (n_boot,) first zero-crossing λ per draw (NaN where a
        draw has none); ``ksi_lo``/``ksi_hi``: (n_lambda,) 16/84 percentiles
        of the per-draw ξ curves.
    """
    lam = samples.lambda_values
    arr = samples.samples  # (n_lambda, n_boot, order)
    if lambda_max is not None:
        mask = lam <= lambda_max
        lam = lam[mask]
        arr = arr[mask]

    n_boot = arr.shape[1]
    crossings = np.full(n_boot, np.nan)
    ksi_curves = np.full((n_boot, len(lam)), np.nan)
    for b in range(n_boot):
        ksi_b = compute_ksi_from_real_parts(
            arr[:, b, :].real, pole_idx1=pole_idx1, pole_idx2=pole_idx2
        )
        ksi_curves[b] = ksi_b
        cross_b = find_zero_crossings(
            lam, ksi_b, direction=direction, ignore_above=ignore_above
        )
        if cross_b:
            crossings[b] = cross_b[0]

    # All-NaN lambda slices are expected (failed draws) — summarise quietly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ksi_lo = np.nanpercentile(ksi_curves, 16.0, axis=0)
        ksi_hi = np.nanpercentile(ksi_curves, 84.0, axis=0)
    return crossings, ksi_lo, ksi_hi


def discriminant_from_params(params: np.ndarray, order: int) -> np.ndarray:
    """Discriminant ``Delta = c1**2 - 4*c0`` of the first quadratic pole pair.

    ``params`` follows the ``pack_pole_params`` layout
    ``[r..., c1_1, c0_1, ...]``; any leading axes are preserved.

    Sign convention::

        Delta < 0  ->  complex-conjugate pair
        Delta = 0  ->  coalescence / EP
        Delta > 0  ->  two real poles
    """
    params = np.asarray(params, dtype=float)
    R = order % 2
    c1 = params[..., R]
    c0 = params[..., R + 1]
    return c1 * c1 - 4.0 * c0


def discriminant_crossing_samples(
    samples,
    *,
    lambda_max: float | None = None,
    direction: str = "any",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-realization zero crossing of the discriminant ``Delta(lambda)``.

    The preferred EP diagnostic: for each bootstrap realization ``b``,
    compute ``Delta(lambda) = c1**2 - 4*c0`` from the fitted quadratic
    parameters and locate its first zero crossing — the transition between
    the oscillating (``Delta < 0``) and purely-decaying (``Delta > 0``)
    regimes.

    Parameters
    ----------
    samples:
        A ``PoleSamples`` with ``param_samples`` (n_lambda, n_boot, n_par).
    lambda_max:
        Restrict the analysis to λ ≤ lambda_max.
    direction:
        Passed to ``find_zero_crossings``.

    Returns
    -------
    (crossings, delta_med, delta_lo, delta_hi)
        ``crossings``: (n_boot,) first crossing λ per realization (NaN where
        none); ``delta_med``/``delta_lo``/``delta_hi``: (n_lambda,) median
        and 16/84 percentiles of the per-realization Delta curves.
    """
    if samples.param_samples is None:
        raise ValueError(
            "discriminant_crossing_samples: samples.param_samples is None — "
            "run a bootstrap/covariance mode that stores quadratic parameters."
        )
    lam = samples.lambda_values
    params = samples.param_samples             # (n_lambda, n_boot, n_par)
    if lambda_max is not None:
        m = lam <= lambda_max
        lam = lam[m]
        params = params[m]

    order = samples.order
    delta = discriminant_from_params(params, order)   # (n_lambda, n_boot)
    n_boot = delta.shape[1]
    crossings = np.full(n_boot, np.nan)
    for b in range(n_boot):
        cross_b = find_zero_crossings(lam, delta[:, b], direction=direction)
        if cross_b:
            crossings[b] = cross_b[0]

    # All-NaN lambda slices are expected (failed draws) — summarise quietly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        delta_med = np.nanmedian(delta, axis=1)
        delta_lo = np.nanpercentile(delta, 16.0, axis=1)
        delta_hi = np.nanpercentile(delta, 84.0, axis=1)
    return crossings, delta_med, delta_lo, delta_hi


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KsiCrossingResult:
    """Result of the ksi crossing analysis.

    Attributes
    ----------
    lambda_values:
        Lambda axis, shape (n_lambda,).
    ksi_values:
        ξ(λ), shape (n_lambda,).
    zero_crossings:
        List of λ values where ξ crosses zero (EP candidates).
    pole_idx1, pole_idx2:
        Pole indices used for the ksi computation.
    ksi_lower, ksi_upper:
        Optional 16/84 percentiles of the per-draw ξ curves, shape
        (n_lambda,).  ``None`` when no raw samples were available.
    crossing_samples:
        Optional (n_boot,) first zero-crossing λ per bootstrap draw
        (NaN where a draw has none).
    crossing_quantiles:
        Optional (16th, 50th, 84th) percentiles of ``crossing_samples``.
    """
    lambda_values: np.ndarray
    ksi_values: np.ndarray
    zero_crossings: list[float]
    pole_idx1: int = 1
    pole_idx2: int = 2
    ksi_lower: np.ndarray | None = None
    ksi_upper: np.ndarray | None = None
    crossing_samples: np.ndarray | None = None
    crossing_quantiles: tuple[float, float, float] | None = None

    @property
    def lambda_ep(self) -> float | None:
        """First zero crossing, or ``None`` if no crossing was found."""
        return self.zero_crossings[0] if self.zero_crossings else None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KsiCrossingPipeline:
    """Compute the ξ crossing from a ``ZenoAnalysisResult``.

    Parameters
    ----------
    result:
        A ``ZenoAnalysisResult`` with at least 3 poles.
    lambda_max:
        If given, restrict the analysis to λ ≤ lambda_max.
    pole_idx1, pole_idx2:
        Pole indices for the ksi ratio (default 1, 2 — matching old code).
    """

    def __init__(
        self,
        result,
        *,
        lambda_max: float | None = None,
        pole_idx1: int = 1,
        pole_idx2: int = 2,
    ) -> None:
        self._result = result
        self._lambda_max = lambda_max
        self._pole_idx1 = pole_idx1
        self._pole_idx2 = pole_idx2

    @classmethod
    def from_result(
        cls,
        result,
        **kwargs,
    ) -> "KsiCrossingPipeline":
        """Convenience factory."""
        return cls(result, **kwargs)

    def run(self) -> KsiCrossingResult:
        """Compute ξ(λ) and find zero crossings.

        Returns
        -------
        KsiCrossingResult
        """
        lam = self._result.lambda_values.copy()
        poles = self._result.poles

        if self._lambda_max is not None:
            mask = lam <= self._lambda_max
            lam = lam[mask]
            poles = poles[mask]

        ksi_vals = compute_ksi_curve(
            poles,
            pole_idx1=self._pole_idx1,
            pole_idx2=self._pole_idx2,
        )
        crossings = find_zero_crossings(lam, ksi_vals)

        if crossings:
            print(
                f"KsiCrossingPipeline: found {len(crossings)} zero crossing(s): "
                + ", ".join(f"lam_EP ~ {c:.4f}" for c in crossings)
            )
        else:
            print("KsiCrossingPipeline: no zero crossings found.")

        # Bootstrap uncertainty from the raw pole samples, when present.
        # NOTE: samples are read from the *unsliced* PolesResult — slicing a
        # PolesResult drops raw_samples — and lambda_max is applied inside
        # ksi_crossing_samples instead.
        ksi_lo = ksi_hi = cross_samples = quantiles = None
        raw = self._result.poles.raw_samples
        if raw is not None:
            cross_samples, ksi_lo, ksi_hi = ksi_crossing_samples(
                raw,
                pole_idx1=self._pole_idx1,
                pole_idx2=self._pole_idx2,
                lambda_max=self._lambda_max,
            )
            n_valid = int(np.sum(~np.isnan(cross_samples)))
            if n_valid > 0:
                q16, q50, q84 = np.nanquantile(cross_samples, [0.16, 0.5, 0.84])
                quantiles = (float(q16), float(q50), float(q84))
                frac_none = 1.0 - n_valid / len(cross_samples)
                print(
                    f"KsiCrossingPipeline (bootstrap, n_boot={len(cross_samples)}): "
                    f"lam_EP = {q50:.4f} +{q84 - q50:.4f} / -{q50 - q16:.4f}  "
                    f"(16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}; "
                    f"{frac_none:.1%} of draws had no crossing)"
                )
            else:
                print("KsiCrossingPipeline: no bootstrap draw had a zero crossing.")

        return KsiCrossingResult(
            lambda_values=lam,
            ksi_values=ksi_vals,
            zero_crossings=crossings,
            pole_idx1=self._pole_idx1,
            pole_idx2=self._pole_idx2,
            ksi_lower=ksi_lo,
            ksi_upper=ksi_hi,
            crossing_samples=cross_samples,
            crossing_quantiles=quantiles,
        )
