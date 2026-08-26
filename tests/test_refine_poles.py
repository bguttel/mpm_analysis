"""Tests for the LSQ pole refinement, Hungarian matching and ksi bootstrap.

Covers the new machinery added around the MPM pipeline:
- pack/unpack of the conjugate-pair (real-quadratic) parametrisation
- Hungarian pole matching and cross-lambda branch tracking
- weighted variable-projection refinement + covariance
- direction-robust zero crossings and per-draw ksi analysis
- soft-bounds critical-point fit
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mpm_analysis.analysis.matrix_pencil import (
    matrix_pencil_method,
    pack_pole_params,
    refine_poles,
    solve_linear_amplitudes,
    unpack_pole_params,
)
from mpm_analysis.analysis.pole_sorting import (
    match_poles_hungarian,
    track_branches_across_lambda,
)
from mpm_analysis.analysis.ksi_crossing import (
    find_zero_crossings,
    ksi_crossing_samples,
)
from mpm_analysis.data_types.pole_samples import PoleSamples


# ---------------------------------------------------------------------------
# pack / unpack round trip
# ---------------------------------------------------------------------------

def _same_pole_set(a, b, tol=1e-9):
    a = np.sort_complex(np.asarray(a))
    b = np.sort_complex(np.asarray(b))
    return np.allclose(a, b, atol=tol)


def test_pack_unpack_conjugate_pair_plus_real():
    poles = np.array([-0.5 + 0.0j, -0.2 - 0.7j, -0.2 + 0.7j])
    p = pack_pole_params(poles)
    assert p.shape == (3,)
    assert _same_pole_set(unpack_pole_params(p, 3), poles)


def test_pack_unpack_three_real_poles():
    poles = np.array([-1.3 + 0.0j, -0.6 + 0.0j, -0.1 + 0.0j])
    p = pack_pole_params(poles)
    assert _same_pole_set(unpack_pole_params(p, 3), poles)


def test_pack_unpack_near_degenerate():
    # Discriminant ~ 0: conjugate pair with tiny imaginary part
    poles = np.array([-0.4 - 1e-8j, -0.4 + 1e-8j])
    p = pack_pole_params(poles)
    out = unpack_pole_params(p, 2)
    assert np.allclose(out.real, [-0.4, -0.4], atol=1e-9)
    assert np.all(np.abs(out.imag) < 1e-6)


# ---------------------------------------------------------------------------
# Hungarian matching / branch tracking
# ---------------------------------------------------------------------------

def test_match_poles_hungarian_recovers_permutation():
    ref = np.array([-1.0 + 0.5j, -0.5 - 0.3j, -0.1 + 0.0j])
    noisy = ref[[2, 0, 1]] + 1e-3 * (1 + 1j)
    matched = match_poles_hungarian(noisy, ref)
    assert np.allclose(matched, ref + 1e-3 * (1 + 1j))


def test_match_poles_hungarian_nan_tolerant():
    ref = np.array([-1.0 + 0.0j, -0.5 + 0.0j])
    poles = np.array([np.nan + 1j * np.nan, -0.99 + 0.0j])
    matched = match_poles_hungarian(poles, ref)
    # The finite pole must land next to its reference; NaN fills the rest
    assert np.isclose(matched[0], -0.99)
    assert np.isnan(matched[1])


def test_track_branches_keeps_identity_through_crossing():
    # Two branches crossing in Re but distinguished by Im: canonical
    # (Re-ascending) sorting swaps labels at the crossing; Hungarian chaining
    # (complex distance) must not.
    lams = np.linspace(0, 1, 21)
    branch_a = -1.0 + lams + 0.4j    # rises in Re, carries Im=+0.4
    branch_b = -0.5 + 0.0j * lams    # constant, real
    poles_list = []
    for a, b in zip(branch_a, branch_b):
        pair = sorted([a, b], key=lambda p: (p.real, p.imag))  # canonical feed
        poles_list.append(np.array(pair, dtype=complex))
    tracked = track_branches_across_lambda(poles_list)
    col0 = np.array([p[0] for p in tracked])
    col1 = np.array([p[1] for p in tracked])
    # Column 0 starts on branch_a (Im=0.4) and must stay on it through the
    # Re crossing at -0.5, ending near 0+0.4j.
    assert np.allclose(col0.imag, 0.4)
    assert np.isclose(col0[0], -1.0 + 0.4j)
    assert np.isclose(col0[-1], 0.0 + 0.4j)
    assert np.allclose(col1, -0.5)


# ---------------------------------------------------------------------------
# Weighted refinement + covariance
# ---------------------------------------------------------------------------

def _synthetic_two_observables(rng, sigma_noise=1e-3):
    t = np.linspace(0, 20, 200)
    poles_true = np.array([-0.8 + 0.0j, -0.15 - 0.4j, -0.15 + 0.4j])
    amps = np.array(
        [[0.5 + 0.0j, 0.25 - 0.1j, 0.25 + 0.1j],
         [0.2 + 0.0j, -0.3 + 0.2j, -0.3 - 0.2j]]
    )
    Phi = np.exp(np.outer(t, poles_true))
    Y = np.real(Phi @ amps.T).T
    Y_noisy = Y + sigma_noise * rng.standard_normal(Y.shape)
    return t, Y_noisy, poles_true


def test_refine_poles_conjugate_recovers_truth():
    rng = np.random.default_rng(0)
    t, Y, poles_true = _synthetic_two_observables(rng)
    poles0, _, _ = matrix_pencil_method(t, Y, 3)
    refined = refine_poles(t, Y, poles0, conjugate_pairs=True)
    assert _same_pole_set(refined, poles_true, tol=5e-3)
    # Conjugate structure enforced: pole set closed under conjugation
    assert _same_pole_set(refined, np.conj(refined), tol=1e-12)


def test_refine_poles_reduces_residual():
    rng = np.random.default_rng(1)
    t, Y, _ = _synthetic_two_observables(rng, sigma_noise=5e-3)
    poles0, _, _ = matrix_pencil_method(t, Y, 3)

    def rss(poles):
        from mpm_analysis.analysis.matrix_pencil import reconstruct_signal
        amps = solve_linear_amplitudes(t, Y, poles)
        return float(np.sum((Y - reconstruct_signal(t, poles, amps)) ** 2))

    refined = refine_poles(t, Y, poles0, conjugate_pairs=True)
    assert rss(refined) <= rss(poles0) * (1 + 1e-9)


def test_refine_poles_weights_downweight_corrupted_points():
    rng = np.random.default_rng(2)
    t, Y, poles_true = _synthetic_two_observables(rng, sigma_noise=1e-4)
    # Corrupt a block of points in observable 0 with huge sigma
    Y_corr = Y.copy()
    Y_corr[0, 50:70] += 0.3
    weights = np.ones_like(Y_corr)
    weights[0, 50:70] = 1e-3          # 1/sigma small → downweighted
    poles0, _, _ = matrix_pencil_method(t, Y, 3)   # init from clean data
    refined_w = refine_poles(t, Y_corr, poles0, weights=weights,
                             conjugate_pairs=True)
    refined_uw = refine_poles(t, Y_corr, poles0, conjugate_pairs=True)

    def err(poles):
        return np.abs(np.sort_complex(poles) - np.sort_complex(poles_true)).max()

    assert err(refined_w) < err(refined_uw)


def test_conjugate_refine_keeps_poles_decaying():
    """bounds_real upper limit is honoured in conjugate mode (Routh-Hurwitz).

    Without it the optimiser can reach growing exponentials, where
    exp(t*Re) overflows to inf and every downstream lstsq receives a
    non-finite matrix (LAPACK *LASCLS 'illegal value' spam).
    """
    rng = np.random.default_rng(7)
    t = np.linspace(0, 19, 60)          # same span as the real scan
    # Nasty data: pure noise, no decaying structure to latch onto
    Y = rng.standard_normal((3, len(t)))
    poles0 = np.array([2.0 + 0.0j, 1.5 + 3.0j, 1.5 - 3.0j])   # all GROWING
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)         # overflow -> error
        refined = refine_poles(t, Y, poles0, conjugate_pairs=True,
                               bounds_real=(-np.inf, 0.0))
    assert np.all(np.isfinite(refined))
    assert np.all(refined.real <= 1e-9), refined       # every pole decays
    # exp(t*Re) can no longer overflow
    assert np.max(np.outer(t, refined).real) <= 1e-6


def test_conjugate_refine_unbounded_when_no_bounds_given():
    """bounds_real=None keeps the old unconstrained behaviour."""
    rng = np.random.default_rng(8)
    t, Y, poles_true = _synthetic_two_observables(rng)
    refined = refine_poles(t, Y, matrix_pencil_method(t, Y, 3)[0],
                           conjugate_pairs=True, bounds_real=None)
    assert _same_pole_set(refined, poles_true, tol=5e-3)


def test_refine_poles_cov_is_symmetric_psd():
    rng = np.random.default_rng(3)
    t, Y, _ = _synthetic_two_observables(rng)
    poles0, _, _ = matrix_pencil_method(t, Y, 3)
    refined, p_opt, cov = refine_poles(
        t, Y, poles0, conjugate_pairs=True, return_cov=True
    )
    assert p_opt.shape == (3,)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert np.all(eigvals > -1e-12)
    # Parametric draws through unpack scatter around the refined poles
    draws = rng.multivariate_normal(p_opt, cov, size=200, method="svd")
    poles_d = np.array([unpack_pole_params(d, 3) for d in draws])
    assert np.abs(np.median(poles_d.real, axis=0)
                  - np.sort(refined.real)).max() < 0.05


# ---------------------------------------------------------------------------
# Zero crossings + ksi bootstrap
# ---------------------------------------------------------------------------

def test_find_zero_crossings_both_directions():
    lam = np.array([0.0, 1.0, 2.0, 3.0])
    down = np.array([2.0, 1.0, -1.0, -2.0])   # + → −
    up = -down
    assert find_zero_crossings(lam, down) == [1.5]
    assert find_zero_crossings(lam, up) == [1.5]
    assert find_zero_crossings(lam, down, direction="up") == []
    assert find_zero_crossings(lam, up, direction="up") == [1.5]


def test_find_zero_crossings_ignore_above():
    lam = np.array([0.0, 1.0])
    through_inf = np.array([50.0, -50.0])     # denominator sign flip
    assert find_zero_crossings(lam, through_inf, ignore_above=10.0) == []
    assert find_zero_crossings(lam, through_inf) == [0.5]


def test_ksi_crossing_samples_brackets_known_crossing():
    # Build synthetic samples: pole 1 fixed at Re=-1; pole 2 moves so that
    # ksi = (2*Re(p2) - Re(p1)) / (Re(p1) - Re(p2)) crosses zero where
    # Re(p2) = Re(p1)/2 = -0.5, i.e. at lambda = 0.5 here.
    lams = np.linspace(0, 1, 51)
    re_p2 = -1.0 + lams                        # crosses -0.5 at lam=0.5
    n_boot = 300
    rng = np.random.default_rng(4)
    samples = np.empty((len(lams), n_boot, 3), dtype=complex)
    for i, lam in enumerate(lams):
        jitter = 0.01 * rng.standard_normal(n_boot)
        samples[i, :, 0] = -2.0                # fastest branch, unused
        samples[i, :, 1] = -1.0 + 0j
        samples[i, :, 2] = re_p2[i] + jitter
    ps = PoleSamples(samples=samples, observable_key="survival",
                     lambda_values=lams)
    crossings, ksi_lo, ksi_hi = ksi_crossing_samples(ps)
    q16, q50, q84 = np.nanquantile(crossings, [0.16, 0.5, 0.84])
    assert q16 < 0.5 < q84
    assert abs(q50 - 0.5) < 0.05
    assert ksi_lo.shape == lams.shape and ksi_hi.shape == lams.shape
    assert np.all(ksi_lo[np.isfinite(ksi_lo)] <= ksi_hi[np.isfinite(ksi_hi)])


# ---------------------------------------------------------------------------
# Soft-bounds critical point fit
# ---------------------------------------------------------------------------

def test_soft_bounds_sqrt_fit_recovers_lambda_c():
    from mpm_analysis.analysis.critical_point import fit_sqrt_to_eigenvalues
    from mpm_analysis.data_types.poles import PolesResult

    rng = np.random.default_rng(5)
    lam = np.linspace(0.8, 1.6, 80)
    lam_c_true, a_true, b_true = 1.32, 0.7, 0.2
    d = lam - lam_c_true
    real_split = a_true * np.sqrt(np.clip(d, 0, None)) \
        + b_true * np.clip(d, 0, None) ** 1.5
    imag_split = a_true * np.sqrt(np.clip(-d, 0, None)) \
        + b_true * np.clip(-d, 0, None) ** 1.5
    noise = 5e-3
    real_split = real_split + noise * rng.standard_normal(len(lam))
    imag_split = imag_split + noise * rng.standard_normal(len(lam))
    err = np.full(len(lam), noise)

    poles = PolesResult(
        decay_rates=np.zeros((len(lam), 3)),
        decay_rates_lower=np.zeros((len(lam), 3)),
        decay_rates_upper=np.zeros((len(lam), 3)),
        frequencies=np.zeros((len(lam), 3)),
        frequencies_lower=np.zeros((len(lam), 3)),
        frequencies_upper=np.zeros((len(lam), 3)),
        real_splitting=real_split,
        real_splitting_lower=err,
        real_splitting_upper=err,
        imag_splitting=imag_split,
        imag_splitting_lower=err,
        imag_splitting_upper=err,
    )
    fit = fit_sqrt_to_eigenvalues(
        poles, lam, 1.25,
        window_left=60, window_right=60,
        soft_bounds=True, lambda_c_bounds=(1.0, 1.5),
    )
    assert 1.0 <= fit.lambda_c <= 1.5
    assert abs(fit.lambda_c - lam_c_true) < 0.05
