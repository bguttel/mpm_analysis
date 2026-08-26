"""Round-trip tests for the shifted-repetition reconstruction.

Generates a known fine-bin click-probability curve ``q_f``, forms the coarse
per-readout probabilities ``p[j, m]`` it implies exactly, and checks that the
reconstruction recovers ``q_f`` and ``P(n, t)``.  Also checks behaviour under
binomial shot noise at the planned budget.
"""
from __future__ import annotations

import numpy as np
import pytest

from mpm_analysis.io.simulated.shifted_resolution import (
    fine_click_probs,
    no_click_curve,
    poisson_binomial_pnt,
    reconstruct,
)

N_SHIFTS = 18
N_COARSE = 40


def true_q(n_fine: int, *, flat_head: bool = False) -> np.ndarray:
    """A smooth, decaying fine-bin click probability with some structure.

    ``flat_head`` makes the first coarse bin constant, which is exactly the
    condition under which Step 3's anchor assumption is not an assumption at all.
    """
    f = np.arange(n_fine, dtype=float)
    q = 0.012 * (1.0 + 0.4 * np.sin(f / 37.0)) * np.exp(-f / 900.0)
    if flat_head:
        q[:N_SHIFTS] = q[0]
    return q


def exact_p(q: np.ndarray, n_shifts: int, n_coarse: int) -> np.ndarray:
    """The p[j, m] that ``q`` implies exactly (Step 2, forward direction).

    Readout m of shift j covers fine bins [j + n_shifts*m, ... + n_shifts).
    """
    log1m_q = np.log1p(-q)
    p = np.empty((n_shifts, n_coarse), dtype=float)
    for j in range(n_shifts):
        for m in range(n_coarse):
            lo = j + n_shifts * m
            p[j, m] = -np.expm1(log1m_q[lo : lo + n_shifts].sum())
    return p


@pytest.fixture(scope="module")
def noiseless():
    # exact_p needs fine bins up to (n_shifts-1) + n_shifts*n_coarse.
    q = true_q(N_SHIFTS * (N_COARSE + 1))
    p = exact_p(q, N_SHIFTS, N_COARSE)
    return q, p


@pytest.fixture(scope="module")
def noiseless_flat_head():
    q = true_q(N_SHIFTS * (N_COARSE + 1), flat_head=True)
    p = exact_p(q, N_SHIFTS, N_COARSE)
    return q, p


def test_window_coverage_is_a_partition(noiseless):
    """Every fine index is the left edge of exactly one (j, m) window."""
    edges = sorted(
        j + N_SHIFTS * m for j in range(N_SHIFTS) for m in range(N_COARSE)
    )
    assert edges == list(range(N_SHIFTS * N_COARSE))


def test_no_click_curve_matches_ground_truth(noiseless):
    """Step 2-3: N(F) is the running product of (1 - q_f)."""
    q, p = noiseless
    n_fine = N_SHIFTS * N_COARSE
    got = no_click_curve(p)
    want = np.concatenate([[1.0], np.cumprod(1.0 - q[:n_fine])])
    # The first coarse bin rests on the flat-rate assumption of Step 3, so only
    # its 18 interior points are approximate; boundaries are exact throughout.
    assert got.size == n_fine + 1
    np.testing.assert_allclose(
        got[N_SHIFTS::N_SHIFTS], want[N_SHIFTS::N_SHIFTS], rtol=1e-12
    )


def test_recovers_q_exactly_when_first_bin_is_flat(noiseless_flat_head):
    """Step 4 inverts Step 2 exactly when Step 3's anchor assumption holds.

    A flat first coarse bin makes ``N(j) = (1-p[0,0])**(j/18)`` exact, so every
    chain starts on its true value and the whole fine grid is recovered.
    """
    q, p = noiseless_flat_head
    got = fine_click_probs(p)
    np.testing.assert_allclose(
        got, q[: N_SHIFTS * N_COARSE], rtol=1e-9, atol=1e-12
    )


def test_anchor_error_stays_inside_coarse_bins(noiseless):
    """With a non-flat first bin, the distortion does not leak across boundaries.

    Step 3's assumption puts a constant factor on each chain, so individual
    ``q_f`` are wrong within a coarse bin — but every bin contains one full turn
    through all 18 chains, so the factors telescope and each bin's total no-click
    probability is exact.  That total is what P(n, t) is built from.
    """
    q, p = noiseless
    got = fine_click_probs(p)

    # Per-bin totals: exact.
    got_tot = np.log1p(-got).reshape(N_COARSE, N_SHIFTS).sum(axis=1)
    want_tot = np.log1p(
        -q[: N_SHIFTS * N_COARSE]
    ).reshape(N_COARSE, N_SHIFTS).sum(axis=1)
    np.testing.assert_allclose(got_tot, want_tot, rtol=1e-10)

    # Individual bins: distorted, and visibly so — this is the documented limit.
    assert np.max(np.abs(got - q[: N_SHIFTS * N_COARSE])) > 1e-4


def test_first_coarse_bin_is_flattened_but_totals_agree(noiseless):
    """Step 3's assumption is confined to the first bin, and preserves its total."""
    q, p = noiseless
    got = fine_click_probs(p)
    head = got[:N_SHIFTS]
    assert np.allclose(head, head[0]), "first bin should come out flat by assumption"
    # ...and its no-click product still equals the directly measured value.
    assert np.log1p(-head).sum() == pytest.approx(np.log1p(-p[0, 0]), rel=1e-12)


def test_pnt_is_normalized(noiseless):
    """Sum_n P(n, t) == 1 at every coarse time."""
    q, _ = noiseless
    pnt = poisson_binomial_pnt(
        q[: N_SHIFTS * N_COARSE], n_shifts=N_SHIFTS, drop_first=False
    )
    np.testing.assert_allclose(pnt.sum(axis=0), 1.0, rtol=0, atol=1e-12)
    assert np.all(pnt >= 0.0)


def test_pnt_mean_matches_sum_of_q(noiseless):
    """Mean count through coarse bin M is the sum of that prefix's q_f."""
    q, _ = noiseless
    n_fine = N_SHIFTS * N_COARSE
    pnt = poisson_binomial_pnt(q[:n_fine], n_shifts=N_SHIFTS, drop_first=False)
    n_arr = np.arange(pnt.shape[0], dtype=float)
    got = n_arr @ pnt
    want = np.cumsum(q[:n_fine])[N_SHIFTS - 1 :: N_SHIFTS]
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_pnt_variance_matches_poisson_binomial(noiseless):
    """Second moment too, so the fold is not merely mean-preserving."""
    q, _ = noiseless
    n_fine = N_SHIFTS * N_COARSE
    qq = q[:n_fine]
    pnt = poisson_binomial_pnt(qq, n_shifts=N_SHIFTS, drop_first=False)
    n_arr = np.arange(pnt.shape[0], dtype=float)
    mean = n_arr @ pnt
    var = (n_arr**2) @ pnt - mean**2
    want = np.cumsum(qq * (1.0 - qq))[N_SHIFTS - 1 :: N_SHIFTS]
    np.testing.assert_allclose(var, want, rtol=1e-8)


def test_single_coin_limit():
    """n_shifts = 1 degenerates to 'each readout is its own fine bin'."""
    p = np.array([[0.1, 0.2, 0.3]])
    q = fine_click_probs(p)
    np.testing.assert_allclose(q, [0.1, 0.2, 0.3], rtol=1e-12)


def test_drop_first_removes_exactly_one_column(noiseless):
    q, _ = noiseless
    qq = q[: N_SHIFTS * N_COARSE]
    full = poisson_binomial_pnt(qq, n_shifts=N_SHIFTS, drop_first=False)
    cut = poisson_binomial_pnt(qq, n_shifts=N_SHIFTS, drop_first=True)
    assert cut.shape[1] == full.shape[1] - 1
    np.testing.assert_array_equal(cut, full[:, 1:])


def test_reconstruct_pkt_is_dft_of_pnt(noiseless):
    """Step 6 consistency, and P(k=0, t) == 1 by normalization."""
    q, p = noiseless
    n_trials = 300_000
    clicks = np.rint(p * n_trials).astype(np.int64)
    k_values = np.linspace(0.01, 2.0 * np.pi - 0.01, 32)
    out = reconstruct(clicks, n_trials, k_values=k_values)

    n_arr = np.arange(out["pnt"].shape[0], dtype=float)
    want = np.exp(1j * np.outer(k_values, n_arr)) @ out["pnt"].astype(complex)
    # pkt is computed from the float64 pnt and only stored as float32, so the
    # agreement is limited by float32 rounding of ~2000 summands.
    np.testing.assert_allclose(out["pkt"], want, rtol=0, atol=1e-6)

    at_k0 = np.exp(1j * 0.0 * n_arr) @ out["pnt"].astype(complex)
    np.testing.assert_allclose(at_k0.real, 1.0, atol=1e-6)


def test_noise_guard_keeps_probabilities_valid_and_normalized(noiseless):
    """Under shot noise: q stays in [0, 1] and coarse-bin totals stay measured."""
    q, p = noiseless
    rng = np.random.default_rng(0)
    n_trials = 300_000
    p_noisy = rng.binomial(n_trials, p) / n_trials

    q_hat = fine_click_probs(p_noisy)
    assert np.all(q_hat >= 0.0) and np.all(q_hat < 1.0)

    # Each coarse bin's no-click product is pinned to its directly measured value.
    per_bin = np.log1p(-q_hat).reshape(N_COARSE, N_SHIFTS).sum(axis=1)
    np.testing.assert_allclose(per_bin, np.log1p(-p_noisy[0, :]), rtol=1e-9)

    pnt = poisson_binomial_pnt(q_hat, n_shifts=N_SHIFTS, drop_first=True)
    np.testing.assert_allclose(pnt.sum(axis=0), 1.0, atol=1e-12)


def test_noise_bias_on_coarse_mean_is_small(noiseless):
    """The coarse-grid mean count survives shot noise at the planned budget.

    The per-bin q_f curve is noisy, but each coarse bin's total is pinned to a
    measured number, so the quantity that feeds P(n,t) is far more robust.
    """
    q, p = noiseless
    rng = np.random.default_rng(1)
    n_trials = 300_000

    errs = []
    for seed_shift in range(5):
        rng = np.random.default_rng(seed_shift)
        p_noisy = rng.binomial(n_trials, p) / n_trials
        q_hat = fine_click_probs(p_noisy)
        got = np.cumsum(q_hat)[N_SHIFTS - 1 :: N_SHIFTS]
        want = np.cumsum(q[: N_SHIFTS * N_COARSE])[N_SHIFTS - 1 :: N_SHIFTS]
        errs.append(np.abs(got - want)[-1] / want[-1])

    assert np.mean(errs) < 0.02, f"relative error on total count: {errs}"
