"""Tests for the data-resampling bootstrap (measurement statistics).

Covers:
- multinomial ens_avg resampling (sum-to-one, correct moments, correlations)
- binomial post_selected resampling (shared survival draw across channels)
- run_data_bootstrap end-to-end (serial + parallel reproducibility)
- measured best fit preserved as the central value
- discriminant Delta = c1**2 - 4*c0 per-realization crossing
"""
from __future__ import annotations

import numpy as np
import pytest

from mpm_analysis.analysis.bootstrap import (
    bootstrap_ens_avg_measurements,
    bootstrap_post_selected_measurements,
    run_data_bootstrap,
    aggregate_bootstrap,
)
from mpm_analysis.analysis.ksi_crossing import (
    discriminant_from_params,
    discriminant_crossing_samples,
)
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.data_types.pole_samples import PoleSamples


# ---------------------------------------------------------------------------
# ens_avg: multinomial resampling
# ---------------------------------------------------------------------------

def _ens_avg_stats(n_t=40, n_shots=3000):
    rng = np.random.default_rng(0)
    D = 0.5 + 0.3 * np.exp(-np.linspace(0, 3, n_t))
    G = 0.25 * (1 - np.exp(-np.linspace(0, 3, n_t)))
    B = 0.1 * np.ones(n_t)
    E = 1.0 - D - G - B
    return {
        "data_mode": "ens_avg",
        "P": np.vstack([D, G, B, E]),
        "channels": ["D", "G", "B", "error"],
        "N_shots": n_shots,
    }


def test_ens_avg_resample_sums_to_one():
    stats = _ens_avg_stats()
    rng = np.random.default_rng(1)
    Y = bootstrap_ens_avg_measurements(stats, ["D", "G", "B", "error"], rng)
    assert Y.shape == (4, 40)
    assert np.allclose(Y.sum(axis=0), 1.0)          # multinomial constraint
    # counts are multiples of 1/N
    assert np.allclose(Y * stats["N_shots"], np.round(Y * stats["N_shots"]))


def test_ens_avg_resample_moments_and_correlation():
    stats = _ens_avg_stats()
    rng = np.random.default_rng(2)
    n_rep = 3000
    draws = np.stack([
        bootstrap_ens_avg_measurements(stats, ["D", "G"], rng)
        for _ in range(n_rep)
    ])                                              # (n_rep, 2, N)
    N = stats["N_shots"]
    P = stats["P"]
    t0 = 20    # mid-scan: all four channels have non-zero probability
    # Mean ≈ P, Var ≈ P(1-P)/N (binomial marginal of the multinomial)
    assert abs(draws[:, 0, t0].mean() - P[0, t0]) < 5e-3
    var_expected = P[0, t0] * (1 - P[0, t0]) / N
    assert 0.7 * var_expected < draws[:, 0, t0].var() < 1.3 * var_expected
    # D and G share the shots → negative correlation, never independent
    c = np.corrcoef(draws[:, 0, t0], draws[:, 1, t0])[0, 1]
    assert c < -0.1


def test_ens_avg_Z_is_D_minus_G():
    stats = _ens_avg_stats()
    Y1 = bootstrap_ens_avg_measurements(stats, ["D", "G", "Z"],
                                        np.random.default_rng(3))
    assert np.allclose(Y1[2], Y1[0] - Y1[1])


def test_ens_avg_unknown_key_raises():
    with pytest.raises(KeyError):
        bootstrap_ens_avg_measurements(_ens_avg_stats(), ["survival"],
                                       np.random.default_rng(0))


# ---------------------------------------------------------------------------
# post_selected: binomial resampling with shared survival
# ---------------------------------------------------------------------------

def _post_selected_stats(n_t=50):
    t = np.linspace(0, 10, n_t)
    return {
        "data_mode": "post_selected",
        "survival": np.exp(-0.25 * t),
        "X": 0.6 * np.cos(1.1 * t),
        "Z": 0.7 * np.exp(-0.1 * t),
        "N_survival": 180_000,
        "N_X": np.maximum((60_000 * np.exp(-0.25 * t)).astype(int), 1),
        "N_Z": np.maximum((60_000 * np.exp(-0.25 * t)).astype(int), 1),
    }


def test_post_selected_shared_survival_draw():
    # With effectively infinite tomography statistics, X* → X and Z* → Z, so
    # rows "X" and "Z" reveal the survival draw: same s_boot in all channels.
    stats = _post_selected_stats()
    stats["N_X"] = np.full_like(stats["N_X"], 10**9)
    stats["N_Z"] = np.full_like(stats["N_Z"], 10**9)
    Y = bootstrap_post_selected_measurements(
        stats, ["survival", "X", "Z"], np.random.default_rng(4)
    )
    s_boot = Y[0]
    assert np.allclose(Y[1], s_boot * stats["X"], atol=2e-4)
    assert np.allclose(Y[2], s_boot * stats["Z"], atol=2e-4)
    # And the survival draw itself differs from the measured curve (resampled)
    assert not np.allclose(s_boot, stats["survival"])


def test_post_selected_survival_variance():
    stats = _post_selected_stats()
    rng = np.random.default_rng(5)
    t0 = 10
    s = stats["survival"][t0]
    draws = np.array([
        bootstrap_post_selected_measurements(stats, ["survival"], rng)[0, t0]
        for _ in range(4000)
    ])
    var_expected = s * (1 - s) / stats["N_survival"]
    assert 0.8 * var_expected < draws.var() < 1.2 * var_expected


def _per_experiment_stats(n_t=50, navg=30_000):
    """New-style stats: each tomography run has its own survival + attempts."""
    t = np.linspace(0, 10, n_t)
    n_phase = np.maximum((navg * 2 * np.exp(-0.25 * t)).astype(int), 1)
    n_z = np.maximum((navg * 2 * np.exp(-0.30 * t)).astype(int), 1)
    return {
        "data_mode": "post_selected",
        "survival": np.exp(-0.25 * t),
        "N_survival": navg * 6,
        "SX": n_phase / (navg * 2),
        "N_attempts_SX": navg * 4,
        "SZ": n_z / (navg * 2),
        "N_attempts_SZ": navg * 2,
        "X": 0.6 * np.cos(1.1 * t),
        "Y": 0.2 * np.sin(1.1 * t),
        "Z": 0.7 * np.exp(-0.1 * t),
        "N_X": n_phase, "N_Y": n_phase, "N_Z": n_z,
    }


def test_products_use_their_own_experiment_survival():
    """X must carry SX, Z must carry SZ — never the dedicated survival."""
    stats = _per_experiment_stats()
    # infinite tomography statistics -> V* == V, so the product reveals S_exp*
    stats["N_X"] = np.full_like(stats["N_X"], 10 ** 9)
    stats["N_Z"] = np.full_like(stats["N_Z"], 10 ** 9)
    keys = ["S", "SX", "SZ", "X", "Z"]
    Y = bootstrap_post_selected_measurements(stats, keys,
                                             np.random.default_rng(11))
    S_b, SX_b, SZ_b, X_b, Z_b = Y
    # the product equals its own survival draw times the Bloch value
    assert np.allclose(X_b, SX_b * stats["X"], atol=2e-4)
    assert np.allclose(Z_b, SZ_b * stats["Z"], atol=2e-4)
    # and NOT the dedicated survival (the old bug)
    assert not np.allclose(X_b, S_b * stats["X"], atol=2e-4)
    # the three survival draws are independent experiments
    assert not np.allclose(SX_b, SZ_b)
    assert not np.allclose(S_b, SX_b)


def test_survival_attempt_counts_set_the_variance():
    """S/SX/SZ are binomial on navg*6 / navg*4 / navg*2 attempts."""
    stats = _per_experiment_stats()
    rng = np.random.default_rng(12)
    idx = 5
    draws = {k: [] for k in ("S", "SX", "SZ")}
    for _ in range(3000):
        y = bootstrap_post_selected_measurements(stats, ["S", "SX", "SZ"], rng)
        for k, row in zip(("S", "SX", "SZ"), y):
            draws[k].append(row[idx])
    for key, frac_field, n_field in (("S", "survival", "N_survival"),
                                     ("SX", "SX", "N_attempts_SX"),
                                     ("SZ", "SZ", "N_attempts_SZ")):
        p = stats[frac_field][idx]
        expected = p * (1 - p) / stats[n_field]
        got = np.var(draws[key])
        assert 0.75 * expected < got < 1.3 * expected, key
    # twice the attempts -> half the variance
    assert 1.5 < np.var(draws["SZ"]) / np.var(draws["SX"]) < 2.7


def test_survival_draw_shared_between_channel_and_product():
    """A run fitting [SX, X] must use one SX draw for both."""
    stats = _per_experiment_stats()
    stats["N_X"] = np.full_like(stats["N_X"], 10 ** 9)
    Y = bootstrap_post_selected_measurements(stats, ["SX", "X"],
                                             np.random.default_rng(13))
    assert np.allclose(Y[1], Y[0] * stats["X"], atol=2e-4)


def test_Y_shares_the_phase_axis_survival():
    stats = _per_experiment_stats()
    stats["N_X"] = np.full_like(stats["N_X"], 10 ** 9)
    stats["N_Y"] = np.full_like(stats["N_Y"], 10 ** 9)
    Y = bootstrap_post_selected_measurements(stats, ["SX", "X", "Y"],
                                             np.random.default_rng(14))
    assert np.allclose(Y[1], Y[0] * stats["X"], atol=2e-4)
    assert np.allclose(Y[2], Y[0] * stats["Y"], atol=2e-4)


def test_legacy_stats_fall_back_to_dedicated_survival():
    """Old stats dicts (no SX/SZ) still work, sharing one survival draw."""
    stats = _post_selected_stats()
    stats["N_X"] = np.full_like(stats["N_X"], 10 ** 9)
    stats["N_Z"] = np.full_like(stats["N_Z"], 10 ** 9)
    Y = bootstrap_post_selected_measurements(stats, ["survival", "X", "Z"],
                                             np.random.default_rng(15))
    assert np.allclose(Y[1], Y[0] * stats["X"], atol=2e-4)
    assert np.allclose(Y[2], Y[0] * stats["Z"], atol=2e-4)


def test_post_selected_zero_shots_keeps_value():
    stats = _post_selected_stats()
    stats["N_X"] = np.zeros_like(stats["N_X"])
    Y = bootstrap_post_selected_measurements(
        stats, ["survival", "X"], np.random.default_rng(6)
    )
    # X untouched where N_X == 0 (only survival factor resampled)
    assert np.allclose(Y[1] / np.maximum(Y[0], 1e-12), stats["X"], atol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end data bootstrap
# ---------------------------------------------------------------------------

def _synthetic_records(n_lambda=3, n_t=80):
    """post_selected-like records with attached bootstrap_stats."""
    t = np.linspace(0, 15, n_t)
    records = []
    for i in range(n_lambda):
        lam = 0.5 + 0.3 * i
        poles = np.array([-0.6 - 0.1 * i, -0.12 + 0.35j, -0.12 - 0.35j])
        s = np.real(0.6 * np.exp(poles[0] * t)
                    + 0.2 * np.exp(poles[1] * t) + 0.2 * np.exp(poles[2] * t))
        s = np.clip(s, 1e-4, 1.0)
        X = np.clip(0.5 * np.cos(0.35 * t), -1, 1)
        Z = np.clip(0.6 * np.exp(-0.05 * t), -1, 1)
        stats = {
            "data_mode": "post_selected",
            "survival": s, "X": X, "Z": Z,
            "N_survival": 180_000,
            "N_X": np.full(n_t, 30_000),
            "N_Z": np.full(n_t, 30_000),
        }
        meta = {"bootstrap_stats": stats}
        for key, sig in [("survival", s), ("X", s * X), ("Z", s * Z)]:
            records.append(ObservableRecord(
                lambda_val=lam, t=t, signal=sig,
                observable_key=key, metadata=meta,
            ))
    return records


def test_run_data_bootstrap_end_to_end_and_center():
    records = _synthetic_records()
    keys = ["survival", "Z", "X"]
    results = run_data_bootstrap(
        records, order=3, n_boot=15, observable_key=keys,
        seed=7, n_jobs=1,
        refine_settings={"conjugate_pairs": True,
                         "bounds_real": None, "bounds_imag": None},
    )
    assert len(results) == 3
    for br in results:
        assert np.isfinite(br.boot_real).any()
        assert br.boot_params is not None
        assert np.isfinite(br.boot_params).any()

    # Central value = measured best fit, not the bootstrap median
    center = [np.array([-0.6 - 0.1 * i, -0.12 - 0.35j, -0.12 + 0.35j])
              for i in range(3)]
    poles_result = aggregate_bootstrap(results, observable_key="survival+Z+X",
                                       center_poles=center)
    for i in range(3):
        assert np.allclose(poles_result.decay_rates[i],
                           -center[i].real, atol=1e-12)
        assert np.allclose(poles_result.frequencies[i],
                           center[i].imag, atol=1e-12)
    assert poles_result.raw_samples.param_samples is not None


def test_run_data_bootstrap_reproducible_and_parallel():
    records = _synthetic_records(n_lambda=2, n_t=60)
    kwargs = dict(order=3, n_boot=6, observable_key=["survival", "Z", "X"],
                  seed=42)
    a = run_data_bootstrap(records, n_jobs=1, **kwargs)
    b = run_data_bootstrap(records, n_jobs=1, **kwargs)
    c = run_data_bootstrap(records, n_jobs=2, **kwargs)
    for x, y in [(a, b), (a, c)]:
        for bx, by in zip(x, y):
            assert np.allclose(bx.boot_real, by.boot_real, equal_nan=True)
            assert np.allclose(bx.boot_imag, by.boot_imag, equal_nan=True)


def test_run_data_bootstrap_requires_stats():
    records = _synthetic_records(n_lambda=1)
    for r in records:
        r.metadata = {}
    with pytest.raises(ValueError, match="bootstrap_stats"):
        run_data_bootstrap(records, order=3, n_boot=2,
                           observable_key=["survival", "Z", "X"], n_jobs=1)


# ---------------------------------------------------------------------------
# Discriminant transition
# ---------------------------------------------------------------------------

def test_discriminant_from_params():
    # order 3: params = [r, c1, c0]
    p = np.array([-0.5, 0.3, -0.1])
    assert np.isclose(discriminant_from_params(p, 3), 0.3**2 + 0.4)


def test_discriminant_crossing_samples():
    # Delta(lambda) crosses zero at lambda = 0.6 in every realization
    lams = np.linspace(0, 1, 41)
    n_boot = 200
    rng = np.random.default_rng(8)
    params = np.empty((len(lams), n_boot, 3))
    delta_true = lams - 0.6                       # target discriminant
    for b in range(n_boot):
        c1 = 1.0 + 0.01 * rng.standard_normal(len(lams))
        c0 = (c1**2 - (delta_true + 0.01 * rng.standard_normal(len(lams)))) / 4
        params[:, b, 0] = -0.5
        params[:, b, 1] = c1
        params[:, b, 2] = c0
    samples = PoleSamples(
        samples=np.zeros((len(lams), n_boot, 3), dtype=complex),
        observable_key="survival", lambda_values=lams,
        param_samples=params,
    )
    crossings, d_med, d_lo, d_hi = discriminant_crossing_samples(samples)
    q16, q50, q84 = np.nanpercentile(crossings, [16, 50, 84])
    assert abs(q50 - 0.6) < 0.02
    assert q16 < q50 < q84
    assert np.all(d_lo[np.isfinite(d_lo)] <= d_hi[np.isfinite(d_hi)])


# ---------------------------------------------------------------------------
# post_selected: segment-pooled survival (multinomial over the first-click
# histogram)
# ---------------------------------------------------------------------------

def _pooled_stats(n_seg=8, n_axes=2, n_reps=400, seed=0):
    """Synthetic segmented record → first-click histogram → stats dict."""
    from mpm_analysis.io.experimental.click_record_npz import (
        ClickRecordData, first_click_counts, pooled_survival_from_counts,
    )
    rng = np.random.default_rng(seed)
    lengths = np.arange(1, n_seg + 1)
    traj = rng.random((n_reps * n_axes, int(lengths.sum()))) < 0.1
    rec = ClickRecordData(
        lambda_val=1.0, t=np.arange(n_seg + 1) * 0.32, trajectories=traj,
        source_file="synthetic", metadata={"n_axes": n_axes},
        segment_lengths=lengths,
    )
    counts = first_click_counts(rec)
    frac, n_succ, n_att = pooled_survival_from_counts(counts)
    stats = {
        "data_mode": "post_selected",
        "survival": frac.copy(), "N_survival": n_reps * n_axes,
        "S_pooled": frac, "N_attempts_S_pooled": n_att,
        "pooled_counts": counts, "pooled_mask": None,
    }
    return rec, counts, frac, n_succ, n_att, stats


def test_first_click_counts_matches_brute_force():
    rec, counts, frac, n_succ, n_att, _ = _pooled_stats()
    o = rec.segment_offsets
    n_seg = rec.n_segments
    assert counts.shape == (2, n_seg, n_seg + 1)
    assert (counts.sum(-1) == 400).all()
    for i in range(1, n_seg + 1):
        succ = att = 0
        for L in range(i, n_seg + 1):
            seg = rec.segment(L - 1)
            succ += int((~seg[:, :i].any(axis=1)).sum())
            att += seg.shape[0]
        assert n_succ[i] == succ and n_att[i] == att
    # diagonal (own segment only) = the scan's stored definition of p_0(t_i)
    for i in range(1, n_seg + 1):
        own = ~rec.segment(i - 1).any(axis=1)
        assert counts[:, i - 1, 0].sum() == own.sum()
    assert frac[0] == 1.0 and n_att[0] == rec.N * n_seg


def test_pooled_resample_is_multinomial_and_keeps_nesting():
    _, counts, frac, _, n_att, stats = _pooled_stats()
    rng = np.random.default_rng(3)
    n_rep = 3000
    draws = np.array([
        bootstrap_post_selected_measurements(stats, ["S_pooled"], rng)[0]
        for _ in range(n_rep)
    ])
    assert draws.shape == (n_rep, len(frac))
    assert np.allclose(draws[:, 0], 1.0)                    # anchor at p = 1
    # per-point variance = binomial on the pooled attempts
    var_exp = frac[1:] * (1 - frac[1:]) / n_att[1:]
    assert np.allclose(draws[:, 1:].var(axis=0), var_exp, rtol=0.15)
    # consecutive pooled points share shots → positive correlation; the
    # per-point binomial fallback would give ~0
    c = np.corrcoef(draws[:, 1], draws[:, 2])[0, 1]
    assert c > 0.5
    stats_fb = {k: v for k, v in stats.items() if k != "pooled_counts"}
    draws_fb = np.array([
        bootstrap_post_selected_measurements(stats_fb, ["S_pooled"], rng)[0]
        for _ in range(n_rep)
    ])
    assert abs(np.corrcoef(draws_fb[:, 1], draws_fb[:, 2])[0, 1]) < 0.1


def test_pooled_resample_respects_mask_and_anchor():
    _, counts, frac, _, n_att, stats = _pooled_stats()
    mask = np.ones(len(frac), dtype=bool)
    mask[0] = False                          # drop_first_point
    mask[-2:] = False                        # t_max cut
    s = dict(stats, S_pooled=frac[mask], N_attempts_S_pooled=n_att[mask],
             pooled_mask=mask)
    rng = np.random.default_rng(4)
    y = bootstrap_post_selected_measurements(s, ["S_pooled"], rng)[0]
    assert y.shape == (int(mask.sum()),)
    assert np.all((y >= 0) & (y <= 1))
    # init_error-style anchor: the t=0 point follows the stored value
    s2 = dict(stats); s2["S_pooled"] = frac.copy(); s2["S_pooled"][0] = 0.98
    ys = np.array([bootstrap_post_selected_measurements(s2, ["S_pooled"],
                                                        rng)[0][0]
                   for _ in range(500)])
    assert abs(ys.mean() - 0.98) < 2e-3
