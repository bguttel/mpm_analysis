"""
Tests for the click-probability chain.

Both modes measure time from each run's own t = 0 and differ only in how many
clicks are tolerated before t (``max_clicks``):

* ``conditional``   (k = 0)    — P(n=0, t), the post-selected no-jump curve
* ``unconditional`` (k = None) — P(no click at t), full dynamics
* ``partial``       (k)        — the interpolation between them

Covered: the counting kernel against hand-computed examples, the analytic
Bernoulli limits, the k-family interpolation, termination handling and the
``min_alive_frac`` truncation, source equality (compressed == dense ==
streamed runs), the retired-name guard, ObservableRecord conversion, and the
full pipeline end to end for every source (including both simulated NPZ
formats and the shifted reconstruction).
"""
from __future__ import annotations

import numpy as np
import pytest

from mpm_analysis.analysis.click_correlations import (
    COND_CLICK,
    COND_NO_CLICK,
    PARTIAL_CLICK,
    PARTIAL_NO_CLICK,
    UNCOND_CLICK,
    UNCOND_NO_CLICK,
    _click_counts,
    _clicks_to_sparse,
    compute_all_click_correlations,
    compute_click_correlations,
    compute_click_correlations_dense,
    compute_click_correlations_runs,
    to_observable_records,
)
from mpm_analysis.io.experimental.click_record_npz import ClickRecordData
from mpm_analysis.io.experimental.compressed_click_record import (
    CompressedClickRecord,
)
from mpm_analysis.pipeline.click_correlation_pipeline import (
    ClickCorrelationPipeline,
)


# ---------------------------------------------------------------------------
# Builders — the same click matrix as a dense record and a compressed record
# ---------------------------------------------------------------------------

def make_dense(clicks: np.ndarray, lambda_val: float = 1.0,
               dt_us: float = 1.0) -> ClickRecordData:
    """Wrap a boolean (N, T) click matrix as a dense ClickRecordData."""
    n, t = clicks.shape
    return ClickRecordData(
        lambda_val=lambda_val,
        t=np.arange(t, dtype=float) * dt_us,
        trajectories=clicks.astype(np.float32),
        source_file="synthetic",
        metadata={"omega_DG": 1.0},
    )


def make_compressed(clicks: np.ndarray, n_runs: int,
                    lambda_val: float = 1.0,
                    dt_us: float = 1.0,
                    lengths: np.ndarray | None = None) -> CompressedClickRecord:
    """Wrap the same boolean (N, T) click matrix as a CompressedClickRecord.

    ``lengths`` sets per-shot compressed lengths (default: full T).  Only the
    fields the estimator reads carry real data.
    """
    n, t = clicks.shape
    assert n % n_runs == 0
    per_run = n // n_runs
    if lengths is None:
        lengths = np.full(n, t, dtype=np.int32)

    click_times, shot_offsets = [], [0]
    for i in range(n):
        ct = np.nonzero(clicks[i])[0]
        click_times.extend(ct.tolist())
        shot_offsets.append(shot_offsets[-1] + len(ct))

    alive = np.full((n_runs, t), per_run, dtype=np.int32)
    return CompressedClickRecord(
        lambda_val=lambda_val,
        rate_khz=100,
        t=np.arange(t, dtype=float) * dt_us,
        time_raw_us=np.arange(t, dtype=float) * dt_us,
        dt_us=dt_us,
        pnt=np.zeros((t, 2)),
        counts_hist=np.zeros((n_runs, t, 2)),
        alive=alive,
        shots_per_time=alive.sum(axis=0),
        runs_per_time=np.full(t, n_runs, dtype=np.int32),
        jump_counts=clicks.sum(axis=1).astype(np.int32),
        compressed_lengths=np.asarray(lengths, dtype=np.int32),
        click_times=np.array(click_times, dtype=np.int32),
        shot_offsets=np.array(shot_offsets, dtype=np.int64),
        run_shot_offsets=np.arange(n_runs + 1, dtype=np.int64) * per_run,
        fourier_k_array=np.zeros(3),
        pkt_real=np.zeros((t, 3)),
        pkt_imag=np.zeros((t, 3)),
        omega_dg=1.0,
        n_runs=n_runs,
        n_max=1,
        source_file="synthetic",
        metadata={},
    )


def bernoulli_clicks(p: float, n: int, t: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, t)) < p


# ---------------------------------------------------------------------------
# Counting kernel
# ---------------------------------------------------------------------------

# Run A: clicks at 2, 4, 7   length 8
# Run B: click  at 0         length 5
# Run C: no clicks           length 8
# Run D: clicks at 3, 4      length 6
_CT = np.array([2, 4, 7, 0, 3, 4])
_SO = np.array([0, 3, 4, 4, 6])
_LEN = np.array([8, 5, 8, 6])


def _brute_force(click_times, shot_offsets, lengths, width, max_clicks):
    """Straight-from-the-definition reference for :func:`_click_counts`."""
    k = np.inf if max_clicks is None else max_clicks
    n_runs = len(lengths)
    click = np.zeros(width)
    no_click = np.zeros(width)
    alive = np.zeros(width)
    for i in range(n_runs):
        times = set(click_times[shot_offsets[i]:shot_offsets[i + 1]].tolist())
        for j in range(width):
            if j >= lengths[i]:
                continue                       # terminated: out of everything
            alive[j] += 1
            c = sum(1 for x in times if x < j)  # clicks strictly before j
            if c > k:
                continue                       # too many already
            if j in times:
                click[j] += 1
            else:
                no_click[j] += 1
    return click, no_click, alive


@pytest.mark.parametrize("max_clicks", [0, 1, 2, None])
def test_kernel_matches_definition(max_clicks):
    """Vectorised kernel == the literal definition, on random records."""
    rng = np.random.default_rng(3)
    clicks = np.zeros((60, 14), dtype=bool)
    for s in range(60):                        # no two clicks adjacent
        t = 0
        while t < 14:
            if rng.random() < 0.35:
                clicks[s, t] = True
                t += 2
            else:
                t += 1
    lengths = rng.integers(5, 15, size=60)
    clicks &= np.arange(14)[None, :] < lengths[:, None]
    ct, so = _clicks_to_sparse(clicks)

    got = _click_counts(ct, so, lengths, 14, max_clicks)
    want = _brute_force(ct, so, lengths, 14, max_clicks)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    np.testing.assert_array_equal(got[2], want[2])


def test_kernel_conditional():
    """max_clicks=0: silence required in every bin up to t."""
    click, no_click, alive, n_clicked, n_runs = _click_counts(
        _CT, _SO, _LEN, width=8, max_clicks=0,
    )
    assert n_clicked == 3.0 and n_runs == 4
    # denominator: runs not yet terminated (B ends after bin 4, D after 5)
    assert alive.tolist() == [4, 4, 4, 4, 4, 3, 2, 2]
    # first click exactly at t: B@0, A@2, D@3
    assert click.tolist() == [1, 0, 1, 1, 0, 0, 0, 0]
    # silent through t: A until 1, C throughout, D until 2 (B clicks at 0)
    assert no_click.tolist() == [3, 3, 2, 1, 1, 1, 1, 1]
    # every run is either silent, clicking now, or already disqualified
    assert np.all(click + no_click <= alive)


def test_kernel_unconditional():
    """max_clicks=None: only the bin at t matters."""
    click, no_click, alive, _, _ = _click_counts(
        _CT, _SO, _LEN, width=8, max_clicks=None,
    )
    # every click contributes at its own bin (A@2,4,7  B@0  D@3,4)
    assert click.tolist() == [1, 0, 1, 1, 2, 0, 0, 1]
    # the two channels partition the surviving runs exactly
    np.testing.assert_array_equal(click + no_click, alive)


def test_kernel_partial_interpolates():
    """k=0 and k>=max clicks reproduce the two fixed modes."""
    cond = _click_counts(_CT, _SO, _LEN, width=8, max_clicks=0)
    uncond = _click_counts(_CT, _SO, _LEN, width=8, max_clicks=None)
    k0 = _click_counts(_CT, _SO, _LEN, width=8, max_clicks=0)
    k_big = _click_counts(_CT, _SO, _LEN, width=8, max_clicks=99)
    np.testing.assert_array_equal(k0[0], cond[0])
    np.testing.assert_array_equal(k0[1], cond[1])
    np.testing.assert_array_equal(k_big[0], uncond[0])
    np.testing.assert_array_equal(k_big[1], uncond[1])

    # k=1 tolerates one earlier click, so A stays eligible through its 2nd
    click, no_click, _, _, _ = _click_counts(
        _CT, _SO, _LEN, width=8, max_clicks=1,
    )
    assert click.tolist() == [1, 0, 1, 1, 2, 0, 0, 0]
    assert no_click.tolist() == [3, 4, 3, 3, 2, 1, 1, 1]


def test_kernel_no_clicks_at_all():
    click, no_click, alive, n_clicked, n_runs = _click_counts(
        np.array([], dtype=int), np.array([0, 0, 0]), np.array([5, 3]),
        width=6, max_clicks=0,
    )
    assert n_clicked == 0.0 and n_runs == 2
    assert click.tolist() == [0] * 6
    assert alive.tolist() == [2, 2, 2, 1, 1, 0]
    np.testing.assert_array_equal(no_click, alive)


# ---------------------------------------------------------------------------
# Analytic limits
# ---------------------------------------------------------------------------

def test_bernoulli_conditional_is_survival():
    """i.i.d. clicks: P(n=0,t) = (1-p)^(t+1)."""
    p, n, t = 0.3, 60_000, 20
    res = compute_click_correlations_dense(
        make_dense(bernoulli_clicks(p, n, t, seed=4)),
        mode="conditional", n_bootstrap=50,
    )
    assert res.mode == "conditional" and res.max_clicks == 0
    assert len(res.t_us) == t                      # the record's own axis
    for j in range(6):
        expected = (1 - p) ** (j + 1)
        se = 5 * np.sqrt(expected * (1 - expected) / n) + 1e-12
        assert res.no_click[j] == pytest.approx(expected, abs=se)
        # first click exactly at j
        expected_first = p * (1 - p) ** j
        assert res.click[j] == pytest.approx(
            expected_first, abs=5 * np.sqrt(expected_first / n))
    assert 0 < res.no_click_err[3] < 1e-2


def test_bernoulli_unconditional_is_per_bin():
    """i.i.d. clicks: P(click at t) = p at every t, flat in time."""
    p, n, t = 0.3, 60_000, 20
    res = compute_click_correlations_dense(
        make_dense(bernoulli_clicks(p, n, t, seed=5)),
        mode="unconditional", n_bootstrap=0,
    )
    assert res.mode == "unconditional" and res.max_clicks is None
    se = 5 * np.sqrt(p * (1 - p) / n)
    for j in range(6):
        assert res.click[j] == pytest.approx(p, abs=se)
    # complementary by construction, at every time
    np.testing.assert_allclose(res.click + res.no_click, 1.0)


def test_conditional_matches_counting_statistics():
    """The conditional no-click curve IS the n=0 row of P(n,t)."""
    clicks = bernoulli_clicks(0.25, 20_000, 15, seed=6)
    res = compute_click_correlations_dense(
        make_dense(clicks), mode="conditional", n_bootstrap=0)
    # P(n=0 through bin j) computed directly from the click matrix
    direct = (np.cumsum(clicks, axis=1) == 0).mean(axis=0)
    np.testing.assert_allclose(res.no_click, direct, atol=1e-12)


# ---------------------------------------------------------------------------
# Termination handling and the alive cut
# ---------------------------------------------------------------------------

def test_min_alive_frac_only_truncates_variable_lengths():
    n, t = 400, 10
    clicks = np.zeros((n, t), dtype=bool)
    clicks[:n // 2, 2] = True
    lengths = np.concatenate((np.full(n // 2, t), np.full(n // 2, 4)))
    rec = make_compressed(clicks, n_runs=2, lengths=lengths)

    tight = compute_click_correlations(rec, min_alive_frac=0.6, n_bootstrap=0)
    assert len(tight.t_us) == 4                    # alive halves at bin 4
    loose = compute_click_correlations(rec, min_alive_frac=0.0, n_bootstrap=0)
    assert len(loose.t_us) == t

    # dense records span the full window -> the cut never bites
    dense = compute_click_correlations_dense(
        make_dense(clicks), min_alive_frac=0.99, n_bootstrap=0)
    assert len(dense.t_us) == t


def test_terminated_runs_leave_the_denominator():
    """Runs that ended are out of both numerator and denominator."""
    clicks = np.zeros((4, 6), dtype=bool)
    rec = make_compressed(clicks, n_runs=2,
                          lengths=np.array([6, 6, 3, 3]))
    res = compute_click_correlations(rec, min_alive_frac=0.0, n_bootstrap=0)
    # all runs silent -> no_click is 1 wherever anyone is still recorded
    np.testing.assert_allclose(res.no_click, 1.0)
    assert res.window["alive_first"] == 4
    assert res.window["alive_last"] == 2


def test_max_time_cap():
    clicks = bernoulli_clicks(0.3, 500, 30, seed=7)
    res = compute_click_correlations_dense(
        make_dense(clicks), max_time_us=4.0, min_alive_frac=0.0, n_bootstrap=0)
    assert len(res.t_us) == 5                      # t = 0 … 4 µs


# ---------------------------------------------------------------------------
# Source equality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["conditional", "unconditional"])
def test_compressed_matches_dense(mode):
    clicks = bernoulli_clicks(0.3, 1000, 20, seed=2)
    res_d = compute_click_correlations_dense(
        make_dense(clicks), mode=mode, n_bootstrap=50, seed=7, n_groups=4)
    res_c = compute_click_correlations(
        make_compressed(clicks, n_runs=4), mode=mode, n_bootstrap=50, seed=7)

    np.testing.assert_array_equal(res_d.t_us, res_c.t_us)
    np.testing.assert_allclose(res_d.click, res_c.click)
    np.testing.assert_allclose(res_d.no_click, res_c.no_click)
    np.testing.assert_allclose(res_d.no_click_err, res_c.no_click_err)
    assert res_d.r_bar == pytest.approx(res_c.r_bar)


def test_runs_estimator_matches_dense():
    clicks = bernoulli_clicks(0.3, 900, 18, seed=8)
    t = np.arange(18, dtype=float)
    runs = (clicks[i * 300:(i + 1) * 300] for i in range(3))

    res_r = compute_click_correlations_runs(
        runs, t, lambda_val=1.0, rate_khz=250, n_bootstrap=40, seed=9)
    res_d = compute_click_correlations_dense(
        make_dense(clicks), n_bootstrap=40, seed=9, n_groups=3)
    np.testing.assert_allclose(res_r.no_click, res_d.no_click)
    np.testing.assert_allclose(res_r.click, res_d.click)
    np.testing.assert_allclose(res_r.no_click_err, res_d.no_click_err)
    assert res_r.window["n_groups"] == 3
    assert res_r.rate_khz == 250


def test_dispatch_and_sorting():
    clicks = bernoulli_clicks(0.2, 200, 10, seed=3)
    recs = [make_dense(clicks, lambda_val=2.0),
            make_compressed(clicks, n_runs=2, lambda_val=0.5)]
    out = compute_all_click_correlations(recs, n_bootstrap=0)
    assert [r.lambda_val for r in out] == [0.5, 2.0]
    with pytest.raises(TypeError):
        compute_all_click_correlations([object()])


# ---------------------------------------------------------------------------
# Mode plumbing
# ---------------------------------------------------------------------------

def test_partial_mode_requires_max_clicks():
    clicks = bernoulli_clicks(0.3, 200, 10, seed=0)
    with pytest.raises(ValueError, match="needs max_clicks"):
        compute_click_correlations_dense(make_dense(clicks), mode="partial")


def test_fixed_modes_reject_max_clicks():
    clicks = bernoulli_clicks(0.3, 200, 10, seed=0)
    with pytest.raises(ValueError, match="mode='partial'"):
        compute_click_correlations_dense(
            make_dense(clicks), mode="conditional", max_clicks=2)


@pytest.mark.parametrize("retired", ["no_click", "first_click", "conditioned"])
def test_retired_mode_names_are_refused(retired):
    """The names changed meaning, so they must not be silently remapped."""
    clicks = bernoulli_clicks(0.3, 200, 10, seed=0)
    with pytest.raises(ValueError, match="no longer exists"):
        compute_click_correlations_dense(make_dense(clicks), mode=retired)


def test_partial_bridges_the_two_modes():
    clicks = bernoulli_clicks(0.3, 4000, 14, seed=12)
    cond = compute_click_correlations_dense(
        make_dense(clicks), mode="conditional", n_bootstrap=0)
    uncond = compute_click_correlations_dense(
        make_dense(clicks), mode="unconditional", n_bootstrap=0)
    k0 = compute_click_correlations_dense(
        make_dense(clicks), mode="partial", max_clicks=0, n_bootstrap=0)
    k_big = compute_click_correlations_dense(
        make_dense(clicks), mode="partial", max_clicks=50, n_bootstrap=0)

    np.testing.assert_allclose(k0.no_click, cond.no_click)
    np.testing.assert_allclose(k_big.no_click, uncond.no_click)
    assert k0.keys == (PARTIAL_CLICK, PARTIAL_NO_CLICK)
    # k=1 sits between the two everywhere
    k1 = compute_click_correlations_dense(
        make_dense(clicks), mode="partial", max_clicks=1, n_bootstrap=0)
    assert np.all(k1.no_click >= cond.no_click - 1e-12)
    assert np.all(k1.no_click <= uncond.no_click + 1e-12)


# ---------------------------------------------------------------------------
# ObservableRecord conversion
# ---------------------------------------------------------------------------

def test_to_observable_records():
    clicks = bernoulli_clicks(0.25, 2000, 15, seed=6)
    results = [compute_click_correlations_dense(
        make_dense(clicks, lambda_val=lv), n_bootstrap=10) for lv in (0.4, 0.8)]

    recs = to_observable_records(results)
    assert len(recs) == 4
    assert {r.observable_key for r in recs} == {COND_CLICK, COND_NO_CLICK}
    for r in recs:
        assert r.t[0] == 0.0
        assert len(r.metadata["signal_err"]) == len(r.signal)
        assert r.metadata["mode"] == "conditional"
        assert r.metadata["max_clicks"] == 0

    trimmed = to_observable_records(results, observables=(COND_NO_CLICK,),
                                    skip_first=2)
    assert len(trimmed) == 2
    assert trimmed[0].t[0] == pytest.approx(results[0].t_us[2])

    uncond = [compute_click_correlations_dense(
        make_dense(clicks), mode="unconditional", n_bootstrap=0)]
    assert {r.observable_key for r in to_observable_records(uncond)} == {
        UNCOND_CLICK, UNCOND_NO_CLICK}


# ---------------------------------------------------------------------------
# Pipeline, all sources
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end(tmp_path):
    # i.i.d. clicks: P(n=0,t) = (1-p)^(t+1) is a single real exponential with
    # decay -ln(1-p)/dt, which order-1 MPM must recover.
    dt = 1.0
    p_of_lam = {0.5: 0.15, 1.0: 0.25, 1.5: 0.35}
    records = [
        make_dense(bernoulli_clicks(p, 50_000, 25, seed=int(10 * lam)),
                   lambda_val=lam, dt_us=dt)
        for lam, p in p_of_lam.items()
    ]

    out = ClickCorrelationPipeline().run(
        records=records, source="json", out_dir=tmp_path,
        mode="conditional", observables=(COND_NO_CLICK,), order=1,
        n_boot=25, n_bootstrap=25, verbose=False,
    )

    result = out["mpm_results"][COND_NO_CLICK]
    for lam, gamma in zip(result.lambda_values, result.poles.decay_rates[:, 0]):
        expected = -np.log(1 - p_of_lam[float(lam)]) / dt
        assert gamma == pytest.approx(expected, rel=0.05)

    run_dir = out["out_dir"]
    assert run_dir.exists() and out["correlations_path"].exists()
    assert out["result_paths"][COND_NO_CLICK].exists()
    figures = out["figures"]
    assert len(figures) == 5   # svd + 2 raw channels + click prob + poles
    assert sum(1 for f in figures if f.name.startswith("corr_raw_")) == 2
    assert all(f.exists() and f.suffix == ".png" for f in figures)

    z = np.load(out["correlations_path"])
    assert "t_us_0" in z.files and "no_click_err_2" in z.files


def _write_individual_runs_scan(tmp_path, rates, n_runs=3, n_shots=400,
                                n_bins=14, seed=0):
    """Synthetic scan folder with an ``individual_runs`` subfolder."""
    import json as _json

    rng = np.random.default_rng(seed)
    runs_dir = tmp_path / "scan_experiment_results_99" / "individual_runs"
    runs_dir.mkdir(parents=True)
    t_ns = (np.arange(n_bins) * 320.0).tolist()

    expected = {}
    for rate in rates:
        lam = rate / 1000.0
        raws = [rng.random((n_shots, n_bins)) < 0.4 for _ in range(n_runs)]
        for i, raw in enumerate(raws, start=1):
            stem = (f"click_record_repeated_data__OBG0_{rate}_kHz"
                    f"_run_{i}_of_{n_runs}_19_06_2026__13_32_07")
            np.savez(runs_dir / f"{stem}.npz",
                     assigned_states=raw.astype(bool).ravel())
            (runs_dir / f"{stem}.json").write_text(_json.dumps({
                "data": {"time_array": t_ns},
                "parameters": {
                    "lambda_zeno_estimation": lam,
                    "Omega_DG": 628318.5307,
                    "navg_repeat_experiment": n_shots,
                },
            }), encoding="utf-8")
        raw_all = np.vstack(raws)
        clicks = np.zeros_like(raw_all, dtype=bool)
        clicks[:, 1:] = raw_all[:, :-1] & ~raw_all[:, 1:]
        expected[rate] = (lam, clicks)
    return runs_dir.parent, expected


def test_pipeline_json_individual_runs(tmp_path):
    scan_dir, expected = _write_individual_runs_scan(
        tmp_path, rates=(1455, 2310), n_runs=3)

    out = ClickCorrelationPipeline().run(
        scan_dir, source="json", run_mpm=False, save_figures=False,
        n_bootstrap=30, verbose=False)

    correlations = out["correlations"]
    assert [round(r.lambda_val, 4) for r in correlations] == [1.455, 2.31]
    assert correlations[0].rate_khz == 1455
    assert correlations[0].mode == "conditional"
    assert correlations[0].dt_us == pytest.approx(0.32)
    assert correlations[0].t_us[0] == 0.0
    for res in correlations:
        lam, clicks = expected[res.rate_khz]
        ref = compute_click_correlations_dense(
            make_dense(clicks, lambda_val=lam, dt_us=0.32),
            n_bootstrap=0, n_groups=3)
        np.testing.assert_allclose(res.no_click, ref.no_click)
        np.testing.assert_allclose(res.click, ref.click)


def _write_raw_sim_npz(tmp_path, *, key, store_telegraph, n_lambda=2,
                       n_trials=300, n_steps=12, drop_pnt_for=(), seed=0):
    """Raw-format simulation NPZ in one of the generator vintages."""
    rng = np.random.default_rng(seed)
    payload = {
        "lambda_values": np.arange(1, n_lambda + 1, dtype=float) / 2,
        "k_values": np.linspace(0, 2 * np.pi, 5),
        "t_values": np.arange(n_steps, dtype=float) * 0.18,
        "n_trials": np.array(n_trials), "n_steps": np.array(n_steps),
        "dt": np.array(0.18), "Omega": np.array(0.628),
    }
    expected = {}
    for i in range(n_lambda):
        if store_telegraph:
            tele = np.zeros((n_trials, n_steps), dtype=bool)
            for s in range(n_trials):
                t = 0
                while t < n_steps:
                    if rng.random() < 0.3:
                        run = min(rng.integers(1, 4), n_steps - t)
                        tele[s, t:t + run] = True
                        t += run + 1
                    else:
                        t += 1
            stored = tele
            clicks = np.zeros_like(tele)
            clicks[:, 1:] = tele[:, :-1] & ~tele[:, 1:]
        else:
            clicks = np.zeros((n_trials, n_steps), dtype=bool)
            for s in range(n_trials):
                t = 0
                while t < n_steps:
                    if rng.random() < 0.3:
                        clicks[s, t] = True
                        t += 2
                    else:
                        t += 1
            stored = clicks
        expected[i] = clicks
        payload[f"{key}_{i}"] = np.packbits(stored.T.ravel().astype(np.uint8))
        if i not in drop_pnt_for:
            payload[f"pnt_{i}"] = np.zeros((n_steps + 1, n_steps), dtype=np.float32)
            payload[f"pkt_{i}"] = np.zeros((5, n_steps), dtype=np.complex64)

    path = tmp_path / f"click_record_raw_{key}.npz"
    np.savez(path, **payload)
    return path, expected


@pytest.mark.parametrize("key, store_telegraph", [
    ("proc_packed", False),
    ("raw_packed", False),
    ("raw_packed", True),
])
def test_sim_loader_click_conventions(tmp_path, key, store_telegraph):
    from mpm_analysis.io.simulated.click_record_raw_npz import (
        SimClickRecordLoader,
    )

    path, expected = _write_raw_sim_npz(
        tmp_path, key=key, store_telegraph=store_telegraph)
    for i, rec in enumerate(SimClickRecordLoader().load(path)):
        got = rec.trajectories > 0
        np.testing.assert_array_equal(got, expected[i])
        assert not np.any(got[:, :-1] & got[:, 1:])


def test_sim_loader_partial_file(tmp_path):
    from mpm_analysis.io.simulated.click_record_raw_npz import (
        SimClickRecordLoader,
    )

    path, _ = _write_raw_sim_npz(
        tmp_path, key="raw_packed", store_telegraph=False,
        n_lambda=3, drop_pnt_for=(2,))
    loader = SimClickRecordLoader()
    assert len(loader.load(path)) == 3
    assert len(loader.load_observable(path)) == 2


def test_pipeline_simulated_raw_stream(tmp_path):
    path, expected = _write_raw_sim_npz(
        tmp_path, key="raw_packed", store_telegraph=False, n_lambda=2)
    out = ClickCorrelationPipeline().run(
        path, source="simulated", mode="unconditional",
        run_mpm=False, save_figures=False, save_results=False,
        n_bootstrap=0, verbose=False)
    correlations = out["correlations"]
    assert len(correlations) == 2
    for i, res in enumerate(correlations):
        assert res.mode == "unconditional"
        ref = compute_click_correlations_dense(
            make_dense(expected[i], dt_us=0.18), mode="unconditional",
            n_bootstrap=0)
        np.testing.assert_allclose(res.no_click, ref.no_click)
        assert res.metadata["omega_DG"] == pytest.approx(0.628)


def _write_shifted_npz(tmp_path, lambdas, n_fine=12, delta=0.1, seed=0):
    rng = np.random.default_rng(seed)
    qs = [rng.uniform(0.02, 0.3, size=n_fine) for _ in lambdas]
    payload = {
        "lambda_values": np.array(lambdas),
        "delta_fine": np.array(delta),
        "n_shifts": np.array(4),
        "n_trials_per_shift": np.array(1000),
        "Omega": np.array(0.628),
        "dt": np.array(delta * 4),
    }
    for i, q in enumerate(qs):
        payload[f"q_{i}"] = q
        payload[f"clicks_{i}"] = np.zeros((4, n_fine // 4))
    path = tmp_path / "click_record_shifted_test.npz"
    np.savez(path, **payload)
    return path, qs


@pytest.mark.parametrize("mode", ["conditional", "unconditional"])
def test_pipeline_simulated_shifted(tmp_path, mode):
    path, qs = _write_shifted_npz(tmp_path, lambdas=[0.5, 1.0])
    out = ClickCorrelationPipeline().run(
        path, source="simulated", mode=mode, run_mpm=False,
        save_figures=False, save_results=False, verbose=False)

    for res, q in zip(out["correlations"], qs):
        survival = np.cumprod(1.0 - q)
        if mode == "conditional":
            np.testing.assert_allclose(res.no_click, survival)
            np.testing.assert_allclose(
                res.click, q * np.concatenate(([1.0], survival[:-1])))
        else:
            np.testing.assert_allclose(res.no_click, 1.0 - q)
            np.testing.assert_allclose(res.click, q)
        assert res.dt_us == pytest.approx(0.1)
        assert res.mode == mode

    with pytest.raises(ValueError, match="partial"):
        ClickCorrelationPipeline().run(
            path, source="simulated", mode="partial", max_clicks=1,
            run_mpm=False, save_figures=False, save_results=False,
            verbose=False)


def test_pipeline_survives_short_windows(tmp_path, capsys):
    """Too-short windows are skipped with guidance, not a crash."""
    n, t = 400, 12
    clicks = np.zeros((n, t), dtype=bool)
    lengths = np.concatenate((np.full(n // 2, t), np.full(n // 2, 3)))
    rec = make_compressed(clicks, n_runs=2, lengths=lengths)

    out = ClickCorrelationPipeline().run(
        records=[rec], source="compressed", out_dir=tmp_path,
        min_alive_frac=0.98, order=3, n_bootstrap=0, n_boot=2,
        save_figures=False, verbose=True)
    assert out["correlations"][0].window["n_bins"] < 7
    assert out["mpm_results"] == {}
    assert out["correlations_path"].exists()
    msg = capsys.readouterr().out
    assert "min_alive_frac" in msg and "order" in msg


def test_pipeline_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError, match="Unknown source"):
        ClickCorrelationPipeline().run(tmp_path, source="csv")


def test_pipeline_requires_input():
    with pytest.raises(ValueError, match="folder"):
        ClickCorrelationPipeline().run(source="compressed")


# ---------------------------------------------------------------------------
# MPM robustness (regressions from mixed-length slices)
# ---------------------------------------------------------------------------

def test_bootstrap_tolerates_mixed_lengths():
    from mpm_analysis.analysis.bootstrap import run_bootstrap
    from mpm_analysis.data_types.observables import ObservableRecord

    long_t = np.arange(30, dtype=float)
    short_t = np.arange(4, dtype=float)
    recs = [
        ObservableRecord(lambda_val=0.5, t=long_t,
                         signal=np.exp(-0.2 * long_t), observable_key="x"),
        ObservableRecord(lambda_val=1.0, t=short_t,
                         signal=np.exp(-0.2 * short_t), observable_key="x"),
    ]
    out = run_bootstrap(recs, order=3, n_boot=3, L_used=[10, 10],
                        rng=np.random.default_rng(0))
    assert np.all(np.isfinite(out[0].boot_real))
    assert np.all(np.isnan(out[1].boot_real))


def test_matrix_pencil_rejects_short_signal():
    from mpm_analysis.analysis.matrix_pencil import matrix_pencil_method

    t = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        matrix_pencil_method(t, np.exp(-t), order=3)
