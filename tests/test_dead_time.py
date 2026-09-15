"""Tests for the detector dead-time compression sweep.

- time-axis remap: identity at c = 0, monotonicity guard, record plumbing
- uniform regrid of a non-uniform record
- direct non-uniform fit recovers the generating poles of a synthetic
  three-pole signal sampled on a compressed axis
- refine-only data bootstrap (pencil=False) runs on a non-uniform axis
- end-to-end sweep on synthetic records: c = 0 reproduces the raw poles,
  the sweep tracks a known drift, JSON round trip
"""
from __future__ import annotations

import numpy as np
import pytest

from mpm_analysis.analysis.dead_time import (
    compress_records, compress_time_axis, first_fold, resample_uniform,
)
from mpm_analysis.analysis.matrix_pencil import (
    matrix_pencil_method, refine_poles, reconstruct_signal,
)
from mpm_analysis.data_types.observables import ObservableRecord


# A slow real pole dominating a faster conjugate pair keeps the synthetic
# "D probability" inside (0, 1) — the multinomial resampler clips at 0.
POLES = np.array([-0.15 + 0.0j, -0.5 - 1.7j, -0.5 + 1.7j])
AMPS = np.array([0.6 + 0j, 0.15 - 0.05j, 0.15 + 0.05j])
T = 0.32 * np.arange(102)


def _p_click(t):
    return 0.45 * (1.0 - np.exp(-t / 3.0))


def _signal(t):
    return reconstruct_signal(t, POLES, AMPS)


def _records(lambdas=(0.8, 1.0, 1.2), n_shots=3000):
    """Synthetic ens_avg-like records with a click probability per λ."""
    recs = []
    for lam in lambdas:
        y = _signal(T)
        P = np.vstack([y, 1 - y - 0.01, np.full_like(y, 0.005), np.full_like(y, 0.005)])
        meta = dict(omega_DG=1.0, bootstrap_stats=dict(
            data_mode="ens_avg", P=P, channels=["D", "G", "B", "error"],
            N_shots=n_shots))
        recs.append(ObservableRecord(lam, T, y, "D", metadata=meta))
        recs.append(ObservableRecord(lam, T, _p_click(T), "click_probability",
                                     metadata=meta))
    return recs


def _same_pole_set(a, b, tol=1e-6):
    return np.allclose(np.sort_complex(np.asarray(a)), np.sort_complex(np.asarray(b)),
                       atol=tol)


# ---------------------------------------------------------------------------
# axis remap
# ---------------------------------------------------------------------------

def test_compress_time_axis_identity_and_bounds():
    t2 = compress_time_axis(T, _p_click(T), 0.0)
    assert np.array_equal(t2, T)
    t1 = compress_time_axis(T, _p_click(T), 1.0)
    assert np.all(np.diff(t1) > 0) and t1[-1] < T[-1]
    with pytest.raises(ValueError):
        compress_time_axis(T, _p_click(T), 1.5)


def _folding_records():
    recs = _records(lambdas=(1.0,))
    p = np.zeros_like(T)
    p[10:] = 0.99            # a step in p_click folds the pointwise axis at bin 10
    recs[1].signal = p
    return recs


def test_pointwise_fold_detection_and_truncation():
    p = _folding_records()[1].signal
    assert first_fold(compress_time_axis(T, p, 1.0)) == 10
    assert first_fold(compress_time_axis(T, p, 0.0)) is None
    with pytest.raises(ValueError):
        compress_records(_folding_records(), 1.0, keys=["D"])
    out = compress_records(_folding_records(), 1.0, keys=["D"], on_fold="truncate")
    rec = out[0]
    assert rec.t.size == 10 and rec.metadata["fold_index"] == 10
    assert np.all(np.diff(rec.t) > 0)
    np.testing.assert_array_equal(rec.signal, _signal(T)[:10])
    # time-resolved bootstrap statistics are cut to the same length
    assert rec.metadata["bootstrap_stats"]["P"].shape == (4, 10)
    assert rec.metadata["bootstrap_stats"]["N_shots"] == 3000


def test_cumulative_remap_never_folds():
    p = _folding_records()[1].signal
    t_c = compress_time_axis(T, p, 1.0, remap="cumulative")
    assert first_fold(t_c) is None
    # slope 1 - c p: exactly t before the step, shrinking after it
    np.testing.assert_allclose(t_c[:10], T[:10])
    assert t_c[-1] < T[-1]
    out = compress_records(_folding_records(), 1.0, keys=["D"], remap="cumulative")
    assert out[0].t.size == T.size and out[0].metadata["remap"] == "cumulative"


def test_compress_records_plumbing():
    recs = _records()
    out = compress_records(recs, 0.5, keys=["D"])
    assert {r.observable_key for r in out} == {"D"}
    assert len(out) == 3
    for r in out:
        assert r.metadata["compress_coeff"] == 0.5
        assert "bootstrap_stats" in r.metadata
        np.testing.assert_array_equal(r.signal, _signal(T))   # y untouched
        assert r.t[-1] < T[-1]
    with pytest.raises(KeyError):
        compress_records([r for r in recs if r.observable_key == "D"], 0.5)


def test_resample_uniform():
    rec = compress_records(_records(), 1.0, keys=["D"])[0]
    u = resample_uniform(rec, 0.32)
    assert np.allclose(np.diff(u.t), 0.32)
    assert u.t[0] == rec.t[0] and u.t[-1] <= rec.t[-1]
    assert np.allclose(u.signal, np.interp(u.t, rec.t, rec.signal))


# ---------------------------------------------------------------------------
# direct fit on the non-uniform axis
# ---------------------------------------------------------------------------

def test_refine_recovers_poles_on_compressed_axis():
    t_c = compress_time_axis(T, _p_click(T), 1.0)
    y = _signal(t_c)                       # generated ON the compressed axis
    mpm_poles, *_ = matrix_pencil_method(T, _signal(T), 3)   # raw-axis seed
    fitted = refine_poles(t_c, y, mpm_poles, conjugate_pairs=True,
                          bounds_real=(-np.inf, 0.0))
    assert _same_pole_set(fitted, POLES, tol=1e-4)


def test_refine_only_bootstrap_on_nonuniform_axis():
    from mpm_analysis.analysis.bootstrap import run_data_bootstrap

    # Generate the modal signal ON the compressed axis, so that after
    # compress_records(c=0.7) the true poles are exactly POLES.
    recs = _records(lambdas=(1.0,))
    t_c = compress_time_axis(T, _p_click(T), 0.7)
    for r in recs:
        if r.observable_key == "D":
            r.signal = _signal(t_c)
            P = r.metadata["bootstrap_stats"]["P"]
            P[0], P[1] = r.signal, 1 - r.signal - 0.01   # columns still sum to 1
    recs = compress_records(recs, 0.7, keys=["D"])
    settings = dict(conjugate_pairs=True, bounds_real=(-np.inf, 0.0), bounds_imag=None)
    with pytest.raises(ValueError):
        run_data_bootstrap(recs, order=3, n_boot=2, observable_key="D",
                           pencil=False, n_jobs=1)
    out = run_data_bootstrap(recs, order=3, n_boot=8, observable_key="D",
                             clean_poles=[POLES], refine_settings=settings,
                             pencil=False, n_jobs=1, seed=0)
    assert len(out) == 1
    assert np.all(np.isfinite(out[0].boot_real))
    # Draws scatter around the clean poles
    assert np.allclose(np.median(out[0].boot_real, axis=0), POLES.real, atol=0.05)


# ---------------------------------------------------------------------------
# sweep pipeline
# ---------------------------------------------------------------------------

def test_sweep_pipeline_end_to_end(tmp_path):
    from mpm_analysis.io.analysis_json import load_dead_time_sweep
    from mpm_analysis.pipeline.dead_time_pipeline import DeadTimeSweepPipeline

    recs = _records()
    pipe = DeadTimeSweepPipeline(recs, [0.0, 0.5, 1.0], order=3,
                                 observable_keys=["D"], fallback=True,
                                 metadata={"data_mode": "simulation"})
    res = pipe.run()
    assert res.direct.poles.shape == (3, 3, 3)
    assert res.fallback is not None and res.fallback.poles.shape == (3, 3, 3)
    # c = 0: the direct fit reproduces the generating poles
    for il in range(3):
        assert _same_pole_set(res.direct.poles[0, il], POLES, tol=1e-4)
        assert _same_pole_set(res.fallback.poles[0, il], POLES, tol=1e-3)
    # compressing the axis shortens every time scale: |Re| and |Im| grow
    assert np.all(-res.direct.poles[2].real > -res.direct.poles[0].real - 1e-9)
    assert np.all(np.abs(res.direct.poles[2].imag) >= np.abs(res.direct.poles[0].imag) - 1e-9)
    # both routes agree where both converge (interpolation error only)
    ok = np.isfinite(res.fallback.poles).all(axis=2)
    assert np.allclose(res.direct.poles[ok], res.fallback.poles[ok], atol=0.1)
    assert np.all(np.isfinite(res.direct.rmse))
    rows = res.summary_rows()
    assert len(rows) == 3 and "lambda_ep_direct" in rows[0]

    # one c as a standard result: the same poles, zero half-widths (no bootstrap)
    res0 = res.to_analysis_result(0)
    np.testing.assert_allclose(-res0.poles.decay_rates, res.direct.poles[0].real)
    assert res0.metadata["compress_coeff"] == 0.0 and res0.poles.decay_rates_lower.max() == 0

    path = pipe.save(tmp_path)
    assert path.exists() and path.with_suffix(".csv").exists()
    back = load_dead_time_sweep(path)
    np.testing.assert_allclose(back.direct.poles, res.direct.poles)
    np.testing.assert_allclose(back.compress_coeffs, res.compress_coeffs)
    assert back.fallback.method == "regrid_mpm"


# ---------------------------------------------------------------------------
# theory transition vs dead time
# ---------------------------------------------------------------------------

def test_theory_transition_moves_up_as_dead_time_is_removed():
    from mpm_analysis.simulation.liouvillian_spectrum import (
        tau_B_effective, theory_transition, theory_transition_vs_compression,
    )

    kw = dict(T1=15.0, T2=21.0, Omega=0.628)
    ep4 = theory_transition(tau_B=4.0, **kw)
    assert 1.25 < ep4 < 1.35                     # scan-50 parameters, tau_B = 4 us
    # c = tau_B_comp / tau_B: removing dead time shortens tau_B_eff ...
    np.testing.assert_allclose(tau_B_effective([0.0, 0.5, 1.0], 4.0), [4.0, 2.0, 1e-3])
    # ... and the transition climbs monotonically towards the ideal lambda = 2
    line = theory_transition_vs_compression(np.linspace(0, 1, 6), tau_B=4.0, **kw)
    assert np.all(np.diff(line) > 0) and line[0] == ep4 and 1.9 < line[-1] < 2.1


# ---------------------------------------------------------------------------
# figures: panel grids and HTML sliders (smoke tests, headless)
# ---------------------------------------------------------------------------

def test_panel_grids_and_html_sliders(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from mpm_analysis.pipeline.dead_time_pipeline import DeadTimeSweepPipeline
    from mpm_analysis.plotting.dead_time_plots import (
        export_poles_slider_html, export_traces_slider_html,
        plot_poles_panel_grid, plot_traces_panel_grid,
    )

    recs = _records()
    sweep = DeadTimeSweepPipeline(recs, [0.0, 0.5, 1.0], observable_keys=["D"],
                                  fallback=True, metadata={"data_mode": "simulation"}).run()
    theory = dict(tau_B=4.0, T1=15.0, T2=21.0)
    fig = plot_traces_panel_grid(recs, sweep, max_panels=2)
    assert fig.axes and len([a for a in fig.axes if a.get_subplotspec()]) == 3 * 2
    fig = plot_poles_panel_grid(sweep, theory_params=theory)
    assert len(fig.axes) == 3 * 2
    p1 = export_traces_slider_html(recs, sweep, tmp_path / "traces.html", show_fallback=True)
    p2 = export_poles_slider_html(sweep, tmp_path / "poles.html", theory_params=theory,
                                  method="regrid_mpm")
    for p in (p1, p2):
        text = open(p, encoding="utf-8").read()
        assert "plotly" in text and "compress coeff c" in text and text.count('"label"') >= 3


# ---------------------------------------------------------------------------
# synthetic 4x4 generator through the same sweep
# ---------------------------------------------------------------------------

def test_synthetic_generator_sweep_recovers_theory_ep():
    from mpm_analysis.pipeline.dead_time_pipeline import DeadTimeSweepPipeline
    from mpm_analysis.simulation.liouvillian_spectrum import (
        simulate_populations, synthetic_records, theory_transition,
    )

    th = dict(tau_B=4.0, T1=15.0, T2=21.0, Omega=0.628)
    t = 0.32 * np.arange(102)
    pops = simulate_populations(1.0, t, **th)
    np.testing.assert_allclose(pops["B"] + pops["G"] + pops["D"], 1.0, atol=1e-9)
    assert pops["D"][0] == 1.0 and pops["B"][0] == 0.0

    lams = np.linspace(0.9, 2.3, 29)
    recs = synthetic_records(lams, t, **th)
    assert {r.observable_key for r in recs} == {"D", "G", "B", "click_probability"}
    # exact 4-mode data: an order-4 direct fit at c = 0 reproduces the
    # generator's own coalescence (theory_transition) to the lambda spacing
    sweep = DeadTimeSweepPipeline(recs, [0.0, 0.3], order=4, observable_keys=["D"],
                                  fallback=False, metadata={"data_mode": "synthetic_4x4"}).run()
    ep_th = theory_transition(**th)
    assert abs(sweep.direct.lambda_ep[0] - ep_th) < 0.06
    # compressing the generator's own dead time pushes its EP up as well
    assert np.isfinite(sweep.direct.lambda_ep[1])
    assert sweep.direct.lambda_ep[1] > sweep.direct.lambda_ep[0] + 0.05
