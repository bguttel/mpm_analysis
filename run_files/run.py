"""MPM analysis driver for the Zeno paper transitions.

How to run
----------
    python run_files/run.py

Everything is configured by the ``if (RUN_... := True/False):`` toggles
below — flip the boolean in the ``if`` line to switch a task on or off.  The
settings a task needs live inside its own clause, so a disabled clause is
simply skipped and nothing else has to change.

Two independent entry points, one per paper transition:

    RUN_FIRST_TRANSITION   post-selected survival  (scan 10)  → first transition
    RUN_THIRD_TRANSITION   ensemble-average  D     (scan 50)  → third transition

Inside each, the nested toggles select what is done with the loaded data:

    ..._MPM                 matrix-pencil poles + bootstrap error bars   (required)
        ..._TRANSITIONS     state-freezing (ksi = 0) and exceptional point (Delta = 0)
        ..._CRITICAL_POINT  sqrt fit of the splitting → lambda_c
        ..._SLIDER          interactive per-λ trace browser with the fitted poles
        ..._FIGURE          paper figure built from this run's result
        THIRD_EXPORT        (third transition only) write the result JSON + per-λ
                            fit curves that run_files/make_third_transition_figure.py
                            reads, into a fresh fitting_results_<stamp> Dropbox folder

The stand-alone checks at the end (simulation test, benchmark, Liouvillian
spectrum) need no experimental data.
"""
from __future__ import annotations

from pathlib import Path
from mpm_analysis.utils.lab_paths import get_lab_path

# ===========================================================================
# GLOBAL SETTINGS  (apply to every task)
# ===========================================================================

DEBUG = False               # True → few files, small bootstrap, nothing saved
SHOW_PLOTS = True           # plt.show() at the end of the run
SAVE_FIGURES = False        # png into OUTPUT_DIR (paper figures: PDF+SVG+PNG into PAPER_FIGURES_DIR)
SAVE_FULL_ANALYSIS = False  # also write raw pole samples + refine params/cov into the result JSON
SEED: int | None = None     # RNG seed for every bootstrap; None → OS entropy
N_JOBS = -1                 # parallel workers for the data bootstrap (-1 → all cores)
MAX_FILES = 10 if DEBUG else None   # load only the first N λ files (debugging)

# ===========================================================================
# PATHS
# ===========================================================================
# Dropbox root is auto-detected from the hostname (utils/lab_paths.py); add
# your machine there if you get a "hostname not recognised" ValueError.

_ZT = r"Common\QuantumZeno\Paper\zeno_paper_data"
_NEW_SCAN_FOLDER = r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\click_record_with_post_selected_tomography\Omega_BG0_scan"
POST_SELECTED_DROPBOX = Path(get_lab_path(_ZT + r"\scan_experiment_results_10"))
ENS_AVG_DROPBOX = Path(get_lab_path(_ZT + r"\scan_experiment_results_50"))
PARITY_FOLDER = Path(get_lab_path(
    _ZT + r"\click_record_repeated\Omega_BG0_scan\scan_experiment_results_12"
))
# Data folders actually loaded.  Local mirrors hold the same files, BUT the
# ens_avg HMM λ table (scan-10's misfiled results.csv) is only discovered
# inside the Dropbox tree — from the local mirror the ens_avg λ silently
# falls back to the stored JSON value (~11 % lower), and would no longer
# match what make_third_transition_figure.py loads for panel A.
# All 4 iterations of scan 3 merged per point (raw_data_tests/combine_scan_iterations.py)
POST_SELECTED_FOLDER = Path(_NEW_SCAN_FOLDER) / "scan_experiment_results_4" / "scan_iteration"
# POST_SELECTED_FOLDER = Path(_NEW_SCAN_FOLDER) / "scan_experiment_results_3" / "scan_iteration"
ENS_AVG_FOLDER = r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab\Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons\survival_probability\Omega_BG0_scan\scan_experiment_results_0"
# ENS_AVG_FOLDER = Path(r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old_survival_data\scan_experiment_results_50")

# λ axis.  Default: each file's stored parameters.lambda_zeno_estimation.
# Alternatively a per-drive HMM table (hmm_rates_per_drive.csv: converged
# offline HMM of the alpha click records, one row per drive):
# λ = gamma_BG_per_s_mean(drive) / (2·Ω_DG), looked up exactly at each file's
# drive — never interpolated; a drive missing from the table is an error.
# ("auto" would search the scan tree for a run-keyed table — scan-50 only.)
LAMBDA_SOURCE: bool | str | Path = False
if (HMM_DRIVE_LAMBDA := False):
    LAMBDA_SOURCE = Path(_NEW_SCAN_FOLDER) / "scan_experiment_results_3" / "hmm_rates_per_drive.csv"

OUTPUT_DIR = Path(__file__).parent.parent / "mpm_analysis_output"
PAPER_FIGURES_DIR = Path(__file__).parent.parent / "paper_figures"
PAPER_FIGURES_DPI = 600

# ===========================================================================
# TASKS
# ===========================================================================
# Every enabled transition clause appends its settings to JOBS; main runs
# them in order.  Each job is a plain dict — the keys are documented once,
# in the FIRST transition clause; the THIRD clause uses the same keys.

JOBS: list[dict] = []

# ---------------------------------------------------------------------------
# FIRST TRANSITION — post-selected survival probability (scan 10)
# ---------------------------------------------------------------------------
if (RUN_FIRST_TRANSITION := True):
    FIRST = dict(
        label="first_transition",
        data_mode="post_selected",      # "post_selected" | "ens_avg" | "parity"  (loader + folder)
        observables=["S"],       # keys to load; 1 key → classic MPM, >1 → joint MPM

                                        #   (common poles, per-observable amplitudes).
                                        #   post_selected: "S" (stored p_0_t_array),
                                        #   "S_pooled" (segment-pooled rebuild from the raw
                                        #   NPZ, multinomial data bootstrap — written by
                                        #   run_files/raw_data_tests/run_pooled_p0.py),
                                        #   "S_prev_click" / "S_prev_noclick", X/Y/Z …
        order=3,                        # MPM model order (3 for both scans)
        lambda_range=(0, 4),         # λ window to load
        t_max=None,                     # µs; None → full record
        drop_first_point=False,         # drop the by-construction t=0 anchor (S=1) from the
                                        #   data AND the bootstrap statistics.  The anchor is
                                        #   inconsistent with the first-window loss / false
                                        #   clicks (order-3 EP 1.01 vs 1.15 with it, but the EP
                                        #   is poorly resolved without it — see
                                        #   mpm_benchmarks/noise_robustness/v3_transient.py).
    )

    if (FIRST_MPM := True):
        FIRST["mpm"] = dict(
            uncertainty_source="data_bootstrap",
            #   "data_bootstrap"     resample the raw measurement statistics, rerun the
            #                        pencil (+ refinement) per draw — the reported error bars
            #   "fit_covariance"     fast Monte-Carlo propagation of the LSQ covariance
            #   "residual_bootstrap" legacy residual resampling
            n_boot=100,                 # bootstrap draws / covariance samples
            refine_poles=False,         # weighted complex-exponential LSQ after the pencil
            conjugate_pairs=True,       # constrain the refinement to conjugate pairs
            plot_svd=True,              # SVD-vs-λ model-order calibration figure
        )

        # The two transitions on the pole strands — do not conflate them:
        #   state freezing : zero of ksi = (2·Re(e3) − Re(e2)) / (Re(e2) − Re(e3))
        #   exceptional pt : zero of the pair discriminant Delta = c1² − 4·c0
        # Delta is linear in λ through a sqrt EP, so a line fitted over
        # ±half_width around the sign change ("linfit") uses every point in
        # the window; "first" takes the bracketing pair (reads ~0.1 low).
        # Both are refitted inside every bootstrap draw → 16/84 error bar.
        if (FIRST_TRANSITIONS := True):
            FIRST["transitions"] = dict(
                method="linfit",        # "linfit" (recommended) | "first"
                half_width=0.45,        # λ half-window of the line fit
                lambda_max=None,        # e.g. 1.55 to match the critical-point window
                plot_ksi=True,         # ksi(λ) with error bars + the ksi = 0 line
                plot_discriminant=True, # Delta(λ) with error bars + the fitted line
            )

        # sqrt fit of the pole splitting: only λ_c is bounded, a/b are free.
        # λ_c is fitted on the central values (aggregate) and, with
        # per_realization, separately in every bootstrap draw (16/50/84).
        if (FIRST_CRITICAL_POINT := True):
            FIRST["critical_point"] = dict(
                lambda_c_guess=1.2,
                lambda_c_bounds=(0.85, 1.5),   # guess ± 0.15
                lambda_max=1.6,                 # exclude λ above this from the fit
                per_realization=True,
            )

        if (FIRST_SLIDER := False):
            FIRST["slider"] = dict(
                scales={"S": "log", "S_pooled": "log"},   # key → "log"/"linear"
                take_abs=False,         # plot/fit |signal| (e.g. for parity)
            )

        # Paper figure from this run's result.
        #   "discriminant" → 2 panels: A spectrum, B Delta(λ) line fit
        #   "sqrt"         → 3 panels: A spectrum, B/C splittings + sqrt fit
        #                    (needs the critical point above)
        if (FIRST_FIGURE := False):
            FIRST["figure"] = dict(
                layout="discriminant",
                lambda_max=1.5,                     # λ cutoff of every panel
                fit_validity=1.2,
                real_axis_scale="log",
                real_axis_ticks=None,
                real_split_axis_ticks=[0.0, 1.0, 2.0, 3.0],
            )

    JOBS.append(FIRST)

# ---------------------------------------------------------------------------
# THIRD TRANSITION — ensemble-average D population (scan 50)
# ---------------------------------------------------------------------------
# Same keys as FIRST.  D alone at order 3: the joint D/G/B/error pencil is a
# B-dominated truncation bias (mpm_benchmarks/noise_robustness/v3_channelcheck.py).
if (RUN_THIRD_TRANSITION := False):
    THIRD = dict(
        label="third_transition",
        data_mode="ens_avg",
        observables=["D","G","B","error"],              # "D" | "G" | "B" | "error", or several for a joint fit
        order=3,
        lambda_range=(0, 1.65),
        t_max=None,
        drop_first_point=False,         # the t=0 point is an inconsistent anchor here too
                                        #   (λ₃ 1.13 → 1.03 without it)
    )

    if (THIRD_MPM := True):
        THIRD["mpm"] = dict(
            uncertainty_source="data_bootstrap",
            n_boot=1000,
            refine_poles=False,
            conjugate_pairs=True,
            plot_svd=True,
        )

        if (THIRD_TRANSITIONS := True):
            THIRD["transitions"] = dict(
                method="linfit",
                half_width=0.15,
                lambda_max=None,
                plot_ksi=False,
                plot_discriminant=True,
            )

        if (THIRD_CRITICAL_POINT := False):
            THIRD["critical_point"] = dict(
                lambda_c_guess=1.1,
                lambda_c_bounds=(0.95, 1.25),   # guess ± 0.15
                lambda_max=1.7,
                per_realization=True,
            )

        if (THIRD_SLIDER := False):
            THIRD["slider"] = dict(
                scales={},                      # linear axes
                take_abs=False,
            )

        if (THIRD_FIGURE := False):
            THIRD["figure"] = dict(
                layout="discriminant",
                lambda_max=1.5,
                fit_validity=1.3,
                real_axis_scale="linear",
                real_axis_ticks=[0.0, 1.0, 2.0],
                real_split_axis_ticks=[0.0, 0.6, 1.2],
            )

        # Data files for run_files/make_third_transition_figure.py, written
        # into a NEW  <parent>/fitting_results_<UTC stamp>/  folder (the old
        # folders are never touched; the figure script picks the newest):
        #   mpm_analysis_results_*.json      panel B — this run's result, with
        #                                    the sqrt fit if THIRD_CRITICAL_POINT ran
        #   <scan file>_mpm_fit.json         panel A — one per λ: the modal fit
        #                                    Σ A_k exp(p_k t) of `observable` on a
        #                                    smooth grid (times_smooth / fit_curve),
        #                                    poles = this run's, amplitudes re-solved
        #                                    on that trace; matched by kHz in the name
        if (THIRD_EXPORT := False):
            THIRD["export"] = dict(
                parent=ENS_AVG_DROPBOX,     # where the figure script reads from
                observable="D",             # trace the fit curve describes — must be
                                            #   what panel A plots: the loader's
                                            #   "survival" key = D_probability
                n_smooth=300,               # points of the smooth time grid
            )

    JOBS.append(THIRD)

# ---------------------------------------------------------------------------
# Stand-alone checks (no experimental data, no settings of their own)
# ---------------------------------------------------------------------------
if (RUN_SIMULATION_TEST := False):
    pass    # small synthetic simulation through MPM + bootstrap

if (RUN_BENCHMARK := False):
    pass    # robustness benchmark with injected shot noise

if (RUN_LIOUVILLIAN_SPECTRUM := False):
    pass    # 4×4 Liouvillian eigenvalues vs λ at k=π


# ===========================================================================
# IMPORTS
# ===========================================================================

import json
import shutil
from datetime import datetime, timezone

import numpy as np
import matplotlib.pyplot as plt
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep
from mpm_analysis.analysis.ksi_crossing import (
    KsiCrossingPipeline, discriminant_crossing_samples, discriminant_from_poles,
    linfit_zero_crossing, find_zero_crossings,
)
from mpm_analysis.analysis.matrix_pencil import (
    solve_linear_amplitudes, reconstruct_signal,
)
from mpm_analysis.analysis.critical_point import fit_sqrt_per_realization
from mpm_analysis.data_types.pole_samples import PoleSamples
from mpm_analysis.plotting.ksi_plot import plot_ksi_curve, plot_discriminant_curve
from mpm_analysis.io.analysis_json import (
    augment_json_with_ksi, augment_json_with_transitions,
)
from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.liouvillian_spectrum import LiouvillianSpectrum
from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader, ParityLoader
from mpm_analysis.simulation.noise import ShotNoise
from mpm_analysis.plotting.exploratory import plot_spectrum_panels, plot_time_traces_slider
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap
from mpm_analysis.io.file_naming import build_figure_filename, save_result


# ===========================================================================
# HELPERS
# ===========================================================================

# Strong references to interactive widgets: matplotlib only weak-references
# slider callbacks, so a dropped slider keeps drawing but stops responding.
_LIVE_WIDGETS: list = []


def _show_and_save(fig, figure_tag: str, *, data_mode: str = "unknown",
                   scan_number: int | None = None) -> None:
    """Save the figure as png into OUTPUT_DIR (if enabled) and show it."""
    if SAVE_FIGURES and not DEBUG:
        filename = build_figure_filename(
            figure_tag=figure_tag, data_mode=data_mode,
            scan_number=scan_number, extension="png",
        )
        path = OUTPUT_DIR / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"  Figure saved: {path.name}")
    if SHOW_PLOTS:
        plt.show(block=False)


def _result_tags(result) -> tuple[str, int | None]:
    """(data_mode, scan_number) of a result, for figure file names."""
    data_mode = str(result.metadata.get("data_mode",
                                        result.metadata.get("source_tag", "unknown")))
    scan_number = result.metadata.get("scan_number")
    return data_mode, (int(scan_number) if scan_number is not None else None)


def _make_source(data_mode: str):
    """Resolve a data mode to (loader, folder)."""
    if data_mode == "ens_avg":
        return EnsembleAverageLoader(csv_path=LAMBDA_SOURCE), Path(ENS_AVG_FOLDER)
    if data_mode == "post_selected":
        return PostSelectedLoader(csv_path=LAMBDA_SOURCE), Path(POST_SELECTED_FOLDER)
    if data_mode == "parity":
        return ParityLoader(), Path(PARITY_FOLDER)
    raise ValueError(f"Unknown data_mode '{data_mode}'; expected "
                     f"'ens_avg', 'post_selected' or 'parity'.")


def _resolve_uncertainty_source(source: str, data_mode: str) -> str:
    """Validate the uncertainty source and apply per-mode fallbacks."""
    if source not in ("data_bootstrap", "fit_covariance", "residual_bootstrap"):
        raise ValueError(f"Unknown uncertainty_source '{source}'.")
    if source == "data_bootstrap" and data_mode not in ("ens_avg", "post_selected"):
        print(f"Warning: data bootstrap has no measurement statistics for "
              f"'{data_mode}' — falling back to residual_bootstrap.")
        source = "residual_bootstrap"
    return source


def _build_steps(job: dict) -> tuple[list, str]:
    """Assemble MPM → (Refine) → Bootstrap steps from a job's settings."""
    mpm = job["mpm"]
    src = _resolve_uncertainty_source(mpm["uncertainty_source"], job["data_mode"])
    n_boot = min(mpm["n_boot"], 20) if DEBUG else mpm["n_boot"]
    steps = [
        MPMStep(order=job["order"], observable_key=job["observables"],
                plot_svd=mpm["plot_svd"], n_sv=10, save_dir=OUTPUT_DIR),
    ]
    if mpm["refine_poles"] or src == "fit_covariance":
        # fit_covariance needs the refinement covariance → RefineStep forced on
        steps.append(RefineStep(observable_key=job["observables"],
                                conjugate_pairs=mpm["conjugate_pairs"]))
    mode = {"data_bootstrap": "data", "fit_covariance": "parametric",
            "residual_bootstrap": "residual"}[src]
    steps.append(BootstrapStep(n_boot=n_boot, seed=SEED, mode=mode, n_jobs=N_JOBS))
    return steps, src


# ===========================================================================
# TASKS
# ===========================================================================

def run_mpm(job: dict, loader, folder: Path):
    """MPM + bootstrap for one job.  Returns (result, saved JSON path)."""
    data_mode, observables = job["data_mode"], job["observables"]
    print(f"\n=== MPM ({data_mode}, observables={observables}, order={job['order']}) ===")
    steps, src = _build_steps(job)
    pipeline = MPMPipeline.from_loader(
        loader, folder,
        data_mode=data_mode,
        observables=observables,
        steps=steps,
        lambda_range=job["lambda_range"],
        max_files=MAX_FILES,
        t_max=job["t_max"],
        drop_first_point=job["drop_first_point"],
        uncertainty_source=src,
    )
    result = pipeline.run()

    fig, _ = plot_spectrum_panels(result, title=data_mode)
    fig.tight_layout()
    tag_mode, scan_number = _result_tags(result)
    _show_and_save(fig, "spectrum", data_mode=tag_mode, scan_number=scan_number)

    saved_path = None
    if not DEBUG:
        saved_path = pipeline.save(OUTPUT_DIR, full=SAVE_FULL_ANALYSIS)
        print(f"Saved: {saved_path.name}")
    return result, saved_path


def run_transitions(result, job: dict, json_path=None):
    """The two transitions on the pole strands, each with its own error bar.

    state freezing : lam(ksi=0), ksi = (2·Re(e3) − Re(e2)) / (Re(e2) − Re(e3))
    exceptional pt : lam_EP, zero of the pair discriminant Delta = c1² − 4·c0

    Both errors are per-realization: the crossing (or the line fit through it)
    is redone inside every bootstrap draw and the 16/84 spread of those values
    is the error bar.  The central value comes from the unresampled poles.
    """
    cfg = job["transitions"]
    method, half_width, lambda_max = cfg["method"], cfg["half_width"], cfg["lambda_max"]
    print(f"\n=== Transitions ({job['label']}) ===")
    data_mode, scan_number = _result_tags(result)
    obs_tag = "+".join(job["observables"])
    transitions = {}

    # --- state freezing: ksi = 0 --------------------------------------------
    ksi = KsiCrossingPipeline(result, lambda_max=lambda_max).run()
    ksi_central = ksi.lambda_ksi_zero
    if ksi.crossing_quantiles is not None:
        q16, q50, q84 = ksi.crossing_quantiles
        centre = ksi_central if ksi_central is not None else q50
        print(f"  STATE FREEZING  lam(ksi=0) = {centre:.4f} +/- {(q84 - q16) / 2:.4f}   "
              f"[bootstrap 16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}]")
    elif ksi_central is not None:
        print(f"  STATE FREEZING  lam(ksi=0) = {ksi_central:.4f}  "
              f"(no bootstrap draws — no error bar)")
    else:
        print("  STATE FREEZING  ksi has no zero crossing.")
    if cfg["plot_ksi"]:
        ax = plot_ksi_curve(ksi, title=f"State freezing: ksi = 0  ({data_mode}, {obs_tag})")
        _show_and_save(ax.get_figure(), "state_freezing_ksi",
                       data_mode=data_mode, scan_number=scan_number)

    # --- exceptional point: Delta = 0 ---------------------------------------
    # Delta of the EP pair, followed along λ: past the EP it stays on the two
    # real poles that continue the pair, not on whichever two are closest.
    d_quantiles = d_med = d_lo = d_hi = None
    raw = result.poles.raw_samples
    if raw is not None:
        d_cross, d_med, d_lo, d_hi = discriminant_crossing_samples(
            raw, lambda_max=lambda_max, method=method, half_width=half_width,
        )
        if np.any(np.isfinite(d_cross)):
            q16, q50, q84 = np.nanpercentile(d_cross, [16, 50, 84])
            d_quantiles = (q16, q50, q84)
            transitions["delta_crossing_samples"] = d_cross
            transitions["delta_crossing_quantiles"] = [q16, q50, q84]
            transitions["delta_crossing_method"] = method
        else:
            print("  EXCEPTIONAL PT  no bootstrap realization had a Delta crossing.")
    else:
        print("  EXCEPTIONAL PT  no pole draws stored — the discriminant error bar "
              "needs a bootstrap/covariance run (n_boot >= 1).")

    lam_d = np.asarray(result.lambda_values, dtype=float)
    central = -result.poles.decay_rates + 1j * result.poles.frequencies
    d_central = discriminant_from_poles(central)
    if lambda_max is not None:
        m = lam_d <= lambda_max
        lam_d, d_central = lam_d[m], d_central[m]
    # Error bar of Delta = the 16/84 spread of the draws (None without them):
    # it weights the line fit and is drawn on the points.
    has_band = d_med is not None and len(d_med) == len(lam_d)
    d_sigma = (d_hi - d_lo) / 2 if has_band else None
    lam_ep, fit = linfit_zero_crossing(lam_d, d_central, sigma=d_sigma,
                                       half_width=half_width, return_fit=True)
    first = find_zero_crossings(lam_d, d_central)
    if d_quantiles is not None:
        q16, q50, q84 = d_quantiles
        print(f"  EXCEPTIONAL PT  lam_EP     = {lam_ep:.4f} +/- {(q84 - q16) / 2:.4f}   "
              f"[bootstrap 16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}, "
              f"{np.mean(np.isnan(transitions['delta_crossing_samples'])):.1%} without a crossing]")
    elif np.isfinite(lam_ep):
        print(f"  EXCEPTIONAL PT  lam_EP     = {lam_ep:.4f}  (no bootstrap draws — no error bar)")
    print(f"                  ({'weighted ' if fit.get('weighted') else ''}{method} "
          "on Delta; raw first sign change at "
          + (f"{first[0]:.4f})" if first else "none)"))
    if np.isfinite(lam_ep):
        transitions["delta_linfit_central"] = lam_ep

    if cfg["plot_discriminant"]:
        band = (d_lo, d_hi) if has_band else (None, None)
        ax_d = plot_discriminant_curve(
            lam_d, d_central, fit=fit, lambda_ep=lam_ep,
            first_crossing=first[0] if first else None,
            crossing_quantiles=d_quantiles, delta_lower=band[0], delta_upper=band[1],
            title=f"Exceptional point: Delta = 0  ({data_mode}, {obs_tag})",
        )
        _show_and_save(ax_d.get_figure(), "exceptional_point_discriminant",
                       data_mode=data_mode, scan_number=scan_number)

    if json_path is not None and not DEBUG:
        augment_json_with_ksi(json_path, ksi)
        if transitions:
            augment_json_with_transitions(json_path, transitions)
    return ksi


def run_critical_point(result, job: dict, json_path=None):
    """sqrt fit of the pole splitting → lambda_c (aggregate + per realization)."""
    cfg = job["critical_point"]
    guess, bounds, lambda_max = cfg["lambda_c_guess"], cfg["lambda_c_bounds"], cfg["lambda_max"]
    print(f"\n=== Critical-point fit ({job['label']}) ===")
    cp = CriticalPointPipeline(result, lambda_max=lambda_max)
    fit = cp.fit(
        lambda_c_guess=guess, soft_bounds=True, lambda_c_bounds=bounds,
        window_left=50, window_right=15,
    )
    print(f"  lam_c (aggregate fit) = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}   "
          f"[guess {guess}, bounds {bounds}]")
    print(f"  chi2_red = {fit.chi2_reduced:.3f}  R2 = {fit.r_squared:.4f}")

    if cfg["per_realization"] and result.poles.raw_samples is not None:
        raw = result.poles.raw_samples
        if lambda_max is not None:
            m = raw.lambda_values <= lambda_max
            raw = PoleSamples(
                samples=raw.samples[m],
                observable_key=raw.observable_key,
                lambda_values=raw.lambda_values[m],
                param_samples=(raw.param_samples[m]
                               if raw.param_samples is not None else None),
            )
        lam_c_samples = fit_sqrt_per_realization(
            raw, lambda_c_guess=guess, lambda_c_bounds=bounds,
            window_left=50, window_right=15, omega_s=result.omega_s,
        )
        if np.any(np.isfinite(lam_c_samples)):
            q16, q50, q84 = np.nanpercentile(lam_c_samples, [16, 50, 84])
            print(f"  lam_c (per realization) = {q50:.4f} "
                  f"+{q84 - q50:.4f} / -{q50 - q16:.4f}  "
                  f"({np.mean(np.isnan(lam_c_samples)):.1%} of fits failed)")
            if json_path is not None and not DEBUG:
                augment_json_with_transitions(json_path, {
                    "sqrt_lambda_c_samples": lam_c_samples,
                    "sqrt_lambda_c_quantiles": [q16, q50, q84],
                })
        else:
            print("  Per-realization sqrt fit: no realization converged.")

    # 3-panel appendix-style figure; the fit lands on the result so the
    # "sqrt" figure layout can reuse it.
    cp.result.critical_point_sqrt_fit = fit
    fig = plot_appendix_bootstrap(
        cp.result, lambda_max=lambda_max, fit_validity_limit=lambda_max,
        real_axis_ticks=[0.0, 1, 2], real_split_axis_ticks=[0.0, 0.6, 1.2],
    )
    data_mode, scan_number = _result_tags(cp.result)
    _show_and_save(fig, "critical_point_appendix",
                   data_mode=data_mode, scan_number=scan_number)

    if json_path is not None and not DEBUG:
        path = cp.save(json_path)
        print(f"Saved: {path.name}")
    return fit


def run_slider_plot(result, job: dict, loader, folder: Path):
    """Interactive per-λ trace browser; the fit overlay uses the result's poles."""
    cfg = job["slider"]
    records = loader.load(
        folder,
        observables=job["observables"],
        lambda_range=job["lambda_range"],
        max_files=MAX_FILES,
        t_max=job["t_max"],
        drop_first_point=job["drop_first_point"],
    )
    fixed_poles = {
        float(lam): -result.poles.decay_rates[i] + 1j * result.poles.frequencies[i]
        for i, lam in enumerate(result.lambda_values)
    }
    slider = plot_time_traces_slider(
        records,
        observable_key=job["observables"],
        scales=cfg["scales"],
        take_abs=cfg["take_abs"],
        mpm_order=job["order"],
        fixed_poles=fixed_poles,
    )
    _LIVE_WIDGETS.append(slider)
    return slider


def run_figure(result, job: dict):
    """Paper transition figure from this run's result (see the clause comment)."""
    from mpm_analysis.plotting.paper_figures.first_transition_main import (
        plot_first_transition_figure,
    )
    cfg = job["figure"]
    layout = cfg["layout"]
    print(f"\n=== Paper figure ({job['label']}, {layout}) ===")
    if layout == "sqrt" and getattr(result, "critical_point_sqrt_fit", None) is None:
        print("[figure] result carries no sqrt fit — enable the critical point "
              "(or switch to the 'discriminant' layout).")
        return None
    half_width = job.get("transitions", {}).get("half_width", 0.15)
    fig = plot_first_transition_figure(
        result,
        layout=layout,
        lambda_max=cfg["lambda_max"],
        fit_validity_limit=cfg["fit_validity"],
        real_axis_scale=cfg["real_axis_scale"],
        real_axis_ticks=cfg["real_axis_ticks"],
        real_split_axis_ticks=cfg["real_split_axis_ticks"],
        discriminant_half_width=half_width,
    )
    data_mode, scan_number = _result_tags(result)
    # Paper figures: own folder, PDF + SVG + high-res PNG, stem naming the
    # transition, layout, data mode, scan, observables and order + UTC stamp.
    if SAVE_FIGURES and not DEBUG:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        stem = build_figure_filename(
            figure_tag=(f"{job['label']}_{layout}_{'+'.join(job['observables'])}"
                        f"_order{job['order']}"),
            data_mode=data_mode, scan_number=scan_number, extension="",
        ).rstrip(".")
        save_figure_pdf_svg(fig, stem, PAPER_FIGURES_DIR, dpi=PAPER_FIGURES_DPI, png=True)
    if SHOW_PLOTS:
        plt.show(block=False)
    return fig


def run_export_figure_data(result, job: dict, json_path, loader, folder: Path):
    """Write the files run_files/make_third_transition_figure.py reads.

    A fresh ``<parent>/fitting_results_<UTC stamp>/`` folder receives this
    run's result JSON (panel B) and one ``<scan file>_mpm_fit.json`` per λ
    (panel A overlays): ``times_smooth`` / ``fit_curve`` of the modal model
    with this run's poles at that λ and amplitudes re-solved on the exported
    observable's own trace — the same construction as the slider overlay.
    Returns the folder.
    """
    cfg = job["export"]
    key = cfg["observable"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = Path(cfg["parent"]) / f"fitting_results_{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"\n=== Export for make_third_transition_figure.py ({job['label']}) ===")

    # Panel B: the result JSON (already augmented with the sqrt fit if the
    # critical point ran; DEBUG never saved one, so save a fresh copy here).
    if json_path is not None:
        result_name = Path(shutil.copy2(json_path, out / Path(json_path).name)).name
    else:
        result_name = save_result(result, out, full=False).name
    print(f"  {result_name}")

    # Panel A: per-λ fit curves of `key` with this run's poles.
    records = loader.load(
        folder, observables=[key],
        lambda_range=job["lambda_range"], max_files=MAX_FILES,
        t_max=job["t_max"], drop_first_point=job["drop_first_point"],
    )
    lam_res = np.asarray(result.lambda_values, dtype=float)
    poles_res = -result.poles.decay_rates + 1j * result.poles.frequencies
    n_written = n_skipped = 0
    for rec in records:
        i = int(np.argmin(np.abs(lam_res - rec.lambda_val)))
        poles = poles_res[i]
        if abs(lam_res[i] - rec.lambda_val) > 1e-6 or not np.all(np.isfinite(poles)):
            n_skipped += 1
            continue
        amps = solve_linear_amplitudes(rec.t, rec.signal, poles)
        t_smooth = np.linspace(float(rec.t[0]), float(rec.t[-1]), int(cfg["n_smooth"]))
        payload = {
            "times_smooth": t_smooth.tolist(),
            "fit_curve": reconstruct_signal(t_smooth, poles, amps).tolist(),
            "lambda": float(rec.lambda_val),
            "observable": key,
            "poles_re_1_per_us": np.real(poles).tolist(),
            "poles_im_rad_per_us": np.imag(poles).tolist(),
            "amplitudes_re": np.real(amps).tolist(),
            "amplitudes_im": np.imag(amps).tolist(),
            "mpm_order": int(job["order"]),
            "lambda_source": loader.lambda_source,
            "analysis_json": result_name,
            "original_file": str(folder / rec.source_file),
        }
        with open(out / f"{Path(rec.source_file).stem}_mpm_fit.json", "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        n_written += 1
    print(f"  {n_written} *_mpm_fit.json files ({key}, {cfg['n_smooth']} points)"
          + (f", {n_skipped} records without a matching λ / finite poles" if n_skipped else ""))
    print(f"  Folder: {out}")
    return out


def run_job(job: dict) -> None:
    """One transition: MPM, then every nested task the clause enabled."""
    print(f"\n##### {job['label']}  ({job['data_mode']}) #####")
    if "mpm" not in job:
        print("  MPM disabled — nothing to do (every other step needs its result).")
        return
    loader, folder = _make_source(job["data_mode"])
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        return
    result, json_path = run_mpm(job, loader, folder)
    if "transitions" in job:
        run_transitions(result, job, json_path)
    if "critical_point" in job:
        run_critical_point(result, job, json_path)   # before the figure: "sqrt" layout reuses the fit
    if "slider" in job:
        run_slider_plot(result, job, loader, folder)
    if "figure" in job:
        run_figure(result, job)
    if "export" in job:
        run_export_figure_data(result, job, json_path, loader, folder)


# ---- stand-alone checks ---------------------------------------------------

def run_simulation_test():
    """Quick end-to-end test on synthetic data (no real data needed)."""
    print("\n=== Simulation test ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=30, simulation_duration=25.0, min_alpha=0.1, max_alpha=2.0,
    )
    records = sim.to_observable_records()
    print(f"  Simulated {len(records)} records")
    steps = [MPMStep(order=3),
             BootstrapStep(n_boot=20 if DEBUG else 100, seed=SEED, mode="residual")]
    result = MPMPipeline.from_records(records, steps=steps).run()
    print(f"  Poles shape: {result.poles.decay_rates.shape}")
    fig, _ = plot_spectrum_panels(result, title="simulation")
    fig.tight_layout()
    _show_and_save(fig, "spectrum", data_mode="simulation")
    return result


def run_benchmark():
    """Analysis robustness with injected shot noise."""
    print("\n=== Benchmark ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=20, simulation_duration=40.0, min_alpha=0.1, max_alpha=2.0,
    )
    bench = BenchmarkPipeline(
        clean_records=sim.to_observable_records(),
        noise_models=[ShotNoise(n_shots=6 * 3e4, seed=0)],
        analysis_steps=[MPMStep(order=3),
                        BootstrapStep(n_boot=20 if DEBUG else 100, seed=SEED)],
        n_trials=5 if DEBUG else 20,
    )
    result = bench.run(lambda_c_guess=1.25)
    print(f"  lam_c bias = {result.lambda_c_bias:.4f}  std = {result.lambda_c_std:.4f}")
    return result


def run_liouvillian_spectrum():
    """Eigenvalues of the 4×4 Liouvillian vs λ at k=π."""
    print("\n=== Liouvillian spectrum (k=π) ===")
    fig, _ = LiouvillianSpectrum().plot()
    fig.suptitle("4×4 Liouvillian eigenvalues vs λ  (k=π)", fontsize=13)
    _show_and_save(fig, "liouvillian_spectrum", data_mode="theory")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for job in JOBS:
        run_job(job)

    if RUN_SIMULATION_TEST:
        run_simulation_test()
    if RUN_BENCHMARK:
        run_benchmark()
    if RUN_LIOUVILLIAN_SPECTRUM:
        run_liouvillian_spectrum()

    print("\nDone.")
    if SHOW_PLOTS:
        plt.show()
