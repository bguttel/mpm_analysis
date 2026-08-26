from __future__ import annotations

from pathlib import Path
from mpm_analysis.utils.lab_paths import get_lab_path

# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================

DEBUG = False          # True → small n_boot, few files, no saving
ORDER = 3              # MPM model order
N_LAMBDA_TEST = None   # set to e.g. 10 to load only first 10 files (debug)
LAMBDA_RANGE = (0,1.4)    # e.g. (0.1, 1.6) — restrict lambda window
SEED: int | None = None  # RNG seed (all uncertainty modes); None → truly random (OS entropy)
T_MAX = None            # max time to load from experimental records (µs), default is None

SHOW_PLOTS  = True     # plt.show() after each task
SAVE_FIGURES = True # save figures as png into OUTPUT_DIR
SAVE_FULL_ANALYSIS = True  # write raw pole samples + refine params/cov + ksi into the output JSON

if DEBUG:
    N_LAMBDA_TEST = 10

# ===========================================================================
# PATHS
# ===========================================================================
# Dropbox root is auto-detected from the current hostname — see
# utils/lab_paths.py.  Add your machine there if you get a "hostname not
# recognised" ValueError.

_ZT = r"Common\QuantumZeno\Paper\zeno_paper_data"

# ---- Ensemble-average scan ----
ENS_AVG_FOLDER = get_lab_path(
    _ZT+
    r"\scan_experiment_results_50",
)

# ---- Post-selected scan + CSV for HMM lambda scale ----
_PS = r"Common\QuantumZeno\Paper\zeno_paper_data\scan_experiment_results_10"

_PARITY = _ZT + r"\click_record_repeated\Omega_BG0_scan\scan_experiment_results_12"

# POST_SELECTED_FOLDER = get_lab_path(_PS)
POST_SELECTED_FOLDER = r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old data\scan_experiment_results_10"
ENS_AVG_FOLDER = r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old_survival_data\scan_experiment_results_50"
PARITY_FOLDER = get_lab_path(_PARITY)
ENSEMBLE_AVERAGE_CSV = get_lab_path(_PS, "results.csv")
ENSEMBLE_AVERAGE_CSV = None
# HMM results CSV; set to None to use lambda_zeno_estimation from JSON

# ---- HMM lambda correction (post-selected) ----
# results.csv holds Gamma_up_3_states_hmm (= alpha) per OBG0 drive setting;
# the loader converts it to lambda = alpha / (2 * Omega_DG).  Only the
# scan_experiment_results_10 folder ships one, so auto-detection applies the
# correction to that scan and to nothing else.
#   None          -> auto-detect "results.csv" inside POST_SELECTED_FOLDER
#   Path(...)     -> force a specific CSV
#   False         -> disable, always use lambda_zeno_estimation from the JSON
POST_SELECTED_CSV = r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old data\scan_experiment_results_10\results.csv"
POST_SELECTED_CSV = False
# ---- Output directory ----
OUTPUT_DIR = Path(__file__).parent.parent / "mpm_analysis_output"

# ===========================================================================
# TASKS
# ===========================================================================

# ---- MPM + bootstrap (unified for all data modes) ----
RUN_MPM     = True
DATA_MODE   = "ens_avg"   # "ens_avg" | "post_selected" | "parity"
OBSERVABLES = ["D","G"]  # keys to load from the records (must be present in the data)


# OBSERVABLES = ["SX","X"]      # 1 key → classic MPM; >1 keys → joint MPM
                                # (common poles, per-observable amplitudes)
PLOT_SVD    = True             # SVD-vs-λ model-order calibration figure

# ---- Pole refinement / uncertainty source ----
REFINE_POLES       = False    # weighted complex-exponential LSQ smoothing after MPM
CONJUGATE_PAIRS    = None    # None → auto: True for "post_selected"/"ens_avg", False otherwise
UNCERTAINTY_SOURCE = "residual_bootstrap"
# "data_bootstrap"     — DEFAULT: resample the raw measurement statistics
#                        (multinomial D/G/B/error for ens_avg, binomial
#                        survival/X/Z for post_selected), rerun MPM + LSQ
#                        refinement per draw.  Final reported error bars.
# "fit_covariance"     — fast approximation: Monte-Carlo propagation of the
#                        LSQ parameter covariance (NOT a bootstrap).
# "residual_bootstrap" — legacy residual resampling.
# One choice for the whole run: spectrum errorbars AND transition crossings.
N_BOOTSTRAP   = 100   # data/residual bootstrap realizations
N_COV_SAMPLES = 100    # Monte-Carlo covariance-propagation samples
N_JOBS        = -1      # parallel workers for the data bootstrap (-1 → all cores)
if DEBUG:
    N_BOOTSTRAP = 50

# ---- Ksi zero crossing (2·Re(e3) − Re(e2)) + discriminant crossing ----
RUN_KSI_CROSSING = False
KSI_LAMBDA_MAX   = None      # e.g. 1.55 to match the critical-point window

# ---- Interactive per-λ trace browser ----
RUN_SLIDER_PLOT  = True
PLOT_OBSERVABLES = OBSERVABLES          # which keys the slider shows
PLOT_SCALES      = {"survival": "log"}  # key → "log"/"linear"; unlisted → linear
PLOT_TAKE_ABS    = False                # plot/fit |signal| (e.g. for parity)

# ---- Other tasks ----
RUN_CRITICAL_POINT       = True  # Fit sqrt model (fresh MPM result or existing JSON)
RUN_SIMULATION_TEST      = False   # Small synthetic simulation (no real data needed)
RUN_BENCHMARK            = False  # Robustness benchmark with noise injection
RUN_LIOUVILLIAN_SPECTRUM = False  # 4×4 Liouvillian eigenvalues vs λ at k=π

# For RUN_CRITICAL_POINT: path to a previously saved analysis JSON
# CRITICAL_POINT_JSON = Path(
#     r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code"
#     r"\mpm_analysis_results_ens_avg_scan_experiment_results_50_20260129_225403.json"
# )

CRITICAL_POINT_JSON = None  # set to None to require a fresh MPM result for critical point fit
CP_LAMBDA_MAX  = 1.75    # exclude lambda above this for the fit

# Soft guess: only λ_c is bounded (CP_LAMBDA_C_BOUNDS), a/b essentially free.
# CP_SOFT_GUESS=False falls back to the per-mode preset (±10 % hard box).
CP_SOFT_GUESS      = True
CP_LAMBDA_C_BOUNDS = (1.0, 1.5)
CP_LAMBDA_C_GUESS  = 1.25   # p0 for λ_c (always around 1–1.5)

# Per-realization transition extraction: λ_c fitted separately for every
# bootstrap realization (16/50/84 from the distribution), IN ADDITION to the
# aggregate fit from the central values + error bars above.  Both reported.
CP_PER_REALIZATION = True


# ===========================================================================
# IMPORTS
# ===========================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep, MPMRRHAStep, RefineStep
from mpm_analysis.analysis.ksi_crossing import (
    KsiCrossingPipeline, discriminant_crossing_samples,
)
from mpm_analysis.analysis.critical_point import fit_sqrt_per_realization
from mpm_analysis.data_types.pole_samples import PoleSamples
from mpm_analysis.plotting.ksi_plot import plot_ksi_curve
from mpm_analysis.io.analysis_json import (
    augment_json_with_ksi, augment_json_with_transitions,
)
from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.ensemble_average import EnsembleAverageDynamics
from mpm_analysis.simulation.liouvillian_spectrum import LiouvillianSpectrum
from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader, ParityLoader
from mpm_analysis.simulation.noise import ShotNoise
from mpm_analysis.plotting.exploratory import (
    plot_eigenvalue_spectrum, plot_time_traces_slider
)
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap
from mpm_analysis.analysis.critical_point_guesses import get_guess, list_presets
from mpm_analysis.io.file_naming import build_figure_filename


# ===========================================================================
# PLOT HELPER
# ===========================================================================

# Strong references to interactive widgets (sliders).  Matplotlib holds only
# weak references to widget callbacks, so anything dropped here stops
# responding while still being drawn.
_LIVE_WIDGETS: list = []


def _show_and_save(
    fig,
    figure_tag: str,
    *,
    data_mode: str = "unknown",
    scan_number: int | None = None,
) -> None:
    """Optionally save figure to OUTPUT_DIR and/or show it."""
    import matplotlib.pyplot as plt
    if SAVE_FIGURES and not DEBUG:
        filename = build_figure_filename(
            figure_tag=figure_tag,
            data_mode=data_mode,
            scan_number=scan_number,
            extension="png",
        )
        path = OUTPUT_DIR / filename
        fig.savefig(path, bbox_inches="tight")
        print(f"  Figure saved: {path.name}")
    if SHOW_PLOTS:
        plt.show(block=False)


def _plot_mpm_result(records, result, tag: str) -> None:
    """Standard 2-panel diagnostic: raw data, decay rates, frequencies."""
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(tag, fontsize=14)

    plot_eigenvalue_spectrum(result, ax=axes[0], mode="decay")
    axes[0].set_title("Decay rates", fontsize=12)
    axes[0].tick_params(labelsize=10)

    plot_eigenvalue_spectrum(result, ax=axes[1], mode="frequency")
    axes[1].set_title("Frequencies", fontsize=12)
    axes[1].tick_params(labelsize=10)

    for ax in axes:
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    data_mode = str(result.metadata.get("data_mode",
                                        result.metadata.get("source_tag", tag)))
    scan_number = result.metadata.get("scan_number")
    if scan_number is not None:
        scan_number = int(scan_number)
    _show_and_save(
        fig,
        "spectrum",
        data_mode=data_mode,
        scan_number=scan_number,
    )


# ===========================================================================
# TASKS
# ===========================================================================

def _resolve_hmm_csv(folder, setting):
    """Locate the HMM results.csv that rescales lambda for this folder.

    ``setting`` is the config value: ``False`` disables the correction,
    an explicit path forces one, ``None`` auto-detects ``results.csv`` beside
    the data.  A CSV without the ``Gamma_up_3_states_hmm`` column is ignored,
    so a bare ``results_df.csv`` never triggers a bogus rescale.
    """
    if setting is False:
        print("[post_selected] HMM correction disabled, "
              "using lambda_zeno_estimation from JSON")
        return None
    csv_path = Path(setting) if setting else Path(folder) / "results.csv"
    if not csv_path.exists():
        print(f"[post_selected] no {csv_path.name} in {Path(folder).name}, "
              "using lambda_zeno_estimation from JSON")
        return None
    import csv as _csv
    with open(csv_path, newline="") as f:
        cols = _csv.DictReader(f).fieldnames or []
    if "Gamma_up_3_states_hmm" not in cols or "filename" not in cols:
        print(f"[post_selected] {csv_path.name} lacks filename/"
              "Gamma_up_3_states_hmm — using lambda_zeno_estimation from JSON")
        return None
    print(f"[post_selected] HMM lambda correction from {csv_path.name} "
          "(lambda = Gamma_up_3_states_hmm / 2*Omega_DG)")
    return csv_path


def _make_source(name: str):
    """Resolve a data mode to (loader, folder, data_mode, guess_preset)."""
    if name == "ens_avg":
        csv = ENSEMBLE_AVERAGE_CSV if (ENSEMBLE_AVERAGE_CSV and ENSEMBLE_AVERAGE_CSV.exists()) else None
        if csv is None:
            print("[ens_avg] CSV not found, using lambda_zeno_estimation from JSON")
        return EnsembleAverageLoader(csv_path=csv), ENS_AVG_FOLDER, "ens_avg", "ens_avg_experimental"
    if name == "post_selected":
        csv = _resolve_hmm_csv(POST_SELECTED_FOLDER, POST_SELECTED_CSV)
        return (PostSelectedLoader(csv_path=csv), POST_SELECTED_FOLDER,
                "post_selected", "post_selected_experimental")
    if name == "parity":
        return ParityLoader(), PARITY_FOLDER, "parity", None
    raise ValueError(f"Unknown DATA_MODE '{name}'; expected "
                     f"'ens_avg', 'post_selected' or 'parity'.")


def _resolve_uncertainty_source(data_mode: str) -> str:
    """Validate UNCERTAINTY_SOURCE and apply per-mode fallbacks."""
    src = UNCERTAINTY_SOURCE
    if src == "fit_errorbars":     # legacy name
        print("Note: UNCERTAINTY_SOURCE 'fit_errorbars' is now 'fit_covariance'.")
        src = "fit_covariance"
    if src not in ("data_bootstrap", "fit_covariance", "residual_bootstrap"):
        raise ValueError(f"Unknown UNCERTAINTY_SOURCE '{UNCERTAINTY_SOURCE}'.")
    if src == "data_bootstrap" and data_mode not in ("ens_avg", "post_selected"):
        print(f"Warning: data bootstrap has no measurement statistics for "
              f"'{data_mode}' — falling back to residual_bootstrap.")
        src = "residual_bootstrap"
    return src


def _build_steps(data_mode: str, observables: list[str]):
    """Assemble MPM → (Refine) → Bootstrap steps from the global config."""
    src = _resolve_uncertainty_source(data_mode)
    conj = (CONJUGATE_PAIRS if CONJUGATE_PAIRS is not None
            else data_mode in ("post_selected", "ens_avg"))
    steps = [
        MPMStep(order=ORDER, observable_key=observables,
                plot_svd=PLOT_SVD, n_sv=10, save_dir=OUTPUT_DIR),
    ]
    if REFINE_POLES or src == "fit_covariance":
        # fit_covariance needs the refinement covariance, so RefineStep is
        # forced on in that mode.
        steps.append(RefineStep(observable_key=observables, conjugate_pairs=conj))
    if src == "data_bootstrap":
        steps.append(BootstrapStep(n_boot=N_BOOTSTRAP, seed=SEED,
                                   mode="data", n_jobs=N_JOBS))
    elif src == "fit_covariance":
        steps.append(BootstrapStep(n_boot=N_COV_SAMPLES, seed=SEED,
                                   mode="parametric"))
    else:
        steps.append(BootstrapStep(n_boot=N_BOOTSTRAP, seed=SEED,
                                   mode="residual"))
    return steps, src


def run_mpm(loader, folder, *, data_mode: str, observables: list[str]):
    """Unified MPM + bootstrap entry point for any loader/data mode."""
    folder = Path(folder)
    if not folder.exists():
        print(f"[{data_mode}] Folder not found: {folder}")
        return None, None

    print(f"\n=== MPM ({data_mode}, observables={observables}) ===")
    steps, src = _build_steps(data_mode, observables)
    pipeline = MPMPipeline.from_loader(
        loader, folder,
        data_mode=data_mode,
        observables=observables,
        steps=steps,
        lambda_range=LAMBDA_RANGE,
        max_files=N_LAMBDA_TEST,
        t_max=T_MAX,
        uncertainty_source=src,
    )
    result = pipeline.run()
    _plot_mpm_result(pipeline.records, result, data_mode)
    saved_path = None
    if not DEBUG:
        saved_path = pipeline.save(OUTPUT_DIR, full=SAVE_FULL_ANALYSIS)
        print(f"Saved: {saved_path.name}")
    return result, saved_path


def run_slider_plot(loader, folder, result=None):
    """Interactive per-λ trace browser for the configured observables.

    When an analysis ``result`` is given, the fit overlay uses its (refined,
    branch-tracked) median poles instead of re-running MPM per λ.
    """
    folder = Path(folder)
    if not folder.exists():
        print(f"[slider] Folder not found: {folder}")
        return
    records = loader.load(
        folder,
        observables=PLOT_OBSERVABLES,
        lambda_range=LAMBDA_RANGE,
        max_files=N_LAMBDA_TEST,
        t_max=T_MAX,
    )
    fixed_poles = None
    if result is not None:
        fixed_poles = {
            float(lam): -result.poles.decay_rates[i] + 1j * result.poles.frequencies[i]
            for i, lam in enumerate(result.lambda_values)
        }
    # Keep the widget alive: matplotlib only weak-references slider callbacks,
    # so a dropped reference leaves a slider that draws but does not respond.
    slider = plot_time_traces_slider(
        records,
        observable_key=PLOT_OBSERVABLES,
        scales=PLOT_SCALES,
        take_abs=PLOT_TAKE_ABS,
        mpm_order=ORDER,
        fixed_poles=fixed_poles,
    )
    _LIVE_WIDGETS.append(slider)
    return slider


def run_ksi_crossing(result, json_path=None):
    """Ksi (2·Re(e3) − Re(e2)) and discriminant zero crossings, per realization."""
    print(f"\n=== Ksi zero crossing ===")
    ksi = KsiCrossingPipeline(result, lambda_max=KSI_LAMBDA_MAX).run()
    if ksi.crossing_quantiles is not None:
        q16, q50, q84 = ksi.crossing_quantiles
        print(f"  lam_EP (ksi) = {q50:.4f} +{q84 - q50:.4f} / -{q50 - q16:.4f}")
    elif ksi.lambda_ep is not None:
        print(f"  lam_EP (ksi) = {ksi.lambda_ep:.4f} (no realization samples)")

    # Discriminant Delta = c1^2 - 4*c0 per realization: the preferred EP
    # diagnostic (Delta < 0 conjugate pair, = 0 EP, > 0 two real poles).
    transitions = {}
    raw = result.poles.raw_samples
    if raw is not None and raw.param_samples is not None:
        d_cross, _, _, _ = discriminant_crossing_samples(
            raw, lambda_max=KSI_LAMBDA_MAX
        )
        if np.any(np.isfinite(d_cross)):
            q16, q50, q84 = np.nanpercentile(d_cross, [16, 50, 84])
            print(f"  lam_EP (Delta=0) = {q50:.4f} +{q84 - q50:.4f} / -{q50 - q16:.4f}  "
                  f"({np.mean(np.isnan(d_cross)):.1%} of realizations had no crossing)")
            transitions["delta_crossing_samples"] = d_cross
            transitions["delta_crossing_quantiles"] = [q16, q50, q84]
        else:
            print("  Delta=0: no realization had a crossing.")

    ax = plot_ksi_curve(ksi)
    data_mode = str(result.metadata.get("data_mode",
                                        result.metadata.get("source_tag", "unknown")))
    scan_number = result.metadata.get("scan_number")
    if scan_number is not None:
        scan_number = int(scan_number)
    _show_and_save(
        ax.get_figure(), "ksi_crossing",
        data_mode=data_mode, scan_number=scan_number,
    )
    if json_path is not None and not DEBUG:
        augment_json_with_ksi(json_path, ksi)
        if transitions:
            augment_json_with_transitions(json_path, transitions)
    return ksi


def run_critical_point(result=None, guess=None, json_path=None):
    """Fit critical-point sqrt model."""
    if result is not None:
        cp = CriticalPointPipeline(result, lambda_max=CP_LAMBDA_MAX)
    elif CRITICAL_POINT_JSON is not None and CRITICAL_POINT_JSON.exists():
        cp = CriticalPointPipeline.from_json(CRITICAL_POINT_JSON, lambda_max=CP_LAMBDA_MAX)
        json_path = CRITICAL_POINT_JSON
    else:
        print(f"[critical_point] No result and no valid JSON — skipping.")
        return None

    print(f"\n=== Critical-point fit ===")
    if guess is None or CP_SOFT_GUESS:
        # Soft guess: only λ_c bounded (CP_LAMBDA_C_BOUNDS), a/b free.
        fit = cp.fit(
            lambda_c_guess=CP_LAMBDA_C_GUESS,
            soft_bounds=True, lambda_c_bounds=CP_LAMBDA_C_BOUNDS,
            window_left=50, window_right=15,
        )
    else:
        lambda_c_guess, a_guess, b_guess = guess["lambda_c"], guess["a"], guess["b"]
        fit = cp.fit(lambda_c_guess=lambda_c_guess, a_guess=a_guess, b_guess=b_guess, window_left=50, window_right=15)
    print(f"  lam_c (aggregate fit) = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}")
    print(f"  chi2_red = {fit.chi2_reduced:.3f}  R2 = {fit.r_squared:.4f}")

    # Per-realization sqrt fit: lambda_c^(b) fitted separately for every
    # bootstrap realization — uncertainty read off the distribution.
    if (CP_PER_REALIZATION and result is not None
            and result.poles.raw_samples is not None):
        raw = result.poles.raw_samples
        if CP_LAMBDA_MAX is not None:
            m = raw.lambda_values <= CP_LAMBDA_MAX
            raw = PoleSamples(
                samples=raw.samples[m],
                observable_key=raw.observable_key,
                lambda_values=raw.lambda_values[m],
                param_samples=(raw.param_samples[m]
                               if raw.param_samples is not None else None),
            )
        lam_c_samples = fit_sqrt_per_realization(
            raw,
            lambda_c_guess=CP_LAMBDA_C_GUESS,
            lambda_c_bounds=CP_LAMBDA_C_BOUNDS,
            window_left=50, window_right=15,
            omega_s=result.omega_s,
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

    # Plot: 3-panel appendix style
    cp.result.critical_point_sqrt_fit = fit
    fig = plot_appendix_bootstrap(
        cp.result,
        lambda_max=CP_LAMBDA_MAX,
        fit_validity_limit=CP_LAMBDA_MAX,
        real_axis_ticks=[0.0, 1, 2],
        real_split_axis_ticks=[0.0, 0.6, 1.2],
    )
    cp_data_mode = str(cp.result.metadata.get("data_mode",
                       cp.result.metadata.get("source_tag", "unknown")))
    cp_scan_number = cp.result.metadata.get("scan_number")
    if cp_scan_number is not None:
        cp_scan_number = int(cp_scan_number)
    _show_and_save(
        fig,
        "critical_point_appendix",
        data_mode=cp_data_mode,
        scan_number=cp_scan_number,
    )

    if not DEBUG:
        path = cp.save(json_path)
        print(f"Saved: {path.name}")
    return fit


def run_simulation_test():
    """Quick end-to-end test on synthetic data (no real data needed)."""
    print(f"\n=== Simulation test ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=30,
        simulation_duration=25.0,
        min_alpha=0.1,
        max_alpha=2.0,
    )
    records = sim.to_observable_records()
    print(f"  Simulated {len(records)} records")

    src = _resolve_uncertainty_source("simulation")   # → never data_bootstrap
    steps = [MPMStep(order=ORDER)]
    if REFINE_POLES or src == "fit_covariance":
        # conjugate_pairs forced on here for coverage of the constrained fit
        steps.append(RefineStep(conjugate_pairs=True))
    if src == "fit_covariance":
        steps.append(BootstrapStep(n_boot=N_COV_SAMPLES, seed=SEED, mode="parametric"))
    else:
        steps.append(BootstrapStep(n_boot=N_BOOTSTRAP, seed=SEED, mode="residual"))
    result = MPMPipeline.from_records(records, steps=steps).run()
    print(f"  Poles shape: {result.poles.decay_rates.shape}")
    _plot_mpm_result(records, result, "simulation")
    return result


def run_liouvillian_spectrum():
    """Plot eigenvalues of the 4x4 Liouvillian vs λ at k=π."""
    print("\n=== Liouvillian spectrum (k=π) ===")
    sim = LiouvillianSpectrum(
    )
    fig, _ = sim.plot()
    fig.suptitle("4×4 Liouvillian eigenvalues vs λ  (k=π)", fontsize=13)
    _show_and_save(fig, "liouvillian_spectrum", data_mode="theory")


def run_benchmark():
    """Test analysis robustness with injected Gaussian noise."""
    print(f"\n=== Benchmark ===")
    sim = PostSelectedDynamicsSimulator(
        num_lambdas=20, simulation_duration=40.0,
        min_alpha=0.1, max_alpha=2.0,
    )
    clean_records = sim.to_observable_records()

    bench = BenchmarkPipeline(
        clean_records=clean_records,
        noise_models=[ShotNoise(n_shots=6*3e4, seed=0)],
        analysis_steps=[MPMStep(order=ORDER), BootstrapStep(n_boot=N_BOOTSTRAP, seed=SEED)],
        n_trials=5 if DEBUG else 20,
    )
    result = bench.run(lambda_c_guess=1.25)
    print(f"  lam_c bias = {result.lambda_c_bias:.4f}  std = {result.lambda_c_std:.4f}")
    return result


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
  
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mpm_result = None
    mpm_json_path = None
    guess = None

    if RUN_MPM or RUN_SLIDER_PLOT:
        loader, folder, data_mode, guess_preset = _make_source(DATA_MODE)
        if RUN_MPM:
            mpm_result, mpm_json_path = run_mpm(
                loader, folder, data_mode=data_mode, observables=OBSERVABLES
            )
            if guess_preset is not None:
                guess = get_guess(guess_preset)
        if RUN_SLIDER_PLOT:
            run_slider_plot(loader, folder, result=mpm_result)

    if RUN_KSI_CROSSING and mpm_result is not None:
        run_ksi_crossing(mpm_result, json_path=mpm_json_path)

    if RUN_CRITICAL_POINT:
        run_critical_point(result=mpm_result, guess=guess, json_path=mpm_json_path)

    if RUN_SIMULATION_TEST:
        run_simulation_test()

    if RUN_BENCHMARK:
        run_benchmark()

    if RUN_LIOUVILLIAN_SPECTRUM:
        run_liouvillian_spectrum()

    print("\nDone.")
    if SHOW_PLOTS:
        plt.show()
