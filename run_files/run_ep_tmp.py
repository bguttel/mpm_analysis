from __future__ import annotations

from pathlib import Path
from mpm_analysis.utils.lab_paths import get_lab_path

# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================

DEBUG = False          # True → small n_boot, few files, no saving
ORDER = 3              # MPM model order
N_LAMBDA_TEST = None   # set to e.g. 10 to load only first 10 files (debug)
LAMBDA_RANGE = (0, 1.65)    # e.g. (0.1, 1.6) — restrict lambda window
SEED: int | None = None  # RNG seed (all uncertainty modes); None → truly random (OS entropy)
T_MAX = None            # max time to load from experimental records (µs), default is None
PROPAGATE_CORR = None   # (f_last, strict): None -> off.  See block below.
S1_FROM_SEG2 = None   # path to agent1 z6_seg2.json to replace S(1) by segment 2's first window
S1_FROM_SEG2_KEY = "S1_seg2"   # "S1_seg2" (unconditioned, like every other S(n)) or "S1_seg2_prevdark"
S0_ANCHOR = None    # set e.g. 0.995 to overwrite the t=0 sample of every
                   # survival trace before the pencil sees it (the by-construction
                   # S(0)=1 is not a measured point).  None -> leave as loaded.
DROP_FIRST_POINT = False  # True → drop the t=0 sample of every file (the
                          #        by-construction S=1 anchor) from the
                          #        observables AND the bootstrap statistics
                          #        before any analysis step.  For the
                          #        post-selected S the anchor is inconsistent
                          #        with the measured points (first-window
                          #        loss ~2-4 %, ~7 % G admixture at t=0,
                          #        ~1.5 %/window false clicks); through it
                          #        the order-3 pencil spends a pole on the
                          #        step and the EP reads 1.01 instead of the
                          #        generator's 1.15 (mpm_benchmarks/
                          #        noise_robustness/v3_transient.py).  Without
                          #        the anchor those artifacts are pure
                          #        multiplicative factors the pencil cannot
                          #        see; the price is that the EP is then
                          #        poorly resolved from S alone (no Delta
                          #        crossing in ~half the bootstrap draws).
SPLICE_FIRST_FROM_NOCLICK = False   # True → replace the FIRST MEASURED point of
                          #        the survival trace (index 1, t = 0.32 µs —
                          #        index 0 is the t=0 anchor) with the same
                          #        point of the prev-no-click conditional
                          #        curve, leaving every other point as it is.
                          #        The rest of the trace, the t=0 anchor and
                          #        every other observable are untouched.
                          #
                          #        Why: the first measured bin is the one the
                          #        Schmitt carry-over and the first-window
                          #        false-bright excess corrupt (the preceding
                          #        segment seeds the filter).  The prev-no-click
                          #        sub-ensemble is the one that entered that bin
                          #        with the filter in the dark state, so its
                          #        value is the uncontaminated estimate of the
                          #        same physical quantity.  Splicing only that
                          #        point tests whether the S data-model gap is
                          #        carried by it alone.
                          #
                          #        Needs data.p_0_t_array_prev_noclick in the
                          #        scan JSONs (written by
                          #        run_files/raw_data_tests/run_conditional_p0.py).
                          #        The bootstrap statistics are spliced at the
                          #        same index, on that point's own class size,
                          #        so the error bar there reflects the smaller
                          #        sub-ensemble instead of the full row count.
                          #        False → data loaded as it is.
SPLICE_INDEX = 1          # Which time index to splice.  1 = first measured
                          #        point (t = 0.32 µs on this scan's axis);
                          #        0 would be the by-construction t=0 anchor,
                          #        which carries no information.
SPLICE_SOURCE = "S_prev_noclick"   # Observable key to take the point FROM.
                          #        "S_prev_noclick" | "S_prev_click".
SPLICE_TARGETS = ("S",)   # Observable keys to splice INTO.  Only the plain
                          #        survival by default — the conditional
                          #        channels are the source and must stay intact.

SHOW_PLOTS  = False    # plt.show() after each task
SAVE_FIGURES = False # save figures as png into OUTPUT_DIR
SAVE_FULL_ANALYSIS = False  # write raw pole samples + refine params/cov + ksi into the output JSON

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

# ---- Post-selected scan ----
_PS = r"Common\QuantumZeno\Paper\zeno_paper_data\scan_experiment_results_10"

_PARITY = _ZT + r"\click_record_repeated\Omega_BG0_scan\scan_experiment_results_12"

# POST_SELECTED_FOLDER = get_lab_path(_PS)
POST_SELECTED_FOLDER = r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old data\scan_experiment_results_10"
ENS_AVG_FOLDER = r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\old_survival_data\scan_experiment_results_50"
PARITY_FOLDER = get_lab_path(_PARITY)
# ---- Lambda scale (both data modes) ----
# The loader finds the HMM table that provably belongs to the folder being
# loaded — matched on the full run id (drive frequency AND timestamp), so a
# table filed next to the wrong scan can never be applied.  When no table
# covers the scan, lambda comes from the JSON ``lambda_zeno_estimation``,
# which for the post-selected data already IS the HMM value
# (Gamma_BG_HMM / 2*Omega_DG).  Either way the run prints which it used and
# records it in the result metadata.  There is nothing to configure per scan.
#   "auto"    -> discover (default; correct for every scan)
#   False     -> force the stored JSON lambda
#   Path(...) -> force one table; raises if it does not belong to the folder
LAMBDA_SOURCE = "auto"

# ---- Output directory ----
OUTPUT_DIR = Path(__file__).parent.parent / "mpm_analysis_output"

# Paper figures get their own folder, and all three formats: PDF + SVG (the
# vector deliverables) and a high-res PNG preview, at PAPER_FIGURES_DPI.
PAPER_FIGURES_DIR = Path(__file__).parent.parent / "paper_figures"
PAPER_FIGURES_DPI = 600

# ===========================================================================
# SHARED ANALYSIS PARAMETERS  (read by more than one task — keep them here)
# ===========================================================================

# ---- Data source (MPM, slider, transitions, first-transition figure) ----
DATA_MODE   = "post_selected"   # "ens_avg" | "post_selected" | "parity"
OBSERVABLES = ["S"]  # keys to load from the records (must be present in the data)
                     # post_selected: "S" (stored p_0_t_array), "S_pooled"
                     # (segment-pooled rebuild from the raw NPZ, up to 59x the
                     # shots at early t — written by
                     # run_files/raw_data_tests/run_pooled_p0.py; its data
                     # bootstrap is a multinomial over the first-click
                     # histogram), "S_prev_click"/"S_prev_noclick", X/Y/Z…
# OBSERVABLES = ["D","B","G","error"]      # 1 key → classic MPM; >1 keys → joint MPM
                                # (common poles, per-observable amplitudes)

# ---- Pole refinement / uncertainty source (MPM, simulation test, benchmark,
#      dead-time sweep) ----
REFINE_POLES       = False    # weighted complex-exponential LSQ smoothing after MPM
CONJUGATE_PAIRS    = None    # None → auto: True for "post_selected"/"ens_avg", False otherwise
UNCERTAINTY_SOURCE = "data_bootstrap"  # how to get error bars on the poles and derived quantities:

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
N_JOBS        = 1      # parallel workers for the data bootstrap (-1 → all cores)
if DEBUG:
    N_BOOTSTRAP = 50

# ---- Transition extraction (transitions, first-transition figure, dead-time
#      sweep) ----
# Delta is linear in lambda through a square-root EP, so a straight line fitted
# over a window around the crossing uses every point in that window instead of
# the two that bracket the sign change ("first", which reads ~0.1 low).  The
# line is refitted inside every bootstrap realization; the error bar on the EP
# is the 16/84 spread of those per-realization fits.
DISCRIMINANT_HALFWIDTH = 0.15      # half-window in lambda for the line fit
TRANSITION_LAMBDA_MAX  = None      # e.g. 1.55 to match the critical-point window

# ---- agent1 batch overrides (environment variables; unset -> values above) ----
import os as _os
AGENT1_TAG = _os.environ.get("AGENT1_TAG", "run")
if _os.environ.get("AGENT1_OBS"):
    OBSERVABLES = _os.environ["AGENT1_OBS"].split(",")
if _os.environ.get("AGENT1_FOLDER"):
    POST_SELECTED_FOLDER = _os.environ["AGENT1_FOLDER"]
if _os.environ.get("AGENT1_ORDER"):
    ORDER = int(_os.environ["AGENT1_ORDER"])
if _os.environ.get("AGENT1_DROP_FIRST"):
    DROP_FIRST_POINT = _os.environ["AGENT1_DROP_FIRST"] == "1"
if _os.environ.get("AGENT1_LAMBDA_MAX"):
    LAMBDA_RANGE = (0, float(_os.environ["AGENT1_LAMBDA_MAX"]))
if _os.environ.get("AGENT1_LAMBDA_SOURCE") == "json":
    LAMBDA_SOURCE = False
print(f"[agent1] tag={AGENT1_TAG} obs={OBSERVABLES} order={ORDER} drop_first={DROP_FIRST_POINT} "
      f"lambda_range={LAMBDA_RANGE} folder={POST_SELECTED_FOLDER}")

# ===========================================================================
# TASKS — flip the walrus in the `if` line; each clause holds only the
# settings that task alone uses (the ones above are shared).
# ===========================================================================

# ---- MPM + bootstrap (unified for all data modes) ----
if (RUN_MPM := True):
    PLOT_SVD = False             # SVD-vs-λ model-order calibration figure

# ---- Interactive per-λ trace browser ----
if (RUN_SLIDER_PLOT := False):
    PLOT_OBSERVABLES = OBSERVABLES          # which keys the slider shows
    PLOT_SCALES      = {"survival": "log"}  # key → "log"/"linear"; unlisted → linear
    PLOT_TAKE_ABS    = False                # plot/fit |signal| (e.g. for parity)

# ---- Transitions read off the pole strands (needs RUN_MPM) ----
# TWO DIFFERENT transitions, do not conflate them:
#   state freezing : zero of ksi = (2·Re(e3) − Re(e2)) / (Re(e2) − Re(e3))
#   exceptional pt : zero of the pair discriminant Delta = c1^2 - 4*c0
# On the post-selected scan they sit ~0.14 apart in lambda.
if (RUN_TRANSITIONS := True):
    PLOT_KSI            = True      # ksi(lambda) with error bars + the ksi = 0 line
    PLOT_DISCRIMINANT   = True      # Delta(lambda) with error bars + the fitted line
    DISCRIMINANT_METHOD = "linfit"  # "linfit" (recommended) | "first"

# ---- Critical-point sqrt fit (fresh MPM result or an existing JSON) ----
if (RUN_CRITICAL_POINT := True):
    # Path to a previously saved analysis JSON, used when no fresh MPM result
    # exists in this run; None → require a fresh result.
    # CRITICAL_POINT_JSON = Path(
    #     r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code"
    #     r"\mpm_analysis_results_ens_avg_scan_experiment_results_50_20260129_225403.json"
    # )
    CRITICAL_POINT_JSON = None
    CP_LAMBDA_MAX  = 1.3    # exclude lambda above this for the fit

    # Soft guess: only λ_c is bounded (CP_LAMBDA_C_BOUNDS), a/b essentially free.
    # CP_SOFT_GUESS=False falls back to the per-mode preset (±10 % hard box).
    CP_SOFT_GUESS      = True
    CP_LAMBDA_C_BOUNDS = (0.95, 1.2)
    CP_LAMBDA_C_GUESS  = 1   # p0 for λ_c (always around 1–1.5)

    # Per-realization transition extraction: λ_c fitted separately for every
    # bootstrap realization (16/50/84 from the distribution), IN ADDITION to the
    # aggregate fit from the central values + error bars above.  Both reported.
    CP_PER_REALIZATION = False

# ---- First-transition paper figure (built from this run's MPM result) ----
if (RUN_FIRST_TRANSITION_FIGURE := False):
    FIRST_TRANSITION_LAYOUT = "discriminant"    # "sqrt"         → 3 panels: A spectrum,
                                        #                  B/C splittings + sqrt fit
                                        #                  (needs RUN_CRITICAL_POINT)
                                        # "discriminant" → 2 panels: A spectrum,
                                        #                  B Δ12(λ) line fit
                                        #                  (replaces panels B and C)
    FT_LAMBDA_MAX = 1.5                 # λ cutoff of every panel
    # Per-data-mode style, mirroring first_transition_plotting.py's parameter
    # dicts: post_selected = first transition, ens_avg = third transition
    # (linear panel A with ticks [0, 1, 2], split ticks [0, 0.6, 1.2],
    # fit validity 1.3).
    FT_STYLE = {
        "post_selected": dict(label="first_transition", fit_validity=1.2,
                              real_axis_scale="log", real_axis_ticks=None,
                              real_split_axis_ticks=[0.0, 1.0, 2, 3]),
        "ens_avg":       dict(label="third_transition", fit_validity=1.3,
                              real_axis_scale="linear", real_axis_ticks=[0.0, 1, 2],
                              real_split_axis_ticks=[0.0, 0.6, 1.2]),
    }

# ---- Detector dead-time compression sweep ----
# Uses p_click(t) — the fast-readout bright fraction, i.e. the ensemble
# probability of occupying |B> (data.click_probability of the ens_avg scans)
# — to remap ONLY the time axis,  t' = t·[1 − c·p_click(t)],  and refits the
# same modal form on the non-uniform axis: c = 0 is the raw data, c = 1 full
# compression.  The measured y-data are never modified.  The direct fit is
# the package's complex-exponential LSQ (RefineStep), seeded at c = 0 from
# the MPM poles and then from the previous c; the EP (discriminant line-fit
# zero) is read off exactly as in run_transitions.  Fallback: each remapped
# trace is interpolated back onto the raw 320 ns grid and the ordinary MPMf
# runs on it — reported alongside the direct route.
if (RUN_DEAD_TIME_SWEEP := False):
    DT_DATA_MODE         = "ens_avg"    # only the ens_avg JSONs carry click_probability
    DT_OBSERVABLES       = ["D","G"]        # fitted key(s): the D probability; >1 → joint fit
    DT_CLICK_KEY         = "click_probability"   # p_click source:
                                        # "click_probability" — fast-readout bright
                                        #   fraction per bin (the click detector's own
                                        #   |B> occupation; index 0 is a placeholder 0)
                                        # "B_plus_error"      — projective readout's
                                        #   |B> occupation (B + error probabilities,
                                        #   sampled at t; ~0.04 lower at late times)
    DT_REMAP             = "cumulative"  # "pointwise": t' = t·[1 − c·p_click(t)]  (as specified)
                                        # "cumulative": t' = t − c·∫p_click dt  (accumulated
                                        #   dead time; never folds).  The pointwise map folds
                                        #   where a noisy late bin of p_click jumps
                                        #   (c·[p + k·Δp] > 1); such traces are truncated at
                                        #   the first fold and the cut is printed.
    DT_COMPRESS_COEFFS   = tuple(i / 25 for i in range(11))   # c = 0.0 … 0.44
    DT_FALLBACK_REGRID   = True         # also run the regrid + MPM fallback
    DT_REGRID_DT         = None         # µs; None → the raw spacing (0.32)
    DT_N_BOOTSTRAP       = 0            # >0 → data bootstrap per c (refine-only
                                        #      draws: the pencil never sees t'),
                                        #      giving 16/84 error bars on lam_EP
    DT_REPRESENTATIVE_C  = (0.1, 0.2, 0.3)   # c values of the traces / per-c pole figures
    DT_REPRESENTATIVE_LAMBDAS = None    # e.g. (0.5, 1.0, 1.4); None → 3 evenly spaced
    DT_TRACES_SHOW_FALLBACK = False     # also draw the regrid + MPM fit in the traces figures
    DT_PANEL_GRIDS       = True         # static multi-panel figures = the slider frames side
                                        #   by side: traces (rows λ × cols c) and poles (rows c)
    DT_GRID_MAX_PANELS   = 6            # at most this many c columns/rows (evenly spaced)
    DT_SLIDER_HTML       = True         # also write the two sliders as standalone Plotly
                                        #   HTML files into OUTPUT_DIR (open in any browser)
    DT_SYNTHETIC         = True         # run the SAME sweep on the 4×4 generator's own
                                        #   P_D/P_G (fit) and P_B (p_click), integrated at
                                        #   the data's λ values and sampled on its 320 ns
                                        #   grid (needs DT_THEORY); prints experiment vs
                                        #   synthetic vs tau_B_eff theory per c and overlays
                                        #   the experiment on the synthetic EP-vs-c figure
    DT_SYNTHETIC_ORDER   = 4            # pole order for the synthetic sweep (the generator
                                        #   has 4 modes; order 3 biases its c=0 EP up by
                                        #   ~0.08).  None → the data's ORDER.
    # Theory overlay: 4×4 Liouvillian (simulation/liouvillian_spectrum.py) at
    # k = 0 — the ensemble-average generator — with the dead time mapped as
    # tau_B_eff = (1 − c)·tau_B, i.e. c = tau_B_comp / tau_B with
    # tau_B_comp = c·tau_B the compressed-away dwell time.  Omega is taken
    # from the data (omega_DG); T1/T2 as in LiouvillianSpectrum's defaults.
    # Draws the theory lambda_EP(c) line in the transition figure and the
    # theory eigenvalues + both transitions in the per-c pole figures.
    # None → no theory overlay.
    DT_THEORY = dict(tau_B=4.0, T1=93.0, T2=26.0)   # µs

# ---- Self-contained checks (no settings of their own beyond the shared ones) ----
if (RUN_SIMULATION_TEST := False):
    pass    # small synthetic simulation through MPM + bootstrap (no real data)

if (RUN_BENCHMARK := False):
    pass    # robustness benchmark with injected shot noise

if (RUN_LIOUVILLIAN_SPECTRUM := False):
    pass    # 4×4 Liouvillian eigenvalues vs λ at k=π


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
    KsiCrossingPipeline, discriminant_crossing_samples, discriminant_from_params,
    linfit_zero_crossing, find_zero_crossings,
)
from mpm_analysis.analysis.matrix_pencil import pack_pole_params
from mpm_analysis.analysis.critical_point import fit_sqrt_per_realization
from mpm_analysis.data_types.pole_samples import PoleSamples
from mpm_analysis.plotting.ksi_plot import plot_ksi_curve, plot_discriminant_curve
from mpm_analysis.io.analysis_json import (
    augment_json_with_ksi, augment_json_with_transitions,
)
from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.ensemble_average import EnsembleAverageDynamics
from mpm_analysis.simulation.liouvillian_spectrum import LiouvillianSpectrum
from mpm_analysis.io.experimental import EnsembleAverageLoader, PostSelectedLoader, ParityLoader
from mpm_analysis.simulation.noise import ShotNoise
from mpm_analysis.plotting.exploratory import (
    plot_spectrum_panels, plot_time_traces_slider
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
    """Standard 2-panel diagnostic: decay rates and frequencies of all poles."""
    fig, _ = plot_spectrum_panels(result, title=tag)
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

def _make_source(name: str):
    """Resolve a data mode to (loader, folder, data_mode, guess_preset)."""
    if name == "ens_avg":
        return (EnsembleAverageLoader(csv_path=LAMBDA_SOURCE), ENS_AVG_FOLDER,
                "ens_avg", "ens_avg_experimental")
    if name == "post_selected":
        return (PostSelectedLoader(csv_path=LAMBDA_SOURCE), POST_SELECTED_FOLDER,
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
        drop_first_point=DROP_FIRST_POINT,
        uncertainty_source=src,
    )
    if S0_ANCHOR is not None:
        n_patched = 0
        for rec in pipeline.records:
            if rec.observable_key not in observables:
                continue
            if len(rec.t) == 0 or abs(float(rec.t[0])) > 1e-12:
                continue              # this trace has no t=0 sample
            rec.signal[0] = S0_ANCHOR
            n_patched += 1
        print(f"[S0_ANCHOR] overwrote the t=0 sample with {S0_ANCHOR} "
              f"on {n_patched} traces")
    if PROPAGATE_CORR is not None:
        # propagated instrumental correction (agent1 z2/t3/x1):
        #   S(1)   += STRICT + F_LAST   (segment-1 bin: strict row-start classifier
        #                                + last-window signal loss, S(1) only)
        #   S(n>=2) /= (1 - F_LAST)     (every segment ends with a last window)
        f_last, strict = PROPAGATE_CORR
        n_patched = 0
        for rec in pipeline.records:
            if rec.observable_key not in observables:
                continue
            if len(rec.t) < 3 or abs(float(rec.t[0])) > 1e-12:
                continue
            rec.signal[1] = rec.signal[1] + strict + f_last
            rec.signal[2:] = rec.signal[2:] / (1.0 - f_last)
            n_patched += 1
        print(f"[PROPAGATE_CORR] S(1) += {strict + f_last:.4f}, S(n>=2) /= {1 - f_last:.4f} "
              f"on {n_patched} traces")
    if S1_FROM_SEG2 is not None:
        # replace the t = 320 ns sample (segment-1 single bin) by the survival of
        # the FIRST window of segment 2 (agent1 z6), everything else untouched
        import json as _json
        tab = {float(v["lam"]): v for v in _json.load(open(S1_FROM_SEG2)).values()}
        lams = sorted(tab)
        n_patched = 0
        for rec in pipeline.records:
            if rec.observable_key not in observables:
                continue
            if len(rec.t) < 2 or abs(float(rec.t[0])) > 1e-12:
                continue
            lam_near = min(lams, key=lambda x: abs(x - float(rec.lambda_val)))
            assert abs(lam_near - float(rec.lambda_val)) < 1e-6, (lam_near, rec.lambda_val)
            v = tab[lam_near]
            assert abs(float(rec.signal[1]) - v["S1_json"]) < 1e-9, (rec.signal[1], v["S1_json"])
            rec.signal[1] = v[S1_FROM_SEG2_KEY]
            n_patched += 1
        print(f"[S1_FROM_SEG2] replaced S(1) by {S1_FROM_SEG2_KEY} on {n_patched} traces")
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
        drop_first_point=DROP_FIRST_POINT,
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


def run_transitions(result, json_path=None):
    """The two transitions on the pole strands, each with its own name and error.

    state freezing : lam(ksi=0), ksi = (2·Re(e3) − Re(e2)) / (Re(e2) − Re(e3))
    exceptional pt : lam_EP, zero of the pair discriminant Delta = c1^2 - 4*c0

    Both errors are per-realization: the crossing (or the line fit through it)
    is redone inside every bootstrap draw and the 16/84 spread of those values
    is the error bar.  The central value comes from the unresampled poles.
    """
    print(f"\n=== Transitions ===")
    data_mode = str(result.metadata.get("data_mode",
                                        result.metadata.get("source_tag", "unknown")))
    scan_number = result.metadata.get("scan_number")
    if scan_number is not None:
        scan_number = int(scan_number)
    obs_tag = "+".join(OBSERVABLES)
    transitions = {}

    # --- state freezing: ksi = 0 --------------------------------------------
    ksi = KsiCrossingPipeline(result, lambda_max=TRANSITION_LAMBDA_MAX).run()
    ksi_central = ksi.lambda_ksi_zero
    if ksi.crossing_quantiles is not None:
        q16, q50, q84 = ksi.crossing_quantiles
        half = (q84 - q16) / 2
        centre = ksi_central if ksi_central is not None else q50
        print(f"  STATE FREEZING  lam(ksi=0) = {centre:.4f} +/- {half:.4f}   "
              f"[bootstrap 16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}]")
    elif ksi_central is not None:
        print(f"  STATE FREEZING  lam(ksi=0) = {ksi_central:.4f}  "
              f"(no bootstrap draws — no error bar)")
    else:
        print("  STATE FREEZING  ksi has no zero crossing.")
    if PLOT_KSI:
        ax = plot_ksi_curve(ksi, title=f"State freezing: ksi = 0  ({data_mode}, {obs_tag})")
        _show_and_save(ax.get_figure(), "state_freezing_ksi",
                       data_mode=data_mode, scan_number=scan_number)

    # --- exceptional point: Delta = 0 ---------------------------------------
    d_quantiles = d_med = d_lo = d_hi = None
    raw = result.poles.raw_samples
    if raw is not None and raw.param_samples is not None:
        d_cross, d_med, d_lo, d_hi = discriminant_crossing_samples(
            raw, lambda_max=TRANSITION_LAMBDA_MAX,
            method=DISCRIMINANT_METHOD, half_width=DISCRIMINANT_HALFWIDTH,
        )
        if np.any(np.isfinite(d_cross)):
            q16, q50, q84 = np.nanpercentile(d_cross, [16, 50, 84])
            d_quantiles = (q16, q50, q84)
            transitions["delta_crossing_samples"] = d_cross
            transitions["delta_crossing_quantiles"] = [q16, q50, q84]
            transitions["delta_crossing_method"] = DISCRIMINANT_METHOD
        else:
            print("  EXCEPTIONAL PT  no bootstrap realization had a Delta crossing.")
    else:
        print("  EXCEPTIONAL PT  no quadratic parameters stored — the discriminant "
              "needs a bootstrap/covariance run (UNCERTAINTY_SOURCE + N_BOOTSTRAP >= 1).")

    lam_d = np.asarray(result.lambda_values, dtype=float)
    central = -result.poles.decay_rates + 1j * result.poles.frequencies
    order = central.shape[1]
    d_central = np.array([
        discriminant_from_params(pack_pole_params(p), order) for p in central
    ], dtype=float)
    if TRANSITION_LAMBDA_MAX is not None:
        m = lam_d <= TRANSITION_LAMBDA_MAX
        lam_d, d_central = lam_d[m], d_central[m]
    lam_ep, fit = linfit_zero_crossing(
        lam_d, d_central, half_width=DISCRIMINANT_HALFWIDTH, return_fit=True
    )
    first = find_zero_crossings(lam_d, d_central)
    # agent1: dump the CENTRAL (unresampled) poles and discriminant for comparison
    import json as _json
    _json.dump(dict(lambda_values=lam_d.tolist(), delta_central=d_central.tolist(),
                    decay_rates=result.poles.decay_rates.tolist(),
                    frequencies=result.poles.frequencies.tolist()),
               open(Path(__file__).parent / f"ep_central_{AGENT1_TAG}.json", "w"), indent=1)
    if d_quantiles is not None:
        q16, q50, q84 = d_quantiles
        print(f"  EXCEPTIONAL PT  lam_EP     = {lam_ep:.4f} +/- {(q84 - q16) / 2:.4f}   "
              f"[bootstrap 16/50/84 = {q16:.4f}/{q50:.4f}/{q84:.4f}, "
              f"{np.mean(np.isnan(transitions['delta_crossing_samples'])):.1%} without a crossing]")
    elif np.isfinite(lam_ep):
        print(f"  EXCEPTIONAL PT  lam_EP     = {lam_ep:.4f}  (no bootstrap draws — no error bar)")
    print(f"                  ({DISCRIMINANT_METHOD} on Delta; raw first sign change at "
          + (f"{first[0]:.4f})" if first else "none)"))
    if np.isfinite(lam_ep):
        transitions["delta_linfit_central"] = lam_ep

    if PLOT_DISCRIMINANT:
        band = (d_lo, d_hi) if (d_med is not None and len(d_med) == len(lam_d)) else (None, None)
        ax_d = plot_discriminant_curve(
            lam_d, d_central, fit=fit, lambda_ep=lam_ep,
            first_crossing=first[0] if first else None,
            crossing_quantiles=d_quantiles, delta_lower=band[0], delta_upper=band[1],
            title=f"Exceptional point: Delta = 0  ({data_mode}, {obs_tag})",
        )
        _show_and_save(
            ax_d.get_figure(), "exceptional_point_discriminant",
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


def run_first_transition_figure(result):
    """Paper-style first-transition figure from this run's result.

    ``FIRST_TRANSITION_LAYOUT`` picks the 3-panel sqrt layout (A full
    spectrum, B/C splittings with the sqrt fit — needs the critical-point
    fit, so run with RUN_CRITICAL_POINT on) or the 2-panel discriminant
    layout, where the Δ12(λ) straight-line fit through its zero crossing
    replaces panels B and C in the same style.
    """
    from mpm_analysis.plotting.paper_figures.first_transition_main import (
        plot_first_transition_figure,
    )

    print(f"\n=== First-transition figure ({FIRST_TRANSITION_LAYOUT}) ===")
    if (FIRST_TRANSITION_LAYOUT == "sqrt"
            and getattr(result, "critical_point_sqrt_fit", None) is None):
        print("[first_transition] result carries no sqrt fit — enable "
              "RUN_CRITICAL_POINT (or switch to the 'discriminant' layout).")
        return None
    data_mode = str(result.metadata.get("data_mode",
                                        result.metadata.get("source_tag", "unknown")))
    style = FT_STYLE.get(data_mode)
    if style is None:
        print(f"[first_transition] no FT_STYLE preset for data mode "
              f"'{data_mode}' — using the post_selected style.")
        style = FT_STYLE["post_selected"]
    fig = plot_first_transition_figure(
        result,
        layout=FIRST_TRANSITION_LAYOUT,
        lambda_max=FT_LAMBDA_MAX,
        fit_validity_limit=style["fit_validity"],
        real_axis_scale=style["real_axis_scale"],
        real_axis_ticks=style["real_axis_ticks"],
        real_split_axis_ticks=style["real_split_axis_ticks"],
        discriminant_half_width=DISCRIMINANT_HALFWIDTH,
    )
    scan_number = result.metadata.get("scan_number")
    if scan_number is not None:
        scan_number = int(scan_number)

    # Paper figures: their own folder, PDF + SVG + high-res PNG, with a stem
    # naming what the figure is — transition, layout, data mode, scan,
    # observables, MPM order — plus a UTC stamp so nothing is overwritten.
    if SAVE_FIGURES and not DEBUG:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        stem = build_figure_filename(
            figure_tag=(f"{style['label']}_{FIRST_TRANSITION_LAYOUT}"
                        f"_{'+'.join(OBSERVABLES)}_order{ORDER}"),
            data_mode=data_mode,
            scan_number=scan_number,
            extension="",
        ).rstrip(".")
        save_figure_pdf_svg(
            fig, stem, PAPER_FIGURES_DIR,
            dpi=PAPER_FIGURES_DPI, png=True,
        )
    if SHOW_PLOTS:
        plt.show(block=False)
    return fig


def run_dead_time_sweep(mpm_result=None):
    """Detector dead-time compression sweep: c = 0 (raw) → c = 1 (fully compressed).

    Loads the DT_OBSERVABLES together with the click probability, sweeps the
    compression coefficient with ``DeadTimeSweepPipeline`` (direct
    non-uniform fit + regrid/MPM fallback), prints and saves the transition
    table (JSON + CSV) and the three figures: traces with fits, poles vs c,
    discriminant/transitions vs c.  When this run's MPM result describes the
    same data, its poles seed the c = 0 fit.
    """
    from mpm_analysis.pipeline.dead_time_pipeline import DeadTimeSweepPipeline
    from mpm_analysis.plotting.dead_time_plots import (
        export_poles_slider_html, export_traces_slider_html,
        plot_compression_traces, plot_poles_at_compression,
        plot_poles_compression_slider, plot_poles_panel_grid,
        plot_traces_compression_slider, plot_traces_panel_grid,
        plot_transition_vs_compression,
    )

    loader, folder, data_mode, _ = _make_source(DT_DATA_MODE)
    folder = Path(folder)
    if not folder.exists():
        print(f"[dead_time] Folder not found: {folder}")
        return None
    print(f"\n=== Dead-time compression sweep ({data_mode}, {DT_REMAP}, "
          f"observables={DT_OBSERVABLES}, c={DT_COMPRESS_COEFFS[0]:g}..{DT_COMPRESS_COEFFS[-1]:g}) ===")
    records = loader.load(
        folder,
        observables=[*DT_OBSERVABLES, DT_CLICK_KEY],
        lambda_range=LAMBDA_RANGE,
        max_files=N_LAMBDA_TEST,
        t_max=T_MAX,
        drop_first_point=DROP_FIRST_POINT,
    )

    # Seed c = 0 from this run's MPM poles when they describe the same data.
    initial = None
    if (mpm_result is not None
            and mpm_result.metadata.get("data_mode") == data_mode
            and list(mpm_result.metadata.get("observables", [])) == list(DT_OBSERVABLES)):
        initial = {
            float(lam): -mpm_result.poles.decay_rates[i] + 1j * mpm_result.poles.frequencies[i]
            for i, lam in enumerate(mpm_result.lambda_values)
        }

    conj = CONJUGATE_PAIRS if CONJUGATE_PAIRS is not None else True
    pipe = DeadTimeSweepPipeline(
        records, DT_COMPRESS_COEFFS,
        order=ORDER,
        observable_keys=DT_OBSERVABLES,
        click_key=DT_CLICK_KEY,
        remap=DT_REMAP,
        discriminant_half_width=DISCRIMINANT_HALFWIDTH,
        lambda_max=TRANSITION_LAMBDA_MAX,
        conjugate_pairs=conj,
        fallback=DT_FALLBACK_REGRID,
        regrid_dt=DT_REGRID_DT,
        n_boot=DT_N_BOOTSTRAP,
        seed=SEED,
        n_jobs=N_JOBS,
        initial_poles=initial,
        metadata=dict(data_mode=data_mode, source_folder=str(folder),
                      observables=list(DT_OBSERVABLES),
                      lambda_source=loader.lambda_source),
    )
    sweep = pipe.run()

    rows = sweep.summary_rows()
    cols = [k for k in rows[0] if k.startswith(("compress", "lambda_ep_direct",
                                                "lambda_ep_q", "lambda_ep_regrid"))]
    print("\n  " + "  ".join(f"{c:>24s}" for c in cols))
    for row in rows:
        print("  " + "  ".join(f"{row[c]:>24.4f}" for c in cols))
    if not DEBUG:
        path = pipe.save(OUTPUT_DIR)
        print(f"Saved: {path.name} (+ .csv)")

    scan = sweep.metadata.get("scan_number")
    scan = int(scan) if scan is not None else None
    figs = [
        (plot_compression_traces(records, sweep, c_values=DT_REPRESENTATIVE_C,
                                 target_lambdas=DT_REPRESENTATIVE_LAMBDAS,
                                 click_key=DT_CLICK_KEY,
                                 show_fallback=DT_TRACES_SHOW_FALLBACK), "dead_time_traces"),
        (plot_transition_vs_compression(sweep, c_values=DT_REPRESENTATIVE_C,
                                        lambda_max=TRANSITION_LAMBDA_MAX,
                                        theory_params=DT_THEORY),
         "dead_time_transition"),
    ]
    # One standard pole-spectrum figure (all poles per panel) per representative c
    for c in DT_REPRESENTATIVE_C:
        fig, _ = plot_poles_at_compression(sweep, c, lambda_max=TRANSITION_LAMBDA_MAX,
                                           theory_params=DT_THEORY)
        fig.tight_layout()
        c_used = sweep.compress_coeffs[sweep.index_of(c)]
        figs.append((fig, f"dead_time_poles_c{c_used:.2f}"))
    if DT_PANEL_GRIDS:
        # The slider frames as static multi-panel figures
        figs.append((plot_traces_panel_grid(records, sweep, target_lambdas=DT_REPRESENTATIVE_LAMBDAS,
                                            click_key=DT_CLICK_KEY, max_panels=DT_GRID_MAX_PANELS,
                                            show_fallback=DT_TRACES_SHOW_FALLBACK),
                     "dead_time_traces_grid"))
        figs.append((plot_poles_panel_grid(sweep, lambda_max=TRANSITION_LAMBDA_MAX,
                                           theory_params=DT_THEORY, max_panels=DT_GRID_MAX_PANELS),
                     "dead_time_poles_grid"))
    for fig, tag in figs:
        _show_and_save(fig, tag, data_mode=data_mode, scan_number=scan)
    if DT_SLIDER_HTML and not DEBUG:
        # Standalone HTML sliders (Plotly): same content as the interactive ones
        for tag, export in (("dead_time_traces_slider", lambda p: export_traces_slider_html(
                                 records, sweep, p, target_lambdas=DT_REPRESENTATIVE_LAMBDAS,
                                 click_key=DT_CLICK_KEY, show_fallback=DT_TRACES_SHOW_FALLBACK)),
                            ("dead_time_poles_slider", lambda p: export_poles_slider_html(
                                 sweep, p, lambda_max=TRANSITION_LAMBDA_MAX,
                                 theory_params=DT_THEORY))):
            path = OUTPUT_DIR / build_figure_filename(figure_tag=tag, data_mode=data_mode,
                                                      scan_number=scan, extension="html")
            print(f"  HTML saved: {Path(export(path)).name}")
    if DT_SYNTHETIC and DT_THEORY is not None:
        run_dead_time_synthetic(sweep, records, conj=conj, scan=scan)
    if SHOW_PLOTS:
        # Sliders over c — the traces and the two pole panels (interactive
        # only; keep the widgets referenced or they stop responding).
        _LIVE_WIDGETS.append(
            plot_traces_compression_slider(records, sweep,
                                           target_lambdas=DT_REPRESENTATIVE_LAMBDAS,
                                           click_key=DT_CLICK_KEY,
                                           show_fallback=DT_TRACES_SHOW_FALLBACK)
        )
        _LIVE_WIDGETS.append(
            plot_poles_compression_slider(sweep, lambda_max=TRANSITION_LAMBDA_MAX, theory_params=DT_THEORY, method="regrid_mpm")
        )
        plt.show(block=False)
    return sweep


def run_dead_time_synthetic(sweep, records, *, conj: bool, scan):
    """The same compression sweep on the 4×4 generator's own populations.

    The generator of ``DT_THEORY`` (Omega from the data) is integrated at the
    experiment's λ values and sampled on the experiment's time grid; its
    P_D / P_G are the fitted observables and its P_B the p_click, so the
    identical remap + fits + EP extraction run on exact theory data.  Prints
    a c-by-c comparison (experiment | synthetic | tau_B_eff theory), saves
    the synthetic sweep and two figures: the EP-vs-c figure with the
    experiment overlaid, and the synthetic poles grid.
    """
    from mpm_analysis.pipeline.dead_time_pipeline import DeadTimeSweepPipeline
    from mpm_analysis.plotting.dead_time_plots import (
        plot_poles_panel_grid, plot_transition_vs_compression,
    )
    from mpm_analysis.simulation.liouvillian_spectrum import (
        synthetic_records, theory_transition_vs_compression,
    )

    keys = list(DT_OBSERVABLES)
    if not set(keys) <= {"D", "G", "B"}:
        print(f"[dead_time synthetic] generator has only D/G/B populations — "
              f"cannot fit {keys}; skipped.")
        return None
    th = {"Omega": sweep.omega_s, "k": 0.0, **DT_THEORY}
    t_grid = next(r.t for r in records if r.observable_key == keys[0])
    order = ORDER if DT_SYNTHETIC_ORDER is None else DT_SYNTHETIC_ORDER
    print(f"\n=== Same sweep on the synthetic 4x4 generator "
          f"(tau_B={th['tau_B']:g}, T1={th['T1']:g}, T2={th['T2']:g} us; order {order} "
          f"vs {ORDER} for the data; {len(sweep.lambda_values)} lambdas on the data's "
          f"{t_grid[1] - t_grid[0]:.3f} us grid) ===")
    syn_records = synthetic_records(
        sweep.lambda_values, t_grid, tau_B=th["tau_B"], T1=th["T1"], T2=th["T2"],
        Omega=th["Omega"], k=th["k"], click_key=DT_CLICK_KEY,
    )
    pipe = DeadTimeSweepPipeline(
        syn_records, DT_COMPRESS_COEFFS,
        order=order, observable_keys=keys, click_key=DT_CLICK_KEY, remap=DT_REMAP,
        discriminant_half_width=DISCRIMINANT_HALFWIDTH, lambda_max=TRANSITION_LAMBDA_MAX,
        conjugate_pairs=conj, fallback=DT_FALLBACK_REGRID, regrid_dt=DT_REGRID_DT,
        n_boot=0,          # exact data: nothing to resample
        metadata=dict(data_mode="synthetic_4x4", observables=keys, theory=dict(th),
                      source_folder=str(sweep.metadata.get("source_folder", ""))),
    )
    syn = pipe.run()

    ep_th = theory_transition_vs_compression(
        syn.compress_coeffs, tau_B=th["tau_B"], T1=th["T1"], T2=th["T2"],
        Omega=th["Omega"], k=th["k"])
    print("\n  c        experiment   synthetic 4x4   theory tau_B_eff=(1-c)tau_B")
    for c, e, s_, t_ in zip(syn.compress_coeffs, sweep.direct.lambda_ep,
                            syn.direct.lambda_ep, ep_th):
        print(f"  {c:5.2f}   {e:10.4f}   {s_:13.4f}   {t_:16.4f}")
    if not DEBUG:
        path = pipe.save(OUTPUT_DIR)
        print(f"Saved: {path.name} (+ .csv)")
    figs = [
        (plot_transition_vs_compression(
            syn, c_values=DT_REPRESENTATIVE_C, lambda_max=TRANSITION_LAMBDA_MAX,
            theory_params=DT_THEORY, references={"experiment": sweep},
            title="Exceptional point vs c: synthetic 4x4 generator (circles) vs experiment"),
         "dead_time_transition_synthetic"),
        (plot_poles_panel_grid(syn, lambda_max=TRANSITION_LAMBDA_MAX, theory_params=DT_THEORY,
                               max_panels=DT_GRID_MAX_PANELS,
                               title="Synthetic 4x4 generator: poles vs dead-time compression"),
         "dead_time_poles_grid_synthetic"),
    ]
    for fig, tag in figs:
        _show_and_save(fig, tag, data_mode="synthetic_4x4", scan_number=scan)
    return syn


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

    if RUN_TRANSITIONS and mpm_result is not None:
        run_transitions(mpm_result, json_path=mpm_json_path)

    if RUN_CRITICAL_POINT:
        run_critical_point(result=mpm_result, guess=guess, json_path=mpm_json_path)

    if RUN_FIRST_TRANSITION_FIGURE and mpm_result is not None:
        # After the critical point: its sqrt fit lands on mpm_result, which
        # the "sqrt" layout reuses.
        run_first_transition_figure(mpm_result)

    if RUN_DEAD_TIME_SWEEP:
        run_dead_time_sweep(mpm_result)

    if RUN_SIMULATION_TEST:
        run_simulation_test()

    if RUN_BENCHMARK:
        run_benchmark()

    if RUN_LIOUVILLIAN_SPECTRUM:
        run_liouvillian_spectrum()

    print("\nDone.")
    if SHOW_PLOTS:
        plt.show()
