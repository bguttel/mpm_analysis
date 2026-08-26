"""
run_kspace_analysis.py
----------------------
Single entry point for the k-space pole analysis ("FDG transition"), for all
data sources behind one selector:

    SOURCE = "json"        -> experiment scan JSONs   (KSpaceJSONLoader)
    SOURCE = "compressed"  -> compressed click records (CompressedClickKSpaceLoader)
    SOURCE = "simulated"   -> simulation NPZ from generate_click_record_raw.py
                              (SimClickKSpaceLoader)

It drives the unified ``KSpacePipeline``: load -> KSpaceAnalysis -> save
``KSpacePolesResult`` -> save the full figure set (eigenvalues-vs-λ at k≈π,
poles-vs-k decay/freq, parametric, SVD spectrum, raw k-space traces + HTML
slider, 3-D grids + HTML, braiding-ν).  Options: run-level bootstrap error bars
(compressed only), theory overlay + braiding comparison.

Figures use the ``Agg`` backend and are always saved; ``OPEN_FIGURES`` opens
them (PNG via the OS viewer, HTML in the browser) after the run so they are easy
to see — this also sidesteps the Windows long-path viewing limit.

The legacy 1-D eigenvalues-vs-λ-at-fixed-k task (json source) is preserved as an
optional toggle.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")               # keep Agg; we open saved files at the end

from pathlib import Path

from mpm_analysis.pipeline.kspace_pipeline import KSpacePipeline, _open_paths
from mpm_analysis.pipeline.click_correlation_pipeline import ClickCorrelationPipeline
from mpm_analysis.utils.lab_paths import get_lab_path

# ===========================================================================
# SETTINGS
# ===========================================================================

# ---- Source + data location ----
SOURCE = "json"          # "json" | "compressed" | "simulated"
SCAN_NUMBER = 22

# "simulated" points directly at a click_record_raw_*.npz file produced by
# misc/generate_click_record_raw.py — set the path and skip the lab-path
# lookup used by the other two sources.
SIMULATED_NPZ = Path(r"click_record_raw_20260730_131459.npz")

if SOURCE == "simulated":
    FOLDER = SIMULATED_NPZ
else:
    _SCAN = get_lab_path(
        "Experiments", "Zeno", "CavityRecAl6", "Chip_TransSap46_5", "OPX",
        "two_transmons", "click_record_repeated", "Omega_BG0_scan",
        f"scan_experiment_results_{SCAN_NUMBER}",
    )
    # json source reads the scan folder directly; compressed source reads the
    # compressed_click_analysis subfolder produced by run_kspace_preprocessing.py.
    FOLDER = _SCAN if SOURCE == "json" else _SCAN / "compressed_click_analysis"

# ---- Tasks ----
RUN_KSPACE_FULL = False                 # the unified k-space pole analysis
RUN_CORRELATIONS = True                 # fixed-anchor click correlations -> MPM
RUN_EIGENVALUES_VS_LAMBDA_1D = False    # legacy fixed-k MPM (json source only)
K_IDX_FOR_EIGENVALUES = 50              # k index (≈ π) for the legacy 1-D task

# ---- Analysis parameters ----
MAX_ORDER = 2
MIN_ORDER = 2
SVD_THRESHOLD = 0.2
N_BOOTSTRAP = 1                # residual resampling inside KSpaceAnalysis
BOOTSTRAP_OVER_RUNS = False    # compressed only: per-run analysis -> pole error bars
SIGNAL_COMPONENT = "complex"   # "complex" | "real" | "imag"
T_MAX_US = None
LAMBDA_RANGE = None
MAX_FILES = None               # json source only
K_INDEX_RANGE = (1,-1)# None keeps the full selected/recomputed k grid
# Compressed source only: recompute P(k,t) from stored P(n,t) on a denser open
# grid.  The endpoints are shifted inward to avoid the exact k=0 and k=2pi
# numerical issue while still getting very close to both ends.
K_GRID_N = 401                 # None = stored 101-point grid; 401 ≈ 4x denser
K_ENDPOINT_EPS = 1e-3          # first/last k are eps and 2pi-eps
# Compressed source only. A shot contributes at compressed time t iff it still
# has data there (t < compressed_length). Each retained time bin is normalized
# by the number of surviving shots. No padding/freezing of ended shots.
# Keep bins with alive shots >= frac*initial. 0.5 means "only times where at
# least half the compressed trajectories still exist"; tune this as needed.
MIN_ALIVE_FRAC = 0.98

# ---- Correlation parameters (RUN_CORRELATIONS task) ----
# Click statistics (analysis/click_correlations.py).  Both modes measure time
# from each run's own t = 0 - no anchoring, no lag axis - and differ only in
# how many clicks are tolerated between 0 and t:
#   "conditional" (default) - post-selected / no-jump dynamics: the run must
#     be silent in EVERY bin up to t.  no_click[t] = P(n=0, t) exactly, with
#     all runs that have not terminated in the denominator; click[t] is the
#     first-click probability.  Fitting it gives the post-selected
#     (non-Hermitian) eigenvalues.  Keys: cond_click / cond_no_click.
#   "unconditional" - full dynamics: only the bin at t matters, whatever
#     happened in between.  no_click[t] = P(no click at t), click[t] its
#     complement.  Fitting it gives the full Liouvillian eigenvalues.
#     Keys: uncond_click / uncond_no_click.
#   "partial" - the interpolation: at most CORR_MAX_CLICKS clicks strictly
#     before t (0 reproduces conditional, unlimited reproduces
#     unconditional).  Keys: partial_click / partial_no_click.
CORR_MODE = "unconditional"
CORR_MAX_CLICKS = None         # only for mode="partial": int >= 0
CORR_OBSERVABLES = None        # None = both channels of the chosen mode
CORR_MAX_TIME_US = None        # largest time (us); None = full record
# Truncate once too few runs are still recorded.  Only ever active for the
# compressed source, whose per-shot compressed lengths differ.
CORR_MIN_ALIVE_FRAC = 0.98
CORR_N_BOOTSTRAP = 1         # run/group bootstrap draws for error bars
CORR_N_GROUPS = 25             # dense sources without runs: bootstrap groups
CORR_ORDER = 3                 # MPM model order on the time axis
CORR_N_BOOT = 1                # MPM residual-bootstrap draws
CORR_SKIP_FIRST = 0            # drop this many leading time points from the fit
CORR_PLOT_SVD = True           # save the Hankel-SVD order-calibration figure

# ---- Theory + braiding ----
COMPUTE_BRAIDING = False        # braiding characterization of the data (ν vs λ)
# Theory overlay / comparison with experimental-imperfection parameters.
# Set THEORY = None to disable.  Knobs:
#   model    : "3x3" or "4x4"
#   T1, T2   : relaxation / dephasing times (µs)
#   tau_B    : bright-state lifetime (µs; 4x4 only; None -> 1/(λΩ))
#   k_fp     : false-positive count rate (rad/µs)  [you can set from data's alpha]
#   fn       : readout false-negative probability (4x4 only; e^{ik} -> fn+(1-fn)e^{ik})
#   t_bin    : detection time-bin length (µs; 4x4 only; 0 -> continuous limit)
#   Omega    : Rabi (rad/µs); omit -> use data's Omega_DG
#   lambda_scale : multiplier on λ before the model.  USE THIS to calibrate the
#     click-rate: the model's rate is α=2λΩ, but the data clicks ~8x slower for
#     this scan, so the theory eigenvalues are ~8x too fast.  Setting
#     lambda_scale≈0.13 (≈ measured_rate / 2λΩ) aligns the scales.
THEORY = dict(
    model="3x3",
    Omega=0.628 * 1.5,
    T1=120.0,
    T2=120.0,
    tau_B=None,
    k_fp=1e-3,
    lambda_scale=1.0,
)

# ---- Output / figures ----
SAVE_RESULT = True
SAVE_FIGURES = True
OPEN_FIGURES = True            # pop the saved figures/HTML open after the run
INCLUDE_3D = True
INCLUDE_HTML = True
GRID_N_PANELS = 25
GRID_N_COLS = 5
RAW_COMPONENT = "real"


def _run_legacy_1d() -> None:
    """Legacy: eigenvalues vs λ at a single k via the 1-D ParityLoader + MPM."""
    import matplotlib.pyplot as plt
    from mpm_analysis.io.experimental import ParityLoader
    from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
    from mpm_analysis.analysis.steps import MPMStep, BootstrapStep
    from mpm_analysis.plotting.pole_spectrum import plot_decay_and_freq_panel
    from mpm_analysis.io.experimental.compressed_click_record import _unique_path, _timestamp
    from mpm_analysis.utils.windows_paths import win_long_path

    print(f"\n=== Legacy eigenvalues vs λ (k_idx={K_IDX_FOR_EIGENVALUES}) ===")
    records = ParityLoader(k_idx=K_IDX_FOR_EIGENVALUES).load(
        FOLDER, observables=["parity_real", "parity_imag"],
        lambda_range=LAMBDA_RANGE, max_files=MAX_FILES, t_max=T_MAX_US,
    )
    records_real = [r for r in records if r.observable_key == "parity_real"]
    pipeline = MPMPipeline.from_records(
        records_real, steps=[MPMStep(order=MAX_ORDER), BootstrapStep(n_boot=N_BOOTSTRAP)],
    )
    result_1d = pipeline.run()
    ax_decay, _ = plot_decay_and_freq_panel(result_1d)
    fig = ax_decay.get_figure()
    fig.suptitle(f"k-space MPM (1-D)  |  k_idx = {K_IDX_FOR_EIGENVALUES}", fontsize=11)
    
    out_dir = FOLDER / "pole_analysis"
    import os
    os.makedirs(win_long_path(out_dir), exist_ok=True)
    path = _unique_path(out_dir / f"eigenvalues_vs_lambda_kidx{K_IDX_FOR_EIGENVALUES}_{_timestamp()}.png")
    fig.savefig(win_long_path(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")
    if OPEN_FIGURES:
        _open_paths([path])


def main() -> None:
    if RUN_EIGENVALUES_VS_LAMBDA_1D:
        if SOURCE != "json":
            print("[skip] legacy 1-D task needs SOURCE='json'.")
        else:
            _run_legacy_1d()

    if RUN_KSPACE_FULL:
        print(f"\n=== Full k-space MPM (source={SOURCE}) ===")
        out = KSpacePipeline().run(
            FOLDER,
            source=SOURCE,
            lambda_range=LAMBDA_RANGE, t_max=T_MAX_US,
            k_index_range=K_INDEX_RANGE, min_alive_frac=MIN_ALIVE_FRAC,
            k_grid_n=K_GRID_N, k_endpoint_eps=K_ENDPOINT_EPS,
            max_files=MAX_FILES,
            max_order=MAX_ORDER, min_order=MIN_ORDER, threshold=SVD_THRESHOLD,
            n_bootstrap=N_BOOTSTRAP, signal_component=SIGNAL_COMPONENT,
            bootstrap_over_runs=BOOTSTRAP_OVER_RUNS,
            braiding=COMPUTE_BRAIDING, theory=None,
            save_result=SAVE_RESULT, save_figures=SAVE_FIGURES,
            open_figures=OPEN_FIGURES,
            grid_n_panels=GRID_N_PANELS, grid_n_cols=GRID_N_COLS,
            raw_component=RAW_COMPONENT, include_3d=INCLUDE_3D,
            include_html=INCLUDE_HTML, scan_number=SCAN_NUMBER,
        )
        print(f"\nResult: {out['result_path'].name if out['result_path'] else '(not saved)'}")
        print(f"Figures ({len(out['figures'])}) in: {out['out_dir']}")
    
    if RUN_CORRELATIONS:
        print(f"\n=== Click correlations [{CORR_MODE}] (source={SOURCE}) ===")
        out = ClickCorrelationPipeline().run(
            FOLDER,
            source=SOURCE,
            lambda_range=LAMBDA_RANGE, t_max_us=T_MAX_US, max_files=MAX_FILES,
            mode=CORR_MODE, max_clicks=CORR_MAX_CLICKS,
            max_time_us=CORR_MAX_TIME_US, min_alive_frac=CORR_MIN_ALIVE_FRAC,
            n_bootstrap=CORR_N_BOOTSTRAP, n_groups=CORR_N_GROUPS,
            observables=CORR_OBSERVABLES, skip_first=CORR_SKIP_FIRST,
            order=CORR_ORDER, n_boot=CORR_N_BOOT, plot_svd=CORR_PLOT_SVD,
            save_results=SAVE_RESULT, save_figures=SAVE_FIGURES,
            scan_number=SCAN_NUMBER,
        )
        for key, path in out["result_paths"].items():
            print(f"Result [{key}]: {path.name}")
        print(f"Correlations: {out['correlations_path'].name if out['correlations_path'] else '(not saved)'}")
        print(f"Figures ({len(out['figures'])}) in: {out['out_dir']}")
        if OPEN_FIGURES:
            _open_paths(out["figures"])

    print("\nDone.")


if __name__ == "__main__":
    main()
