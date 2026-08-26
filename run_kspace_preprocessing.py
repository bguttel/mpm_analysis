"""
run_kspace_preprocessing.py
---------------------------
Preprocessing entry point: build **compressed click records** from the raw
``individual_runs`` of a click_record_repeated scan (compression + counting
statistics + Fourier-over-k + stat figures).

This is the heavy, one-time step.  The k-space pole analysis itself is a
separate, fast entry point: ``run_kspace_analysis.py`` (reads the compressed
records this produces).

Compression rule (``MIN_DARK``):
  1 -> every bright run is a click (``1->0``, default).
  2 -> "2-click" rule: an isolated single-0 gap (``101``) is a false negative,
       so two bright runs split by it merge into one click (``1->00``).

Outputs go to ``<scan_folder>/compressed_click_analysis/`` (timestamped, never
overwriting).  Figures use the ``Agg`` backend and are saved; set
``OPEN_FIGURES`` to pop them open after the run.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")               # headless-safe; we open saved files at the end

from mpm_analysis.pipeline.compressed_click_pipeline import CompressedClickPipeline
from mpm_analysis.pipeline.kspace_pipeline import _open_paths
from mpm_analysis.utils.lab_paths import get_lab_path

# --- configuration ---------------------------------------------------------
SCAN_NUMBERS = [28]
RATES = None            # e.g. [1425, 1590]; None = all rates
MAX_RUNS = None        # e.g. 2 for a quick test; None = all 25 runs
MIN_DARK = 1            # 1 = default 1->0 rule; 2 = "2-click" 1->00 rule
MAKE_FIGURES = True     # counting-statistics + per-λ summary + dead-time PNGs
OPEN_FIGURES = True    # open the saved stat figures after the run
OPEN_PER_LAMBDA = False # also pop open the one-per-λ dead-time histograms

# INDIVIDUAL_RUNS = get_lab_path(
#     "Experiments", "Zeno", "CavityRecAl6", "Chip_TransSap46_5", "OPX",
#     "two_transmons", "click_record_repeated", "Omega_BG0_scan",
#     f"scan_experiment_results_{SCAN_NUMBER}", "individual_runs",
# )


def main() -> None:
    for scan in SCAN_NUMBERS:
        INDIVIDUAL_RUNS = get_lab_path(
            "Experiments", "Zeno", "CavityRecAl6", "Chip_TransSap46_5", "OPX",
            "two_transmons", "click_record_repeated", "Omega_BG0_scan",
            f"scan_experiment_results_{scan}", "individual_runs",
        )
        
        out = CompressedClickPipeline().run(
            INDIVIDUAL_RUNS,
            rates=RATES,
            max_runs=MAX_RUNS,
            min_dark=MIN_DARK,
            make_figures=MAKE_FIGURES,
            run_poles=False,          # analysis is run_kspace_analysis.py's job
        )
        print(f"\nWrote {len(out['npz_paths'])} compressed NPZ(s) to:\n  {out['out_dir']}")
        for p in out["figures"]:
            print(f"  figure: {p.name}")
        for p in out["data_files"]:
            print(f"  data:   {p.name}")
        to_open = out["figures"] if OPEN_PER_LAMBDA else [
            p for p in out["figures"] if "dead_time_hist_lam" not in p.name
        ]
        if OPEN_FIGURES and to_open:
            _open_paths(to_open)


if __name__ == "__main__":
    main()
