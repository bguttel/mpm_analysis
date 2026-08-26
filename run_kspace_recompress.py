"""
run_kspace_recompress.py
------------------------
Fast re-compression entry point: build stricter ``min_dark`` compressed click
records from already-compressed records, without re-reading ``individual_runs``.

Typical use:
  SOURCE_MIN_DARK = 1
  TARGET_MIN_DARK = 2

This writes to ``<scan_folder>/compressed_click_analysis_mindark2/`` by default.
Point ``run_kspace_analysis.py`` at that folder to analyze the recompressed data.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from mpm_analysis.io.experimental import (
    CompressedClickLoader,
    recompress_folder_min_dark,
)
from mpm_analysis.plotting.compressed_click_stats import (
    save_compressed_click_figures,
)
from mpm_analysis.pipeline.kspace_pipeline import _open_paths
from mpm_analysis.utils.lab_paths import get_lab_path

# --- configuration ---------------------------------------------------------
SCAN_NUMBER = 22
RATES = None                 # e.g. [1425, 1590]; None = all rates
SOURCE_MIN_DARK = 1          # the rule used by the existing compressed files
TARGET_MIN_DARK = 2          # the stricter rule to build
MAKE_FIGURES = True
OPEN_FIGURES = False

SCAN_FOLDER = get_lab_path(
    "Experiments", "Zeno", "CavityRecAl6", "Chip_TransSap46_5", "OPX",
    "two_transmons", "click_record_repeated", "Omega_BG0_scan",
    f"scan_experiment_results_{SCAN_NUMBER}",
)

INPUT_DIR = SCAN_FOLDER / "compressed_click_analysis"
OUT_DIR = SCAN_FOLDER / f"compressed_click_analysis_mindark{TARGET_MIN_DARK}"


def main() -> None:
    npz_paths = recompress_folder_min_dark(
        INPUT_DIR,
        min_dark=TARGET_MIN_DARK,
        source_min_dark=SOURCE_MIN_DARK,
        rates=RATES,
        out_dir=OUT_DIR,
        latest_only=True,
        verbose=True,
    )
    print(f"\nWrote {len(npz_paths)} recompressed NPZ(s) to:\n  {OUT_DIR}")

    figures = []
    if MAKE_FIGURES:
        records = CompressedClickLoader().load(OUT_DIR)
        for p in save_compressed_click_figures(records, OUT_DIR):
            print(f"  {'figure' if p.suffix == '.png' else 'data  '}: {p.name}")
            # Skip the one-per-λ dead-time histograms when opening.
            if p.suffix == ".png" and "dead_time_hist_lam" not in p.name:
                figures.append(p)

    if OPEN_FIGURES and figures:
        _open_paths(figures)


if __name__ == "__main__":
    main()
