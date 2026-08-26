"""
run_kspace_theory_braiding.py
-----------------------------
Theory-side braiding report using the same plotting and paper-index machinery
as the experimental k-space analysis.

The model implementation lives in Theory/braid_tools_complete.py, which is the
cleaned-up/validated version of the fake Lindbladian script. It uses the same
3x3 generator as Theory/fake_lindbladian_braid_fixed_2.py.

Run:
    python run_kspace_theory_braiding.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import numpy as np

from mpm_analysis.analysis import braiding as braid
from mpm_analysis.kspace import KSpacePolesResult, save_kspace_figures
from mpm_analysis.pipeline.kspace_pipeline import _open_paths
from mpm_analysis.io.experimental.compressed_click_record import _timestamp

# --- parameters ------------------------------------------------------------
MODEL = "3x3"                 # "3x3" | "4x4"
OMEGA = 0.628 * 1.5           # Omega_S, rad/us
T1 = 120.0                    # us
T2 = 120.0                    # us
TAU_B = 0.004                 # us; 4x4 only
K_FP = 1e-3                   # false-positive rate, rad/us
FN = 0.0                      # readout false-negative probability; 4x4 only
T_BIN = 0.0                   # detection time-bin length, us; 4x4 only
LAMBDA_SCALE = 1.0

LAM = np.linspace(0.2, 3.5, 30)
K = np.linspace(0.0, 2.0 * np.pi, 201)
K_INDEX_RANGE = (1, 200)      # drop exact endpoints for the display figures

OUT_DIR = Path(__file__).parent / "results" / "theory_braiding"
OPEN_FIGURES = False


def main() -> None:
    tp = braid.theory_poles_on_grid(
        LAM, K, Omega=OMEGA, T1=T1, T2=T2, tau_B=TAU_B, k_fp=K_FP,
        fn=FN, t_bin=T_BIN, lambda_scale=LAMBDA_SCALE, model=MODEL,
    )
    sl = slice(*K_INDEX_RANGE)
    k = K[sl]
    poles = tp[:, sl, :]
    order = poles.shape[2]

    result = KSpacePolesResult(
        lambda_values=LAM,
        k_array=k,
        t=np.arange(2, dtype=float),
        poles=poles,
        amplitudes=np.full_like(poles, np.nan),
        pole_order=np.full((len(LAM), len(k)), order, dtype=int),
        svd_singvals=np.zeros((len(LAM), len(k), 1)),
        metadata={
            "source": "theory",
            "max_order": order,
            "min_order": order,
            "model": MODEL,
            "Omega": OMEGA,
            "T1": T1,
            "T2": T2,
            "k_fp": K_FP,
            "fn": FN,
            "t_bin": T_BIN,
        },
    )
    braiding = braid.theory_braiding(
        LAM, Omega=OMEGA, T1=T1, T2=T2, tau_B=TAU_B, k_fp=K_FP,
        fn=FN, t_bin=T_BIN, lambda_scale=LAMBDA_SCALE, model=MODEL,
    )
    braid.print_braiding_summary(braiding, label="theory")

    run_dir = OUT_DIR / f"theory_braiding_{_timestamp()}"
    figs = save_kspace_figures(
        result,
        run_dir,
        records=None,
        source_tag="theory",
        include_svd=False,
        include_3d=True,
        include_html=True,
        braiding=braiding,
        braiding_label="theory",
    )
    print(f"\nFigures ({len(figs)}) in: {run_dir}")
    if OPEN_FIGURES:
        _open_paths(figs)


if __name__ == "__main__":
    main()
