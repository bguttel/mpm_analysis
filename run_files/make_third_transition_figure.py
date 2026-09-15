"""
make_third_transition_figure.py
--------------------------------
Entry point: reproduce the third-transition main-text figure.

The analysis inputs are produced by ``run_files/run.py`` with the
``THIRD_EXPORT`` clause on: every run writes a new
``<scan folder>/fitting_results_<stamp>/`` holding the result JSON (panel B)
and one ``*_mpm_fit.json`` per λ (panel A fit overlays).  By default the
NEWEST such folder is used; pin ``FIT_DIR`` to reproduce an older figure.

Edit the configuration section, then run:
    python make_third_transition_figure.py
"""
from pathlib import Path
import numpy as np

from mpm_analysis.io.analysis_json import load_analysis_result
from mpm_analysis.io.experimental import EnsembleAverageLoader
from mpm_analysis.plotting.paper_figures import make_third_transition_figure
from mpm_analysis.utils.lab_paths import get_lab_path
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
# Dropbox root is auto-detected from the hostname (utils/lab_paths.py), the
# same way run.py builds ENS_AVG_DROPBOX — the two must point at one folder.

_ZT = r"Common\QuantumZeno\Paper\zeno_paper_data"
PANEL_A_DATA_FOLDER = Path(get_lab_path(_ZT + r"\scan_experiment_results_50"))

# None → newest fitting_results_* folder inside PANEL_A_DATA_FOLDER;
# or an explicit Path, e.g. PANEL_A_DATA_FOLDER / "fitting_results_20260102_155334".
FIT_DIR: Path | None = None

# None → the result JSON inside FIT_DIR (mpm_*.json, newest if several).
PANEL_B_JSON: Path | None = None

SAVE_DIR = Path(get_lab_path(r"Common\QuantumZeno\Paper\Plots\transitions"))

TARGET_LAMBDAS = np.array([0.4, 1.1, 1.5])
LAMBDA_MAX_B = 1.55
YTICKS_FREQ = [0.0, 0.4, 0.8]

# ============================================================
# RUN
# ============================================================

if FIT_DIR is None:
    candidates = sorted(PANEL_A_DATA_FOLDER.glob("fitting_results_*"))
    if not candidates:
        raise FileNotFoundError(f"No fitting_results_* folder in {PANEL_A_DATA_FOLDER}")
    FIT_DIR = candidates[-1]
if PANEL_B_JSON is None:
    jsons = sorted(FIT_DIR.glob("mpm_*.json"), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise FileNotFoundError(f"No mpm_*.json result file in {FIT_DIR}")
    PANEL_B_JSON = jsons[-1]
print(f"Fit folder : {FIT_DIR}")
print(f"Panel B    : {PANEL_B_JSON.name}")

result = load_analysis_result(PANEL_B_JSON)
print(f"Loaded result: {result.n_lambda} lambda points")

records = []
if PANEL_A_DATA_FOLDER.exists():
    records = EnsembleAverageLoader().load(PANEL_A_DATA_FOLDER)
    print(f"Loaded {len(records)} panel A records")

fig = make_third_transition_figure(
    panel_b_result=result,
    panel_a_records=records,
    fit_dir=FIT_DIR,
    target_lambdas=TARGET_LAMBDAS,
    lambda_max_b=LAMBDA_MAX_B,
    save_dir=SAVE_DIR if SAVE_DIR.exists() else None,
    yticks_freq=YTICKS_FREQ,
)
plt.show()
