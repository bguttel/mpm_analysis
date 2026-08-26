"""
make_third_transition_figure.py
--------------------------------
Entry point: reproduce the third-transition main-text figure.

Edit the configuration section, then run:
    python make_third_transition_figure.py
"""
from pathlib import Path
import numpy as np

from mpm_analysis.io.analysis_json import load_analysis_result
from mpm_analysis.io.experimental import EnsembleAverageLoader
from mpm_analysis.plotting.paper_figures import make_third_transition_figure
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

PANEL_B_JSON = Path(
    r"C:\Users\admin\OneDrive - weizmann.ac.il\Msc\Zeno\Code\misc\mpm_analysis_results_ens_avg_scan_experiment_results_50_20260129_204832.json"
)

PANEL_A_DATA_FOLDER = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    r"\Common\QuantumZeno\Paper\zeno_paper_data\scan_experiment_results_50"
)
#Common\QuantumZeno\Paper\zeno_paper_data
PANEL_A_FIT_DIR = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    r"\Common\QuantumZeno\Paper\zeno_paper_data\scan_experiment_results_50"
    r"\fitting_results_20260102_155334"
)
    
SAVE_DIR = Path(
    r"C:\Users\admin\Weizmann Institute Dropbox\Noam Nezer\Quantum Circuits Lab"
    r"\Common\QuantumZeno\Paper\Plots\transitions"
)

TARGET_LAMBDAS = np.array([0.4, 1.1, 1.5])
LAMBDA_MAX_B = 1.55
YTICKS_FREQ = [0.0, 0.4, 0.8]

# ============================================================
# RUN
# ============================================================

result = load_analysis_result(PANEL_B_JSON)
print(f"Loaded result: {result.n_lambda} lambda points")

records = []
if PANEL_A_DATA_FOLDER.exists():
    records = EnsembleAverageLoader().load(PANEL_A_DATA_FOLDER)
    print(f"Loaded {len(records)} panel A records")

fig = make_third_transition_figure(
    panel_b_result=result,
    panel_a_records=records,
    fit_dir=PANEL_A_FIT_DIR if PANEL_A_FIT_DIR.exists() else None,
    target_lambdas=TARGET_LAMBDAS,
    lambda_max_b=LAMBDA_MAX_B,
    save_dir=SAVE_DIR if SAVE_DIR.exists() else None,
    yticks_freq=YTICKS_FREQ,
)
plt.show()
