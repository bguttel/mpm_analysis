"""
io/simulated
------------
Loaders for simulation output produced by ``generate_click_record_raw.py``.

Usage
~~~~~
    from mpm_analysis.io.simulated import SimClickRecordLoader

    # Full trajectories — compatible with connected_correlator() etc.
    records = SimClickRecordLoader().load("click_record_raw_20260525_195204.npz")

    # Pre-computed ⟨n(t)⟩ from saved P(n,t) — fast, no trajectory processing
    obs = SimClickRecordLoader().load_observable("click_record_raw_20260525_195204.npz")
"""
from mpm_analysis.io.simulated.click_record_raw_npz import SimClickRecordLoader
from mpm_analysis.io.simulated.shifted_resolution import (
    fine_click_probs,
    no_click_curve,
    poisson_binomial_pnt,
    reconstruct,
)

__all__ = [
    "SimClickRecordLoader",
    "no_click_curve",
    "fine_click_probs",
    "poisson_binomial_pnt",
    "reconstruct",
]
