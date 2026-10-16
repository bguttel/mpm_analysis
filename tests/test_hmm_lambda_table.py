"""Tests for HMM lambda-table discovery and provenance.

The bug these lock down: the table was keyed on the ``OBG0`` frequency
fragment alone, so a results.csv filed next to a *different* scan silently
matched files by drive frequency and rescaled the whole lambda axis by ~10 %.

Covers:
- run ids keep frequency AND timestamp, across the differing file prefixes
- a table is discovered from a sibling scan folder when it covers this one
- a foreign table is never discovered, and raises when passed explicitly
- csv_path=False forces the stored JSON lambda
- duplicate rows raise instead of last-one-wins
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from mpm_analysis.io.experimental import (
    EnsembleAverageLoader,
    HMMTableMismatch,
    discover_hmm_table,
)
from mpm_analysis.io.experimental.survival_probability import _hmm_key

OMEGA_DG_HZ = 628318.53          # 2*pi*100 kHz, as in the real files
OMEGA_DG_MHZ = OMEGA_DG_HZ * 1e-6


def _write_scan(folder, runs, *, lam_json, prefix="survival probability_data__"):
    """A minimal ens_avg scan folder: one JSON per run id."""
    folder.mkdir(parents=True, exist_ok=True)
    t_ns = np.arange(0, 20) * 100.0
    for run in runs:
        jd = {
            "parameters": {
                "Omega_DG": OMEGA_DG_HZ,
                "lambda_zeno_estimation": lam_json,
                "navg_repeat_experiment": 1000,
            },
            "data": {
                "time_array": t_ns.tolist(),
                "D_probability": np.full(t_ns.size, 0.5).tolist(),
                "G_probability": np.full(t_ns.size, 0.3).tolist(),
                "B_probability": np.full(t_ns.size, 0.1).tolist(),
                "error_probability": np.full(t_ns.size, 0.1).tolist(),
            },
        }
        (folder / f"{prefix}{run}.json").write_text(json.dumps(jd))
    return folder


def _write_table(path, runs, gamma_up, *, prefix="click_record_data__"):
    """A results.csv keyed by filename, as the HMM pipeline emits it."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "Gamma_up_3_states_hmm"])
        w.writeheader()
        for run in runs:
            w.writerow({"filename": f"{prefix}{run}",
                        "Gamma_up_3_states_hmm": gamma_up})


MINE = ["OBG0_1020_kHz_07_12_2025__11_45_42", "OBG0_1035_kHz_07_12_2025__11_48_06"]
# same drive frequencies, different day: this is what used to collide
THEIRS = ["OBG0_1020_kHz_12_09_2025__06_11_53", "OBG0_1035_kHz_12_09_2025__06_57_49"]


def test_run_id_keeps_frequency_and_timestamp():
    assert _hmm_key("survival probability_data__" + MINE[0]) == MINE[0]
    assert _hmm_key("click_record_data__" + MINE[0] + ".json") == MINE[0]
    # the collision the old fragment key produced
    assert _hmm_key("x_data__" + MINE[0]) != _hmm_key("x_data__" + THEIRS[0])


def test_discovers_table_from_sibling_folder(tmp_path, capsys):
    mine = _write_scan(tmp_path / "scan_50", MINE, lam_json=0.5)
    _write_scan(tmp_path / "scan_10", THEIRS, lam_json=0.5,
                prefix="click_record_with_post_selected_tomography_data__")
    # filed next to the wrong scan, exactly as in the real dataset
    _write_table(tmp_path / "scan_10" / "results.csv", MINE, gamma_up=1.0)

    assert discover_hmm_table(mine) == tmp_path / "scan_10" / "results.csv"

    records = EnsembleAverageLoader().load(mine, observables=["D"])
    assert sorted({r.lambda_val for r in records}) == pytest.approx(
        [1.0 / (2 * OMEGA_DG_MHZ)]
    )
    assert "results.csv" in capsys.readouterr().out


def test_foreign_table_is_never_applied(tmp_path):
    theirs = _write_scan(tmp_path / "scan_10", THEIRS, lam_json=0.5)
    table = tmp_path / "scan_10" / "results.csv"
    _write_table(table, MINE, gamma_up=1.0)   # covers the *other* scan only

    # not discovered even though it sits in this very folder
    assert discover_hmm_table(theirs) is None
    records = EnsembleAverageLoader().load(theirs, observables=["D"])
    assert {r.lambda_val for r in records} == {0.5}

    # and forcing it is an error, not a silent 10 % shift
    with pytest.raises(HMMTableMismatch, match="describes no run"):
        EnsembleAverageLoader(csv_path=table).load(theirs, observables=["D"])


def test_false_forces_json_lambda(tmp_path):
    mine = _write_scan(tmp_path / "scan_50", MINE, lam_json=0.5)
    _write_table(tmp_path / "scan_50" / "results.csv", MINE, gamma_up=1.0)

    assert discover_hmm_table(mine) is not None
    records = EnsembleAverageLoader(csv_path=False).load(mine, observables=["D"])
    assert {r.lambda_val for r in records} == {0.5}


def test_duplicate_rows_raise(tmp_path):
    mine = _write_scan(tmp_path / "scan_50", MINE, lam_json=0.5)
    _write_table(tmp_path / "scan_50" / "results.csv", MINE + MINE[:1], gamma_up=1.0)

    with pytest.raises(HMMTableMismatch, match="duplicate row"):
        EnsembleAverageLoader().load(mine, observables=["D"])


def test_non_hmm_csv_is_ignored(tmp_path):
    """results_df.csv holds only Omega_BG0 — never a lambda rescale."""
    mine = _write_scan(tmp_path / "scan_50", MINE, lam_json=0.5)
    with open(tmp_path / "scan_50" / "results_df.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Omega_BG0"])
        w.writerow([0, 6.3])
    assert discover_hmm_table(mine) is None


def _write_drive_table(path, khz, gamma_per_s):
    """A per-drive table, as hmm_rates_per_drive.csv (Gamma_up in 1/s)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drive_khz", "gamma_BG_per_s_mean"])
        w.writerows(zip(khz, gamma_per_s))


def test_drive_table_exact_lookup(tmp_path):
    """Keyed by drive, not run: exact lookup at each file's drive, explicit only."""
    mine = _write_scan(tmp_path / "scan_0", MINE, lam_json=0.5)   # 1020, 1035 kHz
    table = mine / "hmm_rates_per_drive.csv"
    _write_drive_table(table, [1020, 1035, 1100], [1e6, 2e6, 3e6])

    assert discover_hmm_table(mine) is None
    records = EnsembleAverageLoader(csv_path=table).load(mine, observables=["D"])
    assert sorted({r.lambda_val for r in records}) == pytest.approx(
        [1.0 / (2 * OMEGA_DG_MHZ), 2.0 / (2 * OMEGA_DG_MHZ)]
    )

    _write_drive_table(table, [1000, 1035, 1100], [1e6, 2e6, 3e6])   # 1020 missing
    with pytest.raises(HMMTableMismatch, match="no row for drives \[1020\]"):
        EnsembleAverageLoader(csv_path=table).load(mine, observables=["D"])
