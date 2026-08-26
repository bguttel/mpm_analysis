"""
click_record_npz.py
-------------------
Loader for paired JSON + NPZ click-record files.

Each experiment produces two files with the same stem:
  <stem>.json   — parameters, time_array, lambda, etc.
  <stem>.npz    — raw arrays: assigned_states (bool, flat), raw_states, etc.

The ``assigned_states`` array is a flat 1-D boolean array of length N*T,
laid out in (repetition, time) order so that::

    s = assigned_states.astype(float).reshape(N, T)   # (N, T)

gives a binary click matrix where s[i, t] = 1 iff trajectory i had a click
at time step t.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mpm_analysis.io.experimental.base import _get_nested
from mpm_analysis.io.experimental.survival_probability import (
    _read_lambda_from_json,
    _read_time_us,
)
from mpm_analysis.utils.windows_paths import win_long_path


@dataclass
class ClickRecordData:
    """Per-lambda click record loaded from a JSON+NPZ pair.

    Attributes
    ----------
    lambda_val : float
        Lambda (Zeno parameter) read from the JSON.
    t : np.ndarray, shape (T,)
        Time axis in µs.
    trajectories : np.ndarray, shape (N, T), dtype float32
        Binary click matrix: 1 if a click was recorded, 0 otherwise.
    source_file : str
        Stem of the source file pair (without extension).
    metadata : dict
        Additional parameters extracted from the JSON.
    """
    lambda_val: float
    t: np.ndarray
    trajectories: np.ndarray
    source_file: str
    metadata: dict = field(default_factory=dict)

    @property
    def N(self) -> int:
        return self.trajectories.shape[0]

    @property
    def T(self) -> int:
        return self.trajectories.shape[1]


class ClickRecordNpzLoader:
    """Load click-record data from paired JSON + NPZ files in a folder.

    Usage
    -----
    ::

        records = ClickRecordNpzLoader().load(folder)
        # records[i].trajectories.shape == (N, T)
    """

    def load(
        self,
        folder: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        t_max: float | None = None,
    ) -> list[ClickRecordData]:
        """Load all paired JSON+NPZ files in ``folder``.

        Parameters
        ----------
        folder : str | Path
        lambda_range : (lo, hi) or None
            Keep only files with lambda in [lo, hi].
        max_files : int or None
            Load at most this many files.
        t_max : float or None
            Truncate time axis to t <= t_max µs.

        Returns
        -------
        list[ClickRecordData]
            One record per file, sorted by lambda ascending.
        """
        folder = Path(folder)
        npz_files = sorted(folder.glob("*.npz"))
        if max_files is not None:
            npz_files = npz_files[:max_files]

        records: list[ClickRecordData] = []
        for npz_path in npz_files:
            json_path = npz_path.with_suffix(".json")
            if not Path(win_long_path(json_path)).exists():
                print(f"[ClickRecordNpzLoader] Warning: no JSON for {npz_path.name}, skipping.")
                continue
            try:
                rec = self._load_file(npz_path, json_path, t_max=t_max)
            except Exception as exc:
                print(f"[ClickRecordNpzLoader] Warning: skipping {npz_path.name}: {exc}")
                continue

            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= rec.lambda_val <= hi):
                    continue
            records.append(rec)

        records.sort(key=lambda r: r.lambda_val)
        return records

    def _load_file(
        self,
        npz_path: Path,
        json_path: Path,
        *,
        t_max: float | None,
    ) -> ClickRecordData:
        with open(win_long_path(json_path), "r", encoding="utf-8") as f:
            jd = json.load(f)

        lambda_val = _read_lambda_from_json(jd)
        t_us = _read_time_us(jd)

        params = jd.get("parameters", jd)
        N = int(params.get("navg_repeat_experiment", params.get("navg", 1)))
        T = len(t_us)

        npz = np.load(win_long_path(npz_path), allow_pickle=False)
        raw = npz["assigned_states"]
        if raw.size != N * T:
            raise ValueError(
                f"assigned_states has {raw.size} elements, expected N*T = {N}*{T} = {N*T}."
            )
        raw_trajectories = raw.astype(np.float32).reshape(N, T)
        trajectories = self.raw_to_click(raw_trajectories)

        if t_max is not None:
            mask = t_us <= t_max
            t_us = t_us[mask]
            trajectories = trajectories[:, mask]

        meta: dict = {}
        for k in ("navg_repeat_experiment", "navg", "n_avg",
                  "false_positive_max", "false_negative_max"):
            v = params.get(k)
            if v is not None:
                meta[k] = v
        omega_raw = params.get("Omega_DG") or params.get("omega_DG")
        if omega_raw is not None:
            meta["omega_DG"] = float(omega_raw) * 1e-6

        return ClickRecordData(
            lambda_val=lambda_val,
            t=t_us,
            trajectories=trajectories,
            source_file=npz_path.stem,
            metadata=meta,
        )
    
    def raw_to_click(self, raw: np.ndarray) -> np.ndarray:
        """convert a binary sequence to a click sequence, where a click is defined as a transition from 1 to 0."""
        # Find transitions from 1 to 0
        transitions = np.diff(raw, prepend=0) == -1
        return transitions.astype(np.float32)


# ---------------------------------------------------------------------------
# Individual-run groups — per-rate streaming access to the raw run files
# ---------------------------------------------------------------------------
#
# Scans that keep the shot-resolved record only in ``individual_runs/`` (the
# scan-level NPZs store empty ``assigned_states``) have one JSON+NPZ pair per
# run and several runs per rate/λ.  These helpers group the run files by rate
# and stream one boolean click matrix per run, so a rate's runs are never
# concatenated in memory and each run can serve as a bootstrap resampling
# unit downstream.

_RUN_RE = re.compile(r"_run_(\d+)_of_(\d+)_", re.IGNORECASE)


def group_run_files(
    folder: str | Path,
    *,
    max_groups: int | None = None,
) -> list[tuple[str, list[Path]]]:
    """Group ``individual_runs`` NPZs by rate, runs ordered by run index.

    The group key is the filename with the ``_run_i_of_n_`` segment removed;
    NPZs without that segment or without a JSON twin are skipped.
    """
    folder = Path(folder)
    groups: dict[str, list[tuple[int, Path]]] = {}
    for npz_path in sorted(folder.glob("*.npz")):
        m = _RUN_RE.search(npz_path.name)
        if not m:
            continue
        if not Path(win_long_path(npz_path.with_suffix(".json"))).exists():
            print(f"[group_run_files] Warning: no JSON for {npz_path.name}, skipping.")
            continue
        key = npz_path.name[:m.start()] + "_" + npz_path.name[m.end():]
        groups.setdefault(key, []).append((int(m.group(1)), npz_path))

    out = [(key, [p for _, p in sorted(items)])
           for key, items in sorted(groups.items())]
    if max_groups is not None:
        out = out[:max_groups]
    return out


def read_run_header(json_path: str | Path) -> dict:
    """Read the per-λ header shared by all runs of a rate group.

    Returns ``dict(lambda_val, t, rate_khz, metadata)`` — time axis in µs,
    ``omega_DG`` in rad/µs inside ``metadata``.
    """
    from mpm_analysis.io.experimental.base import _khz_from_name, _load_json

    json_path = Path(json_path)
    jd = _load_json(json_path)
    params = jd.get("parameters", jd)

    meta: dict = {}
    for k in ("navg_repeat_experiment", "navg", "n_avg",
              "false_positive_max", "false_negative_max"):
        v = params.get(k)
        if v is not None:
            meta[k] = v
    omega_raw = params.get("Omega_DG") or params.get("omega_DG")
    if omega_raw is not None:
        meta["omega_DG"] = float(omega_raw) * 1e-6

    try:
        rate_khz = _khz_from_name(json_path)
    except ValueError:
        rate_khz = None

    return dict(
        lambda_val=_read_lambda_from_json(jd),
        t=_read_time_us(jd),
        rate_khz=rate_khz,
        metadata=meta,
    )


def load_run_click_matrix(npz_path: str | Path, n_bins: int) -> np.ndarray:
    """Load one run NPZ as a boolean ``(N, T)`` click matrix (falling edge).

    ``N`` is inferred from the flat ``assigned_states`` length; the click
    convention is the same 1→0 transition as :meth:`ClickRecordNpzLoader.raw_to_click`
    (bin 0 never clicks), but computed in-place on booleans to stay lean for
    10^5-shot runs.
    """
    from mpm_analysis.io.experimental.compressed_click_record import _load_npz_retry

    z = _load_npz_retry(Path(npz_path))
    raw = z["assigned_states"]
    if raw.size == 0 or raw.size % n_bins:
        raise ValueError(
            f"{Path(npz_path).name}: assigned_states has {raw.size} elements, "
            f"not a multiple of {n_bins} time bins."
        )
    raw = (raw.reshape(-1, n_bins) > 0)
    clicks = np.zeros_like(raw)
    clicks[:, 1:] = raw[:, :-1] & ~raw[:, 1:]
    return clicks


def iter_click_run_groups(
    folder: str | Path,
    *,
    lambda_range: tuple[float, float] | None = None,
    max_groups: int | None = None,
):
    """Yield one dict per rate group of an ``individual_runs`` folder.

    Each yielded dict has ``key, lambda_val, rate_khz, t, metadata, runs``,
    where ``runs`` is a *generator* of boolean click matrices (one per run
    file, in run order).  Consume each group's generator before advancing —
    only one run matrix is ever in memory.
    """
    for key, npz_paths in group_run_files(folder, max_groups=max_groups):
        header = read_run_header(npz_paths[0].with_suffix(".json"))
        if lambda_range is not None:
            lo, hi = lambda_range
            if not (lo <= header["lambda_val"] <= hi):
                continue
        n_bins = len(header["t"])
        meta = dict(header["metadata"])
        meta["source_file"] = key
        meta["n_run_files"] = len(npz_paths)

        def _runs(paths=npz_paths, nb=n_bins):
            for p in paths:
                yield load_run_click_matrix(p, nb)

        yield dict(
            key=key,
            lambda_val=header["lambda_val"],
            rate_khz=header["rate_khz"],
            t=header["t"],
            metadata=meta,
            runs=_runs(),
        )