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

Segments (post-selected tomography format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``click_record_with_post_selected_tomography`` scans (e.g. scan 10) store a
*segmented* record instead: every point of the ``T``-point time axis is
measured by its own run — segment ``i`` (``i = 1 … T-1``) evolves till
``t_i = i·dt`` and records ``i`` fast-measurement bins.  Their NPZ's
``assigned_states`` is 3-D bool, shape ``(n_reps, n_axes, Σ i)``:
repetitions × tomography measurement settings × the segments concatenated in
ascending-length order (verified on scan 10: per-bin means collapse when
segments are aligned at their starts).  The loader detects this layout and
returns one record per file with

* the measurement-settings axis flattened into the repetition axis —
  ``trajectories`` has ``n_reps · n_axes`` rows (180 000 for scan 10), row
  ``rep · n_axes + axis``, preserving acquisition order;
* the ragged time axis kept concatenated, described by
  ``ClickRecordData.segment_lengths`` (= ``1, 2, …, T-1``), so stored segment
  ``i`` (0-based, ``i+1`` bins) is the view ``rec.segment(i)`` and
  consecutive segments stay adjacent for correlation / bias checks.

These trajectories hold the thresholded bright/dark assignments *verbatim*
(bool) — no falling-edge click conversion: this scan's own ``p_0_t``
(``p0_definition = no_clicks_count``) counts a repetition as surviving iff no
bin of the segment is bright, i.e. a bright bin is itself the click.
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
    trajectories : np.ndarray, shape (N, T) or (N, Σ segment_lengths)
        Classic records: binary click matrix (float32), 1 if a click was
        recorded, 0 otherwise, one column per point of ``t``.  Segmented
        records (``segment_lengths`` set): the thresholded bright/dark
        assignments verbatim (bool), segments concatenated along the second
        axis — see the module docstring.
    source_file : str
        Stem of the source file pair (without extension).
    metadata : dict
        Additional parameters extracted from the JSON.  Segmented records add
        ``n_reps`` and ``n_axes`` (row = rep·n_axes + axis).
    segment_lengths : np.ndarray or None, shape (n_segments,)
        Length in time bins of each acquisition segment concatenated along
        ``trajectories``' second axis, in stored (acquisition) order — for
        the tomography format ``1, 2, …, T-1``, stored segment ``i``
        (0-based) evolving till ``t[i+1]``.  ``None`` means a single uniform
        time axis (classic record).
    """
    lambda_val: float
    t: np.ndarray
    trajectories: np.ndarray
    source_file: str
    metadata: dict = field(default_factory=dict)
    segment_lengths: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.segment_lengths is not None:
            self.segment_lengths = np.asarray(self.segment_lengths,
                                              dtype=np.int64)
            if int(self.segment_lengths.sum()) != self.trajectories.shape[1]:
                raise ValueError(
                    f"segment_lengths sum to {int(self.segment_lengths.sum())} "
                    f"but trajectories has {self.trajectories.shape[1]} columns."
                )

    @property
    def N(self) -> int:
        return self.trajectories.shape[0]

    @property
    def T(self) -> int:
        """Number of columns — for segmented records this is Σ segment_lengths,
        not the length of ``t``."""
        return self.trajectories.shape[1]

    # ------------------------------------------------------------------
    # Segment access — views into the concatenated ragged time axis
    # ------------------------------------------------------------------

    @property
    def n_segments(self) -> int:
        return 1 if self.segment_lengths is None else len(self.segment_lengths)

    @property
    def segment_offsets(self) -> np.ndarray:
        """Column offsets, shape (n_segments+1,): segment ``i`` spans
        ``trajectories[:, segment_offsets[i]:segment_offsets[i+1]]``."""
        lengths = (np.array([self.trajectories.shape[1]], dtype=np.int64)
                   if self.segment_lengths is None else self.segment_lengths)
        return np.concatenate(([0], np.cumsum(lengths)))

    @property
    def segment_index(self) -> np.ndarray:
        """Segment id (0-based) of every column of ``trajectories`` —
        handy for grouping bins by segment."""
        lengths = (np.array([self.trajectories.shape[1]], dtype=np.int64)
                   if self.segment_lengths is None else self.segment_lengths)
        return np.repeat(np.arange(len(lengths)), lengths)

    def segment(self, i: int) -> np.ndarray:
        """Record of stored segment ``i`` (0-based): an
        ``(N, segment_lengths[i])`` view — every repetition of the run that
        lasted ``segment_lengths[i]`` time bins."""
        o = self.segment_offsets
        return self.trajectories[:, o[i]:o[i + 1]]

    def iter_segments(self):
        """Yield each segment's ``(N, length)`` record, in acquisition order."""
        o = self.segment_offsets
        for i in range(len(o) - 1):
            yield self.trajectories[:, o[i]:o[i + 1]]


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

        Files with a 3-D ``assigned_states`` (the segmented post-selected
        tomography format — see module docstring) come back segment-aware:
        ``segment_lengths`` set, measurement-settings axis flattened into the
        repetition axis.

        Parameters
        ----------
        folder : str | Path
        lambda_range : (lo, hi) or None
            Keep only files with lambda in [lo, hi].
        max_files : int or None
            Load at most this many files.
        t_max : float or None
            Truncate time axis to t <= t_max µs (segmented records: drop the
            segments that evolve past ``t_max``).

        Returns
        -------
        list[ClickRecordData]
            One record per file, sorted by lambda ascending.
        """
        records = list(self.iter_records(
            folder, lambda_range=lambda_range, max_files=max_files, t_max=t_max,
        ))
        records.sort(key=lambda r: r.lambda_val)
        return records

    def iter_records(
        self,
        folder: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        t_max: float | None = None,
    ):
        """Yield records one file at a time, in filename (NOT lambda) order.

        Streaming counterpart of :meth:`load` for folders whose per-file
        records are too large to hold together (a segmented tomography file
        is ~300 MB in memory) — only one record is alive per iteration.
        Same parameters and skip behaviour as :meth:`load`.
        """
        folder = Path(folder)
        npz_files = sorted(folder.glob("*.npz"))
        if max_files is not None:
            npz_files = npz_files[:max_files]

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
            yield rec

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

        meta: dict = {}
        for k in ("navg_repeat_experiment", "navg", "n_avg",
                  "false_positive_max", "false_negative_max"):
            v = params.get(k)
            if v is not None:
                meta[k] = v
        omega_raw = params.get("Omega_DG") or params.get("omega_DG")
        if omega_raw is not None:
            meta["omega_DG"] = float(omega_raw) * 1e-6

        if raw.ndim == 3:
            # Segmented post-selected tomography format — see module
            # docstring.  (n_reps, n_axes, Σ i) bool: segments of ascending
            # length 1..T-1 concatenated on the last axis, one per t > 0.
            n_reps, n_axes, total = raw.shape
            segment_lengths = np.arange(1, T, dtype=np.int64)
            if int(segment_lengths.sum()) != total:
                raise ValueError(
                    f"assigned_states last axis has {total} bins, expected "
                    f"1+2+…+{T - 1} = {T * (T - 1) // 2} for the {T}-point "
                    f"time axis."
                )
            trajectories = raw.reshape(n_reps * n_axes, total)  # view, bool
            if t_max is not None:
                n_seg = int(np.count_nonzero(t_us[1:] <= t_max))
                segment_lengths = segment_lengths[:n_seg]
                trajectories = trajectories[:, :int(segment_lengths.sum())]
                t_us = t_us[:n_seg + 1]
            meta["n_reps"] = n_reps
            meta["n_axes"] = n_axes
            return ClickRecordData(
                lambda_val=lambda_val,
                t=t_us,
                trajectories=trajectories,
                source_file=npz_path.stem,
                metadata=meta,
                segment_lengths=segment_lengths,
            )

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
# Segment-pooled survival — first-click histogram of a segmented record
# ---------------------------------------------------------------------------
#
# The scan's own ``p_0_t_array[i]`` uses only the run that evolved till
# ``t_i`` (stored segment ``i``, ``i`` bins).  But every LONGER segment also
# passed through bin ``i``: a shot of segment ``L > i`` that has no bright bin
# among its first ``i`` bins is a success for ``t_i`` just the same.  Pooling
# the segments ``L >= i`` multiplies the attempts behind ``t_i`` by
# ``n_seg - i + 1`` (59x at the first point, 1x at the last).
#
# The sufficient statistic for all of this is, per axis and per segment, the
# histogram of the FIRST bright bin: a shot is a success for ``t_i`` iff its
# first bright bin is later than ``i`` (or it never went bright).  From that
# histogram alone one gets the original curve (segment ``i`` only), the
# pooled curve, its binomial error, and an exact shot-noise bootstrap that
# keeps the nesting of the time points (one multinomial draw per axis and
# segment, never independent binomials per time point).

def first_click_counts(rec: "ClickRecordData") -> np.ndarray:
    """Histogram of the first bright bin, per axis and per stored segment.

    Returns an int64 array ``counts[a, L-1, k]`` of shape
    ``(n_axes, n_seg, n_seg + 1)``: shots of tomography axis ``a`` in the
    segment of ``L`` bins (``L = 1 … n_seg``) whose first bright bin is bin
    ``k`` (1-based, ``k = 1 … L``); ``k = 0`` counts the shots with no bright
    bin in the segment.  Entries ``k > L`` are zero.  Each row sums to the
    repetitions per axis (``n_reps``).

    Only segmented records qualify (``segment_lengths = 1 … n_seg``);
    rows are ``rep · n_axes + axis`` as the loader lays them out.
    """
    if rec.segment_lengths is None:
        raise ValueError("first_click_counts needs a segmented record.")
    lengths = rec.segment_lengths
    n_seg = len(lengths)
    if not np.array_equal(lengths, np.arange(1, n_seg + 1)):
        raise ValueError("first_click_counts expects segment lengths 1 … n_seg "
                         f"(got {lengths[:5]}…).")
    n_axes = int(rec.metadata.get("n_axes", 1))
    rows = rec.trajectories
    axis_of_row = np.arange(rows.shape[0]) % n_axes
    o = rec.segment_offsets

    counts = np.zeros((n_axes, n_seg, n_seg + 1), dtype=np.int64)
    for j in range(n_seg):
        seg = rows[:, o[j]:o[j + 1]]                    # (N, L) bool
        clicked = seg.any(axis=1)
        first = np.where(clicked, seg.argmax(axis=1) + 1, 0)   # 0 = none
        for a in range(n_axes):
            counts[a, j, :] = np.bincount(first[axis_of_row == a],
                                          minlength=n_seg + 1)
    return counts


def pooled_survival_from_counts(
    counts: np.ndarray,
    *,
    min_extra_bins: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment-pooled survival curve from a first-click histogram.

    ``counts`` is the ``(n_axes, n_seg, n_seg + 1)`` array of
    :func:`first_click_counts` (or one multinomial resample of it).  For the
    time index ``i = 1 … n_seg`` every segment ``L >= i + min_extra_bins``
    contributes all its shots as attempts and, as successes, those whose first
    bright bin is later than ``i`` (or absent).  Index 0 is the ``t = 0``
    anchor: every shot survives, so its attempts are the whole file.

    ``min_extra_bins = 0`` (default) pools every segment that reaches ``t_i``;
    ``1`` leaves out the segment that ENDS at ``t_i`` — i.e. each segment's
    own last bin, which the assignment filter is known to read bright too
    often — at the price of an undefined last point (NaN).

    Returns ``(fraction, n_success, n_attempts)``, each of shape
    ``(n_seg + 1,)`` aligned with the file's ``time_array``.  The fraction is
    the pooled ratio over the axes (they carry equal attempts, so it is also
    the mean of the per-axis fractions).
    """
    counts = np.asarray(counts, dtype=np.int64)
    n_axes, n_seg, n_k = counts.shape
    if n_k != n_seg + 1:
        raise ValueError(f"counts has {n_k} first-click slots for {n_seg} "
                         f"segments; expected n_seg + 1.")
    per_seg = counts.sum(axis=-1)                       # (A, n_seg) shots
    # tail[a, L-1, m] = shots whose first bright bin is >= m  (m = 1 … n_seg),
    # with a zero column appended so tail[..., n_seg + 1] exists.
    tail = np.cumsum(counts[:, :, :0:-1], axis=-1)[:, :, ::-1]
    tail = np.concatenate([tail, np.zeros_like(tail[:, :, :1])], axis=-1)

    L = np.arange(1, n_seg + 1)
    n_success = np.zeros(n_seg + 1, dtype=np.int64)
    n_attempts = np.zeros(n_seg + 1, dtype=np.int64)
    n_success[0] = n_attempts[0] = int(per_seg.sum())
    for i in range(1, n_seg + 1):
        use = L >= i + min_extra_bins
        if not use.any():
            continue
        # survivors to bin i in segment L: never bright + first bright > i
        surv = counts[:, use, 0] + tail[:, use, i]      # tail index i ↔ m=i+1
        n_success[i] = int(surv.sum())
        n_attempts[i] = int(per_seg[:, use].sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n_attempts > 0,
                            n_success / np.maximum(n_attempts, 1), np.nan)
    return fraction, n_success, n_attempts


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

if __name__ == "__main__":
    pass    