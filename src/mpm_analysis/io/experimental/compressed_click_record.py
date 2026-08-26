"""
compressed_click_record.py
--------------------------
Build and load **compressed click records** from the raw per-run click-record
NPZ files of a ``click_record_repeated`` scan.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~~~
The raw per-shot thresholded record lives only in the ``individual_runs/``
sub-folder of a scan (e.g.
``…/Omega_BG0_scan/scan_experiment_results_<N>/individual_runs/``).  Each run
NPZ holds ``assigned_states`` — a flat bool array of length ``N·T`` (here
``80000·280 = 22 400 000``), laid out in (shot, time) order — and there are 25
runs per rate, ~21 rates, ~69 GB total.  This is far too large to analyse
repeatedly, so we pre-process it once into a compact, analysis-ready form.

Physical picture
~~~~~~~~~~~~~~~~~
Each shot is a dark(0)/bright(1) telegraph.  A **click** is a bright run (a run
of consecutive 1s); only the ``1→0`` structure carries information.  We register
one click at the **first 1** of every bright run (including a trailing bright run
that never returns to 0 within the window).

Compression ("bright run → 1 step")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each maximal bright run of length ``L`` is collapsed to a single retained ``1``
(the click), immediately followed by its ``0``.  The other ``L-1`` ones are
dropped, and everything after shifts earlier by ``L-1``.  Equivalently, a raw
time step ``i`` is *removed* iff it is a 1 whose predecessor is also a 1; the
compressed index of any surviving step is ``i − (#removed steps before i)``.
This is O(#clicks) per shot (one prefix sum), never O(T·#clicks).

The compressed time axis is ``t_compressed = index × dt`` (dt = raw step, µs).
Because each shot removes a different amount of bright dwell, **compressed
lengths differ per shot** — captured in ``compressed_lengths``.

What we store (and why it is enough)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Only the **compressed click times** per shot are kept as the raw record.  Since
every bright run is normalised to length 1, the dark-run length between two
consecutive clicks is exactly ``Δt_click − 1``.  So the full dark-gap structure
is recoverable from click times alone, and future rules (e.g. ``1→00`` is a
click, ``101`` is a false negative) can be applied **without** re-reading the
69 GB originals — see :func:`recompress_min_dark`.

Counting statistics ``P(n,t)`` and per-run histograms are also stored so that
bootstrapping (resampling the 25 runs) needs no per-shot reprocessing, together
with the Fourier transform over the counting field ``k`` on the existing
101-point grid (reusing :func:`mpm_analysis.kspace.loader.compute_pkt_from_pnt`).

Dead time (bright-run length) statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The compression throws the ``L-1`` extra ``1``s of each bright run away, and
only the per-shot *total* dwell survives (as ``T − compressed_length``).  The
**distribution** of those ``L`` — how long the ion stays bright after a click,
i.e. the detector dead time — is therefore not recoverable afterwards, so
:func:`compress_run` accumulates it during the single pass over the raw data and
the builder stores it as ``bright_hist`` (per run; bin ``L`` = number of bright
runs of exactly ``L`` raw steps).  Cost is ``O(#clicks)``: the run lengths come
out of the prefix sum the compression already computes, with no second scan of
the ``(N,T)`` matrix.  ``bright_hist`` always describes the raw ``1->0`` bright
runs, independent of ``min_dark``.  :func:`fit_dead_time_exponential` gives the
MLE exponential decay constant straight from such a histogram.

Reuses existing infrastructure: ``win_long_path``, ``_load_json``,
``_read_lambda_from_json``, ``_read_time_us``, ``_khz_from_name``.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mpm_analysis.io.experimental.base import _khz_from_name, _load_json
from mpm_analysis.io.experimental.survival_probability import (
    _read_lambda_from_json,
    _read_time_us,
)
from mpm_analysis.utils.windows_paths import win_long_path

SCHEMA_VERSION = 2      # 2 adds ``bright_hist`` (the dead-time distribution)
DEFAULT_OUT_SUBDIR = "compressed_click_analysis"
DEFAULT_K_GRID_N = 101

_RUN_RE = re.compile(r"_run_(\d+)_of_(\d+)_", re.IGNORECASE)
_COMPRESSED_TS_RE = re.compile(r"_(\d{8}_\d{6})(?:_\d+)?\.npz$", re.IGNORECASE)


def _load_npz_retry(path, *, retries: int = 5, delay: float = 3.0):
    """``np.load`` with retries for cloud-sync (Dropbox/OneDrive) placeholders.

    These raw run NPZs are large online-only files.  The first read can hit an
    unhydrated placeholder and fail (numpy reads an empty magic and raises).
    Retrying with a short pause lets the sync client finish downloading.
    """
    for attempt in range(retries):
        try:
            return np.load(win_long_path(path), allow_pickle=False)
        except (OSError, ValueError, EOFError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# ===========================================================================
# Core compression — the validated, linear-in-clicks algorithm
# ===========================================================================

def compress_run(assigned: np.ndarray) -> dict:
    """Compress one run's ``assigned_states`` matrix into per-shot click times.

    Parameters
    ----------
    assigned : np.ndarray, shape (N, T), bool/uint8
        Thresholded telegraph for ``N`` shots over ``T`` raw time steps.
        ``assigned[i, t] == 1`` iff shot ``i`` is bright at step ``t``.

    Returns
    -------
    dict with keys
        ``click_times`` : int32, shape (Σ clicks,)
            Compressed click-time indices, grouped by shot then time-ordered.
        ``shot_offsets`` : int64, shape (N+1,)
            ``click_times[shot_offsets[i]:shot_offsets[i+1]]`` are shot ``i``'s
            clicks.
        ``compressed_lengths`` : int32, shape (N,)
            Per-shot compressed length ``Lc = T − Σ(L−1)``.
        ``jump_counts`` : int32, shape (N,)
            Per-shot number of clicks (= number of bright runs).
        ``bright_hist`` : int64, shape (T+1,)
            Dead-time histogram: ``bright_hist[L]`` = number of bright runs of
            exactly ``L`` raw steps, pooled over the run's ``N`` shots.  This is
            the distribution of the ``1``s the compression collapses, and the
            only place it is ever available — see Notes.

    Notes
    -----
    A step ``i`` is *removed* iff ``assigned[i] and assigned[i-1]`` (a 1 that is
    not the first of its bright run).  A bright run *starts* where
    ``assigned[i] and not assigned[i-1]``; that first-1 is the click.  Its
    compressed time is ``i − cum_removed_before(i)``.

    The bright-run lengths fall out of ``cum_removed`` for free.  No removals
    happen in the dark stretch between two runs, so the removed count at
    consecutive starts differs by exactly ``L−1``:

        ``L_j = cum_removed[start_{j+1}] − cum_removed[start_j] + 1``

    with the row's total removed count standing in for the last run's successor.
    That is ``O(#clicks)`` on arrays the compression already built — no second
    pass over the ``(N, T)`` matrix and no extra ``(N, T)`` temporary.
    """
    ass = np.ascontiguousarray(assigned).astype(np.uint8, copy=False)
    if ass.ndim != 2:
        raise ValueError(f"assigned must be 2-D (N, T); got shape {ass.shape}")
    N, T = ass.shape

    prev = np.zeros_like(ass)
    prev[:, 1:] = ass[:, :-1]                       # predecessor along time

    removed = ass & prev                            # 1 with a 1 before it
    cum_removed = np.cumsum(removed, axis=1, dtype=np.int32)   # inclusive

    starts = ass & (prev ^ 1)                       # 1 with a 0 (or edge) before
    # compressed index of every step; at a start, removed[i]==0 so inclusive ==
    # exclusive prefix, i.e. exactly "i − #removed strictly before i".
    compressed_index = (
        np.arange(T, dtype=np.int32)[None, :] - cum_removed
    )

    rows, cols = np.nonzero(starts)                 # row-major → grouped by shot
    click_times = compressed_index[rows, cols].astype(np.int32, copy=False)

    jump_counts = np.bincount(rows, minlength=N).astype(np.int32)
    shot_offsets = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(jump_counts, out=shot_offsets[1:])

    compressed_lengths = (T - cum_removed[:, -1]).astype(np.int32)

    # Bright-run (dead-time) lengths from the same prefix sum — see Notes.
    removed_before = cum_removed[rows, cols].astype(np.int64)
    removed_after = np.empty_like(removed_before)
    if removed_before.size:
        removed_after[:-1] = removed_before[1:]
        has_click = jump_counts > 0
        last_of_shot = np.cumsum(jump_counts, dtype=np.int64) - 1
        removed_after[last_of_shot[has_click]] = cum_removed[:, -1][has_click]
    bright_hist = np.bincount(
        removed_after - removed_before + 1, minlength=T + 1,
    ).astype(np.int64)

    return dict(
        click_times=click_times,
        shot_offsets=shot_offsets,
        compressed_lengths=compressed_lengths,
        jump_counts=jump_counts,
        bright_hist=bright_hist,
    )


def run_counting_histogram(
    click_times: np.ndarray,
    shot_offsets: np.ndarray,
    compressed_lengths: np.ndarray,
    *,
    n_max: int,
    t_c: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of cumulative click count ``n`` vs compressed time ``t`` for one run.

    Parameters
    ----------
    click_times, shot_offsets, compressed_lengths : as returned by
        :func:`compress_run`.
    n_max : int
        Largest click count to resolve (counts are clipped to ``n_max``).
    t_c : int
        Number of compressed time bins (>= max compressed length in the rate).

    Returns
    -------
    H : int32, shape (t_c, n_max+1)
        ``H[t, n]`` = number of shots **alive at t** with cumulative click
        count ``n`` by compressed time ``t``.  A shot is alive at ``t`` iff
        ``t < compressed_length``.
    alive : int32, shape (t_c,)
        Number of shots alive at each compressed time.
    """
    n_shots = len(compressed_lengths)
    counts = np.diff(shot_offsets)
    rows = np.repeat(np.arange(n_shots), counts)

    inc = np.zeros((n_shots, t_c), dtype=np.int32)
    if click_times.size:
        np.add.at(inc, (rows, click_times), 1)
    cum_n = np.cumsum(inc, axis=1)                  # n(t) per shot, (N, t_c)

    t_idx = np.arange(t_c)
    alive_mask = t_idx[None, :] < compressed_lengths[:, None]

    n_vals = np.minimum(cum_n[alive_mask], n_max)
    tt = np.broadcast_to(t_idx, (n_shots, t_c))[alive_mask]
    combo = tt.astype(np.int64) * (n_max + 1) + n_vals
    H = np.bincount(combo, minlength=t_c * (n_max + 1)).astype(np.int32)
    H = H.reshape(t_c, n_max + 1)
    alive = alive_mask.sum(axis=0).astype(np.int32)
    return H, alive


def fit_dead_time_exponential(
    bright_hist: np.ndarray,
    dt_us: float,
    *,
    l_min: int | None = None,
) -> dict:
    """Maximum-likelihood exponential fit to a bright-run-length histogram.

    The dead time ``L`` is an integer number of raw steps, so the discrete
    counterpart of ``p(t) ∝ e^{-t/τ}`` is the geometric law

        ``P(L) = (1 − q) q^{L − L_min}``,  ``q = e^{−dt/τ}``,  ``L ≥ L_min``,

    whose MLE is fixed by the sample mean alone — exact, with no binning choice
    and no optimiser to converge.  ``L_min`` defaults to the shortest run that
    actually occurs (1 for a raw bright run).

    Parameters
    ----------
    bright_hist : 1-D counts indexed by bright-run length in steps.
    dt_us : raw step in µs.
    l_min : optional explicit support lower bound, in steps.  Pass the pooled
        value when fitting individual runs so their fits stay comparable.

    Returns
    -------
    dict with ``mean_us`` (empirical mean dead time), ``mean_steps``,
    ``tau_us`` (fitted decay constant), ``rate_per_us`` (``1/τ``), ``l_min``,
    ``n_runs_counted`` (number of bright runs in the sample) and ``pmf_fit``
    (fitted probabilities on the same bins as ``bright_hist``).
    """
    h = np.asarray(bright_hist, dtype=np.float64).ravel()
    lengths = np.arange(h.size, dtype=np.float64)
    n = float(h.sum())
    nan = float("nan")
    if n <= 0:
        return dict(mean_us=nan, mean_steps=nan, tau_us=nan, rate_per_us=nan,
                    l_min=0, n_runs_counted=0, pmf_fit=np.zeros_like(h))

    mean_l = float(h @ lengths / n)
    if l_min is None:
        l_min = int(np.nonzero(h)[0][0])
    l_min = int(l_min)

    # Geometric MLE: E[L] = L_min + q/(1-q)  ->  q = s/(1+s), s = E[L] - L_min.
    s = max(mean_l - l_min, 0.0)
    q = s / (1.0 + s)
    tau_us = -dt_us / np.log(q) if 0.0 < q < 1.0 else nan

    pmf_fit = np.zeros_like(h)
    if 0.0 < q < 1.0:
        pmf_fit[l_min:] = (1.0 - q) * q ** (lengths[l_min:] - l_min)
    return dict(
        mean_us=mean_l * dt_us,
        mean_steps=mean_l,
        tau_us=float(tau_us),
        rate_per_us=float(1.0 / tau_us) if np.isfinite(tau_us) else nan,
        l_min=l_min,
        n_runs_counted=int(n),
        pmf_fit=pmf_fit,
    )


def recompress_min_dark(
    click_times: np.ndarray,
    compressed_length: int,
    *,
    min_dark: int = 2,
) -> tuple[np.ndarray, int]:
    """Re-derive clicks under a stricter rule, from compressed click times only.

    Default compression treats every ``1→0`` as a click.  A common refinement
    is to require a dark gap of at least ``min_dark`` zeros before accepting a
    click — i.e. a single isolated 0 between two bright runs (the ``101``
    pattern) is a **false negative**, so the two bright runs are really one
    click.  This merge is exact from click times because every bright run is
    width 1, so the dark gap before click ``j`` is ``click[j] − click[j-1] − 1``.

    Parameters
    ----------
    click_times : 1-D int array (one shot's compressed click times, sorted).
    compressed_length : int
        That shot's compressed length ``Lc``.
    min_dark : int
        Minimum dark-run length (in compressed steps) for a click to survive a
        merge.  ``min_dark=1`` is a no-op (the default ``1→0`` rule);
        ``min_dark=2`` merges single-0 ``101`` false negatives.

    Returns
    -------
    new_click_times : int32 1-D
    new_compressed_length : int
        Length after removing the merged bright steps and their short dark gaps.
    """
    c = np.asarray(click_times, dtype=np.int64).ravel()
    if c.size == 0 or min_dark <= 1:
        return c.astype(np.int32), int(compressed_length)

    kept = [int(c[0])]
    removed_total = 0           # compressed steps removed so far (shift)
    for j in range(1, c.size):
        gap = int(c[j] - c[j - 1] - 1)           # dark-run length before click j
        if gap < min_dark:
            # false negative: drop click j; remove its bright step (1) + the
            # short dark gap from the axis → later clicks shift left.
            removed_total += gap + 1
        else:
            kept.append(int(c[j] - removed_total))
    new_len = int(compressed_length - removed_total)
    return np.asarray(kept, dtype=np.int32), new_len


def _apply_min_dark_to_run(comp: dict, min_dark: int) -> dict:
    """Re-derive a whole run's per-shot record under the stricter ``min_dark`` rule.

    Applies :func:`recompress_min_dark` to every shot's compressed click times and
    rebuilds the run's ``click_times`` / ``shot_offsets`` / ``compressed_lengths`` /
    ``jump_counts``.  Used by the builder when ``min_dark > 1`` (e.g. the ``1->00``
    "2-click" rule) so the counting statistics reflect the stricter rule.
    """
    so = comp["shot_offsets"]
    ct = comp["click_times"]
    cl = comp["compressed_lengths"]
    n_shots = len(cl)

    new_lists: list[np.ndarray] = []
    new_lengths = np.empty(n_shots, dtype=np.int32)
    new_counts = np.empty(n_shots, dtype=np.int32)
    for i in range(n_shots):
        c = ct[so[i]:so[i + 1]]
        nc, nl = recompress_min_dark(c, int(cl[i]), min_dark=min_dark)
        new_lists.append(nc)
        new_lengths[i] = nl
        new_counts[i] = nc.size

    new_offsets = np.zeros(n_shots + 1, dtype=np.int64)
    np.cumsum(new_counts, out=new_offsets[1:])
    new_click_times = (
        np.concatenate(new_lists).astype(np.int32) if new_lists
        else np.zeros(0, dtype=np.int32)
    )
    return dict(
        click_times=new_click_times,
        shot_offsets=new_offsets,
        compressed_lengths=new_lengths,
        jump_counts=new_counts,
    )


# ===========================================================================
# Metadata helpers
# ===========================================================================

def _run_number(fp: Path) -> int:
    m = _RUN_RE.search(fp.name)
    return int(m.group(1)) if m else -1


def _compressed_latest_key(fp: Path) -> tuple[str, float]:
    """Sort key for newest compressed output, independent of click-rule tag."""
    m = _COMPRESSED_TS_RE.search(fp.name)
    timestamp = m.group(1) if m else ""
    try:
        mtime = os.stat(win_long_path(fp)).st_mtime
    except OSError:
        mtime = 0.0
    return timestamp, mtime


def _timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_optimal_threshold(jd: dict) -> dict:
    otd = jd.get("data", {}).get("optimal_threshold_data", {}) or {}
    return dict(
        alpha=otd.get("alpha"),
        beta=otd.get("beta"),
        snr=otd.get("snr"),
        optimal_threshold_v=otd.get("optimal_threshold_v"),
    )


def _find_main_json(main_folder: Path, rate_khz: int) -> Path | None:
    """The single combined JSON for ``rate_khz`` in the main folder (not a run)."""
    cands = [
        p for p in main_folder.glob("*.json")
        if _RUN_RE.search(p.name) is None and f"OBG0_{rate_khz}_kHz" in p.name
    ]
    return cands[0] if cands else None


# ===========================================================================
# Builder
# ===========================================================================

class CompressedClickBuilder:
    """Stream the raw ``individual_runs/`` NPZs into compact per-rate NPZ files.

    One run's ``assigned_states`` (~22 MB) is held in memory at a time.  Output
    files are written to ``<main_folder>/<out_subdir>/`` with timestamped names
    that never overwrite earlier results.  Generic over scan number — pass any
    scan's ``individual_runs`` folder.
    """

    def __init__(
        self,
        *,
        out_subdir: str = DEFAULT_OUT_SUBDIR,
        n_max_cap: int | None = None,
        min_dark: int = 1,
    ) -> None:
        """``min_dark`` controls the click rule (the "extra compression"):

        * ``1`` (default) — every bright run is a click (``1->0``).
        * ``2`` — an isolated single-0 gap (``101``) is a false negative, so the
          two surrounding bright runs merge into one click (``1->00`` rule).
          Applied via :func:`recompress_min_dark` before computing statistics.
        """
        self.out_subdir = out_subdir
        self.n_max_cap = n_max_cap
        self.min_dark = int(min_dark)

    # ------------------------------------------------------------------
    def build(
        self,
        individual_runs_folder: str | Path,
        *,
        rates: list[int] | None = None,
        max_runs: int | None = None,
        out_dir: str | Path | None = None,
        verbose: bool = True,
    ) -> list[Path]:
        """Build one compressed NPZ per rate.

        Parameters
        ----------
        individual_runs_folder : path to the ``individual_runs`` directory.
        rates : optional list of integer rate tags (kHz) to process; default all.
        max_runs : optional cap on runs per rate (for quick tests).
        out_dir : optional explicit output directory.  Default writes to
            ``<main_folder>/<out_subdir>/`` (the parent of ``individual_runs``).
            Override e.g. to write test outputs outside the data folder.
        verbose : print progress.

        Returns
        -------
        list[Path] : the written NPZ paths (one per processed rate).
        """
        ir = Path(individual_runs_folder)
        if ir.name != "individual_runs":
            # Be forgiving: allow passing the scan folder directly.
            cand = ir / "individual_runs"
            if cand.is_dir():
                ir = cand
        if not ir.is_dir():
            raise FileNotFoundError(f"individual_runs folder not found: {ir}")

        main_folder = ir.parent
        out_dir = Path(out_dir) if out_dir is not None else main_folder / self.out_subdir
        os.makedirs(win_long_path(out_dir), exist_ok=True)

        scan_number = _scan_number_from_folder(main_folder)

        # Group run NPZ files by rate tag.
        by_rate: dict[int, list[Path]] = {}
        for npz in sorted(ir.glob("*.npz")):
            try:
                rate = _khz_from_name(npz)
            except ValueError:
                continue
            by_rate.setdefault(rate, []).append(npz)

        target_rates = sorted(by_rate) if rates is None else sorted(rates)
        written: list[Path] = []
        for rate in target_rates:
            run_files = sorted(by_rate.get(rate, []), key=_run_number)
            if max_runs is not None:
                run_files = run_files[:max_runs]
            if not run_files:
                if verbose:
                    print(f"[build] rate {rate} kHz: no runs found, skipping.")
                continue
            path = self._build_rate(
                rate, run_files, main_folder, out_dir, scan_number, verbose,
            )
            written.append(path)
        return written

    # ------------------------------------------------------------------
    def _build_rate(
        self,
        rate: int,
        run_files: list[Path],
        main_folder: Path,
        out_dir: Path,
        scan_number: int | None,
        verbose: bool,
    ) -> Path:
        # ---- Phase 1: stream runs, compress (one assigned_states in RAM) -----
        per_run: list[dict] = []
        bright_hists: list[np.ndarray] = []
        alpha_pr: list[float] = []
        beta_pr: list[float] = []
        snr_pr: list[float] = []
        time_us = None
        lam = omega_dg = omega_bg = None
        navg = nrep = None
        run_numbers: list[int] = []
        source_files: list[str] = []

        for npz_path in run_files:
            json_path = npz_path.with_suffix(".json")
            jd = _load_json(json_path)
            if time_us is None:
                time_us = _read_time_us(jd)
                params = jd.get("parameters", {})
                lam = _read_lambda_from_json(jd)
                omega_dg = params.get("Omega_DG")
                omega_bg = params.get("Omega_BG")
                navg = int(params.get("navg_repeat_experiment", 0)) or None
                nrep = int(params.get("num_experiment_repeats", 0)) or None
            otd = _read_optimal_threshold(jd)
            alpha_pr.append(_nan(otd["alpha"]))
            beta_pr.append(_nan(otd["beta"]))
            snr_pr.append(_nan(otd["snr"]))

            z = _load_npz_retry(npz_path)
            a = z["assigned_states"]
            T = len(time_us)
            N = a.size // T
            if N * T != a.size:
                raise ValueError(
                    f"{npz_path.name}: assigned_states size {a.size} not "
                    f"divisible by T={T}."
                )
            comp = compress_run(a.reshape(N, T))
            del a, z
            # Pop before any min_dark re-derivation: the dead-time histogram
            # describes the *raw* bright runs and cannot be rebuilt later.
            bright_hists.append(comp.pop("bright_hist"))
            if self.min_dark > 1:
                comp = _apply_min_dark_to_run(comp, self.min_dark)
            per_run.append(comp)
            run_numbers.append(_run_number(npz_path))
            source_files.append(npz_path.name)
            if verbose:
                jc = comp["jump_counts"]
                print(f"  [rate {rate} kHz] {npz_path.name}: "
                      f"N={N} mean_clicks={jc.mean():.3f}")

        # ---- Phase 2: assemble counting statistics + Fourier (small arrays) --
        n_runs = len(per_run)
        n_max = max(int(c["jump_counts"].max(initial=0)) for c in per_run)
        if self.n_max_cap is not None:
            n_max = min(n_max, self.n_max_cap)
        t_c = max(int(c["compressed_lengths"].max(initial=1)) for c in per_run)

        counts_hist = np.zeros((n_runs, t_c, n_max + 1), dtype=np.int32)
        alive = np.zeros((n_runs, t_c), dtype=np.int32)
        for r, comp in enumerate(per_run):
            H, al = run_counting_histogram(
                comp["click_times"], comp["shot_offsets"],
                comp["compressed_lengths"], n_max=n_max, t_c=t_c,
            )
            counts_hist[r] = H
            alive[r] = al

        # Concatenate the per-shot raw record across runs.
        click_times = np.concatenate([c["click_times"] for c in per_run])
        per_run_shot_counts = [len(c["compressed_lengths"]) for c in per_run]
        n_shots = int(sum(per_run_shot_counts))
        shot_offsets = np.zeros(n_shots + 1, dtype=np.int64)
        cursor = 0
        base = 0
        for comp in per_run:
            so = comp["shot_offsets"]
            ns = len(so) - 1
            shot_offsets[cursor + 1: cursor + ns + 1] = so[1:] + base
            cursor += ns
            base += so[-1]
        run_shot_offsets = np.concatenate(
            [[0], np.cumsum(per_run_shot_counts)]
        ).astype(np.int64)
        jump_counts = np.concatenate([c["jump_counts"] for c in per_run])
        compressed_lengths = np.concatenate(
            [c["compressed_lengths"] for c in per_run]
        )

        # Stored P(n,t) over surviving shots.  For k-space analysis the loader
        # uses this survivor-normalized signal and truncates the low-alive tail.
        total_H = counts_hist.sum(axis=0).astype(np.float64)        # (t_c, n_max+1)
        shots_per_time = alive.sum(axis=0).astype(np.int32)         # (t_c,)
        runs_per_time = (alive > 0).sum(axis=0).astype(np.int32)    # (t_c,)
        with np.errstate(invalid="ignore", divide="ignore"):
            pnt = np.where(
                shots_per_time[:, None] > 0,
                total_H / shots_per_time[:, None],
                0.0,
            ).astype(np.float32)

        # Fourier over the counting field k, on the existing 101-grid.
        from mpm_analysis.kspace.loader import compute_pkt_from_pnt
        k_array = np.linspace(0.0, 2.0 * np.pi, DEFAULT_K_GRID_N)
        p_kt, _ = compute_pkt_from_pnt(pnt.astype(np.float64), k_array=k_array)

        dt_us = float(time_us[1] - time_us[0])
        t_compressed_us = np.arange(t_c, dtype=np.float64) * dt_us
        bright_hist = np.stack(bright_hists).astype(np.int64)   # (n_runs, T+1)

        # Authoritative alpha/beta from the main combined JSON; else per-run mean.
        alpha = beta = snr = thr = None
        alpha_beta_source = "main_combined_json"
        main_json = _find_main_json(main_folder, rate)
        if main_json is not None:
            mj = _load_json(main_json)
            mt = _read_optimal_threshold(mj)
            alpha, beta, snr, thr = (
                mt["alpha"], mt["beta"], mt["snr"], mt["optimal_threshold_v"]
            )
        if alpha is None:
            alpha = float(np.nanmean(alpha_pr))
            beta = float(np.nanmean(beta_pr))
            snr = float(np.nanmean(snr_pr))
            alpha_beta_source = "per_run_mean_fallback"
            print(f"  [rate {rate} kHz] no main JSON; using per-run mean alpha/beta.")

        if self.min_dark > 1:
            click_rule = f"1->{'0' * self.min_dark} (min_dark={self.min_dark})"
            rule_tag = f"_mindark{self.min_dark}"
        else:
            click_rule = "1->0 (every bright run)"
            rule_tag = ""

        metadata = dict(
            scan_number=scan_number,
            rate_khz=rate,
            run_numbers=run_numbers,
            source_files=source_files,
            alpha_beta_source=alpha_beta_source,
            main_json=(main_json.name if main_json is not None else None),
            min_dark=self.min_dark,
            click_rule=click_rule,
            built_utc=_timestamp(),
        )

        stem = f"click_record_compressed__OBG0_{rate}_kHz{rule_tag}_{_timestamp()}"
        out_path = _unique_path(out_dir / f"{stem}.npz")

        np.savez_compressed(
            win_long_path(out_path),
            schema_version=SCHEMA_VERSION,
            rate_khz=rate,
            lambda_val=float(lam),
            alpha=_nan(alpha), beta=_nan(beta), snr=_nan(snr),
            optimal_threshold_v=_nan(thr),
            alpha_per_run=np.asarray(alpha_pr, dtype=np.float64),
            beta_per_run=np.asarray(beta_pr, dtype=np.float64),
            omega_dg=_nan(omega_dg), omega_bg=_nan(omega_bg),
            dt_us=dt_us, n_runs=n_runs, n_shots_per_run=(n_shots // n_runs),
            n_shots=n_shots, T_raw=len(time_us), n_max=n_max,
            min_dark=self.min_dark,
            click_rule=click_rule,
            time_raw_us=np.asarray(time_us, dtype=np.float64),
            t_compressed_us=t_compressed_us,
            click_times=click_times.astype(np.int32),
            shot_offsets=shot_offsets,
            run_shot_offsets=run_shot_offsets,
            jump_counts=jump_counts.astype(np.int32),
            compressed_lengths=compressed_lengths.astype(np.int32),
            counts_hist=counts_hist,
            alive=alive,
            pnt=pnt,
            shots_per_time=shots_per_time,
            runs_per_time=runs_per_time,
            bright_hist=bright_hist,
            fourier_k_array=k_array,
            pkt_real=p_kt.real.astype(np.float32),
            pkt_imag=p_kt.imag.astype(np.float32),
            metadata_json=np.bytes_(json.dumps(metadata).encode()),
        )
        if verbose:
            mb = os.stat(win_long_path(out_path)).st_size / 1e6
            dead = fit_dead_time_exponential(bright_hist.sum(axis=0), dt_us)
            print(f"[build] rate {rate} kHz -> {out_path.name} "
                  f"({mb:.1f} MB, n_runs={n_runs}, T_c={t_c}, n_max={n_max})")
            print(f"         lambda = {float(lam):.4f}   dead time: "
                  f"mean = {dead['mean_us']:.4f} us "
                  f"({dead['mean_steps']:.3f} steps), "
                  f"tau_fit = {dead['tau_us']:.4f} us, "
                  f"N = {dead['n_runs_counted']:,} bright runs")
        return out_path


# ===========================================================================
# Loader
# ===========================================================================

@dataclass
class CompressedClickRecord:
    """One rate/λ slice loaded from a compressed-click NPZ.

    Most arrays are exposed verbatim; a few convenience properties derive the
    quantities used downstream.  See module docstring for the full schema.
    """
    lambda_val: float
    rate_khz: int
    t: np.ndarray                       # compressed time axis, µs
    time_raw_us: np.ndarray
    dt_us: float
    pnt: np.ndarray                     # (T_c, n_max+1) P(n,t)
    counts_hist: np.ndarray             # (n_runs, T_c, n_max+1)
    alive: np.ndarray                   # (n_runs, T_c)
    shots_per_time: np.ndarray          # (T_c,)
    runs_per_time: np.ndarray           # (T_c,)
    jump_counts: np.ndarray             # (n_shots,)
    compressed_lengths: np.ndarray      # (n_shots,)
    click_times: np.ndarray             # (Σ clicks,)
    shot_offsets: np.ndarray            # (n_shots+1,)
    run_shot_offsets: np.ndarray        # (n_runs+1,)
    fourier_k_array: np.ndarray         # (101,)
    pkt_real: np.ndarray                # (T_c, 101)
    pkt_imag: np.ndarray                # (T_c, 101)
    # Dead time: bin L = number of raw bright runs of exactly L steps, per run.
    # ``None`` for NPZs built before this was recorded — it cannot be recovered
    # from them (the collapsed 1s are gone), so those need a rebuild.
    bright_hist: np.ndarray | None = None       # (n_runs, T_raw+1)
    alpha: float = float("nan")
    beta: float = float("nan")
    snr: float = float("nan")
    omega_dg: float = float("nan")
    omega_bg: float = float("nan")
    n_runs: int = 0
    n_max: int = 0
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Dead time — the length of the bright runs the compression collapsed
    # ------------------------------------------------------------------

    @property
    def has_dead_time(self) -> bool:
        """Whether this NPZ carries the dead-time histogram (needs a rebuild if not)."""
        return self.bright_hist is not None

    @property
    def dead_time_steps(self) -> np.ndarray:
        """Bright-run lengths in raw steps — the histogram's bin index."""
        return np.arange(self.bright_hist.shape[1], dtype=np.int64)

    @property
    def dead_time_us(self) -> np.ndarray:
        """Bright-run lengths in µs — the dead-time axis."""
        return self.dead_time_steps * self.dt_us

    def dead_time_counts(self) -> np.ndarray:
        """Dead-time histogram pooled over all runs."""
        return self.bright_hist.sum(axis=0).astype(np.int64)

    def dead_time_pmf(self) -> np.ndarray:
        """Normalised dead-time distribution over bright runs (sums to 1)."""
        h = self.dead_time_counts().astype(np.float64)
        total = h.sum()
        return h / total if total > 0 else h

    def dead_time_summary(self) -> dict:
        """Mean dead time and exponential fit, with run-to-run error bars.

        Uncertainties are the standard error over the ``n_runs`` runs — the same
        resampling unit used for the counting statistics.
        """
        pooled = fit_dead_time_exponential(self.dead_time_counts(), self.dt_us)
        # Fit each run on the pooled support so the per-run values are comparable.
        per_run = [
            fit_dead_time_exponential(h, self.dt_us, l_min=pooled["l_min"])
            for h in self.bright_hist
        ]

        def _sem(key: str) -> float:
            v = np.array([p[key] for p in per_run], dtype=np.float64)
            v = v[np.isfinite(v)]
            return float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0

        return dict(
            lambda_val=self.lambda_val, rate_khz=self.rate_khz,
            mean_us=pooled["mean_us"], mean_us_err=_sem("mean_us"),
            mean_steps=pooled["mean_steps"],
            tau_us=pooled["tau_us"], tau_us_err=_sem("tau_us"),
            l_min=pooled["l_min"], n_bright_runs=pooled["n_runs_counted"],
            dt_us=self.dt_us, n_runs=len(per_run),
        )

    @property
    def signal_pkt(self) -> np.ndarray:
        """Stored survival-conditioned P(k,t), shape (n_k, n_time)."""
        return (self.pkt_real + 1j * self.pkt_imag).T

    @property
    def n_array(self) -> np.ndarray:
        return np.arange(self.pnt.shape[1], dtype=np.float64)

    def mean_clicks(self) -> np.ndarray:
        """⟨n⟩(t) over alive shots, from the aggregated P(n,t)."""
        return self.pnt @ self.n_array

    def var_clicks(self) -> np.ndarray:
        """Var(n)(t) over alive shots."""
        n = self.n_array
        m = self.pnt @ n
        m2 = self.pnt @ (n * n)
        return m2 - m * m

    def pnt_run(self, r: int) -> np.ndarray:
        """P(n,t) for a single run ``r``, over that run's alive shots.

        Shape ``(T_c, n_max+1)``.  Enables run-level bootstrap of the poles
        (one full pole analysis per run → spread across runs = error bars).
        """
        H = self.counts_hist[r].astype(np.float64)
        al = self.alive[r].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(al[:, None] > 0, H / al[:, None], 0.0)


class CompressedClickLoader:
    """Load compressed-click NPZs from an analysis folder into records."""

    def load(
        self,
        analysis_folder: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        latest_only: bool = True,
    ) -> list[CompressedClickRecord]:
        """Load every compressed NPZ in ``analysis_folder``.

        Parameters
        ----------
        analysis_folder : the ``compressed_click_analysis`` directory (or any
            folder of ``click_record_compressed__*.npz``).
        lambda_range : keep only records with λ in [lo, hi].
        latest_only : if multiple timestamped files exist for the same rate,
            keep only the newest (filenames sort by timestamp).
        """
        folder = Path(analysis_folder)
        files = sorted(folder.glob("click_record_compressed__*.npz"))
        if latest_only:
            newest: dict[int, Path] = {}
            newest_key: dict[int, tuple[str, float]] = {}
            for fp in files:
                try:
                    rate = _khz_from_name(fp)
                except ValueError:
                    continue
                key = _compressed_latest_key(fp)
                if rate not in newest or key > newest_key[rate]:
                    newest[rate] = fp
                    newest_key[rate] = key
            files = [newest[r] for r in sorted(newest)]

        records: list[CompressedClickRecord] = []
        for fp in files:
            rec = self._load_file(fp)
            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= rec.lambda_val <= hi):
                    continue
            records.append(rec)
        records.sort(key=lambda r: r.lambda_val)
        return records

    @staticmethod
    def _load_file(fp: Path) -> CompressedClickRecord:
        z = np.load(win_long_path(fp), allow_pickle=False)
        meta_raw = z["metadata_json"]
        metadata = json.loads(bytes(meta_raw).decode()) if meta_raw.size else {}
        return CompressedClickRecord(
            lambda_val=float(z["lambda_val"]),
            rate_khz=int(z["rate_khz"]),
            t=z["t_compressed_us"],
            time_raw_us=z["time_raw_us"],
            dt_us=float(z["dt_us"]),
            pnt=z["pnt"],
            counts_hist=z["counts_hist"],
            alive=z["alive"],
            shots_per_time=z["shots_per_time"],
            runs_per_time=z["runs_per_time"],
            jump_counts=z["jump_counts"],
            compressed_lengths=z["compressed_lengths"],
            click_times=z["click_times"],
            shot_offsets=z["shot_offsets"],
            run_shot_offsets=z["run_shot_offsets"],
            fourier_k_array=z["fourier_k_array"],
            pkt_real=z["pkt_real"],
            pkt_imag=z["pkt_imag"],
            bright_hist=(z["bright_hist"] if "bright_hist" in z.files else None),
            alpha=float(z["alpha"]), beta=float(z["beta"]), snr=float(z["snr"]),
            omega_dg=float(z["omega_dg"]), omega_bg=float(z["omega_bg"]),
            n_runs=int(z["n_runs"]), n_max=int(z["n_max"]),
            source_file=fp.name, metadata=metadata,
        )


# ===========================================================================
# Re-compression from existing compressed records
# ===========================================================================

def _npz_metadata(z: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in z.files:
        return {}
    meta_raw = z["metadata_json"]
    if not meta_raw.size:
        return {}
    try:
        return json.loads(bytes(meta_raw).decode())
    except Exception:
        return {}


def _npz_min_dark(fp: Path) -> int:
    with np.load(win_long_path(fp), allow_pickle=False) as z:
        if "min_dark" in z.files:
            return int(np.asarray(z["min_dark"]).item())
        return int(_npz_metadata(z).get("min_dark", 1))


def recompress_file_min_dark(
    source_npz: str | Path,
    *,
    min_dark: int = 2,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """Create a stricter ``min_dark`` compressed NPZ from an existing one.

    This uses only the stored per-shot ``click_times`` and
    ``compressed_lengths``. It does not read the original ``individual_runs``
    raw threshold traces.
    """
    source_npz = Path(source_npz)
    out_dir = Path(out_dir) if out_dir is not None else source_npz.parent
    os.makedirs(win_long_path(out_dir), exist_ok=True)

    with np.load(win_long_path(source_npz), allow_pickle=False) as z:
        source_min_dark = int(z["min_dark"]) if "min_dark" in z.files else 1
        if min_dark <= source_min_dark:
            raise ValueError(
                f"target min_dark={min_dark} must be stricter than "
                f"source min_dark={source_min_dark} for {source_npz.name}."
            )

        all_click_times = z["click_times"]
        all_shot_offsets = z["shot_offsets"]
        all_lengths = z["compressed_lengths"]
        all_counts = z["jump_counts"]
        run_shot_offsets_src = z["run_shot_offsets"]

        per_run: list[dict] = []
        for r in range(len(run_shot_offsets_src) - 1):
            s0 = int(run_shot_offsets_src[r])
            s1 = int(run_shot_offsets_src[r + 1])
            c0 = int(all_shot_offsets[s0])
            c1 = int(all_shot_offsets[s1])
            comp = dict(
                click_times=all_click_times[c0:c1],
                shot_offsets=(all_shot_offsets[s0:s1 + 1] - c0).astype(np.int64),
                compressed_lengths=all_lengths[s0:s1],
                jump_counts=all_counts[s0:s1],
            )
            per_run.append(_apply_min_dark_to_run(comp, min_dark))

        n_runs = len(per_run)
        n_max = max(int(c["jump_counts"].max(initial=0)) for c in per_run)
        t_c = max(int(c["compressed_lengths"].max(initial=1)) for c in per_run)

        counts_hist = np.zeros((n_runs, t_c, n_max + 1), dtype=np.int32)
        alive = np.zeros((n_runs, t_c), dtype=np.int32)
        for r, comp in enumerate(per_run):
            H, al = run_counting_histogram(
                comp["click_times"], comp["shot_offsets"],
                comp["compressed_lengths"], n_max=n_max, t_c=t_c,
            )
            counts_hist[r] = H
            alive[r] = al

        click_times = (
            np.concatenate([c["click_times"] for c in per_run]).astype(np.int32)
            if per_run else np.zeros(0, dtype=np.int32)
        )
        per_run_shot_counts = [len(c["compressed_lengths"]) for c in per_run]
        n_shots = int(sum(per_run_shot_counts))
        shot_offsets = np.zeros(n_shots + 1, dtype=np.int64)
        cursor = 0
        base = 0
        for comp in per_run:
            so = comp["shot_offsets"]
            ns = len(so) - 1
            shot_offsets[cursor + 1: cursor + ns + 1] = so[1:] + base
            cursor += ns
            base += int(so[-1])
        run_shot_offsets = np.concatenate(
            [[0], np.cumsum(per_run_shot_counts)]
        ).astype(np.int64)
        jump_counts = np.concatenate([c["jump_counts"] for c in per_run])
        compressed_lengths = np.concatenate(
            [c["compressed_lengths"] for c in per_run]
        )

        total_H = counts_hist.sum(axis=0).astype(np.float64)
        shots_per_time = alive.sum(axis=0).astype(np.int32)
        runs_per_time = (alive > 0).sum(axis=0).astype(np.int32)
        with np.errstate(invalid="ignore", divide="ignore"):
            pnt = np.where(
                shots_per_time[:, None] > 0,
                total_H / shots_per_time[:, None],
                0.0,
            ).astype(np.float32)

        from mpm_analysis.kspace.loader import compute_pkt_from_pnt
        k_array = np.asarray(
            z["fourier_k_array"] if "fourier_k_array" in z.files
            else np.linspace(0.0, 2.0 * np.pi, DEFAULT_K_GRID_N),
            dtype=np.float64,
        )
        p_kt, _ = compute_pkt_from_pnt(pnt.astype(np.float64), k_array=k_array)

        rate = int(z["rate_khz"])
        dt_us = float(z["dt_us"])
        time_raw_us = np.asarray(z["time_raw_us"], dtype=np.float64)
        t_compressed_us = np.arange(t_c, dtype=np.float64) * dt_us
        click_rule = f"1->{'0' * min_dark} (min_dark={min_dark})"
        metadata = _npz_metadata(z)
        metadata.update(
            source_compressed_file=source_npz.name,
            source_min_dark=source_min_dark,
            min_dark=min_dark,
            click_rule=click_rule,
            recompressed_utc=_timestamp(),
        )

        stem = f"click_record_compressed__OBG0_{rate}_kHz_mindark{min_dark}_{_timestamp()}"
        out_path = _unique_path(out_dir / f"{stem}.npz")

        np.savez_compressed(
            win_long_path(out_path),
            schema_version=SCHEMA_VERSION,
            rate_khz=rate,
            lambda_val=float(z["lambda_val"]),
            alpha=float(z["alpha"]), beta=float(z["beta"]), snr=float(z["snr"]),
            optimal_threshold_v=(
                float(z["optimal_threshold_v"])
                if "optimal_threshold_v" in z.files else float("nan")
            ),
            alpha_per_run=(
                z["alpha_per_run"] if "alpha_per_run" in z.files
                else np.full(n_runs, float("nan"))
            ),
            beta_per_run=(
                z["beta_per_run"] if "beta_per_run" in z.files
                else np.full(n_runs, float("nan"))
            ),
            omega_dg=float(z["omega_dg"]), omega_bg=float(z["omega_bg"]),
            dt_us=dt_us, n_runs=n_runs,
            n_shots_per_run=(n_shots // n_runs if n_runs else 0),
            n_shots=n_shots, T_raw=len(time_raw_us), n_max=n_max,
            min_dark=min_dark,
            click_rule=click_rule,
            time_raw_us=time_raw_us,
            t_compressed_us=t_compressed_us,
            click_times=click_times,
            shot_offsets=shot_offsets,
            run_shot_offsets=run_shot_offsets,
            jump_counts=jump_counts.astype(np.int32),
            compressed_lengths=compressed_lengths.astype(np.int32),
            counts_hist=counts_hist,
            alive=alive,
            pnt=pnt,
            shots_per_time=shots_per_time,
            runs_per_time=runs_per_time,
            # Carried through verbatim: it describes the raw bright runs, which
            # the min_dark rule does not change (it only merges *clicks*).
            **({"bright_hist": z["bright_hist"]} if "bright_hist" in z.files else {}),
            fourier_k_array=k_array,
            pkt_real=p_kt.real.astype(np.float32),
            pkt_imag=p_kt.imag.astype(np.float32),
            metadata_json=np.bytes_(json.dumps(metadata).encode()),
        )

    if verbose:
        mb = os.stat(win_long_path(out_path)).st_size / 1e6
        print(f"[recompress] {source_npz.name} -> {out_path.name} ({mb:.1f} MB)")
    return out_path


def recompress_folder_min_dark(
    analysis_folder: str | Path,
    *,
    min_dark: int = 2,
    source_min_dark: int = 1,
    rates: list[int] | None = None,
    out_dir: str | Path | None = None,
    latest_only: bool = True,
    verbose: bool = True,
) -> list[Path]:
    """Recompress a folder of existing compressed NPZs to stricter ``min_dark``."""
    folder = Path(analysis_folder)
    if out_dir is None:
        out_dir = folder.with_name(f"{folder.name}_mindark{min_dark}")
    out_dir = Path(out_dir)

    rate_filter = set(rates) if rates is not None else None
    files_by_rate: dict[int, list[Path]] = {}
    for fp in sorted(folder.glob("click_record_compressed__*.npz")):
        try:
            rate = _khz_from_name(fp)
        except ValueError:
            continue
        if rate_filter is not None and rate not in rate_filter:
            continue
        if _npz_min_dark(fp) != source_min_dark:
            continue
        files_by_rate.setdefault(rate, []).append(fp)

    if latest_only:
        files = [
            max(fps, key=_compressed_latest_key)
            for _, fps in sorted(files_by_rate.items())
        ]
    else:
        files = [fp for _, fps in sorted(files_by_rate.items()) for fp in fps]

    if not files:
        raise RuntimeError(
            f"No source min_dark={source_min_dark} compressed NPZs found in {folder}."
        )

    return [
        recompress_file_min_dark(
            fp, min_dark=min_dark, out_dir=out_dir, verbose=verbose,
        )
        for fp in files
    ]


# ===========================================================================
# small utilities
# ===========================================================================

_SCAN_RE = re.compile(r"_(\d+)$")


def _scan_number_from_folder(folder: Path) -> int | None:
    m = _SCAN_RE.search(folder.name)
    return int(m.group(1)) if m else None


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def _unique_path(path: Path) -> Path:
    """Return ``path`` if free, else append ``_1``, ``_2``… — never overwrite.

    Uses ``win_long_path`` for the existence test so the guarantee holds for
    paths longer than the Windows 260-character limit (where a plain
    ``Path.exists()`` can wrongly report ``False`` and clobber an existing file).
    """
    if not os.path.exists(win_long_path(path)):
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not os.path.exists(win_long_path(cand)):
            return cand
        i += 1
