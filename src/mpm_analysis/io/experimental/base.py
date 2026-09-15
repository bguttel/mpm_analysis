"""
base.py
-------
Abstract base class for all experimental data loaders.

Extensibility model
~~~~~~~~~~~~~~~~~~~
Each concrete loader subclass handles a specific data source layout (e.g.
post-selected tomography folder, ensemble-average folder).  Adding a new
observable type does *not* require a new subclass — call ``add_extractor``
with a key and a function that extracts the signal from the raw JSON dict:

    loader = PostSelectedLoader()
    loader.add_extractor(
        "G_probability",
        lambda jd, t_us: np.asarray(jd["data"]["G_probability"], dtype=float),
    )
    records = loader.load(folder, observables=["survival", "G_probability"])
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np


class HMMTableMismatch(RuntimeError):
    """A lambda-rescaling CSV that describes a different scan than the folder.

    Raised, and deliberately not swallowed by the per-file skip handler in
    :meth:`ExperimentalLoader.load`: a missing row for one file is a gap, but a
    table from the wrong scan makes every lambda wrong, so the run must stop.
    """

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.utils.windows_paths import win_long_path

# Default regex for extracting λ from filenames (kHz variant)
_KHZ_RE = re.compile(r"(\d+)_kHz", re.IGNORECASE)
_OBG0_RE = re.compile(r"OBG0_(\d+)_kHz", re.IGNORECASE)
# Regex to parse trailing integer from folder name (e.g. scan_experiment_results_50 → 50)
_FOLDER_SCAN_RE = re.compile(r"_(\d+)$")


def _khz_from_name(fp: Path) -> int:
    """Return the integer that appears before ``_kHz`` in the filename."""
    m = _OBG0_RE.search(fp.name)
    if m:
        return int(m.group(1))
    m = _KHZ_RE.search(fp.name)
    if not m:
        raise ValueError(f"Cannot find '<N>_kHz' in filename: {fp.name}")
    return int(m.group(1))


def _load_json(path: Path, *, retries: int = 5, delay: float = 3.0) -> dict:
    # Raw run files live in Dropbox as online-only placeholders; the first
    # access can transiently raise FileNotFoundError while the sync client
    # hydrates the file.  Retry with a short pause before giving up.
    for attempt in range(retries):
        try:
            with open(win_long_path(path), "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# Bootstrap-statistics fields that are survival probabilities, and so carry
# the same t=0 normalisation anchor as the survival observables.  The Bloch
# coordinates (X/Y/Z) and the raw shot counts are measurements at t=0 and are
# never touched by ``init_error``.
_INIT_ERROR_STAT_FIELDS = ("survival", "SX", "SZ", "S_pooled")


def _apply_init_error(y: np.ndarray, init_error: float) -> tuple[np.ndarray, bool]:
    """Replace a first sample of exactly ``1`` with ``1 - init_error``.

    Only an exact ``1.0`` is rewritten.  That value is the normalisation
    anchor the experiment writes at t=0 (``p_0_t_array[0]`` is 1 by
    construction in every file of the post-selected scans), not a
    measurement — while the initialisation is imperfect, so the true survival
    at t=0 is ``1 - init_error``.  A first point that was actually measured
    (SX/SZ start around 0.981, the ensemble-average channels far from 1) is
    left exactly as it is, because ``init_error`` has nothing to say about it.

    Returns the (possibly new) array and whether it was rewritten.
    """
    if y.size == 0 or y[0] != 1.0:
        return y, False
    y = y.astype(float, copy=True)
    y[0] = 1.0 - init_error
    return y, True


def _stats_with_init_error(
    stats: dict, init_error: float, hits: dict[str, int]
) -> dict:
    """Copy of ``stats`` with the survival fields' t=0 anchor set to 1 - eps.

    The bootstrap draws from these arrays, so leaving them at 1 would keep
    resampling a point the fit no longer sees at 1 (and a binomial at p=1 has
    zero spread, which understates the error bar at t=0).
    """
    out = dict(stats)
    for field in _INIT_ERROR_STAT_FIELDS:
        v = out.get(field)
        if v is None:
            continue
        v = np.asarray(v, dtype=float)
        v, hit = _apply_init_error(v, init_error)
        if hit:
            out[field] = v
            hits[field] = hits.get(field, 0) + 1
    return out


# Survival channels whose bootstrap attempt count lives in a *scalar* field.
# Splicing one point in from another sub-ensemble makes that count vary with
# time, so the scalar is promoted to a per-point array — which the resampler
# already accepts (it reads the attempts field with np.asarray either way).
_SPLICE_ATTEMPTS_FIELD = {"survival": "N_survival"}


def _splice_point(
    y: np.ndarray, src: np.ndarray, index: int
) -> tuple[np.ndarray, bool]:
    """Copy ``src[index]`` into ``y[index]``, leaving every other point alone.

    Both arrays are the file's *full-length* traces (before any ``t_max`` /
    drop mask), so ``index`` always means the same physical time regardless of
    the window the run happens to fit.  A non-finite source value is refused
    rather than silently poisoning the trace: the conditional curves are NaN
    wherever their sub-ensemble was empty.

    Returns the (possibly new) array and whether the splice happened.
    """
    if y.size <= index or src.size <= index:
        return y, False
    v = float(src[index])
    if not np.isfinite(v):
        return y, False
    y = y.astype(float, copy=True)
    y[index] = v
    return y, True


def _stats_with_splice(
    stats: dict, source_tag: str, index: int, targets: tuple[str, ...],
) -> tuple[dict, bool]:
    """Copy one point of a conditional channel into the survival statistics.

    The bootstrap must resample the spliced point on the *sub-ensemble* that
    actually produced it, not on the full row count — otherwise that point
    keeps the error bar of an ensemble it no longer comes from.  Both the
    fraction and the attempts count are therefore taken from the source
    channel at ``index``, and the scalar attempts field is promoted to a
    per-point array so the rest of the curve keeps its own count.
    """
    src_frac = stats.get(f"S_{source_tag}")
    src_n = stats.get(f"N_attempts_S_{source_tag}")
    if src_frac is None or src_n is None:
        return stats, False

    src_frac = np.asarray(src_frac, dtype=float)
    src_n = np.asarray(src_n, dtype=np.int64)
    out = dict(stats)
    done = False
    for key in targets:
        frac_field = "survival" if key in ("S", "survival") else key
        v = out.get(frac_field)
        n_field = _SPLICE_ATTEMPTS_FIELD.get(frac_field)
        if v is None or n_field is None:
            continue
        v = np.asarray(v, dtype=float)
        v, hit = _splice_point(v, src_frac, index)
        if not hit or src_n.size <= index:
            continue
        n = np.broadcast_to(
            np.asarray(out[n_field], dtype=np.int64), v.shape
        ).copy()
        n[index] = src_n[index]
        out[frac_field] = v
        out[n_field] = n
        done = True
    return out, done


def _get_nested(d: dict, keys: list[str]):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


class ExperimentalLoader(ABC):
    """Base class for experimental data loaders.

    Subclasses implement ``_discover_files`` and ``_extract_lambda_and_time``,
    which handle the folder layout specific to each experiment type.

    The set of observables is kept extensible: the default ``"survival"``
    extractor is registered at init time; call ``add_extractor`` to add more.
    """

    def __init__(self) -> None:
        self._extractors: dict[str, Callable[[dict, np.ndarray], np.ndarray]] = {}
        self._register_default_extractors()

    @abstractmethod
    def _register_default_extractors(self) -> None:
        """Register the default observable extractors."""

    @abstractmethod
    def _discover_files(self, folder: Path) -> list[Path]:
        """Return a list of raw JSON files in ``folder``, sorted by λ."""

    @abstractmethod
    def _extract_lambda_and_time(
        self, jd: dict, fp: Path
    ) -> tuple[float, np.ndarray]:
        """Extract (lambda_val, t_us) from the raw JSON dict."""

    def _bootstrap_stats(self, jd: dict, mask: np.ndarray | None) -> dict | None:
        """Measurement statistics for the data-resampling bootstrap.

        Subclasses return a dict describing the raw counting statistics of
        one file (probabilities + shot counts), with all time-dependent
        arrays sliced by ``mask`` (the ``t_max`` cut).  Attached to every
        record of the file as ``metadata["bootstrap_stats"]``.  Default:
        ``None`` — data bootstrap unavailable for this loader.
        """
        return None

    @property
    def lambda_source(self) -> str:
        """Provenance of the lambda axis, recorded in the saved result.

        ``"json:lambda_zeno_estimation"`` when lambda comes from each file's
        own parameters, or ``"hmm_csv:<name>"`` when a results CSV rescales it
        (lambda = Gamma_up_3_states_hmm / 2*Omega_DG).  Written into the
        result metadata so a saved analysis always states which lambda axis
        it was produced on — the CSV never travels between scan folders
        silently.
        """
        csv = getattr(self, "_csv_path", None)
        return "json:lambda_zeno_estimation" if csv is None \
            else f"hmm_csv:{Path(csv).name}"

    def add_extractor(
        self,
        key: str,
        fn: Callable[[dict, np.ndarray], np.ndarray],
    ) -> None:
        """Register a custom observable extractor.

        Parameters
        ----------
        key:
            Observable key string, e.g. ``"G_probability"``.
        fn:
            Callable ``(raw_json_dict, t_us) -> np.ndarray``.  The function
            receives the full JSON dict and the already-converted time axis.
        """
        self._extractors[key] = fn

    def load(
        self,
        folder: str | Path,
        observables: list[str] | None = None,
        *,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        t_max: float | None = None,
        drop_first_point: bool = False,
        init_error: float | None = None,
        splice_index: int | None = None,
        splice_source: str = "S_prev_noclick",
        splice_targets: tuple[str, ...] = ("S",),
    ) -> list[ObservableRecord]:
        """Load experimental data from a folder.

        Parameters
        ----------
        folder:
            Path to the folder containing per-lambda JSON files.
        observables:
            List of observable keys to extract.  Defaults to ``["survival"]``.
            Each key must have a registered extractor (see ``add_extractor``).
        lambda_range:
            If given, only load files whose λ is within
            ``[lambda_range[0], lambda_range[1]]``.
        max_files:
            If given, load only the first ``max_files`` files (useful for
            quick local testing).
        t_max:
            If given, truncate each signal to times ``t <= t_max`` (in µs).
            Default ``None`` loads the full time axis.
        drop_first_point:
            If ``True``, discard the first sample of the time axis (and of
            every signal and bootstrap-statistics array) of each file, so the
            whole analysis — MPM, refinement, bootstrap, plots — never sees
            it.  Default ``False`` keeps every point.
        init_error:
            Initialisation error ``eps`` in ``[0, 1)``.  When given, a first
            sample of exactly ``1`` — the t=0 normalisation anchor, not a
            measurement — becomes ``1 - eps`` in both the observable and the
            bootstrap statistics, instead of keeping the perfect-preparation
            value.  Measured first points are never rewritten, and what was
            changed is printed.  Default ``None`` leaves the data untouched.
        splice_index:
            If given, replace this ONE time index of every observable in
            ``splice_targets`` with the same index of ``splice_source``,
            leaving all other points untouched.  The index counts on the
            file's full time axis (before ``t_max`` / ``drop_first_point``),
            so it always names the same physical time; index 1 is the first
            measured point of the post-selected survival, index 0 its
            by-construction t=0 anchor.  The bootstrap statistics are spliced
            at the same index and on the source's own per-point class size,
            so the resampled error bar there belongs to the sub-ensemble the
            value came from.  Default ``None`` splices nothing.
        splice_source:
            Observable key to take the spliced point from, e.g.
            ``"S_prev_noclick"``.  Its bootstrap fields are ``S_<tag>`` /
            ``N_attempts_S_<tag>`` as written by ``run_conditional_p0.py``.
        splice_targets:
            Observable keys to splice into.  Default ``("S",)``.

        Returns
        -------
        list[ObservableRecord]
            One record per (file × observable), sorted by λ then observable.
        """
        if observables is None:
            observables = ["survival"]
        if isinstance(observables,str):
            observables = [observables]
        for key in observables:
            if key not in self._extractors:
                raise KeyError(
                    f"No extractor registered for observable '{key}'. "
                    f"Available: {list(self._extractors)}. "
                    "Call add_extractor() to register a new one."
                )

        if init_error is not None:
            if isinstance(init_error, bool):
                raise TypeError(
                    "init_error is the initialisation error itself (e.g. 0.02), "
                    "not an on/off flag; pass a number or None."
                )
            if not (0.0 <= float(init_error) < 1.0):
                raise ValueError(
                    f"init_error must be in [0, 1), got {init_error!r}."
                )
            init_error = float(init_error)
            if drop_first_point:
                print("[init_error] note: drop_first_point removes the very "
                      "sample init_error rewrites — init_error has no effect.")

        splice_targets = tuple(splice_targets or ())
        if splice_index is not None:
            if isinstance(splice_index, bool) or int(splice_index) < 0:
                raise TypeError(
                    "splice_index is a time index (e.g. 1), not a flag; "
                    "pass a non-negative integer or None."
                )
            splice_index = int(splice_index)
            if splice_source not in self._extractors:
                raise KeyError(
                    f"splice_source '{splice_source}' has no registered "
                    f"extractor. Available: {list(self._extractors)}."
                )
            if drop_first_point and splice_index == 0:
                print("[splice] note: drop_first_point removes index 0 — "
                      "the splice would have no effect.")

        folder = Path(folder)
        files = self._discover_files(folder)
        if max_files is not None:
            files = files[:max_files]

        records: list[ObservableRecord] = []
        skipped: list[tuple[str, str]] = []
        init_hits: dict[str, int] = {}      # observables init_error rewrote
        init_seen: dict[str, int] = {}      # ... out of how many
        stat_hits: dict[str, int] = {}      # bootstrap-stats fields rewritten
        splice_hits: dict[str, int] = {}    # observables the splice rewrote
        splice_stat_files = 0               # files whose stats were spliced
        splice_failed: list[str] = []       # files where the source was unusable
        for fp in files:
            try:
                jd = _load_json(fp)
                lambda_val, t_us = self._extract_lambda_and_time(jd, fp)
            except HMMTableMismatch:
                raise          # wrong table for this folder — never a "skip"
            except Exception as exc:
                # Collected and summarised after the loop: a lambda-scale CSV
                # that covers only part of a folder would otherwise bury the
                # run in one long warning per file.
                skipped.append((fp.name, str(exc)))
                continue

            # One mask for the whole file: the t_max cut and the optional
            # drop of the first sample.  Everything time-dependent (the time
            # axis, every observable, the bootstrap statistics) is sliced by
            # this same mask, so the arrays can never drift apart.
            mask = t_us <= t_max if t_max is not None else None
            if drop_first_point and t_us.size:
                if mask is None:
                    mask = np.ones(t_us.shape, dtype=bool)
                mask[0] = False
            if mask is not None:
                t_us = t_us[mask]
            # Whether the t=0 sample survived the mask — init_error has
            # nothing to rewrite once drop_first_point removed it.
            first_kept = mask is None or bool(mask[0])

            if lambda_range is not None:
                if not (lambda_range[0] <= lambda_val <= lambda_range[1]):
                    continue

            # Build metadata dict from file
            params = jd.get("parameters", jd)
            meta: dict = {}
            for k in ("navg_repeat_experiment", "navg", "n_avg",
                      "false_positive_max", "false_negative_max"):
                v = params.get(k)
                if v is not None:
                    meta[k] = v
            # Omega_DG is stored with capital O in JSON; normalise to MHz (rad/us)
            omega_raw = params.get("Omega_DG") or params.get("omega_DG")
            if omega_raw is not None:
                meta["omega_DG"] = float(omega_raw) * 1e-6
            # Auto-parse scan number from the folder name (e.g. results_50 → 50)
            m_scan = _FOLDER_SCAN_RE.search(folder.name)
            if m_scan:
                meta["scan_number"] = int(m_scan.group(1))

            # Measurement statistics for the data-resampling bootstrap
            try:
                stats = self._bootstrap_stats(jd, mask)
            except Exception as exc:
                print(f"Warning: no bootstrap stats for {fp.name}: {exc}")
                stats = None
            if stats is not None:
                if init_error is not None and first_kept:
                    stats = _stats_with_init_error(stats, init_error, stat_hits)
                if splice_index is not None:
                    stats, did = _stats_with_splice(
                        stats, splice_source.removeprefix("S_"),
                        splice_index, splice_targets,
                    )
                    splice_stat_files += int(did)
                meta["bootstrap_stats"] = stats

            # Source curve for the splice, on the file's FULL time axis — the
            # mask is applied afterwards, so splice_index names the same
            # physical time whatever window this run fits.
            splice_src = None
            if splice_index is not None:
                try:
                    splice_src = np.asarray(
                        self._extractors[splice_source](jd, t_us), dtype=float
                    )
                except Exception as exc:
                    splice_failed.append(f"{fp.name}: {exc}")

            for key in observables:
                try:
                    signal = self._extractors[key](jd, t_us)
                    if init_error is not None and first_kept:
                        signal, hit = _apply_init_error(
                            np.asarray(signal, dtype=float), init_error
                        )
                        init_hits[key] = init_hits.get(key, 0) + int(hit)
                        init_seen[key] = init_seen.get(key, 0) + 1
                    if splice_src is not None and key in splice_targets:
                        signal, hit = _splice_point(
                            np.asarray(signal, dtype=float),
                            splice_src, splice_index,
                        )
                        splice_hits[key] = splice_hits.get(key, 0) + int(hit)
                        if not hit:
                            splice_failed.append(
                                f"{fp.name}: {key}[{splice_index}] not spliced "
                                "(source point missing or NaN)"
                            )
                    if mask is not None:
                        signal = signal[mask]
                    records.append(
                        ObservableRecord(
                            lambda_val=lambda_val,
                            t=t_us,
                            signal=signal,
                            observable_key=key,
                            source_file=fp.name,
                            metadata=meta,
                        )
                    )
                except Exception as exc:
                    print(f"Warning: could not extract '{key}' from {fp.name}: {exc}")

        if skipped:
            print(f"Warning: skipped {len(skipped)}/{len(files)} files:")
            for name, exc in skipped[:5]:
                print(f"    {name}: {exc}")
            if len(skipped) > 5:
                print(f"    ... and {len(skipped) - 5} more")

        if init_error is not None and init_seen:
            done = ", ".join(f"{k} {init_hits.get(k, 0)}/{n}"
                             for k, n in sorted(init_seen.items()))
            stat_done = ", ".join(f"{k} {n}" for k, n in sorted(stat_hits.items()))
            print(f"[init_error] eps={init_error:g}: first sample 1 -> "
                  f"{1.0 - init_error:g} in {done}"
                  + (f"; bootstrap stats {stat_done}" if stat_done else ""))
            untouched = [k for k, n in init_seen.items() if init_hits.get(k, 0) < n]
            if untouched:
                print(f"[init_error] left alone (first sample is a measurement, "
                      f"not an exact 1): {', '.join(sorted(untouched))}")

        if splice_index is not None:
            done = ", ".join(f"{k} {n}" for k, n in sorted(splice_hits.items()))
            print(f"[splice] index {splice_index} <- {splice_source}: "
                  f"{done or 'nothing'} file(s) spliced"
                  + (f"; bootstrap stats {splice_stat_files}"
                     if splice_stat_files else
                     "; bootstrap stats NOT spliced (source counts missing — "
                     "the error bar at that point still belongs to the full "
                     "ensemble)"))
            if splice_failed:
                print(f"[splice] {len(splice_failed)} splice(s) did not apply:")
                for msg in splice_failed[:5]:
                    print(f"    {msg}")
                if len(splice_failed) > 5:
                    print(f"    ... and {len(splice_failed) - 5} more")

        records.sort(key=lambda r: (r.lambda_val, r.observable_key))
        return records
