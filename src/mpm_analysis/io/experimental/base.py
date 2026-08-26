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

        folder = Path(folder)
        files = self._discover_files(folder)
        if max_files is not None:
            files = files[:max_files]

        records: list[ObservableRecord] = []
        skipped: list[tuple[str, str]] = []
        for fp in files:
            try:
                jd = _load_json(fp)
                lambda_val, t_us = self._extract_lambda_and_time(jd, fp)
            except Exception as exc:
                # Collected and summarised after the loop: a lambda-scale CSV
                # that covers only part of a folder would otherwise bury the
                # run in one long warning per file.
                skipped.append((fp.name, str(exc)))
                continue

            if t_max is not None:
                mask = t_us <= t_max
                t_us = t_us[mask]
            else:
                mask = None

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
                meta["bootstrap_stats"] = stats

            for key in observables:
                try:
                    signal = self._extractors[key](jd, t_us)
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

        records.sort(key=lambda r: (r.lambda_val, r.observable_key))
        return records
