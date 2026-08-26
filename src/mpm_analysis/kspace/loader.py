"""
loader.py
---------
ParityKSpaceLoader — loads parity-scan JSON folders into KSpaceRecord lists.

Each JSON file in the folder corresponds to one lambda value.  The loader
reads the full 2-D k-space signal from::

    data["fourier_transform_pk_array"]["real part"]     # shape (n_time, n_k)
    data["fourier_transform_pk_array"]["imaginary part"]

and the k-axis from::

    data["fourier_k_array"]                             # shape (n_k,)

The resulting ``signal`` stored in ``KSpaceRecord`` has shape
``(n_k, n_time)`` (transposed from the JSON layout).

Time is converted from nanoseconds to microseconds.  Lambda is read from
``parameters.lambda_zeno_estimation``.

Fallback for older scans
~~~~~~~~~~~~~~~~~~~~~~~~
Older parity-scan JSONs (e.g. ``scan_experiment_results_12``) only store
the FT at k=π and do **not** contain ``fourier_transform_pk_array``.
For those files the loader transparently computes::

    P(k, t) = Σ_n  P(n, t) · exp(+i k n)

from ``data.jump_probability_distribution`` (already normalized so
Σ_n P(n,t) = 1) on the same k-grid as the newer scans
(``np.linspace(0, 2π, 101)``).  A clear stdout banner is printed the first
time this fallback fires per ``load()`` call so it's never silent.

Reuses helpers from the existing experimental loader infrastructure
(_read_lambda_from_json, _read_time_us, _load_json) — no duplication.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from mpm_analysis.io.experimental.base import _load_json, _get_nested
from mpm_analysis.io.experimental.survival_probability import (
    _read_lambda_from_json,
    _read_time_us,
)
from mpm_analysis.kspace.data_types import KSpaceRecord
from mpm_analysis.kspace.sim_loader import SimClickKSpaceLoader
from mpm_analysis.utils.windows_paths import win_long_path


# ---------------------------------------------------------------------------
# DFT helper — used by the fallback path for scans that lack P(k,t).
# ---------------------------------------------------------------------------

# Convention reverse-engineered from scan_experiment_results_14:
#   P(k, t) = Σ_n P(n, t) · exp(+i k n)
# k-grid is uniform on [0, 2π] with 101 points (dk = 2π/100, both endpoints
# included — k=0 and k=2π are the same physical point and are stored twice
# for plotting symmetry).
DEFAULT_K_GRID_N = 101


def compute_pkt_from_pnt(
    P_nt: np.ndarray,
    k_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P(k, t) by DFT of P(n, t) over the n axis.

        P(k, t) = Σ_n  P(n, t) · exp(+i k n)

    No window function — straightforward DFT.  P(n, t) is assumed already
    normalized so that Σ_n P(n, t) = 1 for every t (this guarantees
    P(k=0, t) = 1).

    Parameters
    ----------
    P_nt : np.ndarray, shape (n_time, n_max+1) or (n_max+1, n_time)
        Joint distribution P(n, t).  The axis whose length matches the
        k-grid's "small" dimension (n_max+1) is treated as the n axis;
        if both dims have the same size, axis=1 is taken as n (the
        canonical layout used by the JSON files).
    k_array : np.ndarray or None
        k values (rad).  If None, defaults to ``np.linspace(0, 2π, 101)``
        — the same grid used by the newer experimental scans.

    Returns
    -------
    P_kt : np.ndarray complex, shape (n_time, n_k)
        Fourier-transformed signal.
    k_array : np.ndarray, shape (n_k,)
        The k-grid used (echoed back for convenience).
    """
    P_nt = np.asarray(P_nt, dtype=float)
    if P_nt.ndim != 2:
        raise ValueError(f"P_nt must be 2-D; got shape {P_nt.shape}")
    if k_array is None:
        k_array = np.linspace(0.0, 2.0 * np.pi, DEFAULT_K_GRID_N)
    k_array = np.asarray(k_array, dtype=float)

    n_max = P_nt.shape[1]
    n_idx = np.arange(n_max)
    # exp(+i k n)  →  shape (n_max, n_k)
    phase = np.exp(1j * np.outer(n_idx, k_array))
    # (n_time, n_max) @ (n_max, n_k) → (n_time, n_k)
    return P_nt @ phase, k_array


def make_open_k_grid(k_grid_n: int, endpoint_eps: float = 0.0) -> np.ndarray:
    """Uniform k grid on ``[eps, 2*pi-eps]``.

    Use ``endpoint_eps > 0`` to get points close to 0 and 2*pi while avoiding
    the exact duplicate endpoints.
    """
    k_grid_n = int(k_grid_n)
    endpoint_eps = float(endpoint_eps)
    if k_grid_n < 2:
        raise ValueError(f"k_grid_n must be >= 2; got {k_grid_n}.")
    if endpoint_eps < 0:
        raise ValueError(f"endpoint_eps must be >= 0; got {endpoint_eps}.")
    lo = endpoint_eps
    hi = 2.0 * np.pi - endpoint_eps
    if hi <= lo:
        raise ValueError(
            f"endpoint_eps={endpoint_eps} leaves no k interval inside [0, 2*pi]."
        )
    return np.linspace(lo, hi, k_grid_n)


def _alive_cut(shots_per_time, min_alive_frac: float | None) -> int | None:
    """Number of leading time bins where alive-shot count >= frac * initial.

    Returns ``None`` if no truncation should be applied.  ``shots_per_time`` is
    monotonically decreasing, so this is the contiguous well-sampled prefix.
    """
    if min_alive_frac is None:
        return None
    spt = np.asarray(shots_per_time, dtype=float)
    if spt.size == 0 or spt[0] <= 0:
        return None
    cut = int(np.count_nonzero(spt >= min_alive_frac * spt[0]))
    return cut


class KSpaceJSONLoader:
    """Load k-space experiment JSON folders into ``KSpaceRecord`` lists.

    Each JSON yields one ``KSpaceRecord`` (one λ, all k values).  This is the
    ``source="json"`` k-space loader (reads ``P(k,t)`` directly from the scan
    JSONs, with a DFT fallback from ``P(n,t)``).

    Usage
    -----
    ::

        records = KSpaceJSONLoader().load(folder)
        # records[i].signal.shape == (n_k, n_time)

    Parameters
    ----------
    (none at construction — all options passed to ``load()``)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        folder: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        t_max: float | None = None,
        k_index_range: tuple[int | None, int | None] | None = None,
    ) -> list[KSpaceRecord]:
        """Load all per-lambda JSON files in ``folder``.

        Parameters
        ----------
        folder:
            Path to the folder containing per-lambda JSON files.
        lambda_range:
            If given, only keep records with λ ∈ [lo, hi].
        max_files:
            Load at most this many files (for quick tests).
        t_max:
            Truncate time axis to t ≤ t_max µs.
        k_index_range:
            ``(k_min_idx, k_max_idx)`` — keep only k values in
            ``k_array[k_min_idx : k_max_idx]`` (Python-style slice; upper
            bound is exclusive).  Either bound may be ``None`` to leave
            that end unbounded.  Default ``None`` keeps all k values.

        Returns
        -------
        list[KSpaceRecord]
            One record per file, sorted by λ ascending.
        """
        folder = Path(folder)
        files = sorted(folder.glob("*.json"))
        if max_files is not None:
            files = files[:max_files]

        # Track which loading path each file took, for end-of-load summary.
        n_native = 0
        n_fallback = 0
        fallback_announced = False

        records: list[KSpaceRecord] = []
        for fp in files:
            try:
                rec, used_fallback = self._load_file(
                    fp, t_max=t_max, k_index_range=k_index_range,
                )
            except Exception as exc:
                warnings.warn(f"Skipping {fp.name}: {exc}")
                continue

            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= rec.lambda_val <= hi):
                    continue

            if used_fallback:
                n_fallback += 1
                if not fallback_announced:
                    print(
                        "[ParityKSpaceLoader] *** P(k,t) FALLBACK ACTIVE ***\n"
                        "  'fourier_transform_pk_array' is missing in this folder; "
                        "computing P(k,t) on-the-fly via\n"
                        "      P(k,t) = sum_n P(n,t) * exp(+i*k*n)\n"
                        f"  on k-grid np.linspace(0, 2*pi, {DEFAULT_K_GRID_N}) "
                        "from data.jump_probability_distribution."
                    )
                    fallback_announced = True
                print(f"  [fallback DFT] {fp.name}  lam={rec.lambda_val:.4f}  "
                      f"n_max={rec.metadata.get('n_max_used', '?')}")
            else:
                n_native += 1

            records.append(rec)

        records.sort(key=lambda r: r.lambda_val)

        if n_fallback > 0:
            print(f"[ParityKSpaceLoader] loaded {len(records)} records "
                  f"(native FT: {n_native}, fallback DFT: {n_fallback})")

        return records

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_file(
        self,
        fp: Path,
        *,
        t_max: float | None,
        k_index_range: tuple[int | None, int | None] | None = None,
    ) -> tuple[KSpaceRecord, bool]:
        """Load one JSON file → (record, used_fallback).

        ``used_fallback`` is True iff the P(k,t) array had to be computed
        from P(n,t) instead of read directly from the JSON.
        """
        jd = _load_json(fp)
        data = jd.get("data", {})
        params = jd.get("parameters", jd)

        lambda_val = _read_lambda_from_json(jd)
        t_us = _read_time_us(jd)

        used_fallback = False
        meta: dict = {}

        pk_arr = _get_nested(jd, ["data", "fourier_transform_pk_array"])
        if pk_arr is not None and isinstance(pk_arr, dict):
            # ----- native path: P(k,t) is stored directly -----
            k_raw = data.get("fourier_k_array")
            if k_raw is None:
                raise KeyError(
                    f"'fourier_k_array' not found in data keys: {list(data.keys())}"
                )
            k_array = np.asarray(k_raw, dtype=float)
            real_part = np.asarray(pk_arr["real part"], dtype=float)      # (n_time, n_k)
            imag_part = np.asarray(pk_arr["imaginary part"], dtype=float)  # (n_time, n_k)
            signal = (real_part + 1j * imag_part).T  # (n_k, n_time)
        else:
            # ----- fallback: compute DFT from jump_probability_distribution -----
            P_nt_raw = data.get("jump_probability_distribution")
            if P_nt_raw is None:
                raise KeyError(
                    "Neither 'fourier_transform_pk_array' nor "
                    "'jump_probability_distribution' present in data; "
                    f"keys = {list(data.keys())}"
                )
            P_nt = np.asarray(P_nt_raw, dtype=float)  # (n_time, n_max+1)
            if P_nt.shape[0] != len(t_us):
                raise ValueError(
                    f"P(n,t) time-axis length {P_nt.shape[0]} does not match "
                    f"time_array length {len(t_us)}."
                )
            P_kt, k_array = compute_pkt_from_pnt(P_nt, k_array=None)
            signal = P_kt.T  # (n_k, n_time)
            used_fallback = True
            meta["pkt_source"] = "fallback_dft_from_pnt"
            meta["n_max_used"] = int(P_nt.shape[1])

        # --- optional k slicing ---
        if k_index_range is not None:
            k_lo, k_hi = k_index_range
            sl = slice(k_lo, k_hi)
            k_array = k_array[sl]
            signal = signal[sl, :]
            if k_array.size == 0:
                raise ValueError(
                    f"k_index_range={k_index_range} produced an empty k axis."
                )

        # --- optional time truncation ---
        if t_max is not None:
            mask = t_us <= t_max
            t_us = t_us[mask]
            signal = signal[:, mask]

        # --- metadata ---
        for key in ("navg_repeat_experiment", "navg", "n_avg",
                    "false_positive_max", "false_negative_max"):
            v = params.get(key)
            if v is not None:
                meta[key] = v
        omega_raw = params.get("Omega_DG") or params.get("omega_DG")
        if omega_raw is not None:
            meta["omega_DG"] = float(omega_raw) * 1e-6

        rec = KSpaceRecord(
            lambda_val=lambda_val,
            t=t_us,
            k_array=k_array,
            signal=signal,
            source_file=fp.name,
            metadata=meta,
        )
        return rec, used_fallback


class CompressedClickKSpaceLoader:
    """Load **compressed-click** analysis NPZs into ``KSpaceRecord`` lists.

    Drop-in replacement for :class:`KSpaceJSONLoader` that feeds the same
    ``KSpaceAnalysis`` / pole pipeline, but reads the pre-computed counting
    statistics produced by
    :class:`mpm_analysis.io.experimental.compressed_click_record.CompressedClickBuilder`.

    Each compressed NPZ stores ``P(k,t)`` arrays (``pkt_real``/``pkt_imag``)
    normalized by the shots that still have compressed data at each time.  For
    the pole analysis this loader keeps only the well-sampled leading time
    window, controlled by ``min_alive_frac``.

    Unlike the parity loader, the compressed time axis length ``T_c`` differs
    per rate/λ; we keep each record's native axis and do **not** force a common
    grid (``KSpaceAnalysis`` runs per record, so this is fine).

    Usage
    -----
    ::

        records = CompressedClickKSpaceLoader().load(analysis_folder)
        result  = KSpaceAnalysis().run(records)
    """

    def load(
        self,
        folder: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        t_max: float | None = None,
        k_index_range: tuple[int | None, int | None] | None = None,
        min_alive_frac: float | None = 0.5,
        k_grid_n: int | None = None,
        k_endpoint_eps: float = 0.0,
        latest_only: bool = True,
    ) -> list[KSpaceRecord]:
        """Load all compressed NPZs in ``folder`` into ``KSpaceRecord`` list.

        Parameters mirror :meth:`KSpaceJSONLoader.load` where applicable.
        ``latest_only`` keeps only the newest timestamped file per rate.

        ``P(k,t)`` is built from the recorded data — each shot's click times plus
        its **total compressed length** (which includes the trailing dark after
        the last click).  A shot contributes at compressed time ``t`` iff
        ``t < compressed_length``; each retained time bin is normalized by that
        time's surviving-shot count.

        ``min_alive_frac`` truncates each record's compressed time axis to bins
        with alive-shot count ≥ ``min_alive_frac`` × initial.  The default
        ``0.5`` keeps only times where at least half the compressed data exists.
        ``None`` keeps the full axis for diagnostics.

        ``k_grid_n`` recomputes ``P(k,t)`` from stored ``P(n,t)`` on a fresh
        uniform grid instead of using the stored 101-point grid.  With
        ``k_endpoint_eps > 0`` the grid is ``[eps, 2*pi-eps]``, so it can get
        close to the endpoints without evaluating exactly at 0 or 2*pi.
        """
        # Lazy import avoids any import-time coupling between kspace and io.
        from mpm_analysis.io.experimental.compressed_click_record import (
            CompressedClickLoader,
        )

        cc_records = CompressedClickLoader().load(
            folder, lambda_range=lambda_range, latest_only=latest_only,
        )

        custom_k_array = (
            make_open_k_grid(k_grid_n, k_endpoint_eps)
            if k_grid_n is not None else None
        )

        records: list[KSpaceRecord] = []
        for cc in cc_records:
            if custom_k_array is None:
                k_array = np.asarray(cc.fourier_k_array, dtype=float)
                signal = cc.signal_pkt             # survivor-normalized P(k,t)
                k_grid_source = "stored_npz"
            else:
                p_kt, k_array = compute_pkt_from_pnt(
                    cc.pnt.astype(float), k_array=custom_k_array,
                )
                signal = p_kt.T
                k_grid_source = "recomputed_from_pnt"
            t_us = np.asarray(cc.t, dtype=float)

            if k_index_range is not None:
                sl = slice(*k_index_range)
                k_array = k_array[sl]
                signal = signal[sl, :]
                if k_array.size == 0:
                    raise ValueError(
                        f"k_index_range={k_index_range} produced an empty k axis."
                    )
            cut = _alive_cut(cc.shots_per_time, min_alive_frac)
            if cut is not None:
                t_us = t_us[:cut]
                signal = signal[:, :cut]
            if t_max is not None:
                mask = t_us <= t_max
                t_us = t_us[mask]
                signal = signal[:, mask]

            meta = dict(cc.metadata)
            meta.update(
                source="compressed_click",
                rate_khz=cc.rate_khz,
                alpha=cc.alpha, beta=cc.beta, snr=cc.snr,
                n_runs=cc.n_runs, n_shots=int(cc.jump_counts.size),
                dt_us=cc.dt_us,
                pkt_normalization="surviving_shots",
                min_alive_frac=min_alive_frac,
                k_grid_source=k_grid_source,
                k_grid_n=int(k_array.size),
                k_endpoint_eps=(
                    float(k_endpoint_eps) if custom_k_array is not None else 0.0
                ),
                k_min=float(k_array[0]) if k_array.size else float("nan"),
                k_max=float(k_array[-1]) if k_array.size else float("nan"),
                alive_initial=int(cc.shots_per_time[0]) if cc.shots_per_time.size else 0,
                alive_min_retained=(
                    int(cc.shots_per_time[:len(t_us)].min()) if len(t_us) else 0
                ),
            )
            if np.isfinite(cc.omega_dg):
                meta["omega_DG"] = float(cc.omega_dg) * 1e-6   # Hz -> rad/µs
            if np.isfinite(cc.omega_bg):
                meta["omega_BG"] = float(cc.omega_bg) * 1e-6

            records.append(
                KSpaceRecord(
                    lambda_val=cc.lambda_val,
                    t=t_us,
                    k_array=k_array,
                    signal=signal,
                    source_file=cc.source_file,
                    metadata=meta,
                )
            )
        records.sort(key=lambda r: r.lambda_val)
        return records

    def load_per_run(
        self,
        folder: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        t_max: float | None = None,
        k_index_range: tuple[int | None, int | None] | None = None,
        min_alive_frac: float | None = 0.5,
        k_grid_n: int | None = None,
        k_endpoint_eps: float = 0.0,
        latest_only: bool = True,
    ) -> list[list[KSpaceRecord]]:
        """Load **per-run** ``KSpaceRecord`` lists for run-level bootstrap.

        Each compressed NPZ stores per-run counting histograms (``counts_hist``,
        ``alive``).  This rebuilds ``P(k,t)`` separately for each individual run
        ``r`` (``pnt_run(r)`` -> :func:`compute_pkt_from_pnt`) so that a full
        pole analysis can be run independently per run and the spread across runs
        used as error bars.

        Returns
        -------
        list[list[KSpaceRecord]]
            Outer index = run; inner list = one ``KSpaceRecord`` per λ (sorted by
            λ).  Only runs present in **all** loaded records are returned (the
            common ``min`` run count), so the per-run grids are aligned.
        """
        from mpm_analysis.io.experimental.compressed_click_record import (
            CompressedClickLoader,
        )

        cc_records = CompressedClickLoader().load(
            folder, lambda_range=lambda_range, latest_only=latest_only,
        )
        cc_records.sort(key=lambda c: c.lambda_val)
        if not cc_records:
            return []
        n_runs = min(int(c.n_runs) for c in cc_records)
        custom_k_array = (
            make_open_k_grid(k_grid_n, k_endpoint_eps)
            if k_grid_n is not None else None
        )

        per_run: list[list[KSpaceRecord]] = []
        for r in range(n_runs):
            recs: list[KSpaceRecord] = []
            for cc in cc_records:
                pnt_r = cc.pnt_run(r)                         # (T_c, n_max+1)
                p_kt, k_array = compute_pkt_from_pnt(
                    pnt_r.astype(float),
                    k_array=(
                        custom_k_array if custom_k_array is not None
                        else np.asarray(cc.fourier_k_array, float)
                    ),
                )
                k_grid_source = (
                    "recomputed_from_pnt" if custom_k_array is not None
                    else "stored_npz"
                )
                signal = p_kt.T                            # (n_k, n_time)
                t_us = np.asarray(cc.t, dtype=float)
                if k_index_range is not None:
                    sl = slice(*k_index_range)
                    k_array = k_array[sl]
                    signal = signal[sl, :]
                cut = _alive_cut(cc.alive[r], min_alive_frac)
                if cut is not None:
                    t_us = t_us[:cut]
                    signal = signal[:, :cut]
                if t_max is not None:
                    mask = t_us <= t_max
                    t_us = t_us[mask]
                    signal = signal[:, mask]
                meta = dict(cc.metadata)
                meta.update(source="compressed_click", rate_khz=cc.rate_khz,
                            run_index=r, dt_us=cc.dt_us,
                            pkt_normalization="surviving_shots",
                            k_grid_source=k_grid_source,
                            k_grid_n=int(k_array.size),
                            k_endpoint_eps=(
                                float(k_endpoint_eps)
                                if custom_k_array is not None else 0.0
                            ),
                            k_min=float(k_array[0]) if k_array.size else float("nan"),
                            k_max=float(k_array[-1]) if k_array.size else float("nan"),
                            min_alive_frac=min_alive_frac,
                            alive_initial=(
                                int(cc.alive[r, 0]) if cc.alive.shape[1] else 0
                            ),
                            alive_min_retained=(
                                int(cc.alive[r, :len(t_us)].min()) if len(t_us) else 0
                            ))
                if np.isfinite(cc.omega_dg):
                    meta["omega_DG"] = float(cc.omega_dg) * 1e-6
                recs.append(KSpaceRecord(
                    lambda_val=cc.lambda_val, t=t_us, k_array=k_array,
                    signal=signal, source_file=cc.source_file, metadata=meta,
                ))
            per_run.append(recs)
        return per_run


# Back-compat alias: the JSON k-space loader was previously named
# ``ParityKSpaceLoader``.  Kept so older imports keep working.
ParityKSpaceLoader = KSpaceJSONLoader

# ---------------------------------------------------------------------------
# Source registry — pick the k-space loader by a simple source key.
# ---------------------------------------------------------------------------

KSPACE_LOADERS = {
    "json": KSpaceJSONLoader,
    "compressed": CompressedClickKSpaceLoader,
    "simulated": SimClickKSpaceLoader,
}


def get_kspace_loader(source: str):
    """Return the k-space loader **class** for a source key.

    Parameters
    ----------
    source : ``"json"`` (experiment JSON, ``KSpaceJSONLoader``),
        ``"compressed"`` (compressed click records, ``CompressedClickKSpaceLoader``), or
        ``"simulated"`` (simulation NPZ from ``generate_click_record_raw.py``,
        ``SimClickKSpaceLoader``).
    """
    try:
        return KSPACE_LOADERS[source]
    except KeyError:
        raise ValueError(
            f"Unknown k-space source {source!r}. "
            f"Valid: {sorted(KSPACE_LOADERS)}."
        ) from None
