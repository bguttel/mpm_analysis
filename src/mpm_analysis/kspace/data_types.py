"""
data_types.py
-------------
Core data structures for k-space parity analysis.

KSpaceRecord
    One lambda slice of the full k-space parity signal.
    Holds a complex 2-D array signal[k_idx, time_idx] alongside the
    k-axis and time-axis.  Interoperates with ObservableRecord via
    ``at_k()``, making it compatible with the existing MPMPipeline.

KSpacePolesResult
    MPM poles across the full (lambda, k) grid.  Supports save/load
    (npz round-trip), convenience slices, and direct plotting hooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord


# ---------------------------------------------------------------------------
# KSpaceRecord
# ---------------------------------------------------------------------------

@dataclass
class KSpaceRecord:
    """One lambda slice of the full k-space parity signal.

    Parameters
    ----------
    lambda_val:
        Dimensionless Zeno parameter λ for this slice.
    t:
        Time axis in **microseconds** (n_time,).
    k_array:
        Quasi-momentum values in radians (n_k,).
    signal:
        Complex parity signal, shape (n_k, n_time).
        ``signal[i, j]`` is the complex parity at k[i] and t[j].
    source_file:
        Original JSON filename for traceability.
    metadata:
        Free-form dict (e.g. omega_DG, navg).
    """

    lambda_val: float
    t: np.ndarray        # (n_time,) µs
    k_array: np.ndarray  # (n_k,)
    signal: np.ndarray   # (n_k, n_time) complex
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.k_array = np.asarray(self.k_array, dtype=float)
        self.signal = np.asarray(self.signal, dtype=complex)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def n_k(self) -> int:
        return len(self.k_array)

    @property
    def n_time(self) -> int:
        return len(self.t)

    # ------------------------------------------------------------------
    # K-axis lookup
    # ------------------------------------------------------------------

    def nearest_k_idx(self, k_val: float) -> int:
        """Return the index of the k value closest to ``k_val``."""
        return int(np.argmin(np.abs(self.k_array - k_val)))

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def at_k(
        self,
        k_idx_or_val,
        *,
        component: str = "real",
    ) -> ObservableRecord:
        """Extract a 1-D time trace at a specific k as an ObservableRecord.

        Parameters
        ----------
        k_idx_or_val:
            Integer index into ``k_array``, or a float k value (nearest
            neighbour lookup).
        component:
            ``"real"`` (default), ``"imag"``, or ``"complex"`` — which
            part of the complex signal to store in the record.  Note:
            ObservableRecord.signal is float-typed, so ``"complex"``
            stores the real part (use the raw KSpaceRecord for both parts).

        Returns
        -------
        ObservableRecord
            Compatible with ``MPMPipeline.from_records()``.
        """
        if isinstance(k_idx_or_val, (int, np.integer)):
            k_idx = int(k_idx_or_val)
        else:
            k_idx = self.nearest_k_idx(float(k_idx_or_val))

        sig = self.signal[k_idx, :]
        if component == "real":
            signal_1d = sig.real
        elif component == "imag":
            signal_1d = sig.imag
        else:
            signal_1d = sig.real  # fallback

        key = f"parity_{component}" if component in ("real", "imag") else "parity_real"
        meta = dict(self.metadata, k_val=float(self.k_array[k_idx]), k_idx=k_idx)
        return ObservableRecord(
            lambda_val=self.lambda_val,
            t=self.t.copy(),
            signal=signal_1d,
            observable_key=key,
            source_file=self.source_file,
            metadata=meta,
        )

    def time_slice(self, t_max_us: float) -> "KSpaceRecord":
        """Return a new record restricted to ``t <= t_max_us`` µs."""
        mask = self.t <= t_max_us
        return KSpaceRecord(
            lambda_val=self.lambda_val,
            t=self.t[mask],
            k_array=self.k_array.copy(),
            signal=self.signal[:, mask],
            source_file=self.source_file,
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# KSpacePolesResult
# ---------------------------------------------------------------------------

@dataclass
class KSpacePolesResult:
    """MPM poles across the full (lambda, k) grid.

    Parameters
    ----------
    lambda_values:
        Sorted λ values, shape (n_lambda,).
    k_array:
        Quasi-momentum axis, shape (n_k,).
    t:
        Representative time axis (µs), shape (n_time,).
    poles:
        Complex poles in rad/µs, shape (n_lambda, n_k, max_order).
        NaN fills cells where the inferred order is below max_order.
    amplitudes:
        Complex amplitudes, shape (n_lambda, n_k, max_order).  NaN
        where poles are NaN.
    pole_order:
        Actual model order used per cell, shape (n_lambda, n_k).
    svd_singvals:
        Hankel singular values, shape (n_lambda, n_k, n_sv).  Used for
        diagnostics and order-calibration inspection.
    metadata:
        Free-form provenance dict.
    """

    lambda_values: np.ndarray   # (n_lambda,)
    k_array: np.ndarray         # (n_k,)
    t: np.ndarray               # (n_time,) µs
    poles: np.ndarray           # (n_lambda, n_k, max_order) complex
    amplitudes: np.ndarray      # (n_lambda, n_k, max_order) complex
    pole_order: np.ndarray      # (n_lambda, n_k) int
    svd_singvals: np.ndarray    # (n_lambda, n_k, n_sv) float
    metadata: dict = field(default_factory=dict)
    # Optional run-bootstrap error bars (set by KSpacePipeline when
    # bootstrap_over_runs=True).  Complex convention: ``.real`` is the std of
    # Re(pole) (decay-rate error), ``.imag`` is the std of Im(pole) (frequency
    # error), per (lambda, k, order).  ``None`` when not computed.
    poles_err: np.ndarray | None = None
    amplitudes_err: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.lambda_values = np.asarray(self.lambda_values, dtype=float)
        self.k_array = np.asarray(self.k_array, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.poles = np.asarray(self.poles, dtype=complex)
        self.amplitudes = np.asarray(self.amplitudes, dtype=complex)
        self.pole_order = np.asarray(self.pole_order, dtype=int)
        self.svd_singvals = np.asarray(self.svd_singvals, dtype=float)
        if self.poles_err is not None:
            self.poles_err = np.asarray(self.poles_err, dtype=complex)
        if self.amplitudes_err is not None:
            self.amplitudes_err = np.asarray(self.amplitudes_err, dtype=complex)

    # ------------------------------------------------------------------
    # Shape / index helpers
    # ------------------------------------------------------------------

    @property
    def n_lambda(self) -> int:
        return len(self.lambda_values)

    @property
    def n_k(self) -> int:
        return len(self.k_array)

    @property
    def max_order(self) -> int:
        return self.poles.shape[2]

    def nearest_k_idx(self, k_val: float) -> int:
        return int(np.argmin(np.abs(self.k_array - k_val)))

    def nearest_lambda_idx(self, lam_val: float) -> int:
        return int(np.argmin(np.abs(self.lambda_values - lam_val)))

    def _resolve_k(self, k_idx_or_val) -> int:
        if isinstance(k_idx_or_val, (int, np.integer)):
            return int(k_idx_or_val)
        return self.nearest_k_idx(float(k_idx_or_val))

    def _resolve_lambda(self, lam_idx_or_val) -> int:
        if isinstance(lam_idx_or_val, (int, np.integer)):
            return int(lam_idx_or_val)
        return self.nearest_lambda_idx(float(lam_idx_or_val))

    # ------------------------------------------------------------------
    # Derived observables
    # ------------------------------------------------------------------

    def decay_rates(self) -> np.ndarray:
        """Decay rates −Re(poles), shape (n_lambda, n_k, max_order)."""
        return -self.poles.real

    def frequencies(self) -> np.ndarray:
        """Oscillation frequencies Im(poles), shape (n_lambda, n_k, max_order)."""
        return self.poles.imag

    # ------------------------------------------------------------------
    # Slicing helpers
    # ------------------------------------------------------------------

    def poles_at_k(self, k_idx_or_val) -> np.ndarray:
        """Return poles for all lambdas at a fixed k, shape (n_lambda, max_order)."""
        return self.poles[:, self._resolve_k(k_idx_or_val), :]

    def poles_at_lambda(self, lam_idx_or_val) -> np.ndarray:
        """Return poles for all k at a fixed lambda, shape (n_k, max_order)."""
        return self.poles[self._resolve_lambda(lam_idx_or_val), :, :]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a compressed NPZ file."""
        import os
        from mpm_analysis.utils.windows_paths import win_long_path
        path = Path(path)
        os.makedirs(win_long_path(path.parent), exist_ok=True)
        np.savez_compressed(
            win_long_path(path),
            lambda_values=self.lambda_values,
            k_array=self.k_array,
            t=self.t,
            poles_real=self.poles.real,
            poles_imag=self.poles.imag,
            amplitudes_real=self.amplitudes.real,
            amplitudes_imag=self.amplitudes.imag,
            pole_order=self.pole_order,
            svd_singvals=self.svd_singvals,
            # optional run-bootstrap error bars (empty array = absent)
            poles_err_real=(self.poles_err.real if self.poles_err is not None
                            else np.zeros(0)),
            poles_err_imag=(self.poles_err.imag if self.poles_err is not None
                            else np.zeros(0)),
            amplitudes_err_real=(self.amplitudes_err.real if self.amplitudes_err is not None
                                 else np.zeros(0)),
            amplitudes_err_imag=(self.amplitudes_err.imag if self.amplitudes_err is not None
                                 else np.zeros(0)),
            # metadata serialised as a single string (JSON)
            metadata_json=np.bytes_(
                __import__("json").dumps(self.metadata).encode()
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "KSpacePolesResult":
        """Load from an NPZ file written by ``save()``."""
        import json as _json
        from mpm_analysis.utils.windows_paths import win_long_path
        data = np.load(win_long_path(path), allow_pickle=False)
        meta_raw = data["metadata_json"]
        metadata = _json.loads(bytes(meta_raw).decode()) if meta_raw.size else {}

        def _opt_err(real_key: str, imag_key: str):
            if real_key not in data:
                return None
            re_, im_ = data[real_key], data[imag_key]
            if re_.size == 0:
                return None
            return re_ + 1j * im_

        return cls(
            lambda_values=data["lambda_values"],
            k_array=data["k_array"],
            t=data["t"],
            poles=data["poles_real"] + 1j * data["poles_imag"],
            amplitudes=data["amplitudes_real"] + 1j * data["amplitudes_imag"],
            pole_order=data["pole_order"],
            svd_singvals=data["svd_singvals"],
            metadata=metadata,
            poles_err=_opt_err("poles_err_real", "poles_err_imag"),
            amplitudes_err=_opt_err("amplitudes_err_real", "amplitudes_err_imag"),
        )
