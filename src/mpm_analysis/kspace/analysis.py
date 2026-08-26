"""
analysis.py
-----------
KSpaceAnalysis — run MPM on every (lambda, k) cell of a k-space dataset.

For each cell the workflow is:
  1. Extract the complex 1-D time trace y = signal[k_idx, :].
  2. Build the Hankel matrix and compute its SVD (stored for diagnostics).
  3. Infer the model order via ``infer_pole_order()`` (or use a fixed order).
  4. Run ``matrix_pencil_method()`` with the inferred order.
  5. Solve for complex amplitudes.
  6. Sort poles canonically.

The result is a ``KSpacePolesResult`` with poles padded to ``max_order``
(NaN where a cell used fewer poles than max_order).

Bootstrap (n_bootstrap > 1) uses residual resampling per cell, matching
the existing ``bootstrap.py`` approach.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from mpm_analysis.analysis.matrix_pencil import (
    build_hankel_matrix,
    matrix_pencil_method,
    solve_linear_amplitudes,
)
from mpm_analysis.analysis.pole_sorting import sort_poles_canonical
from mpm_analysis.kspace.data_types import KSpacePolesResult, KSpaceRecord
from mpm_analysis.kspace.pole_order import PoleOrderConfig, infer_pole_order


class KSpaceAnalysis:
    """Extract MPM poles for every (lambda, k) cell in a list of KSpaceRecords.

    Parameters
    ----------
    pole_order_config:
        Controls automatic and manual model-order selection per cell.
        Default: ``PoleOrderConfig()`` (max_order=3, threshold=0.5).
    n_bootstrap:
        Number of residual-resampling bootstrap draws per cell.
        ``1`` (default) disables bootstrap — poles are the direct MPM
        result with no uncertainty estimate.
    method:
        ``"mpm"`` (default) — standard matrix pencil.
        ``"rrha"`` — reduced-rank Hankel approximation variant (slightly
        more robust near exceptional points).
    L:
        Pencil parameter override.  ``None`` (default) uses the
        spectral-gap optimised value from ``matrix_pencil_method``.
    signal_component:
        Which part of the complex signal to feed into MPM.
        ``"complex"`` (default) uses the full complex trace.
        ``"real"`` or ``"imag"`` uses only that component.
    """

    def __init__(
        self,
        pole_order_config: PoleOrderConfig | None = None,
        n_bootstrap: int = 1,
        method: str = "mpm",
        L: int | None = None,
        signal_component: str = "complex",
    ) -> None:
        self.pole_order_config = pole_order_config or PoleOrderConfig()
        self.n_bootstrap = n_bootstrap
        self.method = method
        self.L = L
        self.signal_component = signal_component

        if method not in ("mpm", "rrha"):
            raise ValueError(f"Unknown method '{method}'. Choose 'mpm' or 'rrha'.")
        if signal_component not in ("complex", "real", "imag"):
            raise ValueError(
                f"Unknown signal_component '{signal_component}'. "
                "Choose 'complex', 'real', or 'imag'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, records: list[KSpaceRecord]) -> KSpacePolesResult:
        """Run MPM on all (lambda, k) cells.

        Parameters
        ----------
        records:
            List of ``KSpaceRecord`` objects (one per lambda), e.g. from
            ``KSpaceJSONLoader().load(folder)``.  All records must share the
            same k-axis.  Each record is fit against its **own** time axis, so
            records may have different ``n_time`` (e.g. per-rate ``min_alive_frac``
            truncation of compressed records).  ``dt`` must be uniform.

        Returns
        -------
        KSpacePolesResult
            Poles, amplitudes, orders, and SVD singular values for every
            cell.  Shape: (n_lambda, n_k, max_order).
        """
        if not records:
            raise ValueError("records is empty.")

        records = sorted(records, key=lambda r: r.lambda_val)
        n_lambda = len(records)
        n_k = records[0].n_k
        cfg = self.pole_order_config
        max_order = cfg.max_order
        n_sv_store = max_order + 4  # store a few extra singular values

        lambda_values = np.array([r.lambda_val for r in records])
        k_array = records[0].k_array.copy()
        t = records[0].t.copy()

        poles_out = np.full((n_lambda, n_k, max_order), np.nan, dtype=complex)
        amps_out = np.full((n_lambda, n_k, max_order), np.nan, dtype=complex)
        order_out = np.zeros((n_lambda, n_k), dtype=int)
        svd_out = np.full((n_lambda, n_k, n_sv_store), np.nan, dtype=float)

        for li, rec in enumerate(records):
            # Use this record's own time axis: with per-rate truncation
            # (e.g. min_alive_frac), records may have different n_time.  dt is
            # uniform, so this is correct; for equal-length sources it is a no-op.
            t_rec = rec.t
            for ki in range(n_k):
                y = self._extract_signal(rec, ki)
                try:
                    poles, sv, amps, order = self._fit_cell(y, t_rec, li, ki)
                except Exception as exc:
                    warnings.warn(
                        f"MPM failed at lambda_idx={li}, k_idx={ki}: {exc}"
                    )
                    continue

                poles_out[li, ki, :order] = poles
                amps_out[li, ki, :order] = amps
                order_out[li, ki] = order
                n_sv = min(len(sv), n_sv_store)
                svd_out[li, ki, :n_sv] = sv[:n_sv]

        return KSpacePolesResult(
            lambda_values=lambda_values,
            k_array=k_array,
            t=t,
            poles=poles_out,
            amplitudes=amps_out,
            pole_order=order_out,
            svd_singvals=svd_out,
            metadata={
                "method": self.method,
                "n_bootstrap": self.n_bootstrap,
                "signal_component": self.signal_component,
                "threshold": cfg.threshold,
                "max_order": max_order,
                "min_order": cfg.min_order,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_signal(self, rec: KSpaceRecord, ki: int) -> np.ndarray:
        """Return the 1-D signal for cell ki of record rec."""
        sig = rec.signal[ki, :]
        if self.signal_component == "real":
            return sig.real
        if self.signal_component == "imag":
            return sig.imag
        return sig  # complex

    def _fit_cell(
        self,
        y: np.ndarray,
        t: np.ndarray,
        li: int,
        ki: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Run MPM for one cell; return (poles, singvals, amplitudes, order)."""
        cfg = self.pole_order_config

        # Build Hankel once at L = len(y)//3 (good default) and reuse its SVD
        # for both order inference and the MPM pencil solve, avoiding the costly
        # optimal-L search that iterates over many L values per cell.
        N = len(y)
        L = self.L if self.L is not None else max(cfg.max_order + 1, N // 3)
        L = max(cfg.max_order + 1, min(L, N - cfg.max_order - 1))
        H = build_hankel_matrix(y, L)
        sv = np.linalg.svd(H, compute_uv=False)

        order = infer_pole_order(sv, cfg, lam_idx=li, k_idx=ki)

        # --- MPM — pass L directly to skip the internal L-search ---
        if self.method == "rrha":
            from mpm_analysis.analysis.matrix_pencil_rrha import mpm_rrha
            poles_raw, sv_used, L_used = mpm_rrha(t, y, order, L=L)
        else:
            poles_raw, sv_used, L_used = matrix_pencil_method(t, y, order, L=L)

        if self.n_bootstrap > 1:
            poles_raw = self._bootstrap_poles(t, y, poles_raw, order)

        poles_sorted = sort_poles_canonical(poles_raw)
        amps = solve_linear_amplitudes(t, y, poles_sorted)

        return poles_sorted, sv, amps, order

    def _bootstrap_poles(
        self,
        t: np.ndarray,
        y: np.ndarray,
        poles_init: np.ndarray,
        order: int,
    ) -> np.ndarray:
        """Residual-resampling bootstrap; return median poles."""
        from mpm_analysis.analysis.matrix_pencil import reconstruct_signal

        amps = solve_linear_amplitudes(t, y, poles_init)
        y_clean = reconstruct_signal(t, poles_init, amps)
        residuals = y - y_clean

        rng = np.random.default_rng()
        boot_poles = []
        for _ in range(self.n_bootstrap):
            y_boot = y_clean + rng.choice(residuals, size=len(residuals), replace=True)
            try:
                if self.method == "rrha":
                    from mpm_analysis.analysis.matrix_pencil_rrha import mpm_rrha
                    p, _, _ = mpm_rrha(t, y_boot, order, L=self.L)
                else:
                    p, _, _ = matrix_pencil_method(t, y_boot, order, L=self.L)
                boot_poles.append(sort_poles_canonical(p))
            except Exception:
                continue

        if not boot_poles:
            return poles_init

        boot_arr = np.array(boot_poles)  # (n_boot, order)
        return (
            np.median(boot_arr.real, axis=0)
            + 1j * np.median(boot_arr.imag, axis=0)
        )
