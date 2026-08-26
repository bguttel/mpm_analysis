"""
refine_step.py
--------------
Optional pipeline step: nonlinear refinement of MPM poles.

This step reads ``state["poles_raw"]`` (from ``MPMStep``) and refines
each set of poles using ``analysis.matrix_pencil.refine_poles`` — a
weighted complex-exponential least-squares fit (variable projection:
amplitudes re-solved at every iteration), jointly over all configured
observables.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.data_types.observables import ObservableRecord


class RefineStep(AnalysisStep):
    """Nonlinear refinement of raw MPM poles.

    Reads ``state["poles_raw"]`` and ``state["lambdas"]``, writes
    ``state["poles_raw"]`` in-place with refined values (Hungarian-matched to
    the initial poles, so branch labels from ``MPMStep`` are preserved) and
    updates ``state["amplitudes_raw"]`` accordingly.  Also writes:

        ``state["refine_params"]``    : list of optimal parameter vectors (or None)
        ``state["refine_cov"]``       : list of parameter covariances (or None)
        ``state["refine_conjugate"]`` : bool — parametrisation used

    Parameters
    ----------
    bounds_real:
        Optional ``(lower, upper)`` bounds on Re(pole).  Ignored when
        ``conjugate_pairs=True``.
    bounds_imag:
        Optional ``(lower, upper)`` bounds on Im(pole).  Ignored when
        ``conjugate_pairs=True``.
    observable_key:
        Observable(s) to use for residuals.  ``None`` → all keys analysed by
        ``MPMStep`` (``state["observable_keys"]``); a list fits them jointly
        (shared poles, per-observable amplitudes).
    conjugate_pairs:
        Constrain poles to conjugate pairs / real poles (roots of real
        polynomials); the fit moves smoothly between the two regimes.
    use_sigma:
        Weight residuals by ``1/record.sigma`` where available.
    """

    def __init__(
        self,
        bounds_real: tuple[float, float] | None = (-np.inf, 0.0),
        bounds_imag: tuple[float, float] | None = None,
        observable_key: str | list[str] | None = None,
        *,
        conjugate_pairs: bool = False,
        use_sigma: bool = True,
    ) -> None:
        self.bounds_real = bounds_real
        self.bounds_imag = bounds_imag
        self.observable_key = observable_key
        self.conjugate_pairs = conjugate_pairs
        self.use_sigma = use_sigma

    def _weights(self, recs: list[ObservableRecord], n: int) -> np.ndarray | None:
        """Build a (k, N) 1/sigma weight matrix; None if uniform everywhere."""
        if not self.use_sigma:
            return None
        rows = []
        any_sigma = False
        for rec in recs:
            if rec.sigma is not None and np.any(rec.sigma > 0):
                eps = np.finfo(float).tiny
                rows.append(1.0 / np.maximum(rec.sigma, eps))
                any_sigma = True
            else:
                rows.append(np.ones(n))
        return np.vstack(rows) if any_sigma else None

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from mpm_analysis.analysis.matrix_pencil import (
            refine_poles,
            solve_linear_amplitudes,
        )
        from mpm_analysis.analysis.pole_sorting import match_poles_hungarian
        from mpm_analysis.data_types.observables import group_records_by_lambda

        poles_raw: list = state.get("poles_raw", [])
        amplitudes_raw: list = state.get("amplitudes_raw", [])
        lambdas: list = state.get("lambdas", [])

        # Resolve keys: explicit → as given; None → whatever MPMStep analysed
        key = self.observable_key
        if key is None:
            keys = state.get("observable_keys") or [records[0].observable_key]
        elif isinstance(key, str):
            keys = [key]
        else:
            keys = list(key)

        if len(keys) == 1:
            by_lambda = {
                r.lambda_val: (r.t, r.signal, [r])
                for r in records
                if r.observable_key == keys[0]
            }
        else:
            by_lambda = {
                lam: (t, Y, recs)
                for lam, t, Y, recs in group_records_by_lambda(records, keys)
            }

        poles_refined = []
        refine_params: list[np.ndarray | None] = []
        refine_cov: list[np.ndarray | None] = []
        for i, (lam, poles) in enumerate(zip(lambdas, poles_raw)):
            group = by_lambda.get(lam)
            if np.any(np.isnan(poles)) or group is None:
                poles_refined.append(poles)
                refine_params.append(None)
                refine_cov.append(None)
                continue
            t, Y, recs = group
            weights = self._weights(recs, len(t)) if Y.ndim == 2 else None
            if Y.ndim == 1 and self.use_sigma:
                rec = recs[0]
                if rec.sigma is not None and np.any(rec.sigma > 0):
                    weights = 1.0 / np.maximum(rec.sigma, np.finfo(float).tiny)
            try:
                poles_r, p_opt, cov_p = refine_poles(
                    t,
                    Y,
                    poles,
                    weights=weights,
                    conjugate_pairs=self.conjugate_pairs,
                    bounds_real=self.bounds_real,
                    bounds_imag=self.bounds_imag,
                    return_cov=True,
                )
                # Keep the branch labels assigned by MPMStep's cross-lambda
                # tracking: match refined poles back to their initial poles.
                poles_r = match_poles_hungarian(poles_r, poles)
                poles_refined.append(poles_r)
                refine_params.append(p_opt)
                refine_cov.append(cov_p)
                if i < len(amplitudes_raw):
                    amplitudes_raw[i] = solve_linear_amplitudes(
                        t, Y, poles_r, weights=weights
                    )
            except Exception as exc:
                print(f"Warning: RefineStep failed for lam={lam:.3f}: {exc}")
                poles_refined.append(poles)
                refine_params.append(None)
                refine_cov.append(None)

        state["poles_raw"] = poles_refined
        state["amplitudes_raw"] = amplitudes_raw
        state["refine_params"] = refine_params
        state["refine_cov"] = refine_cov
        state["refine_conjugate"] = self.conjugate_pairs
        # Full settings, so the data bootstrap can rerun the same refinement
        # on every resampled dataset.
        state["refine_settings"] = {
            "conjugate_pairs": self.conjugate_pairs,
            "bounds_real": self.bounds_real,
            "bounds_imag": self.bounds_imag,
            "use_sigma": self.use_sigma,
        }
        return state

    def description(self) -> str:
        return (
            f"RefineStep(obs={self.observable_key}, "
            f"conjugate_pairs={self.conjugate_pairs}, use_sigma={self.use_sigma}, "
            f"bounds_real={self.bounds_real}, bounds_imag={self.bounds_imag})"
        )
