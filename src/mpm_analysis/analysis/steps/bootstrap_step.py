"""
bootstrap_step.py
-----------------
Pipeline step: run bootstrap uncertainty quantification.
"""
from __future__ import annotations

from typing import Any

from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.data_types.observables import ObservableRecord


class BootstrapStep(AnalysisStep):
    """Run bootstrap and aggregate results into a ``PolesResult``.

    Reads ``state["observable_key"]`` set by ``MPMStep``.

    After this step, ``state`` contains:
        ``"boot_results"``  : list[_LambdaBootstrapResult]
        ``"poles_result"``  : PolesResult

    Parameters
    ----------
    n_boot:
        Number of bootstrap resamples per lambda slice.
    order:
        Model order.  If ``None``, uses ``state["poles_raw"][0].shape[0]``.
    seed:
        RNG seed for reproducibility.  ``None`` → truly random (OS entropy).
    mode:
        ``"data"`` — data-resampling bootstrap: resample the raw measurement
        statistics (multinomial / binomial, from
        ``record.metadata["bootstrap_stats"]``), rerun stacked MPM + LSQ
        refinement per draw.  The statistically correct default.
        ``"residual"`` — residual resampling + MPM refit per draw (legacy).
        ``"parametric"`` — Monte-Carlo covariance propagation: Gaussian draws
        of the refinement parameter vector from its least-squares covariance
        (requires a ``RefineStep`` before this step; NOT a bootstrap).
        All modes Hungarian-match draws to the clean poles and produce the
        same kind of samples; central values stay the measured best fit.
    n_jobs:
        Parallel workers for ``mode="data"`` (process pool across lambdas).
        ``-1`` → all cores.
    """

    def __init__(
        self,
        n_boot: int = 2000,
        order: int | None = None,
        seed: int | None = None,
        mode: str = "residual",
        n_jobs: int = -1,
    ) -> None:
        if mode not in ("data", "residual", "parametric"):
            raise ValueError(f"BootstrapStep: unknown mode '{mode}'.")
        self.n_boot = n_boot
        self.order = order
        self.seed = seed
        self.mode = mode
        self.n_jobs = n_jobs

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        import numpy as np
        from mpm_analysis.analysis.bootstrap import run_bootstrap, aggregate_bootstrap

        obs_key = state.get("observable_key")
        if obs_key is None:
            if not records:
                raise ValueError(
                    "BootstrapStep: records is empty and state has no 'observable_key'."
                )
            obs_key = records[0].observable_key
        # Joint MPM: the list of keys drives the bootstrap; obs_key (a plain
        # string, '+'-joined for multiple keys) labels the aggregated result.
        obs_keys = state.get("observable_keys") or obs_key

        order = self.order
        if order is None:
            raw = state.get("poles_raw")
            order = int(raw[0].shape[0]) if raw else 3

        rng = np.random.default_rng(self.seed)

        if self.mode == "parametric":
            boot_results = self._parametric_draws(state, order, rng)
        elif self.mode == "data":
            from mpm_analysis.analysis.bootstrap import run_data_bootstrap
            boot_results = run_data_bootstrap(
                records,
                order=order,
                n_boot=self.n_boot,
                observable_key=obs_keys,
                L_used=state.get("L_used"),
                seed=self.seed,
                clean_poles=state.get("poles_raw"),
                refine_settings=state.get("refine_settings"),
                n_jobs=self.n_jobs,
            )
        else:
            # Pass the per-slice pencil parameters, not a single representative:
            # slices may differ in length, and another slice's L can be invalid
            # (or negative-dimensioned) here.
            boot_results = run_bootstrap(
                records,
                order=order,
                n_boot=self.n_boot,
                observable_key=obs_keys,
                L_used=state.get("L_used"),
                rng=rng,
                clean_poles=state.get("poles_raw"),
            )
        # Central values = fit of the ORIGINAL measured dataset; draws only
        # set the uncertainty intervals.
        poles_result = aggregate_bootstrap(
            boot_results,
            observable_key=obs_key,
            center_poles=state.get("poles_raw"),
        )

        state["boot_results"] = boot_results
        state["poles_result"] = poles_result
        return state

    def _parametric_draws(self, state: dict[str, Any], order: int, rng):
        """Gaussian parameter-space draws from the RefineStep covariance."""
        import numpy as np
        from mpm_analysis.analysis.bootstrap import _LambdaBootstrapResult
        from mpm_analysis.analysis.matrix_pencil import (
            unpack_pole_params,
            pack_pole_params,
        )
        from mpm_analysis.analysis.pole_sorting import match_poles_hungarian

        params = state.get("refine_params")
        covs = state.get("refine_cov")
        if params is None or covs is None:
            raise ValueError(
                "BootstrapStep(mode='parametric') requires a RefineStep before "
                "it — state['refine_params'] / state['refine_cov'] not found."
            )
        conjugate = bool(state.get("refine_conjugate", False))
        lambdas = state.get("lambdas", [])
        clean_poles = state.get("poles_raw", [])

        results: list[_LambdaBootstrapResult] = []
        for i, lam in enumerate(lambdas):
            boot_real = np.full((self.n_boot, order), np.nan, dtype=float)
            boot_imag = np.full((self.n_boot, order), np.nan, dtype=float)
            boot_params = np.full((self.n_boot, order), np.nan, dtype=float)
            p_opt = params[i] if i < len(params) else None
            cov = covs[i] if i < len(covs) else None
            if p_opt is None or cov is None:
                print(f"Warning: covariance propagation skipped for lam={lam:.3f} "
                      f"(no refinement fit).")
                results.append(_LambdaBootstrapResult(lam, boot_real, boot_imag,
                                                      boot_params))
                continue
            try:
                cov_sym = 0.5 * (cov + cov.T)
                draws = rng.multivariate_normal(
                    p_opt, cov_sym, size=self.n_boot, method="svd"
                )
                reference = np.asarray(clean_poles[i], dtype=complex)
                M = len(p_opt)
                for b in range(self.n_boot):
                    if conjugate:
                        poles_b = unpack_pole_params(draws[b], order)
                        boot_params[b] = draws[b]
                    else:
                        poles_b = draws[b][: M // 2] + 1j * draws[b][M // 2:]
                        if np.all(np.isfinite(poles_b)):
                            boot_params[b] = pack_pole_params(poles_b)
                    poles_b = match_poles_hungarian(poles_b, reference)
                    boot_real[b] = poles_b.real
                    boot_imag[b] = poles_b.imag
            except Exception as exc:
                print(f"Warning: covariance propagation failed for lam={lam:.3f}: {exc}")
            results.append(_LambdaBootstrapResult(lam, boot_real, boot_imag,
                                                  boot_params))
        return results

    def description(self) -> str:
        return (f"BootstrapStep(n_boot={self.n_boot}, seed={self.seed}, "
                f"mode={self.mode}, n_jobs={self.n_jobs})")
