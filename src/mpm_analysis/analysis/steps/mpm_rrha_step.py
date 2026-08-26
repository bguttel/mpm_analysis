"""
mpm_rrha_step.py
----------------
Pipeline step: extract poles using the RRHA-enhanced Matrix Pencil Method.

This is a drop-in replacement for ``MPMStep`` that writes the same state keys
so it can be combined with ``BootstrapStep`` unchanged::

    steps = [MPMRRHAStep(order=3), BootstrapStep(n_boot=2000)]
    result = MPMPipeline.from_records(records, steps=steps).run()

See ``matrix_pencil_rrha.mpm_rrha`` for algorithm details.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.data_types.observables import ObservableRecord


class MPMRRHAStep(AnalysisStep):
    """Extract poles using RRHA-enhanced MPM on each lambda slice.

    After this step, ``state`` contains:
        ``"poles_raw"``      : list[np.ndarray]   complex poles per slice
        ``"L_used"``         : list[int]           pencil parameter used per slice
        ``"lambdas"``        : list[float]         lambda values in order
        ``"observable_key"`` : str

    Parameters
    ----------
    order:
        Model order (number of poles).
    L:
        Pencil parameter override.  ``None`` → auto (N//3).
    observable_key:
        Which observable to analyse.  ``None`` → derived from first record.
    max_iter:
        Maximum RRHA iterations per slice.  Raises on non-convergence.
    rtol:
        RRHA stop criterion tolerance.
    plot_svd:
        If ``True``, plot the Hankel SVD singular values vs λ after processing.
        The singular values shown come from the **first RRHA iteration** (i.e. the
        SVD of the original Hankel matrix), which is the most informative snapshot
        for choosing model order before noise-suppression kicks in.
    n_sv:
        Number of singular values to display when ``plot_svd=True``.
    save_dir:
        Directory to save the SVD figure as PNG when ``plot_svd=True``.
        ``None`` → do not save.
    """

    def __init__(
        self,
        order: int = 3,
        L: int | None = None,
        observable_key: str | None = None,
        *,
        max_iter: int = 10,
        rtol: float = 1e-3,
        plot_svd: bool = False,
        n_sv: int = 10,
        save_dir: Path | str | None = None,
    ) -> None:
        self.order = order
        self.L = L
        self.observable_key = observable_key
        self.max_iter = max_iter
        self.rtol = rtol
        self.plot_svd = plot_svd
        self.n_sv = n_sv
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from mpm_analysis.analysis.matrix_pencil_rrha import mpm_rrha, build_hankel_rrha
        from scipy.linalg import svd as _svd
        print("Running MPM+RRHA on each lambda slice...")

        obs_key = self.observable_key
        if obs_key is None:
            if not records:
                raise ValueError(
                    "MPMRRHAStep: records is empty; cannot derive observable_key."
                )
            obs_key = records[0].observable_key

        slices = sorted(
            [r for r in records if r.observable_key == obs_key],
            key=lambda r: r.lambda_val,
        )

        if not slices:
            raise ValueError(
                f"MPMRRHAStep: no records found with observable_key='{obs_key}'. "
                f"Available keys: {sorted({r.observable_key for r in records})}."
            )

        poles_raw = []
        L_used = []
        lambdas = []
        sv_list = []  # singular values from first RRHA iteration (pre-reduction)

        for rec in slices:
            # Capture singular values of the raw Hankel matrix (first RRHA step)
            if self.plot_svd:
                try:
                    N = len(rec.t)
                    L_init = self.L if self.L is not None else N // 3
                    L_init = max(self.order, min(L_init, N - self.order - 1))
                    H_init = build_hankel_rrha(rec.signal, L_init)
                    sv_init = _svd(H_init, compute_uv=False)
                    sv_list.append(sv_init)
                except Exception:
                    sv_list.append(np.array([np.nan]))

            try:
                poles, _, L_u = mpm_rrha(
                    rec.t,
                    rec.signal,
                    self.order,
                    L=self.L,
                    max_iter=self.max_iter,
                    rtol=self.rtol,
                )
                poles_raw.append(poles)
                L_used.append(L_u)
                lambdas.append(rec.lambda_val)
            except Exception as exc:
                print(
                    f"Warning: MPMRRHAStep failed for lam={rec.lambda_val:.3f}: {exc}"
                )
                poles_raw.append(np.full(self.order, np.nan, dtype=complex))
                L_used.append(0)
                lambdas.append(rec.lambda_val)

        if self.plot_svd and sv_list:
            import matplotlib.pyplot as plt
            from mpm_analysis.plotting.exploratory import plot_svd_singular_values

            valid = [(lam, sv) for lam, sv in zip(lambdas, sv_list)
                     if not np.all(np.isnan(sv))]
            if valid:
                lams_v, svs_v = zip(*valid)
                ax = plot_svd_singular_values(
                    np.array(lams_v), list(svs_v),
                    n_sv=self.n_sv,
                    title=(
                        f"RRHA Hankel SVD (first iteration) — "
                        f"model order calibration (order={self.order})"
                    ),
                )
                fig = ax.get_figure()
                fig.tight_layout()
                if self.save_dir is not None:
                    fname = self.save_dir / f"svd_rrha_{obs_key}.png"
                    fig.savefig(fname, bbox_inches="tight")
                    print(f"  SVD figure saved: {fname.name}")
                plt.show(block=False)

        state["poles_raw"] = poles_raw
        state["L_used"] = L_used
        state["lambdas"] = lambdas
        state["observable_key"] = obs_key
        return state

    def description(self) -> str:
        return (
            f"MPMRRHAStep(order={self.order}, L={self.L}, "
            f"max_iter={self.max_iter}, obs='{self.observable_key}', "
            f"plot_svd={self.plot_svd})"
        )
