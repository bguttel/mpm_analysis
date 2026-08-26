"""
mpm_step.py
-----------
Pipeline step: run the Matrix Pencil Method on each lambda slice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.data_types.observables import ObservableRecord


class MPMStep(AnalysisStep):
    """Extract poles from each lambda slice using the Matrix Pencil Method.

    After this step, ``state`` contains:
        ``"poles_raw"``   : list[np.ndarray]   complex poles per slice
        ``"L_used"``      : list[int]           pencil parameter used per slice
        ``"lambdas"``     : list[float]         lambda values in order
        ``"observable_key"`` : str              single key, or keys joined by '+'
        ``"observable_keys"`` : list[str]       keys actually analysed
        ``"amplitudes_raw"`` : list[np.ndarray] complex amplitudes per slice,
                                                shape (order,) or (k, order)

    Parameters
    ----------
    order:
        Model order (number of poles).
    L:
        Pencil parameter override.  ``None`` → auto (N//3).
    observable_key:
        Which observable(s) to analyse.  A single key runs the classic 1-D
        pencil; a list of keys runs the joint (multi-signal) pencil — one
        common pole set per lambda, amplitudes solved per observable.
    plot_svd:
        If ``True``, plot the Hankel SVD singular values vs λ after processing.
        Use this as a calibration step to visually select the model order:
        look for a gap between line ``order`` and ``order+1``.
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
        observable_key: str | list[str] | None = None,
        *,
        plot_svd: bool = False,
        n_sv: int = 10,
        save_dir: Path | str | None = None,
    ) -> None:
        self.order = order
        self.L = L
        self.observable_key = observable_key  # None → derived from first record
        self.plot_svd = plot_svd
        self.n_sv = n_sv
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def process(
        self,
        records: list[ObservableRecord],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        from mpm_analysis.analysis.matrix_pencil import (
            matrix_pencil_method,
            solve_linear_amplitudes,
        )
        from mpm_analysis.analysis.pole_sorting import (
            sort_poles_canonical,
            enforce_pole_count,
            track_branches_across_lambda,
        )
        from mpm_analysis.data_types.observables import group_records_by_lambda

        # Normalise observable_key to a list of keys
        if self.observable_key is None:
            if not records:
                raise ValueError("MPMStep: records is empty; cannot derive observable_key.")
            keys = [records[0].observable_key]
        elif isinstance(self.observable_key, str):
            keys = [self.observable_key]
        else:
            keys = list(self.observable_key)
        obs_key = keys[0] if len(keys) == 1 else "+".join(keys)

        if len(keys) == 1:
            slices = sorted(
                [r for r in records if r.observable_key == keys[0]],
                key=lambda r: r.lambda_val,
            )
            groups = [(r.lambda_val, r.t, r.signal, [r]) for r in slices]
        else:
            groups = group_records_by_lambda(records, keys)

        if not groups:
            raise ValueError(
                f"MPMStep: no records found with observable_key(s) {keys}. "
                f"Available keys: {sorted({r.observable_key for r in records})}."
            )

        poles_raw = []
        amplitudes_raw = []
        L_used = []
        lambdas = []
        sv_list = []

        for lam, t, Y, _recs in groups:
            try:
                poles, sv, L_u = matrix_pencil_method(t, Y, self.order, L=self.L)
                poles = sort_poles_canonical(enforce_pole_count(poles, self.order))
                amps = solve_linear_amplitudes(t, Y, poles)
                poles_raw.append(poles)
                amplitudes_raw.append(amps)
                L_used.append(L_u)
                lambdas.append(lam)
                sv_list.append(sv)
            except Exception as exc:
                print(f"Warning: MPMStep failed for lam={lam:.3f}: {exc}")
                poles_raw.append(np.full(self.order, np.nan, dtype=complex))
                amp_shape = self.order if len(keys) == 1 else (len(keys), self.order)
                amplitudes_raw.append(np.full(amp_shape, np.nan, dtype=complex))
                L_used.append(0)
                lambdas.append(lam)
                sv_list.append(np.array([np.nan]))

        if self.plot_svd and sv_list:
            import matplotlib
            import matplotlib.pyplot as plt
            from mpm_analysis.plotting.exploratory import plot_svd_singular_values
            from mpm_analysis.utils.windows_paths import win_long_path

            valid = [(lam, sv) for lam, sv in zip(lambdas, sv_list)
                     if not np.all(np.isnan(sv))]
            if valid:
                lams_v, svs_v = zip(*valid)
                ax = plot_svd_singular_values(
                    np.array(lams_v), list(svs_v),
                    n_sv=self.n_sv,
                    title=f"MPM Hankel SVD — model order calibration (order={self.order})",
                )
                fig = ax.get_figure()
                fig.tight_layout()
                if self.save_dir is not None:
                    fname = self.save_dir / f"svd_{obs_key}.png"
                    # win_long_path: deep lab folders exceed MAX_PATH.
                    fig.savefig(win_long_path(fname), bbox_inches="tight")
                    print(f"  SVD figure saved: {fname.name}")
                if matplotlib.get_backend().lower() != "agg":
                    plt.show(block=False)
                else:
                    plt.close(fig)      # headless: the figure is already saved

        # Branch tracking: Hungarian-match each slice's poles to the previous
        # slice so index i follows one continuous branch across lambda (no
        # label switching near the EP).  Amplitudes are permuted alongside.
        poles_raw, perms = track_branches_across_lambda(poles_raw, return_indices=True)
        amplitudes_raw = [
            amps[..., idx] for amps, idx in zip(amplitudes_raw, perms)
        ]

        state["poles_raw"] = poles_raw
        state["amplitudes_raw"] = amplitudes_raw
        state["L_used"] = L_used
        state["lambdas"] = lambdas
        state["observable_key"] = obs_key
        state["observable_keys"] = keys
        return state

    def description(self) -> str:
        return (
            f"MPMStep(order={self.order}, L={self.L}, "
            f"obs='{self.observable_key}', plot_svd={self.plot_svd})"
        )
