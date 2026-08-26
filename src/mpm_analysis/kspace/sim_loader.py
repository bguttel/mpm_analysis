"""
sim_loader.py
-------------
SimClickKSpaceLoader — loads the multi-lambda simulation NPZ produced by
``misc/generate_click_record_raw.py`` directly into ``KSpaceRecord`` lists.

The NPZ already stores ``pkt_{i}`` = P(k,t), shape (n_k, n_steps) complex64,
exactly the layout ``KSpaceRecord.signal`` expects — no DFT/transpose needed,
unlike the JSON/compressed loaders.  This is the ``source="simulated"``
k-space loader.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mpm_analysis.kspace.data_types import KSpaceRecord

_SCALAR_KEYS = (
    "n_trials", "n_steps", "dt", "substeps", "dt_fine", "T_total",
    "Omega", "T1", "T2", "tau_B", "fp", "fn",
    "gamma_1", "gamma_2", "gamma_deph", "gamma_1B",
    "epsilon", "fp_bright", "fn_bright", "fp_dark", "fn_dark",
)


class SimClickKSpaceLoader:
    """Load a simulation NPZ (from ``generate_click_record_raw.py``) into
    ``KSpaceRecord`` lists.

    Usage
    -----
    ::

        records = SimClickKSpaceLoader().load(path_to_npz)
        # records[i].signal.shape == (n_k, n_time)
    """

    def load(
        self,
        path: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        t_max: float | None = None,
        k_index_range: tuple[int | None, int | None] | None = None,
    ) -> list[KSpaceRecord]:
        path = Path(path)
        npz = np.load(path, allow_pickle=False)

        lambda_values = npz["lambda_values"]
        t_values = npz["t_values"]
        k_values = npz["k_values"]
        meta_scalars = self._extract_scalars(npz, path.name)
        omega = meta_scalars.get("Omega")

        keys = set(npz.files)
        records: list[KSpaceRecord] = []
        for i, lam in enumerate(lambda_values):
            if f"pkt_{i}" not in keys:
                continue
            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= float(lam) <= hi):
                    continue

            t = t_values
            k_array = k_values
            signal = npz[f"pkt_{i}"]   # (n_k, n_time) complex64, already P(k,t)

            if k_index_range is not None:
                sl = slice(*k_index_range)
                k_array = k_array[sl]
                signal = signal[sl, :]
                if k_array.size == 0:
                    raise ValueError(
                        f"k_index_range={k_index_range} produced an empty k axis."
                    )

            if t_max is not None:
                mask = t <= t_max
                t = t[mask]
                signal = signal[:, mask]

            meta = dict(meta_scalars)
            if omega is not None:
                meta["omega_DG"] = float(omega)

            records.append(
                KSpaceRecord(
                    lambda_val=float(lam),
                    t=t,
                    k_array=k_array,
                    signal=signal,
                    source_file=path.name,
                    metadata=meta,
                )
            )

        records.sort(key=lambda r: r.lambda_val)
        return records

    @staticmethod
    def _extract_scalars(npz: np.lib.npyio.NpzFile, source: str) -> dict:
        meta = {"source_file": source}
        for k in _SCALAR_KEYS:
            if k in npz:
                meta[k] = npz[k].item()
        return meta
