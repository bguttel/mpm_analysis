"""
observables.py
--------------
Universal intermediate data type.  Every data source (experimental JSON folder,
analytical NPZ, Monte Carlo simulation) produces ``ObservableRecord`` objects.
Every analysis step consumes them.

Design goals
~~~~~~~~~~~~
* Only numpy as a dependency.
* A record captures one lambda slice of one observable.
* ``observable_key`` makes the type extensible: adding a new observable type
   requires only a new loader — not a new class.

Typed aliases
~~~~~~~~~~~~~
``SurvivalRecord`` and ``ZRecord`` are just ``ObservableRecord`` instances with
a specific ``observable_key``.  They are provided as factory functions so that
call sites read clearly:

    rec = SurvivalRecord(lambda_val=0.4, t=t_us, signal=P_survival)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ObservableRecord:
    """One lambda slice of one measured/simulated observable.

    Parameters
    ----------
    lambda_val:
        Dimensionless Zeno parameter λ for this slice.
    t:
        Time axis in **microseconds** (always µs — loaders must normalise).
    signal:
        Observable values, same shape as ``t``.  Normalised to [0, 1] for
        survival probability; arbitrary units for other observables.
    observable_key:
        String tag identifying the observable, e.g. ``"survival"``, ``"Z"``,
        ``"click_rate"``.  Used by analysis steps to select the right signal.
    source_file:
        Original filename (for traceability).  May be empty.
    metadata:
        Free-form dict for any additional per-record context (e.g. navg,
        false-positive rates, omega_DG).
    sigma:
        Optional per-point 1-sigma errors, same shape as ``signal``.
        ``None`` (default) → uniform weights in any weighted fit.
    """

    lambda_val: float
    t: np.ndarray
    signal: np.ndarray
    observable_key: str
    source_file: str = ""
    metadata: dict = field(default_factory=dict)
    sigma: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.signal = np.asarray(self.signal, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)



    @property
    def n_points(self) -> int:
        return len(self.t)

    def downsample(self, factor: int) -> "ObservableRecord":
        """Return a new record with every ``factor``-th point kept."""
        return ObservableRecord(
            lambda_val=self.lambda_val,
            t=self.t[::factor],
            signal=self.signal[::factor],
            observable_key=self.observable_key,
            source_file=self.source_file,
            metadata=self.metadata,
            sigma=self.sigma[::factor] if self.sigma is not None else None,
        )

    def time_slice(self, t_min: float | None = None, t_max: float | None = None) -> "ObservableRecord":
        """Return a new record restricted to ``[t_min, t_max]`` µs."""
        mask = np.ones(len(self.t), dtype=bool)
        if t_min is not None:
            mask &= self.t >= t_min
        if t_max is not None:
            mask &= self.t <= t_max
        return ObservableRecord(
            lambda_val=self.lambda_val,
            t=self.t[mask],
            signal=self.signal[mask],
            observable_key=self.observable_key,
            source_file=self.source_file,
            metadata=self.metadata,
            sigma=self.sigma[mask] if self.sigma is not None else None,
        )


# ---------------------------------------------------------------------------
# Multi-observable grouping
# ---------------------------------------------------------------------------

def group_records_by_lambda(
    records: list[ObservableRecord],
    keys: list[str],
) -> list[tuple[float, np.ndarray, np.ndarray, list[ObservableRecord]]]:
    """Align records of several observables on a shared time axis, per lambda.

    Returns ``[(lambda_val, t, Y, recs)]`` sorted by lambda, where ``Y`` has
    shape ``(k, N)`` with rows ordered like ``keys`` and ``recs`` are the
    matching records in the same order.  A lambda missing any key, or whose
    records disagree on the time axis, is dropped with a printed warning —
    this single rule keeps every joint consumer (MPM step, bootstrap,
    plotting) index-aligned.
    """
    by_lambda: dict[float, dict[str, ObservableRecord]] = {}
    for rec in records:
        if rec.observable_key in keys:
            by_lambda.setdefault(rec.lambda_val, {})[rec.observable_key] = rec

    groups = []
    for lam in sorted(by_lambda):
        found = by_lambda[lam]
        missing = [k for k in keys if k not in found]
        if missing:
            print(f"Warning: lambda={lam:.4f} missing observable(s) "
                  f"{missing} — slice dropped.")
            continue
        recs = [found[k] for k in keys]
        t = recs[0].t
        if any(not np.array_equal(r.t, t) for r in recs[1:]):
            print(f"Warning: lambda={lam:.4f} has mismatched time axes "
                  f"across {keys} — slice dropped.")
            continue
        Y = np.vstack([r.signal for r in recs])
        groups.append((lam, t, Y, recs))
    return groups


# ---------------------------------------------------------------------------
# Factory helpers — typed aliases with a clear call site
# ---------------------------------------------------------------------------

def SurvivalRecord(
    lambda_val: float,
    t: np.ndarray,
    signal: np.ndarray,
    source_file: str = "",
    metadata: dict | None = None,
) -> ObservableRecord:
    """Factory: an ObservableRecord with observable_key='survival'."""
    return ObservableRecord(
        lambda_val=lambda_val,
        t=t,
        signal=signal,
        observable_key="survival",
        source_file=source_file,
        metadata=metadata or {},
    )


def ZRecord(
    lambda_val: float,
    t: np.ndarray,
    signal: np.ndarray,
    source_file: str = "",
    metadata: dict | None = None,
) -> ObservableRecord:
    """Factory: an ObservableRecord with observable_key='Z'."""
    return ObservableRecord(
        lambda_val=lambda_val,
        t=t,
        signal=signal,
        observable_key="Z",
        source_file=source_file,
        metadata=metadata or {},
    )
