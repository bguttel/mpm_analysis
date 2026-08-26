"""
pole_samples.py
---------------
Container for raw bootstrap pole samples, exposing the full
(n_lambda, n_boot, order) complex array for downstream analysis.

The median/uncertainty view lives in ``PolesResult``; this container stores
the raw draws so callers can apply any aggregation they like (e.g. ksi
crossing, histogram inspection, custom percentiles).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoleSamples:
    """Raw complex pole samples from the bootstrap.

    Attributes
    ----------
    samples:
        Complex array of shape ``(n_lambda, n_boot, order)``.
        Within each draw, poles are Hungarian-matched to the clean
        (branch-tracked) poles of that lambda, so a fixed column index means
        one continuous branch — and draw ``b`` across all lambdas is one
        complete synthetic realization of the scan.
    observable_key:
        The observable key used to produce these samples.
    lambda_values:
        1-D array of λ values, shape ``(n_lambda,)``.
    param_samples:
        Optional float array of shape ``(n_lambda, n_boot, n_par)`` with the
        quadratic pole parameters ``[r..., c1, c0, ...]`` per draw (see
        ``pack_pole_params``).  The discriminant ``c1**2 - 4*c0`` is the
        preferred EP diagnostic near a coalescence.
    """

    samples: np.ndarray        # complex (n_lambda, n_boot, order)
    observable_key: str
    lambda_values: np.ndarray  # float  (n_lambda,)
    param_samples: np.ndarray | None = None  # float (n_lambda, n_boot, n_par)

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=complex)
        self.lambda_values = np.asarray(self.lambda_values, dtype=float)
        if self.samples.ndim != 3:
            raise ValueError(
                f"PoleSamples.samples must be 3-D (n_lambda, n_boot, order), "
                f"got shape {self.samples.shape}."
            )
        if len(self.lambda_values) != self.samples.shape[0]:
            raise ValueError(
                f"lambda_values length {len(self.lambda_values)} does not match "
                f"samples.shape[0] = {self.samples.shape[0]}."
            )
        if self.param_samples is not None:
            self.param_samples = np.asarray(self.param_samples, dtype=float)
            if self.param_samples.shape[:2] != self.samples.shape[:2]:
                raise ValueError(
                    f"param_samples shape {self.param_samples.shape} does not "
                    f"match samples shape {self.samples.shape} in "
                    "(n_lambda, n_boot)."
                )

    @property
    def n_lambda(self) -> int:
        return self.samples.shape[0]

    @property
    def n_boot(self) -> int:
        return self.samples.shape[1]

    @property
    def order(self) -> int:
        return self.samples.shape[2]

    def get_lambda(self, idx: int) -> np.ndarray:
        """Return the ``(n_boot, order)`` sample slice for lambda index ``idx``."""
        return self.samples[idx]
