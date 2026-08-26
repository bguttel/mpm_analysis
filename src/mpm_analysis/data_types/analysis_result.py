"""
analysis_result.py
------------------
Top-level result containers.  Maps 1-to-1 to the existing JSON output schema so
that ``io/analysis_json.py`` is pure (de)serialisation with no business logic.

JSON schema reference
~~~~~~~~~~~~~~~~~~~~~
The ``mpm_analysis_results_*.json`` files produced by ``first_transition_analysis.py``
and ``object_oriented_poles.py`` have this top-level structure:

    {
      "metadata":  {...},
      "parameters": {..., "omega_DG": float, ...},
      "lambda_values": [...],
      "poles": {
        "decay_rates": {"median": [...], "lower_uncertainty": [...], "upper_uncertainty": [...]},
        "frequencies": {...},
      },
      "critical_point_sqrt_fit": {   // optional, added by CriticalPointsAnalyzer
        "fit": {
          "results": {"lambda_c": float, "a": float, ...},
          "window":  {...},
          "data_in_window": {"lambda": [...], "imag_split": [...], "imag_sigma": [...]},
          "curves":  {"lambda_plot": [...], "imag_fit_curve": [...], ...},
        }
      }
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np

from mpm_analysis.data_types.poles import PolesResult

if TYPE_CHECKING:
    pass  # no additional TYPE_CHECKING imports needed here


# ---------------------------------------------------------------------------
# Sqrt-fit result
# ---------------------------------------------------------------------------

@dataclass
class SqrtFitResult:
    """Results of fitting Im(Δe₁₂) ~ a√|λ−λ_c| + b(λ−λ_c)^1.5 near the critical point.

    Fields mirror the ``critical_point_sqrt_fit.fit`` block in the output JSON.
    """
    # Fit parameters
    lambda_c: float
    a: float
    b: float
    lambda_c_err: float
    a_err: float
    b_err: float

    # Fit quality
    chi2_reduced: float
    rmse: float
    r_squared: float

    # Window description (maps to the "window" sub-block in JSON)
    window_lambda_min: float
    window_lambda_max: float
    window_n_points: int
    window_left_points: int
    window_right_points: int

    # Data used in the fit
    lambda_data: np.ndarray      # lambda points in window
    imag_split_data: np.ndarray  # imaginary splitting data
    imag_split_err: np.ndarray   # uncertainties used in fit
    real_split_data: np.ndarray | None = None
    real_split_err: np.ndarray | None = None

    # Smooth fit curves for plotting
    lambda_plot: np.ndarray | None = None
    imag_fit_curve: np.ndarray | None = None
    real_fit_curve: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.lambda_data = np.asarray(self.lambda_data, dtype=float)
        self.imag_split_data = np.asarray(self.imag_split_data, dtype=float)
        self.imag_split_err = np.asarray(self.imag_split_err, dtype=float)


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Output of BenchmarkPipeline: bias and variance of the analysis under noise.

    Attributes
    ----------
    ground_truth_lambda_c:
        The true critical point from the clean simulation.
    extracted_lambda_c_samples:
        Lambda_c estimated in each trial.
    noise_description:
        Dict describing the noise model(s) used (from ``NoiseModel.description()``).
    pole_bias:
        Per-lambda, per-pole bias (extracted median − ground truth).
        Shape (n_lambda, order).
    pole_variance:
        Per-lambda, per-pole variance across trials.  Shape (n_lambda, order).
    """
    ground_truth_lambda_c: float
    extracted_lambda_c_samples: np.ndarray   # (n_trials,)
    noise_description: dict
    pole_bias: np.ndarray | None = None      # (n_lambda, order)
    pole_variance: np.ndarray | None = None  # (n_lambda, order)

    @property
    def lambda_c_bias(self) -> float:
        return float(np.mean(self.extracted_lambda_c_samples) - self.ground_truth_lambda_c)

    @property
    def lambda_c_std(self) -> float:
        return float(np.std(self.extracted_lambda_c_samples))

    @property
    def n_trials(self) -> int:
        return len(self.extracted_lambda_c_samples)


# ---------------------------------------------------------------------------
# Poles-only benchmark result (no critical-point fit)
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPolesResult:
    """Output of ``BenchmarkPipeline.run(fit_critical_point=False)``.

    Stores per-trial ``PolesResult`` objects so the caller can inspect the
    raw pole distributions without committing to a critical-point fit.

    Attributes
    ----------
    trial_poles:
        One ``PolesResult`` per successful trial.
    noise_description:
        Dict describing the noise models used.
    lambda_values:
        Lambda axis (from the first successful trial, or None).
    """
    trial_poles: "list[PolesResult]"
    noise_description: dict
    lambda_values: np.ndarray | None = None

    @property
    def n_trials(self) -> int:
        return len(self.trial_poles)

    @property
    def n_successful(self) -> int:
        return len(self.trial_poles)


# ---------------------------------------------------------------------------
# Top-level analysis result
# ---------------------------------------------------------------------------

@dataclass
class ZenoAnalysisResult:
    """Complete output of the MPM bootstrap analysis pipeline.

    This is the in-memory representation of an ``mpm_analysis_results_*.json``
    file.  Loading and saving are handled by ``io/analysis_json.py``.

    Attributes
    ----------
    metadata:
        Provenance info (timestamp, analysis_type, source_folder, …).
    parameters:
        Experiment / analysis parameters (omega_DG, pole_order, n_bootstrap, …).
    lambda_values:
        1-D array of λ values at which poles were extracted.
    poles:
        Bootstrap-aggregated poles across the lambda scan.
    critical_point_sqrt_fit:
        Optional; present after ``CriticalPointPipeline`` has been run.
    benchmark_results:
        Optional; list of ``BenchmarkResult`` from ``BenchmarkPipeline``.
    extras:
        Optional dict of additional analysis data harvested from the pipeline
        state (refine_params, refine_cov, amplitudes_raw, L_used, …).
        Serialised to the ``analysis_extras`` JSON block when saving with
        ``full=True``.
    """
    metadata: dict
    parameters: dict
    lambda_values: np.ndarray
    poles: PolesResult
    critical_point_sqrt_fit: SqrtFitResult | None = None
    benchmark_results: list[BenchmarkResult] = field(default_factory=list)
    extras: dict | None = None

    def __post_init__(self) -> None:
        self.lambda_values = np.asarray(self.lambda_values, dtype=float)

    # ------------------------------------------------------------------
    # Convenience properties

    @property
    def omega_s(self) -> float:
        """Return omega_DG from parameters (normalization factor)."""
        import warnings
        val = self.parameters.get("omega_DG")
        if val is None:
            warnings.warn(
                "omega_DG not found in result.parameters — normalization falls back to 1.0 "
                "(poles will be in raw units 1/µs). Check that Omega_DG is present in the "
                "experiment JSON files and that the pipeline ran successfully.",
                UserWarning,
                stacklevel=2,
            )
            return 1.0
        return float(val)

    @property
    def n_lambda(self) -> int:
        return len(self.lambda_values)

    @property
    def analysis_type(self) -> str:
        return self.metadata.get("analysis_type", "unknown")
