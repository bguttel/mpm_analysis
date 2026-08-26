"""
critical_point_pipeline.py
--------------------------
High-level pipeline: load an analysis JSON → fit sqrt model → save back.

Replaces ``CriticalPointsAnalyzer`` from ``Code/critical_points_analysis.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mpm_analysis.data_types.analysis_result import ZenoAnalysisResult, SqrtFitResult
from mpm_analysis.analysis.critical_point import (
    fit_sqrt_to_eigenvalues,
    calibrate_window,
    WindowCalibrationResult,
)


class CriticalPointPipeline:
    """Fit the sqrt model to the imaginary pole splitting.

    Parameters
    ----------
    result:
        A loaded ``ZenoAnalysisResult``.
    source_path:
        If the result was loaded from a file, pass the path here so that
        ``save()`` knows where to write back.
    lambda_max:
        Optional upper limit on λ — points above this are excluded from the fit.
    """

    def __init__(
        self,
        result: ZenoAnalysisResult,
        *,
        source_path: Path | None = None,
        lambda_max: float | None = None,
    ) -> None:
        self.result = result
        self.source_path = source_path

        if lambda_max is not None:
            mask = result.lambda_values <= lambda_max
            self._poles = result.poles[mask]
            self._lambdas = result.lambda_values[mask]
        else:
            self._poles = result.poles
            self._lambdas = result.lambda_values

        self._last_fit: SqrtFitResult | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        lambda_max: float | None = None,
    ) -> "CriticalPointPipeline":
        """Load a result JSON and return a ready-to-use pipeline.

        Parameters
        ----------
        path:
            Path to an ``mpm_analysis_results_*.json`` file.
        lambda_max:
            Optional upper λ cutoff for the fit.
        """
        from mpm_analysis.io.analysis_json import load_analysis_result
        path = Path(path)
        result = load_analysis_result(path)
        return cls(result, source_path=path, lambda_max=lambda_max)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        lambda_c_guess: float | None = None,
        *,
        preset: str | None = None,
        window_left: int = 50,
        window_right: int = 15,
        a_guess: float | None = None,
        b_guess: float | None = None,
        fit_real_part: bool = True,
        soft_bounds: bool = False,
        lambda_c_bounds: tuple[float, float] = (1.0, 1.5),
    ) -> SqrtFitResult:
        """Fit the sqrt model and return the result.

        Parameters
        ----------
        lambda_c_guess:
            Initial guess for λ_c.
        preset:
            Named preset key from ``critical_point_guesses`` (e.g.
            ``"ens_avg_experimental"``).  If given, overrides ``lambda_c_guess``,
            ``a_guess``, and ``b_guess`` unless those are also explicitly provided.
        window_left, window_right:
            Number of lambda points to include left/right of ``lambda_c_guess``.
        a_guess, b_guess:
            Initial parameter guesses.  If ``None`` and no preset is given,
            defaults to 1.0 / 0.0.
        fit_real_part:
            Whether to also compute the real-splitting fit curve.
        soft_bounds:
            Soft guess mode — only λ_c is bounded (``lambda_c_bounds``),
            ``a``/``b`` are essentially free.  See
            ``fit_sqrt_to_eigenvalues``.
        lambda_c_bounds:
            λ_c bounds used when ``soft_bounds=True``.

        Returns
        -------
        SqrtFitResult

        Raises
        ------
        ValueError
            If no splitting data is available.
        """
        from mpm_analysis.analysis.critical_point_guesses import get_guess

        if not self._poles.has_splitting:
            raise ValueError(
                "No splitting data in this result.  "
                "Re-run the MPM pipeline with a BootstrapStep that computes splittings."
            )

        # Resolve guesses from preset (explicit args take priority)
        if preset is not None:
            defaults = get_guess(preset)
            if lambda_c_guess is None:
                lambda_c_guess = defaults["lambda_c"]
            if a_guess is None:
                a_guess = defaults["a"]
            if b_guess is None:
                b_guess = defaults["b"]

        # Fall back to neutral defaults if still None
        if a_guess is None:
            a_guess = 1.0
        if b_guess is None:
            b_guess = 0.0

        if lambda_c_guess is None:
            lambda_c_guess = float(np.median(self._lambdas))

        omega_s = self.result.omega_s

        self._last_fit = fit_sqrt_to_eigenvalues(
            self._poles,
            self._lambdas,
            lambda_c_guess,
            window_left=window_left,
            window_right=window_right,
            a_guess=a_guess,
            b_guess=b_guess,
            fit_real_part=fit_real_part,
            omega_s=omega_s,
            soft_bounds=soft_bounds,
            lambda_c_bounds=lambda_c_bounds,
        )

        print(
            f"Sqrt fit complete: lam_c = {self._last_fit.lambda_c:.4f} +/- "
            f"{self._last_fit.lambda_c_err:.4f},  chi2_red = {self._last_fit.chi2_reduced:.3f}"
        )
        return self._last_fit

    def calibrate(
        self,
        lambda_c_guess: float | None = None,
        left_range: tuple[int, int] = (20, 100),
        right_range: tuple[int, int] = (5, 30),
        a_guess: float | None = None,
        b_guess: float | None = None,
        *,
        preset: str | None = None,
    ) -> WindowCalibrationResult:
        """Sweep window sizes and return a calibration result.

        Useful for diagnosing fit stability.
        """
        from mpm_analysis.analysis.critical_point_guesses import get_guess

        if preset is not None:
            defaults = get_guess(preset)
            if lambda_c_guess is None:
                lambda_c_guess = defaults["lambda_c"]
            if a_guess is None:
                a_guess = defaults["a"]
            if b_guess is None:
                b_guess = defaults["b"]

        if a_guess is None:
            a_guess = 1.0
        if b_guess is None:
            b_guess = 0.0

        if lambda_c_guess is None:
            lambda_c_guess = float(np.median(self._lambdas))

        return calibrate_window(
            self._poles,
            self._lambdas,
            lambda_c_guess,
            left_range=left_range,
            right_range=right_range,
            a_guess=a_guess,
            b_guess=b_guess,
            omega_s=self.result.omega_s,
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path | None = None,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Save the fit result back into the source JSON.

        Parameters
        ----------
        path:
            Target JSON file.  Defaults to ``self.source_path``.
        overwrite:
            If ``False`` and the key already exists, raise ``ValueError``.

        Returns
        -------
        Path
            Path to the saved file.
        """
        if self._last_fit is None:
            raise RuntimeError("Call fit() before save().")

        target = Path(path) if path is not None else self.source_path
        if target is None:
            raise ValueError(
                "No path to save to. Pass a path or use from_json() to load from a file."
            )

        from mpm_analysis.io.analysis_json import augment_json_with_sqrt_fit
        return augment_json_with_sqrt_fit(target, self._last_fit, overwrite=overwrite)
