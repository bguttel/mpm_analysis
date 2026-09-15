"""
benchmark_pipeline.py
---------------------
Robustness and bias testing pipeline.

Runs a known-ground-truth simulation through a sequence of noise models and
analysis steps, then measures bias (extracted − true) and variance of the
critical-point estimate and pole spectrum.

Design
~~~~~~
* Modular noise: pass any ``NoiseModel`` or ``CompositeNoise``.
* Modular analysis: pass any list of ``AnalysisStep`` objects.
* Multiple trials per noise instance for variance estimation.
* ``sweep_noise`` lets you vary a single noise parameter while fixing others.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mpm_analysis.data_types.analysis_result import (
    ZenoAnalysisResult,
    BenchmarkResult,
    BenchmarkPolesResult,
)
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.analysis.steps.mpm_step import MPMStep
from mpm_analysis.analysis.steps.bootstrap_step import BootstrapStep
from mpm_analysis.simulation.noise.base import NoiseModel

if TYPE_CHECKING:
    from mpm_analysis.data_types.analysis_result import SqrtFitResult

class BenchmarkPipeline:
    """Benchmark the MPM + bootstrap pipeline against injected noise.

    Parameters
    ----------
    clean_records:
        Ground-truth ``ObservableRecord`` objects (from a simulator or
        from ``io/simulation.load_analytical_npz``).
    noise_models:
        List of ``NoiseModel`` objects applied in order.  Each trial applies
        all models and re-runs the full analysis.
    analysis_steps:
        ``AnalysisStep`` list defining the analysis pipeline.  Default:
        ``[MPMStep(3), BootstrapStep(n_boot=200)]`` (low n_boot for speed).
    n_trials:
        Number of independent noisy trials to average over.
    ground_truth_lambda_c:
        Known true λ_c for bias calculation.  If ``None``, the benchmark
        estimates it from the clean-data analysis.

    Example
    -------
    ::

        from mpm_analysis.simulation import PostSelectedDynamicsSimulator
        from mpm_analysis.simulation.noise import GaussianNoise, FiniteSampling
        from mpm_analysis.analysis.steps import MPMStep, BootstrapStep

        sim = PostSelectedDynamicsSimulator(num_lambdas=40)
        clean = sim.to_observable_records()

        bench = BenchmarkPipeline(
            clean_records=clean,
            noise_models=[GaussianNoise(sigma=0.02)],
            analysis_steps=[MPMStep(3), BootstrapStep(200)],
            n_trials=20,
        )
        result = bench.run()
        print(f"lam_c bias: {result.lambda_c_bias:.4f} +/- {result.lambda_c_std:.4f}")
    """

    def __init__(
        self,
        clean_records: list[ObservableRecord],
        noise_models: list[NoiseModel],
        analysis_steps: list[AnalysisStep] | None = None,
        n_trials: int = 20,
        ground_truth_lambda_c: float | None = None,
    ) -> None:
        self.clean_records = clean_records
        self.noise_models = noise_models
        self.analysis_steps = analysis_steps or [MPMStep(3), BootstrapStep(n_boot=200)]
        self.n_trials = n_trials
        self._ground_truth_lambda_c = ground_truth_lambda_c

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_analysis(self, records: list[ObservableRecord]) -> ZenoAnalysisResult | None:
        from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
        try:
            p = MPMPipeline.from_records(records, steps=self.analysis_steps)
            return p.run()
        except Exception as exc:
            print(f"  [BenchmarkPipeline] trial failed: {exc}")
            return None

    def _apply_noise(self, records: list[ObservableRecord]) -> list[ObservableRecord]:
        noisy = records
        for model in self.noise_models:
            noisy = model.apply(noisy)
        return noisy

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        lambda_c_guess: float | None = None,
        *,
        a_guess: float | None = None,
        b_guess: float | None = None,
        preset: str | None = None,
        plot: bool = False,
        fit_critical_point: bool = True,
        soft_bounds: bool = False,
        lambda_c_bounds: tuple[float, float] = (1.0, 1.5),
    ) -> "BenchmarkResult | BenchmarkPolesResult":
        """Run ``n_trials`` noisy analyses and compute bias/variance.

        Parameters
        ----------
        lambda_c_guess:
            Passed to ``CriticalPointPipeline.fit()``.
        a_guess, b_guess:
            Initial guesses for the sqrt fit.
        preset:
            Named preset from ``critical_point_guesses`` (e.g.
            ``"post_selected_simulation"``).  Overridden by explicit args.
        plot:
            If ``True``, show the appendix figure for the clean data fit.
        fit_critical_point:
            If ``True`` (default), fit the sqrt critical-point model and return
            a ``BenchmarkResult``.  If ``False``, skip the fit and return a
            ``BenchmarkPolesResult`` with per-trial ``PolesResult`` objects so
            you can inspect the raw pole distributions directly.
        soft_bounds, lambda_c_bounds:
            Passed to ``CriticalPointPipeline.fit()``.  With the default
            hard-box mode a ``b_guess`` of exactly 0 makes the b-bounds
            degenerate and every fit fails — use ``soft_bounds=True`` unless
            you have calibrated a/b guesses.

        Returns
        -------
        BenchmarkResult
            When ``fit_critical_point=True``.
        BenchmarkPolesResult
            When ``fit_critical_point=False``.
        """
        if not fit_critical_point:
            return self._run_poles_only(plot=plot)

        from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline

        print(f"BenchmarkPipeline: {self.n_trials} trials")

        # Resolve guesses from preset if needed
        if preset is not None:
            from mpm_analysis.analysis.critical_point_guesses import get_guess
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

        # Ground-truth analysis on clean data
        clean_result = self._run_analysis(self.clean_records)

        # Ground-truth λ_c
        true_lambda_c = self._ground_truth_lambda_c
        clean_fit = None
        if true_lambda_c is None and clean_result is not None:
            try:
                cp = CriticalPointPipeline(clean_result)
                clean_fit = cp.fit(
                    lambda_c_guess=lambda_c_guess,
                    a_guess=a_guess,
                    b_guess=b_guess,
                    soft_bounds=soft_bounds,
                    lambda_c_bounds=lambda_c_bounds,
                )
                true_lambda_c = clean_fit.lambda_c
                print(f"  Ground-truth lam_c (from clean data): {true_lambda_c:.4f}")
            except Exception as exc:
                print(f"  Could not fit ground-truth lam_c: {exc}")
                true_lambda_c = float("nan")

        if plot and clean_result is not None and clean_fit is not None:
            _plot_benchmark_fit(clean_result, clean_fit, label="clean data")

        extracted_lambda_cs = []
        first_trial_info: tuple[ZenoAnalysisResult, "SqrtFitResult"] | None = None
        for trial in range(self.n_trials):
            noisy = self._apply_noise(self.clean_records)
            result = self._run_analysis(noisy)
            if result is None:
                extracted_lambda_cs.append(float("nan"))
                continue
            try:
                cp = CriticalPointPipeline(result)
                fit = cp.fit(
                    lambda_c_guess=lambda_c_guess or true_lambda_c,
                    a_guess=a_guess,
                    b_guess=b_guess,
                    soft_bounds=soft_bounds,
                    lambda_c_bounds=lambda_c_bounds,
                )
                extracted_lambda_cs.append(fit.lambda_c)
                if first_trial_info is None:
                    result.critical_point_sqrt_fit = fit
                    first_trial_info = (result, fit)
            except Exception:
                extracted_lambda_cs.append(float("nan"))

            if (trial + 1) % max(1, self.n_trials // 5) == 0:
                print(f"  Trial {trial + 1}/{self.n_trials}")

        # Expose first successful noisy trial for sweep_noise() comparison plots.
        self._last_trial_info = first_trial_info

        noise_desc = {
            "models": [m.description() for m in self.noise_models]
        }

        benchmark = BenchmarkResult(
            ground_truth_lambda_c=float(true_lambda_c) if true_lambda_c is not None else float("nan"),
            extracted_lambda_c_samples=np.asarray(extracted_lambda_cs, dtype=float),
            noise_description=noise_desc,
        )
        print(
            f"Benchmark done: lam_c bias={benchmark.lambda_c_bias:.4f}, "
            f"std={benchmark.lambda_c_std:.4f}"
        )
        return benchmark

    def _run_poles_only(self, *, plot: bool = False) -> BenchmarkPolesResult:
        """Run trials without fitting the critical point.

        Returns a ``BenchmarkPolesResult`` containing per-trial pole spectra
        so the user can inspect the raw distributions.
        """
        print(f"BenchmarkPipeline (poles-only): {self.n_trials} trials")

        noise_desc = {"models": [m.description() for m in self.noise_models]}
        trial_poles = []
        lambda_values = None

        for trial in range(self.n_trials):
            noisy = self._apply_noise(self.clean_records)
            result = self._run_analysis(noisy)
            if result is None:
                continue
            trial_poles.append(result.poles)
            if lambda_values is None:
                lambda_values = result.lambda_values

            if (trial + 1) % max(1, self.n_trials // 5) == 0:
                print(f"  Trial {trial + 1}/{self.n_trials}")

        print(f"Benchmark (poles-only) done: {len(trial_poles)}/{self.n_trials} successful trials.")
        if plot and trial_poles:
            _plot_benchmark_poles(trial_poles, lambda_values)

        return BenchmarkPolesResult(
            trial_poles=trial_poles,
            noise_description=noise_desc,
            lambda_values=lambda_values,
        )

    # ------------------------------------------------------------------
    # Noise parameter sweep
    # ------------------------------------------------------------------

    def sweep_noise(
        self,
        noise_factory,
        param_values: list,
        lambda_c_guess: float | None = None,
        *,
        a_guess: float = 1.0,
        b_guess: float = 0.0,
        plot: bool = False,
        param_label: str = "noise",
        soft_bounds: bool = False,
        lambda_c_bounds: tuple[float, float] = (1.0, 1.5),
    ) -> list[BenchmarkResult]:
        """Run the benchmark for a range of noise levels.

        Parameters
        ----------
        noise_factory:
            Callable ``(param_value) -> list[NoiseModel]``.  Called once per
            value in ``param_values``.
        param_values:
            List of parameter values to sweep.
        lambda_c_guess:
            Passed to each ``run()`` call.

        Returns
        -------
        list[BenchmarkResult]
            One result per value in ``param_values``.

        Example
        -------
        ::

            results = bench.sweep_noise(
                noise_factory=lambda sigma: [GaussianNoise(sigma)],
                param_values=[0.01, 0.02, 0.05, 0.10],
            )
        """
        all_results = []
        comparison: list[tuple[str, ZenoAnalysisResult, "SqrtFitResult"]] = []
        for val in param_values:
            print(f"\n--- sweep {param_label} = {val} ---")
            self.noise_models = noise_factory(val)
            bench = self.run(
                lambda_c_guess=lambda_c_guess,
                a_guess=a_guess,
                b_guess=b_guess,
                plot=False,  # defer plotting until the sweep finishes
                soft_bounds=soft_bounds,
                lambda_c_bounds=lambda_c_bounds,
            )
            all_results.append(bench)
            info = getattr(self, "_last_trial_info", None)
            if info is not None:
                comparison.append((f"{param_label} = {val}", info[0], info[1]))

        if plot and comparison:
            _plot_splitting_comparison(comparison)
        return all_results

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path, stem: str | None = None) -> Path:
        """Save the last benchmark result as JSON.

        Parameters
        ----------
        output_dir:
            Directory to write into (created if absent).
        stem:
            Filename stem.  Auto-generated if ``None``.

        Returns
        -------
        Path
        """
        raise NotImplementedError(
            "BenchmarkResult saving not yet implemented.  "
            "Use the returned BenchmarkResult to access data programmatically."
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _plot_benchmark_fit(
    result: "ZenoAnalysisResult",
    fit: "SqrtFitResult",
    *,
    label: str = "",
) -> None:
    """Show the appendix figure (splitting + sqrt fit) for a benchmark result.

    Called when ``BenchmarkPipeline.run(plot=True)`` — once for the clean
    ground-truth fit, and once for the first noisy trial, so the two can be
    compared side-by-side.  ``plot_appendix_bootstrap`` reads the fit off
    ``result.critical_point_sqrt_fit``, so we attach it before rendering.
    """
    import matplotlib.pyplot as plt
    from mpm_analysis.plotting.paper_figures.appendix_bootstrap import plot_appendix_bootstrap

    result.critical_point_sqrt_fit = fit
    print(
        f"  [plot] {label or 'fit'}: lam_c = {fit.lambda_c:.4f} "
        f"+/- {fit.lambda_c_err:.4f}"
    )
    fig = plot_appendix_bootstrap(result)
    if label:
        fig.suptitle(label)
    plt.show(block=False)


def _plot_benchmark_poles(
    trial_poles: list,
    lambda_values: "np.ndarray | None",
) -> None:
    """Quick complex-plane scatter of poles across all trials, coloured by λ.

    Parameters
    ----------
    trial_poles:
        List of ``PolesResult`` objects, one per trial.
    lambda_values:
        Lambda axis for the colour scale.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not trial_poles or lambda_values is None:
        print("[_plot_benchmark_poles] nothing to plot.")
        return

    n_lam = len(lambda_values)
    cmap = cm.viridis
    norm = plt.Normalize(vmin=float(lambda_values.min()), vmax=float(lambda_values.max()))

    fig, ax = plt.subplots(figsize=(5, 4))
    for poles in trial_poles:
        re_poles = -poles.decay_rates   # Re(pole) = -decay_rate, shape (n_lam, order)
        im_poles = poles.frequencies
        for i in range(n_lam):
            color = cmap(norm(float(lambda_values[i])))
            ax.scatter(re_poles[i], im_poles[i], color=color, s=6, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$\lambda$")
    ax.set_xlabel(r"Re(pole)")
    ax.set_ylabel(r"Im(pole)")
    ax.set_title(f"Benchmark poles ({len(trial_poles)} trials)")
    ax.grid(False)
    fig.tight_layout()
    plt.show(block=False)


def _plot_splitting_comparison(
    entries: list[tuple[str, "ZenoAnalysisResult", "SqrtFitResult"]],
) -> None:
    """Compare splitting + sqrt fit across a sweep of noise parameter values.

    Produces a 2-row grid (imag splitting on top, real splitting on bottom)
    with one column per swept value, so the user can eyeball how the data
    and fit degrade as the noise parameter (e.g., n_shots) changes.
    """
    import matplotlib.pyplot as plt

    n = len(entries)
    fig, axes = plt.subplots(
        2, n, figsize=(3.2 * n, 5.2), sharex=True, squeeze=False,
    )

    for j, (label, result, fit) in enumerate(entries):
        if not result.poles.has_splitting:
            continue
        omega_s = result.omega_s
        lam = result.lambda_values
        p = result.poles

        im = (p.imag_splitting / omega_s).flatten()
        im_err = ((p.imag_splitting_upper + p.imag_splitting_lower) / 2.0 / omega_s).flatten()
        re = (p.real_splitting / omega_s).flatten()
        re_err = ((p.real_splitting_upper + p.real_splitting_lower) / 2.0 / omega_s).flatten()

        ax_im, ax_re = axes[0, j], axes[1, j]
        ax_im.errorbar(lam, im, yerr=im_err, fmt="o", ms=3, color="steelblue")
        ax_re.errorbar(lam, re, yerr=re_err, fmt="o", ms=3, color="steelblue")

        if fit.lambda_plot is not None:
            if fit.imag_fit_curve is not None:
                ax_im.plot(fit.lambda_plot, fit.imag_fit_curve, "k--", lw=1.2)
            if fit.real_fit_curve is not None:
                ax_re.plot(fit.lambda_plot, fit.real_fit_curve, "k--", lw=1.2)
        for ax in (ax_im, ax_re):
            ax.axvline(fit.lambda_c, color="crimson", ls=":", lw=1.0, alpha=0.8)

        ax_im.set_title(f"{label}\nλ_c = {fit.lambda_c:.3f} ± {fit.lambda_c_err:.3f}",
                        fontsize=9)
        ax_re.set_xlabel(r"$\lambda$")
        if j == 0:
            ax_im.set_ylabel(r"$\mathrm{Im}(\Delta e_{12}) / \Omega_S$")
            ax_re.set_ylabel(r"$\mathrm{Re}(\Delta e_{12}) / \Omega_S$")

    fig.suptitle("Benchmark splitting comparison across swept noise parameter")
    fig.tight_layout()
    plt.show(block=False)
