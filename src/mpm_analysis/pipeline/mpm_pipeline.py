"""
mpm_pipeline.py
---------------
High-level pipeline: load data → run analysis steps → return / save result.

Replaces the monolithic ``ZenoTimeDomainAnalyzer`` / ``object_oriented_poles.py``
entry points with a clean class that delegates to ``io/`` and ``analysis/``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mpm_analysis.data_types.analysis_result import ZenoAnalysisResult
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.analysis.steps.base import AnalysisStep
from mpm_analysis.analysis.steps.mpm_step import MPMStep
from mpm_analysis.analysis.steps.bootstrap_step import BootstrapStep


_DEFAULT_STEPS = lambda order, n_boot: [MPMStep(order=order), BootstrapStep(n_boot=n_boot)]


class MPMPipeline:
    """Matrix Pencil + Bootstrap analysis pipeline.

    Parameters
    ----------
    records:
        Pre-loaded list of ``ObservableRecord`` objects.
    steps:
        Ordered list of ``AnalysisStep`` objects.  Default:
        ``[MPMStep(order=3), BootstrapStep(n_boot=2000)]``.
    metadata:
        Extra key-value pairs to embed in the result metadata.

    Typical usage
    ~~~~~~~~~~~~~
    ::

        pipeline = MPMPipeline.from_ensemble_average(folder, order=3, n_boot=500)
        result = pipeline.run()
        pipeline.save(output_dir=Path("results/"))

    Or with custom steps::

        from mpm_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep
        steps = [MPMStep(order=3), RefineStep(), BootstrapStep(n_boot=2000)]
        pipeline = MPMPipeline.from_ensemble_average(folder, steps=steps)
    """

    def __init__(
        self,
        records: list[ObservableRecord],
        steps: list[AnalysisStep] | None = None,
        *,
        metadata: dict | None = None,
        order: int = 3,
        n_boot: int = 2000,
    ) -> None:
        self.records = records
        self.steps = steps if steps is not None else _DEFAULT_STEPS(order, n_boot)
        self.metadata = dict(metadata or {})
        self._result: ZenoAnalysisResult | None = None

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_loader(
        cls,
        loader,
        folder: str | Path,
        *,
        data_mode: str | None = None,
        source_tag: str | None = None,
        observables: list[str] | str | None = None,
        steps: list[AnalysisStep] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        t_max: float | None = None,
        drop_first_point: bool = False,
        splice_index: int | None = None,
        splice_source: str = "S_prev_noclick",
        splice_targets: tuple[str, ...] = ("S",),
        order: int = 3,
        n_boot: int = 2000,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load data via any ``ExperimentalLoader`` and return a pipeline.

        Parameters
        ----------
        loader:
            A configured loader instance (``EnsembleAverageLoader``,
            ``PostSelectedLoader``, ``ParityLoader``, ...).
        data_mode:
            Experimental data mode tag (``"ens_avg"`` | ``"post_selected"``
            | ``"parity"`` | ...).  ``source_tag`` is accepted as a
            deprecated alias.
        observables:
            Observable key(s) to load.  ``None`` → the loader's default.

        splice_index:
            Replace this one time index of ``splice_targets`` with the same
            index of ``splice_source`` (see ``ExperimentalLoader.load``).
            Recorded in the result metadata when given, so a saved analysis
            always states that one of its points came from another
            sub-ensemble.
        drop_first_point:
            Drop the first sample of every file (the by-construction t=0
            anchor of the survival scans) before any analysis step, in the
            observables and the bootstrap statistics alike (see
            ``ExperimentalLoader.load``).  Recorded in the result metadata.
        """
        if data_mode is None:
            data_mode = source_tag
        if data_mode is None:
            raise TypeError("from_loader: data_mode is required.")
        folder = Path(folder)
        records = loader.load(
            folder,
            observables=observables,
            lambda_range=lambda_range,
            max_files=max_files,
            t_max=t_max,
            drop_first_point=drop_first_point,
            splice_index=splice_index,
            splice_source=splice_source,
            splice_targets=splice_targets,
        )
        if observables is None:
            obs_list = sorted({r.observable_key for r in records})
        elif isinstance(observables, str):
            obs_list = [observables]
        else:
            obs_list = list(observables)
        meta = {
            "analysis_type": "mpm",
            "source_folder": str(folder),
            "data_mode": data_mode,
            "source_tag": data_mode,   # legacy alias kept for old readers
            "observables": obs_list,
            # Which lambda axis this result lives on — see
            # ExperimentalLoader.lambda_source
            "lambda_source": getattr(loader, "lambda_source",
                                     "json:lambda_zeno_estimation"),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        if drop_first_point:
            meta["drop_first_point"] = True
        if splice_index is not None:
            # A spliced trace is not the file's own survival any more, so the
            # saved analysis records exactly which point was swapped in.
            meta["splice_index"] = int(splice_index)
            meta["splice_source"] = str(splice_source)
            meta["splice_targets"] = list(splice_targets)
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    @classmethod
    def from_ensemble_average(
        cls,
        folder: str | Path,
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        order: int = 3,
        n_boot: int = 2000,
        csv_path: str | Path | bool | None = "auto",
        t_max: float | None = None,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load ensemble-average data and return a ready-to-run pipeline."""
        from mpm_analysis.io.experimental import EnsembleAverageLoader
        return cls.from_loader(
            EnsembleAverageLoader(csv_path=csv_path), folder,
            data_mode="ens_avg",
            observables=observables or ["survival"],
            steps=steps, lambda_range=lambda_range, max_files=max_files,
            t_max=t_max, order=order, n_boot=n_boot, **metadata_kwargs,
        )

    @classmethod
    def from_post_selected(
        cls,
        folder: str | Path,
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        order: int = 3,
        n_boot: int = 2000,
        csv_path: str | Path | bool | None = "auto",
        t_max: float | None = None,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load post-selected tomography data and return a ready-to-run pipeline.

        Parameters
        ----------
        csv_path:
            ``"auto"`` (default) finds the HMM table that provably belongs
            to ``folder`` and falls back to the JSON ``lambda_zeno_estimation``
            when none does.  ``False`` forces the JSON value; an explicit path
            is provenance-checked.  See ``io.experimental.discover_hmm_table``.
        """
        from mpm_analysis.io.experimental import PostSelectedLoader
        return cls.from_loader(
            PostSelectedLoader(csv_path=csv_path), folder,
            data_mode="post_selected",
            observables=observables or ["survival"],
            steps=steps, lambda_range=lambda_range, max_files=max_files,
            t_max=t_max, order=order, n_boot=n_boot, **metadata_kwargs,
        )

    @classmethod
    def from_parity(
        cls,
        folder: str | Path,
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        order: int = 3,
        n_boot: int = 2000,
        t_max: float | None = None,
   
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load parity-scan data and return a ready-to-run pipeline."""
        from mpm_analysis.io.experimental import ParityLoader
        return cls.from_loader(
            ParityLoader(), folder,
            data_mode="parity",
            observables=observables or ["parity_real"],
            steps=steps, lambda_range=lambda_range, max_files=max_files,
            t_max=t_max, order=order, n_boot=n_boot, **metadata_kwargs,
        )

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        data_type: str = "analytical",
        steps: list[AnalysisStep] | None = None,
        observables: list[str] | None = None,
        lambda_range: tuple[float, float] | None = None,
        order: int = 3,
        n_boot: int = 2000,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Load analytical / MC NPZ and return a ready-to-run pipeline.

        Parameters
        ----------
        data_type:
            ``"analytical"`` or ``"monte_carlo"``.
        """
        from mpm_analysis.io.simulation import load_analytical_npz, load_monte_carlo_npz
        path = Path(path)
        loader_fn = load_analytical_npz if data_type == "analytical" else load_monte_carlo_npz
        records = loader_fn(
            path,
            observables=observables or ["survival"],
            lambda_range=lambda_range,
        )
        meta = {
            "analysis_type": "mpm",
            "source_folder": str(path),
            "data_mode": data_type,
            "source_tag": data_type,   # legacy alias
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    @classmethod
    def from_records(
        cls,
        records: list[ObservableRecord],
        steps: list[AnalysisStep] | None = None,
        order: int = 3,
        n_boot: int = 2000,
        **metadata_kwargs,
    ) -> "MPMPipeline":
        """Build a pipeline from an already-loaded list of records.

        Useful when records come from a simulator's ``to_observable_records()``.
        """
        meta = {
            "analysis_type": "mpm",
            "data_mode": "simulation",
            "source_tag": "simulation",   # legacy alias
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **metadata_kwargs,
        }
        return cls(records, steps, metadata=meta, order=order, n_boot=n_boot)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> ZenoAnalysisResult:
        """Execute all analysis steps and return a ``ZenoAnalysisResult``.

        The result is also stored in ``self._result`` for later access or saving.
        """
        state: dict[str, Any] = {}
        for step in self.steps:
            print(f"  > {step.description()}")
            state = step.process(self.records, state)

        poles_result = state.get("poles_result")
        if poles_result is None:
            raise RuntimeError(
                "No 'poles_result' in state after running steps. "
                "Make sure a BootstrapStep is included."
            )

        lambdas = np.asarray(state.get("lambdas", []))
        step_descriptions = [s.description() for s in self.steps]

        params = {
            "pole_order": state.get("poles_raw", [np.array([])])[0].shape[0]
                          if state.get("poles_raw") else 3,
            "step_descriptions": step_descriptions,
        }
        if state.get("observable_keys") is not None:
            params["observable_keys"] = list(state["observable_keys"])

        if "scan_number" not in self.metadata:
            scan_numbers = {
                int(r.metadata["scan_number"])
                for r in self.records
                if r.metadata.get("scan_number") is not None
            }
            if len(scan_numbers) == 1:
                self.metadata["scan_number"] = scan_numbers.pop()

        # Pull omega_DG from first record that has it
        omega_dg = next(
            (r.metadata.get("omega_DG") for r in self.records if r.metadata.get("omega_DG") is not None),
            None,
        )
        if omega_dg is not None:
            params["omega_DG"] = float(omega_dg)
        else:
            import warnings
            warnings.warn(
                "omega_DG not found in any record metadata — pole normalization will fall back "
                "to 1.0. Check that Omega_DG is present in the experiment JSON files.",
                UserWarning,
                stacklevel=2,
            )

        # Harvest everything else the steps produced, for full-output saving.
        extras: dict = {}
        for key in ("refine_params", "refine_cov", "refine_conjugate",
                    "amplitudes_raw", "L_used"):
            if state.get(key) is not None:
                extras[key] = state[key]

        self._result = ZenoAnalysisResult(
            metadata=self.metadata,
            parameters=params,
            lambda_values=lambdas,
            poles=poles_result,
            extras=extras or None,
        )
        print(f"Pipeline complete: {len(lambdas)} lambda points, poles shape {poles_result.decay_rates.shape}")
        return self._result

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        output_dir: str | Path,
        stem: str | None = None,
        *,
        full: bool = False,
    ) -> Path:
        """Save the most recent ``run()`` result to ``output_dir``.

        Parameters
        ----------
        output_dir:
            Directory to write into.
        stem:
            Filename stem (without ``.json``).  Auto-generated if ``None``.
        full:
            If ``True``, also serialise the raw bootstrap pole samples and
            the ``extras`` block (refine params/covariances, amplitudes, …)
            — everything the analysis produced.

        Returns
        -------
        Path
            Path to the saved file.
        """
        if self._result is None:
            raise RuntimeError("Call run() before save().")
        from mpm_analysis.io.file_naming import save_result
        return save_result(self._result, output_dir, stem=stem, full=full)
