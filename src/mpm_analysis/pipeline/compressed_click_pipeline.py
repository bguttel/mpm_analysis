"""
compressed_click_pipeline.py
----------------------------
High-level orchestrator for the compressed-click workflow:

    raw individual_runs NPZs
        -> CompressedClickBuilder        (compress + counting statistics
                                          + dead-time histograms + FT-over-k)
        -> CompressedClickLoader          (load the compact per-rate NPZs)
        -> summary / histogram figures    (plotting.compressed_click_stats,
                                          including one dead-time PNG per λ,
                                          the mean-vs-λ PNG and its data NPZ)
        -> [optional] k-space pole analysis (kspace pipeline) + its figures

Every output — compressed NPZs, statistics figures, and pole-analysis products —
is written into the **same** analysis subfolder
(``<main_folder>/compressed_click_analysis/`` by default), with timestamped,
non-clobbering names.  Generic over scan number.
"""
from __future__ import annotations

from pathlib import Path

from mpm_analysis.io.experimental.compressed_click_record import (
    CompressedClickBuilder, CompressedClickLoader, DEFAULT_OUT_SUBDIR,
)


class CompressedClickPipeline:
    """Build compressed click records and (optionally) run the pole analysis."""

    def __init__(self, *, out_subdir: str = DEFAULT_OUT_SUBDIR) -> None:
        self.out_subdir = out_subdir

    def run(
        self,
        individual_runs_folder: str | Path,
        *,
        rates: list[int] | None = None,
        max_runs: int | None = None,
        min_dark: int = 1,
        out_dir: str | Path | None = None,
        make_figures: bool = True,
        run_poles: bool = False,
        pole_kwargs: dict | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run the full workflow.

        Parameters
        ----------
        individual_runs_folder : the scan's ``individual_runs`` directory.
        rates, max_runs : forwarded to :meth:`CompressedClickBuilder.build`.
        out_dir : explicit output directory (default ``<main>/<out_subdir>``).
        make_figures : save the counting-statistics + summary PNGs, the per-λ
            dead-time PNGs, and the ``dead_time_vs_lambda_*.npz`` data file.
        run_poles : also run the k-space MPM pole analysis on the compressed
            records and save ``kspace_poles_compressed_*.npz`` + pole figures.
        pole_kwargs : options forwarded to ``KSpaceAnalysis`` / loader, e.g.
            ``dict(max_order=2, min_order=2, threshold=0.5, k_index_range=(1, 100))``.

        Returns
        -------
        dict with keys ``npz_paths``, ``out_dir``, ``records``, ``figures``,
        ``data_files`` (the saved non-figure products, e.g. the dead-time
        table), ``poles`` (or None), ``poles_path`` (or None).
        """
        builder = CompressedClickBuilder(out_subdir=self.out_subdir, min_dark=min_dark)
        npz_paths = builder.build(
            individual_runs_folder, rates=rates, max_runs=max_runs,
            out_dir=out_dir, verbose=verbose,
        )
        if not npz_paths:
            raise RuntimeError("No compressed NPZs were produced.")
        analysis_dir = npz_paths[0].parent

        records = CompressedClickLoader().load(analysis_dir)

        figures: list[Path] = []
        data_files: list[Path] = []
        if make_figures:
            from mpm_analysis.plotting.compressed_click_stats import (
                save_compressed_click_figures,
            )
            saved = save_compressed_click_figures(records, analysis_dir)
            figures = [p for p in saved if p.suffix.lower() == ".png"]
            data_files = [p for p in saved if p.suffix.lower() != ".png"]

        poles = None
        poles_path = None
        if run_poles:
            poles, poles_path = self._run_poles(
                analysis_dir, pole_kwargs or {}, verbose,
            )

        return dict(
            npz_paths=npz_paths, out_dir=analysis_dir, records=records,
            figures=figures, data_files=data_files,
            poles=poles, poles_path=poles_path,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _run_poles(analysis_dir: Path, pk: dict, verbose: bool):
        """Delegate the k-space pole analysis to the unified ``KSpacePipeline``.

        No duplicated load/analyze/save/plot logic — the compressed pipeline only
        owns the *build* (raw -> compressed) and the compressed *stat* figures.
        """
        from mpm_analysis.pipeline.kspace_pipeline import KSpacePipeline

        out = KSpacePipeline().run(
            analysis_dir, source="compressed",
            k_index_range=pk.get("k_index_range", (1, 100)),
            t_max=pk.get("t_max"),
            min_alive_frac=pk.get("min_alive_frac", 0.5),
            max_order=pk.get("max_order", 2),
            min_order=pk.get("min_order", 2),
            threshold=pk.get("threshold", 0.5),
            n_bootstrap=pk.get("n_bootstrap", 1),
            bootstrap_over_runs=pk.get("bootstrap_over_runs", False),
            theory=pk.get("theory"),
            braiding=pk.get("braiding", True),
            save_figures=pk.get("save_figures", True),
            open_figures=pk.get("open_figures", False),
            verbose=verbose,
        )
        return out["result"], out["result_path"]

