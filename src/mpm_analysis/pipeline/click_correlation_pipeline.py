"""
click_correlation_pipeline.py
-----------------------------
One pipeline for the click-probability chain — post-selected (conditional)
no-jump curves by default, full-dynamics (unconditional) curves as a toggle —
mirroring the package architecture used for every other quantity:

    load (source switch)  ->  correlation analysis  ->  ObservableRecords
        ->  matrix pencil (MPMStep + BootstrapStep)  ->  figures + saving

Source switch (same idiom as :class:`~mpm_analysis.pipeline.kspace_pipeline.KSpacePipeline`):

    source="compressed" -> CompressedClickLoader   (compressed click NPZs)
    source="json"       -> ClickRecordNpzLoader    (paired JSON+NPZ click records)
    source="simulated"  -> SimClickRecordLoader    (multi-λ simulation NPZ)

All three yield shot-resolved click records; ``compute_all_click_correlations``
dispatches on the record type, so the estimator, MPM stage, figures and file
layout are identical across sources.

Every run writes into its own timestamped sub-folder (correlation data NPZ,
one ``ZenoAnalysisResult`` JSON per fitted observable, and the figure set) so
products from different runs never mix.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from mpm_analysis.analysis.click_correlations import (
    DEFAULT_MIN_ALIVE_FRAC,
    DEFAULT_MODE,
    MODE_KEYS,
    MODE_MAX_CLICKS,
    ClickCorrelationResult,
    compute_all_click_correlations,
    to_observable_records,
)
from mpm_analysis.io.experimental.compressed_click_record import (
    _timestamp, _unique_path,
)
from mpm_analysis.utils.windows_paths import win_long_path

#: Where the run folder goes, relative to the input, per source.  The
#: compressed source already points at an analysis folder; the simulated
#: source points at a single NPZ file (its parent is used).
_DEFAULT_SUBDIR = {"json": "correlation_analysis", "compressed": ".", "simulated": "."}

_VALID_SOURCES = ("compressed", "json", "simulated")


class ClickCorrelationPipeline:
    
    """Load -> correlate -> matrix pencil -> plot/save, for any click source."""

    def run(
        self,
        folder: str | Path | None = None,
        *,
        source: str = "compressed",
        records: list | None = None,
        out_dir: str | Path | None = None,
        # loading
        lambda_range: tuple[float, float] | None = None,
        max_files: int | None = None,
        t_max_us: float | None = None,
        latest_only: bool = True,
        # correlation estimator
        mode: str = DEFAULT_MODE,
        max_time_us: float | None = None,
        min_alive_frac: float = DEFAULT_MIN_ALIVE_FRAC,
        max_clicks: int | None = None,
        n_bootstrap: int = 200,
        seed: int = 0,
        n_groups: int = 25,
        # matrix pencil
        observables: tuple[str, ...] | None = None,
        skip_first: int = 0,
        order: int = 3,
        L: int | None = None,
        n_boot: int = 500,
        mpm_seed: int | None = 0,
        run_mpm: bool = True,
        plot_svd: bool = True,
        # output
        save_results: bool = True,
        save_figures: bool = True,
        scan_number: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run the full correlation workflow for ``source`` on ``folder``.

        Parameters
        ----------
        folder :
            Input location per source — the ``compressed_click_analysis``
            folder (``"compressed"``), the folder of paired JSON+NPZ click
            records (``"json"``), or the multi-λ simulation NPZ file
            (``"simulated"``).  Ignored when ``records`` is given.
        records :
            Pre-loaded records (``CompressedClickRecord`` /
            ``ClickRecordData``) — bypasses loading entirely.  ``out_dir``
            is then required for saving.
        mode :
            ``"conditional"`` (default): post-selected no-jump dynamics —
            ``no_click[t] = P(n = 0, t)``, silence required in every bin, so
            the fit returns the non-Hermitian eigenvalues.
            ``"unconditional"``: full dynamics — ``no_click[t] =
            P(no click at t)`` with intermediate clicks irrelevant, so the
            fit returns the full Liouvillian eigenvalues.
            ``"partial"``: at most ``max_clicks`` clicks before ``t``, the
            interpolation between the two.
            Both curves start at each run's own ``t = 0``; there is no lag
            axis and no anchoring.
        max_clicks :
            Clicks tolerated strictly before ``t``.  Fixed by the two named
            modes (0 / unlimited); required for ``mode="partial"``.
        max_time_us, min_alive_frac, n_bootstrap, seed, n_groups :
            Forwarded to the correlation estimator
            (:func:`~mpm_analysis.analysis.click_correlations.compute_all_click_correlations`).
            ``min_alive_frac`` truncates once too few runs are still
            recorded — only ever active for the compressed source.
        t_max_us :
            Truncate each record at this absolute time when loading
            (dense sources; the compressed source keeps its native axis).
        observables :
            Which channels get an MPM fit.  ``None`` (default) fits both
            channels of the chosen mode, e.g. ``("cond_click",
            "cond_no_click")``.
        skip_first :
            Drop this many leading time points from the MPM fit (for records
            whose first bins carry an initialisation transient).
        order, L, n_boot, mpm_seed :
            Matrix-pencil model order, pencil parameter and bootstrap draws.
        run_mpm :
            Set ``False`` to stop after the correlation stage (no pole fits).
        plot_svd :
            Save the Hankel-SVD model-order calibration figure of the MPM
            step (one per fitted observable) into the run folder.  Needs
            ``save_figures``.

        Returns
        -------
        dict with keys ``records``, ``correlations``, ``observable_records``,
        ``mpm_results`` (``{key: ZenoAnalysisResult}``), ``result_paths``,
        ``correlations_path``, ``figures``, ``out_dir``.
        """
        if source not in _VALID_SOURCES:
            raise ValueError(f"Unknown source {source!r}. Valid: {_VALID_SOURCES}.")

        # ---- 1+2. load and correlate ----------------------------------------
        # The json source prefers the scan's ``individual_runs`` folder when it
        # exists: the scan-level NPZs may store empty ``assigned_states``, and
        # streaming the per-run files keeps memory flat and gives real
        # run-resampled error bars.  Correlations are then computed during the
        # (streamed) load, so both stages live in this block.
        correlations = None
        out_root = None if folder is None else Path(folder)
        estimator_kwargs = dict(
            mode=mode, max_clicks=max_clicks, max_time_us=max_time_us,
            min_alive_frac=min_alive_frac,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        if records is None:
            if folder is None:
                raise ValueError("Provide either `folder` or pre-loaded `records`.")
            folder = Path(folder)
            if source == "simulated":
                records = []
                if self._is_shifted_npz(folder):
                    # Shifted-resolution NPZs carry no per-shot record — only
                    # the reconstructed per-fine-bin click probabilities,
                    # whose running product IS the no-click curve.
                    correlations = self._correlations_from_shifted(
                        folder, lambda_range=lambda_range, mode=mode,
                        max_clicks=max_clicks, max_time_us=max_time_us,
                        t_max_us=t_max_us, verbose=verbose,
                    )
                else:
                    # Raw simulation NPZs hold ~N·T-bit records per λ —
                    # stream one λ at a time instead of loading all of them.
                    correlations = self._correlate_sim_stream(
                        folder, lambda_range=lambda_range,
                        max_records=max_files, t_max_us=t_max_us,
                        n_groups=n_groups,
                        estimator_kwargs=estimator_kwargs, verbose=verbose,
                    )
                if not correlations:
                    raise RuntimeError(
                        f"No lambda entries loaded from {folder} "
                        "(simulated source)."
                    )
            if source == "json":
                runs_dir = self._find_runs_dir(folder)
                if runs_dir is not None:
                    out_root = runs_dir.parent
                    records = []
                    correlations = self._correlate_run_groups(
                        runs_dir, lambda_range=lambda_range,
                        max_groups=max_files, t_max_us=t_max_us,
                        estimator_kwargs=estimator_kwargs, verbose=verbose,
                    )
                    if not correlations:
                        raise RuntimeError(
                            f"No run groups loaded from {runs_dir} (source=json)."
                        )
            if correlations is None:
                records = self._load(
                    folder, source,
                    lambda_range=lambda_range, max_files=max_files,
                    t_max_us=t_max_us, latest_only=latest_only,
                )
        if correlations is None:
            if not records:
                raise RuntimeError(
                    f"No records loaded from {folder} (source={source})."
                )
            if verbose:
                lam = sorted(r.lambda_val for r in records)
                print(f"[corr] source={source}: {len(records)} records "
                      f"(lam {lam[0]:.3f}..{lam[-1]:.3f})")
            correlations = compute_all_click_correlations(
                records, n_groups=n_groups, **estimator_kwargs,
            )
        if verbose:
            for res in correlations:
                print(f"  lam={res.lambda_val:6.3f}  r_bar={res.r_bar:.4f}  "
                      f"{len(res.t_us)} bins (dt={res.dt_us:.3f} us)")

        # ---- 3. observable records ------------------------------------------
        if observables is None:
            observables = correlations[0].keys
        obs_records = to_observable_records(
            correlations, observables=observables, skip_first=skip_first,
        )

        # ---- 4. output folder (before MPM so the SVD figure can land there) --
        run_dir = None
        if save_results or save_figures:
            run_dir = self._make_run_dir(out_root, source, out_dir)
            if verbose:
                print(f"[corr] run folder: {run_dir}")

        # ---- 5. matrix pencil ------------------------------------------------
        mpm_results: dict = {}
        mpm_pipelines: dict = {}
        svd_figures: list[Path] = []
        if run_mpm:
            from mpm_analysis.analysis.steps.bootstrap_step import BootstrapStep
            from mpm_analysis.analysis.steps.mpm_step import MPMStep
            from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline

            svd_on = plot_svd and save_figures and run_dir is not None
            min_points = 2 * order + 1
            for key in observables:
                key_records = [r for r in obs_records if r.observable_key == key]
                key_records, dropped = self._drop_short_slices(
                    key_records, min_points, key=key, order=order,
                    mode=mode, min_alive_frac=min_alive_frac, verbose=verbose,
                )
                if not key_records:
                    continue
                if verbose:
                    print(f"[corr] MPM on '{key}' ({len(key_records)} lam, "
                          f"order {order}"
                          + (f", {dropped} lam too short" if dropped else "")
                          + ")")
                pipeline = MPMPipeline.from_records(
                    key_records,
                    steps=[
                        MPMStep(order=order, L=L, observable_key=key,
                                plot_svd=svd_on, save_dir=run_dir),
                        BootstrapStep(n_boot=n_boot, seed=mpm_seed),
                    ],
                    source_tag=f"{key}_{source}",
                )
                mpm_results[key] = pipeline.run()
                mpm_pipelines[key] = pipeline
                if svd_on:
                    p = run_dir / f"svd_{key}.png"
                    if os.path.exists(win_long_path(p)):
                        svd_figures.append(p)

        correlations_path = None
        result_paths: dict = {}
        if save_results:
            correlations_path = self._save_correlations(
                correlations, run_dir, source,
            )
            if verbose:
                print(f"[corr] saved {correlations_path.name}")
            for key, pipe in mpm_pipelines.items():
                result_paths[key] = pipe.save(run_dir)
                if verbose:
                    print(f"[corr] saved {result_paths[key].name}")

        figures: list[Path] = []
        if save_figures:
            from mpm_analysis.plotting.click_correlation_plots import (
                save_click_correlation_figures,
            )
            if scan_number is None:
                scan_number = next(
                    (int(r.metadata["scan_number"]) for r in correlations
                     if r.metadata.get("scan_number") is not None),
                    None,
                )
            figures = save_click_correlation_figures(
                correlations, mpm_results, run_dir,
                source_tag=f"corr_{source}", scan_number=scan_number,
            )
            figures = svd_figures + figures
            if verbose:
                for p in figures:
                    print(f"[corr] saved {p.name}")

        return dict(
            records=records, correlations=correlations,
            observable_records=obs_records, mpm_results=mpm_results,
            result_paths=result_paths, correlations_path=correlations_path,
            figures=figures, out_dir=run_dir,
        )

    # ------------------------------------------------------------------
    # Loading — the source switch
    # ------------------------------------------------------------------

    @staticmethod
    def _find_runs_dir(folder: Path) -> Path | None:
        """The ``individual_runs`` folder for a json scan, if it has run files.

        Accepts either the scan folder (looks for the ``individual_runs``
        subfolder) or the ``individual_runs`` folder itself; returns ``None``
        when no ``*_run_i_of_n_*`` NPZs are found, in which case the flat
        paired-file loader is used instead.
        """
        from mpm_analysis.io.experimental.click_record_npz import _RUN_RE

        for cand in (folder, folder / "individual_runs"):
            if cand.is_dir() and any(
                _RUN_RE.search(p.name) for p in cand.glob("*.npz")
            ):
                return cand
        return None

    @staticmethod
    def _drop_short_slices(
        key_records: list,
        min_points: int,
        *,
        key: str,
        order: int,
        mode: str,
        min_alive_frac: float,
        verbose: bool,
    ) -> tuple[list, int]:
        """Keep only slices long enough for an ``order``-pole pencil.

        A λ whose retained window is shorter than ``2*order + 1`` points
        cannot be fitted at all; dropping it here (with a message naming the
        knobs that control the window) beats a wall of NaNs downstream.
        """
        keep = [r for r in key_records if len(r.signal) >= min_points]
        dropped = len(key_records) - len(keep)
        if dropped and verbose:
            lengths = sorted(len(r.signal) for r in key_records)
            print(f"[corr] '{key}': {dropped}/{len(key_records)} lam have "
                  f"fewer than {min_points} points (order {order}); "
                  f"lengths {lengths[0]}..{lengths[-1]}")
        if not keep:
            hint = ("lower min_alive_frac (currently "
                    f"{min_alive_frac:g}) to keep more time points")
            if mode == "conditional":
                hint += (", or switch to mode='unconditional' "
                         "(its window is not shortened by clicks)")
            print(f"[corr] '{key}': no lam long enough for order {order} — "
                  f"skipping the fit.  Try: {hint}, or a smaller order.")
        return keep, dropped

    @staticmethod
    def _is_shifted_npz(path: Path) -> bool:
        """True if ``path`` is a shifted-resolution simulation NPZ.

        Those files (``io/simulated/shifted_resolution.py``) store per-λ
        reconstructed fine-bin click probabilities ``q_{i}`` instead of the
        per-shot ``proc_packed_{i}`` trajectories of the raw format.
        """
        import numpy as np

        if not path.is_file():
            return False
        try:
            z = np.load(win_long_path(path), allow_pickle=False)
        except Exception:
            return False
        return "delta_fine" in z.files and any(
            k.startswith("q_") for k in z.files
        )

    @staticmethod
    def _correlations_from_shifted(
        path: Path,
        *,
        lambda_range,
        mode: str,
        max_clicks,
        max_time_us,
        t_max_us,
        verbose: bool,
    ) -> list:
        """Click / no-click curves straight from a shifted-resolution NPZ.

        The file stores the reconstructed per-fine-bin click probabilities
        ``q_f`` (bin width ``delta_fine``) instead of any per-shot record.
        Both fixed modes follow in closed form on that grid::

            conditional   : no_click[j] = prod_{f<=j} (1 - q_f)   ( = P(n=0,t) )
                            click[j]    = q_j * prod_{f<j} (1 - q_f)
            unconditional : no_click[j] = 1 - q_j ,  click[j] = q_j

        ``mode="partial"`` is not available here: bounding the number of
        intermediate clicks needs the counting distribution, which for these
        independent fine bins would come from
        :func:`~mpm_analysis.io.simulated.shifted_resolution.poisson_binomial_pnt`.
        No bootstrap is possible either — the errors are zero.
        """
        import numpy as np

        if mode == "partial":
            raise ValueError(
                "mode='partial' is not available for shifted-resolution NPZs: "
                "they store per-bin click probabilities, not per-shot records. "
                "Use 'conditional' or 'unconditional' here."
            )
        z = np.load(win_long_path(path), allow_pickle=False)
        lam_values = np.atleast_1d(z["lambda_values"])
        delta = float(z["delta_fine"])

        meta_common: dict = {"source_file": Path(path).name}
        for k in ("Omega", "T1", "T2", "dt", "n_shifts", "n_trials_per_shift"):
            if k in z.files:
                meta_common[k] = z[k].item()
        if "Omega" in z.files:
            meta_common["omega_DG"] = float(z["Omega"])

        caps = [v for v in (max_time_us, t_max_us) if v is not None]
        t_cap = min(caps) if caps else None

        out = []
        for i, lam in enumerate(lam_values):
            key = f"q_{i}"
            if key not in z.files:
                continue
            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= float(lam) <= hi):
                    continue
            q = z[key].astype(np.float64)
            t = np.arange(len(q), dtype=np.float64) * delta
            survival = np.cumprod(1.0 - q)             # P(no click in 0..j)
            r_bar = float(1.0 - survival[-1])
            if mode == "conditional":
                no_click = survival
                # first click exactly at j: silent through j-1, then click
                click = q * np.concatenate(([1.0], survival[:-1]))
            else:
                no_click = 1.0 - q
                click = q.copy()
            if t_cap is not None:
                n_keep = max(2, int(np.searchsorted(t, t_cap, side="right")))
                t, no_click, click = t[:n_keep], no_click[:n_keep], click[:n_keep]
            if verbose:
                print(f"[corr] shifted: lam={float(lam):.3f}  "
                      f"{len(t)} fine bins (delta={delta:.4f} us)")
            out.append(ClickCorrelationResult(
                lambda_val=float(lam), rate_khz=None, dt_us=delta,
                t_us=t,
                click=click, click_err=np.zeros_like(click),
                no_click=no_click, no_click_err=np.zeros_like(no_click),
                r_bar=r_bar, mode=mode,
                max_clicks=MODE_MAX_CLICKS[mode],
                window=dict(
                    estimator=mode, source="shifted_reconstruction",
                    max_clicks=MODE_MAX_CLICKS[mode],
                    min_alive_frac=float("nan"), n_bins=len(t),
                ),
                metadata=dict(meta_common),
            ))
        out.sort(key=lambda r: r.lambda_val)
        return out

    @staticmethod
    def _correlate_sim_stream(
        path: Path,
        *,
        lambda_range,
        max_records,
        t_max_us,
        n_groups: int,
        estimator_kwargs: dict,
        verbose: bool,
    ) -> list:
        """Stream a raw simulation NPZ into correlation results, one λ at a time.

        Each λ's boolean click matrix (derived from ``proc_packed_{i}`` or,
        as a fallback, the raw telegraph ``raw_packed_{i}``) exists only
        while its correlation is computed.
        """
        import numpy as np

        from mpm_analysis.analysis.click_correlations import (
            compute_click_correlations_dense,
        )
        from mpm_analysis.io.simulated.click_record_raw_npz import (
            SimClickRecordLoader,
        )

        correlations = []
        loader = SimClickRecordLoader(load_trajectories=True)
        for rec in loader.iter_records(
            path, lambda_range=lambda_range, max_records=max_records,
        ):
            if t_max_us is not None:
                n_keep = max(2, int(np.searchsorted(rec.t, t_max_us,
                                                    side="right")))
                rec.t = rec.t[:n_keep]
                rec.trajectories = rec.trajectories[:, :n_keep]
            if verbose:
                print(f"[corr] simulated: lam={rec.lambda_val:.3f} "
                      f"({rec.trajectories.shape[0]} trials x "
                      f"{rec.trajectories.shape[1]} steps)")
            # keep only scalar metadata — pnt/pkt arrays are per-λ baggage
            rec.metadata = {k: v for k, v in rec.metadata.items()
                            if np.isscalar(v) or isinstance(v, str)}
            if "Omega" in rec.metadata:
                rec.metadata.setdefault("omega_DG", float(rec.metadata["Omega"]))
            correlations.append(compute_click_correlations_dense(
                rec, n_groups=n_groups, **estimator_kwargs,
            ))
        correlations.sort(key=lambda r: r.lambda_val)
        return correlations

    @staticmethod
    def _correlate_run_groups(
        runs_dir: Path,
        *,
        lambda_range,
        max_groups,
        t_max_us,
        estimator_kwargs: dict,
        verbose: bool,
    ) -> list:
        """Stream per-rate run groups into correlation results, one λ at a time.

        Only one run's click matrix is ever in memory; each run is a bootstrap
        resampling unit (mirroring the compressed source's run bootstrap).
        ``t_max_us`` truncates each record before the first-click anchoring.
        """
        import numpy as np

        from mpm_analysis.analysis.click_correlations import (
            compute_click_correlations_runs,
        )
        from mpm_analysis.io.experimental.click_record_npz import (
            iter_click_run_groups,
        )

        correlations = []
        for group in iter_click_run_groups(
            runs_dir, lambda_range=lambda_range, max_groups=max_groups,
        ):
            if verbose:
                print(f"[corr] json runs: {group['key']} "
                      f"(lam={group['lambda_val']:.3f}, "
                      f"{group['metadata']['n_run_files']} runs)")
            t = np.asarray(group["t"], dtype=float)
            runs = group["runs"]
            if t_max_us is not None:
                n_keep = max(2, int(np.searchsorted(t, t_max_us, side="right")))
                t = t[:n_keep]
                runs = (m[:, :n_keep] for m in runs)
            correlations.append(compute_click_correlations_runs(
                runs, t,
                lambda_val=group["lambda_val"], rate_khz=group["rate_khz"],
                metadata=group["metadata"], **estimator_kwargs,
            ))
        correlations.sort(key=lambda r: r.lambda_val)
        return correlations

    @staticmethod
    def _load(
        folder: Path,
        source: str,
        *,
        lambda_range,
        max_files,
        t_max_us,
        latest_only,
    ) -> list:
        if source == "compressed":
            from mpm_analysis.io.experimental.compressed_click_record import (
                CompressedClickLoader,
            )
            return CompressedClickLoader().load(
                folder, lambda_range=lambda_range, latest_only=latest_only,
            )
        if source == "json":
            from mpm_analysis.io.experimental.click_record_npz import (
                ClickRecordNpzLoader,
            )
            return ClickRecordNpzLoader().load(
                folder, lambda_range=lambda_range,
                max_files=max_files, t_max=t_max_us,
            )
        # "simulated" never reaches here — run() streams it via
        # _correlate_sim_stream / _correlations_from_shifted.
        raise ValueError(f"_load does not handle source {source!r}.")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    @staticmethod
    def _make_run_dir(
        folder: str | Path | None,
        source: str,
        out_dir: str | Path | None,
    ) -> Path:
        if out_dir is None:
            if folder is None:
                raise ValueError(
                    "out_dir is required when running from pre-loaded records."
                )
            folder = Path(folder)
            root = folder.parent if folder.is_file() else folder
            sub = _DEFAULT_SUBDIR.get(source, ".")
            base = root if sub == "." else root / sub
        else:
            base = Path(out_dir)
        run_dir = Path(base) / f"corr_run_{source}_{_timestamp()}"
        os.makedirs(win_long_path(run_dir), exist_ok=True)
        return run_dir

    @staticmethod
    def _save_correlations(
        correlations: list[ClickCorrelationResult],
        run_dir: Path,
        source: str,
    ) -> Path:
        """Save every per-λ correlation into one NPZ (lag axes may differ)."""
        payload: dict = {
            "lambda_values": np.array([r.lambda_val for r in correlations]),
            "r_bar": np.array([r.r_bar for r in correlations]),
            "dt_us": np.array([r.dt_us for r in correlations]),
            "rate_khz": np.array(
                [-1 if r.rate_khz is None else r.rate_khz for r in correlations]
            ),
        }
        meta = []
        for i, r in enumerate(correlations):
            payload[f"t_us_{i}"] = r.t_us
            payload[f"click_{i}"] = r.click
            payload[f"click_err_{i}"] = r.click_err
            payload[f"no_click_{i}"] = r.no_click
            payload[f"no_click_err_{i}"] = r.no_click_err
            meta.append(dict(window=r.window,
                             metadata={k: v for k, v in r.metadata.items()
                                       if isinstance(v, (str, int, float, bool))}))
        payload["per_lambda_json"] = np.frombuffer(
            json.dumps(meta).encode(), dtype=np.uint8,
        )
        path = _unique_path(
            run_dir / f"click_correlations_{source}_{_timestamp()}.npz"
        )
        np.savez_compressed(win_long_path(path), **payload)
        return path
