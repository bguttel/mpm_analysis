"""
kspace_pipeline.py
------------------
Unified k-space pole-analysis pipeline.  One class drives both data sources
behind a simple selector:

    source="json"        -> KSpaceJSONLoader        (experiment scan JSONs)
    source="compressed"  -> CompressedClickKSpaceLoader (compressed click records)

Both loaders return ``list[KSpaceRecord]`` and feed the same
``KSpaceAnalysis`` -> ``KSpacePolesResult``.  This pipeline adds: per-source
default output folder, timestamped non-clobbering save, the full figure set
(via ``kspace.save_kspace_figures``), optional **run-level bootstrap** error
bars (compressed source), optional **theory overlay** + **braiding
characterization**, and optional auto-opening of the saved figures.

Replaces the duplicated load->analyze->save->plot logic that used to live in
``parity_transition.py`` and ``compressed_click_pipeline.py``.
"""
from __future__ import annotations

import os
import webbrowser
from pathlib import Path

import numpy as np

from mpm_analysis.kspace import (
    KSpaceAnalysis, PoleOrderConfig, KSpacePolesResult,
    get_kspace_loader, CompressedClickKSpaceLoader, save_kspace_figures,
)
from mpm_analysis.io.experimental.compressed_click_record import (
    _timestamp, _unique_path,
)
from mpm_analysis.utils.windows_paths import win_long_path

_DEFAULT_SUBDIR = {"json": "pole_analysis", "compressed": ".", "simulated": "."}


def _open_paths(paths) -> None:
    """Open saved figures so they are easy to see (PNG via OS, HTML via browser)."""
    for p in paths:
        p = Path(p)
        try:
            if p.suffix == ".html":
                webbrowser.open(p.resolve().as_uri())
            elif os.name == "nt":
                os.startfile(win_long_path(p))      # noqa: SLF001 (Windows only)
            else:
                webbrowser.open(p.resolve().as_uri())
        except Exception as exc:                    # opening is best-effort
            print(f"  [open] could not open {p.name}: {exc}")


class KSpacePipeline:
    """Load -> analyze -> save -> plot, for either k-space source."""

    def run(
        self,
        folder: str | Path,
        *,
        source: str = "json",
        out_dir: str | Path | None = None,
        # loading
        lambda_range: tuple[float, float] | None = None,
        t_max: float | None = None,
        k_index_range: tuple[int | None, int | None] | None = (1, 100),
        min_alive_frac: float | None = 0.5,
        k_grid_n: int | None = None,
        k_endpoint_eps: float = 0.0,
        max_files: int | None = None,
        # analysis
        max_order: int = 2,
        min_order: int = 2,
        threshold: float = 0.5,
        n_bootstrap: int = 1,
        signal_component: str = "complex",
        bootstrap_over_runs: bool = False,
        # theory / braiding
        braiding: bool = True,
        theory: dict | None = None,
        # output / figures
        save_result: bool = True,
        save_figures: bool = True,
        open_figures: bool = False,
        grid_n_panels: int = 16,
        grid_n_cols: int = 4,
        raw_component: str = "real",
        include_3d: bool = True,
        include_html: bool = True,
        scan_number: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run the k-space analysis for ``source`` on ``folder``.

        Returns ``dict(records, result, result_path, figures, braiding,
        theory_poles, out_dir)``.
        """
        folder = Path(folder)
        loader_cls = get_kspace_loader(source)
        load_kwargs = dict(lambda_range=lambda_range, t_max=t_max,
                           k_index_range=k_index_range)
        if source == "json":
            load_kwargs["max_files"] = max_files
        elif source == "compressed":
            load_kwargs["min_alive_frac"] = min_alive_frac
            load_kwargs["k_grid_n"] = k_grid_n
            load_kwargs["k_endpoint_eps"] = k_endpoint_eps
        # "simulated" only accepts lambda_range / t_max / k_index_range
        records = loader_cls().load(folder, **load_kwargs)
        if not records:
            raise RuntimeError(f"No records loaded from {folder} (source={source}).")
        if verbose:
            print(f"[kspace] source={source}: {len(records)} records "
                  f"(λ {records[0].lambda_val:.3f}..{records[-1].lambda_val:.3f}), "
                  f"signal {records[0].signal.shape}")

        cfg = PoleOrderConfig(max_order=max_order, min_order=min_order,
                              threshold=threshold)
        result = KSpaceAnalysis(pole_order_config=cfg, n_bootstrap=n_bootstrap,
                                signal_component=signal_component).run(records)
        if scan_number is None:
            scan_number = records[0].metadata.get("scan_number")
        result.metadata.setdefault("source", source)

        # ---- optional run-level bootstrap → per-pole error bars -------------
        if bootstrap_over_runs:
            if source != "compressed":
                raise ValueError(
                    "bootstrap_over_runs requires source='compressed' "
                    "(per-run statistics live only in compressed records)."
                )
            result.poles_err = self._run_bootstrap(
                folder, result, cfg, signal_component,
                lambda_range, t_max, k_index_range, min_alive_frac,
                k_grid_n, k_endpoint_eps,
                verbose,
            )

        # ---- theory poles + braiding ---------------------------------------
        theory_poles = None
        braiding_exp = braiding_theory = None
        if braiding or theory is not None:
            from mpm_analysis.analysis import braiding as braid
            if braiding:
                braiding_exp = braid.experimental_braiding(result)
                if verbose:
                    braid.print_braiding_summary(braiding_exp, label="experiment")
            if theory is not None:
                tp = dict(theory)
                Omega = tp.pop("Omega", None)
                if Omega is None:
                    Omega = records[0].metadata.get("omega_DG")
                if Omega is None:
                    raise ValueError(
                        "Theory overlay needs Omega (rad/µs): pass theory={'Omega':…} "
                        "or ensure records carry omega_DG metadata."
                    )
                theory_path = tp.pop("theory_path", None)
                # overlay poles on the (possibly sliced) analysis k-grid
                theory_poles = braid.theory_poles_on_grid(
                    result.lambda_values, result.k_array, Omega=Omega,
                    theory_path=theory_path, **tp,
                )
                # braiding on the FULL closed loop k∈[0,2π] -> integer ν
                braiding_theory = braid.theory_braiding(
                    result.lambda_values, Omega=Omega,
                    theory_path=theory_path, **tp,
                )
                if verbose:
                    braid.print_braiding_summary(braiding_theory, label="theory")

        # ---- output location ------------------------------------------------
        # Each run gets its own timestamped sub-folder so figures/results from
        # different runs never mix together.
        if out_dir is None:
            # the simulated source points at a single .npz file, not a folder
            root = folder.parent if folder.is_file() else folder
            sub = _DEFAULT_SUBDIR.get(source, ".")
            base = root if sub == "." else root / sub
        else:
            base = Path(out_dir)
        out_dir = Path(base) / f"kspace_run_{source}_{_timestamp()}"
        os.makedirs(win_long_path(out_dir), exist_ok=True)
        if verbose:
            print(f"[kspace] run folder: {out_dir}")

        result_path = None
        if save_result:
            tag = "kspace_poles" if source == "json" else "kspace_poles_compressed"
            result_path = _unique_path(out_dir / f"{tag}_{_timestamp()}.npz")
            result.save(result_path)
            if verbose:
                print(f"[kspace] saved {result_path.name}")

        figures = []
        if save_figures:
            figures = save_kspace_figures(
                result, out_dir, records=records, source_tag=f"kspace_{source}",
                scan_number=scan_number, grid_n_panels=grid_n_panels,
                grid_n_cols=grid_n_cols, raw_component=raw_component,
                include_3d=include_3d, include_html=include_html,
                theory_poles=theory_poles, braiding=braiding_exp,
                braiding_theory=braiding_theory,
            )
            if open_figures:
                _open_paths(figures)

        return dict(records=records, result=result, result_path=result_path,
                    figures=figures, braiding=braiding_exp,
                    theory_poles=theory_poles, out_dir=out_dir)

    # ------------------------------------------------------------------
    @staticmethod
    def _run_bootstrap(folder, agg_result, cfg, signal_component,
                       lambda_range, t_max, k_index_range, min_alive_frac,
                       k_grid_n, k_endpoint_eps,
                       verbose) -> np.ndarray:
        """Run one pole analysis per individual run; return per-pole std (complex).

        ``.real`` = std of Re(pole), ``.imag`` = std of Im(pole), aligned to the
        aggregate result's pole ordering per (λ, k) cell.
        """
        from scipy.optimize import linear_sum_assignment

        per_run = CompressedClickKSpaceLoader().load_per_run(
            folder, lambda_range=lambda_range, t_max=t_max,
            k_index_range=k_index_range, min_alive_frac=min_alive_frac,
            k_grid_n=k_grid_n, k_endpoint_eps=k_endpoint_eps,
        )
        if verbose:
            print(f"[kspace] run-bootstrap over {len(per_run)} runs ...")
        agg = agg_result.poles                          # (n_λ, n_k, order)
        n_l, n_k, order = agg.shape
        stack = np.full((len(per_run), n_l, n_k, order), np.nan, dtype=complex)
        for r, recs in enumerate(per_run):
            res_r = KSpaceAnalysis(pole_order_config=cfg, n_bootstrap=1,
                                   signal_component=signal_component).run(recs)
            pr = res_r.poles                            # (n_λ, n_k, order)
            # align per-cell pole identity to the aggregate ordering
            for li in range(n_l):
                for ki in range(n_k):
                    a = agg[li, ki]; b = pr[li, ki]
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                        cost = np.abs(b[:, None] - a[None, :])
                        rows, cols = linear_sum_assignment(cost)
                        aligned = np.full(order, np.nan, dtype=complex)
                        aligned[cols] = b[rows]
                        stack[r, li, ki] = aligned
                    else:
                        stack[r, li, ki] = b
        std_re = np.nanstd(stack.real, axis=0)
        std_im = np.nanstd(stack.imag, axis=0)
        return std_re + 1j * std_im
