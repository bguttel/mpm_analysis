"""
dead_time_pipeline.py
---------------------
Sweep the detector dead-time compression coefficient ``c`` and track the
extracted poles and transitions.

For every ``c`` the time axis of the fitted observables is remapped —
``t' = t (1 - c p_click(t))`` (pointwise, default) or ``t' = t - c * int(p_click)``
(cumulative), see :mod:`mpm_analysis.analysis.dead_time` — while the y-data
stay as measured.  Where the pointwise remap folds (late, noisy bins of
``p_click``) the trace is truncated at its first fold and the cut is
reported; a λ left with too few points for the model order is NaN at that
``c``.  Two fitting routes run on the remapped records:

direct
    The same modal form (sum of ``order`` complex exponentials, conjugate
    pairs / real poles via the real-quadratic parametrisation) fitted
    directly on the non-uniform axis with ``RefineStep`` — i.e.
    :func:`mpm_analysis.analysis.matrix_pencil.refine_poles`, which never
    assumed uniform sampling.  ``c = 0`` starts from the raw-data MPM poles;
    every following ``c`` starts from the previous fit, so the sweep follows
    one continuous branch.

regrid_mpm  (fallback, optional)
    Each remapped trace is linearly interpolated back onto a uniform grid of
    the original spacing and the ordinary ``MPMStep`` (plus the same
    ``RefineStep``) runs on it.  Reported next to the direct route wherever
    both converge.

From every fit the decay rates / frequencies and the two transitions of the
standard analysis are read off exactly as in ``run.py``'s ``run_transitions``:
the exceptional point as the straight-line zero of the pair discriminant
``Delta = c1**2 - 4 c0`` (``linfit_zero_crossing``).  With ``n_boot > 0`` the
data bootstrap reruns
the direct fit on every resampled dataset (refine-only draws — the pencil is
never run on the non-uniform axis) and the per-realization EP quantiles are
stored as well.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mpm_analysis.analysis.dead_time import (
    DEFAULT_CLICK_KEY,
    REMAPS,
    compress_records,
    resample_uniform,
)
from mpm_analysis.data_types.observables import ObservableRecord


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CompressionTrack:
    """Poles and transitions of one fitting route along the ``c`` sweep.

    Arrays are indexed ``[c, lambda(, pole)]``.  ``poles`` are complex
    (``-decay + 1j * frequency``, 1/µs), branch-labelled consistently across
    ``c`` and λ.  ``lambda_ep`` is the straight-line zero of the
    discriminant and ``lambda_ep_first`` the raw first sign change; NaN
    where none exists.
    ``lambda_ep_quantiles`` (16/50/84 of the per-realization line fits) and
    the four pole half-width arrays are present only when a bootstrap ran.
    """
    method: str
    poles: np.ndarray                 # complex (n_c, n_lambda, order)
    delta: np.ndarray                 # (n_c, n_lambda)
    rmse: np.ndarray                  # (n_c, n_lambda)   fit residual rms
    lambda_ep: np.ndarray             # (n_c,)
    lambda_ep_first: np.ndarray       # (n_c,)
    lambda_ep_quantiles: np.ndarray | None = None   # (n_c, 3)
    decay_lower: np.ndarray | None = None           # (n_c, n_lambda, order)
    decay_upper: np.ndarray | None = None
    freq_lower: np.ndarray | None = None
    freq_upper: np.ndarray | None = None

    @property
    def n_c(self) -> int:
        return self.poles.shape[0]

    @property
    def order(self) -> int:
        return self.poles.shape[2]


@dataclass
class DeadTimeSweepResult:
    """Output of :class:`DeadTimeSweepPipeline`.

    Attributes
    ----------
    compress_coeffs:
        The swept ``c`` values, ascending.
    lambda_values:
        λ axis shared by every track.
    direct:
        Direct non-uniform fit track.
    fallback:
        Regrid + MPM track, or ``None`` when the fallback was not run.
    metadata, parameters:
        Provenance (data mode, source folder, scan number, observables,
        click key) and the fit configuration (order, half-width, λ cut,
        regrid spacing, bootstrap size, ``omega_DG``).
    """
    compress_coeffs: np.ndarray
    lambda_values: np.ndarray
    direct: CompressionTrack
    fallback: CompressionTrack | None
    metadata: dict
    parameters: dict

    @property
    def omega_s(self) -> float:
        """``omega_DG`` (rad/µs) for normalising poles; 1.0 when unknown."""
        val = self.parameters.get("omega_DG")
        return 1.0 if val is None else float(val)

    @property
    def tracks(self) -> list[CompressionTrack]:
        return [self.direct] + ([self.fallback] if self.fallback is not None else [])

    def track(self, method: str) -> CompressionTrack:
        """The track of one fitting route by name (``"direct"`` | ``"regrid_mpm"``)."""
        for tr in self.tracks:
            if tr.method == method:
                return tr
        raise ValueError(f"unknown fit method {method!r}; this sweep has "
                         f"{[t.method for t in self.tracks]}.")

    def index_of(self, c: float) -> int:
        """Index of the swept coefficient closest to ``c``."""
        return int(np.argmin(np.abs(self.compress_coeffs - float(c))))

    def to_analysis_result(self, ic: int, method: str = "direct",
                           *, lambda_max: float | None = None):
        """The pole spectrum at one ``c`` as a ``ZenoAnalysisResult``.

        Lets every standard consumer (``plot_spectrum_panels``, the JSON
        writer, …) work on a single slice of the sweep.  Half-widths are the bootstrap ones when a bootstrap
        ran, zero otherwise.
        """
        from mpm_analysis.data_types.analysis_result import ZenoAnalysisResult
        from mpm_analysis.data_types.poles import PolesResult

        tr = self.track(method)
        lam = self.lambda_values
        m = np.ones(len(lam), dtype=bool) if lambda_max is None else lam <= lambda_max
        P = tr.poles[ic][m]

        def _hw(a):
            return np.zeros(P.shape) if a is None else a[ic][m]

        poles = PolesResult(
            decay_rates=-P.real,
            decay_rates_lower=_hw(tr.decay_lower), decay_rates_upper=_hw(tr.decay_upper),
            frequencies=P.imag,
            frequencies_lower=_hw(tr.freq_lower), frequencies_upper=_hw(tr.freq_upper),
        )
        c = float(self.compress_coeffs[ic])
        return ZenoAnalysisResult(
            metadata={**self.metadata, "compress_coeff": c, "fit_method": method},
            parameters=dict(self.parameters),
            lambda_values=lam[m],
            poles=poles,
        )

    def summary_rows(self) -> list[dict]:
        """One dict per ``c``: transitions of every track (for tables / CSV)."""
        import warnings

        rows = []
        for i, c in enumerate(self.compress_coeffs):
            row = {"compress_coeff": float(c)}
            for tr in self.tracks:
                row[f"lambda_ep_{tr.method}"] = float(tr.lambda_ep[i])
                row[f"lambda_ep_first_{tr.method}"] = float(tr.lambda_ep_first[i])
                with warnings.catch_warnings():        # all-NaN row at a c → NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row[f"rmse_mean_{tr.method}"] = float(np.nanmean(tr.rmse[i]))
                row[f"n_lambda_fitted_{tr.method}"] = int(np.isfinite(tr.rmse[i]).sum())
                if tr.lambda_ep_quantiles is not None:
                    q16, q50, q84 = tr.lambda_ep_quantiles[i]
                    row[f"lambda_ep_q16_{tr.method}"] = float(q16)
                    row[f"lambda_ep_q50_{tr.method}"] = float(q50)
                    row[f"lambda_ep_q84_{tr.method}"] = float(q84)
            rows.append(row)
        return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _groups(records: list[ObservableRecord], keys: list[str]):
    """``[(lambda, t, Y)]`` exactly as ``MPMStep`` / ``RefineStep`` see them."""
    from mpm_analysis.data_types.observables import group_records_by_lambda

    if len(keys) == 1:
        slices = sorted(
            [r for r in records if r.observable_key == keys[0]],
            key=lambda r: r.lambda_val,
        )
        return [(r.lambda_val, r.t, r.signal) for r in slices]
    return [(lam, t, Y) for lam, t, Y, _ in group_records_by_lambda(records, keys)]


def _fit_rmse(records: list[ObservableRecord], keys: list[str],
              lambdas: np.ndarray, poles: list[np.ndarray]) -> np.ndarray:
    """Residual rms of the modal reconstruction per λ (NaN where the fit failed)."""
    from mpm_analysis.analysis.matrix_pencil import (
        reconstruct_signal, solve_linear_amplitudes,
    )

    by_lambda = {lam: (t, Y) for lam, t, Y in _groups(records, keys)}
    out = np.full(len(lambdas), np.nan)
    for i, (lam, p) in enumerate(zip(lambdas, poles)):
        p = np.asarray(p, dtype=complex)
        grp = by_lambda.get(lam)
        if grp is None or not np.all(np.isfinite(p)):
            continue
        t, Y = grp
        y_hat = reconstruct_signal(t, p, solve_linear_amplitudes(t, Y, p))
        out[i] = float(np.sqrt(np.mean((np.asarray(Y) - y_hat) ** 2)))
    return out


def _transitions(lambdas: np.ndarray, poles: np.ndarray, *,
                 half_width: float, lambda_max: float | None):
    """``(delta, lambda_ep, lambda_ep_first)`` from central poles.

    The same recipe as ``run.py``'s ``run_transitions``: ``Delta`` of the EP
    pair (followed along λ past the EP), its straight-line zero and raw
    first sign change.
    """
    from mpm_analysis.analysis.ksi_crossing import (
        discriminant_from_poles, find_zero_crossings, linfit_zero_crossing,
    )

    delta = discriminant_from_poles(poles)
    lam, d = np.asarray(lambdas, dtype=float), delta
    if lambda_max is not None:
        m = lam <= lambda_max
        lam, d = lam[m], d[m]
    lam_ep = linfit_zero_crossing(lam, d, half_width=half_width)
    first = find_zero_crossings(lam, d)
    return delta, float(lam_ep), float(first[0]) if first else np.nan


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DeadTimeSweepPipeline:
    """Sweep the dead-time compression coefficient and refit the poles.

    Parameters
    ----------
    records:
        Loaded records holding the fitted observable(s) AND the click
        probability (``click_key``) for every λ.
    compress_coeffs:
        The ``c`` values to sweep (sorted ascending internally).
    order:
        Number of poles.
    observable_keys:
        Observable(s) to fit — one key for the classic fit, several for the
        joint (shared-pole) fit.
    click_key:
        Observable key carrying ``p_click(t)``.
    remap:
        ``"pointwise"`` (``t (1 - c p)``) or ``"cumulative"`` (``t - c int(p)``).
    discriminant_half_width, lambda_max:
        Line-fit half-window and λ cut of the transition extraction (the
        ``DISCRIMINANT_HALFWIDTH`` / ``TRANSITION_LAMBDA_MAX`` of ``run.py``).
    conjugate_pairs, bounds_real:
        ``RefineStep`` configuration shared by both routes.
    fallback:
        Also run the regrid + MPM route.
    regrid_dt:
        Uniform spacing (µs) of the fallback grid; ``None`` → the raw
        records' own spacing.
    fallback_refine:
        Apply the same ``RefineStep`` after the fallback MPM (so the two
        routes differ only in the interpolation).
    n_boot, seed, n_jobs:
        Data-bootstrap size for the direct route (``0`` → central values
        only); reruns the refinement from the clean poles on every draw.
    initial_poles:
        Optional ``{lambda: poles}`` from an existing MPM result to seed
        ``c = 0`` with; λ's not covered fall back to a fresh MPM.
    metadata:
        Provenance to carry into the result.
    """

    def __init__(
        self,
        records: list[ObservableRecord],
        compress_coeffs,
        *,
        order: int = 3,
        observable_keys: list[str] | str = ("D",),
        click_key: str = DEFAULT_CLICK_KEY,
        remap: str = "pointwise",
        discriminant_half_width: float = 0.15,
        lambda_max: float | None = None,
        conjugate_pairs: bool = True,
        bounds_real: tuple[float, float] | None = (-np.inf, 0.0),
        fallback: bool = True,
        regrid_dt: float | None = None,
        fallback_refine: bool = True,
        n_boot: int = 0,
        seed: int | None = None,
        n_jobs: int = -1,
        initial_poles: dict[float, np.ndarray] | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.records = records
        self.compress_coeffs = np.array(sorted(float(c) for c in compress_coeffs))
        if self.compress_coeffs.size == 0:
            raise ValueError("DeadTimeSweepPipeline: compress_coeffs is empty.")
        self.order = int(order)
        self.keys = ([observable_keys] if isinstance(observable_keys, str)
                     else list(observable_keys))
        self.click_key = click_key
        if remap not in REMAPS:
            raise ValueError(f"DeadTimeSweepPipeline: unknown remap '{remap}'; expected {REMAPS}.")
        self.remap = remap
        self.half_width = float(discriminant_half_width)
        self.lambda_max = lambda_max
        self.conjugate_pairs = conjugate_pairs
        self.bounds_real = bounds_real
        self.fallback = fallback
        self.regrid_dt = regrid_dt
        self.fallback_refine = fallback_refine
        self.n_boot = int(n_boot)
        self.seed = seed
        self.n_jobs = n_jobs
        self.initial_poles = initial_poles
        self.metadata = dict(metadata or {})
        self._result: DeadTimeSweepResult | None = None

    # ------------------------------------------------------------------

    def _seed_state(self) -> dict:
        """``c = 0`` starting point: MPM poles of the raw records."""
        from mpm_analysis.analysis.steps import MPMStep

        state = MPMStep(order=self.order, observable_key=self.keys).process(
            self.records, {}
        )
        if self.initial_poles:
            n_hit = 0
            for i, lam in enumerate(state["lambdas"]):
                p = self.initial_poles.get(float(lam))
                if p is not None and np.all(np.isfinite(p)) and len(p) == self.order:
                    state["poles_raw"][i] = np.asarray(p, dtype=complex)
                    n_hit += 1
            print(f"  seed: {n_hit}/{len(state['lambdas'])} lambda points "
                  "initialised from the existing MPM poles")
        return state

    def _compress(self, c: float) -> list[ObservableRecord]:
        """Remapped records for ``c``; traces that fold are truncated and reported.

        A λ whose truncated trace is shorter than ``2 * order + 1`` points
        cannot support the model and is dropped for this ``c`` (its poles
        stay NaN) — printed, never silent.
        """
        recs = compress_records(self.records, c, click_key=self.click_key,
                                keys=self.keys, remap=self.remap, on_fold="truncate")
        n_min = 2 * self.order + 1
        cut = sorted({(r.lambda_val, r.metadata["fold_index"]) for r in recs
                      if "fold_index" in r.metadata})
        keep = [r for r in recs if r.t.size >= n_min]
        if cut:
            worst = min(cut, key=lambda x: x[1])
            print(f"    fold: {len(cut)} lambda(s) truncated at c={c:g} "
                  f"(kept {min(k for _, k in cut)}..{max(k for _, k in cut)} of "
                  f"{self.records[0].t.size} points; shortest at lambda={worst[0]:.3f})")
        dropped = sorted({r.lambda_val for r in recs} - {r.lambda_val for r in keep})
        if dropped:
            print(f"    dropped {len(dropped)} lambda(s) with < {n_min} points at c={c:g}: "
                  + ", ".join(f"{l:.3f}" for l in dropped))
        return keep

    def _empty_track(self, method: str, n_lam: int) -> CompressionTrack:
        n_c = len(self.compress_coeffs)
        return CompressionTrack(
            method=method,
            poles=np.full((n_c, n_lam, self.order), np.nan, dtype=complex),
            delta=np.full((n_c, n_lam), np.nan),
            rmse=np.full((n_c, n_lam), np.nan),
            lambda_ep=np.full(n_c, np.nan),
            lambda_ep_first=np.full(n_c, np.nan),
        )

    def _fill(self, track: CompressionTrack, ic: int, lambdas: np.ndarray,
              poles: list[np.ndarray], records: list[ObservableRecord]) -> None:
        P = np.array([np.asarray(p, dtype=complex) for p in poles])
        track.poles[ic] = P
        track.rmse[ic] = _fit_rmse(records, self.keys, lambdas, poles)
        (track.delta[ic], track.lambda_ep[ic],
         track.lambda_ep_first[ic]) = _transitions(
            lambdas, P, half_width=self.half_width, lambda_max=self.lambda_max,
        )

    def run(self) -> DeadTimeSweepResult:
        from mpm_analysis.analysis.ksi_crossing import discriminant_crossing_samples
        from mpm_analysis.analysis.pole_sorting import match_poles_hungarian
        from mpm_analysis.analysis.steps import BootstrapStep, MPMStep, RefineStep

        obs_key = self.keys[0] if len(self.keys) == 1 else "+".join(self.keys)
        refine = RefineStep(observable_key=self.keys,
                            conjugate_pairs=self.conjugate_pairs,
                            bounds_real=self.bounds_real)
        print(f"  > {refine.description()}")

        seed = self._seed_state()
        lambdas = np.asarray(seed["lambdas"], dtype=float)
        n_lam = len(lambdas)
        prev = [np.asarray(p, dtype=complex) for p in seed["poles_raw"]]

        direct = self._empty_track("direct", n_lam)
        fb = self._empty_track("regrid_mpm", n_lam) if self.fallback else None
        if self.n_boot > 0:
            direct.lambda_ep_quantiles = np.full((len(self.compress_coeffs), 3), np.nan)
            shape = direct.poles.shape
            direct.decay_lower = np.full(shape, np.nan)
            direct.decay_upper = np.full(shape, np.nan)
            direct.freq_lower = np.full(shape, np.nan)
            direct.freq_upper = np.full(shape, np.nan)

        raw_t = next(r.t for r in self.records if r.observable_key == self.keys[0])
        dt = float(self.regrid_dt) if self.regrid_dt is not None \
            else float(np.median(np.diff(raw_t)))

        for ic, c in enumerate(self.compress_coeffs):
            recs_c = self._compress(c)
            present = {r.lambda_val for r in recs_c}
            nan_poles = np.full(self.order, np.nan, dtype=complex)
            # --- direct fit on the non-uniform axis, seeded from the last c
            state = {
                "poles_raw": [p.copy() for p in prev],
                "amplitudes_raw": list(seed["amplitudes_raw"]),
                "lambdas": list(lambdas),
                "observable_key": obs_key,
                "observable_keys": self.keys,
            }
            state = refine.process(recs_c, state)
            # RefineStep passes a missing λ through untouched; it is NaN here.
            prev = [np.asarray(p, dtype=complex) if lam in present else nan_poles
                    for lam, p in zip(lambdas, state["poles_raw"])]
            self._fill(direct, ic, lambdas, prev, recs_c)
            msg = (f"  c={c:.3f}: direct lam_EP={direct.lambda_ep[ic]:.4f} "
                   f"rmse={np.nanmean(direct.rmse[ic]):.2e}")

            if self.n_boot > 0:
                state = BootstrapStep(n_boot=self.n_boot, seed=self.seed,
                                      mode="data", n_jobs=self.n_jobs,
                                      pencil=False).process(recs_c, state)
                pr = state["poles_result"]
                # The bootstrap ran on the λ's present at this c — align by λ.
                idx = [int(np.argmin(np.abs(lambdas - l)))
                       for l in pr.raw_samples.lambda_values]
                direct.decay_lower[ic, idx] = pr.decay_rates_lower
                direct.decay_upper[ic, idx] = pr.decay_rates_upper
                direct.freq_lower[ic, idx] = pr.frequencies_lower
                direct.freq_upper[ic, idx] = pr.frequencies_upper
                cross, *_ = discriminant_crossing_samples(
                    pr.raw_samples, lambda_max=self.lambda_max,
                    method="linfit", half_width=self.half_width,
                )
                if np.any(np.isfinite(cross)):
                    direct.lambda_ep_quantiles[ic] = np.nanpercentile(cross, [16, 50, 84])
                    q16, _, q84 = direct.lambda_ep_quantiles[ic]
                    msg += f" +/-{(q84 - q16) / 2:.4f}"

            # --- fallback: regrid to a uniform axis, ordinary MPM (+ refine)
            if fb is not None:
                recs_u = [u for u in (resample_uniform(r, dt) for r in recs_c)
                          if u.t.size >= 2 * self.order + 1]
                su = MPMStep(order=self.order, observable_key=self.keys).process(recs_u, {})
                if self.fallback_refine:
                    su = refine.process(recs_u, su)
                # Same branch labels as the direct route, so the tracks are
                # comparable pole by pole; λ's absent at this c stay NaN.
                by_lam = dict(zip(su["lambdas"], su["poles_raw"]))
                poles_u = [
                    match_poles_hungarian(np.asarray(by_lam[lam], dtype=complex), ref)
                    if lam in by_lam and np.all(np.isfinite(ref)) else nan_poles
                    for lam, ref in zip(lambdas, prev)
                ]
                self._fill(fb, ic, lambdas, poles_u, recs_u)
                msg += (f" | regrid lam_EP={fb.lambda_ep[ic]:.4f} "
                        f"({len(recs_u[0].t)} pts)")
            print(msg)

        omega = next((r.metadata.get("omega_DG") for r in self.records
                      if r.metadata.get("omega_DG") is not None), None)
        params = {
            "pole_order": self.order,
            "observable_keys": list(self.keys),
            "click_key": self.click_key,
            "remap": self.remap,
            "discriminant_half_width": self.half_width,
            "lambda_max": self.lambda_max,
            "conjugate_pairs": self.conjugate_pairs,
            "bounds_real": None if self.bounds_real is None else list(self.bounds_real),
            "regrid_dt": dt if fb is not None else None,
            "fallback_refine": self.fallback_refine,
            "n_bootstrap": self.n_boot,
            "seed": self.seed,
        }
        if omega is not None:
            params["omega_DG"] = float(omega)
        meta = {"analysis_type": "dead_time_sweep", **self.metadata}
        if "scan_number" not in meta:
            scans = {int(r.metadata["scan_number"]) for r in self.records
                     if r.metadata.get("scan_number") is not None}
            if len(scans) == 1:
                meta["scan_number"] = scans.pop()
        self._result = DeadTimeSweepResult(
            compress_coeffs=self.compress_coeffs, lambda_values=lambdas,
            direct=direct, fallback=fb, metadata=meta, parameters=params,
        )
        return self._result

    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path, stem: str | None = None) -> Path:
        """Write the sweep as JSON plus a one-row-per-``c`` CSV summary.

        Returns the JSON path; the CSV sits next to it with the same stem.
        """
        import csv
        import os

        from mpm_analysis.io.analysis_json import save_dead_time_sweep
        from mpm_analysis.io.file_naming import build_result_filename
        from mpm_analysis.utils.windows_paths import win_long_path

        if self._result is None:
            raise RuntimeError("Call run() before save().")
        res = self._result
        output_dir = Path(output_dir)
        os.makedirs(win_long_path(output_dir), exist_ok=True)
        if stem is None:
            lam = res.lambda_values
            scan = res.metadata.get("scan_number")
            stem = build_result_filename(
                analysis_type="dead_time_sweep",
                data_mode=str(res.metadata.get("data_mode", "unknown")),
                lambda_range=(float(lam.min()), float(lam.max())),
                n_lambda=len(lam),
                scan_number=int(scan) if scan is not None else None,
                extension="",
            ).rstrip(".")
        json_path = output_dir / f"{stem}.json"
        save_dead_time_sweep(res, json_path)

        rows = res.summary_rows()
        with open(win_long_path(output_dir / f"{stem}.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return json_path
