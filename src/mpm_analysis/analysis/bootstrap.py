"""
bootstrap.py
------------
Bootstrap uncertainty quantification for the MPM pole pipeline.

Strategy: for each lambda slice, fit a clean model to the data, resample
residuals, and re-extract poles.  Collecting n_boot pole estimates gives an
empirical distribution from which we compute median ± asymmetric uncertainties.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.data_types.poles import PolesResult
from mpm_analysis.data_types.pole_samples import PoleSamples
from mpm_analysis.analysis.pole_sorting import (
    sort_poles_canonical,
    enforce_pole_count,
    match_poles_hungarian,
)


# ---------------------------------------------------------------------------
# Internal result type (per-lambda bootstrap draws)
# ---------------------------------------------------------------------------

@dataclass
class _LambdaBootstrapResult:
    """Bootstrap distribution for a single lambda slice."""
    lambda_val: float
    boot_real: np.ndarray   # (n_boot, order)
    boot_imag: np.ndarray   # (n_boot, order)
    boot_params: np.ndarray | None = None   # (n_boot, n_par) quadratic params


# ---------------------------------------------------------------------------
# Core bootstrap function
# ---------------------------------------------------------------------------

def run_bootstrap(
    records: list[ObservableRecord],
    order: int = 3,
    n_boot: int = 1000,
    observable_key: str | list[str] | None = None,
    *,
    L: int | None = None,
    L_used: "list[int] | None" = None,
    rng: np.random.Generator | None = None,
    clean_poles: "list[np.ndarray] | None" = None,
) -> list[_LambdaBootstrapResult]:
    """Run bootstrap pole extraction for each lambda slice.

    A slice that cannot be fitted (too few time points for ``order``, or a
    numerically failed pencil) yields all-NaN draws and a warning instead of
    aborting the scan — the same per-slice tolerance ``MPMStep`` has.

    Parameters
    ----------
    records:
        List of ``ObservableRecord`` objects.
    order:
        Number of poles to extract.
    n_boot:
        Number of bootstrap resamples per lambda slice.
    observable_key:
        Which observable to analyse.  If ``None``, derived from first record.
        A list of keys runs the joint (multi-signal) pencil per draw: one
        time-index resample is applied to every observable's residuals, so
        equal-time cross-observable correlations are preserved and the error
        bars refer to the common pole set.
    L:
        Pencil parameter override applied to **every** slice.  Use only when
        all slices share a length; prefer ``L_used``.
    L_used:
        Per-slice pencil parameters (aligned with the lambda-sorted slices,
        as produced by ``MPMStep``).  Takes precedence over ``L``.  Values
        that do not fit a given slice are ignored in favour of the automatic
        choice — slices may differ in length.
    rng:
        Numpy random generator (``None`` → fresh default RNG).
    clean_poles:
        Optional per-slice clean poles (aligned with the lambda-sorted
        slices, e.g. ``state["poles_raw"]`` after a ``RefineStep``).  When
        given they define the clean model whose residuals are resampled, and
        every draw's poles are Hungarian-matched to them so bootstrap indices
        follow the same branches.  Per-draw refits remain plain MPM (a
        nonlinear refinement per draw would multiply runtime ~100×); the
        refined model only shapes the residual pool and the branch labels.

    Returns
    -------
    list[_LambdaBootstrapResult]
        One result per lambda slice, sorted by lambda.
    """
    from mpm_analysis.analysis.matrix_pencil import (
        matrix_pencil_method,
        solve_linear_amplitudes,
        reconstruct_signal,
        pack_pole_params,
    )

    from mpm_analysis.data_types.observables import group_records_by_lambda

    if rng is None:
        rng = np.random.default_rng()

    if observable_key is None:
        if not records:
            raise ValueError("records is empty; cannot derive observable_key.")
        observable_key = records[0].observable_key

    joint = not isinstance(observable_key, str)
    if joint and len(observable_key) == 1:
        observable_key = observable_key[0]
        joint = False

    if joint:
        keys = list(observable_key)
        groups = group_records_by_lambda(records, keys)
    else:
        slices = [r for r in records if r.observable_key == observable_key]
        slices.sort(key=lambda r: r.lambda_val)
        groups = [(r.lambda_val, r.t, r.signal, [r]) for r in slices]

    results: list[_LambdaBootstrapResult] = []

    for i, (lam, t, y, _recs) in enumerate(groups):
        N = np.atleast_2d(y).shape[1]

        # Per-slice pencil parameter: slices can differ in length (e.g. an
        # alive-fraction cut), so a neighbour's L may not fit this one.
        L_i = L
        if L_used is not None and i < len(L_used):
            L_i = int(L_used[i]) or None
        if L_i is not None and not 1 <= L_i <= N - 1:
            L_i = None                      # fall back to the automatic choice

        boot_real = np.full((n_boot, order), np.nan, dtype=float)
        boot_imag = np.full((n_boot, order), np.nan, dtype=float)
        boot_params = np.full((n_boot, order), np.nan, dtype=float)
        try:
            poles_clean, _, L_fit = matrix_pencil_method(t, y, order, L=L_i)
            poles_clean = sort_poles_canonical(poles_clean)
            # A refined/branch-tracked clean model, when available, defines
            # the residual pool and the per-draw branch reference.
            if clean_poles is not None and i < len(clean_poles):
                cp = np.asarray(clean_poles[i], dtype=complex)
                if np.all(np.isfinite(cp)):
                    poles_clean = cp
            amps_clean = solve_linear_amplitudes(t, y, poles_clean)
            y_clean = reconstruct_signal(t, poles_clean, amps_clean)
            residuals = y - y_clean

            for b in range(n_boot):
                if joint:
                    # One time-index draw shared by all observables: keeps
                    # equal-time cross-observable residual correlation.
                    idx = rng.integers(0, N, size=N)
                    y_boot = y_clean + residuals[:, idx]
                else:
                    y_boot = y_clean + rng.choice(
                        residuals, size=len(residuals), replace=True
                    )
                poles_b, _, _ = matrix_pencil_method(t, y_boot, order, L=L_fit)
                # Hungarian match to the clean poles: draw index i means the
                # same branch as clean pole i (no label switching).
                poles_b = match_poles_hungarian(
                    enforce_pole_count(poles_b, order), poles_clean
                )
                boot_real[b] = poles_b.real
                boot_imag[b] = poles_b.imag
                if np.all(np.isfinite(poles_b)):
                    boot_params[b] = pack_pole_params(poles_b)
        except Exception as exc:
            print(f"Warning: bootstrap failed for lam={lam:.3f} "
                  f"({N} points, order {order}): {exc}")

        results.append(
            _LambdaBootstrapResult(
                lambda_val=lam,
                boot_real=boot_real,
                boot_imag=boot_imag,
                boot_params=boot_params,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Data-resampling bootstrap (measurement statistics)
# ---------------------------------------------------------------------------

def bootstrap_ens_avg_measurements(
    stats: dict,
    keys: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """One multinomial resample of the ens_avg tomography outcomes.

    The four mutually exclusive outcomes ``D/G/B/error`` come from the same
    ``N_shots`` experimental shots, so each time point is resampled as ONE
    multinomial draw — never as independent Gaussians, which would break the
    multinomial correlations.

    Parameters
    ----------
    stats:
        ``{"P": (4, N) probabilities [D, G, B, error], "channels": [...],
        "N_shots": int}`` as attached by ``EnsembleAverageLoader``.
    keys:
        Observable keys to reconstruct (``"D"``, ``"G"``, ``"B"``,
        ``"error"``, ``"Z"`` = D − G) — the same mapping the loader's
        extractors use on the real data.
    rng:
        Numpy random generator.

    Returns
    -------
    np.ndarray
        Resampled observables, shape ``(len(keys), N)``.
    """
    P = np.clip(np.asarray(stats["P"], dtype=float), 0.0, None)  # (4, N)
    channels = list(stats["channels"])
    n_shots = int(stats["N_shots"])
    col_sum = P.sum(axis=0)
    P = P / np.where(col_sum > 0, col_sum, 1.0)

    counts = rng.multinomial(n_shots, P.T)          # (N, n_channels)
    Pb = counts.T.astype(float) / n_shots           # (n_channels, N)
    ch = {c: Pb[i] for i, c in enumerate(channels)}

    rows = []
    for key in keys:
        if key in ch:
            rows.append(ch[key])
        elif key == "Z":
            rows.append(ch["D"] - ch["G"])
        else:
            raise KeyError(
                f"bootstrap_ens_avg_measurements: no resampling rule for "
                f"observable '{key}' (channels: {channels} + 'Z')."
            )
    return np.vstack(rows)


# Survival channels: observable key -> (fraction field, attempts field).
# Each tomography experiment has its OWN survival, measured on its own
# attempts: the dedicated run uses all 6 axis settings (navg*6), the phase-axis
# run 4 settings (navg*4), the z run 2 settings (navg*2).
_POST_SELECTED_SURVIVAL = {
    "S": ("survival", "N_survival"),
    "survival": ("survival", "N_survival"),      # legacy key name
    "SX": ("SX", "N_attempts_SX"),
    "SZ": ("SZ", "N_attempts_SZ"),
    # Conditioned raw-record survivals (run_conditional_p0.py write-back):
    # their attempt counts are per-point arrays — the class sizes vary with t.
    "S_prev_click": ("S_prev_click", "N_attempts_S_prev_click"),
    "S_prev_noclick": ("S_prev_noclick", "N_attempts_S_prev_noclick"),
    # Segment-pooled survival (run_pooled_p0.py write-back).  Its points share
    # shots, so when the stats carry the first-click histogram
    # (``pooled_counts``) the draw is one multinomial per axis and segment
    # (see ``_resample_pooled_survival``); the per-point binomial below is
    # only the fallback for stats without the histogram.
    "S_pooled": ("S_pooled", "N_attempts_S_pooled"),
}


def _resample_pooled_survival(stats: dict, rng: np.random.Generator) -> np.ndarray:
    """One shot-noise resample of the segment-pooled survival curve.

    ``stats["pooled_counts"]`` is the ``(n_axes, n_seg, n_seg + 1)``
    first-bright-bin histogram (slot 0 = never bright, slot ``k`` = first
    bright at bin ``k``).  A shot's slot fixes its success at every time
    point, so resampling each (axis, segment) row multinomially and
    re-pooling reproduces the joint shot-noise distribution of the whole
    curve — including the correlation between consecutive points, which a
    per-point binomial would drop.

    The curve is rebuilt on the file's full time axis and then cut by
    ``stats["pooled_mask"]`` (the loader's ``t_max`` / drop mask).  The
    ``t = 0`` anchor (every shot survives; ``1`` by construction, or
    ``1 - init_error`` when the loader rewrote it) is drawn binomially on the
    whole file's shot count, like the other survival channels.
    """
    from mpm_analysis.io.experimental.click_record_npz import (
        pooled_survival_from_counts,
    )
    counts = np.asarray(stats["pooled_counts"], dtype=np.int64)
    shape = counts.shape
    n = counts.sum(axis=-1)                                   # (A, n_seg)
    P = counts / np.maximum(n, 1)[..., None]
    draw = rng.multinomial(n.ravel(), P.reshape(-1, shape[-1])).reshape(shape)
    frac, _, n_att = pooled_survival_from_counts(draw)

    mask = stats.get("pooled_mask")
    measured = np.asarray(stats["S_pooled"], dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        frac, n_att = frac[mask], n_att[mask]
        first_kept = bool(mask[0])
    else:
        first_kept = True
    if first_kept and frac.size:
        p0 = float(np.clip(measured[0], 0.0, 1.0))
        frac[0] = rng.binomial(int(n_att[0]), p0) / max(int(n_att[0]), 1)
    return frac

# Product channels: observable key -> (Bloch field, conditional-count field,
# survival channel that multiplies it).
_POST_SELECTED_TOMOGRAPHY = {
    "X": ("X", "N_X", "SX"),
    "Y": ("Y", "N_Y", "SX"),      # orthogonal phase axis: same experiment as X
    "Z": ("Z", "N_Z", "SZ"),
}


def bootstrap_post_selected_measurements(
    stats: dict,
    keys: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """One resample of the post_selected measurements.

    Each tomography axis is measured in its own experiment, on its own
    attempts, so every experiment gets its own two-stage draw:

    1. survival, on that experiment's total attempts::

           S  : n* ~ Binomial(navg*6, s)   / (navg*6)   (dedicated run)
           SX : n* ~ Binomial(navg*4, s_X) / (navg*4)   (phase-axis run)
           SZ : n* ~ Binomial(navg*2, s_Z) / (navg*2)   (z-axis run)

    2. the conditional Bloch coordinate, on the surviving trajectories::

           q = (1+V)/2,  n* ~ Binomial(N_V[t], q),  V* = 2 n*/N_V[t] − 1

       with the actual time-dependent ``N_V[t]``.

    3. the MPM channel is the product ``S_exp* · V*``.

    CRITICAL: the product uses the survival of *its own* experiment, and the
    draws are cached, so a run that fits ``["SZ", "SX", "Z", "X"]`` gets the
    same ``SX*`` in both the ``SX`` channel and the ``X`` product (and the
    same ``SZ*`` in ``SZ`` and ``Z``).  Mixing in the dedicated survival
    instead would impose a correlation the experiment does not have.

    Parameters
    ----------
    stats:
        ``{"survival"/"SX"/"SZ": (N,), "N_survival"/"N_attempts_SX"/
        "N_attempts_SZ": int, "X"/"Y"/"Z": (N,), "N_X"/"N_Y"/"N_Z": (N,)}``
        as attached by ``PostSelectedLoader``.  ``Y`` and the per-experiment
        survival fields are optional; without them the products fall back to
        the dedicated survival.
    keys:
        Observable keys to reconstruct: survival channels ``"S"``
        (``"survival"``), ``"SX"``, ``"SZ"``; product channels ``"X"``,
        ``"Y"``, ``"Z"`` — matching the loader extractors.
    rng:
        Numpy random generator.

    Returns
    -------
    np.ndarray
        Resampled observables, shape ``(len(keys), N)``.
    """
    survival_cache: dict[str, np.ndarray] = {}
    tomo_cache: dict[str, np.ndarray] = {}

    def _survival(skey: str) -> np.ndarray:
        """Binomial draw of one experiment's survival on its own attempts."""
        if skey in survival_cache:
            return survival_cache[skey]
        if skey == "S_pooled" and stats.get("pooled_counts") is not None:
            survival_cache[skey] = _resample_pooled_survival(stats, rng)
            return survival_cache[skey]
        frac_field, n_field = _POST_SELECTED_SURVIVAL[skey]
        if stats.get(frac_field) is None or stats.get(n_field) is None:
            if skey in ("SX", "SZ"):
                # Older stats dicts carry only the dedicated survival: fall
                # back to it rather than inventing per-experiment attempts.
                return _survival("survival")
            raise KeyError(
                "bootstrap_post_selected_measurements: bootstrap_stats "
                f"lacks '{frac_field}'/'{n_field}'."
            )
        s = np.clip(np.asarray(stats[frac_field], dtype=float), 0.0, 1.0)
        # Attempts: a scalar (one count for the whole curve) or a per-point
        # array (the conditioned channels, whose class sizes vary with t).
        n = np.asarray(stats[n_field], dtype=np.int64)
        safe = n > 0
        n_safe = np.where(safe, n, 1)
        draw = rng.binomial(n_safe, s) / n_safe
        survival_cache[skey] = np.where(safe, draw, s)
        return survival_cache[skey]

    def _tomo(key: str) -> np.ndarray:
        """Binomial draw of a conditional Bloch coordinate in [-1, 1]."""
        if key in tomo_cache:
            return tomo_cache[key]
        val_field, n_field, _ = _POST_SELECTED_TOMOGRAPHY[key]
        missing = [f for f in (val_field, n_field) if stats.get(f) is None]
        if missing:
            raise KeyError(
                f"bootstrap_post_selected_measurements: observable '{key}' "
                f"needs {missing} in bootstrap_stats, which the loader did not "
                "provide."
            )
        v = np.clip(np.asarray(stats[val_field], dtype=float), -1.0, 1.0)
        n = np.asarray(stats[n_field], dtype=int)
        safe = n > 0
        q = (1.0 + v) / 2.0
        n_plus = rng.binomial(np.where(safe, n, 1), q)
        v_boot = 2.0 * n_plus / np.where(safe, n, 1) - 1.0
        tomo_cache[key] = np.where(safe, v_boot, v)  # no shots → keep measured
        return tomo_cache[key]

    rows = []
    for key in keys:
        if key in _POST_SELECTED_SURVIVAL:
            rows.append(_survival(key))
            continue
        if key not in _POST_SELECTED_TOMOGRAPHY:
            raise KeyError(
                f"bootstrap_post_selected_measurements: no resampling rule "
                f"for observable '{key}' (supported: "
                f"{', '.join(sorted(set(_POST_SELECTED_SURVIVAL) | set(_POST_SELECTED_TOMOGRAPHY)))})."
            )
        _, _, skey = _POST_SELECTED_TOMOGRAPHY[key]
        # The product carries its OWN experiment's survival, and the draws are
        # cached: the same SX* multiplies the X product and is returned for a
        # standalone "SX" channel, so their correlation is preserved.
        rows.append(_survival(skey) * _tomo(key))
    return np.vstack(rows)


_MEASUREMENT_RESAMPLERS = {
    "ens_avg": bootstrap_ens_avg_measurements,
    "post_selected": bootstrap_post_selected_measurements,
}


def _data_bootstrap_worker(payload: dict) -> tuple:
    """One lambda slice of the data bootstrap (top-level for process pools).

    Per draw: resample measurement statistics → reconstruct observables →
    stacked common-pole MPM → nonlinear LSQ refinement (when configured) →
    Hungarian match to the clean poles → poles + quadratic parameters.

    Returns ``(boot_real, boot_imag, boot_params, failure)`` where ``failure``
    is ``None`` or ``(n_failed, first_error)`` — never swallow errors
    silently, or a systematically broken bootstrap looks like a valid
    all-NaN result.
    """
    from mpm_analysis.analysis.matrix_pencil import (
        matrix_pencil_method,
        refine_poles,
        pack_pole_params,
    )

    t = payload["t"]
    order = payload["order"]
    n_boot = payload["n_boot"]
    keys = payload["keys"]
    stats = payload["stats"]
    refine_settings = payload["refine_settings"]
    rng = np.random.default_rng(payload["seed_seq"])
    resample = _MEASUREMENT_RESAMPLERS[stats["data_mode"]]

    boot_real = np.full((n_boot, order), np.nan, dtype=float)
    boot_imag = np.full((n_boot, order), np.nan, dtype=float)
    boot_params = np.full((n_boot, order), np.nan, dtype=float)

    clean = payload["clean_poles"]
    pencil = payload.get("pencil", True)
    if pencil:
        # Clean fit: only to pin the pencil parameter and (as fallback) the
        # Hungarian reference.  The central result is NOT recomputed here.
        try:
            poles_ref, _, L_fit = matrix_pencil_method(t, payload["Y"], order,
                                                       L=payload["L"])
            poles_ref = sort_poles_canonical(poles_ref)
        except Exception as exc:
            return (boot_real, boot_imag, boot_params,
                    (n_boot, f"clean fit failed: {type(exc).__name__}: {exc}"))
        if clean is not None and np.all(np.isfinite(clean)):
            poles_ref = np.asarray(clean, dtype=complex)
    else:
        # Refine-only draws (non-uniform time axis): every draw starts from
        # the clean poles and the pencil is never run.
        if clean is None or not np.all(np.isfinite(clean)):
            return (boot_real, boot_imag, boot_params,
                    (n_boot, "refine-only bootstrap needs finite clean poles"))
        poles_ref = np.asarray(clean, dtype=complex)

    single = payload["single"]
    n_failed = 0
    first_error: str | None = None
    for b in range(n_boot):
        try:
            y_boot = resample(stats, keys, rng)
            if single:
                y_boot = y_boot[0]
            if pencil:
                poles_b, _, _ = matrix_pencil_method(t, y_boot, order, L=L_fit)
                poles_b = enforce_pole_count(poles_b, order)
            else:
                poles_b = poles_ref.copy()
            if refine_settings is not None and np.all(np.isfinite(poles_b)):
                poles_b = refine_poles(
                    t, y_boot, poles_b,
                    conjugate_pairs=refine_settings["conjugate_pairs"],
                    bounds_real=refine_settings["bounds_real"],
                    bounds_imag=refine_settings["bounds_imag"],
                )
            poles_b = match_poles_hungarian(poles_b, poles_ref)
            boot_real[b] = poles_b.real
            boot_imag[b] = poles_b.imag
            if np.all(np.isfinite(poles_b)):
                boot_params[b] = pack_pole_params(poles_b)
        except Exception as exc:    # NaN row for this draw — but reported
            n_failed += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
    failure = (n_failed, first_error) if n_failed else None
    return boot_real, boot_imag, boot_params, failure


def run_data_bootstrap(
    records: list[ObservableRecord],
    order: int = 3,
    n_boot: int = 1000,
    observable_key: str | list[str] | None = None,
    *,
    L: int | None = None,
    L_used: "list[int] | None" = None,
    seed: int | None = None,
    clean_poles: "list[np.ndarray] | None" = None,
    refine_settings: dict | None = None,
    n_jobs: int = -1,
    pencil: bool = True,
) -> list[_LambdaBootstrapResult]:
    """Data-resampling bootstrap: resample the measurement statistics.

    Each draw is a complete synthetic repeat of one lambda point's
    measurement: the raw counting statistics are resampled (multinomial for
    ens_avg, binomial for post_selected — see
    ``bootstrap_ens_avg_measurements`` / ``bootstrap_post_selected_measurements``),
    the observables are reconstructed with the same mapping the loaders use,
    and the full analysis (stacked MPM + LSQ refinement) is rerun.  Since
    different lambdas use independent shots, draw ``b`` across all lambdas is
    one complete synthetic realization of the scan.

    Requires ``record.metadata["bootstrap_stats"]`` as attached by the
    experimental loaders; lambdas without it yield all-NaN draws and a
    warning (no shot counts are ever invented).

    Parameters
    ----------
    records, order, n_boot, observable_key, L, L_used:
        As in ``run_bootstrap``.
    seed:
        Base seed.  Reproducible regardless of parallelism: per-lambda RNGs
        are spawned from ``np.random.SeedSequence(seed)``.  ``None`` → truly
        random.
    clean_poles:
        Per-slice clean poles (``state["poles_raw"]``) used as the Hungarian
        branch reference; the central result is never altered.
    refine_settings:
        ``{"conjugate_pairs", "bounds_real", "bounds_imag"}`` from the
        ``RefineStep`` configuration; ``None`` → per-draw MPM only.
    n_jobs:
        Parallel workers across lambda slices (process pool).  ``-1`` → all
        cores; ``1`` → serial.
    pencil:
        ``True`` (default) reruns the matrix pencil on every draw and then
        refines.  ``False`` skips the pencil — every draw is the LSQ
        refinement started from ``clean_poles`` — which is the only valid
        choice when the records' time axis is **non-uniform** (e.g. after
        :func:`mpm_analysis.analysis.dead_time.compress_records`), since the
        pencil assumes equal spacing.  Requires ``clean_poles`` and
        ``refine_settings``.

    Returns
    -------
    list[_LambdaBootstrapResult]
        One result per lambda slice (with ``boot_params``), sorted by lambda.
    """
    from mpm_analysis.data_types.observables import group_records_by_lambda

    if not pencil and (clean_poles is None or refine_settings is None):
        raise ValueError(
            "run_data_bootstrap(pencil=False) refines every draw from the "
            "clean poles — pass clean_poles and refine_settings (a RefineStep "
            "before the BootstrapStep provides both)."
        )

    if observable_key is None:
        if not records:
            raise ValueError("records is empty; cannot derive observable_key.")
        observable_key = records[0].observable_key

    joint = not isinstance(observable_key, str)
    if joint and len(observable_key) == 1:
        observable_key = observable_key[0]
        joint = False
    keys = list(observable_key) if joint else [observable_key]

    if joint:
        groups = group_records_by_lambda(records, keys)
    else:
        slices = sorted(
            [r for r in records if r.observable_key == keys[0]],
            key=lambda r: r.lambda_val,
        )
        groups = [(r.lambda_val, r.t, r.signal, [r]) for r in slices]

    if not any(recs[0].metadata.get("bootstrap_stats") for *_x, recs in groups):
        raise ValueError(
            "run_data_bootstrap: no record carries metadata['bootstrap_stats'] "
            "— this loader/data mode does not provide measurement statistics. "
            "Use UNCERTAINTY_SOURCE='fit_covariance' or 'residual_bootstrap'."
        )

    seed_children = np.random.SeedSequence(seed).spawn(len(groups))

    payloads = []
    for i, (lam, t, y, recs) in enumerate(groups):
        stats = recs[0].metadata.get("bootstrap_stats")
        N = np.atleast_2d(y).shape[1]
        L_i = L
        if L_used is not None and i < len(L_used):
            L_i = int(L_used[i]) or None
        if L_i is not None and not 1 <= L_i <= N - 1:
            L_i = None
        cp = None
        if clean_poles is not None and i < len(clean_poles):
            cp = np.asarray(clean_poles[i], dtype=complex)
        payloads.append(dict(
            lambda_val=lam, t=t, Y=y, single=not joint, keys=keys,
            stats=stats, order=order, n_boot=n_boot, L=L_i,
            clean_poles=cp, refine_settings=refine_settings,
            seed_seq=seed_children[i], pencil=pencil,
        ))

    def _empty():
        return (np.full((n_boot, order), np.nan),
                np.full((n_boot, order), np.nan),
                np.full((n_boot, order), np.nan),
                None)

    missing = [p["lambda_val"] for p in payloads if p["stats"] is None]
    for lam in missing:
        print(f"Warning: data bootstrap skipped for lam={lam:.3f} — "
              "no bootstrap_stats in record metadata.")

    todo = [p for p in payloads if p["stats"] is not None]

    # Fail fast: dry-run the resampler once.  A missing field would otherwise
    # make every single draw raise and silently produce an all-NaN result.
    probe = todo[0]
    try:
        _MEASUREMENT_RESAMPLERS[probe["stats"]["data_mode"]](
            probe["stats"], keys, np.random.default_rng(0)
        )
    except Exception as exc:
        raise ValueError(
            f"run_data_bootstrap: cannot resample observables {keys} in "
            f"'{probe['stats']['data_mode']}' mode — {exc}"
        ) from exc

    if n_jobs == 1 or len(todo) <= 1:
        outputs = [_data_bootstrap_worker(p) for p in todo]
    else:
        from concurrent.futures import ProcessPoolExecutor
        max_workers = None if n_jobs in (-1, 0, None) else int(n_jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            outputs = list(pool.map(_data_bootstrap_worker, todo))

    out_iter = iter(outputs)
    results: list[_LambdaBootstrapResult] = []
    total_failed = 0
    first_error: str | None = None
    for p in payloads:
        br, bi, bp, failure = (next(out_iter) if p["stats"] is not None
                               else _empty())
        if failure is not None:
            n_failed, err = failure
            total_failed += n_failed
            if first_error is None:
                first_error = err
            if n_failed == n_boot:
                print(f"Warning: ALL {n_boot} draws failed for "
                      f"lam={p['lambda_val']:.3f}: {err}")
        results.append(_LambdaBootstrapResult(
            lambda_val=p["lambda_val"],
            boot_real=br, boot_imag=bi, boot_params=bp,
        ))

    n_total = n_boot * max(len(todo), 1)
    if total_failed:
        print(f"Warning: {total_failed}/{n_total} bootstrap draws failed "
              f"({total_failed / n_total:.1%}); first error: {first_error}")
    if todo and total_failed == n_total:
        raise RuntimeError(
            "run_data_bootstrap: every bootstrap draw failed — the result "
            f"would be entirely NaN. First error: {first_error}"
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_bootstrap(
    boot_results: list[_LambdaBootstrapResult],
    *,
    percentile_lo: float = 16.0,
    percentile_hi: float = 84.0,
    observable_key: str = "survival",
    center_poles: "list[np.ndarray] | None" = None,
) -> PolesResult:
    """Aggregate bootstrap distributions into a ``PolesResult``.

    Parameters
    ----------
    boot_results:
        Output of ``run_bootstrap`` / ``run_data_bootstrap``.
    percentile_lo, percentile_hi:
        Percentiles for lower/upper uncertainty bounds.  Default 16/84 ≈ ±1σ.
    center_poles:
        Optional per-lambda poles from the fit of the ORIGINAL measured
        dataset (``state["poles_raw"]``).  When given, the central values are
        these measured best-fit poles — not the bootstrap median — and the
        16/84 draws only set the (asymmetric) uncertainty around them.

    Returns
    -------
    PolesResult
    """
    n_lam = len(boot_results)
    order = boot_results[0].boot_real.shape[1] if boot_results else 0

    decay_med = np.zeros((n_lam, order))
    decay_lo  = np.zeros((n_lam, order))
    decay_hi  = np.zeros((n_lam, order))
    freq_med  = np.zeros((n_lam, order))
    freq_lo   = np.zeros((n_lam, order))
    freq_hi   = np.zeros((n_lam, order))

    def _center(i) -> np.ndarray | None:
        if center_poles is None or i >= len(center_poles):
            return None
        c = np.asarray(center_poles[i], dtype=complex)
        return c if np.all(np.isfinite(c)) else None

    for i, br in enumerate(boot_results):
        valid_real = br.boot_real[~np.isnan(br.boot_real).any(axis=1)]
        valid_imag = br.boot_imag[~np.isnan(br.boot_imag).any(axis=1)]

        if len(valid_real) == 0:
            decay_med[i] = decay_lo[i] = decay_hi[i] = np.nan
            freq_med[i]  = freq_lo[i]  = freq_hi[i]  = np.nan
            continue

        c = _center(i)
        # Decay rates: -Re(pole)  (positive = decaying)
        dr = -valid_real
        decay_med[i] = (-c.real) if c is not None else np.median(dr, axis=0)
        decay_lo[i]  = decay_med[i] - np.percentile(dr, percentile_lo, axis=0)
        decay_hi[i]  = np.percentile(dr, percentile_hi, axis=0) - decay_med[i]

        # Frequencies: Im(pole) — keep signed so conjugate pairs appear at ±ω
        fr = valid_imag
        freq_med[i] = c.imag if c is not None else np.median(fr, axis=0)
        freq_lo[i]  = freq_med[i] - np.percentile(fr, percentile_lo, axis=0)
        freq_hi[i]  = np.percentile(fr, percentile_hi, axis=0) - freq_med[i]

    # Compute splittings between poles 0 and 1
    (rs_med, rs_lo, rs_hi,
     is_med, is_lo, is_hi) = _compute_splitting(boot_results, percentile_lo, percentile_hi)
    if center_poles is not None:
        for i in range(n_lam):
            c = _center(i)
            if c is not None and len(c) >= 2 and np.isfinite(rs_med[i]):
                rs_c = np.abs(c[1].real - c[0].real) / 2.0
                is_c = np.abs(c[1].imag - c[0].imag) / 2.0
                # Re-center half-widths: lo = c − p16, hi = p84 − c
                rs_lo[i] -= rs_med[i] - rs_c; rs_hi[i] += rs_med[i] - rs_c
                is_lo[i] -= is_med[i] - is_c; is_hi[i] += is_med[i] - is_c
                rs_med[i], is_med[i] = rs_c, is_c

    # Raw samples container: (n_lam, n_boot, order) complex
    lambda_values = np.array([br.lambda_val for br in boot_results])
    raw_samples_arr = np.stack(
        [br.boot_real + 1j * br.boot_imag for br in boot_results], axis=0
    )
    param_samples = None
    if all(br.boot_params is not None for br in boot_results) and boot_results:
        param_samples = np.stack([br.boot_params for br in boot_results], axis=0)
    raw_samples = PoleSamples(
        samples=raw_samples_arr,
        observable_key=observable_key,
        lambda_values=lambda_values,
        param_samples=param_samples,
    )

    return PolesResult(
        decay_rates=decay_med,
        decay_rates_lower=np.abs(decay_lo),
        decay_rates_upper=np.abs(decay_hi),
        frequencies=freq_med,
        frequencies_lower=np.abs(freq_lo),
        frequencies_upper=np.abs(freq_hi),
        imag_splitting=is_med,
        imag_splitting_lower=np.abs(is_lo),
        imag_splitting_upper=np.abs(is_hi),
        real_splitting=rs_med,
        real_splitting_lower=np.abs(rs_lo),
        real_splitting_upper=np.abs(rs_hi),
        raw_samples=raw_samples,
    )


def _compute_splitting(
    boot_results: list[_LambdaBootstrapResult],
    plo: float,
    phi: float,
) -> tuple[np.ndarray, ...]:
    """Compute real and imaginary pole splittings between poles 0 and 1.

    Poles are sorted by real-part ascending (most negative first = fastest
    decay), so the two EP poles occupy indices 0 and 1.  Splittings are
    absolute values of the difference — absorbs sign ambiguity for conjugate
    pairs and across the EP where pole identities swap.

    Returns
    -------
    (rs_med, rs_lo, rs_hi, is_med, is_lo, is_hi)
        Six 1-D arrays of shape (n_lambda,).
    """
    n = len(boot_results)
    rs_med = np.zeros(n); rs_lo = np.zeros(n); rs_hi = np.zeros(n)
    is_med = np.zeros(n); is_lo = np.zeros(n); is_hi = np.zeros(n)

    for i, br in enumerate(boot_results):
        valid_real = br.boot_real[~np.isnan(br.boot_real).any(axis=1)]
        valid_imag = br.boot_imag[~np.isnan(br.boot_imag).any(axis=1)]

        if valid_real.shape[1] < 2 or len(valid_real) == 0:
            rs_med[i] = rs_lo[i] = rs_hi[i] = np.nan
            is_med[i] = is_lo[i] = is_hi[i] = np.nan
            continue

        # |Re(pole₁ − pole₀)| / 2  — poles 0 and 1 (EP pair)
        rs = np.abs(valid_real[:, 1] - valid_real[:, 0]) / 2.0
        rs_med[i] = np.median(rs)
        rs_lo[i]  = rs_med[i] - np.percentile(rs, plo)
        rs_hi[i]  = np.percentile(rs, phi) - rs_med[i]

        # |Im(pole₁ − pole₀)| / 2  — same pair
        imag_s = np.abs(valid_imag[:, 1] - valid_imag[:, 0]) / 2.0
        is_med[i] = np.median(imag_s)
        is_lo[i]  = is_med[i] - np.percentile(imag_s, plo)
        is_hi[i]  = np.percentile(imag_s, phi) - is_med[i]

    return rs_med, rs_lo, rs_hi, is_med, is_lo, is_hi
