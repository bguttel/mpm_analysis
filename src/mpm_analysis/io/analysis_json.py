"""
analysis_json.py
----------------
Load and save ``ZenoAnalysisResult`` objects to/from the canonical JSON format
produced by the bootstrap analysis pipeline.

This is the **only** place in the package that knows the JSON schema.  All
other code works with typed ``ZenoAnalysisResult`` / ``PolesResult`` objects.

JSON schema reference
~~~~~~~~~~~~~~~~~~~~~
See ``data_types/analysis_result.py`` docstring for the full schema.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mpm_analysis.data_types.analysis_result import (
    ZenoAnalysisResult,
    SqrtFitResult,
)
from mpm_analysis.data_types.poles import PolesResult
from mpm_analysis.utils.windows_paths import win_long_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(win_long_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(data: dict, path: Path, *, indent: int = 2) -> None:
    # win_long_path on the mkdir too — deep lab folders exceed MAX_PATH.
    os.makedirs(win_long_path(Path(path).parent), exist_ok=True)
    with open(win_long_path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def _to_list(arr) -> list:
    """Convert numpy array (or nested) to plain Python list for JSON."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def _jsonify(obj):
    """Recursively convert numpy / complex data to JSON-safe structures.

    Complex arrays and scalars become ``{"__complex__": true, "real": ...,
    "imag": ...}`` so they round-trip through ``_dejsonify``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.bool_, np.integer)):
        return obj.item()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"__complex__": True,
                    "real": obj.real.tolist(),
                    "imag": obj.imag.tolist()}
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return str(obj)  # last resort — never fail a save on an exotic type


def _dejsonify(obj):
    """Inverse of ``_jsonify`` (complex markers → complex ndarrays)."""
    if isinstance(obj, dict):
        if obj.get("__complex__"):
            return np.asarray(obj["real"], dtype=float) + 1j * np.asarray(
                obj["imag"], dtype=float
            )
        return {k: _dejsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_dejsonify(v) for v in obj]
    return obj


def _ua_from_dict(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (median, lower, upper) arrays from a JSON uncertainty-array dict."""
    return (
        np.asarray(d["median"], dtype=float),
        np.asarray(d["lower_uncertainty"], dtype=float),
        np.asarray(d["upper_uncertainty"], dtype=float),
    )


def _ua_to_dict(median, lower, upper) -> dict:
    """Serialise three arrays to the canonical JSON uncertainty-array dict."""
    return {
        "median": _to_list(median),
        "lower_uncertainty": _to_list(lower),
        "upper_uncertainty": _to_list(upper),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_analysis_result(path: str | Path) -> ZenoAnalysisResult:
    """Load a ``ZenoAnalysisResult`` from an ``mpm_analysis_results_*.json`` file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    ZenoAnalysisResult
    """
    path = Path(path)
    jd = _load_json(path)

    # --- poles ---
    poles_jd = jd["poles"]
    dr_med, dr_lo, dr_hi = _ua_from_dict(poles_jd["decay_rates"])
    fr_med, fr_lo, fr_hi = _ua_from_dict(poles_jd["frequencies"])

    # Splitting — accept both underscore and space key variants
    real_key = "real_splitting" if "real_splitting" in poles_jd else "real splitting"
    imag_key = "imaginary_splitting" if "imaginary_splitting" in poles_jd else "imaginary splitting"

    rs_med = rs_lo = rs_hi = None
    is_med = is_lo = is_hi = None
    if real_key in poles_jd and imag_key in poles_jd:
        rs_med, rs_lo, rs_hi = _ua_from_dict(poles_jd[real_key])
        is_med, is_lo, is_hi = _ua_from_dict(poles_jd[imag_key])

    # --- raw pole samples (optional, written with full=True) ---
    raw_samples = None
    if "pole_samples" in jd:
        from mpm_analysis.data_types.pole_samples import PoleSamples

        ps = jd["pole_samples"]
        samples_arr = np.asarray(ps["real"], dtype=float) + 1j * np.asarray(
            ps["imag"], dtype=float
        )
        raw_samples = PoleSamples(
            samples=samples_arr,
            observable_key=str(ps.get("observable_key", "survival")),
            lambda_values=np.asarray(ps["lambda_values"], dtype=float),
            param_samples=(np.asarray(ps["params"], dtype=float)
                           if ps.get("params") is not None else None),
        )

    poles = PolesResult(
        decay_rates=dr_med,
        decay_rates_lower=dr_lo,
        decay_rates_upper=dr_hi,
        frequencies=fr_med,
        frequencies_lower=fr_lo,
        frequencies_upper=fr_hi,
        real_splitting=rs_med,
        real_splitting_lower=rs_lo,
        real_splitting_upper=rs_hi,
        imag_splitting=is_med,
        imag_splitting_lower=is_lo,
        imag_splitting_upper=is_hi,
        raw_samples=raw_samples,
    )

    # --- sqrt fit (optional) ---
    sqrt_fit: SqrtFitResult | None = None
    if "critical_point_sqrt_fit" in jd:
        sqrt_fit = _load_sqrt_fit(jd["critical_point_sqrt_fit"])

    return ZenoAnalysisResult(
        metadata=jd.get("metadata", {}),
        parameters=jd.get("parameters", {}),
        lambda_values=np.asarray(jd["lambda_values"], dtype=float),
        poles=poles,
        critical_point_sqrt_fit=sqrt_fit,
        extras=_dejsonify(jd["analysis_extras"]) if "analysis_extras" in jd else None,
    )


def save_analysis_result(
    result: ZenoAnalysisResult,
    path: str | Path,
    *,
    full: bool = False,
) -> None:
    """Save a ``ZenoAnalysisResult`` to JSON, preserving the canonical schema.

    Parameters
    ----------
    result:
        The result object to serialise.
    path:
        Output file path.  Parent directories are created if needed.
    full:
        If ``True``, additionally write the ``pole_samples`` block (raw
        bootstrap draws, shape n_lambda × n_boot × order — can be tens of MB
        at n_boot=2000) and the ``analysis_extras`` block (refine parameter
        vectors and covariances, amplitudes, pencil parameters, …), so the
        JSON contains everything the analysis produced.  Old-schema readers
        simply ignore the extra blocks.
    """
    path = Path(path)
    p = result.poles
    jd: dict = {
        "metadata": result.metadata,
        "parameters": result.parameters,
        "lambda_values": _to_list(result.lambda_values),
        "poles": {
            "decay_rates": _ua_to_dict(p.decay_rates, p.decay_rates_lower, p.decay_rates_upper),
            "frequencies": _ua_to_dict(p.frequencies, p.frequencies_lower, p.frequencies_upper),
        },
    }
    if p.has_splitting:
        jd["poles"]["real_splitting"] = _ua_to_dict(
            p.real_splitting, p.real_splitting_lower, p.real_splitting_upper
        )
        jd["poles"]["imaginary_splitting"] = _ua_to_dict(
            p.imag_splitting, p.imag_splitting_lower, p.imag_splitting_upper
        )
    if result.critical_point_sqrt_fit is not None:
        jd["critical_point_sqrt_fit"] = _dump_sqrt_fit(result.critical_point_sqrt_fit)

    if full:
        if p.raw_samples is not None:
            rs = p.raw_samples
            jd["pole_samples"] = {
                "real": rs.samples.real.tolist(),
                "imag": rs.samples.imag.tolist(),
                "observable_key": rs.observable_key,
                "lambda_values": _to_list(rs.lambda_values),
                # Quadratic pole parameters [r..., c1, c0] per draw — the
                # discriminant c1**2 - 4*c0 is the preferred EP diagnostic.
                "params": _to_list(rs.param_samples),
            }
        if result.extras:
            jd["analysis_extras"] = _jsonify(result.extras)

    _save_json(jd, path)
    print(f"Saved analysis result: {path}")


def augment_json_with_sqrt_fit(
    path: str | Path,
    fit: SqrtFitResult,
    *,
    overwrite: bool = True,
) -> Path:
    """Add / update the ``critical_point_sqrt_fit`` block in an existing JSON.

    Parameters
    ----------
    path:
        Path to an existing ``mpm_analysis_results_*.json``.
    fit:
        The ``SqrtFitResult`` to embed.
    overwrite:
        If ``False`` and the key already exists, raise ``ValueError``.

    Returns
    -------
    Path
        Same path (for chaining).
    """
    path = Path(path)
    jd = _load_json(path)
    if "critical_point_sqrt_fit" in jd and not overwrite:
        raise ValueError(
            f"'critical_point_sqrt_fit' already present in {path.name}. "
            "Pass overwrite=True to replace it."
        )
    jd["critical_point_sqrt_fit"] = _dump_sqrt_fit(fit)
    _save_json(jd, path)
    print(f"Augmented {path.name} with sqrt-fit (lam_c = {fit.lambda_c:.4f})")
    return path


def augment_json_with_ksi(
    path: str | Path,
    ksi_result,
    *,
    overwrite: bool = True,
) -> Path:
    """Add / update the ``ksi_crossing`` block in an existing JSON.

    Parameters
    ----------
    path:
        Path to an existing ``mpm_analysis_results_*.json``.
    ksi_result:
        A ``KsiCrossingResult`` from ``KsiCrossingPipeline.run()``.
    overwrite:
        If ``False`` and the key already exists, raise ``ValueError``.

    Returns
    -------
    Path
        Same path (for chaining).
    """
    path = Path(path)
    jd = _load_json(path)
    if "ksi_crossing" in jd and not overwrite:
        raise ValueError(
            f"'ksi_crossing' already present in {path.name}. "
            "Pass overwrite=True to replace it."
        )

    cs = ksi_result.crossing_samples
    frac_none = None
    if cs is not None and len(cs):
        frac_none = float(np.mean(np.isnan(cs)))

    jd["ksi_crossing"] = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "pole_idx1": ksi_result.pole_idx1,
        "pole_idx2": ksi_result.pole_idx2,
        "lambda_values": _to_list(ksi_result.lambda_values),
        "ksi_values": _to_list(ksi_result.ksi_values),
        "ksi_lower": _to_list(ksi_result.ksi_lower),
        "ksi_upper": _to_list(ksi_result.ksi_upper),
        "zero_crossings": list(ksi_result.zero_crossings),
        "crossing_samples": _to_list(cs),
        "crossing_quantiles": (
            list(ksi_result.crossing_quantiles)
            if ksi_result.crossing_quantiles is not None else None
        ),
        "fraction_no_crossing": frac_none,
    }
    _save_json(jd, path)
    lam_ep = (ksi_result.crossing_quantiles[1]
              if ksi_result.crossing_quantiles is not None
              else ksi_result.lambda_ep)
    print(f"Augmented {path.name} with ksi crossing"
          + (f" (lam_EP = {lam_ep:.4f})" if lam_ep is not None else ""))
    return path


def augment_json_with_transitions(
    path: str | Path,
    transitions: dict,
    *,
    overwrite: bool = True,
) -> Path:
    """Add / update the ``transition_samples`` block in an existing JSON.

    ``transitions`` is a free-form dict of per-realization transition
    distributions and their quantiles (e.g. ``delta_crossing_samples``,
    ``sqrt_lambda_c_samples``, ``ksi_crossing_samples`` with matching
    ``*_quantiles`` triples).  Numpy content is converted automatically.
    """
    path = Path(path)
    jd = _load_json(path)
    block = jd.get("transition_samples", {})
    new_keys = set(transitions) & set(block)
    if new_keys and not overwrite:
        raise ValueError(
            f"transition_samples keys {sorted(new_keys)} already present in "
            f"{path.name}. Pass overwrite=True to replace them."
        )
    block.update(_jsonify(transitions))
    block["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    jd["transition_samples"] = block
    _save_json(jd, path)
    print(f"Augmented {path.name} with transition samples "
          f"({', '.join(k for k in transitions)})")
    return path


# ---------------------------------------------------------------------------
# Sqrt-fit helpers
# ---------------------------------------------------------------------------

def _load_sqrt_fit(block: dict) -> SqrtFitResult:
    """Deserialise the ``critical_point_sqrt_fit`` JSON block."""
    fit = block.get("fit", block)  # handle both wrapped and flat formats
    results = fit["results"]
    window_d = fit.get("window", {})
    diw = fit.get("data_in_window", {})
    curves = fit.get("curves", {})

    # Accept both "imag_split" and "imag_split_median" key variants
    imag_split_key = "imag_split" if "imag_split" in diw else "imag_split_median"
    imag_sigma_key = "imag_sigma" if "imag_sigma" in diw else "imag_split_sigma_for_fit"
    real_split_key = "real_split" if "real_split" in diw else "real_split_median"
    real_sigma_key = "real_sigma" if "real_sigma" in diw else "real_split_sigma_for_fit"

    def _f(val, default=float("nan")):
        return float(val) if val is not None else default

    return SqrtFitResult(
        lambda_c=float(results["lambda_c"]),
        a=_f(results.get("a")),
        b=_f(results.get("b")),
        lambda_c_err=_f(results.get("lambda_c_err")),
        a_err=_f(results.get("a_err")),
        b_err=_f(results.get("b_err")),
        chi2_reduced=_f(results.get("chi2_reduced")),
        rmse=_f(results.get("rmse")),
        r_squared=_f(results.get("r_squared")),
        window_lambda_min=float(window_d.get("lambda_min", 0.0)),
        window_lambda_max=float(window_d.get("lambda_max", 0.0)),
        window_n_points=int(window_d.get("n_points", 0)),
        window_left_points=int(window_d.get("left_points", 0)),
        window_right_points=int(window_d.get("right_points", 0)),
        lambda_data=np.asarray(diw.get("lambda", []), dtype=float),
        imag_split_data=np.asarray(diw.get(imag_split_key, []), dtype=float),
        imag_split_err=np.asarray(diw.get(imag_sigma_key, []), dtype=float),
        real_split_data=np.asarray(diw[real_split_key], dtype=float) if real_split_key in diw else None,
        real_split_err=np.asarray(diw[real_sigma_key], dtype=float) if real_sigma_key in diw else None,
        lambda_plot=np.asarray(curves.get("lambda_plot", []), dtype=float),
        imag_fit_curve=np.asarray(curves.get("imag_fit_curve", []), dtype=float),
        real_fit_curve=np.asarray(curves.get("real_fit_curve", []), dtype=float),
    )


def _dump_sqrt_fit(fit: SqrtFitResult) -> dict:
    """Serialise a ``SqrtFitResult`` to the canonical JSON block."""
    diw: dict = {
        "lambda": _to_list(fit.lambda_data),
        "imag_split": _to_list(fit.imag_split_data),
        "imag_sigma": _to_list(fit.imag_split_err),
    }
    if fit.real_split_data is not None:
        diw["real_split"] = _to_list(fit.real_split_data)
    if fit.real_split_err is not None:
        diw["real_sigma"] = _to_list(fit.real_split_err)

    curves: dict = {}
    if fit.lambda_plot is not None:
        curves["lambda_plot"] = _to_list(fit.lambda_plot)
    if fit.imag_fit_curve is not None:
        curves["imag_fit_curve"] = _to_list(fit.imag_fit_curve)
    if fit.real_fit_curve is not None:
        curves["real_fit_curve"] = _to_list(fit.real_fit_curve)

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "fit": {
            "results": {
                "lambda_c": fit.lambda_c,
                "a": fit.a,
                "b": fit.b,
                "lambda_c_err": fit.lambda_c_err,
                "a_err": fit.a_err,
                "b_err": fit.b_err,
                "chi2_reduced": fit.chi2_reduced,
                "rmse": fit.rmse,
                "r_squared": fit.r_squared,
            },
            "window": {
                "lambda_min": fit.window_lambda_min,
                "lambda_max": fit.window_lambda_max,
                "n_points": fit.window_n_points,
                "left_points": fit.window_left_points,
                "right_points": fit.window_right_points,
            },
            "data_in_window": diw,
            "curves": curves,
        },
    }
