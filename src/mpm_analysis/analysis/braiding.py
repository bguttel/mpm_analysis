"""
braiding.py
-----------
Theory poles + braiding characterization, **using
``Theory/braid_tools_complete.py``** (the complete, validated braid analysis) for
all the topology — no reimplementation.  From that module we reuse:

* ``scan_k`` / ``build_liouvillian_3x3`` / ``build_liouvillian_4x4`` — the theory
  eigenvalues vs (λ, k);
* ``sort_by_continuity`` — Hungarian continuity sort of eigenvalue strands;
* ``analyze_external_eigenstrands`` → ``BraidAnalysis`` — the paper braid indices
  ``ν_total`` (``round_if_close`` to integers), the closed/component link indices,
  the per-pair link **types** (Hopf link / unlink / half-twist …), the closure
  classification, and the braid word.

Two uses:
1. ``theory_poles_on_grid`` — theory eigenvalues on a (λ, k) grid (for overlay).
2. ``experimental_braiding`` / ``theory_braiding`` — the braid report per λ.

Quantization note: ``ν_total`` is an integer winding **only on the closed loop
k ∈ [0, 2π]**.  ``theory_braiding`` therefore evaluates on the full loop (→
integer ν).  The experiment's ``P(k=0,t)=P(k=2π,t)=1`` is trivially constant, so
its loop omits the exact endpoints and ν is only *approximately* integer
(``round_if_close`` snaps it when within tolerance; residual non-integer values
reflect MPM noise + the open endpoints).

The script is loaded by file path via ``importlib`` (its ``__main__`` is
guarded).  Set the location with ``MPM_BRAID_TOOLS`` env var or ``theory_path=``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_BT_CACHE: dict = {}
_DEFAULT_BT_REL = ("Theory", "braid_tools_complete.py")


def _default_bt_path() -> Path:
    env = os.environ.get("MPM_BRAID_TOOLS")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[4]   # .../Code
    return repo_root.joinpath(*_DEFAULT_BT_REL)


def load_braid_tools(theory_path: str | Path | None = None):
    """Import and cache ``braid_tools_complete.py`` (no side effects)."""
    path = Path(theory_path) if theory_path else _default_bt_path()
    key = str(path)
    if key in _BT_CACHE:
        return _BT_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(
            f"braid_tools_complete.py not found at {path}. Set MPM_BRAID_TOOLS "
            f"or pass theory_path=."
        )
    spec = importlib.util.spec_from_file_location("zeno_braid_tools", str(path))
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (BraidAnalysis) can resolve its module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _BT_CACHE[key] = mod
    return mod


# Backward-compatible alias (older callers used load_theory).
load_theory = load_braid_tools


# ---------------------------------------------------------------------------
# Theory poles on a (λ, k) grid
# ---------------------------------------------------------------------------

def theory_poles_on_grid(
    lambda_values,
    k_array,
    *,
    Omega: float,
    T1: float,
    T2: float,
    tau_B: float | None = None,
    k_fp: float = 0.0,
    lambda_scale: float = 1.0,
    model: str = "3x3",
    theory_path: str | Path | None = None,
    fn: float = 0.0,
    t_bin: float = 0.0,
) -> np.ndarray:
    """Theory eigenvalues over ``(λ, k)`` in rad/µs, shape ``(n_λ, n_k, M)``.

    Imperfection / calibration params: ``T1``, ``T2`` (µs), ``tau_B`` (µs, 4x4
    only; default ``1/(λΩ)``), ``k_fp`` (rad/µs), ``fn`` (readout false-negative
    probability, 4x4 only — the generator counts with ``e^{ik} -> fn + (1-fn)
    e^{ik}``), ``t_bin`` (detection time-bin length in µs, 4x4 only; 0 = the
    continuous limit), and ``lambda_scale`` (multiplier on λ — the click-rate
    calibration knob, since the model rate is ``α=2λΩ``).
    """
    bt = load_braid_tools(theory_path)
    lambda_values = np.asarray(lambda_values, dtype=float)
    k_array = np.asarray(k_array, dtype=float)
    M = 3 if model == "3x3" else 4
    out = np.full((len(lambda_values), len(k_array), M), np.nan, dtype=complex)
    for i, lam0 in enumerate(lambda_values):
        lam = lam0 * lambda_scale
        tb = tau_B
        if tb is None:
            tb = 1.0 / (lam * Omega) if (model == "4x4" and lam * Omega != 0) else 1.0
        # normalize_by_Omega=False -> eigenvalues already in rad/µs
        out[i] = bt.scan_k(lam, k_array, tb, T2, T1, Omega, k_fp=k_fp,
                           model=model, normalize_by_Omega=False,
                           k_fn=fn, t_bin=t_bin)
    return out


# ---------------------------------------------------------------------------
# Braiding characterization (reuse braid_tools_complete.analyze_external_eigenstrands)
# ---------------------------------------------------------------------------

def _ep_k(eigs_nk_m: np.ndarray, k_array=None):
    """(k_idx, k_val, min_gap): k where the smallest pairwise eigenvalue gap over
    the loop is minimized — the candidate exceptional point."""
    eigs = np.asarray(eigs_nk_m, dtype=complex)
    finite = np.all(np.isfinite(eigs), axis=0)
    eigs = eigs[:, finite]
    if eigs.shape[1] < 2:
        return -1, float("nan"), float("nan")
    n_k, M = eigs.shape
    min_gap = np.full(n_k, np.inf)
    for a in range(M):
        for b in range(a + 1, M):
            min_gap = np.minimum(min_gap, np.abs(eigs[:, a] - eigs[:, b]))
    ki = int(np.argmin(min_gap))
    kv = float(k_array[ki]) if k_array is not None else float(ki)
    return ki, kv, float(min_gap[ki])


def _link_label(a) -> str:
    """Human-readable link type from a BraidAnalysis: prefer the closure
    classification; fall back to the per-pair link type."""
    closure = getattr(a, "closure_type", "")
    if closure:
        return str(closure)
    pls = getattr(a, "pair_link_types", {}) or {}
    return next(iter(pls.values()), "") if pls else ""


def _analyze_over_lambda(eigs_lkm, lambda_values, k_array, integer_tol, bt) -> dict:
    lam = np.asarray(lambda_values, dtype=float)
    n = len(lam)
    nu = np.full(n, np.nan)
    links: list[dict] = []
    raw_pairs: list[dict] = []
    component_indices: list[dict] = []
    words: list[str] = []
    link_types: list[str] = []
    ep_k_idx = np.full(n, -1, dtype=int)
    ep_k = np.full(n, np.nan)
    ep_gap = np.full(n, np.nan)
    for i in range(n):
        eigs = np.asarray(eigs_lkm[i], dtype=complex)
        finite = np.all(np.isfinite(eigs), axis=0)
        eigs = eigs[:, finite]
        if eigs.shape[1] < 2:
            links.append({}); raw_pairs.append({}); component_indices.append({})
            words.append(""); link_types.append("")
            continue
        es = bt.sort_by_continuity(eigs)
        try:
            a = bt.analyze_external_eigenstrands(es, integer_tol=integer_tol)
        except Exception:
            links.append({}); raw_pairs.append({}); component_indices.append({})
            words.append(""); link_types.append("")
            continue
        nu[i] = float(a.nu_total)
        comp = a.component_indices or a.raw_pair
        links.append({str(k): float(v) for k, v in comp.items()})
        raw_pairs.append({str(k): float(v) for k, v in (a.raw_pair or {}).items()})
        component_indices.append(
            {str(k): float(v) for k, v in (a.component_indices or {}).items()}
        )
        words.append(str(a.braid_word))
        link_types.append(_link_label(a))
        ep_k_idx[i], ep_k[i], ep_gap[i] = _ep_k(eigs, k_array)
    return {"lambda_values": lam, "nu_total": nu, "links": links,
            "raw_pairs": raw_pairs, "component_indices": component_indices,
            "braid_words": words, "link_types": link_types,
            "ep_k_idx": ep_k_idx, "ep_k": ep_k, "ep_gap": ep_gap}


def experimental_braiding(result, *, integer_tol: float = 2e-2,
                          theory_path: str | Path | None = None) -> dict:
    """Braid report per λ for an experimental ``KSpacePolesResult`` (poles over k).

    Returns dict: ``lambda_values, nu_total, links, braid_words, link_types,
    ep_k, ep_k_idx, ep_gap``.
    """
    bt = load_braid_tools(theory_path)
    return _analyze_over_lambda(result.poles, result.lambda_values,
                                result.k_array, integer_tol, bt)


def theory_braiding(
    lambda_values,
    *,
    Omega: float,
    T1: float,
    T2: float,
    tau_B: float | None = None,
    k_fp: float = 0.0,
    lambda_scale: float = 1.0,
    model: str = "3x3",
    n_k: int = 201,
    integer_tol: float = 1e-1,
    theory_path: str | Path | None = None,
    fn: float = 0.0,
    t_bin: float = 0.0,
) -> dict:
    """Braid report per λ for the theory model, on the **full closed loop**
    ``k ∈ [0, 2π]`` (so ``ν_total`` is an integer winding).  ``fn`` (false-negative
    probability) and ``t_bin`` (time-bin length, µs) are 4x4 only, see
    ``theory_poles_on_grid``."""
    bt = load_braid_tools(theory_path)
    K = np.linspace(0.0, 2.0 * np.pi, n_k)
    tp = theory_poles_on_grid(lambda_values, K, Omega=Omega, T1=T1, T2=T2,
                              tau_B=tau_B, k_fp=k_fp, lambda_scale=lambda_scale,
                              model=model, theory_path=theory_path,
                              fn=fn, t_bin=t_bin)
    return _analyze_over_lambda(tp, lambda_values, K, integer_tol, bt)


# ---------------------------------------------------------------------------
# Paper-style reporting
# ---------------------------------------------------------------------------

def braiding_transitions(braiding: dict) -> dict:
    """λ (and EP-k) where the integer braid invariant ν_total changes."""
    nu = np.asarray(braiding["nu_total"])
    lam = np.asarray(braiding["lambda_values"])
    ek = np.asarray(braiding.get("ep_k", np.full_like(lam, np.nan)))
    idx = [i for i in range(1, len(nu))
           if np.isfinite(nu[i - 1]) and np.isfinite(nu[i])
           and round(nu[i - 1]) != round(nu[i])]
    return {"indices": idx,
            "lambda_mid": [float(0.5 * (lam[i - 1] + lam[i])) for i in idx],
            "nu_from": [float(nu[i - 1]) for i in idx],
            "nu_to": [float(nu[i]) for i in idx],
            "ep_k": [float(ek[i]) for i in idx]}


def print_braiding_summary(braiding: dict, *, label: str = "experiment") -> None:
    """Paper-style table: per λ → ν_total, EP-k, link type, links (ν), braid word."""
    def safe(text) -> str:
        return (
            str(text)
            .replace("ν", "nu")
            .replace("λ", "lambda")
            .replace("→", "->")
            .replace("≈", "~")
            .encode("ascii", "replace")
            .decode("ascii")
        )

    lam = braiding["lambda_values"]; nu = braiding["nu_total"]
    ek = braiding.get("ep_k", [np.nan] * len(lam))
    links = braiding.get("links", [{}] * len(lam))
    words = braiding.get("braid_words", [""] * len(lam))
    ltyp = braiding.get("link_types", [""] * len(lam))
    print(f"\n=== Braiding characterization ({label}) — paper indices ===")
    print(f"{'lambda':>8} {'nu':>5} {'EP k':>6}  {'link type':<34} {'word':<12} indices")
    for i in range(len(lam)):
        link_str = safe(", ".join(f"{k}:{v:+.2f}" for k, v in links[i].items()) or "-")
        nu_s = f"{nu[i]:+.0f}" if np.isfinite(nu[i]) and abs(nu[i] - round(nu[i])) < 0.05 \
            else (f"{nu[i]:+.2f}" if np.isfinite(nu[i]) else " nan")
        ek_s = f"{ek[i]:.2f}" if np.isfinite(ek[i]) else " nan"
        lt = safe(ltyp[i][:33]) if ltyp[i] else "-"
        word = safe(words[i])
        print(f"{lam[i]:8.3f} {nu_s:>5} {ek_s:>6}  {lt:<34} {word:<12} {link_str}")
    tr = braiding_transitions(braiding)
    if tr["indices"]:
        print("  transitions (nu changes = exceptional points):")
        for lm, a, b, k in zip(tr["lambda_mid"], tr["nu_from"], tr["nu_to"], tr["ep_k"]):
            print(f"    lambda~{lm:.3f}:  nu {a:+.0f} -> {b:+.0f}   (EP k~{k:.2f})")
    else:
        print("  no nu transitions detected over this lambda range.")
