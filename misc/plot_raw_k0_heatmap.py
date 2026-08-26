"""Quick heatmap: raw P(k=k_idx, t) real part, time vs lambda.

Optionally also draws line cuts vs time at chosen λ and k values
(``CUT_LAMBDAS`` / ``CUT_KS`` below); the nearest available λ / k is used.
"""
from __future__ import annotations

import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt


from mpm_analysis.kspace import get_kspace_loader
from mpm_analysis.utils.lab_paths import get_lab_path

SOURCE = "compressed"      # "json" (raw scan JSONs) | "compressed" (compressed NPZs)
SCAN_NUMBER = 22
SCAN_FOLDER = get_lab_path(
    "Experiments", "Zeno", "CavityRecAl6", "Chip_TransSap46_5", "OPX",
    "two_transmons", "click_record_repeated", "Omega_BG0_scan",
    f"scan_experiment_results_{SCAN_NUMBER}",
)
# Raw JSONs sit directly in the scan folder; the compressed NPZs live in the
# sub-folder written by the compressed-click builder.
FOLDER = SCAN_FOLDER if SOURCE == "json" else SCAN_FOLDER / "compressed_click_analysis"
K_IDX = 0  # k=0 (idx 0) is the trivial normalization mode -- use first non-trivial k
COMPONENT = "real"   # "real" | "imag" | "abs"

# --- optional line cuts (empty list = skip) ---------------------------------
# CUT_LAMBDAS: λ values -> P(k=K_IDX, t) vs t, one curve per λ.
# CUT_KS:      k values in rad -> P(k, t) vs t at λ = CUT_K_LAMBDA.
# Nearest available λ / k is used in both cases.
CUT_LAMBDAS: list[float] = [0.2,0.8,1,1.5,2,2.2,2.4,2.6]      # e.g. [0.2, 0.6, 1.0]
CUT_KS: list[float] = []           # e.g. [np.pi / 2, np.pi]
CUT_K_LAMBDA: float | None = None  # λ for the CUT_KS panel; None -> median λ

# --- optional 3-D complex trajectory (empty list = skip) --------------------
# One (t, Re, Im) space curve per λ at k_idx=K_IDX -- shows the full complex
# signal instead of projecting onto a single component.
CUT3D_LAMBDAS: list[float] = []    # e.g. [0.2, 0.6, 1.0]

# Log the signal axis: "log" for the positive "abs" component, "symlog" for the
# signed real/imag ones (plain "log" would silently drop every negative point).
# The heatmap has no signal y-axis, so it gets a log colour norm instead.
LOG_SCALE = False

# Log the *axes* (independent variables), separately from the signal above.
# Time starts at 0, which a log axis cannot show, so the first bin drops out.
LOG_TIME = False      # time axis: heatmap x, and x of every line-cut panel

LOG_LAMBDA = False   # λ axis: heatmap y

# --- survival cut, same knob as MIN_ALIVE_FRAC in run_kspace_analysis.py ------
# Compressed source only: keep time bins where the number of shots that still
# have compressed data is >= frac * initial.  0.1 means "only times where at
# least 10% of the trajectories still exist".  None keeps the full axis.
# Ignored for SOURCE="json", whose loader has no survival notion.
MIN_ALIVE_FRAC: float | None = 0.5
T_MAX_US: float | None = None      # extra hard cut on time (µs); None = no cut

# --- k grid, same knobs as K_GRID_N / K_ENDPOINT_EPS in run_kspace_analysis.py -
# Compressed source only: recompute P(k,t) from stored P(n,t) on a denser open
# grid instead of using the stored 101-point one.  The endpoints are pulled
# inward by K_ENDPOINT_EPS so the grid gets close to 0 and 2π without hitting
# either exactly -- which is what puts k[0] at 0.001 rather than at 0.
# K_GRID_N = None falls back to the stored 101-point grid (k[0] = 0 exactly,
# the trivial normalization mode where P(k=0,t) = 1).
K_GRID_N: int | None = 401
K_ENDPOINT_EPS = 1e-3
K_INDEX_RANGE: tuple[int | None, int | None] | None = None  # e.g. (1, -1)

_COMPONENTS = {"real": np.real, "imag": np.imag, "abs": np.abs}
if COMPONENT not in _COMPONENTS:
    raise SystemExit(f"COMPONENT must be one of {sorted(_COMPONENTS)}; got {COMPONENT!r}")


def _comp(a):
    """Project the complex signal onto the selected component."""
    return _COMPONENTS[COMPONENT](a)


def _set_log_y(ax):
    """Put the signal (y) axis on a log scale, if enabled."""
    if not LOG_SCALE:
        return
    if COMPONENT == "abs":
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-3)


def _set_log_time(ax):
    """Put the time (x) axis on a log scale, if enabled."""
    if LOG_TIME:
        ax.set_xscale("log")


# Gated the same way KSpacePipeline.run() does it -- only the compressed loader
# accepts min_alive_frac.
load_kwargs = dict(t_max=T_MAX_US, k_index_range=K_INDEX_RANGE)
if SOURCE == "compressed":
    load_kwargs["min_alive_frac"] = MIN_ALIVE_FRAC
    load_kwargs["k_grid_n"] = K_GRID_N
    load_kwargs["k_endpoint_eps"] = K_ENDPOINT_EPS

records = get_kspace_loader(SOURCE)().load(FOLDER, **load_kwargs)
if not records:
    raise SystemExit(f"No {SOURCE} records found in {FOLDER}")
recs = sorted(records, key=lambda r: r.lambda_val)

_k = recs[0].k_array
print(f"k grid: {_k.size} points, k[0]={_k[0]:.5f}, k[1]={_k[1]:.5f}, "
      f"k[-1]={_k[-1]:.5f} rad  (source={recs[0].metadata.get('k_grid_source', SOURCE)})")
if K_IDX >= _k.size:
    raise SystemExit(f"K_IDX={K_IDX} out of range for a {_k.size}-point k grid")
print(f"K_IDX={K_IDX} -> k = {_k[K_IDX]:.5f} rad")

if SOURCE == "compressed":
    print(f"min_alive_frac={MIN_ALIVE_FRAC} -> retained time bins per λ:")
    for r in recs:
        m = r.metadata
        print(f"  λ={r.lambda_val:6.3f}  {r.t.size:4d} bins  t≤{r.t[-1]:7.2f} µs  "
              f"alive {m.get('alive_min_retained', '?')}/{m.get('alive_initial', '?')}")

lam = np.array([r.lambda_val for r in recs])
# Records can have different-length t axes (compressed source truncates once
# too few shots survive), but all share the same dt -> pad to the longest with
# NaN so pcolormesh gets a rectangular grid; NaN cells render blank.
n_t = max(r.t.size for r in recs)
t = max((r.t for r in recs), key=lambda tt: tt.size)
sig = np.full((len(recs), n_t), np.nan, dtype=complex)
for i, r in enumerate(recs):
    sig[i, :r.t.size] = r.signal[K_IDX, :]
z = _comp(sig)

fig, ax = plt.subplots(figsize=(8, 5))
if LOG_SCALE:
    # Signed components straddle zero -> SymLogNorm; "abs" is strictly positive
    # so a plain LogNorm applies (vmin from the smallest non-zero cell).
    if COMPONENT == "abs":
        pos = z[np.isfinite(z) & (z > 0)]
        norm = matplotlib.colors.LogNorm(
            vmin=pos.min() if pos.size else 1e-6, vmax=np.nanmax(z),
        )
    else:
        norm = matplotlib.colors.SymLogNorm(
            linthresh=1e-3, vmin=np.nanmin(z), vmax=np.nanmax(z),
        )
else:
    norm = None
mesh = ax.pcolormesh(t, lam, z, shading="auto", cmap="viridis", norm=norm)
ax.set_xlabel("time (µs)")
ax.set_ylabel("λ")
_set_log_time(ax)
if LOG_LAMBDA:
    ax.set_yscale("log")
_cut = f"  |  alive ≥ {MIN_ALIVE_FRAC}" if SOURCE == "compressed" else ""
ax.set_title(
    f"[{SOURCE}] {COMPONENT} P(k_idx={K_IDX}, t)  |  "
    f"k ≈ {recs[0].k_array[K_IDX]:.3f} rad{_cut}"
)
fig.colorbar(mesh, ax=ax, label=f"{COMPONENT} P(k, t)")
fig.tight_layout()

# ---------------------------------------------------------------------------
# Line cuts
# ---------------------------------------------------------------------------


def _nearest_rec(lambda_val):
    """Record whose λ is closest to ``lambda_val``."""
    return recs[int(np.argmin(np.abs(lam - lambda_val)))]


if CUT_LAMBDAS or CUT_KS:
    n_panels = bool(CUT_LAMBDAS) + bool(CUT_KS)
    fig2, axes = plt.subplots(
        n_panels, 1, figsize=(8, 4 * n_panels), squeeze=False,
    )
    axes = list(axes[:, 0])

    if CUT_LAMBDAS:
        ax2 = axes.pop(0)
        for lv in CUT_LAMBDAS:
            r = _nearest_rec(lv)
            ax2.plot(r.t, _comp(r.signal[K_IDX, :]), label=f"λ = {r.lambda_val:.3f}")
        ax2.set_xlabel("time (µs)")
        ax2.set_ylabel(f"{COMPONENT} P(k, t)")
        ax2.set_title(
            f"λ cuts at k_idx={K_IDX} (k ≈ {recs[0].k_array[K_IDX]:.3f} rad)"
        )
        _set_log_y(ax2)
        _set_log_time(ax2)
        ax2.legend(fontsize=8)

    if CUT_KS:
        ax2 = axes.pop(0)
        r = _nearest_rec(
            float(np.median(lam)) if CUT_K_LAMBDA is None else CUT_K_LAMBDA
        )
        for kv in CUT_KS:
            ki = int(np.argmin(np.abs(r.k_array - kv)))
            ax2.plot(r.t, _comp(r.signal[ki, :]),
                     label=f"k = {r.k_array[ki]:.3f} rad (idx {ki})")
        ax2.set_xlabel("time (µs)")
        ax2.set_ylabel(f"{COMPONENT} P(k, t)")
        ax2.set_title(f"k cuts at λ = {r.lambda_val:.3f}")
        _set_log_y(ax2)
        _set_log_time(ax2)
        ax2.legend(fontsize=8)

    fig2.tight_layout()

# ---------------------------------------------------------------------------
# 3-D complex trajectories: (t, Re, Im) space curve per λ
# ---------------------------------------------------------------------------
if CUT3D_LAMBDAS:
    fig3 = plt.figure(figsize=(9, 7))
    ax3 = fig3.add_subplot(111, projection="3d")
    for lv in CUT3D_LAMBDAS:
        r = _nearest_rec(lv)
        s = r.signal[K_IDX, :]
        line, = ax3.plot(r.t, s.real, s.imag, lw=1.2,
                         label=f"λ = {r.lambda_val:.3f}")
        # Mark t=0 so the winding direction along the curve is unambiguous.
        ax3.scatter(r.t[:1], s.real[:1], s.imag[:1], s=25,
                    color=line.get_color())
    ax3.set_xlabel("time (µs)")
    ax3.set_ylabel("Re P(k, t)")
    ax3.set_zlabel("Im P(k, t)")
    ax3.set_title(
        f"[{SOURCE}] complex P(k_idx={K_IDX}, t)  |  "
        f"k ≈ {recs[0].k_array[K_IDX]:.3f} rad  (dot = t=0)"
    )
    ax3.legend(fontsize=8)
    fig3.tight_layout()

plt.show()
