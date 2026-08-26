from __future__ import annotations

from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from mpm_analysis.utils.lab_paths import get_lab_path

from mpm_analysis.io.experimental import ParityLoader
from mpm_analysis.io.experimental.click_record_npz import ClickRecordNpzLoader, ClickRecordData
from mpm_analysis.io.simulated import SimClickRecordLoader
from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.analysis.matrix_pencil import matrix_pencil_method
from mpm_analysis.analysis.matrix_pencil_rrha import mpm_rrha
from mpm_analysis.analysis.pole_sorting import sort_poles_canonical
from mpm_analysis.io.simulated import SimClickRecordLoader


# ===========================================================================
# SETTINGS
# ===========================================================================

# ---- Data ----
_ZT = r"Experiments\Zeno\CavityRecAl6\Chip_TransSap46_5\OPX\two_transmons"
_PARITY = _ZT + r"\click_record_repeated\Omega_BG0_scan\scan_experiment_results_12"
PARITY_FOLDER = get_lab_path(_PARITY)

# ===========================================================================
# HELPERS
# ===========================================================================

def _make_mean_clicks_loader() -> ParityLoader:
    """Return a ParityLoader with a 'mean_clicks' extractor registered."""
    loader = ParityLoader()
    loader.add_extractor(
        "mean_clicks",
        lambda jd, _: (
            np.asarray(jd["data"]["jump_probability_distribution"], dtype=float)
            @ np.arange(
                np.asarray(jd["data"]["jump_probability_distribution"]).shape[1],
                dtype=float,
            )
        ),
    )
    return loader


# ===========================================================================
# ANALYSIS
# ===========================================================================

def average_click_number(
    folder: str | Path,
    **loader_kwargs,
) -> list[ObservableRecord]:
    """Compute the ensemble-average number of clicks <n>(t) per lambda.

    Returns one ObservableRecord per lambda file.  Each record's ``.signal``
    is the 1-D array <n>(t) = Σ_n n · P(n,t), and ``.t`` is time in µs.
    """
    loader = _make_mean_clicks_loader()
    return loader.load(folder, observables=["mean_clicks"], **loader_kwargs)


def plot_click_rate(
    folder: str | Path,
    ax: plt.Axes | None = None,
    **loader_kwargs,
) -> plt.Figure:
    """Compute and plot the click rate r(t) = d<n>/dt as a function of time.

    Parameters
    ----------
    folder:
        Path to the parity-scan JSON folder.
    ax:
        Existing axes to plot into.  A new figure is created if None.
    **loader_kwargs:
        Forwarded to the ParityLoader (e.g. lambda_range, max_files, t_max).

    Returns
    -------
    fig : plt.Figure
    """
    return plot_click_rate_from_records(average_click_number(folder, **loader_kwargs), ax)


def plot_click_rate_from_records(
    records: list[ObservableRecord],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot click rate r(t) = d⟨n⟩/dt from pre-loaded ``ObservableRecord`` objects.

    Use this with records from ``SimClickRecordLoader().load_observable(path)``.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for rec in records:
        t_mid, r = _click_rate(rec.t, rec.signal)
        ax.plot(t_mid, r, label=f"λ={rec.lambda_val:.3f}")

    ax.set_xlabel("t (µs)")
    ax.set_ylabel("r(t)  [clicks / µs]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig

def _click_rate(t: np.ndarray, n_avg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_mid, r) where r = d<n>/dt via forward finite differences."""
    dt = np.diff(t)
    r = np.diff(n_avg) / dt
    t_mid = 0.5 * (t[:-1] + t[1:])
    return t_mid, r


def steady_state_stats(
    folder: str | Path,
    T: float = 35.0,
    **loader_kwargs,
) -> list[dict]:
    """Compute click probability and two-time click correlation at steady state.

    The click probability at time t is r(t) = d<n>/dt (the click rate is the
    probability density of a click, since clicks are binary indicators).

    The two-time click correlation is:
        C(t') = r(T) * r(T + t')
    which equals <click(T) · click(T+t')> under the assumption that clicks
    at different times are statistically independent (Poisson/quantum-jump).

    Parameters
    ----------
    folder:
        Path to the parity-scan JSON folder.
    T:
        Steady-state time in µs.

    Returns
    -------
    list of dicts, one per lambda, each containing:
        ``lambda_val``, ``T_actual`` (nearest grid point to T),
        ``r_ss``  (click probability density r(T)),
        ``t_prime``  (t' axis, starting at 0),
        ``click_corr``  (r(T) * r(T+t') for t' >= 0).
    """
    return steady_state_stats_from_records(average_click_number(folder, **loader_kwargs), T)


def steady_state_stats_from_records(
    records: list[ObservableRecord],
    T: float = 35.0,
) -> list[dict]:
    """Compute steady-state stats from pre-loaded ``ObservableRecord`` objects.

    Use this with records from ``SimClickRecordLoader().load_observable(path)``.
    """
    results = []
    for rec in records:
        t_mid, r = _click_rate(rec.t, rec.signal)
        T_idx = int(np.argmin(np.abs(t_mid - T)))
        r_ss = float(r[T_idx])
        t_prime = t_mid[T_idx:] - t_mid[T_idx]
  
        results.append(
            dict(
                lambda_val=rec.lambda_val,
                T_actual=float(t_mid[T_idx]),
                r_ss=r_ss,
                t_prime=t_prime,
            )
        )
    return results


def plot_steady_state_stats(
    folder: str | Path,
    T: float = 35.0,
    **loader_kwargs,
) -> plt.Figure:
    """Plot the click probability r(T) and two-time correlation r(T)*r(T+t')."""
    return plot_steady_state_stats_from_records(
        average_click_number(folder, **loader_kwargs), T
    )


def plot_steady_state_stats_from_records(
    records: list[ObservableRecord],
    T: float = 35.0,
) -> plt.Figure:
    """Plot steady-state stats from pre-loaded ``ObservableRecord`` objects."""
    return _plot_steady_state_stats_impl(steady_state_stats_from_records(records, T), T)


def _plot_steady_state_stats_impl(stats: list[dict], T: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_r, ax_corr = axes

    lambdas = [s["lambda_val"] for s in stats]
    r_ss_vals = [s["r_ss"] for s in stats]
    ax_r.plot(lambdas, r_ss_vals, "o-")
    ax_r.set_xlabel("λ")
    ax_r.set_ylabel(f"r(T={T} µs)  [clicks / µs]")
    ax_r.set_title("Steady-state click probability")

    
    ax_corr.set_xlabel("t'  (µs)")
    ax_corr.set_ylabel("r(T) · r(T+t')")
    ax_corr.set_title(f"Two-time click correlation,  T={T} µs")
    ax_corr.legend(fontsize=7)

    fig.tight_layout()
    return fig


# ===========================================================================
# TWO-POINT CONNECTED CORRELATOR  (from raw NPZ trajectories)
# ===========================================================================

def _correlator_single_anchor_theory(
    s: np.ndarray,
    T_idx: int,
    window: int,
    dt: float,
    as_density: bool = True,
) -> np.ndarray:
    """
    Estimate:
        <J(T+t') J(T)> - <J(T+t')><J(T)>

    For binary click bins s ∈ {0,1}, the binned covariance is
        <s_T s_{T+m}> - <s_T><s_{T+m}>.

    Divide by dt**2 to convert to a jump-rate density.
    """
    sT = s[:, T_idx]
    seg = s[:, T_idx:T_idx + window]

    pT = sT.mean()
    p_tau = seg.mean(axis=0)
    joint = (sT[:, None] * seg).mean(axis=0)

    C_bin = joint - pT * p_tau

    if as_density:
        return C_bin / dt**2
    return C_bin


def connected_correlator(
    rec: ClickRecordData,
    T: float,
    n_anchors: int = 10,
) -> dict:
    """Compute the connected two-point click correlator, averaged over anchor times.

    In steady state C(T, T+t') is stationary, so averaging over n_anchors
    evenly-spaced T values reduces noise without introducing bias.

    Parameters
    ----------
    rec : ClickRecordData
    T : float
        First anchor time in µs (steady-state start).
    n_anchors : int
        Number of evenly-spaced anchors from T to t_end.
        Set to 1 to use a single anchor at T.

    Returns
    -------
    dict with keys: ``lambda_val``, ``T_anchors``, ``t_prime``, ``C_conn``
    """
    s = rec.trajectories    # (N, T_steps)
    t = rec.t
    mean_all = s.mean(axis=0)

    T_start_idx = int(np.argmin(np.abs(t - T)))
    available = len(t) - T_start_idx   # time steps from T to end
    # Keep the first half of available as t' window; spread anchors in the other half
    window = max(available // 2, 5)
    T_end_idx = len(t) - window        # last anchor still has `window` t' points
    if n_anchors > 1 and T_end_idx > T_start_idx:
        anchor_indices = np.unique(
            np.linspace(T_start_idx, T_end_idx, n_anchors, dtype=int)
        )
    else:
        anchor_indices = np.array([T_start_idx], dtype=int)

    dt = float(t[1] - t[0])

    curves = [
        _correlator_single_anchor_theory(s, idx, window, dt, as_density=True)
        for idx in anchor_indices
    ]

    C_conn = np.nanmean(curves, axis=0)
    t_prime = np.arange(window) * dt

    return dict(
        lambda_val=rec.lambda_val,
        T_anchors=[float(t[i]) for i in anchor_indices],
        t_prime=t_prime,
        C_conn=C_conn,
    )


def compute_all_correlators(
    folder: str | Path,
    T: float = 35.0,
    n_anchors: int = 10,
    **loader_kwargs,
) -> list[dict]:
    """Load all NPZ files and compute the averaged connected correlator for each lambda."""
    return compute_all_correlators_from_records(
        ClickRecordNpzLoader().load(folder, **loader_kwargs), T, n_anchors
    )


def compute_all_correlators_from_records(
    records: list[ClickRecordData],
    T: float = 35.0,
    n_anchors: int = 10,
) -> list[dict]:
    """Compute the averaged connected correlator from pre-loaded ``ClickRecordData``.

    Use this with records from ``SimClickRecordLoader().load(path)``.
    """
    return [connected_correlator(rec, T, n_anchors=n_anchors) for rec in records]


# ===========================================================================
# MATRIX PENCIL ON CORRELATOR
# ===========================================================================

def correlator_poles(
    folder: str | Path,
    T: float = 35.0,
    order: int = 3,
    n_anchors: int = 10,
    **loader_kwargs,
) -> list[dict]:
    """Extract MPM poles from C_conn(T, T+t') for each lambda.

    Parameters
    ----------
    folder : str | Path
    T : float
        Steady-state anchor time in µs.
    order : int
        Number of poles to extract.

    Returns
    -------
    list of dicts, one per lambda:
        ``lambda_val``, ``T_actual``, ``t_prime``, ``C_conn``, ``poles``
    """
    return correlator_poles_from_records(
        compute_all_correlators(folder, T=T, n_anchors=n_anchors, **loader_kwargs),
        order,
    )


def correlator_poles_from_records(
    corrs: list[dict],
    order: int = 3,
) -> list[dict]:
    """Extract MPM poles from pre-computed correlator dicts.

    Use this with the output of ``compute_all_correlators_from_records()``.
    """
    results = []
    for c in corrs:
        t = c["t_prime"]
        C = np.real(c["C_conn"])   # guard: ensure real signal

        # Skip t'=0: that point is the self-click Kronecker delta
        # (C[0] = 1 - r_ss·dt ≈ 1 while all subsequent values are O(r_ss·dt)).
        # Its 100:1 amplitude over the rest dominates the Hankel matrix and
        # causes MPM to fit a spurious ultra-fast mode, leaving the real poles
        # as garbage.  The physical decay dynamics live entirely in t' > 0.
        t_mpm = t[1:]
        C_mpm = C[1:]

        try:
            # poles, _, _ = matrix_pencil_method(t_mpm, C_mpm, order)
            poles,_,_ = mpm_rrha(t_mpm, C_mpm, order, rtol = 1e-4, max_iter=100)
            poles = sort_poles_canonical(poles)

            # poles, s, L = matrix_pencil_method(t_mpm, C_mpm, order)

            dt = t_mpm[1] - t_mpm[0]
            z_back = np.exp(poles * dt)
            print("continuous poles:", poles)
            print("discrete roots:", z_back)

        except Exception as exc:
            print(f"[correlator_poles] MPM failed for lam={c['lambda_val']:.3f}: {exc}")
            poles = np.full(order, np.nan + 0j)
        results.append({**c, "poles": poles})
    return results


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_connected_correlator(
    folder: str | Path,
    T: float = 35.0,
    n_anchors: int = 10,
    ax: plt.Axes | None = None,
    **loader_kwargs,
) -> plt.Figure:
    """Plot C_conn(T, T+t') vs t' for all lambda values."""
    return plot_connected_correlator_from_records(
        ClickRecordNpzLoader().load(folder, **loader_kwargs), T, n_anchors, ax
    )


def plot_connected_correlator_from_records(
    records: list[ClickRecordData],
    T: float = 35.0,
    n_anchors: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot C_conn from pre-loaded ``ClickRecordData``.

    Use this with records from ``SimClickRecordLoader().load(path)``.
    """
    return _plot_connected_correlator_impl(
        compute_all_correlators_from_records(records, T, n_anchors), T, ax
    )


def _plot_connected_correlator_impl(
    corrs: list[dict],
    T: float,
    ax: plt.Axes | None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for c in corrs:
        ax.plot(c["t_prime"], c["C_conn"], label=f"λ={c['lambda_val']:.3f}")

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("t'  (µs)")
    ax.set_ylabel(r"$\langle r(T)\,r(T{+}t')\rangle - \langle r(T)\rangle\langle r(T{+}t')\rangle$")
    ax.set_title(f"Connected click correlator,  T={T} µs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_correlator_svd(
    folder: str | Path,
    T: float = 35.0,
    n_anchors: int = 10,
    n_singular_values: int = 8,
    **loader_kwargs,
) -> plt.Figure:
    """Plot Hankel singular values of C_conn as a function of lambda.

    One curve per singular value index k; x-axis is lambda.  Shows how
    each singular value varies across the transition — the signal singular
    values (low k) change with lambda while the noise floor (high k) stays flat.
    """
    return plot_correlator_svd_from_records(
        ClickRecordNpzLoader().load(folder, **loader_kwargs), T, n_anchors, n_singular_values
    )


def plot_correlator_svd_from_records(
    records: list[ClickRecordData],
    T: float = 35.0,
    n_anchors: int = 10,
    n_singular_values: int = 8,
) -> plt.Figure:
    """Plot Hankel SVD of C_conn from pre-loaded ``ClickRecordData``."""
    return _plot_correlator_svd_impl(
        compute_all_correlators_from_records(records, T, n_anchors), T, n_singular_values
    )


def _plot_correlator_svd_impl(
    corrs: list[dict],
    T: float,
    n_singular_values: int,
) -> plt.Figure:
    from mpm_analysis.analysis.matrix_pencil import build_hankel_matrix

    lambdas = np.array([c["lambda_val"] for c in corrs])
    sv_matrix = []
    for c in corrs:
        C = c["C_conn"]
        L = len(C) // 3
        H = build_hankel_matrix(C, L)
        sv = np.linalg.svd(H, compute_uv=False)
        sv_matrix.append(sv[:n_singular_values])
    sv_matrix = np.array(sv_matrix)   # (n_lambda, n_singular_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(n_singular_values):
        ax.semilogy(lambdas, sv_matrix[:, k], "o-", markersize=4, label=f"σ_{k+1}")

    ax.set_xlabel("λ")
    ax.set_ylabel("Singular value  σₖ")
    ax.set_title(f"Hankel SVD of C_conn vs λ,  T={T} µs")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlator_poles(
    folder: str | Path,
    T: float = 35.0,
    order: int = 3,
    n_anchors: int = 10,
    **loader_kwargs,
) -> plt.Figure:
    """Extract MPM poles from C_conn and plot Re/Im vs lambda."""
    return plot_correlator_poles_from_records(
        ClickRecordNpzLoader().load(folder, **loader_kwargs), T, order, n_anchors
    )


def plot_correlator_poles_from_records(
    records: list[ClickRecordData],
    T: float = 35.0,
    order: int = 3,
    n_anchors: int = 10,
) -> plt.Figure:
    """Plot MPM poles from pre-loaded ``ClickRecordData``."""
    return _plot_correlator_poles_impl(
        correlator_poles_from_records(
            compute_all_correlators_from_records(records, T, n_anchors), order
        )
    )


def _plot_correlator_poles_impl(results: list[dict]) -> plt.Figure:
    lambdas = np.array([r["lambda_val"] for r in results])
    all_poles = np.array([r["poles"] for r in results])   # (n_lambda, order)
    order = all_poles.shape[1]

    fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(11, 4))

    for k in range(order):
        ax_re.plot(lambdas, all_poles[:, k].real, "o-", label=f"pole {k+1}")
        ax_im.plot(lambdas, all_poles[:, k].imag, "o-", label=f"pole {k+1}")

    ax_re.set_xlabel("λ")
    ax_re.set_ylabel("Re(pole)  [1/µs]")
    ax_re.set_title("Decay rates from C_conn")
    ax_re.legend(fontsize=7)
    ax_re.axhline(0, color="k", linewidth=0.5, linestyle="--")

    ax_im.set_xlabel("λ")
    ax_im.set_ylabel("Im(pole)  [rad/µs]")
    ax_im.set_title("Oscillation frequencies from C_conn")
    ax_im.legend(fontsize=7)
    ax_im.axhline(0, color="k", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    return fig


# ===========================================================================
# CLICK-NUMBER HISTOGRAMS  P(n, t)
# ===========================================================================

def click_number_histograms(
    rec: ClickRecordData,
    t_values: list[float] | np.ndarray,
) -> list[dict]:
    """Return P(n) at each requested time for one record.

    For each t_j, n_i(t_j) = number of clicks trajectory i accumulated up to t_j.
    Returns one dict per t_j with keys: ``t_actual``, ``n_counts``, ``prob``.
    """
    s = rec.trajectories   # (N, T), binary
    t = rec.t

    # Cumulative click count per trajectory vs time: shape (N, T)
    n_cumul = np.cumsum(s, axis=1)

    results = []
    for t_req in t_values:
        idx = int(np.argmin(np.abs(t - t_req)))
        n_at_t = n_cumul[:, idx].astype(int)   # length N
        n_max = int(n_at_t.max()) if len(n_at_t) > 0 else 0
        counts = np.bincount(n_at_t, minlength=n_max + 1)
        results.append(dict(
            t_actual=float(t[idx]),
            n_counts=counts,
            prob=counts / counts.sum(),
        ))
    return results


def plot_click_number_histograms(
    folder: str | Path,
    t_values: list[float] | None = None,
    n_lambda: int = 5,
    **loader_kwargs,
) -> plt.Figure:
    """Plot histograms of P(n) at several times t for a selection of λ values.

    Parameters
    ----------
    folder : str | Path
    t_values : list of floats (µs)
        Times at which to show P(n).  Default: 5 evenly-spaced values over
        the full time axis of the first loaded record.
    n_lambda : int
        Number of λ values to show (evenly spaced among loaded files).
    **loader_kwargs : forwarded to ClickRecordNpzLoader.load
    """
    records = ClickRecordNpzLoader().load(folder, **loader_kwargs)
    return plot_click_number_histograms_from_records(records, t_values, n_lambda)


def plot_click_number_histograms_from_records(
    records: list[ClickRecordData],
    t_values: list[float] | None = None,
    n_lambda: int = 5,
) -> plt.Figure:
    """Plot P(n) histograms from pre-loaded ClickRecordData."""
    if not records:
        raise ValueError("No records provided.")

    # Select a subset of λ values evenly spaced through the loaded list
    if len(records) <= n_lambda:
        selected = records
    else:
        idx = np.round(np.linspace(0, len(records) - 1, n_lambda)).astype(int)
        selected = [records[i] for i in idx]

    # Default t_values: 5 points from ~10% to ~90% of the time axis
    if t_values is None:
        t_ref = selected[0].t
        t_values = list(np.linspace(t_ref[len(t_ref) // 10], t_ref[-len(t_ref) // 10], 5))

    n_rows = len(selected)
    n_cols = len(t_values)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows),
        sharex=False, sharey=False,
    )
    # Normalise axes to 2-D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, rec in enumerate(selected):
        hists = click_number_histograms(rec, t_values)
        for col, h in enumerate(hists):
            ax = axes[row, col]
            n_vals = np.arange(len(h["prob"]))
            ax.bar(n_vals, h["prob"], width=0.8, color="steelblue", alpha=0.8)
            if row == 0:
                ax.set_title(f"t = {h['t_actual']:.1f} µs", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"λ={rec.lambda_val:.3f}\nP(n)", fontsize=7)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("n clicks", fontsize=7)
            ax.tick_params(labelsize=7)

    fig.suptitle("P(n) — click-number distribution", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # fig_pn = plot_click_number_histograms(PARITY_FOLDER, n_lambda=5)
    # plt.show()

    # fig1 = plot_click_rate(PARITY_FOLDER)
    # fig1.suptitle("Click rate r(t) vs time", fontsize=11)
    # plt.show()

    # fig2 = plot_connected_correlator(PARITY_FOLDER, T=20, n_anchors=4)
    # fig2.suptitle("Connected two-point click correlator", fontsize=11)
    # plt.show()

    # fig3 = plot_correlator_svd(PARITY_FOLDER, T=20, n_anchors=4)
    # fig3.suptitle("Hankel SVD of C_conn — model order selection", fontsize=11)
    # plt.show()

    # fig4 = plot_correlator_poles(PARITY_FOLDER, T=20, order=2, n_anchors=4)
    # fig4.suptitle("MPM poles from C_conn vs λ", fontsize=11)
    # plt.show()
    path = "click_record_raw_20260526_113719.npz"

    # # click rate / steady-state (fast — uses pre-computed P(n,t))
    # # obs = SimClickRecordLoader().load_observable(path)
    # # fig1 = plot_click_rate_from_records(obs)
    # # fig1.suptitle("Click rate r(t) vs time", fontsize=11)
    # # plt.show()

    # # correlator analysis (uses full trajectories)
    recs = SimClickRecordLoader().load(path)
    # # fig2 = plot_connected_correlator_from_records(recs, T=35.0)
    # # fig2.suptitle("Connected two-point click correlator", fontsize=11)
    # # plt.show()

    # fig4 = plot_correlator_svd_from_records(recs, T=30.0, n_singular_values=5)
    # fig4.suptitle("Hankel SVD of C_conn — model order selection", fontsize=11)
    # plt.show()

    fig3 = plot_correlator_poles_from_records(recs, T=30.0, order=3)
    fig3.suptitle("MPM poles from C_conn vs λ", fontsize=11)
    plt.show()