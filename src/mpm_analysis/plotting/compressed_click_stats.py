"""
compressed_click_stats.py
-------------------------
Diagnostic figures for the **compressed click records** — counting-statistics
histograms and per-λ summary statistics in real space (jump number ``n``, not
the counting field ``k``).

Two figures are produced:

* ``counting_statistics_histograms`` — a grid of ``P(n, t)`` bar histograms
  (rows = selected λ, columns = selected compressed times).  Shows how the
  jump-number distribution broadens with time at each λ.
* ``click_summary_vs_lambda`` — four panels vs λ: mean clicks/shot, Fano factor
  ``Var(n)/⟨n⟩``, mean sequence time, and the alive-shot fraction vs compressed
  time (the shot-noise budget).  Error bars are the run-to-run standard error
  (the 25 runs are the resampling unit).

Plus the dead-time set, one figure per λ and one summary:

* ``dead_time_hist_lam<λ>`` — the normalised distribution of bright-run lengths
  (the consecutive ``1``s the compression collapses), with the MLE exponential
  fit and the mean written on the panel.
* ``dead_time_vs_lambda`` — mean dead time and fitted ``τ`` vs λ, with the
  underlying numbers saved alongside as ``dead_time_vs_lambda_*.npz``.

All figures are written to the analysis subfolder with timestamped,
non-clobbering names and informative, metadata-bearing titles.  Styling follows
``plotting/_style.py`` (paper palette, mathtext labels) — deliberately not the
generic matplotlib defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless-safe; we only save PNGs
import matplotlib.pyplot as plt
import numpy as np

from mpm_analysis.io.experimental.compressed_click_record import (
    CompressedClickRecord, _unique_path, fit_dead_time_exponential,
)
from mpm_analysis.io.file_naming import build_figure_filename
from mpm_analysis.plotting._style import (
    ONE_COLUMN_WIDTH, TWO_COLUMN_WIDTH, MARKER_SIZE, LINE_WIDTH,
    FIT_LINEWIDTH, FIT_LINESTYLE, AZURE_BLUE, PURPLE_MAGENTA,
    apply_paper_style, make_ax_lines_wider,
)
from mpm_analysis.utils.windows_paths import win_long_path


# ---------------------------------------------------------------------------
# Per-record statistics with run-to-run error bars
# ---------------------------------------------------------------------------

def _per_run_split(values: np.ndarray, rec: CompressedClickRecord) -> list[np.ndarray]:
    rso = rec.run_shot_offsets
    return [values[int(rso[r]):int(rso[r + 1])] for r in range(rec.n_runs)]


def record_summary(rec: CompressedClickRecord) -> dict:
    """Mean clicks, Fano factor and mean sequence time, with run SEM error bars."""
    jc = rec.jump_counts.astype(np.float64)
    seq_us = rec.compressed_lengths.astype(np.float64) * rec.dt_us

    per_run_mean = np.array([v.mean() for v in _per_run_split(jc, rec)])
    per_run_var = np.array([v.var() for v in _per_run_split(jc, rec)])
    per_run_seq = np.array([v.mean() for v in _per_run_split(seq_us, rec)])
    n = max(rec.n_runs, 1)
    sem = lambda a: float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    mean_clicks = float(jc.mean())
    var_clicks = float(jc.var())
    fano = var_clicks / mean_clicks if mean_clicks > 0 else np.nan
    # Fano error via run spread of per-run Fano.
    per_run_fano = np.where(per_run_mean > 0, per_run_var / per_run_mean, np.nan)

    return dict(
        lambda_val=rec.lambda_val,
        rate_khz=rec.rate_khz,
        mean_clicks=mean_clicks, mean_clicks_err=sem(per_run_mean),
        fano=fano, fano_err=sem(per_run_fano[np.isfinite(per_run_fano)]),
        mean_seq_us=float(seq_us.mean()), mean_seq_us_err=sem(per_run_seq),
    )


# ---------------------------------------------------------------------------
# Title / metadata helper
# ---------------------------------------------------------------------------

def _suptitle(records: list[CompressedClickRecord], what: str) -> str:
    r0 = records[0]
    scan = r0.metadata.get("scan_number")
    rates = sorted(r.rate_khz for r in records)
    n_runs = r0.n_runs
    n_shots = int(r0.jump_counts.size)
    dt = r0.dt_us
    built = r0.metadata.get("built_utc", "")
    return (
        f"{what}  |  scan {scan}  |  "
        f"$\\Omega_{{BG}}/2\\pi$ = {rates[0]}-{rates[-1]} kHz  |  "
        f"{len(records)} $\\lambda$, {n_runs} runs x {n_shots:,} shots  |  "
        f"dt = {dt:.3f} $\\mu$s  |  built {built} UTC"
    )


# ---------------------------------------------------------------------------
# Figure 1 — counting-statistics histograms  P(n, t)
# ---------------------------------------------------------------------------

def plot_counting_histograms(
    records: list[CompressedClickRecord],
    *,
    n_lambda_show: int = 4,
    time_fracs: tuple[float, ...] = (0.25, 0.5, 0.9),
) -> plt.Figure:
    """Grid of ``P(n, t)`` bar histograms: rows = λ, columns = compressed times.

    ``time_fracs`` are fractions of each record's compressed time span at which
    to sample the distribution.
    """
    apply_paper_style()
    recs = sorted(records, key=lambda r: r.lambda_val)
    if len(recs) > n_lambda_show:
        idx = np.linspace(0, len(recs) - 1, n_lambda_show).round().astype(int)
        recs = [recs[i] for i in idx]

    n_rows, n_cols = len(recs), len(time_fracs)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(TWO_COLUMN_WIDTH, 1.7 * n_rows + 0.6),
        squeeze=False, sharex="col",
    )
    for row, rec in enumerate(recs):
        # Use the last bin where most shots are still alive for the time axis.
        t_last = int(np.searchsorted(-rec.shots_per_time, -0.1 * rec.shots_per_time[0]))
        t_last = max(t_last, 2)
        n_axis = np.arange(rec.pnt.shape[1])
        for col, frac in enumerate(time_fracs):
            ax = axes[row][col]
            t_idx = min(int(frac * t_last), rec.t.size - 1)
            pn = rec.pnt[t_idx]
            ax.bar(n_axis, pn, width=0.9, color=AZURE_BLUE, alpha=0.85,
                   edgecolor="none")
            mean_n = float(pn @ n_axis)
            ax.axvline(mean_n, color=PURPLE_MAGENTA, lw=LINE_WIDTH,
                       ls=(0, (4, 2)))
            # Trim the n-axis to where there is mass.
            nz = np.nonzero(pn > 1e-4)[0]
            if nz.size:
                ax.set_xlim(-0.5, nz[-1] + 1.5)
            if row == 0:
                ax.set_title(f"t = {rec.t[t_idx]:.1f} $\\mu$s", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"$\\lambda$ = {rec.lambda_val:.2f}\nP(n)", fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("clicks  n", fontsize=8)
            make_ax_lines_wider(ax, labelsize=8)
    fig.suptitle(_suptitle(records, "Counting statistics P(n,t)"), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — summary statistics vs λ
# ---------------------------------------------------------------------------

def plot_summary_vs_lambda(
    records: list[CompressedClickRecord],
) -> plt.Figure:
    """Four panels vs λ: mean clicks, Fano factor, mean sequence time, survival."""
    apply_paper_style()
    recs = sorted(records, key=lambda r: r.lambda_val)
    summ = [record_summary(r) for r in recs]
    lam = np.array([s["lambda_val"] for s in summ])

    fig, axes = plt.subplots(
        2, 2, figsize=(TWO_COLUMN_WIDTH, 0.62 * TWO_COLUMN_WIDTH),
    )
    (ax_mean, ax_fano), (ax_seq, ax_surv) = axes

    def _eb(ax, y, yerr, color, ylabel, title):
        ax.errorbar(lam, y, yerr=yerr, fmt="o-", color=color,
                    markersize=MARKER_SIZE * 2.5, lw=LINE_WIDTH,
                    capsize=2.0, elinewidth=LINE_WIDTH * 0.8)
        ax.set_xlabel("$\\lambda$  (Zeno parameter)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        make_ax_lines_wider(ax)

    _eb(ax_mean, [s["mean_clicks"] for s in summ], [s["mean_clicks_err"] for s in summ],
        AZURE_BLUE, r"$\langle n \rangle$  clicks/shot", "Mean clicks per shot")
    _eb(ax_fano, [s["fano"] for s in summ], [s["fano_err"] for s in summ],
        PURPLE_MAGENTA, r"Fano  $\mathrm{Var}(n)/\langle n\rangle$", "Counting Fano factor")
    ax_fano.axhline(1.0, color="0.5", lw=LINE_WIDTH * 0.7, ls=(0, (2, 2)))
    _eb(ax_seq, [s["mean_seq_us"] for s in summ], [s["mean_seq_us_err"] for s in summ],
        "#1f8a4c", r"$\langle T_{\mathrm{seq}}\rangle$  ($\mu$s)",
        "Mean compressed sequence time")

    # Survival / shot-noise budget: alive fraction vs compressed time per λ.
    cmap = plt.get_cmap("viridis")
    lam_norm = (lam - lam.min()) / (np.ptp(lam) or 1.0)
    for rec, ln in zip(recs, lam_norm):
        frac = rec.shots_per_time / rec.shots_per_time[0]
        ax_surv.plot(rec.t, frac, color=cmap(ln), lw=LINE_WIDTH * 0.8)
    ax_surv.set_xlabel("compressed time  ($\\mu$s)")
    ax_surv.set_ylabel("alive-shot fraction")
    ax_surv.set_title("Shot-noise budget vs compressed time", fontsize=8)
    make_ax_lines_wider(ax_surv)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(lam.min(), lam.max()))
    fig.colorbar(sm, ax=ax_surv, label="$\\lambda$", fraction=0.046, pad=0.04)

    fig.suptitle(_suptitle(records, "Click summary vs $\\lambda$"), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — dead-time distribution (bright-run lengths), one per λ
# ---------------------------------------------------------------------------

def _dead_time_xlim(counts: np.ndarray, quantile: float = 0.9999) -> int:
    """Last length worth showing: the ``quantile`` point, never below 5 bins."""
    total = counts.sum()
    if total <= 0:
        return 5
    hi = int(np.searchsorted(np.cumsum(counts) / total, quantile) + 1)
    return max(hi, 5)


def plot_dead_time_histogram(rec: CompressedClickRecord) -> plt.Figure:
    """Normalised dead-time distribution for one λ, with an exponential fit.

    The dead time is the length of a bright run — the consecutive ``1``s that
    the compression collapses into a single click.  Bars are the measured
    probabilities (pooled over all runs and shots); the dashed line is the MLE
    geometric/exponential law from :func:`fit_dead_time_exponential`.
    """
    apply_paper_style()
    counts = rec.dead_time_counts()
    pmf = rec.dead_time_pmf()
    t_us = rec.dead_time_us
    summ = rec.dead_time_summary()
    fit = fit_dead_time_exponential(counts, rec.dt_us)

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.72 * ONE_COLUMN_WIDTH))
    ax.bar(t_us, pmf, width=rec.dt_us * 0.85, color=AZURE_BLUE, alpha=0.85,
           edgecolor="none", label="measured")
    # Draw the fit only over its support, else it plunges to 0 below l_min.
    sup = slice(fit["l_min"], None)
    ax.plot(t_us[sup], fit["pmf_fit"][sup], color=PURPLE_MAGENTA, lw=LINE_WIDTH,
            ls=FIT_LINESTYLE,
            label=rf"fit  $\tau$ = {summ['tau_us']:.3f} $\mu$s")
    ax.axvline(summ["mean_us"], color="0.35", lw=FIT_LINEWIDTH, ls=(0, (2, 2)),
               label=rf"mean = {summ['mean_us']:.3f} $\mu$s")

    ax.set_yscale("log")
    hi = _dead_time_xlim(counts)
    ax.set_xlim(0.0, (hi + 1) * rec.dt_us)
    n_tot = max(int(counts.sum()), 1)
    ax.set_ylim(0.5 / n_tot, 2.0)
    ax.set_xlabel(r"dead time  $\tau_{\mathrm{bright}}$  ($\mu$s)")
    ax.set_ylabel("probability")
    ax.set_title(
        rf"Dead-time distribution  |  $\lambda$ = {rec.lambda_val:.3f}  "
        rf"|  {rec.rate_khz} kHz", fontsize=8,
    )
    ax.legend(fontsize=7, frameon=False)
    ax.text(
        0.97, 0.62,
        f"$\\langle \\tau_{{\\mathrm{{bright}}}}\\rangle$ = "
        f"{summ['mean_us']:.4f} $\\pm$ {summ['mean_us_err']:.4f} $\\mu$s\n"
        f"({summ['mean_steps']:.3f} steps of {rec.dt_us:.3f} $\\mu$s)\n"
        f"$\\tau_{{\\mathrm{{fit}}}}$ = {summ['tau_us']:.4f} $\\pm$ "
        f"{summ['tau_us_err']:.4f} $\\mu$s\n"
        f"N = {summ['n_bright_runs']:,} bright runs",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )
    make_ax_lines_wider(ax, labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — mean dead time vs λ
# ---------------------------------------------------------------------------

def dead_time_table(records: list[CompressedClickRecord]) -> dict:
    """Per-λ dead-time numbers as flat arrays, ready to save or tabulate."""
    recs = sorted(
        [r for r in records if r.has_dead_time], key=lambda r: r.lambda_val,
    )
    summ = [r.dead_time_summary() for r in recs]
    width = max((r.bright_hist.shape[1] for r in recs), default=1)
    pooled = np.zeros((len(recs), width), dtype=np.int64)
    for i, r in enumerate(recs):
        c = r.dead_time_counts()
        pooled[i, :c.size] = c
    dt = recs[0].dt_us if recs else float("nan")
    col = lambda k: np.array([s[k] for s in summ], dtype=np.float64)
    return dict(
        lambda_values=col("lambda_val"),
        rate_khz=np.array([s["rate_khz"] for s in summ], dtype=np.int64),
        mean_us=col("mean_us"), mean_us_err=col("mean_us_err"),
        mean_steps=col("mean_steps"),
        tau_us=col("tau_us"), tau_us_err=col("tau_us_err"),
        n_bright_runs=np.array([s["n_bright_runs"] for s in summ], dtype=np.int64),
        l_min=np.array([s["l_min"] for s in summ], dtype=np.int64),
        dt_us=dt,
        dead_time_steps=np.arange(width, dtype=np.int64),
        dead_time_us=np.arange(width, dtype=np.float64) * dt,
        bright_hist_pooled=pooled,
    )


def plot_dead_time_vs_lambda(records: list[CompressedClickRecord]) -> plt.Figure:
    """Mean dead time vs λ, with the fitted ``τ`` alongside it.

    Error bars are the run-to-run standard error.
    """
    apply_paper_style()
    tab = dead_time_table(records)
    lam = tab["lambda_values"]

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 0.72 * ONE_COLUMN_WIDTH))
    ax.errorbar(lam, tab["mean_us"], yerr=tab["mean_us_err"], fmt="o-",
                color=AZURE_BLUE, markersize=MARKER_SIZE * 2.5, lw=LINE_WIDTH,
                capsize=2.0, elinewidth=LINE_WIDTH * 0.8,
                label=r"mean  $\langle \tau_{\mathrm{bright}}\rangle$")
    ax.errorbar(lam, tab["tau_us"], yerr=tab["tau_us_err"], fmt="s--",
                color=PURPLE_MAGENTA, markersize=MARKER_SIZE * 2.5,
                lw=LINE_WIDTH, capsize=2.0, elinewidth=LINE_WIDTH * 0.8,
                label=r"exponential fit  $\tau$")
    ax.set_xlabel(r"$\lambda$  (Zeno parameter)")
    ax.set_ylabel(r"dead time  ($\mu$s)")
    ax.set_title("Mean dead time vs $\\lambda$", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    make_ax_lines_wider(ax, labelsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save the compressed-click stat figures (non-clobbering)
# ---------------------------------------------------------------------------
# NOTE: the k-space pole figures (eigenvalues-vs-λ, poles-vs-k, parametric, SVD
# spectrum, raw traces, braiding) now live in ``kspace.plotting`` and are saved
# by ``kspace.save_kspace_figures`` (used by ``KSpacePipeline``).  This module
# keeps only the compressed-click *counting-statistics* figures.

def _save_fig(fig, tag: str, out_dir: Path, scan, dpi: int) -> Path:
    stem = build_figure_filename(
        tag, "compressed_click", scan_number=scan, extension="png",
    )
    path = _unique_path(out_dir / stem)
    fig.savefig(win_long_path(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] saved {path.name}")
    return path


def save_dead_time_figures(
    records: list[CompressedClickRecord],
    out_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    """Save one dead-time histogram PNG per λ, the vs-λ summary, and the data.

    Prints the mean dead time and fitted ``τ`` for every λ, and writes
    ``dead_time_vs_lambda_*.npz`` holding those numbers together with the pooled
    histograms — everything needed to redo the analysis without re-plotting.
    """
    out_dir = Path(out_dir)
    os.makedirs(win_long_path(out_dir), exist_ok=True)
    scan = records[0].metadata.get("scan_number") if records else None

    usable = sorted(
        [r for r in records if r.has_dead_time], key=lambda r: r.lambda_val,
    )
    skipped = len(records) - len(usable)
    if skipped:
        print(f"[dead-time] {skipped} record(s) predate the dead-time histogram "
              f"and were skipped. Rebuild them from individual_runs to include "
              f"them (the collapsed 1s are not recoverable from the NPZ).")
    if not usable:
        return []

    saved: list[Path] = []
    print(f"[dead-time] mean bright-run length per lambda "
          f"(dt = {usable[0].dt_us:.4f} us):")
    for rec in usable:
        s = rec.dead_time_summary()
        print(f"  lambda = {s['lambda_val']:.4f}  ({s['rate_khz']:>4d} kHz)  "
              f"mean = {s['mean_us']:.4f} +/- {s['mean_us_err']:.4f} us "
              f"({s['mean_steps']:.3f} steps)  "
              f"tau_fit = {s['tau_us']:.4f} +/- {s['tau_us_err']:.4f} us  "
              f"N = {s['n_bright_runs']:,}")
        saved.append(_save_fig(
            plot_dead_time_histogram(rec),
            f"dead_time_hist_lam{rec.lambda_val:.3f}_{rec.rate_khz}kHz",
            out_dir, scan, dpi,
        ))

    saved.append(_save_fig(
        plot_dead_time_vs_lambda(usable), "dead_time_vs_lambda",
        out_dir, scan, dpi,
    ))

    stem = build_figure_filename(
        "dead_time_vs_lambda", "compressed_click", scan_number=scan,
        extension="npz",
    )
    data_path = _unique_path(out_dir / stem)
    np.savez_compressed(win_long_path(data_path), **dead_time_table(usable))
    print(f"[data]   saved {data_path.name}")
    saved.append(data_path)
    return saved


def save_compressed_click_figures(
    records: list[CompressedClickRecord],
    out_dir: str | Path,
    *,
    dpi: int = 300,
    n_lambda_show: int = 4,
    dead_time: bool = True,
) -> list[Path]:
    """Build and save the diagnostic figures as PNGs to ``out_dir``.

    Filenames are timestamped (``build_figure_filename``) and additionally
    guarded by :func:`_unique_path`, so previously stored figures are never
    overwritten.  ``dead_time`` also emits the per-λ dead-time histograms, the
    mean-vs-λ summary and its NPZ.  Returns the saved paths.
    """
    if not records:
        raise ValueError("No records to plot.")
    out_dir = Path(out_dir)
    os.makedirs(win_long_path(out_dir), exist_ok=True)
    scan = records[0].metadata.get("scan_number")

    saved: list[Path] = []
    specs = [
        ("counting_statistics_histograms",
         lambda: plot_counting_histograms(records, n_lambda_show=n_lambda_show)),
        ("click_summary_vs_lambda",
         lambda: plot_summary_vs_lambda(records)),
    ]
    for tag, builder in specs:
        saved.append(_save_fig(builder(), tag, out_dir, scan, dpi))

    if dead_time:
        saved.extend(save_dead_time_figures(records, out_dir, dpi=dpi))
    return saved
