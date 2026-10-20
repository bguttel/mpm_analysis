"""
eigenvalues_picture.py
----------------------
Spectral origin of the hierarchy reversal: postselected Liouvillian spectrum
in the ideal vs. realistic case.

Design intent (see the commented-out caption at the bottom):
  * In the OSCILLATORY regime (lambda < lambda_1^c) the ideal spectrum has
    THREE coincident real parts (e_3 = Re e_1 = Re e_2), so the freezing
    condition Re(e_2) = 2 e_3 can never be met there.
  * Decoherence lifts that locking: the realistic spectrum has only TWO
    coincident real parts (Re e_1 = Re e_2), while e_3 splits off -- which is
    exactly what lets the condition be satisfied at lambda_2^c < lambda_1^c.

The 3-vs-2 overlap is made visible with complementary dash phases drawn over a
thicker solid branch, so every coincident curve shows its own colour.
"""

from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from mpm_analysis.plotting._style import (
    DPI,
    FIT_LINESTYLE,
    LABELS_FONT_SIZE,
    LAMBDA_OBS_COLOR,
    LINE_WIDTH,
    MUSTARD,
    PURPLE_MAGENTA,
    TEAL,
    TWO_COLUMN_WIDTH,
    make_ax_lines_wider,
)

LABELS_FONT_SIZE *= 1.5
LEGEND_FONT_SIZE = LABELS_FONT_SIZE / 1.3
LINE_WIDTH *= 1.5

# ============================================================
# User settings
# ============================================================

OUTDIR = Path(".")
STEM = "hierarchy_reversal_spectrum"

LAMBDA_MIN = 0.4
LAMBDA_MAX = 1.6
NPTS = 2000

# ============================================================
# Physical parameters from the paper
# ============================================================

Omega_S = 2 * np.pi * 100e3  # 100 kHz
Tphi_S = 26e-6  # 26 microseconds
T1_S = 93e-6  # 93 microseconds

gamma_phi = 1.0 / (Omega_S * Tphi_S)
gamma1 = 1.0 / (Omega_S * T1_S)

# ============================================================
# Matplotlib style -- paper defaults
# ============================================================

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "sans-serif",
        "font.size": LABELS_FONT_SIZE,
        "axes.labelsize": LABELS_FONT_SIZE,
        "axes.titlesize": LABELS_FONT_SIZE,
        "legend.fontsize": LABELS_FONT_SIZE - 1,
        "xtick.labelsize": LABELS_FONT_SIZE - 1,
        "ytick.labelsize": LABELS_FONT_SIZE - 1,
        "axes.linewidth": LINE_WIDTH,
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": DPI,
        "figure.dpi": 300,
    }
)

# Paper palette (ThirdTransitionFigureColors), consistent with the
# first-transition figure: e1 -> magenta, e2 -> teal, e3 -> mustard.
# Branches are indexed by decay rate at every lambda (see
# eig_sorted_by_real): e3 is the SLOWEST decaying, e1 the fastest.  Colour
# follows that ordering, so mustard -> purple -> teal always reads
# slow -> fast in both panels.  The 2*e_3 guide shares the e3 colour.
colors = {
    "e3": MUSTARD,
    "e1": PURPLE_MAGENTA,
    "e2": TEAL,
}

# ---- the overlap trick -------------------------------------------------
# e_3 is drawn thick and solid at the bottom; e_2 and e_1 are drawn on top
# with COMPLEMENTARY dash phases, so where branches coincide the eye sees
# alternating teal/magenta segments riding on a mustard halo.  Ideal panel
# => three colours in the stripe; realistic panel => only two, with the
# mustard branch peeled away.

# Each branch is drawn solid in z-order e3 < e2 < e1, then e3 and e2 are
# REDRAWN on top as dashes with complementary phases of a common period.
# Off a coincidence the overlay lands exactly on its own solid curve and is
# invisible; on a coincidence it punches through the branch above it, so the
# shared segment reads as a repeating mustard / teal / magenta rainbow.
DASH_PERIOD = 9.0
DASH_ON = 3.0
DASH_E3 = (0.0, (DASH_ON, DASH_PERIOD - DASH_ON))  # mustard slot
DASH_E2 = (DASH_ON, (DASH_ON, DASH_PERIOD - DASH_ON))  # teal slot
# the remaining third of every period is left to the magenta e1 underneath

# ============================================================
# Model
# ============================================================


def L_postselected(lam, gamma_phi=0.0, gamma1=0.0, Omega=1.0):
    """
    Postselected Liouvillian in the (p, y, z) basis.
    Here lam = alpha/(2*Omega).
    """
    alpha_half = lam * Omega
    gamma2 = gamma_phi + 0.5 * gamma1

    return np.array(
        [
            [-alpha_half, 0.0, alpha_half],
            [0.0, -(gamma2 + alpha_half), Omega],
            [alpha_half - gamma1, -Omega, -(alpha_half + gamma1)],
        ],
        dtype=complex,
    )


def track_branches(lams, gamma_phi=0.0, gamma1=0.0, Omega=1.0):
    """
    Continuously track the three eigenvalue branches.
    We initialize branch 3 as the nearly-real branch at lambda=0.
    """
    vals0 = np.linalg.eigvals(L_postselected(lams[0], gamma_phi, gamma1, Omega))

    idx_real = np.argmin(np.abs(np.imag(vals0)))
    rest = [i for i in range(3) if i != idx_real]
    rest = sorted(rest, key=lambda i: np.imag(vals0[i]))

    prev = np.array([vals0[rest[0]], vals0[rest[1]], vals0[idx_real]], dtype=complex)
    tracked = [prev.copy()]

    for lam in lams[1:]:
        vals = np.linalg.eigvals(L_postselected(lam, gamma_phi, gamma1, Omega))

        best = None
        best_cost = None
        for perm in permutations(range(3)):
            cand = vals[list(perm)]
            cost = np.sum(np.abs(cand - prev))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = cand

        prev = best
        tracked.append(prev.copy())

    return np.array(tracked)  # shape (N, 3)


def eig_sorted_by_real(lams, gamma_phi=0.0, gamma1=0.0, Omega=1.0):
    """
    For each lambda, sort eigenvalues by real part descending:
    [e3 (slowest), e2, e1].

    Real parts tie below lambda_1^c -- for the conjugate pair always, and in
    the ideal case for all three branches at once -- so the Re-sort alone
    leaves the labelling free to jump from one lambda to the next.  Break
    ties by |Im| ascending first, which keeps the purely real branch as e3,
    then by Im descending, so e2 is the +Im partner and e1 the -Im one.
    Past the split every Im is zero and the order is decided by Re alone.
    """
    out = []
    for lam in lams:
        vals = np.linalg.eigvals(L_postselected(lam, gamma_phi, gamma1, Omega))
        # round the tie keys so branches separated only by round-off still
        # count as degenerate
        order = np.lexsort(
            (
                -np.round(np.imag(vals), 9),  # last resort: +Im before -Im
                np.round(np.abs(np.imag(vals)), 9),  # then: most-real branch first
                -np.round(np.real(vals), 9),  # primary: slowest decay first
            )
        )
        out.append(vals[order])
    return np.array(out)


def find_lambda1(lams, gamma_phi=0.0, gamma1=0.0):
    """
    First transition: coalescence of the complex pair, i.e. where
    |Im e_{1,2}| vanishes.
    """
    tr = track_branches(lams, gamma_phi, gamma1)
    imagmag = np.maximum(np.abs(np.imag(tr[:, 0])), np.abs(np.imag(tr[:, 1])))

    idx = np.where(imagmag < 1e-6)[0]
    if len(idx) == 0:
        return np.nan

    i1 = idx[0]
    if i1 == 0:
        return lams[0]

    i0 = i1 - 1
    x0, x1 = lams[i0], lams[i1]
    y0, y1 = imagmag[i0], imagmag[i1]

    # linear interpolation
    return x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)


def find_lambda2(lams, gamma_phi=0.0, gamma1=0.0, lam1=None):
    """
    Second transition: Re(e2) = 2 e3.
    We select the root immediately below lam1, since that is the
    physically relevant one for the hierarchy-reversal discussion.
    """
    s = eig_sorted_by_real(lams, gamma_phi, gamma1)
    e3 = s[:, 0]
    e2 = s[:, 1]

    f = np.real(e2 - 2 * e3)
    roots = []

    for i in range(len(lams) - 1):
        if f[i] == 0 or f[i] * f[i + 1] < 0:
            x0, x1 = lams[i], lams[i + 1]
            y0, y1 = f[i], f[i + 1]
            root = x0 - y0 * (x1 - x0) / (y1 - y0)
            roots.append(root)

    if not roots:
        return np.nan

    if lam1 is None:
        return roots[-1]

    roots_below = [r for r in roots if r < lam1]
    if roots_below:
        return roots_below[-1]
    return roots[-1]


# ============================================================
# Compute spectra
# ============================================================

lams = np.linspace(LAMBDA_MIN, LAMBDA_MAX, NPTS)

ideal_tr = track_branches(lams, gamma_phi=0.0, gamma1=0.0)
real_tr = track_branches(lams, gamma_phi=gamma_phi, gamma1=gamma1)

ideal_sorted = eig_sorted_by_real(lams, gamma_phi=0.0, gamma1=0.0)
real_sorted = eig_sorted_by_real(lams, gamma_phi=gamma_phi, gamma1=gamma1)

lam1_ideal = 1.0
lam2_ideal = 2.0 / np.sqrt(3.0)

lam1_real = find_lambda1(lams, gamma_phi=gamma_phi, gamma1=gamma1)
lam2_real = find_lambda2(lams, gamma_phi=gamma_phi, gamma1=gamma1, lam1=lam1_real)

# ============================================================
# Helpers
# ============================================================


def add_lambda_marker(ax, lam):
    """Star marker sitting on the x-axis spine (paper idiom)."""
    ax.plot(
        lam,
        0.0,
        transform=ax.get_xaxis_transform(),
        marker="*",
        color=LAMBDA_OBS_COLOR,
        markersize=10.0,
        clip_on=False,
        zorder=100,
        ls="none",
    )


def add_critical_lines(ax, lam1, lam2, *, marker=True):
    """Vertical lines at the two transitions, styled as in the appendix figure."""
    ax.axvline(
        lam2,
        color=LAMBDA_OBS_COLOR,
        ls=(0, (1, 1.6)),
        lw=LINE_WIDTH * 1.2,
        alpha=1.0,
        zorder=0,
    )
    ax.axvline(
        lam1,
        color=LAMBDA_OBS_COLOR,
        ls=(0, (5, 2.5)),
        lw=LINE_WIDTH * 1.2,
        alpha=1.0,
        zorder=0,
    )
    if marker:
        add_lambda_marker(ax, lam1)
        add_lambda_marker(ax, lam2)


def add_critical_ticks(ax, lam1, lam2):
    """Secondary bottom axis carrying labelled ticks at the critical lambdas."""
    # sort by position so the labels never swap (in the ideal case
    # lambda_2^c > lambda_1^c, in the realistic case the order reverses)
    pairs = sorted(
        [
            (lam1, r"$\lambda_\mathrm{QJ}$"),
            (lam2, r"$\lambda_\mathrm{SF}$"),
        ]
    )
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([p[0] for p in pairs])
    sec.set_xticklabels([p[1] for p in pairs], va="bottom")
    sec.tick_params(
        axis="x",
        colors=LAMBDA_OBS_COLOR,
        width=0,
        length=0,
        pad=1.0,
        labelsize=LEGEND_FONT_SIZE,
    )

    for lbl, alignment in zip(sec.get_xticklabels(), ("center", "center")):
        lbl.set_horizontalalignment(alignment)
        lbl.set_color(LAMBDA_OBS_COLOR)
    sec.spines["top"].set_linewidth(LINE_WIDTH)
    return sec


def plot_real_parts(ax, sorted_vals):
    """Top-row panel: real parts, with the dash-phase overlap trick."""
    ax.plot(
        lams,
        -np.real(sorted_vals[:, 0]),
        color=colors["e3"],
        lw=LINE_WIDTH,
        solid_capstyle="round",
        label=r"$-e_3$",
        zorder=2,
    )
    ax.plot(
        lams,
        -np.real(sorted_vals[:, 1]),
        color=colors["e2"],
        lw=LINE_WIDTH,
        label=r"$-e_2$",
        zorder=3,
    )
    ax.plot(
        lams,
        -np.real(sorted_vals[:, 2]),
        color=colors["e1"],
        lw=LINE_WIDTH,
        label=r"$-e_1$",
        zorder=4,
    )
    # rainbow overlay on the coincident stretch
    ax.plot(
        lams,
        -np.real(sorted_vals[:, 0]),
        color=colors["e3"],
        lw=LINE_WIDTH,
        ls=DASH_E3,
        zorder=6,
    )
    ax.plot(
        lams,
        -np.real(sorted_vals[:, 1]),
        color=colors["e2"],
        lw=LINE_WIDTH,
        ls=DASH_E2,
        zorder=7,
    )

    ax.plot(
        lams,
        -2 * np.real(sorted_vals[:, 0]),
        color=colors["e3"],
        lw=LINE_WIDTH * 0.8,
        ls=FIT_LINESTYLE,
        label=r"$-2e_3$",
        zorder=5,
    )


def plot_imag_parts(ax, sorted_vals):
    """Bottom-row panel: Im e_n, same branch ordering and dash trick."""
    ax.plot(
        lams,
        np.imag(sorted_vals[:, 0]),
        color=colors["e3"],
        lw=LINE_WIDTH,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        lams,
        np.imag(sorted_vals[:, 1]),
        color=colors["e2"],
        lw=LINE_WIDTH,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        lams, np.imag(sorted_vals[:, 2]), color=colors["e1"], lw=LINE_WIDTH, zorder=3
    )

    # rainbow overlay on the collapsed-to-zero stretch
    ax.plot(
        lams,
        np.imag(sorted_vals[:, 0]),
        color=colors["e3"],
        lw=LINE_WIDTH,
        ls=DASH_E3,
        zorder=4,
    )
    ax.plot(
        lams,
        np.imag(sorted_vals[:, 1]),
        color=colors["e2"],
        lw=LINE_WIDTH,
        ls=DASH_E2,
        zorder=5,
    )


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH * 0.62),
    sharex="col",
    sharey="row",
    layout="constrained",
)

(ax_ir, ax_rr), (ax_ii, ax_ri) = axes

# left-aligned: centred they would collide with the critical-lambda labels
# carried by the secondary top axis
ax_ir.set_title("Ideal", pad=0, fontsize=LABELS_FONT_SIZE, loc="left")
ax_rr.set_title("Realistic", pad=0, fontsize=LABELS_FONT_SIZE, loc="left")

plot_real_parts(ax_ir, ideal_sorted)
plot_real_parts(ax_rr, real_sorted)
plot_imag_parts(ax_ii, ideal_sorted)
plot_imag_parts(ax_ri, real_sorted)

add_critical_ticks(ax_ir, lam1_ideal, lam2_ideal)
add_critical_ticks(ax_rr, lam1_real, lam2_real)

# vertical guides at the two transitions -- same lambda in both rows of a
# column, so the lines read as continuous across the stacked panels.
add_critical_lines(ax_ir, lam1_ideal, lam2_ideal, marker=False)
add_critical_lines(ax_ii, lam1_ideal, lam2_ideal, marker=False)
add_critical_lines(ax_rr, lam1_real, lam2_real, marker=False)
add_critical_lines(ax_ri, lam1_real, lam2_real, marker=False)

# ============================================================
# Axes formatting
# ============================================================

TICK_LABEL_SIZE = LABELS_FONT_SIZE - 4

for ax in (ax_ir, ax_rr):
    ax.set_xlim(LAMBDA_MIN, LAMBDA_MAX)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    make_ax_lines_wider(ax, labelsize=TICK_LABEL_SIZE, linewidth=LINE_WIDTH)

for ax in (ax_ii, ax_ri):
    # pushed to the right end and pulled up: centred it would sit right on
    # top of the lambda_1^c / lambda_2^c tick labels
    ax.set_xlabel(r"$\lambda$", labelpad=2)
    ax.set_xlim(LAMBDA_MIN, LAMBDA_MAX)
    ax.set_yticks([-0.5, 0.0, 0.5])
    # tick at lambda = 1.0 but no label -- the critical-lambda labels live there
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xticklabels(["0.5", "1.0", "1.5"])
    make_ax_lines_wider(ax, labelsize=TICK_LABEL_SIZE, linewidth=LINE_WIDTH)

ax_ir.set_ylabel(r"$-\mathrm{Re}(e_k)/\Omega_\mathrm{S}$")
ax_ii.set_ylabel(r"$\mathrm{Im}(e_k)/\Omega_\mathrm{S}$")

# put both y labels at the same x position so they stack exactly
fig.align_ylabels([ax_ir, ax_ii])

# ---- legends, as in the pole plots ----
leg = ax_ir.legend(
    loc="upper left",
    fontsize=LEGEND_FONT_SIZE,
    ncol=1,
    handlelength=2.8,
    handletextpad=0.5,
    labelspacing=0.3,
    borderpad=0.35,
)

# the curve's own dash pattern is scaled by its line width, so at handle
# length it renders as one long stroke -- give the key a tighter pattern so
# the 2*e_3 guide is unmistakably dashed against the solid branches
leg.legend_handles[-1].set_dashes([4.0, 2.0])

# ============================================================
# Save
# ============================================================

for ext in ("pdf", "svg", "png"):
    path = OUTDIR / f"{STEM}.{ext}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.1, transparent=True)
    print(f"Saved: {path}")

print(f"lambda1_ideal = {lam1_ideal:.6f}")
print(f"lambda2_ideal = {lam2_ideal:.6f}")
print(f"lambda1_real  = {lam1_real:.6f}")
print(f"lambda2_real  = {lam2_real:.6f}")

plt.show()
