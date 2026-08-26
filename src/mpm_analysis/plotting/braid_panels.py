"""
braid_panels.py
---------------
The four panels that make a braid topology readable, following
Pavlov/Gefen/Shnirman, Phys. Rev. B 111, 104301 (2025).

A — **braid diagram over the full period**.  Strand height vs k drawn over
    ``[0, L·2π]`` so a 4π or 6π component is shown actually closing.  The under
    strand is broken at every crossing, so this is a genuine braid diagram
    rather than a line plot; each crossing is labelled σᵢ^{±1}.  Double braiding
    becomes countable: two σ marks of the same generator.  (Paper Fig. 2 lower /
    Fig. 3 left.)

B — **winding accumulator**.  Cumulative winding of each gap Δ_ab = λ_a − λ_b,
    normalized so one counterclockwise loop = 1.  The braid index is the
    *endpoint height*: ends at 2 → double braiding; a 4π pair only reaches an
    integer after two laps.  This is the panel to read the index off.

C — **link cartoon**.  The closed braid as a link: one node per component
    labelled with its periodicity (2π/4π/6π) and self-writhe, edges labelled
    with the linking number lk(A,B) = ν_{A;B}/2.  A stand-in for the paper's
    Fig. 3-right knot pictures that stays legible for any number of strands.

D — **λ phase diagram**.  A strip coloured by topological class with the
    exceptional points marked, and ν(λ) below it.

Each panel comes in two pieces: a ``*_geometry`` builder that returns plain
arrays and is backend-agnostic, and a ``plot_*_panel`` renderer that draws it
into a matplotlib ``ax``.  The plotly HTML export in
``mpm_analysis.kspace.braid_html`` consumes the same geometry builders, so the
two backends can never drift apart.

The ``plot_*_panel(..., ax=...)`` signature matches the convention in
``plotting/survival_curves.py`` and ``plotting/pole_spectrum.py`` that the
``paper_figures`` modules compose.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from mpm_analysis.analysis.braid_topology import (
    DEFAULT_PROJ_MIX,
    BraidScan,
    BraidTopology,
    winding_curve,
)

# Components (panels A, C, D) get their own palette so that "one colour = one
# closed loop of the link" holds everywhere.  Individual strands keep the
# _POLE_COLORS_HEX used by every other k-space figure, so a strand can still be
# traced back to the poles-vs-k plots.  Both are colourblind-distinguishable.
COMPONENT_COLORS = ["#A13D97", "#1ABC9C", "#D4AC0D", "#3498DB", "#8787de", "#d02500"]

#: Classes get stable colours across every figure in a run.
_CLASS_COLORS = {
    "Class I": "#3498DB",
    "Class II": "#A13D97",
    "Class III": "#1ABC9C",
    "Class IV": "#D4AC0D",
    "Class V": "#BBBBBB",
}
#: Distinct muted shades for labels that match no paper class, so that two
#: different unclassified regimes are still visibly different in the strip.
_UNCLASSIFIED_COLORS = ["#5D6D7E", "#7F8C8D", "#34495E", "#95A5A6", "#4A6572"]

#: Strand colours inside a component, matching the k-space pole figures.
_STRAND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def component_color(idx: int) -> str:
    return COMPONENT_COLORS[idx % len(COMPONENT_COLORS)]


def strand_color(idx: int) -> str:
    """Same palette as ``kspace.plotting._POLE_COLORS_HEX`` (track colours)."""
    return _STRAND_COLORS[idx % len(_STRAND_COLORS)]


def class_color(label: str, *, unclassified_order: list[str] | None = None) -> str:
    """Stable colour per class label; mirrored classes share the base colour.

    Unclassified labels are handed distinct shades in order of first appearance
    (pass ``unclassified_order``), so a strip with several different
    unclassified regimes does not paint them all the same.
    """
    base = label.replace(" (mirror)", "")
    if base in _CLASS_COLORS:
        return _CLASS_COLORS[base]
    if unclassified_order is not None and label in unclassified_order:
        i = unclassified_order.index(label)
        return _UNCLASSIFIED_COLORS[i % len(_UNCLASSIFIED_COLORS)]
    return _UNCLASSIFIED_COLORS[0]


def _projection(z: np.ndarray, proj_mix: float = DEFAULT_PROJ_MIX) -> np.ndarray:
    """Braid-diagram height: the same planar projection used to find crossings."""
    return z.real + proj_mix * z.imag


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _superscript(exp: int) -> str:
    """``1`` → ``""``, ``-1`` → ``⁻¹``, ``-2`` → ``⁻²``, ``3`` → ``³``."""
    return "" if exp == 1 else str(int(exp)).translate(_SUPERSCRIPT)


# ---------------------------------------------------------------------------
# Panel A — braid diagram
# ---------------------------------------------------------------------------

@dataclass
class BraidDiagramGeometry:
    """Everything needed to draw panel A, in data coordinates."""

    k: np.ndarray                       # (n_ext,) shared x axis
    height: np.ndarray                  # (n_ext, M) strand heights, NaN in gaps
    strand_component: np.ndarray        # (M,) component index per strand
    crossing_k: np.ndarray              # (n_cross,)
    crossing_height: np.ndarray         # (n_cross,) where to put the label
    crossing_label: list[str]           # "σ2", "σ1⁻¹", or "σ2²" when merged
    seams: np.ndarray                   # k of the Brillouin-zone boundaries
    seam_labels: list[str]              # "2π", "4π", …
    y_lim: tuple[float, float]
    word_str: str
    n_laps: int


def braid_diagram_geometry(
    topo: BraidTopology,
    *,
    gap_frac: float = 0.010,
    merge_frac: float = 0.012,
    proj_mix: float = DEFAULT_PROJ_MIX,
) -> BraidDiagramGeometry:
    """Build the braid diagram: strand polylines with the under-strand broken.

    The break is cut along the **k axis** as a fixed fraction of the drawn k
    span, not in y.  That is what makes the gap robust: strand heights can span
    wildly different scales from one λ to the next (and the y-limits are not
    known until the axes exist), whereas the k span is always exactly
    ``n_laps · 2π``.  Because braid strands run left-to-right, an x-window cut
    is also the geometrically correct way to open an under-crossing.

    Crossings closer together than ``merge_frac`` of the k span are merged into a
    single label (``σ2²``) instead of overprinting each other.
    """
    un = topo.unrolled
    k = np.asarray(un.k, dtype=float)
    height = _projection(un.strands, proj_mix).copy()
    span = float(k[-1] - k[0]) or 1.0
    gap = gap_frac * span

    cross = sorted(topo.unrolled_crossings, key=lambda c: c.k)

    # Break the under strand: blank its samples inside a window around each
    # crossing so matplotlib/plotly render a genuine gap.
    for c in cross:
        inside = np.abs(k - c.k) < gap
        height[inside, c.strand_under] = np.nan

    # Merge near-coincident crossings into one label.
    merged_k: list[float] = []
    merged_h: list[float] = []
    merged_lbl: list[str] = []
    i = 0
    while i < len(cross):
        j = i
        while (j + 1 < len(cross)
               and (cross[j + 1].k - cross[i].k) < merge_frac * span
               and cross[j + 1].generator == cross[i].generator
               and cross[j + 1].sign == cross[i].sign):
            j += 1
        n = j - i + 1
        c = cross[i]
        # One signed exponent, not a sign glued to a power: two σ2⁻¹ crossings
        # are σ2⁻², and writing "σ2⁻¹²" would read as the twelfth inverse power.
        merged_lbl.append(f"σ{c.generator}{_superscript(c.sign * n)}")
        merged_k.append(float(np.mean([cross[m].k for m in range(i, j + 1)])))
        # label sits at the crossing height, read off the over strand
        idx = int(np.argmin(np.abs(k - merged_k[-1])))
        merged_h.append(float(_projection(un.strands[idx, c.strand_over], proj_mix)))
        i = j + 1

    finite = height[np.isfinite(height)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        pad = 0.18 * max(hi - lo, 1e-12)
        y_lim = (lo - pad, hi + pad)
    else:
        y_lim = (-1.0, 1.0)

    seams = np.asarray(un.seams, dtype=float)
    seam_labels = [f"{2 * (i + 1)}π" for i in range(len(seams))]

    return BraidDiagramGeometry(
        k=k, height=height, strand_component=un.component_of_strand,
        crossing_k=np.asarray(merged_k), crossing_height=np.asarray(merged_h),
        crossing_label=merged_lbl, seams=seams, seam_labels=seam_labels,
        y_lim=y_lim, word_str=topo.word_str, n_laps=un.n_laps,
    )


def plot_braid_diagram_panel(
    topo: BraidTopology,
    *,
    ax=None,
    show_word: bool = True,
    label_crossings: bool = True,
    max_labels: int = 12,
    fontsize: int = 7,
):
    """Draw panel A into ``ax`` (created if not supplied).

    When the projection produces more than ``max_labels`` crossings — which
    happens on noisy data, where two nearly degenerate strands cross and
    re-cross without changing the topology — the per-crossing σ labels are
    suppressed and replaced by a count.  The crossings themselves are still
    drawn as breaks: the diagram stays faithful to the data, it just stops
    printing a label per wiggle.  The reduced word in the title is unaffected,
    since it is computed from the single closed loop and freely reduced.
    """
    import matplotlib.pyplot as plt

    from mpm_analysis.plotting._style import LINE_WIDTH

    g = braid_diagram_geometry(topo)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 2.4))

    two_pi = 2.0 * np.pi

    # Brillouin-zone seams: where one lap ends and the next begins.
    for x, lbl in zip(g.seams, g.seam_labels):
        ax.axvline(x, color="0.55", lw=0.7, ls=(0, (3, 3)), zorder=1)
        ax.text(x, g.y_lim[1], f" {lbl}", ha="left", va="top",
                fontsize=fontsize - 1, color="0.45")

    # Each strand is drawn twice: a thick halo in its *component* colour (so one
    # colour = one closed loop of the link, which is what panel C indexes), and a
    # thin core in its *strand* colour (matching the poles-vs-k figures, so an
    # individual branch can still be traced).  Without the core, a 6π component
    # would be three indistinguishable curves of the same colour.
    for s in range(g.height.shape[1]):
        ci = int(g.strand_component[s])
        ax.plot(g.k, g.height[:, s], "-", color=component_color(ci),
                lw=LINE_WIDTH * 2.6, solid_capstyle="round", alpha=0.45, zorder=2)
        ax.plot(g.k, g.height[:, s], "-",
                color=strand_color(topo.strand_labels[s]),
                lw=LINE_WIDTH * 0.9, solid_capstyle="round", zorder=3)

    # Crossing labels.
    if label_crossings and len(g.crossing_k) > max_labels:
        ax.text(0.99, 0.94, f"{len(g.crossing_k)} crossings", transform=ax.transAxes,
                ha="right", va="top", fontsize=fontsize, color="0.35")
    elif label_crossings and len(g.crossing_k):
        last_x = -np.inf
        stagger = 0
        for x, y, lbl in zip(g.crossing_k, g.crossing_height, g.crossing_label):
            # alternate the label height when two crossings are close in k
            stagger = (stagger + 1) % 2 if (x - last_x) < 0.06 * (g.k[-1] - g.k[0]) else 0
            ax.annotate(
                lbl, (x, y), xytext=(0, 6 + 8 * stagger), textcoords="offset points",
                ha="center", va="bottom", fontsize=fontsize, color="#111111",
                zorder=5,
            )
            last_x = x

    ax.set_xlim(g.k[0], g.k[-1])
    ax.set_ylim(*g.y_lim)
    n_ticks = g.n_laps + 1
    ticks = g.k[0] + two_pi * np.arange(n_ticks)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{2 * (i + 1)}π" for i in range(n_ticks - 1)])
    ax.set_xlabel("counting field  k", fontsize=fontsize + 1)
    ax.set_ylabel("Re λ  (projected)", fontsize=fontsize + 1)

    title = f"{topo.class_label}  |  period {topo.period_label}"
    if show_word:
        title += f"  |  word {topo.word_str}"
    ax.set_title(title, fontsize=fontsize + 1, loc="left")

    if not topo.quality.trustworthy:
        ax.text(0.99, 0.03, "unreliable — see quality", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=fontsize, color="#d02500")
    return ax


# ---------------------------------------------------------------------------
# Panel B — winding accumulator
# ---------------------------------------------------------------------------

@dataclass
class WindingGeometry:
    k: np.ndarray
    curves: list[np.ndarray]            # cumulative winding, one per pair
    labels: list[str]
    totals: list[float]
    seams: np.ndarray
    pair_indices: list[tuple[int, int]]


def winding_curves_geometry(topo: BraidTopology) -> WindingGeometry:
    """Cumulative winding of every strand pair over the unrolled k axis.

    Normalized so one full counterclockwise loop is 1 — so the endpoint height
    *is* the ordered index ν^(a,b) accumulated over the drawn period, and the
    paper's ν_{a;b} is twice it.  A pair inside a 4π component only reaches an
    integer after the second lap, which is precisely the statement that it is not
    a 2π-closed pair.
    """
    un = topo.unrolled
    z = un.strands
    M = z.shape[1]
    curves, labels, totals, pairs = [], [], [], []
    for i, j in combinations(range(M), 2):
        cum, total = winding_curve(z[:, i] - z[:, j])
        curves.append(cum)
        totals.append(total)
        pairs.append((i, j))
        labels.append(f"{topo.strand_labels[i]}–{topo.strand_labels[j]}")
    return WindingGeometry(k=un.k, curves=curves, labels=labels, totals=totals,
                           seams=np.asarray(un.seams, dtype=float),
                           pair_indices=pairs)


def plot_winding_accumulator_panel(
    topo: BraidTopology,
    *,
    ax=None,
    fontsize: int = 7,
):
    """Draw panel B into ``ax``: read the index off the right-hand edge."""
    import matplotlib.pyplot as plt

    from mpm_analysis.plotting._style import LINE_WIDTH

    g = winding_curves_geometry(topo)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 2.4))

    for x in g.seams:
        ax.axvline(x, color="0.55", lw=0.7, ls=(0, (3, 3)), zorder=1)

    all_vals = np.concatenate([c for c in g.curves]) if g.curves else np.zeros(1)
    lo = np.floor(np.nanmin(all_vals) - 0.25)
    hi = np.ceil(np.nanmax(all_vals) + 0.25)
    for level in np.arange(lo, hi + 1):
        ax.axhline(level, color="0.88", lw=0.6, zorder=0)

    for idx, (cum, lbl, total) in enumerate(zip(g.curves, g.labels, g.totals)):
        color = COMPONENT_COLORS[idx % len(COMPONENT_COLORS)]
        ax.plot(g.k, cum, "-", color=color, lw=LINE_WIDTH, zorder=3,
                label=f"{lbl}: {total:+.2f}")

    ax.set_xlim(g.k[0], g.k[-1])
    ax.set_ylim(lo, hi)
    n_laps = topo.unrolled.n_laps
    ticks = g.k[0] + 2.0 * np.pi * np.arange(n_laps + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{2 * (i + 1)}π" for i in range(n_laps)])
    ax.set_xlabel("counting field  k", fontsize=fontsize + 1)
    ax.set_ylabel("accumulated winding  [loops]", fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 1, loc="upper left", frameon=True,
              framealpha=0.85, handlelength=1.4, borderpad=0.3)
    ax.set_title("index = endpoint height   (ν$_{a;b}$ = 2 × this)",
                 fontsize=fontsize + 1, loc="left")
    return ax


# ---------------------------------------------------------------------------
# Panel C — link cartoon
# ---------------------------------------------------------------------------

@dataclass
class LinkGraphGeometry:
    node_xy: np.ndarray                 # (C, 2) on the unit circle
    node_labels: list[str]
    node_colors: list[str]
    self_writhe: list[float]
    edges: list[tuple[int, int, float]]  # (a, b, linking number)


def link_graph_geometry(topo: BraidTopology) -> LinkGraphGeometry:
    """Lay the closed components out on a circle with linking numbers on edges."""
    comps = topo.unrolled.components
    n = len(comps)
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / max(n, 1)
    radius = 0.0 if n == 1 else 1.0
    node_xy = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)

    labels, colors, writhe = [], [], []
    for ci, cyc in enumerate(comps):
        laps = topo.unrolled.period_laps[ci]
        members = ",".join(str(topo.strand_labels[s]) for s in cyc)
        labels.append(f"{{{members}}}\n{2 * laps}π")
        colors.append(component_color(ci))
        writhe.append(float(topo.linking[ci, ci]))

    edges = [(a, b, float(topo.linking[a, b])) for a, b in combinations(range(n), 2)]
    return LinkGraphGeometry(node_xy=node_xy, node_labels=labels,
                             node_colors=colors, self_writhe=writhe, edges=edges)


def plot_link_graph_panel(
    topo: BraidTopology,
    *,
    ax=None,
    fontsize: int = 7,
):
    """Draw panel C into ``ax``: components as rings, edges labelled with lk."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    g = link_graph_geometry(topo)
    if ax is None:
        _, ax = plt.subplots(figsize=(2.6, 2.6))

    for a, b, lk in g.edges:
        x = [g.node_xy[a, 0], g.node_xy[b, 0]]
        y = [g.node_xy[a, 1], g.node_xy[b, 1]]
        linked = abs(lk) > 0.25
        ax.plot(x, y,
                ls="-" if linked else (0, (1, 3)),
                lw=(0.8 + 0.9 * min(abs(lk), 3.0)) if linked else 0.7,
                color="#333333" if linked else "0.7", zorder=1)
        ax.annotate(f"{lk:+.0f}" if linked else "0",
                    (float(np.mean(x)), float(np.mean(y))),
                    ha="center", va="center", fontsize=fontsize,
                    color="#111111" if linked else "0.55",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white",
                          "ec": "none", "alpha": 0.9}, zorder=2)

    # A lone component needs a bigger ring (nothing else to scale against);
    # several components sit on the unit circle and need a smaller one.
    radius = 0.62 if len(g.node_labels) == 1 else 0.40
    for xy, lbl, col, w in zip(g.node_xy, g.node_labels, g.node_colors,
                               g.self_writhe):
        ax.add_patch(Circle(xy, radius, facecolor="white", edgecolor=col,
                            lw=2.4, zorder=3))
        text = lbl if abs(w) < 0.25 else f"{lbl}\nw={w:+.0f}"
        ax.annotate(text, xy, ha="center", va="center", fontsize=fontsize - 1,
                    color="#111111", zorder=4)

    lim = (radius + 0.15) if len(g.node_labels) == 1 else (1.0 + radius + 0.15)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    # Caption below rather than a title: the closure sentence is long and would
    # collide with the panel letter.
    ax.text(0.5, -0.04, topo.closure_label.split("  [")[0],
            transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize - 1, color="#333333", wrap=True)
    return ax


# ---------------------------------------------------------------------------
# Panel D — lambda phase diagram
# ---------------------------------------------------------------------------

@dataclass
class PhaseStripGeometry:
    lambda_values: np.ndarray
    edges: np.ndarray                   # (n_lambda + 1,) cell boundaries
    labels: list[str]
    colors: list[str]
    nu: np.ndarray
    transition_lambdas: list[float]
    legend: list[tuple[str, str]]       # (label, colour), unique, in λ order


def phase_strip_geometry(scan: BraidScan) -> PhaseStripGeometry:
    """Class-vs-λ strip with cell boundaries midway between sampled λ."""
    lam = np.asarray(scan.lambda_values, dtype=float)
    if len(lam) > 1:
        mid = 0.5 * (lam[1:] + lam[:-1])
        edges = np.concatenate([[lam[0] - (mid[0] - lam[0])], mid,
                                [lam[-1] + (lam[-1] - mid[-1])]])
    else:
        edges = np.array([lam[0] - 0.5, lam[0] + 0.5])

    labels = scan.class_labels
    unclassified = []
    for l in labels:
        if l.replace(" (mirror)", "") not in _CLASS_COLORS and l not in unclassified:
            unclassified.append(l)
    colors = [class_color(l, unclassified_order=unclassified) for l in labels]

    seen: list[tuple[str, str]] = []
    for l, c in zip(labels, colors):
        if (l, c) not in seen:
            seen.append((l, c))

    return PhaseStripGeometry(
        lambda_values=lam, edges=edges, labels=labels, colors=colors,
        nu=scan.nu_total, transition_lambdas=list(scan.transition_lambdas),
        legend=seen,
    )


def plot_class_strip_panel(
    scan: BraidScan,
    *,
    ax=None,
    ax_nu=None,
    lambda_cursor: float | None = None,
    fontsize: int = 7,
):
    """Draw panel D: the class strip, and ν(λ) beneath it if ``ax_nu`` is given."""
    import matplotlib.pyplot as plt

    from mpm_analysis.plotting._style import LAMBDA_OBS_COLOR, LINE_WIDTH

    g = phase_strip_geometry(scan)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 1.2))

    for i, color in enumerate(g.colors):
        ax.axvspan(g.edges[i], g.edges[i + 1], color=color, lw=0, alpha=0.92)

    for lm in g.transition_lambdas:
        for a in (ax, ax_nu):
            if a is not None:
                a.axvline(lm, color=LAMBDA_OBS_COLOR, lw=1.0, ls=(0, (3, 2)),
                          zorder=5)
    if lambda_cursor is not None:
        for a in (ax, ax_nu):
            if a is not None:
                a.axvline(lambda_cursor, color="black", lw=1.2, zorder=6)

    ax.set_yticks([])
    ax.set_xlim(g.edges[0], g.edges[-1])
    ax.set_ylabel("class", fontsize=fontsize + 1)
    handles = [plt.Line2D([0], [0], marker="s", linestyle="None", markersize=5,
                          markerfacecolor=c, markeredgecolor="none", label=l)
               for l, c in g.legend]
    ax.legend(handles=handles, fontsize=fontsize - 1, ncol=min(len(handles), 5),
              loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False,
              handlelength=1.0, columnspacing=1.0)

    if ax_nu is not None:
        ax_nu.plot(g.lambda_values, g.nu, "o-", color="#A13D97",
                   ms=2.5, lw=LINE_WIDTH)
        ax_nu.axhline(0, color="0.75", lw=0.6)
        ax_nu.set_xlim(g.edges[0], g.edges[-1])
        ax_nu.set_ylabel(r"$\nu$", fontsize=fontsize + 1)
        ax_nu.set_xlabel(r"$\lambda$", fontsize=fontsize + 1)
    else:
        ax.set_xlabel(r"$\lambda$", fontsize=fontsize + 1)
    return ax
