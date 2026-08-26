"""
braid_html.py
-------------
Standalone, self-contained HTML version of the four braid-topology panels, with
a λ slider — the interactive companion to
``plotting.paper_figures.braid_topology_main``.

Panels A-C are rebuilt for every λ and switched by the slider; panel D shows the
whole scan with a λ cursor that follows it, so you can watch the class change
and see the braid rearrange at the same time.

All panel content comes from the geometry builders in
``mpm_analysis.plotting.braid_panels``, the same ones the matplotlib figure
uses, so the two backends cannot drift apart.

plotly is imported lazily inside the export function: it is *not* a declared
dependency of this package (pyproject lists only numpy, scipy and matplotlib),
matching how the rest of ``kspace.plotting`` treats it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mpm_analysis.plotting.braid_panels import (
    braid_diagram_geometry,
    class_color,
    component_color,
    link_graph_geometry,
    phase_strip_geometry,
    strand_color,
    winding_curves_geometry,
)

_TWO_PI = 2.0 * np.pi


def _period_ticks(k0: float, n_laps: int) -> tuple[list[float], list[str]]:
    ticks = [k0 + _TWO_PI * i for i in range(n_laps + 1)]
    labels = ["0"] + [f"{2 * (i + 1)}π" for i in range(n_laps)]
    return ticks, labels


def export_braid_topology_html(
    result,
    path: str,
    *,
    lambda_values=None,
    deriv_weight: float = 0.1,
    max_crossing_labels: int = 10,
    include_plotlyjs: str | bool = "cdn",
    scan=None,
) -> str:
    """Write the four braid panels with a λ slider as one standalone HTML file.

    Parameters
    ----------
    result :
        ``KSpacePolesResult``.  Ignored when ``scan`` is supplied.
    path :
        Destination ``.html``; parent directories are created.
    lambda_values :
        Which λ become slider steps — same conventions as elsewhere in
        ``kspace.plotting`` (``None`` = all).
    deriv_weight :
        Continuity-sort tiebreaker passed through to the track matching.
    max_crossing_labels :
        Above this many crossings the σ labels are suppressed for that λ (noisy
        data re-crosses without changing topology); the breaks are still drawn.
    scan :
        Pre-computed ``BraidScan``, to avoid re-running the analysis.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from mpm_analysis.analysis.braid_topology import analyze_braid_scan
    from mpm_analysis.kspace.plotting import _resolve_lambda_indices
    from mpm_analysis.utils.windows_paths import win_long_path

    if scan is None:
        lam_idx = (None if lambda_values is None
                   else _resolve_lambda_indices(result, lambda_values,
                                                default_n_panels=result.n_lambda))
        scan = analyze_braid_scan(result, deriv_weight=deriv_weight,
                                  lambda_indices=lam_idx)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.68, 0.32],
        row_heights=[0.52, 0.48],
        subplot_titles=("A — braid diagram over the full period",
                        "C — link (components and linking numbers)",
                        "B — winding accumulator (index = endpoint height)",
                        "D — class vs λ"),
        horizontal_spacing=0.07,
        vertical_spacing=0.13,
    )

    # ---- per-λ traces (panels A, B, C) -------------------------------------
    groups: list[list[int]] = []
    for pos, topo in enumerate(scan.topologies):
        visible = pos == 0
        idx: list[int] = []

        # --- A: braid diagram ------------------------------------------------
        gA = braid_diagram_geometry(topo)
        for x in gA.seams:
            idx.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[x, x], y=list(gA.y_lim), mode="lines",
                line={"color": "rgba(0,0,0,0.35)", "width": 1, "dash": "dash"},
                hoverinfo="skip", showlegend=False, visible=visible,
            ), row=1, col=1)

        for s in range(gA.height.shape[1]):
            ci = int(gA.strand_component[s])
            idx.append(len(fig.data))
            fig.add_trace(go.Scatter(                      # component halo
                x=gA.k, y=gA.height[:, s], mode="lines",
                line={"color": component_color(ci), "width": 7},
                opacity=0.40, hoverinfo="skip", showlegend=False, visible=visible,
            ), row=1, col=1)
            idx.append(len(fig.data))
            fig.add_trace(go.Scatter(                      # strand core
                x=gA.k, y=gA.height[:, s], mode="lines",
                line={"color": strand_color(topo.strand_labels[s]), "width": 2},
                name=f"strand {topo.strand_labels[s]}",
                hovertemplate=("k=%{x:.3f}<br>Re λ=%{y:.4f}"
                               f"<extra>strand {topo.strand_labels[s]}, "
                               f"component {ci}</extra>"),
                showlegend=False, visible=visible,
            ), row=1, col=1)

        idx.append(len(fig.data))
        show_labels = len(gA.crossing_k) <= max_crossing_labels
        fig.add_trace(go.Scatter(
            x=gA.crossing_k, y=gA.crossing_height,
            mode="markers+text" if show_labels else "markers",
            marker={"color": "#111111", "size": 5, "symbol": "x-thin",
                    "line": {"color": "#111111", "width": 1.5}},
            text=gA.crossing_label if show_labels else None,
            textposition="top center", textfont={"size": 11},
            customdata=gA.crossing_label,
            hovertemplate="k=%{x:.3f}<br>%{customdata}<extra>crossing</extra>",
            showlegend=False, visible=visible,
        ), row=1, col=1)

        # --- B: winding accumulator ------------------------------------------
        gB = winding_curves_geometry(topo)
        for j, (cum, lbl, total) in enumerate(zip(gB.curves, gB.labels, gB.totals)):
            idx.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=gB.k, y=cum, mode="lines",
                line={"color": component_color(j), "width": 2.2},
                name=f"{lbl}: {total:+.2f}",
                hovertemplate=("k=%{x:.3f}<br>turns=%{y:+.3f}"
                               f"<extra>pair {lbl}</extra>"),
                showlegend=True, legendgroup=f"pair{j}", visible=visible,
            ), row=2, col=1)

        # --- C: link cartoon ---------------------------------------------------
        gC = link_graph_geometry(topo)
        for a, b, lk in gC.edges:
            linked = abs(lk) > 0.25
            idx.append(len(fig.data))
            fig.add_trace(go.Scatter(
                x=[gC.node_xy[a, 0], gC.node_xy[b, 0]],
                y=[gC.node_xy[a, 1], gC.node_xy[b, 1]],
                mode="lines+text",
                line={"color": "#333333" if linked else "rgba(0,0,0,0.25)",
                      "width": 1 + 2 * min(abs(lk), 3.0) if linked else 1,
                      "dash": "solid" if linked else "dot"},
                text=["", f"lk={lk:+.0f}"], textposition="middle center",
                textfont={"size": 11},
                hoverinfo="skip", showlegend=False, visible=visible,
            ), row=1, col=2)

        idx.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=gC.node_xy[:, 0], y=gC.node_xy[:, 1],
            mode="markers+text",
            marker={"size": 62 if len(gC.node_labels) == 1 else 46,
                    "color": "white",
                    "line": {"color": gC.node_colors, "width": 4}},
            text=[l.replace("\n", "<br>") for l in gC.node_labels],
            textposition="middle center", textfont={"size": 10},
            customdata=gC.self_writhe,
            hovertemplate="%{text}<br>self-writhe=%{customdata:+.0f}<extra></extra>",
            showlegend=False, visible=visible,
        ), row=1, col=2)

        groups.append(idx)

    # ---- panel D: the whole scan, never toggled ----------------------------
    gD = phase_strip_geometry(scan)
    seen: set[str] = set()
    for i, (lbl, color) in enumerate(zip(gD.labels, gD.colors)):
        fig.add_trace(go.Scatter(
            x=[gD.edges[i], gD.edges[i + 1], gD.edges[i + 1], gD.edges[i]],
            y=[0, 0, 1, 1], fill="toself",
            fillcolor=color, line={"width": 0},
            mode="lines", name=lbl, legendgroup=lbl,
            showlegend=lbl not in seen,
            hovertemplate=f"λ=%{{x:.3f}}<extra>{lbl}</extra>",
        ), row=2, col=2)
        seen.add(lbl)

    fig.add_trace(go.Scatter(
        x=gD.lambda_values, y=0.5 + 0.42 * _normalize(gD.nu),
        mode="lines+markers",
        line={"color": "#111111", "width": 1.6},
        marker={"color": "#111111", "size": 4},
        customdata=gD.nu, name="ν",
        hovertemplate="λ=%{x:.3f}<br>ν=%{customdata:+.2f}<extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    for lm in gD.transition_lambdas:
        fig.add_trace(go.Scatter(
            x=[lm, lm], y=[0, 1], mode="lines",
            line={"color": "#d02500", "width": 1.2, "dash": "dash"},
            hovertemplate=f"EP near λ={lm:.3f}<extra></extra>",
            showlegend=False,
        ), row=2, col=2)

    # λ cursor on panel D — one per step, toggled with the panel-A/B/C group.
    cursor_idx: list[int] = []
    for pos, topo in enumerate(scan.topologies):
        cursor_idx.append(len(fig.data))
        fig.add_trace(go.Scatter(
            x=[topo.lambda_val, topo.lambda_val], y=[0, 1], mode="lines",
            line={"color": "black", "width": 2.5},
            hoverinfo="skip", showlegend=False, visible=(pos == 0),
        ), row=2, col=2)
    for pos, ci in enumerate(cursor_idx):
        groups[pos].append(ci)

    # ---- slider --------------------------------------------------------------
    # Every step addresses the same list of toggleable traces (the union of all
    # per-λ groups) and states which of them are visible.  Panel D's strip, ν
    # track and EP lines are deliberately outside this list, so they are never
    # toggled and stay on screen for every λ.
    toggleable = [i for g in groups for i in g]
    pos_in_toggleable = {t: i for i, t in enumerate(toggleable)}

    steps = []
    for pos, topo in enumerate(scan.topologies):
        vis = [False] * len(toggleable)
        for t in groups[pos]:
            vis[pos_in_toggleable[t]] = True

        gA = braid_diagram_geometry(topo)
        ticks, labels = _period_ticks(float(gA.k[0]), topo.unrolled.n_laps)
        k_range = [float(gA.k[0]), float(gA.k[-1])]
        steps.append(dict(
            method="update",
            label=f"{topo.lambda_val:.3f}",
            args=[
                {"visible": vis},
                {
                    # panel A is (row 1, col 1) -> xaxis/yaxis;
                    # panel B is (row 2, col 1) -> xaxis3/yaxis3.
                    # The k span changes with the number of laps, so both axes
                    # must be relaid out per λ or a 6π braid would be clipped
                    # to the 2π window of the previous step.
                    "xaxis.range": k_range,
                    "xaxis.tickvals": ticks,
                    "xaxis.ticktext": labels,
                    "yaxis.range": list(gA.y_lim),
                    "xaxis3.range": k_range,
                    "xaxis3.tickvals": ticks,
                    "xaxis3.ticktext": labels,
                    "title.text": _title(topo),
                },
                toggleable,
            ],
        ))

    first = scan.topologies[0]
    gA0 = braid_diagram_geometry(first)
    ticks0, labels0 = _period_ticks(float(gA0.k[0]), first.unrolled.n_laps)

    fig.update_xaxes(title_text="counting field  k", tickvals=ticks0,
                     ticktext=labels0, range=[float(gA0.k[0]), float(gA0.k[-1])],
                     row=1, col=1)
    fig.update_yaxes(title_text="Re λ (projected)", range=list(gA0.y_lim),
                     row=1, col=1)
    fig.update_xaxes(title_text="counting field  k", tickvals=ticks0,
                     ticktext=labels0, range=[float(gA0.k[0]), float(gA0.k[-1])],
                     row=2, col=1)
    fig.update_yaxes(title_text="accumulated winding [loops]", row=2, col=1)
    fig.update_xaxes(visible=False, range=[-1.7, 1.7], row=1, col=2)
    fig.update_yaxes(visible=False, range=[-1.7, 1.7], scaleanchor="x2",
                     scaleratio=1, row=1, col=2)
    fig.update_xaxes(title_text="λ", row=2, col=2)
    fig.update_yaxes(visible=False, range=[0, 1], row=2, col=2)

    fig.update_layout(
        title={"text": _title(first), "x": 0.5, "xanchor": "center",
               "font": {"size": 14}},
        sliders=[dict(
            active=0, steps=steps, pad={"t": 46},
            currentvalue={"prefix": "λ = ", "font": {"size": 13}},
            font={"size": 9},
        )],
        height=880, width=1300,
        margin={"l": 70, "r": 30, "t": 110, "b": 110},
        template="plotly_white",
        legend={"font": {"size": 10}, "y": 0.42, "x": 1.01},
    )

    import os as _os
    out = Path(path)
    _os.makedirs(win_long_path(out.parent), exist_ok=True)
    fig.write_html(win_long_path(out), include_plotlyjs=include_plotlyjs,
                   full_html=True)
    return str(out.resolve())


def _normalize(v: np.ndarray) -> np.ndarray:
    """Map ν onto [-1, 1] so it can share panel D's unit-height axis."""
    v = np.asarray(v, dtype=float)
    scale = np.nanmax(np.abs(v)) if np.any(np.isfinite(v)) else 1.0
    return v / (scale if scale > 0 else 1.0)


def _title(topo) -> str:
    q = topo.quality
    warn = "" if q.trustworthy else "   ⚠ unreliable"
    return (f"Braid topology  |  λ = {topo.lambda_val:.3f}  |  {topo.class_label}"
            f"  |  period {topo.period_label}  |  word {topo.word_str}"
            f"  |  {q.summary()}{warn}")
