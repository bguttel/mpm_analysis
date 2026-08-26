"""
braid_topology_main.py
----------------------
Two-column figure characterising the braid topology of the Lindbladian
eigenvalues at one λ, in the language of Pavlov/Gefen/Shnirman,
Phys. Rev. B 111, 104301 (2025).

Panel A  braid diagram over the full period (0 → L·2π), under-strands broken at
         each crossing, generators labelled — so 4π/6π components are shown
         closing and double braiding is countable.
Panel B  winding accumulator: the braid index read off the endpoint height.
Panel C  the closed braid as a link, components and linking numbers.
Panel D  class vs λ across the whole scan, exceptional points marked, with ν(λ).

Panels A–C describe the selected λ (marked by the cursor in panel D); panel D
spans the scan.
"""
from __future__ import annotations

from pathlib import Path


def make_braid_topology_figure(
    scan,
    lam,
    *,
    save_dir: Path | None = None,
    figure_stem: str = "braid_topology",
    fontsize: int = 7,
):
    """Build the four-panel braid topology figure for one λ of a scan.

    Parameters
    ----------
    scan : mpm_analysis.analysis.braid_topology.BraidScan
        From ``analyze_braid_scan``.
    lam : float
        Which λ panels A–C describe; the nearest sampled λ is used.
    save_dir :
        When given, the figure is written as both PDF and SVG via
        ``save_figure_pdf_svg``.
    figure_stem :
        Filename stem used for that save.

    Returns
    -------
    plt.Figure
    """
    import matplotlib.pyplot as plt

    from mpm_analysis.plotting._style import (
        LABELS_FONT_SIZE,
        TWO_COLUMN_WIDTH,
        apply_paper_style,
        make_ax_lines_wider,
        mm_to_inch,
    )
    from mpm_analysis.plotting.braid_panels import (
        plot_braid_diagram_panel,
        plot_class_strip_panel,
        plot_link_graph_panel,
        plot_winding_accumulator_panel,
    )

    apply_paper_style()
    topo = scan.at(lam)

    fig = plt.figure(figsize=(TWO_COLUMN_WIDTH, TWO_COLUMN_WIDTH * 0.82),
                     layout="constrained")
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1.15, 1.0, 0.34, 0.62], hspace=0.05,
    )
    ax_braid = fig.add_subplot(gs[0, :])
    ax_wind = fig.add_subplot(gs[1, :2])
    ax_link = fig.add_subplot(gs[1, 2])
    ax_strip = fig.add_subplot(gs[2, :])
    ax_nu = fig.add_subplot(gs[3, :], sharex=ax_strip)

    plot_braid_diagram_panel(topo, ax=ax_braid, fontsize=fontsize)
    plot_winding_accumulator_panel(topo, ax=ax_wind, fontsize=fontsize)
    plot_link_graph_panel(topo, ax=ax_link, fontsize=fontsize)
    plot_class_strip_panel(scan, ax=ax_strip, ax_nu=ax_nu,
                           lambda_cursor=topo.lambda_val, fontsize=fontsize)
    ax_strip.tick_params(labelbottom=False)

    for ax in (ax_braid, ax_wind, ax_nu):
        make_ax_lines_wider(ax, labelsize=fontsize)

    for ax, letter in ((ax_braid, "A"), (ax_wind, "B"),
                       (ax_link, "C"), (ax_strip, "D")):
        _add_panel_letter(ax, letter, mm_to_inch, LABELS_FONT_SIZE)

    q = topo.quality
    fig.suptitle(
        f"Braid topology at λ = {topo.lambda_val:.3f}   |   {topo.class_label}"
        f"   |   {q.summary()}",
        fontsize=fontsize + 1,
    )

    if save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
        save_figure_pdf_svg(fig, figure_stem, save_dir)
    return fig


def make_braid_topology_figure_from_result(
    result,
    lam=None,
    *,
    save_dir: Path | None = None,
    figure_stem: str = "braid_topology",
    deriv_weight: float = 0.1,
):
    """Convenience wrapper: run the scan on a ``KSpacePolesResult`` and plot it.

    ``lam`` defaults to ``scan.most_interesting_lambda()`` — the cleanest slice
    with the most exotic periodicity.
    """
    from mpm_analysis.analysis.braid_topology import analyze_braid_scan

    scan = analyze_braid_scan(result, deriv_weight=deriv_weight)
    if lam is None:
        lam = scan.most_interesting_lambda()
    return make_braid_topology_figure(scan, lam, save_dir=save_dir,
                                      figure_stem=figure_stem)


# ---- Helpers ----

def _add_panel_letter(ax, letter: str, mm_to_inch, fontsize: int) -> None:
    """Bold panel letter, offset -4 mm / +4 mm from the axes' top-left corner."""
    from matplotlib.transforms import ScaledTranslation

    ax.text(
        0.06, 1.0, letter,
        transform=(ax.transAxes
                   + ScaledTranslation(-mm_to_inch(4), mm_to_inch(4),
                                       ax.figure.dpi_scale_trans)),
        fontsize=fontsize, va="bottom", fontweight="bold", usetex=False,
    )
