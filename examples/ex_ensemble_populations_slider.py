"""
ex_ensemble_populations_slider.py
---------------------------------
Raw 4x4 ensemble-average generator output: the dark excited-state population
P_1 vs time at three lambda linecuts, compared between two bath return times.

Left panel  (A)  tau_B = 5 us    (experimental value)
Right panel (B)  tau_B = 0.01 us (fast-bath limit)

Styled like panel A of the third-transition main-text figure
(:func:`mpm_analysis.plotting.paper_figures.third_transition_main.make_third_transition_figure`):
same lambda linecuts, colours, fonts and axis labels, laid out side by side.
All physical parameters are the experimental ones (T_D, T1, Omega_S).

Run the file (or call ``main()``); edit the settings block below to change
anything.  ``SAVE_PAPER = True`` writes PDF + SVG + PNG into the paper
figures folder (``mpm_analysis/paper_figures``) with the run.py naming
convention (tag + data mode + UTC stamp); ``SAVE_DIR`` writes the same
trio anywhere else.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mpm_analysis.plotting._style import (
    LABELS_FONT_SIZE,
    LINE_WIDTH,
    LINECUT_COLORS,
    ONE_COLUMN_WIDTH,
    SURVIVAL_PROBABILITY_LABEL,
    TWO_COLUMN_WIDTH,
    add_panel_letter,
    apply_paper_style,
    make_ax_lines_wider,
)
from mpm_analysis.simulation.ensemble_average import EnsembleAverageDynamics

# ===========================================================================
# SETTINGS
# ===========================================================================
TAU_B    = (5.0, 0.01)          # the two bath return times to compare (us): panel A, panel B
LAMBDAS  = (0.4, 1.1, 1.5)      # lambda linecuts drawn in each panel
T1       = 93.0                 # relaxation time (us)
T_D      = 26.0                 # pure dephasing time (us)
OMEGA_S  = 0.628                # sensor Rabi frequency (rad/us)
DURATION = 20.0                 # simulation length (us)
T_INT    = 0.32                 # time step (us)
SHOW_TITLES = True              # tau_B title above each panel

# Paper proportions: 153 x 57 mm (85 % of a two-column width), so each panel
# is ~4:3 and the figure sits centred in a two-column slot.
FIG_WIDTH  = TWO_COLUMN_WIDTH * 0.85
FIG_HEIGHT = ONE_COLUMN_WIDTH * 0.65

SAVE_PAPER        = True        # PDF + SVG + PNG into PAPER_FIGURES_DIR (run.py convention)
PAPER_FIGURES_DIR = Path(__file__).resolve().parent.parent / "paper_figures"
PAPER_FIGURES_DPI = 600
FIGURE_TAG        = "ensemble_populations_tau_B"   # stem prefix; data mode "generator"

SAVE_DIR = None                 # any other directory for the same trio; None → skip
STEM     = FIGURE_TAG           # plain stem used with SAVE_DIR


def main(tau_B=TAU_B, lambdas=LAMBDAS, *, T1=T1, T_D=T_D, omega_S=OMEGA_S,
         duration=DURATION, T_int=T_INT, show_titles=SHOW_TITLES,
         fig_width=FIG_WIDTH, fig_height=FIG_HEIGHT,
         save_paper=SAVE_PAPER, save_dir=SAVE_DIR, stem=STEM, show=True):
    """Build (and optionally save / show) the two-panel figure; returns it."""
    lam_vals = np.array(lambdas, dtype=float)

    apply_paper_style()

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_width, fig_height),
        sharey=True, layout="constrained",
    )

    for ax, tb, letter in zip(axes, tau_B, "ab"):
        sim = EnsembleAverageDynamics(
            T_D=T_D, T1=T1, T_B=tb, omega_S=omega_S,
            simulation_duration=duration, T_int=T_int,
        )
        pop = sim.populations(lambda_vals=lam_vals)
        for i, lam in enumerate(lam_vals):
            ax.plot(pop["t"], pop["d"][i], lw=LINE_WIDTH,
                    color=LINECUT_COLORS[i % len(LINECUT_COLORS)],
                    label=rf"$\lambda$={lam:.1f}", zorder=10)
        if show_titles:
            ax.set_title(rf"$\tau_B = {tb:g}\ \mu$s", fontsize=LABELS_FONT_SIZE)
        ax.set_xlabel(r"Time ($\mu$s)", fontsize=LABELS_FONT_SIZE, labelpad=1.0)
        ax.grid(False)
        make_ax_lines_wider(ax)
        add_panel_letter(ax, letter, fontsize=LABELS_FONT_SIZE)

    axes[0].set_ylabel(SURVIVAL_PROBABILITY_LABEL, fontsize=LABELS_FONT_SIZE)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=LABELS_FONT_SIZE)

    if save_paper or save_dir is not None:
        from mpm_analysis.utils.windows_paths import save_figure_pdf_svg
    if save_paper:
        # Same naming as run.py's paper figures: tag, data mode, UTC stamp.
        from mpm_analysis.io.file_naming import build_figure_filename
        tag = FIGURE_TAG + "_" + "_".join(f"{tb:g}".replace(".", "p") for tb in tau_B)
        paper_stem = build_figure_filename(
            figure_tag=tag, data_mode="generator", extension="").rstrip(".")
        save_figure_pdf_svg(fig, paper_stem, PAPER_FIGURES_DIR,
                            dpi=PAPER_FIGURES_DPI, png=True)
    if save_dir is not None:
        save_figure_pdf_svg(fig, stem, Path(save_dir), dpi=PAPER_FIGURES_DPI, png=True)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    main()
