"""
paper_figures
-------------
Publication-quality paper figure functions.

Functions
~~~~~~~~~
make_third_transition_figure  — Panel A + Panel B main-text figure
plot_appendix_bootstrap       — Appendix bootstrap analysis figure
make_braid_topology_figure    — 4-panel braid topology figure (braid diagram,
                                winding accumulator, link, class-vs-λ)
"""
from mpm_analysis.plotting.paper_figures.third_transition_main import (
    make_third_transition_figure,
)
from mpm_analysis.plotting.paper_figures.appendix_bootstrap import (
    plot_appendix_bootstrap,
)
from mpm_analysis.plotting.paper_figures.braid_topology_main import (
    make_braid_topology_figure,
    make_braid_topology_figure_from_result,
)

__all__ = [
    "make_third_transition_figure",
    "plot_appendix_bootstrap",
    "make_braid_topology_figure",
    "make_braid_topology_figure_from_result",
]
