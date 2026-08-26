"""
kspace — k-space parity analysis module
========================================

Extends the existing MPM framework with a second (k / quasi-momentum)
dimension.  Each parity JSON now contains a full 2-D complex signal
(n_k × n_time); this module loads it, runs the Matrix Pencil Method
on every (lambda, k) cell, and provides interactive and static plots.

Quick start
-----------
::

    from mpm_analysis.kspace import (
        ParityKSpaceLoader,
        KSpaceAnalysis,
        PoleOrderConfig,
        plot_poles_vs_k_slider,
        plot_poles_vs_k_grid,
        plot_eigenvalues_vs_lambda,
        plot_raw_traces,
    )

    records = ParityKSpaceLoader().load(folder)
    result  = KSpaceAnalysis().run(records)

    plot_poles_vs_k_slider(result)
    plot_poles_vs_k_grid(result, n_cols=3)
    plot_eigenvalues_vs_lambda(result, k_idx_or_val=50)   # k ≈ π
    plot_raw_traces(records, lambda_val=1.0, component="real")

    result.save("kspace_result.npz")
    result2 = KSpacePolesResult.load("kspace_result.npz")
"""
from mpm_analysis.kspace.data_types import KSpacePolesResult, KSpaceRecord
from mpm_analysis.kspace.loader import (
    KSpaceJSONLoader,
    ParityKSpaceLoader,        # back-compat alias of KSpaceJSONLoader
    CompressedClickKSpaceLoader,
    SimClickKSpaceLoader,
    KSPACE_LOADERS,
    get_kspace_loader,
    compute_pkt_from_pnt,
)
from mpm_analysis.kspace.analysis import KSpaceAnalysis
from mpm_analysis.kspace.pole_order import PoleOrderConfig, infer_pole_order
from mpm_analysis.kspace.plotting import (
    plot_eigenvalues_vs_lambda,
    plot_raw_traces,
    plot_poles_vs_k_slider,
    plot_poles_vs_k_grid,
    plot_poles_vs_k_3d_grid,
    export_poles_vs_k_3d_html,
    plot_eigenvalues_parametric_grid,
    plot_eigenvalues_parametric_3d,
    export_eigenvalues_parametric_3d_html,
    plot_svd_vs_k_slider,
    plot_svd_spectrum_grid,
    plot_svd_vs_lambda_grid,
    export_svd_vs_lambda_slider_html,
    plot_raw_kspace_traces,
    export_raw_kspace_traces_html,
    export_eigenvalues_vs_lambda_kslider_html,
    export_pole_pair_topology_html,
    plot_braiding_vs_lambda,  # noqa: F401  (re-exported for callers)
    save_kspace_figures,
    sort_poles_along_k,
    track_permutation,
    gap_winding,
)

__all__ = [
    "KSpaceRecord",
    "KSpacePolesResult",
    "KSpaceJSONLoader",
    "ParityKSpaceLoader",
    "CompressedClickKSpaceLoader",
    "SimClickKSpaceLoader",
    "KSPACE_LOADERS",
    "get_kspace_loader",
    "compute_pkt_from_pnt",
    "KSpaceAnalysis",
    "PoleOrderConfig",
    "infer_pole_order",
    "plot_eigenvalues_vs_lambda",
    "plot_raw_traces",
    "plot_poles_vs_k_slider",
    "plot_poles_vs_k_grid",
    "plot_poles_vs_k_3d_grid",
    "export_poles_vs_k_3d_html",
    "plot_eigenvalues_parametric_grid",
    "plot_eigenvalues_parametric_3d",
    "export_eigenvalues_parametric_3d_html",
    "plot_svd_vs_k_slider",
    "plot_svd_vs_lambda_grid",
    "export_svd_vs_lambda_slider_html",
    "export_pole_pair_topology_html",
    "export_braid_topology_html",
    "plot_braiding_vs_lambda",
    "save_kspace_figures",
    "sort_poles_along_k",
    "track_permutation",
    "gap_winding",
]


def __getattr__(name):
    """Lazily expose the braid-topology HTML export.

    ``kspace.braid_html`` pulls in ``plotting.braid_panels`` and the analysis
    core, so importing it eagerly here would make ``import mpm_analysis.kspace``
    noticeably heavier for the many callers that never touch braiding.
    """
    if name == "export_braid_topology_html":
        from mpm_analysis.kspace.braid_html import export_braid_topology_html
        return export_braid_topology_html
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
