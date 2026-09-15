"""
plotting
--------
Plotting utilities for mpm_analysis.

Sub-modules
~~~~~~~~~~~
_style          — shared style constants and helpers (no ZenoPlotting dep)
exploratory     — quick sanity-check plots (raw data, grids, spectra)
                  includes ``plot_svd_singular_values`` for model-order
                  calibration via the Hankel SVD spectrum
survival_curves — panel-A style survival probability plots
pole_spectrum   — panel-B style decay rate + frequency plots
dead_time_plots — dead-time compression sweep: traces + fits, poles vs c,
                  discriminant / transitions vs c
paper_figures/  — publication-quality paper figures

SVD calibration workflow
~~~~~~~~~~~~~~~~~~~~~~~~
Pass ``plot_svd=True`` (and optionally ``n_sv=<int>``) to ``MPMStep`` or
``MPMRRHAStep`` to display the Hankel singular-value spectrum vs λ before
committing to a model order.  For RRHA the singular values shown are from the
first (pre-reduction) SVD, which gives the clearest view of the signal vs noise
subspaces::

    steps = [MPMStep(order=3, plot_svd=True, n_sv=12), BootstrapStep()]
    steps = [MPMRRHAStep(order=3, plot_svd=True, n_sv=12), BootstrapStep()]
"""
