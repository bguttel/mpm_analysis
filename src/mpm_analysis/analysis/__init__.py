"""
analysis
--------
Stateless numerical routines for pole extraction and critical-point analysis.

Heavy imports (scipy, sympy) are deferred to function bodies so that
``import mpm_analysis`` stays fast.

Sub-modules
~~~~~~~~~~~
matrix_pencil   — Matrix Pencil Method (MPM) for pole extraction
bootstrap       — Bootstrap uncertainty quantification
click_correlations — Fixed-anchor two-time click statistics (joint /
                  exclusive), source-agnostic, MPM-ready via
                  ``to_observable_records``
critical_point  — Sqrt-fit for exceptional-point characterisation
braiding        — Thin wrapper over Theory/braid_tools_complete.py
braid_topology  — Paper-convention braid invariants: loop closure, unrolling to
                  the full 4π/6π period, Eqs. 9-13, canonical braid word and
                  Table I classification
steps/          — Pluggable AnalysisStep classes (Strategy pattern)
"""
