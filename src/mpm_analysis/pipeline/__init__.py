"""
pipeline
--------
High-level orchestration classes that wire together io + analysis + saving.

Classes
~~~~~~~
MPMPipeline              — Load data → run steps → save ZenoAnalysisResult
CriticalPointPipeline    — Load result JSON → fit sqrt model → save back
BenchmarkPipeline        — Simulator → noise → analysis → bias/variance report
ClickCorrelationPipeline — Click records (compressed/json/simulated) →
                           fixed-anchor correlations → MPM poles → figures
DeadTimeSweepPipeline    — Detector dead-time compression of the time axis,
                           t' = t·[1 − c·p_click(t)], swept in c: direct
                           non-uniform LSQ fit (+ regrid/MPM fallback) →
                           poles and transitions vs c
"""
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from mpm_analysis.pipeline.click_correlation_pipeline import ClickCorrelationPipeline
from mpm_analysis.pipeline.dead_time_pipeline import DeadTimeSweepPipeline

__all__ = [
    "MPMPipeline",
    "CriticalPointPipeline",
    "BenchmarkPipeline",
    "ClickCorrelationPipeline",
    "DeadTimeSweepPipeline",
]
