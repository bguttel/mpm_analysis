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
"""
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline
from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from mpm_analysis.pipeline.click_correlation_pipeline import ClickCorrelationPipeline

__all__ = [
    "MPMPipeline",
    "CriticalPointPipeline",
    "BenchmarkPipeline",
    "ClickCorrelationPipeline",
]
