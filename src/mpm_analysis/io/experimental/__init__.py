"""
io/experimental
---------------
Loaders for experimental data folders.

Usage
~~~~~
    from mpm_analysis.io.experimental import PostSelectedLoader, EnsembleAverageLoader

    records = PostSelectedLoader().load(folder, observables=["survival", "Z"])
    records = EnsembleAverageLoader().load(folder, observables=["survival"])
"""
from mpm_analysis.io.experimental.base import ExperimentalLoader
from mpm_analysis.io.experimental.survival_probability import (
    ParityLoader,
    PostSelectedLoader,
    EnsembleAverageLoader,
)
from mpm_analysis.io.experimental.click_record_npz import (
    ClickRecordNpzLoader,
    ClickRecordData,
)
from mpm_analysis.io.experimental.compressed_click_record import (
    CompressedClickBuilder,
    CompressedClickLoader,
    CompressedClickRecord,
    compress_run,
    fit_dead_time_exponential,
    recompress_min_dark,
    recompress_file_min_dark,
    recompress_folder_min_dark,
)

__all__ = [
    "ExperimentalLoader",
    "PostSelectedLoader",
    "EnsembleAverageLoader",
    "ParityLoader",
    "ClickRecordNpzLoader",
    "ClickRecordData",
    "CompressedClickBuilder",
    "CompressedClickLoader",
    "CompressedClickRecord",
    "compress_run",
    "fit_dead_time_exponential",
    "recompress_min_dark",
    "recompress_file_min_dark",
    "recompress_folder_min_dark",
]
