"""
io
--
All file I/O for mpm_analysis.

Sub-modules
~~~~~~~~~~~
analysis_json    — load / save ``ZenoAnalysisResult`` ↔ JSON
experimental/    — load experimental data folders → ``list[ObservableRecord]``
simulated/       — load simulation NPZ (generate_click_record_raw.py) → ``list[ClickRecordData]``
simulation       — load analytical / MC NPZ files → ``list[ObservableRecord]``
file_naming      — auto-generate and parse result filenames
"""
from mpm_analysis.io.analysis_json import (
    load_analysis_result,
    save_analysis_result,
    augment_json_with_sqrt_fit,
)
from mpm_analysis.io.file_naming import build_result_filename, save_result

__all__ = [
    "load_analysis_result",
    "save_analysis_result",
    "augment_json_with_sqrt_fit",
    "build_result_filename",
    "save_result",
]
