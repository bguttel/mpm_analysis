"""
file_naming.py
--------------
Auto-generate and parse result filenames so that saved files are always
self-describing.

Filename format
~~~~~~~~~~~~~~~
    <analysis_type>_<data_mode>_N<n_lambda>_lam<min>-<max>_<timestamp>.json

Examples
~~~~~~~~
    mpm_ens_avg_N50_lam0.40-1.60_20260410_143022.json
    mpm_post_selected_N30_lam0.40-1.55_20260412_091500.json
    critical_point_ens_avg_N50_lam0.40-1.60_20260410_145500.json
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Filename building
# ---------------------------------------------------------------------------

def build_result_filename(
    analysis_type: str,
    data_mode: str,
    lambda_range: tuple[float, float],
    n_lambda: int,
    *,
    scan_number: int | None = None,
    extension: str = "json",
) -> str:
    """Build a timestamped filename for an analysis result.

    Parameters
    ----------
    analysis_type:
        Short tag for the analysis, e.g. ``"mpm"``, ``"critical_point"``.
    data_mode:
        Data mode tag, e.g. ``"ens_avg"``, ``"post_selected"``,
        ``"analytical"``.
    lambda_range:
        ``(lambda_min, lambda_max)`` tuple.
    n_lambda:
        Number of lambda points.
    scan_number:
        Integer scan number to embed in the filename (e.g. 50 for
        ``scan_experiment_results_50``).  Omitted from the filename if ``None``.
    extension:
        File extension without leading dot. Default ``"json"``.

    Returns
    -------
    str
        Filename string, e.g.
        ``"mpm_ens_avg_scan50_N50_lam0.40-1.60_20260410_143022.json"``
        or (no scan number)
        ``"mpm_ens_avg_N50_lam0.40-1.60_20260410_143022.json"``.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    lam_min, lam_max = lambda_range
    scan_seg = f"_scan{scan_number}" if scan_number is not None else ""
    return (
        f"{analysis_type}_{data_mode}{scan_seg}_N{n_lambda}"
        f"_lam{lam_min:.2f}-{lam_max:.2f}_{ts}.{extension}"
    )


def build_figure_filename(
    figure_tag: str,
    data_mode: str,
    *,
    scan_number: int | None = None,
    extension: str = "pdf",
) -> str:
    """Build a timestamped filename for a saved figure.

    Parameters
    ----------
    figure_tag:
        Descriptive tag, e.g. ``"third_transition_main"``.
    data_mode:
        Data mode tag, e.g. ``"ens_avg"``.
    scan_number:
        If the figure is from a scan, the scan number.  Omitted if ``None``.
    extension:
        File extension. Default ``"pdf"``.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    scan_seg = f"_scan{scan_number}" if scan_number is not None else ""
    return f"{figure_tag}_{data_mode}{scan_seg}_{ts}.{extension}"


# ---------------------------------------------------------------------------
# Saving convenience
# ---------------------------------------------------------------------------

def save_result(
    result,
    output_dir: str | Path,
    stem: str | None = None,
    *,
    full: bool = False,
) -> Path:
    """Save a ``ZenoAnalysisResult`` to ``output_dir`` with an auto-generated name.

    Parameters
    ----------
    result:
        A ``ZenoAnalysisResult`` instance.
    output_dir:
        Directory to write into (created if absent).
    stem:
        If given, use ``<stem>.json`` instead of an auto-generated name.
    full:
        Passed to ``save_analysis_result`` — also serialise raw pole samples
        and the ``analysis_extras`` block.

    Returns
    -------
    Path
        Path to the saved file.
    """
    # Import here to avoid circular dependency at module level
    import os

    from mpm_analysis.io.analysis_json import save_analysis_result
    from mpm_analysis.utils.windows_paths import win_long_path

    output_dir = Path(output_dir)
    # win_long_path: deep lab folders exceed MAX_PATH on Windows.
    os.makedirs(win_long_path(output_dir), exist_ok=True)

    if stem is None:
        lam = result.lambda_values
        scan_number = result.metadata.get("scan_number")
        if scan_number is not None:
            scan_number = int(scan_number)

        data_mode = str(result.metadata.get("data_mode",
                                           result.metadata.get("source_tag", "unknown")))
        analysis_type = str(result.analysis_type or "mpm")
        # Keep backward compatibility with old metadata where analysis_type was
        # set equal to the data-mode tag, which duplicated filename segments.
        if analysis_type == data_mode:
            analysis_type = "mpm"

        stem = build_result_filename(
            analysis_type=analysis_type,
            data_mode=data_mode,
            lambda_range=(float(lam.min()), float(lam.max())),
            n_lambda=len(lam),
            scan_number=scan_number,
        )
        # stem already includes .json extension from build_result_filename
        out_path = output_dir / stem
    else:
        out_path = output_dir / (stem if stem.endswith(".json") else stem + ".json")

    save_analysis_result(result, out_path, full=full)
    return out_path


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(
    r"^(?P<analysis_type>[a-z_]+)_(?P<data_mode>[a-z_]+)"
    r"(?:_scan(?P<scan_number>\d+))?"   # optional _scanNN segment
    r"_N(?P<n_lambda>\d+)"
    r"_lam(?P<lam_min>[\d.]+)-(?P<lam_max>[\d.]+)"
    r"_(?P<timestamp>\d{8}_\d{6})"
    r"\.(?P<ext>[a-z]+)$"
)


def parse_result_filename(filename: str) -> dict | None:
    """Parse a filename generated by ``build_result_filename``.

    Returns a dict with keys:
        ``analysis_type``, ``data_mode`` (also under legacy key ``source_tag``), ``scan_number`` (int or None),
        ``n_lambda``, ``lam_min``, ``lam_max``, ``timestamp``, ``ext``

    Returns ``None`` if the filename does not match the expected format.
    """
    m = _FNAME_RE.match(Path(filename).name)
    if not m:
        return None
    sn = m.group("scan_number")
    return {
        "analysis_type": m.group("analysis_type"),
        "data_mode": m.group("data_mode"),
        "source_tag": m.group("data_mode"),   # legacy alias
        "scan_number": int(sn) if sn is not None else None,
        "n_lambda": int(m.group("n_lambda")),
        "lam_min": float(m.group("lam_min")),
        "lam_max": float(m.group("lam_max")),
        "timestamp": m.group("timestamp"),
        "ext": m.group("ext"),
    }
