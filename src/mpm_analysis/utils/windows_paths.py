"""
windows_paths.py
----------------
Single canonical implementation of Windows long-path handling and figure saving.

This replaces the ``_win_longpath`` / ``_win_long_path`` helper that was
copy-pasted across four files in the original codebase.
"""
from __future__ import annotations

import os
from pathlib import Path


def win_long_path(p: str | Path) -> str:
    """Return a path string with the ``\\\\?\\`` prefix on Windows.

    On Windows, paths longer than ~260 characters fail unless prefixed with
    ``\\\\?\\`` (extended-length path prefix).  This function is a no-op on
    non-Windows platforms.

    Parameters
    ----------
    p:
        A ``Path`` object or path string.

    Returns
    -------
    str
        Path string safe for use with ``open()``, ``savefig()``, etc.
    """
    s = str(Path(p).resolve())
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        # UNC path  \\server\share\...
        return "\\\\?\\UNC\\" + s.lstrip("\\")
    return "\\\\?\\" + s


def save_figure_pdf_svg(
    fig,
    stem: str,
    output_dir: str | Path,
    *,
    dpi: int = 600,
    png: bool = False,
) -> tuple[Path, Path]:
    """Save a matplotlib figure as PDF and SVG (optionally also PNG).

    Parameters
    ----------
    fig:
        ``matplotlib.figure.Figure`` to save.
    stem:
        Base filename without extension.
    output_dir:
        Directory to write files into (created if it does not exist).
    dpi:
        Resolution for rasterised elements in the PDF, and of the PNG.
    png:
        Also write ``<stem>.png`` at ``dpi`` -- for paper figures, where the
        vector pair is the deliverable and the PNG is the preview.

    Returns
    -------
    (pdf_path, svg_path)
        Absolute ``Path`` objects for the saved vector files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdf_path = out / f"{stem}.pdf"
    svg_path = out / f"{stem}.svg"

    fig.savefig(win_long_path(pdf_path), format="pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(win_long_path(svg_path), format="svg", bbox_inches="tight")
    msg = f"Figure saved:\n  PDF: {pdf_path}\n  SVG: {svg_path}"
    if png:
        png_path = out / f"{stem}.png"
        fig.savefig(win_long_path(png_path), format="png", dpi=dpi,
                    bbox_inches="tight")
        msg += f"\n  PNG: {png_path}"
    print(msg)
    return pdf_path, svg_path
