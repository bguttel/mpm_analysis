"""
run_kspace_smoke_check.py
-------------------------
Validation smoke test for the k-space compressed-click analysis path.

This is intentionally not a demo/synthetic test. It checks the active
``run_kspace_analysis.py`` configuration against the real compressed records,
then runs a tiny real-data MPM + plotting pass.
"""
from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

import run_kspace_analysis as cfg
from mpm_analysis.io.experimental.base import _khz_from_name
from mpm_analysis.io.experimental.compressed_click_record import (
    _compressed_latest_key,
)
from mpm_analysis.kspace import KSpaceAnalysis, PoleOrderConfig
from mpm_analysis.kspace.data_types import KSpaceRecord
from mpm_analysis.kspace.loader import compute_pkt_from_pnt, make_open_k_grid
from mpm_analysis.kspace.plotting import (
    export_eigenvalues_parametric_slider_html,
    plot_eigenvalues_parametric_grid,
)
from mpm_analysis.utils.windows_paths import win_long_path


EXPECTED_SCAN_NUMBER = 22
SMOKE_N_RATES = 2
SMOKE_K_GRID_N = 33
SMOKE_K_ENDPOINT_EPS = 1e-3
FORBIDDEN_DATA_WORDS = re.compile(r"\b(demo|synthetic|fake|mock)\b", re.IGNORECASE)


def _fail(message: str) -> None:
    raise AssertionError(message)


def _metadata_from_npz(z: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in z.files:
        return {}
    raw = z["metadata_json"]
    if not raw.size:
        return {}
    return json.loads(bytes(raw).decode())


def _latest_compressed_files(folder: Path) -> list[Path]:
    newest: dict[int, Path] = {}
    newest_key: dict[int, tuple[str, float]] = {}
    for fp in sorted(folder.glob("click_record_compressed__*.npz")):
        rate = _khz_from_name(fp)
        key = _compressed_latest_key(fp)
        if rate not in newest or key > newest_key[rate]:
            newest[rate] = fp
            newest_key[rate] = key
    return [newest[r] for r in sorted(newest)]


def _assert_config() -> None:
    print("[config] checking run_kspace_analysis.py")
    if cfg.SCAN_NUMBER != EXPECTED_SCAN_NUMBER:
        _fail(
            f"run_kspace_analysis.py is set to scan {cfg.SCAN_NUMBER}, "
            f"but this smoke test expects scan {EXPECTED_SCAN_NUMBER}."
        )
    if cfg.SOURCE != "compressed":
        _fail("Smoke check expects SOURCE='compressed'.")
    if cfg.K_GRID_N is None:
        _fail("K_GRID_N is None; dense recomputed k-grid is disabled.")
    if cfg.K_ENDPOINT_EPS <= 0:
        _fail("K_ENDPOINT_EPS must be > 0 to avoid exact k=0 and k=2pi.")
    if cfg.K_INDEX_RANGE is not None:
        _fail("K_INDEX_RANGE should be None for the full open dense grid.")
    if cfg.N_BOOTSTRAP != 1:
        _fail("N_BOOTSTRAP should be 1 for deterministic no-resampling validation.")
    if cfg.BOOTSTRAP_OVER_RUNS:
        _fail("BOOTSTRAP_OVER_RUNS should be False for this deterministic smoke check.")
    if not Path(cfg.FOLDER).is_dir():
        _fail(f"compressed analysis folder does not exist: {cfg.FOLDER}")
    if cfg.THEORY is not None:
        print("[warn] THEORY overlay is enabled in run_kspace_analysis.py.")
        print("       Data poles are still experimental, but some figures include a model overlay.")
    print(f"[config] OK: scan={cfg.SCAN_NUMBER}, folder={cfg.FOLDER}")


def _assert_files(files: list[Path]) -> None:
    print(f"[files] checking {len(files)} latest compressed NPZ files")
    if not files:
        _fail("No compressed NPZ files found.")
    if len(files) < 5:
        _fail(f"Only {len(files)} compressed files found; expected a scan, not a demo subset.")

    for fp in files:
        if FORBIDDEN_DATA_WORDS.search(fp.name):
            _fail(f"Forbidden demo/synthetic/fake/mock word in filename: {fp.name}")
        with np.load(win_long_path(fp), allow_pickle=False) as z:
            meta = _metadata_from_npz(z)
            source_text = " ".join(
                [fp.name, str(meta.get("source_compressed_file", ""))]
                + [str(x) for x in meta.get("source_files", [])]
            )
            if FORBIDDEN_DATA_WORDS.search(source_text):
                _fail(f"Forbidden demo/synthetic/fake/mock source in {fp.name}")
            scan_meta = meta.get("scan_number")
            if scan_meta is not None and int(scan_meta) != EXPECTED_SCAN_NUMBER:
                _fail(f"{fp.name} metadata scan_number={scan_meta}, expected {EXPECTED_SCAN_NUMBER}.")
            if int(z["n_runs"]) <= 0 or int(z["n_shots"]) <= 0:
                _fail(f"{fp.name} has nonpositive n_runs/n_shots.")
            if int(z["n_runs"]) < 2:
                _fail(f"{fp.name} has only {int(z['n_runs'])} run; this looks like a debug subset.")
    print("[files] OK: real compressed files, no demo/fake/mock markers")


def _assert_probability_and_dft(files: list[Path]) -> None:
    print("[data] checking P(n,t) normalization and stored Fourier transform")
    for fp in files:
        with np.load(win_long_path(fp), allow_pickle=False) as z:
            pnt = np.asarray(z["pnt"], dtype=float)
            shots = np.asarray(z["shots_per_time"], dtype=int)
            live = shots > 0
            row_err = float(np.max(np.abs(pnt[live].sum(axis=1) - 1.0)))
            if row_err > 5e-5:
                _fail(f"{fp.name}: P(n,t) rows are not normalized; max error={row_err:g}.")

    for fp in files[: min(3, len(files))]:
        with np.load(win_long_path(fp), allow_pickle=False) as z:
            pnt = np.asarray(z["pnt"], dtype=float)
            k_stored = np.asarray(z["fourier_k_array"], dtype=float)
            pkt, _ = compute_pkt_from_pnt(pnt, k_array=k_stored)
            stored = np.asarray(z["pkt_real"], dtype=float) + 1j * np.asarray(z["pkt_imag"], dtype=float)
            err = float(np.max(np.abs(pkt - stored)))
            if err > 5e-5:
                _fail(f"{fp.name}: stored P(k,t) does not match DFT from P(n,t); max error={err:g}.")
    print("[data] OK: P(n,t) normalized; stored P(k,t) is direct DFT from P(n,t)")


def _records_from_files(files: list[Path], k_array: np.ndarray) -> list[KSpaceRecord]:
    records: list[KSpaceRecord] = []
    for fp in files:
        with np.load(win_long_path(fp), allow_pickle=False) as z:
            pnt = np.asarray(z["pnt"], dtype=float)
            pkt, _ = compute_pkt_from_pnt(pnt, k_array=k_array)
            signal = pkt.T
            t_us = np.asarray(z["t_compressed_us"], dtype=float)
            shots = np.asarray(z["shots_per_time"], dtype=int)
            cut = int(np.count_nonzero(shots >= cfg.MIN_ALIVE_FRAC * shots[0]))
            t_us = t_us[:cut]
            signal = signal[:, :cut]
            meta = _metadata_from_npz(z)
            meta.update(
                source="compressed_click",
                k_grid_source="recomputed_from_pnt",
                k_grid_n=int(k_array.size),
                k_endpoint_eps=float(SMOKE_K_ENDPOINT_EPS),
                smoke_check=True,
            )
            records.append(KSpaceRecord(
                lambda_val=float(z["lambda_val"]),
                t=t_us,
                k_array=k_array,
                signal=signal,
                source_file=fp.name,
                metadata=meta,
            ))
    records.sort(key=lambda r: r.lambda_val)
    return records


def _assert_open_k_grid(k_array: np.ndarray) -> None:
    print("[k-grid] checking dense open k grid")
    if np.any(np.isclose(k_array, 0.0)):
        _fail("k grid contains exact 0.")
    if np.any(np.isclose(k_array, 2.0 * np.pi)):
        _fail("k grid contains exact 2pi.")
    if not np.all(np.diff(k_array) > 0):
        _fail("k grid is not strictly increasing.")
    if not np.allclose(np.diff(k_array), np.diff(k_array)[0]):
        _fail("k grid is not uniform.")
    print(f"[k-grid] OK: n={k_array.size}, first={k_array[0]:.6g}, last={k_array[-1]:.6g}")


def _assert_analysis_and_plotting(files: list[Path]) -> None:
    print("[analysis] running tiny real-data MPM smoke pass")
    k_array = make_open_k_grid(SMOKE_K_GRID_N, SMOKE_K_ENDPOINT_EPS)
    _assert_open_k_grid(k_array)
    records = _records_from_files(files[:SMOKE_N_RATES], k_array)

    for rec in records:
        if rec.metadata.get("k_grid_source") != "recomputed_from_pnt":
            _fail(f"{rec.source_file}: k grid source is not recomputed_from_pnt.")
        if rec.signal.shape[0] != SMOKE_K_GRID_N:
            _fail(f"{rec.source_file}: wrong k dimension {rec.signal.shape}.")
        if rec.signal.shape[1] < cfg.MAX_ORDER + 4:
            _fail(f"{rec.source_file}: too few time points for MPM smoke pass.")

    analysis = KSpaceAnalysis(
        pole_order_config=PoleOrderConfig(
            max_order=cfg.MAX_ORDER,
            min_order=cfg.MIN_ORDER,
            threshold=cfg.SVD_THRESHOLD,
        ),
        n_bootstrap=1,
        signal_component=cfg.SIGNAL_COMPONENT,
    )
    result = analysis.run(records)
    if result.poles.shape != (len(records), SMOKE_K_GRID_N, cfg.MAX_ORDER):
        _fail(f"unexpected pole shape: {result.poles.shape}")
    finite_frac = float(np.isfinite(result.poles).mean())
    if finite_frac < 0.95:
        _fail(f"too many nonfinite poles in smoke result: finite_frac={finite_frac:.3f}")

    print("[plotting] generating real-data parametric PNG/HTML in a temp folder")
    with tempfile.TemporaryDirectory(prefix="kspace_smoke_") as tmp:
        tmp_path = Path(tmp)
        fig = plot_eigenvalues_parametric_grid(
            result, lambda_values=len(records), n_cols=len(records)
        )
        png_path = tmp_path / "parametric_grid.png"
        fig.savefig(png_path, dpi=80)
        html_path = tmp_path / "parametric_slider.html"
        export_eigenvalues_parametric_slider_html(result, str(html_path))
        if png_path.stat().st_size <= 0 or html_path.stat().st_size <= 0:
            _fail("plot smoke output was empty.")
    print(f"[analysis] OK: real-data MPM finite_frac={finite_frac:.3f}; plotting OK")


def _print_code_audit_note() -> None:
    print("[audit] no interpolation/extrapolation is used to make P(k,t).")
    print("        Dense k points are direct evaluations of sum_n P(n,t)*exp(i*k*n).")
    print("        Plot-time continuity sorting uses a linear prediction only to keep")
    print("        track colors continuous; it does not create or alter pole values.")
    print("        Nearest-neighbor selection is used only for choosing display panels/k~pi.")


def main() -> None:
    _assert_config()
    files = _latest_compressed_files(Path(cfg.FOLDER))
    _assert_files(files)
    _assert_probability_and_dft(files)
    _assert_analysis_and_plotting(files)
    _print_code_audit_note()
    print("\nSMOKE CHECK PASSED: scan 22 path uses real compressed records and direct DFT, not demo/interpolation/extrapolation.")


if __name__ == "__main__":
    main()
