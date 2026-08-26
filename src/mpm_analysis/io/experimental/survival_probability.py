"""
survival_probability.py
-----------------------
Concrete loaders for the two main experimental data formats.

PostSelectedLoader
    Folders of per-lambda JSON files from post-selected quantum-state-tomography.
    Files contain ``data.p_0_t_array`` (survival probability).

    Lambda source (two modes):
      - CSV mode (accurate):  pass ``csv_path`` pointing to a results CSV with a
        ``Gamma_up_3_states_hmm`` column.  Lambda is then computed as
        ``Gamma_up / (2 * Omega_DG * 1e-6)``, matching the original analysis in
        ``object_oriented_poles.py``.  This is the "updated lambda scale" that
        uses HMM-inferred drive rates rather than the stored estimate.
      - JSON mode (fallback):  if no CSV is given, reads
        ``parameters.lambda_zeno_estimation`` directly from each JSON file.

EnsembleAverageLoader
    Folders of per-lambda JSON files from ensemble-average measurements.
    Files contain ``data.D_probability`` (survival) and ``data.G_probability``
    which serves as the Z-like observable (matching ``object_oriented_poles.py``
    where ``z_arr = g_arr = jd["data"]["D_probability"] - jd["data"]["G_probability"]``).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from mpm_analysis.io.experimental.base import (
    ExperimentalLoader,
    _load_json,
    _get_nested,
    _khz_from_name,
)
from mpm_analysis.utils.windows_paths import win_long_path

_SCAN_NUMBER_RE = re.compile(r"_(\d+)$")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_lambda_from_json(jd: dict) -> float:
    """Extract lambda from a raw JSON dict (tries several key variants)."""
    params = jd.get("parameters", jd)
    lam = (
        params.get("lambda_zeno_estimation")
        or params.get("lambda")
        or params.get("lambda_zeno")
        or _get_nested(jd, ["parameters", "lambda_zeno_estimation"])
    )
    if lam is None:
        raise KeyError("Cannot find lambda_zeno_estimation in JSON parameters.")
    return float(lam)


def _read_omega_dg_mhz(jd: dict) -> float:
    """Read Omega_DG from JSON parameters and convert Hz -> MHz (rad/us)."""
    params = jd.get("parameters", jd)
    val = params.get("Omega_DG") or params.get("omega_DG")
    if val is None:
        raise KeyError("Cannot find Omega_DG in JSON parameters.")
    return float(val) * 1e-6  # Hz -> MHz (= rad/us)


def _read_time_us(jd: dict) -> np.ndarray:
    """Extract time array in µs from nested data.time_array (originally in ns)."""
    # Original code: times = np.array(jd["data"]["time_array"]) / 1000.0
    t = _get_nested(jd, ["data", "time_array"]) or _get_nested(jd, ["data", "time_array_ns"])
    if t is None:
        # Fallback: try top-level keys for non-standard files
        t = jd.get("time_array") or jd.get("time_array_ns")
    if t is None:
        raise KeyError("Cannot find time_array in JSON (checked data.time_array and top level).")
    t = np.asarray(t, dtype=float)
    # Convert ns -> µs (heuristic: max > 1000 means ns)
    if np.nanmax(t) > 1_000.0:
        t = t / 1e3
    return t


def _load_csv_gamma_ups(csv_path: Path) -> dict[str, float]:
    """Load HMM Gamma_up values from a results CSV, keyed by filename fragment.

    The CSV must have columns ``filename`` and ``Gamma_up_3_states_hmm``.
    The key stored is the fragment between ``OBG0`` and ``kHz`` in the filename,
    matching the convention used in ``object_oriented_poles.py``.
    """
    gamma_ups: dict[str, float] = {}
    before_string = "OBG0"
    after_string = "kHz"
    with open(win_long_path(csv_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            key = fname.split(before_string)[-1].split(after_string)[0]
            gamma_ups[key] = float(row["Gamma_up_3_states_hmm"])
    return gamma_ups


# ---------------------------------------------------------------------------
# Post-selected loader
# ---------------------------------------------------------------------------

class PostSelectedLoader(ExperimentalLoader):
    """Load post-selected tomography survival-probability data.

    Parameters
    ----------
    csv_path:
        Path to the HMM results CSV file (columns: ``filename``,
        ``Gamma_up_3_states_hmm``).  When given, lambda is computed as
        ``Gamma_up / (2 * Omega_DG * 1e-6)`` — the updated, HMM-calibrated
        lambda scale.  When omitted, falls back to
        ``parameters.lambda_zeno_estimation`` in each JSON.

    Default observables registered:
        ``"survival"``      <- ``data.p_0_t_array``
        ``"Z"``             <- ``data.s_z``  (NaN if absent)
        ``"G_probability"`` <- ``data.G_probability``  (NaN if absent)
        ``"B_probability"`` <- ``data.B_probability``  (NaN if absent)
        ``"D_probability"`` <- ``data.D_probability``  (NaN if absent)
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path is not None else None
        self._gamma_ups: dict[str, float] | None = None  # lazy-loaded
        super().__init__()
    
    

    def _register_default_extractors(self) -> None:
        self.add_extractor(
            "S",
            lambda jd, t: np.asarray(jd["data"]["p_0_t_array"], dtype=float),
        )
        self.add_extractor(
            "Z",
            lambda jd, t: np.asarray(jd["data"]["z_axis_array"], dtype=float)
            * np.asarray(jd["data"]["z_axis_number_of_measurements"], dtype=float)/60000,
        )
        self.add_extractor(
            "X",
            lambda jd, t: np.asarray(jd["data"]["main_phase_axis_array"], dtype=float)
            * np.asarray(jd["data"]["phase_axes_number_of_measurements"], dtype=float)/60000,
        )
        self.add_extractor(
                    "SX",
                    lambda jd, t: np.asarray(jd["data"]["phase_axes_number_of_measurements"], dtype=float)/60000,
                )
        self.add_extractor(
                            "SZ",
                            lambda jd, t: np.asarray(jd["data"]["z_axis_number_of_measurements"], dtype=float)/60000,
                        )
        self.add_extractor(
                    "Y",
                    lambda jd, t: np.asarray(jd["data"]["orthogonal_phase_axis_array"], dtype=float)
                    * np.asarray(jd["data"]["p_0_t_array"], dtype=float),
                )

    def _bootstrap_stats(self, jd: dict, mask: np.ndarray | None) -> dict | None:
        """Binomial statistics for every post-selected channel.

        The scan runs ``navg`` repetitions of each of 6 axis settings
        (z, -z, x, -x, y, -y).  Each measurement therefore has its own number
        of attempts, and its own survival:

        ==========  ===============  ====================================
        channel     attempts         survival fraction
        ==========  ===============  ====================================
        ``S``       ``navg * 6``     ``p_0_t_array``   (dedicated run)
        ``SX``      ``navg * 4``     ``N_phase / (navg*2)``  (x,-x,y,-y)
        ``SZ``      ``navg * 2``     ``N_z / (navg*2)``      (z,-z)
        ==========  ===============  ====================================

        ``phase_axes_number_of_measurements`` counts survivors *per phase
        axis* (out of ``navg*2``), so the phase-axis run pools twice that many
        survivors over twice as many attempts — the fraction is unchanged,
        which is why ``SX`` and ``SZ`` share the ``navg*2`` denominator while
        their attempt counts differ.

        The conditional Bloch coordinates X / Y / Z are resampled on the
        surviving trajectories (``N_X``/``N_Y``/``N_Z``); X and Y share the
        one count array the files provide.
        """
        data = jd["data"]
        params = jd.get("parameters", {})

        def _arr(key, dtype=float, *, required=True):
            v = data.get(key)
            if v is None:
                if required:
                    raise KeyError(f"'{key}' missing from data")
                return None
            v = np.asarray(v, dtype=dtype)
            return v[mask] if mask is not None else v

        navg = int(params.get("actual_navg_repeat_experiment")
                   or params.get("navg_repeat_experiment") or 30_000)

        # Shared count array for both phase axes (survivors per phase axis)
        n_phase = _arr("phase_axes_number_of_measurements", int)
        n_z = _arr("z_axis_number_of_measurements", int)
        per_axis_attempts = navg * 2          # one axis pair = 2 settings

        return {
            "data_mode": "post_selected",
            # --- survival channels: fraction + attempts behind it ---
            "survival": _arr("p_0_t_array"),
            "N_survival": navg * 6,
            "SX": n_phase / per_axis_attempts,
            "N_attempts_SX": navg * 4,        # x, -x, y, -y
            "SZ": n_z / per_axis_attempts,
            "N_attempts_SZ": navg * 2,        # z, -z
            # --- conditional Bloch coordinates + their survivor counts ---
            "X": _arr("main_phase_axis_array"),
            "Y": _arr("orthogonal_phase_axis_array", required=False),
            "Z": _arr("z_axis_array"),
            "N_X": n_phase,
            "N_Y": n_phase,
            "N_Z": n_z,
        }

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"), key=lambda fp: _safe_lambda(fp))

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        t = _read_time_us(jd)

        if self._csv_path is not None:
            # CSV (HMM) lambda: Gamma_up / (2 * Omega_DG)
            if self._gamma_ups is None:
                self._gamma_ups = _load_csv_gamma_ups(self._csv_path)
            before_string = "OBG0"
            after_string = "kHz"
            key = fp.name.split(before_string)[-1].split(after_string)[0]
            if key not in self._gamma_ups:
                raise KeyError(
                    f"no HMM row for OBG0 fragment '{key}' in "
                    f"{self._csv_path.name} ({len(self._gamma_ups)} rows)"
                )
            omega_dg = _read_omega_dg_mhz(jd)
            lam = self._gamma_ups[key] / (2.0 * omega_dg)
        else:
            # Fallback: use stored lambda estimate from JSON
            lam = _read_lambda_from_json(jd)

        return lam, t


# ---------------------------------------------------------------------------
# Ensemble-average loader
# ---------------------------------------------------------------------------

class EnsembleAverageLoader(ExperimentalLoader):
    """Load ensemble-average survival-probability data.

    Default observables registered:
        ``"survival"`` <- ``data.D_probability``
        ``"Z"``        <- ``data.D_probability - data.G_probability``  
    """
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self._csv_path = Path(csv_path) if csv_path is not None else None
        self._gamma_ups: dict[str, float] | None = None  # lazy-loaded
        super().__init__()

    def _register_default_extractors(self) -> None:
        self.add_extractor(
                    "survival",
                    lambda jd, t: (
                        np.asarray(jd["data"]["D_probability"], dtype=float)
                    )),
        self.add_extractor(
            "D",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
            )),
        self.add_extractor(
                    "G",
                    lambda jd, t: (
                        np.asarray(jd["data"]["G_probability"], dtype=float)
                    )),
        self.add_extractor(
                    "B",
                    lambda jd, t: (
                        np.asarray(jd["data"]["B_probability"], dtype=float)
                    )),
        self.add_extractor(
                            "error",
                            lambda jd, t: (
                                np.asarray(jd["data"]["error_probability"], dtype=float)
                            )),
        self.add_extractor(
            "Z",
            lambda jd, t: (
                np.asarray(jd["data"]["D_probability"], dtype=float)
                - np.asarray(jd["data"]["G_probability"], dtype=float)
            ) if "G_probability" in jd.get("data", {})
            else np.full_like(t, float("nan")),
        )


    def _bootstrap_stats(self, jd: dict, mask: np.ndarray | None) -> dict | None:
        """Multinomial statistics: D/G/B/error share the same N_shots.

        The four outcomes are mutually exclusive results of the same shots
        (they sum to 1), so resampling is one multinomial draw per time
        point — never independent noise per channel.
        """
        data = jd["data"]
        params = jd.get("parameters", {})
        navg = (params.get("actual_navg_repeat_experiment")
                or params.get("navg_repeat_experiment"))
        if not navg:
            return None    # never invent shot counts

        P = np.vstack([
            np.asarray(data["D_probability"], dtype=float),
            np.asarray(data["G_probability"], dtype=float),
            np.asarray(data["B_probability"], dtype=float),
            np.asarray(data["error_probability"], dtype=float),
        ])
        if mask is not None:
            P = P[:, mask]
        return {
            "data_mode": "ens_avg",
            "P": P,
            "channels": ["D", "G", "B", "error"],
            "N_shots": int(navg),
        }

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"), key=lambda fp: _safe_lambda(fp))

    def _extract_lambda_and_time(self, jd: dict, fp: Path) -> tuple[float, np.ndarray]:
        if self._csv_path is not None:
            # CSV (HMM) lambda: Gamma_up / (2 * Omega_DG)
            if self._gamma_ups is None:
                self._gamma_ups = _load_csv_gamma_ups(self._csv_path)
            before_string = "OBG0"
            after_string = "kHz"
            key = fp.name.split(before_string)[-1].split(after_string)[0]
            if key not in self._gamma_ups:
                raise KeyError(
                    f"no HMM row for OBG0 fragment '{key}' in "
                    f"{self._csv_path.name} ({len(self._gamma_ups)} rows)"
                )
            omega_dg = _read_omega_dg_mhz(jd)
            lam = self._gamma_ups[key] / (2.0 * omega_dg)
        else:           
            # Fallback: use stored lambda estimate from JSON
            lam = _read_lambda_from_json(jd)
        return lam, _read_time_us(jd)

class ParityLoader(ExperimentalLoader):
    """Loader for parity-scan experiments.

    Extracts ``parity_real`` and ``parity_imag`` observables.

    Parameters
    ----------
    k_idx:
        Index into ``data.fourier_k_array`` to use as the 1-D time trace.
        When set, slices ``data.fourier_transform_pk_array`` at that k
        index (new 2-D data format).  When ``None`` (default), falls back
        to the legacy ``data.fourier_transform_pk`` field (single-k format).
        Use ``ParityKSpaceLoader`` to load all k values simultaneously.
    """

    def __init__(self, k_idx: int | None = None) -> None:
        self._k_idx = k_idx
        # Set on the FIRST file that triggers the DFT fallback so we only
        # print the announcement banner once per ParityLoader instance.
        self._fallback_announced = False
        super().__init__()

    def _compute_pkt_column(self, jd: dict, k_idx: int, component: str) -> np.ndarray:
        """Fallback: compute one column of P(k,t) by DFT of P(n,t).

        Used when ``data.fourier_transform_pk_array`` is missing (older
        scans like scan_experiment_results_12).  Uses the same default
        k-grid (``np.linspace(0, 2*pi, 101)``) as the newer scans, so
        ``k_idx`` keeps the same physical meaning across scans.
        """
        # Local import avoids any chance of circulars at module load.
        from mpm_analysis.kspace.loader import compute_pkt_from_pnt

        P_nt_raw = jd["data"]["jump_probability_distribution"]
        P_nt = np.asarray(P_nt_raw, dtype=float)
        P_kt, _ = compute_pkt_from_pnt(P_nt, k_array=None)  # (n_time, n_k)

        if not self._fallback_announced:
            print(
                f"[ParityLoader] *** P(k,t) FALLBACK ACTIVE (k_idx={k_idx}) ***\n"
                "  'fourier_transform_pk_array' not in JSON; computing column via\n"
                "      P(k,t) = sum_n P(n,t) * exp(+i*k*n)\n"
                "  on default k-grid np.linspace(0, 2*pi, 101) "
                "from data.jump_probability_distribution."
            )
            self._fallback_announced = True

        col = P_kt[:, k_idx]  # (n_time,)
        return col.real if component == "real" else col.imag

    def _register_default_extractors(self) -> None:
        k_idx = self._k_idx

        if k_idx is not None:
            # New 2-D format: slice at the requested k index.
            # JSON layout: fourier_transform_pk_array.real/imag  shape (n_time, n_k)
            # Falls back to DFT of P(n,t) for older scans missing that array.
            def _extract_real(jd: dict, t: np.ndarray) -> np.ndarray:
                pk_arr = jd["data"].get("fourier_transform_pk_array")
                if pk_arr is None:
                    return self._compute_pkt_column(jd, k_idx, "real")
                arr = np.asarray(pk_arr["real part"], dtype=float)
                return arr[:, k_idx]  # (n_time,)

            def _extract_imag(jd: dict, t: np.ndarray) -> np.ndarray:
                pk_arr = jd["data"].get("fourier_transform_pk_array")
                if pk_arr is None:
                    return self._compute_pkt_column(jd, k_idx, "imag")
                arr = np.asarray(pk_arr["imaginary part"], dtype=float)
                return arr[:, k_idx]  # (n_time,)
        else:
            # Legacy single-k format
            def _extract_real(jd: dict, t: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jd["data"]["fourier_transform_pk"]["real part"], dtype=float
                )

            def _extract_imag(jd: dict, t: np.ndarray) -> np.ndarray:
                return np.asarray(
                    jd["data"]["fourier_transform_pk"]["imaginary part"], dtype=float
                )

        self.add_extractor("parity_real", _extract_real)
        self.add_extractor("parity_imag", _extract_imag)

    def _discover_files(self, folder: Path) -> list[Path]:
        return sorted(folder.glob("*.json"))

    def _extract_lambda_and_time(self, jd: dict, fp: Path = None) -> tuple[float, np.ndarray]:
        t = _read_time_us(jd)
        lam = _read_lambda_from_json(jd)
        return lam, t

    def load(self, folder, observables=None, **kwargs):
        if observables is None:
            observables = ["parity_real", "parity_imag"]
        return super().load(folder, observables=observables, **kwargs)
# ---------------------------------------------------------------------------
# Helper: safe lambda for sorting before the JSON is read
# ---------------------------------------------------------------------------

def _safe_lambda(fp: Path) -> float:
    """Return a sort key from the filename (kHz number), fallback to 0."""
    try:
        return float(_khz_from_name(fp))
    except ValueError:
        return 0.0

