"""
click_record_raw_npz.py
-----------------------
Loader for the multi-lambda simulation NPZ produced by
``generate_click_record_raw.py``.

The file stores data for all lambda values in a single NPZ:
  ``lambda_values``     — (n_lambda,) float64
  ``t_values``          — (n_steps,) float64, time axis in µs
  ``proc_packed_{i}``   — packbits of the per-trial **click record**,
                          logical shape (n_steps, n_trials)
  ``raw_packed_{i}``    — same slot, older key name.  In current files it is
                          also a click record; in early (2026-07-08 era)
                          files it is the bright/dark **telegraph**, from
                          which falling-edge (1→0) clicks are derived.
                          :class:`SimClickRecordLoader` tells the two apart
                          automatically — see its docstring.
  ``pnt_{i}``           — P(n,t), shape (n_steps+1, n_steps), float32
  ``pkt_{i}``           — P(k,t), shape (100, n_steps), complex64
  plus scalar metadata (dt, Omega, T1, …)

``pnt_{i}``/``pkt_{i}`` may be absent for the trailing lambdas of a file that
is still being written; the trajectory loaders tolerate that.

Returns
-------
``SimClickRecordLoader.load()``
    → ``list[ClickRecordData]``  compatible with ``connected_correlator()`` and all
      correlator-based analysis in ``click_record_statistics.py``.

``SimClickRecordLoader.load_observable()``
    → ``list[ObservableRecord]`` with ``signal = ⟨n(t)⟩`` read directly from the
      pre-computed ``pnt_{i}``.  Fast — no trajectory processing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord
from mpm_analysis.io.experimental.click_record_npz import ClickRecordData


class SimClickRecordLoader:
    """Load simulated click-record data from a multi-lambda NPZ file.

    The NPZ is produced by ``generate_click_record_raw.py``.

    Trajectory key / convention
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Two key names appear across generator versions, and ``raw_packed_{i}``
    means different things in different vintages:

    * ``proc_packed_{i}`` — the processed **click record** (current
      generator, written when ``config.store_records`` is set).
    * ``raw_packed_{i}`` — in current files also a **click record**; in early
      (2026-07-08 era) files the **raw bright/dark telegraph**, from which
      falling-edge (1→0) clicks must be derived.

    ``click_convention="auto"`` (default) tells them apart on the data
    itself: a click record can never contain two clicks in adjacent slots
    (the readout compression forbids it), whereas a telegraph with any
    bright dwell longer than one slot does.  Verified against each file's
    own ``pnt_{i}``.  Force with ``"clicks"`` or ``"telegraph"``.

    Parameters
    ----------
    load_trajectories : bool
        If False, ``load()`` skips unpacking the (large) trajectory arrays and
        returns ``ClickRecordData`` with empty ``trajectories``.  Useful when
        only P(n,t) statistics are needed.  Default True.
    click_convention : ``"auto"`` | ``"clicks"`` | ``"telegraph"``
        How to interpret the stored per-trial record (see above).
    """

    def __init__(
        self,
        load_trajectories: bool = True,
        *,
        click_convention: str = "auto",
    ) -> None:
        if click_convention not in ("auto", "clicks", "telegraph"):
            raise ValueError(
                f"Unknown click_convention {click_convention!r}; "
                "valid: 'auto', 'clicks', 'telegraph'."
            )
        self._load_trajectories = load_trajectories
        self._click_convention = click_convention

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        path: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
    ) -> list[ClickRecordData]:
        """Load processed click trajectories as ``ClickRecordData`` objects.

        The ``trajectories`` array has shape ``(N, T)`` and contains binary
        click indicators, matching the convention of ``ClickRecordNpzLoader``.
        Files storing a bright/dark telegraph instead of a click record are
        supported — the falling-edge clicks are derived on the fly (see the
        class docstring).

        Parameters
        ----------
        path : str | Path
            Path to the NPZ file.
        lambda_range : (lo, hi) or None
            Keep only lambda values in ``[lo, hi]``.

        Returns
        -------
        list[ClickRecordData]
            One record per lambda, sorted ascending.

        Warning
        -------
        This materialises every λ at once (~N·T float32 each).  For large
        files prefer :meth:`iter_records`, which streams one λ at a time.
        """
        records = [
            self._to_float32(rec)
            for rec in self.iter_records(path, lambda_range=lambda_range)
        ]
        records.sort(key=lambda r: r.lambda_val)
        return records

    def iter_records(
        self,
        path: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
        max_records: int | None = None,
    ):
        """Yield one ``ClickRecordData`` per λ, ascending, one at a time.

        ``trajectories`` is a **boolean** ``(N, T)`` click matrix (memory-lean
        streaming variant of :meth:`load`; the estimators accept it as-is).
        """
        path = Path(path)
        npz = np.load(path, allow_pickle=False)

        lambda_values = npz["lambda_values"]
        t_values = npz["t_values"]
        n_steps = int(npz["n_steps"])
        n_trials = int(npz["n_trials"])
        meta_scalars = self._extract_scalars(npz, path.name)

        available = _available_indices(npz, n_lambda=len(lambda_values))
        order = sorted(available, key=lambda i: float(lambda_values[i]))

        yielded = 0
        for i in order:
            lam = float(lambda_values[i])
            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= lam <= hi):
                    continue
            if max_records is not None and yielded >= max_records:
                return

            try:
                if self._load_trajectories:
                    trajectories = self._load_click_matrix(
                        npz, i, n_steps, n_trials
                    )
                else:
                    trajectories = np.empty((0, n_steps), dtype=bool)
                # pnt/pkt are optional: a simulation still writing its later
                # λ leaves the counting statistics behind the click records.
                meta = {**meta_scalars, "k_values": npz["k_values"]}
                for extra in ("pnt", "pkt"):
                    if f"{extra}_{i}" in npz.files:
                        meta[extra] = npz[f"{extra}_{i}"]
            except Exception as exc:
                print(f"[SimClickRecordLoader] skipping lam={lam:.3f} (index {i}): {exc}")
                continue

            yielded += 1
            yield ClickRecordData(
                lambda_val=lam,
                t=t_values,
                trajectories=trajectories,
                source_file=path.name,
                metadata=meta,
            )

    def _load_click_matrix(
        self,
        npz: np.lib.npyio.NpzFile,
        i: int,
        n_steps: int,
        n_trials: int,
    ) -> np.ndarray:
        """Boolean (N, T) click matrix for λ index ``i``.

        Reads ``proc_packed_{i}`` when present, else ``raw_packed_{i}``, and
        converts a telegraph to falling-edge clicks when needed — see the
        class docstring for how the convention is decided.
        """
        if f"proc_packed_{i}" in npz.files:
            rec = _unpack_trajectories(npz[f"proc_packed_{i}"], n_steps, n_trials)
            convention = "clicks" if self._click_convention == "auto" \
                else self._click_convention
        else:
            rec = _unpack_trajectories(npz[f"raw_packed_{i}"], n_steps, n_trials)
            convention = self._click_convention
            if convention == "auto":
                convention = _detect_convention(rec)

        if convention == "telegraph":
            clicks = np.zeros_like(rec)
            clicks[:, 1:] = rec[:, :-1] & ~rec[:, 1:]
            return clicks
        return rec

    @staticmethod
    def _to_float32(rec: ClickRecordData) -> ClickRecordData:
        """Match the historical float32 trajectory dtype of :meth:`load`."""
        rec.trajectories = rec.trajectories.astype(np.float32)
        return rec

    def load_observable(
        self,
        path: str | Path,
        *,
        lambda_range: tuple[float, float] | None = None,
    ) -> list[ObservableRecord]:
        """Load pre-computed ⟨n(t)⟩ as ``ObservableRecord`` objects.

        Reads ``pnt_{i}`` directly — no trajectory unpacking.  Returns the
        same observable key (``"mean_clicks"``) used by
        ``_make_mean_clicks_loader()`` so these records are drop-in
        replacements for those from ``ParityLoader``.

        Parameters
        ----------
        path : str | Path
        lambda_range : (lo, hi) or None

        Returns
        -------
        list[ObservableRecord]
            ``signal`` is ``⟨n(t)⟩ = Σ_n n · P(n,t)``, same shape as ``t``.
        """
        path = Path(path)
        npz = np.load(path, allow_pickle=False)

        lambda_values = npz["lambda_values"]
        t_values = npz["t_values"]
        n_steps = int(npz["n_steps"])
        n_arr = np.arange(n_steps + 1, dtype=np.float64)
        meta_scalars = self._extract_scalars(npz, path.name)

        available = _available_indices(
            npz, n_lambda=len(lambda_values), require_pnt=True,
        )

        records: list[ObservableRecord] = []
        for i, lam in enumerate(lambda_values):
            if i not in available:
                continue
            if lambda_range is not None:
                lo, hi = lambda_range
                if not (lo <= float(lam) <= hi):
                    continue

            try:
                pnt = npz[f"pnt_{i}"].astype(np.float64)   # (n_steps+1, n_steps)
                mean_n = n_arr @ pnt                         # (n_steps,)
            except Exception as exc:
                print(f"[SimClickRecordLoader] skipping lam={float(lam):.3f} (index {i}): {exc}")
                continue

            records.append(
                ObservableRecord(
                    lambda_val=float(lam),
                    t=t_values,
                    signal=mean_n,
                    observable_key="mean_clicks",
                    source_file=path.name,
                    metadata=meta_scalars,
                )
            )

        records.sort(key=lambda r: r.lambda_val)
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_scalars(npz: np.lib.npyio.NpzFile, source: str) -> dict:
        """Pull all scalar metadata keys into a plain dict."""
        scalar_keys = (
            "n_trials", "n_steps", "dt", "substeps", "dt_fine", "T_total",
            "Omega", "T1", "T2", "tau_B", "fp", "fn",
            "gamma_1", "gamma_2", "gamma_deph", "gamma_1B",
            "unc_lo", "unc_hi",
        )
        meta = {"source_file": source}
        for k in scalar_keys:
            if k in npz:
                meta[k] = npz[k].item()
        return meta


# ---------------------------------------------------------------------------
# Module-level helpers (not exported)
# ---------------------------------------------------------------------------

def _detect_convention(rec: np.ndarray, *, sample: int = 4096) -> str:
    """``"clicks"`` or ``"telegraph"`` for a boolean ``(N, T)`` record.

    A click record can never hold two clicks in adjacent slots (the readout
    compression forbids it), so any adjacent ``1,1`` pair means the array is
    still the bright/dark telegraph.  Only the first ``sample`` trials are
    inspected — one pair is enough to decide, and telegraphs have many.
    """
    sub = rec[:sample]
    if sub.size and np.any(sub[:, :-1] & sub[:, 1:]):
        return "telegraph"
    return "clicks"


def _available_indices(
    npz: np.lib.npyio.NpzFile,
    n_lambda: int,
    *,
    require_pnt: bool = False,
) -> set[int]:
    """Return the set of lambda indices whose required keys are present.

    A λ always needs a per-trial record (``proc_packed_{i}`` or
    ``raw_packed_{i}``); ``require_pnt`` additionally demands ``pnt_{i}``
    (needed by :meth:`SimClickRecordLoader.load_observable`, but not by the
    trajectory loaders — partially-written files are common while a long
    simulation is still running).  Checks key existence only; corrupt entries
    are caught at read time.
    """
    keys = set(npz.files)
    return {
        i for i in range(n_lambda)
        if (f"proc_packed_{i}" in keys or f"raw_packed_{i}" in keys)
        and (not require_pnt or f"pnt_{i}" in keys)
    }


def _unpack_trajectories(
    packed: np.ndarray,
    n_steps: int,
    n_trials: int,
) -> np.ndarray:
    """Unpack a ``packbits``-compressed trajectory array.

    The NPZ stores trajectories in T×N order (time-major) before packing.
    We unpack, reshape to (T, N), then transpose to (N, T) to match the
    ``ClickRecordData.trajectories`` convention.

    Returns
    -------
    np.ndarray, shape (N, T), dtype bool
        (kept boolean for memory; :meth:`SimClickRecordLoader.load` casts to
        the historical float32 on the way out)
    """
    flat = np.unpackbits(packed)[: n_steps * n_trials]
    return flat.reshape(n_steps, n_trials).T.astype(bool)
