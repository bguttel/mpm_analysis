"""
pole_order.py
-------------
SVD-based automatic model-order selection for k-space MPM cells.

The heuristic compares two consecutive singular-value gaps:

    ratio_23 = S[1] / S[2]   (gap between 2nd and 3rd singular value)
    ratio_34 = S[2] / S[3]   (gap between 3rd and 4th singular value)

If the 3rd pole is a genuine signal component, the gap S[2]/S[3] should
be comparable to S[1]/S[2].  If S[2] is already in the noise floor,
ratio_34 will be close to 1 while ratio_23 is large.

Decision rule::

    if ratio_34 >= threshold * ratio_23:
        order = max_order   # 3rd gap is real → use it
    else:
        order = min_order   # 3rd gap is noise → drop it

``threshold`` defaults to 0.5 — the 3rd gap must be at least half the
2nd gap to justify an extra pole.  Lower values are more permissive
(keep 3 poles more often); higher values are more conservative.

Manual overrides (per cell) take precedence over the automatic rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoleOrderConfig:
    """Configuration for automatic + manual pole-order selection.

    Parameters
    ----------
    max_order:
        Maximum number of poles to extract (default 3).
    min_order:
        Minimum number of poles to extract (default 2).
    threshold:
        SVD ratio threshold (see module docstring).  Default 0.5.
    overrides:
        Dict mapping ``(lambda_idx, k_idx)`` → integer order.
        Takes precedence over the automatic rule for that cell.
        Pass ``None`` to disable overrides entirely.
    """

    max_order: int = 3
    min_order: int = 2
    threshold: float = 0.5
    overrides: dict[tuple[int, int], int] | None = None

    def get_override(self, lam_idx: int, k_idx: int) -> int | None:
        """Return the manual override for cell (lam_idx, k_idx), or None."""
        if self.overrides is None:
            return None
        return self.overrides.get((lam_idx, k_idx))


def infer_pole_order(
    singvals: np.ndarray,
    config: PoleOrderConfig,
    *,
    lam_idx: int | None = None,
    k_idx: int | None = None,
) -> int:
    """Infer the model order for one MPM cell from Hankel singular values.

    Parameters
    ----------
    singvals:
        Sorted (descending) singular values of the Hankel matrix.  At
        least ``max_order + 1`` values must be present for the ratio test;
        if fewer are available the function falls back to ``min_order``.
    config:
        ``PoleOrderConfig`` controlling thresholds and overrides.
    lam_idx, k_idx:
        Cell coordinates for override lookup.  Ignored when ``None``.

    Returns
    -------
    int
        Model order to use for this cell.
    """
    # Manual override takes precedence
    if lam_idx is not None and k_idx is not None:
        ov = config.get_override(lam_idx, k_idx)
        if ov is not None:
            return ov

    # Need at least max_order + 1 singular values for the ratio test
    if len(singvals) < config.max_order + 1:
        return config.min_order

    s = np.asarray(singvals, dtype=float)

    # Guard against near-zero denominators
    denom_23 = s[config.min_order]       # S[2] when min_order=2
    denom_34 = s[config.max_order]       # S[3] when max_order=3

    if denom_23 < 1e-12 or denom_34 < 1e-12:
        return config.min_order

    ratio_23 = s[config.min_order - 1] / denom_23   # S[1]/S[2]
    ratio_34 = s[config.min_order] / denom_34        # S[2]/S[3]

    if ratio_34 >= config.threshold * ratio_23:
        return config.max_order
    return config.min_order


# =============================================================================
#  Order MAPS: per-cell automatic calibration over a whole (lambda, k) grid
# =============================================================================
#
# The functions below decide a model order for EVERY cell at once from the
# stacked Hankel singular values, and plug into the existing machinery through
# ``PoleOrderConfig(overrides=overrides_from_map(order_map))`` -- overrides
# beat the built-in ratio rule per cell, so ``KSpaceAnalysis`` and RRHA run
# unmodified.
#
# ``singvals`` is always ``(..., n_sv)``, descending; every leading axis is a
# cell axis.  By convention the FIRST cell axis is lambda (this is what the
# ``monotone`` flag of the ridge rule refers to).


def noise_plateau(singvals: np.ndarray, order: int) -> np.ndarray:
    """The noise floor under a claimed order: median of ``s_i/s_0``, i >= order.

    This is the grey band of the SVD-scan diagnostic figures -- singular
    values below it carry no signal.  Returns one value per cell (the shape
    of ``singvals`` without its last axis).
    """
    s = np.asarray(singvals, dtype=float)
    if s.shape[-1] <= order:
        return np.full(s.shape[:-1], np.nan)
    ratios = s / np.where(s[..., :1] > 0, s[..., :1], np.inf)
    return np.nanmedian(ratios[..., order:], axis=-1)


def ridge_drop_order(
    singvals: np.ndarray,
    *,
    base: int = 3,
    min_order: int = 2,
    margin: float = 3.0,
    monotone: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell order by successive drops: start at ``base``, drop while the
    last claimed singular value sinks into the noise plateau.

    For each cell and each candidate order n (from ``base`` down to
    ``min_order + 1``): if ``s_{n-1}/s_0 < margin * noise_plateau(n)`` the
    n-th pole is noise and the order drops to n-1.  This formalizes the
    ridge-plot reading of the SVD-vs-lambda curves: a strand's singular value
    "sinking into the grey band" is exactly the drop condition.

    ``monotone=True`` makes drops sticky along the first cell axis (lambda by
    convention): once a cell drops, no later lambda climbs back up.  Returns
    ``(order_map, sigma_floor, margin_map)``: the int order per cell, the
    absolute singular-value threshold applied (``margin * plateau * s_0``),
    and ``s_{order-1} / (margin * plateau * s_0)`` -- how far above its own
    floor the last kept singular value sits (< 1 means the kept order is
    itself marginal; only possible at ``min_order``).
    """
    s = np.asarray(singvals, dtype=float)
    cells = s.shape[:-1]
    order = np.full(cells, int(base), dtype=int)
    for n in range(int(base), int(min_order), -1):
        plateau = noise_plateau(s, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            last = np.where(s[..., 0] > 0, s[..., n - 1] / s[..., 0], np.nan)
            drop = (order == n) & (last < margin * plateau)
        order[drop & np.isfinite(last)] = n - 1
    if monotone and order.ndim >= 1 and order.shape[0] > 1:
        order = np.minimum.accumulate(order, axis=0)
    # sigma_floor / margin at the KEPT order:
    floor = np.empty(cells, dtype=float)
    marg = np.empty(cells, dtype=float)
    for n in np.unique(order):
        sel = order == n
        pl = noise_plateau(s, int(n))
        with np.errstate(invalid="ignore", divide="ignore"):
            f = margin * pl * s[..., 0]
            m = np.where(f > 0, s[..., int(n) - 1] / f, np.nan)
        floor[sel] = f[sel]
        marg[sel] = m[sel]
    return order, floor, marg


def sig_digits_order(
    singvals: np.ndarray,
    *,
    p: float | np.ndarray | None = None,
    noise: np.ndarray | None = None,
    signal_max: np.ndarray | None = None,
    max_order: int = 4,
    min_order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell order from the significant digits of the data:
    ``sigma_c / sigma_max = 10^(-p)``.

    Singular values below the data's own precision floor are noise, so the
    order is the count of ``s_i/s_0 >= 10^(-p)``, clipped to
    ``[min_order, max_order]``.  ``p`` per cell comes from (in precedence
    order): the explicit ``p``; ``-log10(noise / signal_max)`` when a
    per-cell noise floor is supplied (e.g. the Monte-Carlo estimator noise
    ``sqrt((1-|P|^2)/N)``); else 12 (analytic data, round-off with margin).

    Returns ``(order_map, sigma_floor)`` with ``sigma_floor = 10^{-p} s_0``.
    """
    s = np.asarray(singvals, dtype=float)
    cells = s.shape[:-1]
    if p is not None:
        p_cell = np.broadcast_to(np.asarray(p, dtype=float), cells)
    elif noise is not None:
        smax = (np.broadcast_to(np.asarray(signal_max, dtype=float), cells)
                if signal_max is not None else np.ones(cells))
        with np.errstate(invalid="ignore", divide="ignore"):
            p_cell = -np.log10(np.asarray(noise, dtype=float) / smax)
        p_cell = np.where(np.isfinite(p_cell), p_cell, 12.0)
    else:
        p_cell = np.full(cells, 12.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = s / np.where(s[..., :1] > 0, s[..., :1], np.inf)
    floor_ratio = 10.0 ** (-p_cell)
    count = np.sum(ratios >= floor_ratio[..., None], axis=-1)
    order = np.clip(count, min_order, max_order).astype(int)
    return order, floor_ratio * s[..., 0]


def infer_order_map(
    singvals: np.ndarray,
    *,
    policy: str = "ridge",
    axis: str = "record",
    base: int | None = None,
    min_order: int = 2,
    margin: float = 3.0,
    monotone: bool = True,
    p: float | np.ndarray | None = None,
    noise: np.ndarray | None = None,
    signal_max: np.ndarray | None = None,
    fixed: int = 3,
    max_order: int | None = None,
) -> tuple[np.ndarray, dict]:
    """The front door: a per-cell order map + diagnostics, by policy.

    ``policy``:
      * ``"ridge"`` (default) -- :func:`ridge_drop_order`, starting at
        ``base`` (default: 3 on the record axis, 4 on the raw axis) and
        dropping toward ``min_order`` where the last singular value sinks
        into the noise plateau;
      * ``"sig_digits"`` -- :func:`sig_digits_order` with the
        ``sigma_c/sigma_max = 10^(-p)`` rule;
      * ``"fixed"`` -- every cell at ``fixed``.

    Returns ``(order_map, diag)``; ``diag`` carries ``policy`` (a string for
    NPZ metadata), ``sigma_floor`` and, for ridge, ``margin_map``.
    """
    s = np.asarray(singvals, dtype=float)
    if policy == "ridge":
        b = base if base is not None else (4 if axis == "raw" else 3)
        order, floor, marg = ridge_drop_order(
            s, base=b, min_order=min_order, margin=margin, monotone=monotone)
        diag = {"policy": f"ridge(base={b},min={min_order},margin={margin:g},"
                          f"monotone={monotone},axis={axis})",
                "sigma_floor": floor, "margin_map": marg}
    elif policy == "sig_digits":
        mo = max_order if max_order is not None else (4 if axis == "raw" else 3)
        order, floor = sig_digits_order(
            s, p=p, noise=noise, signal_max=signal_max,
            max_order=mo, min_order=min_order)
        p_txt = "auto" if p is None else f"{np.max(np.asarray(p)):g}"
        diag = {"policy": f"sig_digits(p={p_txt},max={mo},min={min_order},"
                          f"axis={axis})",
                "sigma_floor": floor}
    elif policy == "fixed":
        order = np.full(s.shape[:-1], int(fixed), dtype=int)
        diag = {"policy": f"fixed({fixed})",
                "sigma_floor": np.full(s.shape[:-1], np.nan)}
    else:
        raise ValueError(
            f"unknown policy {policy!r}; expected ridge / sig_digits / fixed."
        )
    return order, diag


def overrides_from_map(order_map: np.ndarray) -> dict[tuple[int, int], int]:
    """An ``(n_lam, n_k)`` order map as the overrides dict
    :class:`PoleOrderConfig` consumes -- every cell becomes an override, so
    the map is the sole authority and the built-in ratio rule never fires.
    """
    order_map = np.asarray(order_map)
    if order_map.ndim != 2:
        raise ValueError(
            f"overrides_from_map wants an (n_lambda, n_k) map, got shape "
            f"{order_map.shape}."
        )
    return {(int(i), int(j)): int(order_map[i, j])
            for i in range(order_map.shape[0])
            for j in range(order_map.shape[1])}
