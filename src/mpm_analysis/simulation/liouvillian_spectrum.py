
from __future__ import annotations

import numpy as np

from mpm_analysis.analysis.pole_sorting import sort_poles_canonical


def build_liouvillian(lam: float, tau_B: float, T2: float, T1: float, Omega: float, k: float = np.pi) -> np.ndarray:
    """Return the 4x4 Liouvillian matrix as a complex numpy array."""
    eik = np.exp(1j * k)
    alpha = 2 * lam * Omega

    return np.array(
        [[-1/tau_B, alpha/2, 0, -alpha/2],
         [eik/tau_B, -alpha/2, 0, alpha/2],
         [0, 0, -1/T2 - alpha/2, Omega],
         [-eik/tau_B , alpha/2 - 1/T1, -Omega, -alpha/2 - 1/T1]
        ], dtype=complex
    )


class LiouvillianSpectrum:
    """Scan eigenvalues of the 4×4 Liouvillian over a range of λ values"""

    def __init__(
        self,
        Omega: float = 0.628,
        T1: float = 15,
        T2: float = 21.0,
        tau_B: float = 4.0,
        k: float = np.pi,
        lambda_min: float = 0.2,
        lambda_max: float = 1.8,
        n_lambda: int = 200,

    ) -> None:
        self.Omega = Omega
        self.T1 = T1
        self.T2 = T2
        self.tau_B = tau_B
        self.k = k
        self.lambda_vals = np.linspace(lambda_min, lambda_max, n_lambda)

    def compute(self) -> np.ndarray:
        """Compute eigenvalues for all λ values.

        Returns
        -------
        np.ndarray, shape (n_lambda, 4), dtype complex
            Eigenvalues sorted by real part (most negative first) at each λ.
        """
        eigs = np.empty((len(self.lambda_vals), 4), dtype=complex)
        for i, lam in enumerate(self.lambda_vals):
            L = build_liouvillian(lam=lam, tau_B=self.tau_B, T2=self.T2, T1=self.T1, Omega=self.Omega, k=self.k)
            ev = np.linalg.eigvals(L)
            eigs[i] = sort_poles_canonical(ev)
        return eigs

    def plot(self, ax_real=None, ax_imag=None):
        """Plot real and imaginary parts of eigenvalues vs λ.

        Parameters
        ----------
        ax_real, ax_imag:
            Matplotlib axes.  If None, a new figure with two subplots is created.

        Returns
        -------
        fig, (ax_real, ax_imag)
        """
        import matplotlib.pyplot as plt

        eigs = self.compute()
        lams = self.lambda_vals

        if ax_real is None or ax_imag is None:
            fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig = ax_real.get_figure()

        for j in range(4):
            ax_real.plot(lams, eigs[:, j].real, lw=1.2)
            ax_imag.plot(lams, eigs[:, j].imag, lw=1.2)

        ax_real.set_xlabel("λ")
        ax_real.set_ylabel("Re(eigenvalue)")
        ax_real.set_title(f"Decay rates  (k = {self.k/np.pi:.2g}π)")
        ax_real.axhline(0, color="k", lw=0.5, ls="--")
        ax_real.grid(True, alpha=0.3)

        ax_imag.set_xlabel("λ")
        ax_imag.set_ylabel("Im(eigenvalue)")
        ax_imag.set_title(f"Frequencies  (k = {self.k/np.pi:.2g}π)")
        ax_imag.axhline(0, color="k", lw=0.5, ls="--")
        ax_imag.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig, (ax_real, ax_imag)

# ---------------------------------------------------------------------------
# Transition location of the theory, and its dependence on the dead time
# ---------------------------------------------------------------------------

def theory_transition(
    *,
    tau_B: float,
    T1: float,
    T2: float,
    Omega: float,
    k: float = 0.0,
    lambda_range: tuple[float, float] = (0.2, 3.0),
    n_lambda: int = 1401,
    tol: float = 1e-6,
) -> float:
    """λ at which the oscillating eigenvalue pair of the 4×4 Liouvillian
    coalesces (its imaginary part vanishes) — the theory's exceptional point.

    ``k = 0`` is the plain master equation, i.e. the ensemble-average
    dynamics the D-probability fits see.  A coarse λ scan finds the first
    all-real spectrum; a bisection between the last oscillating and the
    first non-oscillating grid point refines it to ``tol``.  Returns NaN if
    the pair never coalesces inside ``lambda_range``.
    """
    def oscillates(lam: float) -> bool:
        ev = np.linalg.eigvals(build_liouvillian(lam, tau_B, T2, T1, Omega, k=k))
        return bool(np.max(np.abs(ev.imag)) > 1e-9)

    lams = np.linspace(*lambda_range, n_lambda)
    flags = np.array([oscillates(l) for l in lams])
    if not flags[0] or flags.all():
        return float("nan")
    i = int(np.argmin(flags))           # first non-oscillating point
    lo, hi = lams[i - 1], lams[i]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if oscillates(mid) else (lo, mid)
    return float(0.5 * (lo + hi))


def tau_B_effective(c, tau_B: float, *, tau_B_min: float = 1e-3):
    """Bright dwell time left after removing a fraction ``c`` of it:
    ``tau_B_eff = (1 - c) * tau_B`` (``c = tau_B_comp / tau_B`` with
    ``tau_B_comp = c * tau_B`` the compressed-away dwell time), floored at
    ``tau_B_min`` so the generator stays finite at ``c = 1``."""
    return np.maximum((1.0 - np.asarray(c, dtype=float)) * tau_B, tau_B_min)


def theory_transition_vs_compression(
    compress_coeffs,
    *,
    tau_B: float,
    T1: float,
    T2: float,
    Omega: float,
    k: float = 0.0,
    **kwargs,
) -> np.ndarray:
    """:func:`theory_transition` at ``tau_B_effective(c, tau_B)`` for every ``c``."""
    return np.array([
        theory_transition(tau_B=float(tb), T1=T1, T2=T2, Omega=Omega, k=k, **kwargs)
        for tb in np.atleast_1d(tau_B_effective(compress_coeffs, tau_B))
    ])


# ---------------------------------------------------------------------------
# Synthetic ensemble-average data from the same generator
# ---------------------------------------------------------------------------
#
# State vector of ``build_liouvillian``:  x = (P_B,  P_G + P_D,  y,  P_D - P_G)
# with y the D-G coherence, so P_D = (x[1] + x[2]) / 2 ... see _populations.
# Initial state |D>: x0 = (0, 1, 0, 1).

_X0_D = np.array([0.0, 1.0, 0.0, 1.0])


def simulate_populations(
    lam: float, t_us: np.ndarray, *, tau_B: float, T1: float, T2: float,
    Omega: float, k: float = 0.0,
) -> dict[str, np.ndarray]:
    """``{"B", "G", "D"}`` populations of the 4×4 generator started in |D⟩,
    sampled at ``t_us`` (µs).  Exact propagation: ``x(t) = expm(L t) x0``."""
    from scipy.linalg import expm

    L = build_liouvillian(lam, tau_B, T2, T1, Omega, k=k)
    t = np.asarray(t_us, dtype=float)
    X = np.array([expm(L * ti) @ _X0_D for ti in t]).real   # (n_t, 4)
    P_B, s, z = X[:, 0], X[:, 1], X[:, 3]
    return {"B": P_B, "G": 0.5 * (s - z), "D": 0.5 * (s + z)}


def synthetic_records(
    lambda_values, t_us, *, tau_B: float, T1: float, T2: float, Omega: float,
    click_key: str = "click_probability", k: float = 0.0,
):
    """``ObservableRecord`` list mimicking an ens_avg scan of the generator.

    Per λ: ``"D"``, ``"G"``, ``"B"`` populations and ``click_key`` (= P_B,
    the generator's own dead-time fraction), all on the given time grid, so
    the records can be pushed through ``DeadTimeSweepPipeline`` exactly like
    the experimental ones.  No bootstrap statistics (the data are exact).
    """
    from mpm_analysis.data_types.observables import ObservableRecord

    t = np.asarray(t_us, dtype=float)
    out = []
    for lam in np.asarray(lambda_values, dtype=float):
        pops = simulate_populations(lam, t, tau_B=tau_B, T1=T1, T2=T2, Omega=Omega, k=k)
        meta = dict(omega_DG=float(Omega), source="4x4 Liouvillian", tau_B=tau_B,
                    T1=T1, T2=T2, k=k)
        for key in ("D", "G", "B"):
            out.append(ObservableRecord(float(lam), t, pops[key], key,
                                        source_file="synthetic", metadata=meta))
        out.append(ObservableRecord(float(lam), t, pops["B"], click_key,
                                    source_file="synthetic", metadata=meta))
    return out
