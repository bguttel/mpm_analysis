
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