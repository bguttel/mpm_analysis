"""
ensemble_average.py
-------------------
Analytical forward simulator for the ensemble-average dynamics.

"""
from __future__ import annotations

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord


class EnsembleAverageDynamics:
    """Simulate ensemble-average Zeno dynamics using a 4×4 EOM.

    Parameters
    ----------
    T_D:
        Pure dephasing time (µs).
    T1:
        Energy relaxation time T₁ (µs).
    T_B:
        Bath correlation / backaction time (µs).
    omega_S:
        Sensor Rabi frequency Ω_S (rad/µs).
    simulation_duration:
        Total simulation time (µs).
    T_int:
        Time step (µs).
    min_alpha, max_alpha, num_lambdas:
        Alpha sweep range.
    """

    def __init__(
        self,
        T_D: float = 21.0,
        T1: float = 66.0,
        T_B: float = 5.0,
        omega_S: float = 0.628,
        simulation_duration: float = 25.0,
        T_int: float = 0.32,
        min_alpha: float = 0.05,
        max_alpha: float = 2.5,
        num_lambdas: int = 80,
    ) -> None:
        self.T_D = T_D
        self.T1 = T1
        self.T_B = T_B
        self.omega_S = omega_S

        self.alpha_vals = np.linspace(min_alpha, max_alpha, num_lambdas)
        self.lambda_vals = self.alpha_vals / (2.0 * omega_S)
        self.t_vals = np.arange(0.0, simulation_duration, T_int)

        self._eom_matrix = None
        self._alpha_sym = None

    # ------------------------------------------------------------------
    # Symbolic matrix (lazy)
    # ------------------------------------------------------------------

    def _build_matrix(self):
        import sympy as sp  # deferred
        alpha = sp.symbols("alpha", real=True)
        T_D, T1, T_B, omega_S = self.T_D, self.T1, self.T_B, self.omega_S
        M = sp.Matrix([
            [-alpha/2, 1/T_B, 0, alpha/2],
            [alpha/2, -1/T_B, 0, -alpha/2],
            [0, 0, -alpha/2 - 1/T_D - 1/(2*T1), omega_S],
            [alpha/2 - 1/T1, -1/T_B, -omega_S, -alpha/2 - 1/T1]
        ])
        self._alpha_sym = alpha
        self._eom_matrix = M

    @property
    def eom_matrix(self):
        if self._eom_matrix is None:
            self._build_matrix()
        return self._eom_matrix

    def _numerical_matrix(self, alpha: float) -> np.ndarray:
        return np.array(
            self.eom_matrix.subs(self._alpha_sym, float(alpha))
        ).astype(np.complex128)

    # ------------------------------------------------------------------
    # Eigenvalue computation
    # ------------------------------------------------------------------

    def compute_eigenvalues(
        self,
        alpha_arr: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute the 4 complex eigenvalues across alpha.

        Parameters
        ----------
        alpha_arr:
            Alpha sweep.  Default: ``self.alpha_vals``.
        Returns
        -------
        np.ndarray
            Shape (n_alpha, 4), sorted by real part.
        """
        if alpha_arr is None:
            alpha_arr = self.alpha_vals

        eigs = []
        for a in alpha_arr:
            M = self._numerical_matrix(a)
            ev = np.linalg.eigvals(M)
            eigs.append(ev[np.argsort(20*ev.real+ev.imag)])
        return np.array(eigs, dtype=complex)

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def simulate(
        self,
        initial_state: np.ndarray | None = None,
        alpha_arr: np.ndarray | None = None,

    ) -> list[dict]:
        """Simulate time evolution for all alpha values.

        Parameters
        ----------
        initial_state:
            [p_S(0), p_B(0), x(0), z(0)] — qubit-manifold population,
            detector-state population, coherence, polarisation (the state
            ordering of the EOM matrix, cf. paper Eq. S33).  Default
            ``[1, 0, 0, 1]``: start in the dark excited state |1>.

        Returns
        -------
        list[dict]
            Keys: ``lambda_val``, ``t``, ``survival`` (= a_t), ``b_t``,
            ``x_t``, ``z_t``, ``eigenvalues``.
        """
        if alpha_arr is None:
            alpha_arr = self.alpha_vals
        if initial_state is None:
            initial_state = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)

        lambda_vals = alpha_arr / (2.0 * self.omega_S)
        results = []

        for alpha, lam in zip(alpha_arr, lambda_vals):
            M = self._numerical_matrix(alpha)

            evals, evecs = np.linalg.eig(M)
            sort_idx = np.argsort(20*evals.real+evals.imag)
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]

            coeffs = np.linalg.solve(evecs, initial_state.astype(complex))

            t = self.t_vals
            state_t = np.zeros((4, len(t)), dtype=complex)
            for k in range(4):
                state_t += coeffs[k] * np.outer(evecs[:, k], np.exp(evals[k] * t))

            results.append({
                "lambda_val": float(lam),
                "t": t.copy(),
                "survival": np.real(state_t[0]),   # p_S: qubit-manifold population
                "b_t": np.real(state_t[1]),        # p_B: detector-state population
                "x_t": np.real(state_t[2]),
                "z_t": np.real(state_t[3]),
                "eigenvalues": evals,
            })

        return results

    def populations(
        self,
        lambda_vals: np.ndarray | None = None,
        initial_state: np.ndarray | None = None,
    ) -> dict:
        """Three-level populations d, g, b vs time for each lambda.

        Derived from ``simulate``: the qubit manifold {|0>, |1>} has population
        ``p_S`` and polarisation ``z``, so ``d = (p_S + z)/2`` (excited, dark),
        ``g = (p_S - z)/2`` (ground, dark) and ``b = p_B`` (bright detector
        state).  ``d + g + b = 1``.

        Parameters
        ----------
        lambda_vals:
            Lambda sweep (``alpha = 2 lambda omega_S``).  Default:
            ``self.lambda_vals``.

        Returns
        -------
        dict
            ``lambda_vals`` (n,), ``t`` (nt,), ``d``, ``g``, ``b`` (n, nt).
        """
        if lambda_vals is None:
            lambda_vals = self.lambda_vals
        lambda_vals = np.asarray(lambda_vals, dtype=float)
        res = self.simulate(initial_state=initial_state,
                            alpha_arr=2.0 * self.omega_S * lambda_vals)
        p_s = np.array([r["survival"] for r in res])
        z = np.array([r["z_t"] for r in res])
        b = np.array([r["b_t"] for r in res])
        return {
            "lambda_vals": lambda_vals,
            "t": self.t_vals.copy(),
            "d": 0.5 * (p_s + z),
            "g": 0.5 * (p_s - z),
            "b": b,
        }

    # ------------------------------------------------------------------
    # Adapter
    # ------------------------------------------------------------------

    def to_observable_records(
        self,
        observable_key: str = "survival",
        initial_state: np.ndarray | None = None,
        alpha_arr: np.ndarray | None = None,
    ) -> list[ObservableRecord]:
        """Run simulation and return as ``ObservableRecord`` objects."""
        _key_map = {
            "survival": "survival",
            "Z": "z_t",
            "x": "x_t",
            "b": "b_t",
        }
        if observable_key not in _key_map:
            raise ValueError(f"observable_key '{observable_key}' not supported.")
        data_key = _key_map[observable_key]

        sim_results = self.simulate(
            initial_state=initial_state,
            alpha_arr=alpha_arr
        )
        records = []
        for r in sim_results:
            records.append(
                ObservableRecord(
                    lambda_val=r["lambda_val"],
                    t=r["t"],
                    signal=r[data_key],
                    observable_key=observable_key,
                    source_file="EnsembleAverageDynamics",
                    metadata={
                        "T_D": self.T_D,
                        "T1": self.T1,
                        "T_B": self.T_B,
                        "omega_S": self.omega_S,
                        # key the analysis pipeline reads for normalization
                        "omega_DG": self.omega_S,
                    },
                )
            )
        return records

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_eigenvalues(
        self,
        alpha_arr: np.ndarray | None = None,
        ax_real=None,
        ax_imag=None,
    ):
        """Plot real and imaginary parts of all 4 eigenvalues vs lambda.

        Example
        -------
        ::

            from mpm_analysis.simulation.ensemble_average import EnsembleAverageDynamics
            import matplotlib.pyplot as plt

            sim = EnsembleAverageDynamics(num_lambdas=100)
            ax_r, ax_i = sim.plot_eigenvalues()
            plt.tight_layout()
            plt.show()


        Parameters
        ----------
        alpha_arr:
            Alpha sweep.  Default: ``self.alpha_vals``.
        ax_real, ax_imag:
            Axes for the real and imaginary panels.  If ``None``, a new
            figure with two side-by-side subplots is created.

        Returns
        -------
        (ax_real, ax_imag)
        """
        import matplotlib.pyplot as plt

        eigs = self.compute_eigenvalues(alpha_arr=alpha_arr)
        lam = (alpha_arr if alpha_arr is not None else self.alpha_vals) / (2.0 * self.omega_S)

        if ax_real is None or ax_imag is None:
            _, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(10, 4))

        for i in range(eigs.shape[1]):
            ax_real.plot(lam, eigs[:, i].real, label=f"eig {i}")
            ax_imag.plot(lam, eigs[:, i].imag, label=f"eig {i}")

        ax_real.set_xlabel(r"$\lambda$")
        ax_real.set_ylabel(r"Re(eigenvalue)")
        ax_real.legend(fontsize=8)

        ax_imag.set_xlabel(r"$\lambda$")
        ax_imag.set_ylabel(r"Im(eigenvalue)")
        ax_imag.legend(fontsize=8)

        return ax_real, ax_imag

    # ------------------------------------------------------------------
    # Save to NPZ
    # ------------------------------------------------------------------

    def save_npz(self, path, initial_state: np.ndarray | None = None) -> None:
        """Save simulation output as NPZ (compatible with ``io/simulation.py``)."""
        from pathlib import Path
        from mpm_analysis.utils.windows_paths import win_long_path

        sim_results = self.simulate(initial_state=initial_state)
        n = len(sim_results)
        nt = len(self.t_vals)
        survival = np.zeros((n, nt))
        z_arr = np.zeros((n, nt))
        lams = np.zeros(n)

        for i, r in enumerate(sim_results):
            lams[i] = r["lambda_val"]
            survival[i] = r["survival"]
            z_arr[i] = r["z_t"]

        path = Path(path)
        np.savez(
            win_long_path(path),
            lambda_values=lams,
            time_array_us=self.t_vals,
            D_probability=survival,
            Z=z_arr,
        )
        print(f"Simulation saved: {path}")
import matplotlib.pyplot as plt
if __name__ == "__main__":
    sim = EnsembleAverageDynamics(num_lambdas=10)
    ax_r, ax_i = sim.plot_eigenvalues()
    plt.tight_layout()
    plt.show()