"""
postselected_dynamics.py
------------------------
Analytical forward simulator for the post-selected quantum Zeno dynamics.

This is a clean re-implementation of the ``PostSelectedDynamicsSimulator``
class from ``Code/postselected_dynamics_analytical_simulation.py``.

Key differences from the original
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``sympy`` is imported lazily (inside ``_build_matrix``) so the class can be
  imported without triggering a multi-second SymPy load.
* ``to_observable_records()`` converts simulation output to ``ObservableRecord``
  objects that feed directly into the analysis pipeline.
* Plotting methods are removed — use ``mpm_analysis.plotting`` instead.
* ``save_npz()`` saves in a format compatible with ``io/simulation.py``.

Physics
~~~~~~~
The 3×3 effective master equation for the post-selected Bloch vector
components (ρ_D, x, z) has the EOM matrix:

    M(α) = [
        [-α/2,                        0,                        α/2  ],
        [0,     -α/2 - 2/T_D - 1/(2T₁),                     -ω_S   ],
        [α/2 - 1/T₁,                 ω_S,    -α/2 - 1/T₁          ],
    ]
"""
from __future__ import annotations

import numpy as np

from mpm_analysis.data_types.observables import ObservableRecord


class PostSelectedDynamicsSimulator:
    """Simulate post-selected quantum Zeno dynamics analytically.

    Parameters
    ----------
    T_D:
        Pure dephasing time (µs).
    T1:
        Energy relaxation time T₁ (µs).
    T_int:
        Measurement interval / time step (µs).
    simulation_duration:
        Total simulation time (µs).
    min_alpha, max_alpha:
        Drive amplitude α range (rad/µs ≡ 2π·MHz).
    num_lambdas:
        Number of λ points in the scan.
    omega_S:
        Sensor Rabi frequency Ω_S (rad/µs).
    """

    def __init__(
        self,
        T_D: float = 21.0,
        T1: float = 66.0,
        T_int: float = 0.32,
        simulation_duration: float = 25.0,
        min_alpha: float = 0.05,
        max_alpha: float = 2.5,
        num_lambdas: int = 80,
        omega_S: float = 0.628,
    ) -> None:
        self.T_D = T_D
        self.T1 = T1
        self.omega_S = omega_S

        self.alpha_vals = np.linspace(min_alpha, max_alpha, num_lambdas)
        self.lambda_vals = self.alpha_vals / (2.0 * omega_S)
        self.t_vals = np.arange(0.0, simulation_duration, T_int)

        # Symbolic matrix — built lazily
        self._eom_matrix = None
        self._alpha_sym = None

    # ------------------------------------------------------------------
    # Symbolic matrix (lazy, deferred sympy import)
    # ------------------------------------------------------------------

    def _build_matrix(self):
        """Build the symbolic 3×3 EOM matrix (imports sympy on first call)."""
        import sympy as sp  
        alpha = sp.symbols("alpha", real=True)
        T_D, T1, omega_S = self.T_D, self.T1, self.omega_S
        M = sp.Matrix([
            [-alpha / 2, 0, alpha / 2],
            [0, -alpha / 2 - 2 / T_D - 1 / (2 * T1), -omega_S],
            [alpha / 2 - 1 / T1, omega_S, -alpha / 2 - 1 / T1],
        ])
        self._alpha_sym = alpha
        self._eom_matrix = M

    @property
    def eom_matrix(self):
        if self._eom_matrix is None:
            self._build_matrix()
        return self._eom_matrix

    # ------------------------------------------------------------------
    # Eigenvalue computation
    # ------------------------------------------------------------------

    def compute_eigenvalues(self, alpha_arr: np.ndarray | None = None) -> np.ndarray:
        """Compute the 3 complex eigenvalues across a range of alpha values.

        Parameters
        ----------
        alpha_arr:
            Alpha values to evaluate at.  Default: ``self.alpha_vals``.

        Returns
        -------
        np.ndarray
            Complex eigenvalue array, shape (n_alpha, 3).  Sorted by real part
            (most negative first).
        """
        if alpha_arr is None:
            alpha_arr = self.alpha_vals

        eigs = []
        for a in alpha_arr:
            M_num = np.array(
                self.eom_matrix.subs(self._alpha_sym, float(a))
            ).astype(np.complex128)
            ev = np.linalg.eigvals(M_num)
            ev = ev[np.argsort(ev.real)]
            eigs.append(ev)
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
            [ρ_D(0), x(0), z(0)].  Default: ``[1, 0, 1]`` (matches reference: a0=1, x0=0, z0=1).
        alpha_arr:
            Override alpha sweep.

        Returns
        -------
        list[dict]
            One dict per lambda slice with keys:
            ``lambda_val``, ``t``, ``survival``, ``x_t``, ``z_t``, ``eigenvalues``.
        """
        if alpha_arr is None:
            alpha_arr = self.alpha_vals
        if initial_state is None:
            initial_state = np.array([1.0, 0.0, 1.0], dtype=float)

        lambda_vals = alpha_arr / (2.0 * self.omega_S)
        results = []

        for alpha, lam in zip(alpha_arr, lambda_vals):
            M_num = np.array(
                self.eom_matrix.subs(self._alpha_sym, float(alpha))
            ).astype(np.complex128)

            evals, evecs = np.linalg.eig(M_num)
            sort_idx = np.argsort(evals.real)
            evals = evals[sort_idx]
            evecs = evecs[:, sort_idx]

            # Decompose initial state in eigenbasis
            coeffs = np.linalg.solve(evecs, initial_state.astype(complex))

            # Time evolution
            t = self.t_vals
            state_t = np.zeros((3, len(t)), dtype=complex)
            for k in range(3):
                state_t += coeffs[k] * np.outer(evecs[:, k], np.exp(evals[k] * t))

            survival = np.real(state_t[0])  # ρ_D = survival probability

            results.append({
                "lambda_val": float(lam),
                "t": t.copy(),
                "survival": np.clip(survival, 0.0, 1.0),
                "x_t": np.real(state_t[1]),
                "z_t": np.real(state_t[2]),
                "eigenvalues": evals,
            })

        return results

    # ------------------------------------------------------------------
    # Adapter: produce ObservableRecord list
    # ------------------------------------------------------------------

    def to_observable_records(
        self,
        observable_key: str = "survival",
        initial_state: np.ndarray | None = None,
        alpha_arr: np.ndarray | None = None,
    ) -> list[ObservableRecord]:
        """Run simulation and return results as ``ObservableRecord`` objects.

        Parameters
        ----------
        observable_key:
            Which simulated quantity to return.  Supported:
            ``"survival"`` (ρ_D), ``"Z"`` (z_t), ``"x"`` (x_t).
        initial_state:
            Initial Bloch vector components [ρ_D, x, z].
        alpha_arr:
            Override the alpha sweep.

        Returns
        -------
        list[ObservableRecord]
        """
        _key_map = {"survival": "survival", "Z": "z_t", "x": "x_t"}
        if observable_key not in _key_map:
            raise ValueError(
                f"observable_key '{observable_key}' not supported. "
                f"Use one of {list(_key_map)}."
            )
        data_key = _key_map[observable_key]

        sim_results = self.simulate(initial_state=initial_state, alpha_arr=alpha_arr)
        records = []
        for r in sim_results:
            records.append(
                ObservableRecord(
                    lambda_val=r["lambda_val"],
                    t=r["t"],
                    signal=r[data_key],
                    observable_key=observable_key,
                    source_file="PostSelectedDynamicsSimulator",
                    metadata={
                        "T_D": self.T_D,
                        "T1": self.T1,
                        "omega_S": self.omega_S,
                        # key the analysis pipeline reads for normalization
                        "omega_DG": self.omega_S,
                    },
                )
            )
        return records

    # ------------------------------------------------------------------
    # Save to NPZ
    # ------------------------------------------------------------------

    def save_npz(self, path, initial_state: np.ndarray | None = None) -> None:
        """Run simulation and save to a ``.npz`` file.

        The file is compatible with ``io/simulation.load_analytical_npz()``.

        Parameters
        ----------
        path:
            Output file path.
        """
        from pathlib import Path
        from mpm_analysis.utils.windows_paths import win_long_path

        sim_results = self.simulate(initial_state=initial_state)
        n = len(sim_results)
        nt = len(self.t_vals)

        survival = np.zeros((n, nt))
        z_arr = np.zeros((n, nt))
        x_arr = np.zeros((n, nt))
        lams = np.zeros(n)

        for i, r in enumerate(sim_results):
            lams[i] = r["lambda_val"]
            survival[i] = r["survival"]
            z_arr[i] = r["z_t"]
            x_arr[i] = r["x_t"]

        path = Path(path)
        np.savez(
            win_long_path(path),
            lambda_values=lams,
            time_array_us=self.t_vals,
            D_probability=survival,
            Z=z_arr,
            x_t=x_arr,
        )
        print(f"Simulation saved: {path}")
