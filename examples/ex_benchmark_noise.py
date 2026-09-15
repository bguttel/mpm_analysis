"""
ex_benchmark_noise.py
---------------------
Example: test analysis robustness by injecting shot noise on
analytical simulation data.

Runs BenchmarkPipeline with:
  - PostSelectedDynamicsSimulator on a lambda scan bracketing the EP
    (experimental-like grid: dt = 0.32 us, ~19 us records)
  - ShotNoise at the experimental S-channel statistics
    (navg * 6 = 180,000 attempts per time point)
  - soft-bounds sqrt fit (the hard +-10% box degenerates when b_guess = 0
    and every fit fails silently — see BenchmarkPipeline.run docstring)

Expected output:
  lambda_c bias and std across trials, plus a shot-count sweep.

NOTE: this is the quick smoke-test version.  The full noise-robustness
study (bias, error-bar coverage, all estimators, record-level MC,
ensemble average, experimental comparison) lives in
Code/mpm_benchmarks/noise_robustness/.
"""
import numpy as np
import matplotlib.pyplot as plt

from mpm_analysis.simulation.postselected_dynamics import PostSelectedDynamicsSimulator
from mpm_analysis.simulation.noise import ShotNoise
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep, RefineStep
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline

N_TRIALS = 10
N_SHOTS = 180_000          # experimental S channel: navg(30k) * 6 settings

# ---- Simulation on an experiment-like grid ----
sim = PostSelectedDynamicsSimulator(
    num_lambdas=40,
    simulation_duration=19.2,   # 60 points of 0.32 us, like the scan
    T_int=0.32,
    min_alpha=0.3,              # lambda ~ 0.24 .. 1.9  (EP at ~1.32)
    max_alpha=2.4,
)
print("Generating clean simulation data...")
clean_records = sim.to_observable_records()
print(f"  {len(clean_records)} records")

# ---- Benchmark with shot noise at the experimental level ----
# seed=None: a fresh noise realization per trial.  A FIXED seed here makes
# every trial identical and the reported std collapses to exactly 0.
# RefineStep(conjugate_pairs=True) mirrors run.py; without it, noise keeps
# the imaginary splitting open past the EP and the sqrt fit pins at the
# upper lambda_c bound.
bench = BenchmarkPipeline(
    clean_records=clean_records,
    noise_models=[ShotNoise(n_shots=N_SHOTS, seed=None)],
    analysis_steps=[MPMStep(order=3), RefineStep(conjugate_pairs=True),
                    BootstrapStep(n_boot=150, seed=42)],
    n_trials=N_TRIALS,
)

print(f"\nRunning benchmark ({N_TRIALS} trials, n_shots={N_SHOTS:.0f})...")
result = bench.run(lambda_c_guess=1.25, soft_bounds=True,
                   lambda_c_bounds=(1.0, 1.5), plot=True)

print("\nResults:")
print(f"  Ground-truth lam_c: {result.ground_truth_lambda_c:.4f}")
print(f"  Extracted samples : {np.round(result.extracted_lambda_c_samples, 4)}")
print(f"  Bias              : {result.lambda_c_bias:+.4f}")
print(f"  Std               : {result.lambda_c_std:.4f}")

# ---- Noise sweep: 10x fewer / experimental / 10x more shots ----
sweep_shots = [N_SHOTS / 10, N_SHOTS, N_SHOTS * 10]
print(f"\nRunning noise sweep (n_shots = {sweep_shots})...")
sweep_results = bench.sweep_noise(
    noise_factory=lambda n_shots: [ShotNoise(n_shots=int(n_shots), seed=None)],
    param_values=sweep_shots,
    lambda_c_guess=1.25, soft_bounds=True, lambda_c_bounds=(1.0, 1.5),
    param_label="n_shots", plot=True)

print("\nNoise sweep summary:")
for s, r in zip(sweep_shots, sweep_results):
    print(f"  n_shots={s:>9.0f}  bias={r.lambda_c_bias:+.4f}  "
          f"std={r.lambda_c_std:.4f}")

plt.show()
