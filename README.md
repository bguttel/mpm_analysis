# mpm_analysis

Analysis package: loading experimental data, extracting poles via the Matrix
Pencil Method (MPM), bootstrap uncertainty quantification, critical-point
fitting, and simulation benchmarks.

---

## Quick start

```bash
# Install
pip install -e .

# Run everything (toggle tasks in run.py first)
python run.py

# Or run a self-contained test on synthetic data (no real data needed)
python examples/ex_run_bootstrap_small.py
```

---

## Multi-machine path setup

Dropbox paths differ per machine.  `run.py` auto-detects the correct root
using the hostname.  The mapping lives in
`src/mpm_analysis/utils/lab_paths.py` and mirrors `ZenoPlotting/base.py`.

**Currently registered machines:**

| Hostname | User | Drive |
|----------|------|-------|
| `DESKTOP-BAEVOT2` | Noam Nezer | C: |
| `DESKTOP-FQN6OUI` | Danielle Gov | C: |
| `CMD-SRStudent1` | Barkay Guttel | E: |
| `cmd-SRLenovoPc2` | Serge Rosenblum | E: |

**Adding a new computer:**

1. Find your hostname: `python -c "import socket; print(socket.gethostname())"`
2. Add an entry to `_DROPBOX_ROOTS` in `src/mpm_analysis/utils/lab_paths.py`.
.

If your hostname is not registered you will get a clear `ValueError` when
`run.py` starts — not a silent wrong-path failure.

---

## No silent defaults — fail-loud policy

This package never guesses unknown inputs or substitutes placeholders.
If a required value is missing (e.g. `lambda_c_guess`, `observable_key`),
the code raises a `ValueError` or `KeyError` immediately.

---

## Package structure

```
mpm_analysis/
├── run.py                   ← Main orchestrator (toggleable tasks, edit paths here)
├── examples/                ← Small runnable demos per module
│   ├── ex_run_bootstrap_small.py
│   ├── ex_load_experimental.py
│   ├── ex_critical_point_fit.py
│   ├── ex_benchmark_noise.py
│   └── ex_paper_figure.py
├── docs/
│   └── adding_a_new_json_format.md  ← How to plug in a new data source
└── src/mpm_analysis/
    ├── data_types/          ← Typed containers (no I/O, no heavy deps)
    ├── io/                  ← Load/save: JSON, NPZ, experimental folders
    ├── analysis/            ← Numerical core: MPM, RRHA, bootstrap, ksi, guesses
    ├── simulation/          ← Forward simulators + noise models
    ├── pipeline/            ← High-level orchestrators
    ├── plotting/            ← Exploratory + paper figures
    ├── diagnostics/         ← Parity tests and written reports
    └── utils/               ← Path helpers, physics constants, lab paths
```

---

## Data flow

```
Experimental data           Simulation
(JSON files / NPZ)          (PostSelectedDynamicsSimulator /
       |                     EnsembleAverageDynamics)
       |                           |
       v                           v
  io/experimental/          simulation/*.py
  EnsembleAverageLoader  ─────────────────────>  list[ObservableRecord]
  PostSelectedLoader                                      |
  (or your custom loader)                                 |
                           ┌──────────────────────────────┘
                           v
                    MPMPipeline.run()
                           |
                    ┌──────┴──────────────┐
                    v                     v
          MPMStep / MPMRRHAStep    BootstrapStep
          (raw poles per λ)        (residual resampling)
                    └──────┬──────────────┘
                           v
                   ZenoAnalysisResult
                   ├── lambda_values
                   └── poles (PolesResult)
                           ├── decay_rates / _lower / _upper  (n_lambda, order)
                           ├── frequencies / _lower / _upper
                           ├── imag_splitting / real_splitting (n_lambda,)
                           └── raw_samples (PoleSamples | None)
                                   |
                    ┌──────────────┴──────────────┐
                    v                             v
        CriticalPointPipeline.fit()     KsiCrossingPipeline.run()
                    |                             |
                    v                             v
             SqrtFitResult                KsiCrossingResult
             └── lambda_c ± err           └── zero_crossings
```

---

## Data types

### `ObservableRecord` — one lambda, one observable

```
ObservableRecord
├── lambda_val: float          dimensionless Zeno parameter λ
├── t: np.ndarray              time axis in µs
├── signal: np.ndarray         measured signal (same length as t)
├── observable_key: str        "survival", "Z", "G_probability", ...
├── source_file: str           original filename (for tracing)
└── metadata: dict             omega_DG, navg_repeat_experiment, scan_number, ...
```

### `PolesResult` — bootstrap-aggregated pole spectrum

All uncertainty fields are **half-widths** (distance from median to the 16th /
84th percentile), not absolute percentile values.

```
PolesResult
├── decay_rates:          np.ndarray   (n_lambda, order) — median
├── decay_rates_lower:    np.ndarray   (n_lambda, order) — half-width to 16th pct
├── decay_rates_upper:    np.ndarray   (n_lambda, order) — half-width to 84th pct
├── frequencies:          np.ndarray   (n_lambda, order)
├── frequencies_lower:    np.ndarray
├── frequencies_upper:    np.ndarray
├── imag_splitting:       np.ndarray | None   (n_lambda,) — |Im(p1-p0)|/2
├── imag_splitting_lower: np.ndarray | None
├── imag_splitting_upper: np.ndarray | None
├── real_splitting:       np.ndarray | None   (n_lambda,) — |Re(p1-p0)|/2
├── real_splitting_lower: np.ndarray | None
├── real_splitting_upper: np.ndarray | None
└── raw_samples:          PoleSamples | None  — (n_lambda, n_boot, order) complex

# Slicing propagates to all arrays:
subset = poles[mask]        # PolesResult with all fields filtered
```

### `ZenoAnalysisResult` — complete pipeline output

```
ZenoAnalysisResult
├── lambda_values: np.ndarray           (n_lambda,)
├── poles: PolesResult
├── critical_point_sqrt_fit: SqrtFitResult | None
├── metadata: dict
└── parameters: dict
```

### `PoleSamples` — raw bootstrap draws

```python
from mpm_analysis.data_types import PoleSamples

ps = result.poles.raw_samples   # populated by BootstrapStep
ps.samples                      # complex, shape (n_lambda, n_boot, order)
ps.lambda_values                # float, shape (n_lambda,)
ps.get_lambda(i)                # (n_boot, order) slice for lambda index i
```

---

## Pole sorting — canonical rule

All producers (MPMStep, MPMRRHAStep, bootstrap inner loop) sort poles by
**real part ascending** (ties broken by imaginary part ascending).

```python
from mpm_analysis.analysis.pole_sorting import sort_poles_canonical

sorted_poles = sort_poles_canonical(poles)  # shape (order,)
# poles[0]: most negative real (fastest decay / EP pole)
# poles[1]: second-most negative  (other EP pole)
# poles[2]: least negative    (slow / non-EP pole)
```

Splittings use poles 0 and 1 (the EP pair):
- `imag_splitting = |Im(p1 - p0)| / 2`  — non-zero in Zeno regime (λ < λ_c)
- `real_splitting = |Re(p1 - p0)| / 2`  — non-zero in classical regime (λ > λ_c)

---

## Loading experimental data

### Ensemble-average (`scan_experiment_results_50`)

```python
from mpm_analysis.io.experimental import EnsembleAverageLoader

loader = EnsembleAverageLoader()
records = loader.load(folder, observables=["survival", "Z"])
# "survival" <- data.D_probability
# "Z"        <- data.D_probability - data.G_probability
```

### Post-selected (`click_record_with_post_selected_tomography`)

```python
from mpm_analysis.io.experimental import PostSelectedLoader

loader = PostSelectedLoader()
records = loader.load(folder, observables=["survival"])
```

Two additional observables load the **conditioned raw-record survivals** that
`run_files/raw_data_tests/run_conditional_p0.py` writes back into the scan
JSONs (p0 conditioned on the preceding segment's outcome):

```python
records = loader.load(folder, observables=["S_prev_click", "S_prev_noclick"])
# <- data.p_0_t_array_prev_click / data.p_0_t_array_prev_noclick;
#    data.p_0_t_conditional_meta records the click rule (condition_on) and
#    the per-point class counts.  Fit from run.py by setting
#    OBSERVABLES = ["S_prev_click"] (etc.).  The data bootstrap resamples
#    these channels binomially on their own per-point class counts, so any
#    UNCERTAINTY_SOURCE works.
```

Files that never ran the write-back lack the keys; the loader warns and
skips the observable.

### Adding a new observable to an existing loader

```python
loader.add_extractor(
    "my_observable",
    lambda jd, t_us: np.asarray(jd["data"]["my_key"], dtype=float),
)
records = loader.load(folder, observables=["survival", "my_observable"])
```

### Adding a completely new JSON format

See **[docs/adding_a_new_json_format.md](docs/adding_a_new_json_format.md)**
for a step-by-step guide with a worked example.

### k-space pole analysis (`click_record_repeated` scans)

The k-space ("FDG transition") pole analysis has **one pipeline** and **two run
files**, with a simple source selector:

- `source="json"` — read `P(k,t)` directly from the experiment scan JSONs
  (`KSpaceJSONLoader`).
- `source="compressed"` — read the compact compressed-click records
  (`CompressedClickKSpaceLoader`).

Both feed the same `KSpaceAnalysis` → `KSpacePolesResult` via
`pipeline.kspace_pipeline.KSpacePipeline`. Pick the source with
`get_kspace_loader(source)` (registry in `kspace/loader.py`).

**1. Preprocessing** (`run_kspace_preprocessing.py`) — build the compressed
records from the raw `individual_runs/` (the raw `assigned_states` is ~69 GB and
lives only there). One-time. `MIN_DARK=2` applies the stricter "2-click"
(`1→00`, `101`=false-negative) rule. Writes to
`<scan_folder>/compressed_click_analysis/` (timestamped, non-clobbering) with
`P(n,t)` histograms + per-λ summary figures.

**2. Analysis** (`run_kspace_analysis.py`) — the single run file. Set `SOURCE`
and the parameters; it loads → analyzes → saves `KSpacePolesResult` → saves the
full figure set (eigenvalues-vs-λ at k≈π, poles-vs-k decay/freq, parametric, SVD
spectrum, raw k-space traces + HTML slider, 3-D grids + HTML, braiding-ν).
Options:

- `BOOTSTRAP_OVER_RUNS=True` (compressed only) — run one pole analysis per
  individual run → per-pole error bars (`KSpacePolesResult.poles_err`).
- `THEORY={…}` — overlay theory poles + braiding from
  `Theory/fake_lindbladian_braid_fixed_2.py` (reused, not duplicated;
  `analysis/braiding.py`). Manual params for now (Ω from data; T1/T2/τ_B/k_fp set
  in the run file).
- `COMPUTE_BRAIDING=True` — braiding characterization ν(λ) of the data.
- Figures use the `Agg` backend and are always saved; `OPEN_FIGURES=True` opens
  them (PNG via the OS viewer, HTML in the browser) after the run.

Programmatic use:

```python
from mpm_analysis.pipeline.kspace_pipeline import KSpacePipeline

out = KSpacePipeline().run(
    folder, source="compressed",          # or "json"
    bootstrap_over_runs=False,            # per-run pole error bars (compressed)
    theory=dict(model="3x3", T1=40, T2=20, tau_B=None, k_fp=0.0),  # or None
    braiding=True, save_figures=True, open_figures=False,
)
```

Full schema, compression algorithm, and the re-compression recipe are in
**[docs/compressed_click_record.md](docs/compressed_click_record.md)**.

### Segmented click records (`click_record_with_post_selected_tomography`)

The scan-10 NPZs keep the shot-resolved fast-measurement record in
*segments*: every point of the 60-point time axis is measured by its own run —
segment `i` (`i = 1…59`) evolves till `t_i = i·dt` and records `i` bins,
concatenated in ascending order.  `ClickRecordNpzLoader` detects the 3-D
`assigned_states` layout automatically:

```python
from mpm_analysis.io.experimental import ClickRecordNpzLoader

loader = ClickRecordNpzLoader()
for rec in loader.iter_records(folder):   # one ~300 MB record at a time
    rec.trajectories        # bool (180000, 1770): reps x settings flattened
                            # into rows (row = rep*n_axes + axis), segments
                            # concatenated along the columns
    rec.segment_lengths     # [1, 2, ..., 59]
    rec.segment(i)          # (180000, i+1) view of stored segment i
```

These trajectories are the thresholded bright/dark assignments **verbatim** —
a bright bin is itself the click in this scan's `no_clicks_count` convention,
so no falling-edge conversion is applied (unlike the flat
`click_record_repeated` telegraph records, which the same loader still
converts to falling-edge clicks).

### Raw-data bias tests (`run_files/raw_data_tests/`)

`run_conditional_p0.py` (IDE toggle block, same style as `run.py`) assembles
the raw `p_0(t)` **conditioned on whether the preceding segment clicked** (and
the opposite), prints and saves the per-λ percentage breakdown of the two
classes, and runs a matrix pencil on each conditional curve (plus the
unconditional baseline on the same time window).  It targets two raw-data
biases that push in opposite directions:

* **initialisation bias** — a failing π-pulse leaves the wrong initial state,
  ending dark more often → extra clicks at segment start;
* **Schmitt-trigger carry-over** — the hysteresis filter is seeded by the
  preceding segment's *last* detector outcome (`CONDITION_ON = "last_bin"`
  probes exactly this mechanism; `"any_click"` is the scan's own click rule).

Outputs — breakdown CSV, stacked conditional-curves NPZ, figures
(conditional `p_0(t)` panels with the PS-generator theory overlay, breakdown
vs t and vs λ, per-λ conditional click-probability differences at the first
bins, per-condition MPM spectra, strand-wise decay comparison), interactive
p₀ and click-rate (−dp₀/dt) sliders, and one standard analysis JSON per
condition — go to `run_files/raw_data_tests/output/`.

With `WRITE_JSON_ARRAYS = True` (default) the run also writes both
conditional curves **into the scan JSONs themselves** —
`data.p_0_t_array_prev_click`, `data.p_0_t_array_prev_noclick` and
`data.p_0_t_conditional_meta` (click rule + per-point class counts),
inserted right after `p_0_t_array` with everything else preserved
byte-for-byte (same mechanism as the `p_0_t_array_corrected` precedent;
re-runs replace the three keys).  After that, the ordinary `run.py` fits
them directly: `OBSERVABLES = ["S_prev_click"]` / `["S_prev_noclick"]` with
the `post_selected` data mode (use a non-data uncertainty source — see the
loader section above).

### Scan number

The loaders automatically parse a trailing integer from the folder name (e.g.
`scan_experiment_results_50` → `scan_number=50`) and attach it to each
record's metadata.

```python
from mpm_analysis.io.file_naming import parse_result_filename
info = parse_result_filename("mpm_ens_avg_scan50_N50_lam0.40-1.60_20260410_143022.json")
# {'analysis_type': 'mpm', 'source_tag': 'ens_avg', 'scan_number': 50, ...}
```

---

## Running the analysis pipeline

### Typical usage (one call)

```python
from mpm_analysis.pipeline.mpm_pipeline import MPMPipeline

result = MPMPipeline.from_ensemble_average(folder, n_boot=2000).run()
```

### With explicit steps — standard MPM

```python
from mpm_analysis.analysis.steps import MPMStep, BootstrapStep

steps = [
    MPMStep(order=3),
    BootstrapStep(n_boot=2000, seed=42),
]
result = MPMPipeline.from_ensemble_average(folder, steps=steps).run()
```

### With RRHA-enhanced MPM (better noise robustness)

```python
from mpm_analysis.analysis.steps import MPMRRHAStep, BootstrapStep

steps = [
    MPMRRHAStep(order=3, max_iter=5),
    BootstrapStep(n_boot=2000, seed=42),
]
result = MPMPipeline.from_ensemble_average(folder, steps=steps).run()
```

`MPMRRHAStep` is a drop-in replacement for `MPMStep`.  It uses iterative
SVD-reduction + Hankel projection to suppress noise before the pencil solve.
Benchmark shows ~1.56× lower real-part error near the EP under shot noise
(see `diagnostics/SHOT_NOISE_REPORT.md`).

### From a pre-built record list (custom loaders)

```python
result = MPMPipeline.from_records(records, steps=steps).run()
```

---

## Critical-point fitting

### With a preset (recommended)

```python
from mpm_analysis.pipeline.critical_point_pipeline import CriticalPointPipeline

cp = CriticalPointPipeline.from_json("result.json", lambda_max=1.55)
fit = cp.fit(preset="ens_avg_experimental", window_left=50, window_right=15)
print(f"lambda_c = {fit.lambda_c:.4f} +/- {fit.lambda_c_err:.4f}")
```

### Available presets

```python
from mpm_analysis.analysis.critical_point_guesses import get_guess, list_presets

list_presets()
# ['post_selected_experimental', 'ens_avg_experimental', 'post_selected_simulation']
```

| Preset | λ_c | a | b | Use for |
|--------|-----|---|---|---------|
| `"post_selected_experimental"` | 1.0 | 1.12 | 0.5 | First transition, post-selected lab data |
| `"ens_avg_experimental"` | 1.1 | 0.62 | −0.15 | Third transition, ensemble-average lab data |
| `"post_selected_simulation"` | 1.3 | 1.0 | 0.0 | Analytical simulator |

### With explicit guesses

```python
fit = cp.fit(lambda_c_guess=1.1, a_guess=0.62, b_guess=-0.15)
```

The sqrt model is fitted jointly to both sides of the EP:

```
Im(Δe₁₂) = a·√|λ−λ_c| + b·(λ−λ_c)^1.5   (λ < λ_c, imaginary splitting)
Re(Δe₁₂) = a·√|λ−λ_c| + b·(λ−λ_c)^1.5   (λ > λ_c, real splitting)
```

`a` is constrained ≥ 0 (physically required); `λ_c` and `b` are unconstrained.

---

## The two transitions: state freezing (ksi = 0) and the exceptional point (Delta = 0)

`ksi = (2*Re(e3) - Re(e2)) / (Re(e2) - e3)` and the pair discriminant
`Delta = c1**2 - 4*c0` mark **different** transitions — on the post-selected
scan they sit ~0.14 apart in lambda.  Never label the ksi crossing as the EP.

```python
from mpm_analysis.analysis.ksi_crossing import (
    KsiCrossingPipeline, discriminant_crossing_samples, linfit_zero_crossing,
)
from mpm_analysis.plotting.ksi_plot import plot_ksi_curve, plot_discriminant_curve

# --- state freezing: the zero of ksi ---
ksi_result = KsiCrossingPipeline.from_result(result, lambda_max=1.55).run()
print(f"state freezing: lambda(ksi=0) = {ksi_result.lambda_ksi_zero:.4f}")
plot_ksi_curve(ksi_result)

# --- exceptional point: the zero of the discriminant ---
# The line fit is redone inside every bootstrap realization; the 16/84 spread
# of those per-realization fits is the error bar.
raw = result.poles.raw_samples
cross, d_med, d_lo, d_hi = discriminant_crossing_samples(
    raw, lambda_max=1.55, method="linfit", half_width=0.15)
q16, q50, q84 = np.nanpercentile(cross, [16, 50, 84])
lam_ep, fit = linfit_zero_crossing(raw.lambda_values, d_med,
                                   half_width=0.15, return_fit=True)
print(f"exceptional point: lambda_EP = {lam_ep:.4f} +/- {(q84 - q16) / 2:.4f}")
plot_discriminant_curve(raw.lambda_values, d_med, fit=fit, lambda_ep=lam_ep,
                        crossing_quantiles=(q16, q50, q84),
                        delta_lower=d_lo, delta_upper=d_hi)
```

`run_files/run.py` does both in one task (`RUN_TRANSITIONS`), printing each
value with its bootstrap error and saving both figures.

---

## Detector dead-time compression sweep

Step-by-step walkthrough of the whole pipeline (inputs, remaps, both fit
routes, bootstrap, theory comparison, outputs, caveats):
**[docs/dead_time_sweep.md](docs/dead_time_sweep.md)**.

While the system occupies |B> the fast readout clicks and the dynamics of
interest is frozen — the detector is *dead*.  The ens_avg scans store the
ensemble |B> occupation as `data.click_probability` (the fraction of shots
whose fast readout at that bin returned bright; loaded as the observable
`"click_probability"`).  The sweep tests whether removing that dead time
moves the extracted transition by remapping **only the time axis** and
refitting the same modal form — the measured y-data are never modified:

| `remap`        | axis                                  | note |
|----------------|---------------------------------------|------|
| `"pointwise"`  | `t' = t·[1 − c·p_click(t)]`           | as specified; a jump `Δp` between bins `k`, `k+1` folds the axis once `c·(p + k·Δp) > 1`, so noisy late bins fold first — such traces are **truncated at the first fold** and the cut is printed |
| `"cumulative"` | `t' = t − c·∫₀ᵗ p_click`              | accumulated dead time; slope `1 − c·p > 0`, never folds |

`c = 0` is the raw data, `c = 1` full compression.  Two fitting routes:

* **direct** — the remapped axis is non-uniform, so instead of the pencil
  the same complex-exponential LSQ (`RefineStep` → `refine_poles`, conjugate
  pairs / real poles, decaying) is fitted directly on `t'`.  `c = 0` is
  seeded from the raw-data MPM poles (or from this run's MPM result when it
  describes the same data); every next `c` starts from the previous fit.
* **regrid_mpm** (fallback) — every remapped trace is interpolated back onto
  a uniform grid of the raw spacing (320 ns) and the ordinary `MPMStep` (+ the
  same `RefineStep`) runs on it; reported next to the direct route wherever
  both converge, Hungarian-matched to the direct branches.

From every fit the decay rates / frequencies and both transitions are read
off exactly as `run_transitions` does — the EP as the straight-line zero of
`Delta = c1² − 4c0` and the state-freezing point as the zero of `ksi`.
`DT_N_BOOTSTRAP > 0` adds the data bootstrap per `c` with **refine-only
draws** (`BootstrapStep(mode="data", pencil=False)`: every draw starts from
the clean poles, the pencil never sees the non-uniform axis) and stores the
per-realization EP quantiles.

```python
from mpm_analysis.io.experimental import EnsembleAverageLoader
from mpm_analysis.pipeline import DeadTimeSweepPipeline
from mpm_analysis.plotting.dead_time_plots import (
    plot_compression_traces, plot_poles_vs_compression, plot_transition_vs_compression,
)

records = EnsembleAverageLoader().load(folder, observables=["D", "click_probability"])
pipe = DeadTimeSweepPipeline(records, [0, 0.25, 0.5, 0.75, 1.0], order=3,
                             observable_keys=["D"], remap="pointwise",
                             discriminant_half_width=0.15, fallback=True, n_boot=0)
sweep = pipe.run()                 # DeadTimeSweepResult: .direct / .fallback tracks
pipe.save(output_dir)              # JSON (+ .csv summary, one row per c)
theory = dict(tau_B=4.0, T1=15.0, T2=21.0)        # µs; Omega from the data
plot_compression_traces(records, sweep, c_values=(0, 0.2, 0.4))   # data + fits
s1 = plot_traces_compression_slider(records, sweep)   # same rows, slider over c
plot_poles_at_compression(sweep, 0.2, theory_params=theory)   # 2-panel spectrum at one c
s2 = plot_poles_compression_slider(sweep, theory_params=theory)   # slider over c
plot_poles_panel_grid(sweep, theory_params=theory)   # slider frames as a static grid
export_poles_slider_html(sweep, "poles_slider.html", theory_params=theory)   # standalone HTML
plot_transition_vs_compression(sweep, theory_params=theory)   # Delta(λ) per c, λ_EP(c)
sweep.to_analysis_result(ic)               # one c as a ZenoAnalysisResult
```

**Theory comparison.**  `theory_params` selects the 4×4 Liouvillian of
`simulation/liouvillian_spectrum.py` at `k = 0` (the ensemble-average
generator) with the compression mapped onto its bright dwell time as
`tau_B_eff = (1 − c)·tau_B` — `c = tau_B_comp / tau_B`, `tau_B_comp = c·tau_B`
being the dwell time compressed away (`tau_B_effective`).
`theory_transition` locates the theory's exceptional point (coalescence of
the oscillating eigenvalue pair) and `theory_transition_vs_compression` gives
the line drawn in the transition figure; the per-c pole figures overlay the
theory eigenvalues and mark both the data's and the theory's `λ_EP`.  For
scan-50 (`Ω = 0.628 rad/µs`, `T1 = 15`, `T2 = 21`, `tau_B = 4 µs`) the theory
EP sits at 1.30 for the raw data and climbs towards the ideal `λ = 2` as the
dead time is removed.

In `run.py` set `RUN_DEAD_TIME_SWEEP = True` (block `DT_*`: data mode,
fitted keys, remap, coefficients, fallback, bootstrap size, representative
`c`/λ, `DT_THEORY`).  Results and the figures — `dead_time_traces`, one
`dead_time_poles_c<c>` spectrum per representative `c` (all poles per panel,
the same layout as the MPM `spectrum` figure), `dead_time_transition` — go to
`OUTPUT_DIR`; with `SHOW_PLOTS` two sliders over `c` open as well (traces and
spectrum; run from a terminal or a GUI backend — an inline backend renders
them as static images).  `io.analysis_json.load_dead_time_sweep` reads a
saved sweep back.

---

## Simulation benchmark

```python
from mpm_analysis.pipeline.benchmark_pipeline import BenchmarkPipeline
from mpm_analysis.simulation.noise import ShotNoise

bench = BenchmarkPipeline(
    clean_records=clean_records,
    noise_models=[ShotNoise(n_shots=1.8e5)],
    analysis_steps=[MPMStep(order=3), BootstrapStep(n_boot=500)],
    n_trials=20,
)
result = bench.run(preset="post_selected_simulation")
print(f"bias = {result.lambda_c_bias:.4f},  std = {result.lambda_c_std:.4f}")
```

### Available noise models

| Class | Parameters | Effect |
|-------|-----------|--------|
| `ShotNoise` | `n_shots`, `seed` | Binomial shot noise: σ = √(s(1−s)/n) |
| `ReadoutError` | `false_positive`, `false_negative` | P_meas = P·(1−fn) + (1−P)·fp |
| `FiniteSampling` | `n_shots`, `seed` | Same formula as ShotNoise |
| `CompositeNoise` | `[model1, model2, ...]` | Apply models in sequence |

---

## Lambda scale: which one to use?

| Data type | Lambda source | How |
|-----------|-------------|-----|
| `ens_avg` | `parameters.lambda_zeno_estimation` in JSON | Automatically read |
| `post_selected` (no CSV) | `parameters.lambda_zeno_estimation` in JSON | Fallback |
| `post_selected` (with CSV) | `Gamma_up_3_states_hmm / (2·Ω_DG·1e-6)` | Pass `csv_path=` to loader |

---

## Debugging tips

### 1. Use `run.py` with `DEBUG = True`

```python
DEBUG = True    # forces n_boot=50, max_files=10
```

### 2. Inspect intermediate state

```python
from mpm_analysis.analysis.steps import MPMStep
state = MPMStep(order=3).process(records, {})
print(state["poles_raw"][0])   # poles for first lambda
```

### 3. Inspect raw bootstrap samples

```python
ps = result.poles.raw_samples  # PoleSamples
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(ps.samples[10].real, ps.samples[10].imag, s=5)
plt.show()
```

### 4. Run examples

```bash
python examples/ex_run_bootstrap_small.py  # synthetic, ~15 s, no data needed
python examples/ex_load_experimental.py    # just loads + prints, no analysis
python examples/ex_critical_point_fit.py   # needs a saved JSON
python examples/ex_benchmark_noise.py      # noise sweep, ~1 min
```

### 5. Run shot-noise parity diagnostic

```bash
python -m mpm_analysis.diagnostics.shot_noise_parity
```

---

## File naming convention

Saved analysis files are auto-named:

```
mpm_ens_avg_scan50_N50_lam0.40-1.60_20260410_143022.json
     ^          ^   ^               ^
   type    scan no  n_lambda      timestamp
```

The `_scanNN_` segment is present only when a scan number was parsed from the
folder name.  Synthetic results omit it.

---

## Shot-noise parity

See `diagnostics/SHOT_NOISE_REPORT.md`.  Short version:

1. The old `object_oriented_poles.py` had shot-noise injection **commented out**.
   All previously published results were on clean noiseless signals.
2. The new pipeline correctly injects noise in benchmarks.
3. **RRHA suppresses noise significantly** near the EP (~1.56× lower error).
