# Compressed click records

Pre-process the raw per-shot click record of a `click_record_repeated` scan into
a compact, analysis-ready form: compressed click streams, full counting
statistics `P(n,t)`, the Fourier transform over the counting field `k`, and
inputs for correlations and the k-space pole analysis.

## Why

The raw thresholded record (`assigned_states`, bool, `N·T = 80000·280` per run,
25 runs × ~21 rates ≈ **69 GB**) lives **only** in the `individual_runs/`
sub-folder of a scan. The merged files in the scan's main folder keep this data
only in their JSON; the combined NPZ is left empty. We pre-process the raw runs
once into per-rate NPZs that are ~1.7 orders of magnitude smaller.

## Physical model and compression

Each shot is a dark(0)/bright(1) telegraph. A **click** is a bright run; we
register one click at the **first 1** of every bright run, including a trailing
bright run that never returns to 0 within the window.

Compression — **"bright run → 1 step"**: each maximal bright run of length `L`
collapses to a single retained `1` (the click) immediately followed by its `0`;
the other `L−1` ones are dropped and everything after shifts earlier by `L−1`.
Equivalently a raw step `i` is *removed* iff `assigned[i] and assigned[i-1]`, and
the compressed index of any surviving step is `i − (#removed before i)`
(one prefix-sum pass — `O(#clicks)`). The compressed time axis is
`t_compressed = index × dt`. Because each shot removes a different amount of
bright dwell, **compressed lengths differ per shot**.

Only the **compressed click times** are kept as the raw record. Since every
bright run is normalised to length 1, the dark-run length between two
consecutive clicks is exactly `Δt_click − 1`, so the entire dark-gap structure
is recoverable from click times alone — see *Re-compression* below.

## Usage

Two run files drive this (see also the README "k-space pole analysis" section):

- **`run_kspace_preprocessing.py`** — build the compressed records from the raw
  `individual_runs/`. `MIN_DARK=2` switches to the "2-click" (`1→00`) rule.
- **`run_kspace_analysis.py`** — the k-space pole analysis, `SOURCE="json"` or
  `"compressed"`, with optional run-bootstrap error bars, theory overlay, and
  braiding.

Programmatic preprocessing:

```python
from mpm_analysis.pipeline.compressed_click_pipeline import CompressedClickPipeline
from mpm_analysis.utils.lab_paths import get_lab_path

individual_runs = get_lab_path(
    "Experiments", "Zeno", "CavityRecAl6", "Chip_TransSap46_5", "OPX",
    "two_transmons", "click_record_repeated", "Omega_BG0_scan",
    "scan_experiment_results_17", "individual_runs",
)

out = CompressedClickPipeline().run(
    individual_runs,
    rates=None,            # all rates; or e.g. [1425, 1590]
    max_runs=None,         # all 25 runs; or 2 for a quick test
    min_dark=1,            # 1 = 1->0 (default); 2 = 1->00 "2-click" rule
    make_figures=True,     # counting-statistics + summary PNGs
    run_poles=False,       # poles are run_kspace_analysis.py's job
)
```

Then the (unified) pole analysis on the built records:

```python
from mpm_analysis.pipeline.kspace_pipeline import KSpacePipeline

KSpacePipeline().run(out["out_dir"], source="compressed",
                     bootstrap_over_runs=False, braiding=True, theory=None)
```

Everything is written to `<scan_folder>/compressed_click_analysis/` with
timestamped, **non-clobbering** names. Or run the low-level steps directly:

```python
from mpm_analysis.io.experimental import CompressedClickBuilder, CompressedClickLoader

paths   = CompressedClickBuilder(min_dark=1).build(individual_runs)  # one NPZ per rate
records = CompressedClickLoader().load(paths[0].parent)              # load them back
```

## Output NPZ schema (one per rate)

| Key | Shape | Meaning |
|-----|-------|---------|
| `lambda_val`, `rate_khz`, `dt_us`, `n_runs`, `n_shots`, `T_raw`, `n_max` | scalar | metadata |
| `alpha`, `beta`, `snr`, `optimal_threshold_v` | scalar | authoritative per-rate threshold (from the main combined JSON; per-run mean fallback) |
| `alpha_per_run`, `beta_per_run` | `(n_runs,)` | per-run threshold values |
| `omega_dg`, `omega_bg` | scalar | drive rates (Hz, as in the JSON) |
| `time_raw_us` | `(T_raw,)` | raw time axis (µs) |
| `t_compressed_us` | `(T_c,)` | compressed time axis (µs) |
| `click_times` | `(Σ clicks,)` | **raw record** — compressed click times, grouped by shot |
| `shot_offsets` | `(n_shots+1,)` | per-shot slices into `click_times` |
| `run_shot_offsets` | `(n_runs+1,)` | per-run slices into the shot axis (bootstrap unit) |
| `jump_counts` | `(n_shots,)` | clicks per shot |
| `compressed_lengths` | `(n_shots,)` | per-shot compressed length (variable) |
| `counts_hist` | `(n_runs, T_c, n_max+1)` | per-run `H[t,n]` of cumulative clicks among alive shots |
| `alive` | `(n_runs, T_c)` | per-run alive-shot count vs compressed time |
| `pnt` | `(T_c, n_max+1)` | aggregated `P(n,t)` over alive shots |
| `shots_per_time` | `(T_c,)` | shot-noise budget (Σ alive over runs) |
| `runs_per_time` | `(T_c,)` | runs contributing at each compressed time |
| `bright_hist` | `(n_runs, T_raw+1)` | **dead time** — per-run histogram of bright-run lengths; bin `L` = number of bright runs of exactly `L` raw steps |
| `fourier_k_array` | `(101,)` | counting-field grid `linspace(0, 2π, 101)` |
| `pkt_real`, `pkt_imag` | `(T_c, 101)` | `P(k,t)` via `compute_pkt_from_pnt(pnt)` |
| `metadata_json` | bytes | run numbers, source filenames, scan id, build time |

## Downstream analyses

**Counting statistics & figures** — `CompressedClickRecord.pnt`, `.mean_clicks()`,
`.var_clicks()`. `plotting.compressed_click_stats.save_compressed_click_figures`
writes the `P(n,t)` histogram grid and the per-λ summary (mean clicks, Fano
factor, mean sequence time, shot-noise budget), with run-to-run error bars.

**Dead time (bright-run lengths)** — the distribution of the consecutive `1`s the
compression collapses. It is the one quantity compression destroys, so it is
accumulated during the single pass over the raw data (`compress_run` returns
`bright_hist`, computed from the prefix sum it already builds — `O(#clicks)`, no
second scan and no extra `(N,T)` temporary) and stored per rate.
`CompressedClickRecord.dead_time_pmf()` / `.dead_time_summary()` give the
normalized distribution and the mean with run-to-run error bars;
`fit_dead_time_exponential` is the MLE geometric/exponential fit
(`P(L) ∝ q^L`, `τ = −dt/ln q`), fixed by the sample mean alone. The figure saver
writes one normalized histogram PNG per λ with the fit and the mean on the panel,
a mean-vs-λ PNG, and `dead_time_vs_lambda_*.npz` with the numbers.
`bright_hist` always describes the raw `1->0` bright runs regardless of
`min_dark` (which merges *clicks*, not runs), and is carried through
re-compression unchanged. NPZs built before it existed load with
`bright_hist=None` (`.has_dead_time == False`) and need a rebuild from
`individual_runs` — the collapsed `1`s are not recoverable.

**k-space pole analysis** — `kspace.CompressedClickKSpaceLoader` reads the NPZs
into `KSpaceRecord`s — a drop-in for `KSpaceJSONLoader`, feeding the same
`KSpaceAnalysis` → `KSpacePolesResult`.

> **Surviving-shot normalization (important).** The stored `pnt`/`pkt` are
> normalized at each compressed time by the shots that still have data there
> (`t < compressed_length`).  The k-space loader uses exactly this signal and
> truncates the low-statistics tail with `min_alive_frac` (default `0.5`), so the
> retained axis contains only times where at least half of the compressed
> trajectories still exist.  There is no padding/freezing of ended shots; records
> at different λ may have different time axes and error bars.

**Correlations** — `analysis.click_correlations.compute_click_correlations`
returns the connected `C(Δt)` and normalized `g²(Δt)` on the compressed click
train, steady-state averaged and with run-bootstrap error bars. This supersedes
the earlier anchor-averaged correlator. Sanity checks: `C(0)=r̄(1−r̄)`,
`g²(0)=1/r̄`, and `g²(1)=0` (two bright runs cannot be adjacent — a one-bin
dead-time).

## Re-compression (future click rules)

A stricter rule — e.g. require a dark gap of ≥ 2 zeros so an isolated `101` is a
false negative — can be applied **without** re-reading the originals, because the
dark gap before a click is `Δt_click − 1`:

```python
from mpm_analysis.io.experimental import recompress_min_dark
new_click_times, new_len = recompress_min_dark(click_times, compressed_length, min_dark=2)
```

`min_dark=1` is the default `1→0` rule (no-op); `min_dark=2` merges single-0
`101` false negatives.

## Performance

Streaming, one run's `assigned_states` (~22 MB) in memory at a time; ~6 s per run
(≈ 50 min for a full 525-run scan). Output ≈ 24 MB per rate (~0.5 GB per scan).
