# Detector dead-time compression sweep — what the pipeline does

`run_files/run.py` → clause `RUN_DEAD_TIME_SWEEP` → `pipeline/dead_time_pipeline.py`.

**Question it answers.** While the system occupies |B⟩ the readout clicks and the
D–G dynamics is frozen (the detector is *dead*). Does removing that dead time from
the time axis move the extracted transition?

## 1. Inputs

| what | where | note |
|---|---|---|
| fitted observable(s) | `DT_OBSERVABLES`, e.g. `["D"]` or `["D", "G"]` | several keys → one joint fit with shared poles |
| p_click(t) | `DT_CLICK_KEY` | `"click_probability"` = fast-readout bright fraction per bin (index 0 is a placeholder 0); `"B_plus_error"` = projective readout's |B⟩ occupation |
| λ window, t window | `LAMBDA_RANGE`, `T_MAX`, `DROP_FIRST_POINT` | the ordinary loader options |

`EnsembleAverageLoader` loads observables and p_click as ordinary records (same
mask, same `bootstrap_stats`). Only the ens_avg JSONs carry these keys.

## 2. Time-axis remap (`analysis/dead_time.py`)

Only `t` changes; every y value stays as measured.

| `DT_REMAP` | formula | behaviour |
|---|---|---|
| `"pointwise"` | t′ = t·[1 − c·p_click(t)] | folds where a noisy late bin jumps (c·(p + k·Δp) > 1); the trace is **truncated at the first fold** — t′, y, σ and the time-resolved bootstrap statistics alike — and the cut is printed. A λ left with < 2·order + 1 points is dropped for that c (NaN poles). |
| `"cumulative"` | t′ = t − c·∫₀ᵗ p_click ds (trapezoid) | slope 1 − c·p > 0: never folds |

c = 0 is the raw data; c = 1 removes the full dead time. Coefficients come from
`DT_COMPRESS_COEFFS` (sorted ascending).

## 3. Fits per c (`DeadTimeSweepPipeline.run`)

1. **Seed.** `MPMStep` on the raw records gives the c = 0 poles (branch-tracked
   across λ). If this run's own MPM result describes the same data and
   observables, its poles replace them.
2. **Direct route** (track `direct`). `RefineStep` = `refine_poles`, the
   variable-projection least-squares fit of `order` complex exponentials
   (conjugate pairs or real poles, `Re ≤ 0`), evaluated on the non-uniform t′.
   The matrix pencil is never run on t′. Each c starts from the previous c's
   poles, so the sweep follows one continuous branch; Hungarian matching keeps
   the pole labels.
3. **Fallback route** (track `regrid_mpm`, `DT_FALLBACK_REGRID`). Each remapped
   trace is linearly interpolated onto a uniform grid of the raw spacing
   (`DT_REGRID_DT`, default 320 ns), then the ordinary `MPMStep` runs, followed
   by the same `RefineStep`. Its poles are matched to the direct branches so the
   two tracks compare pole by pole.
4. **Per c, per track:** poles `(n_c, n_λ, order)`, fit residual rms, the pair
   discriminant Δ(λ) = c₁² − 4c₀ from the real-quadratic parameters
   (`pack_pole_params`), the EP as the straight-line zero of Δ
   (`linfit_zero_crossing`, half-window `DISCRIMINANT_HALFWIDTH`, λ ≤
   `TRANSITION_LAMBDA_MAX`) and the raw first sign change — the same recipe as
   `run_transitions`.
5. **Bootstrap** (`DT_N_BOOTSTRAP > 0`, direct route only). The data bootstrap
   resamples the measurement statistics (multinomial D/G/B/error) and refits
   every draw **from the clean poles with the refinement alone**
   (`BootstrapStep(mode="data", pencil=False)`). Per c it stores 16/84
   half-widths of every pole and the 16/50/84 quantiles of the per-realization
   EP line fits.

## 4. Theory comparison (`simulation/liouvillian_spectrum.py`)

`DT_THEORY = dict(tau_B, T1, T2)` (µs; `Omega` = the data's `omega_DG`) selects
the 4×4 Liouvillian at k = 0, the ensemble-average generator. The compression is
mapped onto its bright dwell time as **τ_B,eff = (1 − c)·τ_B**, i.e.
c = τ_B,comp/τ_B with τ_B,comp = c·τ_B the dwell time compressed away
(`tau_B_effective`). `theory_transition` bisects the λ at which the oscillating
eigenvalue pair coalesces; `theory_transition_vs_compression` gives the line
λ_EP,theory(c). Set `DT_THEORY = None` to drop the overlay.

## 4b. Synthetic check with the same generator (`DT_SYNTHETIC`)

`synthetic_records` integrates the `DT_THEORY` generator exactly
(`expm(L t) x0`, start in |D⟩) at the experiment's λ values on the
experiment's time grid and emits `D`, `G`, `B` populations plus the click key
(= P_B).  `run_dead_time_synthetic` pushes these through the **identical**
sweep (same remap, coefficients, order, refinement, fallback; no bootstrap —
the data are exact) and prints, per c, the experiment's λ_EP, the synthetic
one and the τ_B,eff line.  Two figures: `dead_time_transition_synthetic` (the
synthetic EP vs c with the experiment overlaid as squares) and
`dead_time_poles_grid_synthetic`.  The question it answers: does compressing
the generator's *own* dead time move its fitted EP the way the data's moves —
and does either follow the τ_B,eff = (1 − c)·τ_B mapping?  The synthetic
sweep has its own pole order, `DT_SYNTHETIC_ORDER` (4 = the generator's number
of modes; `None` = the data's `ORDER`): fitting the four-mode generator with
order 3 biases its c = 0 EP up by ≈ 0.08.

## 5. Outputs

* `dead_time_sweep_<mode>_scan<N>_…json` — coefficients, λ axis, both tracks
  (poles as complex, Δ, rmse, λ_EP, λ_EP,first, bootstrap quantiles /
  half-widths when run), metadata and parameters. `load_dead_time_sweep` reads
  it back; `DeadTimeSweepResult.to_analysis_result(ic, method)` turns one c into
  a standard `ZenoAnalysisResult`.
* the `.csv` next to it — one row per c with λ_EP of every track, bootstrap
  quantiles, mean rmse, number of λ fitted.
* figures (`plotting/dead_time_plots.py`), saved with `SAVE_FIGURES`:
  `dead_time_traces` (raw vs remapped data + direct fit, representative c and
  λ), `dead_time_poles_c<c>` (the standard two-panel spectrum at one c, theory
  eigenvalues and both EPs overlaid), `dead_time_transition` (Δ(λ) per
  representative c; λ_EP vs c for both tracks, bootstrap error bars, theory
  line). With `SHOW_PLOTS` two sliders over c open: traces and spectrum
  (`method="regrid_mpm"` shows the fallback). Sliders need a GUI backend; an
  inline backend renders them as static images.
* the sliders in two savable forms: `DT_PANEL_GRIDS` writes them as
  multi-panel figures — `dead_time_traces_grid` (rows λ × columns c) and
  `dead_time_poles_grid` (one row of the two spectrum panels per c), at most
  `DT_GRID_MAX_PANELS` evenly spaced c — and `DT_SLIDER_HTML` writes them as
  standalone Plotly pages, `dead_time_traces_slider.html` and
  `dead_time_poles_slider.html`, that open in any browser (`plotly.js` from
  the CDN; pass `include_plotlyjs=True` to embed it). Programmatic entry
  points: `plot_traces_panel_grid`, `plot_poles_panel_grid`,
  `export_traces_slider_html`, `export_poles_slider_html`.

## 6. Things to know

* The traces figures show the **first** key of a joint fit; the fit itself uses
  all keys.
* A line-fit zero outside the λ window is an extrapolation; the EP axis is
  clipped to the window, the CSV keeps the value. NaN means Δ never crossed zero.
* Beyond c ≈ 0.4 on scan-50 the EP leaves λ ≤ 1.5; widen `LAMBDA_RANGE` to
  follow it.
* Both sliders keep axis limits fixed over the sweep (IQR-robust) so the poles
  move, not the axes.
