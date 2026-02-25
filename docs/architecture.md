# Code Architecture Guide

> A module-by-module tour of the Intel 18A LLAMBO codebase — written for first-time readers with no prior knowledge of Bayesian optimization.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [How the Files Relate](#how-the-files-relate)
3. [Module: `ingestion.py`](#module-ingestionpy)
4. [Module: `context.py`](#module-contextpy)
5. [Module: `surrogate.py`](#module-surrogatepy)
6. [Module: `bayes_loop.py`](#module-bayes_looppy)
7. [Module: `hardening.py`](#module-hardeningpy)
8. [Module: `evaluation.py`](#module-evaluationpy)
9. [Module: `plotting.py`](#module-plottingpy)
10. [Module: `cli.py`](#module-clipy)
11. [Module: `eval_cli.py`](#module-eval_clipy)
12. [Module: `audit_viz.py`](#module-audit_vizpy)
13. [Module: `llambo_integration.py`](#module-llambo_integrationpy)
14. [Scripts: `scripts/ingest/`](#scripts-scriptsingest)
15. [Data Layout](#data-layout)
16. [External Dependency: `external/LLAMBO`](#external-dependency-externallambo)
17. [Dependency Graph](#dependency-graph)

---

## Repository Layout

```
intel-18a-llambo/
│
├── src/intel_18a_llambo/          ← Main Python package
│   ├── __init__.py                # Public API: ForecastPoint, run_forecast
│   ├── ingestion.py               # Load & parse raw data → Observation objects
│   ├── context.py                 # Parse transcripts → TaskContext object
│   ├── surrogate.py               # Predict next yield (mean + stddev)
│   ├── bayes_loop.py              # Month-by-month Bayesian iteration
│   ├── hardening.py               # Robustness: outliers, calibration, heavy-tails
│   ├── evaluation.py              # Backtest harness, scenarios, metrics
│   ├── plotting.py                # Yield learning-curve chart
│   ├── cli.py                     # Forecast command-line entry point
│   ├── eval_cli.py                # Evaluation command-line entry point
│   ├── audit_viz.py               # Audit visualization entry point
│   └── llambo_integration.py      # Detect external LLAMBO repo, get commit SHA
│
├── scripts/
│   ├── ingest/
│   │   ├── fetch_sources.py       # Optional: download raw source files + hash
│   │   └── build_enriched_panel.py # Build the full enriched monthly feature panel
│   └── eval/
│       └── write_cfo_academic_assessment.py # Write delta report for a specific run
│
├── data/
│   ├── raw/                       # Source CSVs manually extracted from Intel IR
│   ├── interim/                   # Intermediate feature files (before model-ready)
│   ├── processed/                 # Model-ready datasets
│   ├── sample_observations.csv    # Two-row starter dataset (Jan/Feb 2026)
│   └── sample_transcript_q1_2026.txt  # Example management transcript
│
├── docs/                          # Documentation (you are reading this)
├── outputs/                       # Generated outputs (CSVs, plots, reports)
├── external/LLAMBO/               # Local clone of upstream LLAMBO reference repo
├── pyproject.toml                 # Package metadata, entry points
└── requirements.txt               # Runtime dependencies
```

---

## How the Files Relate

```
         ┌─────────────────────────────────────────────────┐
         │                   cli.py                         │
         │  (orchestrates everything for a single forecast) │
         └────────────────────────┬────────────────────────┘
                                  │ calls
          ┌───────────────────────┼───────────────────────┐
          ↓                       ↓                       ↓
   ingestion.py             context.py             hardening.py
   (load data →             (transcript text        (robustness
    Observation[])           → TaskContext)          config)
          │                       │                       │
          └───────────────────────┴───────────────────────┘
                                  │ feeds into
                           ┌──────▼──────┐
                           │ bayes_loop  │
                           │  .py        │
                           └──────┬──────┘
                                  │ uses
                           ┌──────▼──────┐
                           │ surrogate   │
                           │  .py        │
                           └──────┬──────┘
                                  │ returns ForecastPoint[]
                           ┌──────▼──────┐
                           │ plotting.py │ (saves PNG chart)
                           └─────────────┘

         ┌─────────────────────────────────────────────────┐
         │                 eval_cli.py                      │
         │  (orchestrates the backtest quality harness)     │
         └────────────────────────┬────────────────────────┘
                                  │ calls
                           ┌──────▼──────┐
                           │ evaluation  │ (many scenarios ×
                           │  .py        │  many horizons ×
                           └─────────────┘  many datasets)
                                  │ internally calls bayes_loop
                                  │ and hardening for each cell
```

---

## Module: `ingestion.py`

**Responsibility:** Read raw data and convert it into typed `Observation` objects.

### Key types

```
Observation(
  month,               ← date(YYYY, M, 1)
  yield_pct,           ← float 0–100
  area_factor,         ← float 0.6–1.6 (die size proxy, 1.0 = nominal)
  cfo_gm_signal_strength,          ← float -1 to +1
  ifs_profitability_timeline_score, ← float -1 to +1
  academic_yield_maturity_signal,   ← float -1 to +1
  disclosure_confidence             ← float 0–1
)
```

### Key functions

| Function | Input | Output | Notes |
|----------|-------|--------|-------|
| `load_observations_csv(path)` | CSV `Path` | `list[Observation]` | Reads `month,yield` columns; optional `area_factor`, `effective_die_area_mm2_proxy`, etc. |
| `parse_inline_observations(s)` | `"Jan=64, Feb=68.5"` | `list[Observation]` | Regex-parses month=value pairs; month names default to year 2026 |
| `load_transcripts(paths)` | `list[Path]` | `str` | Concatenates multiple text files with `\n\n` separator |
| `ensure_bounds(obs)` | `list[Observation]` | `list[Observation]` | Clamps yield to [0, 100] and all signals to valid ranges |
| `month_from_text(value)` | `"Jan"` or `"2026-01"` | `date` | Handles both abbreviated month names and ISO `YYYY-MM` strings |

### CSV column detection logic

When loading from CSV, `area_factor` is calculated as follows (first match wins):
1. Direct `area_factor` column → use as-is
2. `effective_die_area_mm2` column → divide by 180 (nominal reference)
3. `effective_die_area_mm2_proxy` column → divide by 180
4. Neither present → default `area_factor = 1.0`

All area factors are clamped to `[0.6, 1.6]`.

---

## Module: `context.py`

**Responsibility:** Convert raw management transcript text into numerical signals.

### Key types

```
TaskContext(
  guidance_growth_low,   ← 0.07 (7%)
  guidance_growth_high,  ← 0.08 (8%)
  guidance_growth_mid,   ← 0.075
  transcript_confidence, ← 3.0 (count of positive words)
  transcript_risk,       ← 2.0 (count of risk words)
  ribbonfet_mentioned,   ← True/False
  powervia_mentioned,    ← True/False
  s_curve_midpoint,      ← 80.0
  s_curve_steepness,     ← 0.149
  description            ← human-readable summary string
)
```

### Extraction logic

1. **Guidance range** — regex `(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*%` finds patterns like `"7-8%"` or `"7 to 8%"`. Falls back to `0.07/0.08` if nothing is found.

2. **Technology keywords** — simple substring search for `"ribbonfet"` and `"powervia"` (case-insensitive).

3. **Sentiment scoring** — word counting:
   - Positive words: `confidence`, `improved`, `progress`, `ahead`, `reduction`, `stable`
   - Risk words: `risk`, `variability`, `delay`, `challenge`, `uncertain`, `headwind`

4. **S-curve parameters**:
   - `s_curve_midpoint = 78.0` by default; raised to `80.0` if both RibbonFET and PowerVia are mentioned
   - `s_curve_steepness = 0.14 + min(0.03, 0.003 × confidence_score)`

### Default context

`default_task_context()` calls `generate_task_context` with a fixed synthetic string that produces 7-8% guidance, both tech features, and moderate confidence. Use this when no transcript is available.

---

## Module: `surrogate.py`

**Responsibility:** The core prediction engine. Given the current yield, a candidate growth rate, and context, predicts the next yield (mean + uncertainty).

### Key types

```
SurrogatePosterior(
  mean,    ← predicted yield %
  stddev   ← prediction uncertainty
)
```

### `LlamboStyleSurrogate` class

Instantiated once per forecast run:

```python
surrogate = LlamboStyleSurrogate(
    context=task_context,
    seed=18,
    observed_yields=[64.0, 68.5],   # used to estimate data-anchor growth
    hardening=hardening_config,
)
```

#### `posterior_for_candidate(prev_yield, growth_rate, month_index, area_factor)`

The prediction formula step by step:

```
1. headroom = 100 - prev_yield          (how much room left?)
2. phase = sigmoid((prev_yield - midpoint) × steepness)
   phase_gain = 1 - 0.55 × phase        (S-curve slowdown)
3. context_drift = 0.0025×confidence - 0.0020×risk
   (clipped to ±context_drift_clip if hardening enabled)
4. size_drag = 0.03 × (area_factor - 1)  (bigger die → more drag)
5. effective_growth = growth_rate + context_drift - size_drag
   (if hardening: blend with data-anchor growth)
6. mean = prev_yield + headroom × effective_growth × phase_gain
   mean -= 3.0 × (area_factor - 1)      (additional size penalty)
7. stddev = max(0.45,
               (1.7 - min(1.1, 0.12×month_index))
               + 18×|growth_rate - guidance_mid|
               + 0.4×|area_factor - 1|)
   (if hardening: stddev ×= 1.08 for huber, ×1.15 for student_t)
```

#### `pick_candidate_growth(prev_yield, incumbent_best, month_index, area_factor)`

Evaluates 61 candidate growth rates from 0% to 15% in 0.25% steps and returns the one with the highest **Expected Improvement** score.

#### `expected_improvement(mu, sigma, incumbent_best, xi=0.05)`

EI formula:
```
z = (mu - incumbent_best - xi) / sigma
EI = (mu - incumbent_best - xi) × Φ(z) + σ × φ(z)
```
where `Φ` is the normal CDF and `φ` is the normal PDF.

#### `_estimate_data_anchor_growth(observed_yields)`

Computes the **median observed growth rate** from historical yields:
```
growth[i] = (yield[i+1] - yield[i]) / max(1, yield[i])
anchor = median(growth)
```
This gives the surrogate a "data-driven" prior to blend with the context prior.

---

## Module: `bayes_loop.py`

**Responsibility:** Run the surrogate month by month from the last observation to the forecast horizon, propagating uncertainty at each step.

### Key types

```
ForecastPoint(
  month,                 ← date
  observed_yield,        ← float | None
  posterior_mean,        ← float
  posterior_stddev,      ← float
  ci95_low,              ← float
  ci95_high,             ← float
  selected_growth_rate,  ← float | None
  acquisition_value,     ← float | None
  area_factor            ← float | None
)
```

### `run_forecast(observations, context, months_ahead, horizon, seed, hardening, use_area_factor)`

Algorithm:

```
1. Sort observations by month.
2. Emit one ForecastPoint per observed month (stddev=0, CI=point).
3. Estimate drift rates for area_factor, signal_prior, disclosure_confidence
   using median differences across observations.
4. Build in-sample z-scores (if ≥3 observations and hardening enabled):
   For each consecutive pair of observations, run the surrogate and compute
   z = (actual - predicted) / stddev.
5. Build interval calibrator from z-scores.
6. Compute outlier_multiplier and robust_multiplier from z-score statistics.
7. For each forecast step (1 … steps):
   a. Estimate area_factor, signal_prior, disclosure for this month (drift).
   b. Adjust area_factor using signal_prior and disclosure (good signal
      reduces effective die-area penalty).
   c. Call surrogate.pick_candidate_growth → (growth, posterior, acq).
   d. Propagate uncertainty:
      propagated_std = sqrt(posterior.stddev² + (0.35 × prev_std)²)
   e. Apply hardening multipliers to propagated_std.
   f. Cap step size: if delta > outlier_z_clip × innovation_scale, clip it.
   g. Compute 95% CI using calibrator-adjusted z-value.
   h. Append ForecastPoint to output.
```

### Uncertainty propagation

Each step's uncertainty is the quadrature sum of:
- The surrogate's own prediction uncertainty (`posterior.stddev`)
- A fraction (35%) of the previous step's uncertainty (`0.35 × prev_std`)

This models "compound uncertainty": the further ahead you forecast, the less certain you become.

---

## Module: `hardening.py`

**Responsibility:** Pure utility functions for robustness. Contains no forecast logic — it adjusts and calibrates.

### `HardeningConfig`

Dataclass holding all robustness parameters. Key method: `.validated()` sanitizes out-of-range values.

### `build_interval_calibrator(z_scores, config, default_z)`

Takes a list of historical standardized residuals (`z_scores`) and builds an `IntervalCalibrator` that knows what z-value to use for the target coverage.

Methods available:
- `"isotonic"`: Uses empirical CDF inversion — picks the z that achieves `interval_alpha` coverage on in-sample data.
- `"quantile_scale"`: Simple empirical quantile at `interval_alpha`.
- `"none"`: Always returns the default (1.96).

### `robust_weight(z, config)`

Returns a weight in `(0, 1]` for a residual with standardized magnitude `z`:
- Huber: weight = 1.0 if `|z| ≤ delta`, else `delta / |z|`
- Student-t: weight = `(ν+1) / (ν + z²)`

### `outlier_scale_multiplier(z_scores, config)`

Returns a multiplier > 1.0 if any z-score exceeds `outlier_z_clip`. The multiplier scales the prediction interval wider to compensate for detected outlier pressure.

---

## Module: `evaluation.py`

**Responsibility:** The full quality evaluation harness. Runs rolling-origin backtests across multiple scenarios, model versions, and datasets.

### Key concepts

**Rolling-origin backtesting:**
- Fix train set to months 1..k, predict months k+1..k+h (horizon h)
- Slide origin k forward across all available months
- Collect predictions and compute metrics vs actuals

**Scenarios:**
- `prior_7_8_with_killer`: The main realistic scenario with 7-8% guidance and a "killer risk" scenario modifier
- Other scenarios test different guidance levels and risk profiles

**Model versions:**
- `baseline`: No hardening (`HardeningConfig.baseline()`)
- `hardened`: Full hardening with all configured parameters
- `hardened_no_area`: Hardened but with `use_area_factor=False` (ablation)

**Synthetic datasets:**
- Generated programmatically to stress-test the model beyond the limited real data
- Variants: clean S-curve, noisy, with outliers, slow ramp, etc.

### Key functions

| Function | Description |
|----------|-------------|
| `generate_synthetic_datasets(seed)` | Creates a set of synthetic yield histories for stress testing |
| `_run_single_backtest(...)` | Runs one rolling-origin backtest for one (dataset, scenario, model_version, horizon) combination |
| `_compute_metrics(predictions)` | Computes MAE, RMSE, CRPS, coverage95, calibration_error, interval_width |
| `run_quality_evaluation(...)` | Orchestrator: runs all combinations, writes all output files |

---

## Module: `plotting.py`

**Responsibility:** A single function that draws the yield learning-curve chart.

### `plot_learning_curve(points, output_path)`

- Observed months → dark blue dots
- Forecast months → burnt orange dots
- Posterior mean line → green
- 95% CI band → light green fill
- Saves as PNG (150 dpi) to `output_path`

No return value. The file is written to disk and the matplotlib figure is closed.

---

## Module: `cli.py`

**Responsibility:** The command-line interface for a single forward forecast.

### Flow

```
parse_args()
    ↓
_load_observations()      ← CSV or inline or default
    ↓
generate_task_context()   ← from transcript files or default
    ↓
detect_llambo_repo()      ← find external/LLAMBO, get commit SHA
    ↓
HardeningConfig(...)      ← build from CLI args
    ↓
run_forecast(...)         ← bayes_loop.py
    ↓
write_forecast_csv()      ← save output CSV
    ↓
plot_learning_curve()     ← save PNG chart
    ↓
print table to stdout
```

---

## Module: `eval_cli.py`

**Responsibility:** The command-line interface for the quality evaluation harness.

### Flow

```
parse_args()
    ↓
HardeningConfig(...)      ← build from CLI args
    ↓
run_quality_evaluation()  ← evaluation.py
    ↓
print artifact paths to stdout
```

Much simpler than `cli.py` because all the complexity lives in `evaluation.py`.

---

## Module: `audit_viz.py`

**Responsibility:** Generate audit-grade visualizations and a self-contained HTML dashboard.

### What it reads

- The enriched monthly panel CSV (yield history, features, provenance tags)
- The source manifest CSV (URL, tier, confidence for each data source)
- Optional: evaluation output CSVs from previous runs

### What it produces

1. **Bayesian update timeline** — plots how prior → posterior evolved each month
2. **Posterior evolution** — traces mean and uncertainty bands over time
3. **Scenario comparison** — overlays multiple guidance scenarios on one chart
4. **Lineage graph** — DOT/SVG graph: data sources → features → model predictions
5. **Trace matrix** — CSV linking every prediction to its upstream source records
6. **HTML dashboard** — embeds all plots inline as base64 images

---

## Module: `llambo_integration.py`

**Responsibility:** Locate the external LLAMBO repository and return its commit SHA for traceability.

### `detect_llambo_repo(root)`

1. Constructs path `root / "external" / "LLAMBO"`.
2. If the directory does not exist, returns `(path, None)`.
3. If it exists, runs `git -C <path> rev-parse --short HEAD` and returns the short commit hash.
4. On any error (git not installed, not a git repo), returns `(path, None)`.

This information is printed by the forecast CLI but is otherwise cosmetic — it has no effect on model behavior.

---

## Scripts: `scripts/ingest/`

### `fetch_sources.py`

Optional pre-step that attempts to download the raw source files listed in `data/raw/source_manifest.csv` and records SHA-256 hashes in `data/interim/source_hashes.csv`.

- Network access is not required — failures are recorded and the script exits cleanly
- Useful for verifying provenance when running in an environment with internet access

### `build_enriched_panel.py`

The main data engineering pipeline. Reads the four raw CSVs and writes the enriched monthly panel:

```
RAW INPUTS                         OUTPUTS
──────────────────────────────     ─────────────────────────────────
intel_quarterly_financial_signals  → data/interim/enriched_monthly_features.csv
intel_18a_milestones               →
intel_18a_academic_signals         → data/processed/enriched_monthly_panel.csv
intel_cfo_signals                  →
```

**Key steps:**
1. Parse all four raw CSVs into typed dataclasses (`QuarterlyPoint`, `Milestone`, `AcademicSignal`, `CfoSignal`)
2. Iterate every month from `2024-01` to `2026-02`
3. For each month, compute:
   - `milestone_stage_norm` = highest stage reached / 8.0
   - `gm_z`, `rev_z` = z-score normalized financial metrics
   - `cfo_signal`, `ifs_signal` = time-carried and confidence-weighted CFO signals
   - `academic_signal` = recency-decayed, tier-weighted academic signal
   - `area_factor` = engineered proxy for die size
   - `proxy_yield` = linear combination of all features (clipped to [20, 92])
   - `yield_source` = "observed" for Jan/Feb 2026, "proxy" for all earlier months
4. Overwrite the two anchor months with true observed yields (64.0 and 68.5)
5. Write interim feature matrix and final model-ready panel

---

## Data Layout

```
data/
├── raw/
│   ├── intel_quarterly_financial_signals.csv   ← Revenue, gross margin by quarter
│   ├── intel_18a_milestones.csv                ← Process stage milestones with dates
│   ├── intel_18a_academic_signals.csv          ← Technical disclosure signals
│   ├── intel_cfo_signals.csv                   ← CFO commentary signals
│   └── source_manifest.csv                     ← Source provenance catalog
├── interim/
│   ├── enriched_monthly_features.csv           ← Full feature matrix (all columns)
│   └── source_hashes.csv                       ← SHA-256 hashes from fetch run
├── processed/
│   └── enriched_monthly_panel.csv              ← Model-ready: month,yield,features
├── sample_observations.csv                     ← Two-row starter: Jan=64, Feb=68.5
└── sample_transcript_q1_2026.txt               ← Example management transcript
```

### `sample_observations.csv` format

```csv
month,yield
2026-01,64.0
2026-02,68.5
```

### `enriched_monthly_panel.csv` format

```csv
month,yield,yield_source,area_factor,effective_die_area_mm2_proxy,
milestone_stage,milestone_stage_norm,gross_margin_gaap_pct,revenue_bil_usd,
gm_z,rev_z,cfo_gm_signal_strength,ifs_profitability_timeline_score,
academic_yield_maturity_signal,disclosure_confidence
```

---

## External Dependency: `external/LLAMBO`

The project design is inspired by the [LLAMBO paper](https://arxiv.org/abs/2402.02145) and the reference implementation is cloned at `external/LLAMBO`. However, **no code from this directory is imported at runtime**. The `llambo_integration.py` module only reads the git commit SHA for traceability logging.

The LLAMBO *ideas* used:
- Context-rich surrogate (TaskContext feeding into the prediction)
- Candidate scoring via Expected Improvement acquisition function
- The concept of a Bayesian optimization loop adapted for time-series forecasting

---

## Dependency Graph

```
__init__.py
    └── bayes_loop.py
            ├── context.py
            ├── hardening.py
            ├── ingestion.py
            └── surrogate.py
                    ├── context.py
                    └── hardening.py

cli.py
    ├── bayes_loop.py
    ├── context.py
    ├── hardening.py
    ├── ingestion.py
    ├── llambo_integration.py
    └── plotting.py

eval_cli.py
    ├── evaluation.py
    │       ├── bayes_loop.py
    │       ├── context.py
    │       ├── hardening.py
    │       └── ingestion.py
    └── hardening.py

audit_viz.py
    └── context.py (for scenario context generation)
```

All modules are self-contained. There are no circular imports.

---

*Last updated: 2026-02-25*
