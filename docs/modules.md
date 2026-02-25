# Module-by-Module Code Explanation

This document walks through every source file in `src/intel_18a_llambo/`, explaining its purpose, key classes, and important logic.

---

## `__init__.py`

**Purpose:** Package entry point. Re-exports the two most-used symbols so callers can write:

```python
from intel_18a_llambo import ForecastPoint, run_forecast
```

---

## `ingestion.py` — Data Loading

**Purpose:** Convert raw inputs (CSV files, inline strings, transcript text files) into typed Python objects.

### Key Types

- **`Observation(month: date, yield_pct: float)`** — A single monthly yield measurement. `month` is always the 1st of that month.

### Key Functions

| Function | What It Does |
|---|---|
| `month_from_text(value)` | Converts `"Jan"`, `"February"`, or `"2026-02"` into a `date` object. Defaults to year 2026. |
| `load_observations_csv(path)` | Reads a two-column CSV (`month`, `yield`). Returns sorted `list[Observation]`. |
| `parse_inline_observations(inline)` | Parses `"Jan=64, Feb=68.5"` into `list[Observation]` using regex. |
| `load_transcripts(paths)` | Reads one or more text files and concatenates them with double newlines. |
| `ensure_bounds(observations)` | Clamps every `yield_pct` to [0, 100]. |

### Design Notes

- Month parsing supports both short names (`Jan`) and ISO format (`2026-01`).
- All observations are sorted chronologically before being returned.

---

## `context.py` — Transcript Feature Extraction

**Purpose:** Extract structured features from free-text management transcripts to build a `TaskContext`.

### Key Types

- **`TaskContext`** — Frozen dataclass holding all extracted features:

| Field | Type | Meaning |
|---|---|---|
| `guidance_growth_low` | `float` | Lower bound of monthly growth guidance (e.g. 0.07) |
| `guidance_growth_high` | `float` | Upper bound (e.g. 0.08) |
| `guidance_growth_mid` | `float` | Midpoint of the range |
| `transcript_confidence` | `float` | Count of positive-sentiment terms |
| `transcript_risk` | `float` | Count of risk-sentiment terms |
| `ribbonfet_mentioned` | `bool` | Whether "RibbonFET" appears in the transcript |
| `powervia_mentioned` | `bool` | Whether "PowerVia" appears in the transcript |
| `s_curve_midpoint` | `float` | Yield level at which the S-curve inflects (78 or 80) |
| `s_curve_steepness` | `float` | How sharply the S-curve transitions |
| `description` | `str` | Human-readable summary string |

### Key Functions

| Function | What It Does |
|---|---|
| `_extract_guidance_range(text)` | Regex-based extraction of growth-rate guidance (e.g. "7-8%"). Falls back to 7-8 % if nothing matches. |
| `generate_task_context(text)` | Full extraction pipeline → `TaskContext`. |
| `default_task_context()` | Returns a sensible default context when no transcript is supplied. |

### Design Notes

- The extraction is intentionally simple (regex + keyword counting), not a full NLP pipeline. This keeps the project dependency-free.
- S-curve midpoint increases from 78 → 80 when *both* RibbonFET and PowerVia are mentioned, reflecting higher technology maturity confidence.

---

## `surrogate.py` — LLAMBO-Style Surrogate Model

**Purpose:** The core predictive model. Given a previous yield and a candidate monthly growth rate, it produces a posterior distribution and scores candidates using Expected Improvement.

### Key Types

- **`SurrogatePosterior(mean, stddev)`** — Gaussian posterior for one candidate evaluation.
- **`LlamboStyleSurrogate`** — The surrogate class.

### How `posterior_for_candidate()` Works

For a given `prev_yield`, `growth_rate`, and `month_index`:

1. **Headroom** — how much room is left to reach 100 %: `max(0, 100 - prev_yield)`.
2. **S-curve phase** — a logistic function of `prev_yield` relative to the midpoint. As yield approaches the midpoint, the phase approaches 1 and the gain factor decreases (slowing improvement near high yields).
3. **Context drift** — net effect of transcript confidence minus risk, clipped when hardening is enabled.
4. **Effective growth** — `growth_rate + context_drift`, optionally blended with a data-anchor growth rate (median historical growth) via `prior_weight`.
5. **Posterior mean** — `prev_yield + headroom × effective_growth × phase_gain`, clamped to [0, 100].
6. **Posterior stddev** — a base uncertainty that shrinks with `month_index`, plus a penalty for diverging from the guidance midpoint. Inflated under robust-likelihood modes (Huber or Student-t).

### How `expected_improvement()` Works

Standard Expected Improvement (EI) acquisition function:

```
EI(μ, σ, f*) = (μ − f* − ξ) · Φ(z) + σ · φ(z)
```

where `z = (μ − f* − ξ) / σ`, Φ is the normal CDF, φ is the normal PDF, and `f*` is the current best yield.

### How `pick_candidate_growth()` Works

Evaluates a grid of 61 growth rates from 0 % to 15 %, computes the EI for each, and returns the growth rate with the highest EI along with its posterior and acquisition value.

---

## `bayes_loop.py` — Forecast Loop

**Purpose:** Orchestrates the month-by-month forecast by calling the surrogate repeatedly.

### Key Types

- **`ForecastPoint`** — One row of the final output:

| Field | Meaning |
|---|---|
| `month` | Calendar month (date) |
| `observed_yield` | Actual yield if observed, else `None` |
| `posterior_mean` | Predicted yield |
| `posterior_stddev` | Prediction uncertainty |
| `ci95_low / ci95_high` | 95 % confidence interval bounds |
| `selected_growth_rate` | Growth rate chosen by the acquisition function |
| `acquisition_value` | EI score for the selected candidate |

### `run_forecast()` Logic

1. Appends observed data as `ForecastPoint`s with zero uncertainty.
2. Initializes the surrogate with observed yields and hardening config.
3. Computes historical z-scores for interval calibration.
4. Iterates `steps` months into the future:
   - Calls `pick_candidate_growth()` to select the best growth rate.
   - Propagates uncertainty: `sqrt(posterior_std² + (0.35 × prev_std)²)`.
   - Applies hardening: outlier-aware variance inflation, step-cap clipping, robust multiplier.
   - Computes calibrated CI: `mean ± adjusted_z × propagated_std`.

---

## `hardening.py` — Robustness Layer

**Purpose:** Provides configurable robustness features that improve forecast reliability, especially under outliers, short histories, or noisy transcripts.

### Key Types

- **`HardeningConfig`** — Frozen dataclass with all tuneable knobs (see [Configuration](configuration.md)).
- **`IntervalCalibrator`** — Adjusts the z-multiplier for confidence intervals based on historical residuals.

### Key Functions

| Function | What It Does |
|---|---|
| `clip_value(value, lo, hi)` | Simple numeric clamp. |
| `HardeningConfig.validated()` | Returns a new config with all fields clamped to safe ranges. |
| `build_interval_calibrator(z_scores, config)` | Builds an `IntervalCalibrator` using isotonic CDF inversion, quantile-scale, or passthrough. Falls back gracefully when data is too sparse. |
| `robust_weight(z, config)` | Down-weights outlier residuals. Huber: weight = `δ / |z|` when `|z| > δ`. Student-t: `(ν+1) / (ν+z²)`. |
| `outlier_scale_multiplier(z_scores, config)` | Inflates stddev when historical residuals exceed the outlier threshold. |

### Calibration Modes

| Mode | Behavior |
|---|---|
| `none` | No calibration; uses the standard z = 1.96 for 95 % CI. |
| `isotonic` | Empirical CDF inversion (monotone) on absolute z-scores → calibrated z. |
| `quantile_scale` | Uses the empirical quantile of absolute z-scores at `interval_alpha`. |

If isotonic calibration has fewer points than `calibration_min_points`, the system falls back to `calibration_fallback` (typically `quantile_scale`).

---

## `plotting.py` — Chart Rendering

**Purpose:** Renders a learning-curve chart from `list[ForecastPoint]`.

### What The Chart Shows

- **Green line** — Posterior mean across all months.
- **Green shaded band** — 95 % confidence interval.
- **Dark blue dots** — Observed data points.
- **Orange dots** — Forecasted data points.

The chart is saved as a PNG (150 DPI) using the Matplotlib `Agg` backend (no display required).

---

## `llambo_integration.py` — LLAMBO Repo Detection

**Purpose:** Checks for the vendored upstream LLAMBO repository at `external/LLAMBO/` and reads its git HEAD commit hash.

This is purely informational — the vendored repo is not imported or executed. It exists as a reference to the LLAMBO design that inspired this project's surrogate+BO architecture.

---

## `cli.py` — Main Forecast CLI

**Purpose:** Ties the pipeline together for interactive and scripted use.

### Flow

1. Parse arguments → load observations → load transcripts → generate context.
2. Detect LLAMBO repo (informational).
3. Build and validate `HardeningConfig`.
4. Call `run_forecast()`.
5. Write CSV, render plot, print summary.

### Notable Helpers

- `parse_horizon(value)` — converts `"2026-08"` to a `date`.
- `write_forecast_csv(points, path)` — writes the forecast as a CSV with 8 columns.

---

## `eval_cli.py` — Evaluation CLI

**Purpose:** Thin CLI wrapper around `evaluation.run_quality_evaluation()`.

Accepts the same hardening flags as the main CLI, plus evaluation-specific flags like `--max-horizon`, `--no-synthetic`, and `--disable-plots`.

---

## `evaluation.py` — Quality Evaluation Harness

**Purpose:** The largest module. Implements a full backtest + benchmarking pipeline.

### Baseline Models (in addition to LLAMBO)

| Model | Method |
|---|---|
| `persistence` | Last observed value carried forward; uncertainty grows with √horizon. |
| `linear_trend` | OLS linear regression on training window; extrapolates. |
| `logistic_scurve` | Grid-search fit of a logistic curve; fails gracefully on short windows. |
| `gp_surrogate` | Gaussian Process with RBF kernel; analytic posterior. |

### Scenarios

The harness tests four transcript scenarios varying growth priors (3-5 %, 7-8 %, 10-12 %) and yield-killer risk context.

### Synthetic Datasets

Three synthetic stress-test datasets are generated:

- **`synthetic_clean`** — Smooth logistic ramp + mild seasonality.
- **`synthetic_noisy`** — Clean ramp + Gaussian noise (σ = 2.4).
- **`synthetic_outlier`** — Noisy ramp + three negative outlier shocks.

### Metrics

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| CRPS | Continuous Ranked Probability Score (probabilistic accuracy) |
| Coverage95 | Fraction of true values inside the 95 % CI |
| Mean Interval Width | Average width of the 95 % CI |
| Calibration Error | Mean deviation from ideal PIT quantiles |

### Outputs

- `metrics_summary.csv` — Aggregated metrics per dataset × scenario × model × horizon.
- `backtest_predictions.csv` — Every individual prediction.
- `ablation_comparison.csv` — Side-by-side baseline vs. hardened LLAMBO metrics.
- `calibration_plot.png` — Calibration curve (nominal vs. empirical quantiles).
- `benchmark_plot.png` — Bar chart of RMSE and MAE across all models.
- `critical_assessment_hardened.md` — Auto-generated markdown report with findings, regressions, and a verdict.
