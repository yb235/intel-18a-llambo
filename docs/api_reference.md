# CLI & API Reference

> Complete reference for all command-line entry points and Python API objects in the Intel 18A LLAMBO system.

---

## Table of Contents

1. [Entry Points Overview](#entry-points-overview)
2. [Forecast CLI (`intel18a-yield`)](#forecast-cli)
3. [Evaluation CLI (`intel18a-yield-eval`)](#evaluation-cli)
4. [Audit Visualization CLI (`intel18a-audit-viz`)](#audit-visualization-cli)
5. [Python API — Core Objects](#python-api--core-objects)
6. [Python API — Key Functions](#python-api--key-functions)

---

## Entry Points Overview

After installing the package (`pip install -e .`), three command-line tools become available:

| Command | Module | What It Does |
|---------|--------|--------------|
| `intel18a-yield` | `intel_18a_llambo.cli` | Run a yield forecast and save CSV + plot |
| `intel18a-yield-eval` | `intel_18a_llambo.eval_cli` | Run rolling-origin backtest and evaluate model quality |
| `intel18a-audit-viz` | `intel_18a_llambo.audit_viz` | Generate audit-grade Bayesian visualizations |

You can also run them without installation using `PYTHONPATH=src python -m intel_18a_llambo.<module>`.

---

## Forecast CLI

**Entry point:** `intel18a-yield` / `python -m intel_18a_llambo.cli`

Runs a forward-looking yield forecast and writes a CSV and a chart.

### Usage

```bash
intel18a-yield \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/yield_curve.png \
  --horizon 2026-08 \
  --seed 18
```

### Arguments

#### Input data

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--observations-csv` | `Path` | `None` | Path to a CSV file with `month,yield` columns (and optionally `area_factor`, `effective_die_area_mm2`, etc.). Mutually exclusive with `--observations-inline`. |
| `--observations-inline` | `str` | `None` | Inline observations string, e.g. `"Jan=64, Feb=68.5"`. Uses the default year 2026 for bare month names. |
| `--transcript-files` | `Path` (repeatable) | `None` | One or more plain-text transcript files. All files are concatenated. If omitted, a built-in default context is used (7-8% guidance, RibbonFET + PowerVia). |

> If neither `--observations-csv` nor `--observations-inline` is supplied, the built-in sample (`Jan=64, Feb=68.5`) is used.

#### Forecast scope

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--months-ahead` | `int` | `6` | Minimum number of months to forecast beyond the last observation. |
| `--horizon` | `str` | `"2026-08"` | Forecast target month in `YYYY-MM` format. The model forecasts up to this month (or `--months-ahead` steps, whichever is larger). |
| `--seed` | `int` | `18` | Random seed. Fixed seed gives fully reproducible output. |

#### Output paths

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-csv` | `Path` | `outputs/forecast.csv` | Where to write the forecast table (see [Forecast CSV columns](#forecast-csv-columns)). |
| `--output-plot` | `Path` | `outputs/intel18a_yield_curve.png` | Where to save the yield learning-curve chart (PNG, 150 dpi). |

#### Hardening (robustness)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--disable-hardening` | flag | off | If set, disables all hardening and runs baseline-only LLAMBO behavior (wider, less calibrated intervals). |
| `--prior-weight` | `float` | `0.65` | Blend weight between context prior and data-anchor growth. `1.0` = fully trust the prior; `0.0` = fully trust observed data. |
| `--robust-likelihood` | `str` | `"huber"` | Heavy-tail loss mode. Options: `none`, `huber`, `student_t`. |
| `--huber-delta` | `float` | `1.75` | Huber threshold in standardized residual units. Only used when `--robust-likelihood huber`. |
| `--student-t-df` | `float` | `5.0` | Degrees of freedom for Student-t heavy-tail mode. Only used when `--robust-likelihood student_t`. |
| `--context-drift-clip` | `float` | `0.02` | Maximum absolute adjustment that transcript context can apply to growth rate. Prevents an aggressive transcript from distorting predictions. |
| `--outlier-z-clip` | `float` | `3.25` | Z-score threshold above which a historical residual is treated as an outlier. Outliers inflate uncertainty. |
| `--outlier-std-inflation` | `float` | `1.5` | Multiplier applied to standard deviation when outliers are detected. |
| `--interval-calibration` | `str` | `"isotonic"` | Post-hoc recalibration of prediction intervals. Options: `none`, `isotonic`, `quantile_scale`. |
| `--calibration-fallback` | `str` | `"quantile_scale"` | Fallback calibration method when there are too few data points for isotonic regression. |
| `--calibration-min-points` | `int` | `10` | Minimum number of in-sample residual points needed to attempt isotonic calibration. |
| `--interval-alpha` | `float` | `0.95` | Target coverage for the central prediction interval (e.g., `0.95` = 95% CI). |

#### Debug / info

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--print-context` | flag | off | If set, prints the extracted task context to stdout before the forecast table. |

### Forecast CSV columns

The output CSV (`--output-csv`) contains one row per month (including observed months):

| Column | Description |
|--------|-------------|
| `month` | `YYYY-MM` month label |
| `observed_yield` | Actual yield if this was an observed month; empty for forecast months |
| `posterior_mean` | Model's predicted yield mean |
| `posterior_stddev` | Standard deviation of the prediction |
| `ci95_low` | Lower bound of the 95% prediction interval |
| `ci95_high` | Upper bound of the 95% prediction interval |
| `selected_growth_rate` | Growth rate selected by the acquisition function (empty for observed months) |
| `acquisition_value` | Expected Improvement score for the selected growth rate |
| `area_factor` | Die area factor used for this month's prediction |

---

## Evaluation CLI

**Entry point:** `intel18a-yield-eval` / `python -m intel_18a_llambo.eval_cli`

Runs a full rolling-origin backtesting harness. Tests the model on historical data to measure how good the forecasts are.

### Usage

```bash
intel18a-yield-eval \
  --observations-csv data/processed/enriched_monthly_panel.csv \
  --output-dir outputs/quality_hardened \
  --max-horizon 6 \
  --seed 18 \
  --prior-weight 0.65 \
  --robust-likelihood huber \
  --interval-calibration isotonic
```

### Arguments

Most hardening arguments are identical to the [Forecast CLI](#forecast-cli). The evaluation-specific arguments are:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--observations-csv` | `Path` | `data/sample_observations.csv` | CSV with historical yield data. Longer histories produce better backtest results. |
| `--output-dir` | `Path` | `outputs/quality` | Directory where all evaluation artifacts are written. Created if it does not exist. |
| `--assessment-filename` | `str` | `critical_assessment_hardened.md` | Filename for the generated markdown quality report inside `--output-dir`. |
| `--max-horizon` | `int` | `6` | Maximum number of months ahead to test. The evaluator tests all horizons from 1 to this value. |
| `--seed` | `int` | `18` | Random seed. |
| `--no-synthetic` | flag | off | Disable the built-in synthetic stress-test datasets. Not recommended unless your real history is long (≥ 20 months). |
| `--disable-plots` | flag | off | Skip generating PNG calibration and benchmark plots. Useful for fast runs. |
| `--disable-hardening` | flag | off | Only run baseline model; skip the hardened variant comparison. |

### Evaluation output files

| File | Description |
|------|-------------|
| `metrics_summary.csv` | Aggregated metrics (MAE, RMSE, CRPS, coverage95, calibration error) per dataset, scenario, model version, and horizon |
| `backtest_predictions.csv` | Every individual backtest prediction with true value, predicted mean, stddev, CI bounds |
| `calibration_plot.png` | Calibration reliability diagram: predicted vs actual coverage at multiple levels |
| `benchmark_plot.png` | MAE/RMSE comparison bar chart for baseline vs hardened vs hardened_no_area |
| `ablation_comparison.csv` | Pairwise metric comparison between model versions |
| `critical_assessment_hardened.md` | Human-readable assessment summarizing model quality findings |

### Understanding `metrics_summary.csv`

```
dataset,scenario,model,model_version,horizon,mae,rmse,crps_approx,coverage95,calibration_error,interval_width
```

- `dataset`: Which dataset was used (`sample_observations`, `enriched_monthly_panel`, or a synthetic variant)
- `scenario`: Guidance scenario name (e.g., `prior_7_8_with_killer`)
- `model`: Always `llambo_style`
- `model_version`: `baseline` or `hardened` or `hardened_no_area`
- `horizon`: `1` through `max-horizon`, or `all` for the aggregate
- `mae`: Mean Absolute Error (percentage points)
- `rmse`: Root Mean Squared Error (percentage points)
- `crps_approx`: Continuous Ranked Probability Score (lower = better)
- `coverage95`: Fraction of actuals that fell inside the 95% CI (should be ≈ 0.95)
- `calibration_error`: `|coverage95 - 0.95|` (lower = better calibrated)
- `interval_width`: Average width of the 95% CI (percentage points)

---

## Audit Visualization CLI

**Entry point:** `intel18a-audit-viz` / `python -m intel_18a_llambo.audit_viz`

Generates a suite of audit-grade visualizations including Bayesian update timeline, posterior evolution, scenario comparison, data lineage graph, and a full HTML dashboard.

### Usage

```bash
intel18a-audit-viz \
  --repo-root . \
  --panel-csv data/processed/enriched_monthly_panel.csv \
  --manifest-csv data/raw/source_manifest.csv \
  --output-dir outputs/audit \
  --lineage-format svg
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--repo-root` | `Path` | `.` | Root of the repository (used to locate data files). |
| `--panel-csv` | `Path` | Required | Path to the enriched monthly panel CSV. |
| `--manifest-csv` | `Path` | Required | Path to the source manifest CSV with provenance metadata. |
| `--output-dir` | `Path` | `outputs/audit` | Directory where all audit artifacts are saved. |
| `--lineage-format` | `str` | `svg` | Format for the lineage graph. Options: `svg`, `png`. |

### Audit output files

| File | Description |
|------|-------------|
| `bayesian_update_timeline.png` | Timeline of Bayesian updates as data arrived month by month |
| `posterior_evolution.png` | How the posterior mean and uncertainty evolved over time |
| `scenario_comparison.png` | Side-by-side comparison of different guidance scenarios |
| `lineage_graph.svg` | Data provenance graph: sources → features → model |
| `trace_matrix.csv` | Traceability matrix linking predictions to source records |
| `audit_dashboard.html` | Self-contained HTML page embedding all plots and metrics |
| `audit_summary.md` | Text summary of key audit findings |

---

## Python API — Core Objects

These dataclasses are the main data structures flowing through the system.

### `Observation` (`ingestion.py`)

Represents one month of observed yield data.

```python
@dataclass(frozen=True)
class Observation:
    month: date                              # First day of the month, e.g. date(2026, 1, 1)
    yield_pct: float                         # Yield in percent, e.g. 64.0
    area_factor: float = 1.0                 # Die-area scale: 1.0 = nominal 180 mm²
    cfo_gm_signal_strength: float = 0.0     # CFO gross-margin signal [-1, +1]
    ifs_profitability_timeline_score: float = 0.0  # IFS profitability signal [-1, +1]
    academic_yield_maturity_signal: float = 0.0    # Academic maturity signal [-1, +1]
    disclosure_confidence: float = 0.0       # How confident we are in disclosure data [0, 1]
```

### `TaskContext` (`context.py`)

Numerical signals extracted from management transcripts.

```python
@dataclass(frozen=True)
class TaskContext:
    guidance_growth_low: float    # Lower bound of monthly yield growth guidance, e.g. 0.07
    guidance_growth_high: float   # Upper bound, e.g. 0.08
    guidance_growth_mid: float    # Midpoint, e.g. 0.075
    transcript_confidence: float  # Count of positive sentiment words in transcript
    transcript_risk: float        # Count of risk/negative words in transcript
    ribbonfet_mentioned: bool     # Whether "ribbonfet" appeared in transcript
    powervia_mentioned: bool      # Whether "powervia" appeared in transcript
    s_curve_midpoint: float       # Yield level where gains start slowing (default 78–80)
    s_curve_steepness: float      # How sharp the S-curve inflection is
    description: str              # Human-readable summary of this context
```

### `ForecastPoint` (`bayes_loop.py`)

One month in the forecast output.

```python
@dataclass(frozen=True)
class ForecastPoint:
    month: date                        # First day of the forecast month
    observed_yield: float | None       # Actual yield if known; None for future months
    posterior_mean: float              # Model's predicted yield
    posterior_stddev: float            # Uncertainty (standard deviation)
    ci95_low: float                    # Lower bound of 95% prediction interval
    ci95_high: float                   # Upper bound of 95% prediction interval
    selected_growth_rate: float | None # Growth rate chosen by acquisition function
    acquisition_value: float | None    # Expected Improvement score
    area_factor: float | None          # Die area factor used for this step
```

### `HardeningConfig` (`hardening.py`)

Controls all robustness settings.

```python
@dataclass(frozen=True)
class HardeningConfig:
    enabled: bool = False
    prior_weight: float = 0.65
    robust_likelihood: str = "huber"       # "none" | "huber" | "student_t"
    huber_delta: float = 1.75
    student_t_df: float = 5.0
    context_drift_clip: float = 0.02
    outlier_z_clip: float = 3.25
    outlier_std_inflation: float = 1.5
    interval_calibration: str = "isotonic"  # "none" | "isotonic" | "quantile_scale"
    calibration_fallback: str = "quantile_scale"
    calibration_min_points: int = 10
    interval_alpha: float = 0.95
```

Use `HardeningConfig.baseline()` for a no-hardening instance, or call `.validated()` to sanitize user-supplied values.

### `SurrogatePosterior` (`surrogate.py`)

The output of one surrogate model call.

```python
@dataclass(frozen=True)
class SurrogatePosterior:
    mean: float    # Predicted yield mean
    stddev: float  # Prediction uncertainty
```

---

## Python API — Key Functions

### `run_forecast` (`bayes_loop.py`)

The main forecasting entry point.

```python
from intel_18a_llambo import run_forecast

points: list[ForecastPoint] = run_forecast(
    observations=observations,   # list[Observation] — at least one required
    context=context,             # TaskContext — extracted from transcripts
    months_ahead=6,              # int — minimum forecast steps beyond last observation
    horizon=date(2026, 8, 1),    # date | None — forecast up to this month
    seed=18,                     # int — random seed for reproducibility
    hardening=hardening_config,  # HardeningConfig | None — robustness settings
    use_area_factor=True,        # bool — if False, die-size effects are ignored
)
```

Returns a list of `ForecastPoint` objects covering all observed months plus all forecast months.

### `generate_task_context` (`context.py`)

Extracts numerical signals from transcript text.

```python
from intel_18a_llambo.context import generate_task_context

context = generate_task_context(transcript_text="...management guidance 7-8% monthly...")
```

### `default_task_context` (`context.py`)

Returns a sensible default context when no transcript is available.

```python
from intel_18a_llambo.context import default_task_context

context = default_task_context()
# Equivalent to: 7-8% guidance, RibbonFET + PowerVia, moderate confidence
```

### `load_observations_csv` (`ingestion.py`)

Loads observations from a CSV file.

```python
from intel_18a_llambo.ingestion import load_observations_csv
from pathlib import Path

observations = load_observations_csv(Path("data/sample_observations.csv"))
```

### `parse_inline_observations` (`ingestion.py`)

Parses observations from a short inline string.

```python
from intel_18a_llambo.ingestion import parse_inline_observations

observations = parse_inline_observations("Jan=64, Feb=68.5")
observations = parse_inline_observations("2026-01=64, 2026-02=68.5")  # also works
```

### `run_quality_evaluation` (`evaluation.py`)

Runs the full evaluation harness programmatically.

```python
from intel_18a_llambo.evaluation import run_quality_evaluation
from intel_18a_llambo.hardening import HardeningConfig
from pathlib import Path

outputs = run_quality_evaluation(
    observations_csv=Path("data/processed/enriched_monthly_panel.csv"),
    transcript_files=None,
    output_dir=Path("outputs/quality"),
    assessment_filename="assessment.md",
    max_horizon=6,
    seed=18,
    include_synthetic=True,
    enable_plots=True,
    hardening=HardeningConfig(enabled=True, prior_weight=0.65),
)
# outputs is a dict mapping artifact names to Path objects
```

---

*Last updated: 2026-02-25*
