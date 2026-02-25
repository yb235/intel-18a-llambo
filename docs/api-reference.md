# API Reference

This document lists every public class, dataclass, and function exposed by the `intel_18a_llambo` package.

---

## Package Exports (`__init__.py`)

```python
from intel_18a_llambo import ForecastPoint, run_forecast
```

---

## `ingestion` Module

### `Observation`

```python
@dataclass(frozen=True)
class Observation:
    month: date          # First day of the observation month
    yield_pct: float     # Yield percentage [0, 100]
```

### Functions

```python
def month_from_text(value: str, default_year: int = 2026) -> date
```
Converts a month token (`"Jan"`, `"February"`, `"2026-02"`) to a `date` (1st of that month).

---

```python
def load_observations_csv(path: Path) -> list[Observation]
```
Reads a CSV with columns `month` and `yield`. Returns sorted observations.

---

```python
def parse_inline_observations(inline: str, default_year: int = 2026) -> list[Observation]
```
Parses an inline string like `"Jan=64, Feb=68.5"`. Raises `ValueError` on parse failure.

---

```python
def load_transcripts(paths: list[Path]) -> str
```
Reads and concatenates one or more text files. Returns a single string.

---

```python
def ensure_bounds(observations: list[Observation], lo: float = 0.0, hi: float = 100.0) -> list[Observation]
```
Clamps every `yield_pct` to `[lo, hi]`.

---

## `context` Module

### `TaskContext`

```python
@dataclass(frozen=True)
class TaskContext:
    guidance_growth_low: float       # Lower bound of monthly growth guidance
    guidance_growth_high: float      # Upper bound
    guidance_growth_mid: float       # Midpoint
    transcript_confidence: float     # Positive-sentiment term count
    transcript_risk: float           # Risk-sentiment term count
    ribbonfet_mentioned: bool        # RibbonFET keyword detected
    powervia_mentioned: bool         # PowerVia keyword detected
    s_curve_midpoint: float          # Yield inflection point (78 or 80)
    s_curve_steepness: float         # S-curve transition sharpness
    description: str                 # Human-readable summary
```

### Functions

```python
def generate_task_context(transcript_text: str) -> TaskContext
```
Extracts features from transcript text and returns a `TaskContext`.

---

```python
def default_task_context() -> TaskContext
```
Returns a sensible default context (RibbonFET + PowerVia, 7-8 % guidance).

---

## `surrogate` Module

### `SurrogatePosterior`

```python
@dataclass(frozen=True)
class SurrogatePosterior:
    mean: float      # Predicted yield
    stddev: float    # Prediction uncertainty
```

### `LlamboStyleSurrogate`

```python
class LlamboStyleSurrogate:
    def __init__(
        self,
        context: TaskContext,
        seed: int = 18,
        observed_yields: list[float] | None = None,
        hardening: HardeningConfig | None = None,
    ) -> None: ...
```

#### Methods

```python
def posterior_for_candidate(
    self,
    prev_yield: float,      # Previous month's yield
    growth_rate: float,      # Candidate monthly growth rate [0, 0.20]
    month_index: int,        # Step index (1-based)
) -> SurrogatePosterior
```
Computes the posterior for a single candidate growth rate.

---

```python
@staticmethod
def expected_improvement(
    mu: float,               # Posterior mean
    sigma: float,            # Posterior stddev
    incumbent_best: float,   # Best yield seen so far
    xi: float = 0.05,        # Exploration parameter
) -> float
```
Returns the Expected Improvement acquisition value.

---

```python
def pick_candidate_growth(
    self,
    prev_yield: float,
    incumbent_best: float,
    month_index: int,
) -> tuple[float, SurrogatePosterior, float]
```
Evaluates 61 candidates and returns `(best_growth_rate, best_posterior, best_EI_value)`.

---

## `bayes_loop` Module

### `ForecastPoint`

```python
@dataclass(frozen=True)
class ForecastPoint:
    month: date                          # Calendar month
    observed_yield: float | None         # Actual yield (None for forecasted months)
    posterior_mean: float                 # Predicted yield
    posterior_stddev: float              # Prediction uncertainty
    ci95_low: float                      # Lower 95% CI bound
    ci95_high: float                     # Upper 95% CI bound
    selected_growth_rate: float | None   # Growth rate chosen by EI (None for observed)
    acquisition_value: float | None      # EI score (None for observed)
```

### Functions

```python
def run_forecast(
    observations: list[Observation],
    context: TaskContext,
    months_ahead: int = 6,
    horizon: date | None = None,
    seed: int = 18,
    hardening: HardeningConfig | None = None,
) -> list[ForecastPoint]
```
Main forecast entry point. Returns observed + forecasted points. Raises `ValueError` if no observations are provided.

---

```python
def add_months(base: date, count: int) -> date
```
Adds `count` months to a date (wrapping years).

---

```python
def month_distance(start: date, end: date) -> int
```
Returns the number of months between two dates.

---

## `hardening` Module

### `HardeningConfig`

```python
@dataclass(frozen=True)
class HardeningConfig:
    enabled: bool = False
    prior_weight: float = 0.65
    robust_likelihood: str = "huber"         # "none", "huber", "student_t"
    huber_delta: float = 1.75
    student_t_df: float = 5.0
    context_drift_clip: float = 0.02
    outlier_z_clip: float = 3.25
    outlier_std_inflation: float = 1.5
    interval_calibration: str = "isotonic"   # "none", "isotonic", "quantile_scale"
    calibration_fallback: str = "quantile_scale"  # "none", "quantile_scale"
    calibration_min_points: int = 10
    interval_alpha: float = 0.95
```

#### Methods

```python
@staticmethod
def baseline() -> HardeningConfig       # Returns config with enabled=False
def validated(self) -> HardeningConfig   # Returns a copy with all fields clamped to safe ranges
```

### `IntervalCalibrator`

```python
@dataclass(frozen=True)
class IntervalCalibrator:
    method: str        # "none", "isotonic", "quantile_scale", "quantile_scale_fallback"
    z_value: float     # Calibrated z-multiplier
    data_points: int   # Number of residual points used
```

```python
def adjusted_z(self, default_z: float) -> float
```
Returns the calibrated z-value, or `default_z` if calibration is disabled.

### Functions

```python
def clip_value(value: float, lo: float, hi: float) -> float
def build_interval_calibrator(z_scores: list[float], config: HardeningConfig, default_z: float = 1.96) -> IntervalCalibrator
def robust_weight(z: float, config: HardeningConfig) -> float
def outlier_scale_multiplier(z_scores: list[float], config: HardeningConfig) -> float
```

---

## `plotting` Module

```python
def plot_learning_curve(points: list[ForecastPoint], output_path: Path) -> None
```
Renders a learning-curve chart with observed points, posterior mean, and 95 % CI, and saves it as a PNG.

---

## `llambo_integration` Module

```python
def detect_llambo_repo(root: Path) -> tuple[Path, str | None]
```
Returns `(path_to_external_LLAMBO, short_commit_hash_or_None)`.

---

## `evaluation` Module

### Key Types

```python
@dataclass(frozen=True)
class ForecastDistribution:
    mean: float
    stddev: float

@dataclass(frozen=True)
class Scenario:
    name: str
    context: TaskContext
    apply_to_all_models: bool

@dataclass(frozen=True)
class BacktestPrediction:
    dataset: str
    dataset_type: str        # "observed" or "synthetic"
    scenario: str
    model: str
    model_version: str       # "baseline" or "hardened"
    origin_month: date
    target_month: date
    horizon: int
    y_true: float
    y_pred_mean: float
    y_pred_stddev: float
    ci95_low: float
    ci95_high: float
    status: str              # "ok" or "failed"
    note: str
```

### Forecast Functions

```python
def forecast_llambo(train, horizon, context, seed) -> ForecastDistribution
def forecast_llambo_with_config(train, horizon, context, seed, hardening) -> ForecastDistribution
def forecast_persistence(train, horizon) -> ForecastDistribution
def forecast_linear_trend(train, horizon) -> ForecastDistribution
def forecast_logistic_curve(train, horizon) -> ForecastDistribution
def forecast_gp_surrogate(train, horizon) -> ForecastDistribution
```

### Orchestration

```python
def run_quality_evaluation(
    observations_csv: Path,
    transcript_files: list[Path] | None,
    output_dir: Path,
    max_horizon: int = 6,
    seed: int = 18,
    include_synthetic: bool = True,
    enable_plots: bool = True,
    hardening: HardeningConfig | None = None,
) -> dict[str, Path]
```
Top-level evaluation entry point. Returns a dict mapping output names to file paths.
