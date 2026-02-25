# Configuration

All tuneable parameters are controlled through CLI flags that map to the `HardeningConfig` dataclass. This document explains each parameter, its valid range, and its effect on the forecast.

## Hardening Toggle

| Parameter | CLI Flag | Default | Description |
|---|---|---|---|
| `enabled` | `--disable-hardening` | `True` (enabled) | When disabled, the system reverts to baseline LLAMBO behavior — no robust weighting, no calibration, no blending. |

## Prior / Data Blending

| Parameter | CLI Flag | Default | Valid Range | Description |
|---|---|---|---|---|
| `prior_weight` | `--prior-weight` | `0.65` | [0.0, 1.0] | Controls the blend between the context-driven prior growth rate and the data-anchored growth rate (median historical). Higher values trust the transcript more; lower values trust the data more. |

**Effect:** At `prior_weight=1.0`, the surrogate ignores historical growth entirely. At `prior_weight=0.0`, it ignores transcript guidance. The default 0.65 slightly favors the prior.

## Robust Likelihood

| Parameter | CLI Flag | Default | Choices | Description |
|---|---|---|---|---|
| `robust_likelihood` | `--robust-likelihood` | `huber` | `none`, `huber`, `student_t` | Determines how outlier residuals are down-weighted. |
| `huber_delta` | `--huber-delta` | `1.75` | ≥ 0.1 | Threshold (in standardized residual units) above which Huber weighting applies. Residuals within ±δ get full weight; those outside are down-weighted as δ/|z|. |
| `student_t_df` | `--student-t-df` | `5.0` | ≥ 2.1 | Degrees of freedom for Student-t weighting. Lower values = heavier tails = more tolerant of outliers. |

**Effect on uncertainty:** When a robust likelihood is active, the surrogate slightly inflates the predictive stddev (×1.08 for Huber, ×1.15 for Student-t) to avoid over-confident intervals in the presence of heavy-tailed residuals.

## Context Drift Control

| Parameter | CLI Flag | Default | Valid Range | Description |
|---|---|---|---|---|
| `context_drift_clip` | `--context-drift-clip` | `0.02` | ≥ 0.0 | Absolute cap on the net context drift (confidence boost minus risk drag) applied to the growth rate. |

**Effect:** Prevents an extremely confident or extremely risky transcript from dominating the forecast. A value of 0.02 means the transcript can shift the effective growth rate by at most ±2 percentage points.

## Outlier Handling

| Parameter | CLI Flag | Default | Valid Range | Description |
|---|---|---|---|---|
| `outlier_z_clip` | `--outlier-z-clip` | `3.25` | ≥ 1.5 | Threshold for flagging historical residuals as outliers (in z-score-like units). Also used to cap month-to-month step size during forecasting. |
| `outlier_std_inflation` | `--outlier-std-inflation` | `1.5` | ≥ 1.0 | Factor by which the predictive stddev is inflated when historical outlier pressure is detected. |

**Effect:** When past observations show z-scores exceeding `outlier_z_clip`, the forecast widens its uncertainty bands by up to `outlier_std_inflation ×`. This prevents the model from being over-confident when the training data contains shocks.

## Interval Calibration

| Parameter | CLI Flag | Default | Choices / Range | Description |
|---|---|---|---|---|
| `interval_calibration` | `--interval-calibration` | `isotonic` | `none`, `isotonic`, `quantile_scale` | Primary method for recalibrating the z-multiplier used in 95 % CI construction. |
| `calibration_fallback` | `--calibration-fallback` | `quantile_scale` | `none`, `quantile_scale` | Fallback method when the primary calibrator has too few data points. |
| `calibration_min_points` | `--calibration-min-points` | `10` | ≥ 3 | Minimum number of historical residual points required before calibration is attempted. |
| `interval_alpha` | `--interval-alpha` | `0.95` | [0.5, 0.999] | Target central coverage probability for the prediction interval. |

### How Calibration Works

1. The system computes z-scores for every in-sample observation (comparing the actual yield to what the surrogate would have predicted).
2. If there are ≥ `calibration_min_points` absolute z-scores:
   - **Isotonic:** builds an empirical CDF of |z| values, inverts it at `interval_alpha` to find the calibrated z.
   - **Quantile-scale:** directly takes the `interval_alpha`-th quantile of |z| values.
3. If there are fewer points, the system falls back to `calibration_fallback` (or uses the default z = 1.96).
4. The calibrated z replaces 1.96 in the CI formula: `mean ± calibrated_z × stddev`.

## Parameter Interactions

- **Short histories** (< 3 observations) disable calibration regardless of settings — there simply aren't enough residuals.
- **`prior_weight` + `context_drift_clip`** together control how much the transcript influences the mean trajectory. If both are small, the forecast is almost purely data-driven.
- **`robust_likelihood` + `outlier_std_inflation`** together control tail behavior. Turning both off (`none` + `1.0`) gives the narrowest intervals but risks over-confidence.
- **`interval_calibration`** acts *only* on the CI width, not on the posterior mean. Point accuracy is unaffected.
