# Workflow — End-to-End Data Flow

This document traces the complete journey of data through the system, from raw inputs to final outputs.

## High-Level Pipeline

```
Inputs                  Processing                      Outputs
──────                  ──────────                      ───────
observations.csv ──┐
                    ├── ingestion ── context ── bayes_loop ── forecast.csv
transcript.txt  ──┘         │           │          │           yield_curve.png
                            │           │          │
                            └── CLI args ┘          │
                                                    └── console summary
```

## Step-by-Step Walkthrough (Forecast CLI)

### 1. Argument Parsing (`cli.py`)

The `main()` function in `cli.py` uses `argparse` to collect:

- Path to an observations CSV **or** inline observation string (e.g. `"Jan=64, Feb=68.5"`).
- Optional transcript file path(s).
- Forecast horizon (`--horizon 2026-08`), seed, hardening toggles, and output paths.

### 2. Data Ingestion (`ingestion.py`)

Depending on the arguments provided:

- **`load_observations_csv(path)`** — reads a two-column CSV (`month`, `yield`) and returns `list[Observation]`.
- **`parse_inline_observations(inline_str)`** — parses a string like `"Jan=64, Feb=68.5"` into `list[Observation]`.
- **`load_transcripts(paths)`** — reads one or more plain-text transcript files and concatenates them.
- **`ensure_bounds(observations)`** — clamps every yield value to [0, 100].

### 3. Task Context Generation (`context.py`)

If transcript text is available, `generate_task_context(text)` extracts:

| Feature | How it is extracted |
|---|---|
| `guidance_growth_low/high` | Regex for patterns like "7-8% monthly" |
| `ribbonfet_mentioned` | Case-insensitive keyword search |
| `powervia_mentioned` | Case-insensitive keyword search |
| `transcript_confidence` | Count of positive terms ("confidence", "improved", …) |
| `transcript_risk` | Count of risk terms ("risk", "delay", …) |
| `s_curve_midpoint` | 80 if both RibbonFET+PowerVia mentioned, else 78 |
| `s_curve_steepness` | Base 0.14 + up to 0.03 from confidence score |

If no transcript is provided, `default_task_context()` supplies a reasonable default.

### 4. LLAMBO Repo Detection (`llambo_integration.py`)

`detect_llambo_repo(project_root)` checks whether `external/LLAMBO/` exists and, if so, reads its HEAD commit SHA. This is informational only — the vendored repo is not executed.

### 5. Hardening Configuration (`hardening.py`)

A `HardeningConfig` dataclass is built from CLI flags and validated (clamped to safe ranges). When `--disable-hardening` is passed, `enabled=False` and the system falls back to baseline LLAMBO behavior.

### 6. Bayesian Forecast Loop (`bayes_loop.py → run_forecast()`)

This is the core algorithm:

1. **Observed points** are appended first with `stddev=0` (known data).
2. The number of forecast steps is computed from `months_ahead` and `horizon`.
3. A `LlamboStyleSurrogate` is initialized with the task context, observed yields, and hardening config.
4. Historical z-scores are computed for calibration (comparing each observed point to what the surrogate would have predicted).
5. An `IntervalCalibrator` is built from those z-scores (isotonic or quantile-scale).
6. For each future month:
   - The surrogate evaluates a grid of 61 candidate growth rates (0 %–15 %).
   - For each candidate, `posterior_for_candidate()` computes a posterior mean and stddev using the S-curve prior, context drift, and hardening blend.
   - `expected_improvement()` scores each candidate.
   - The best candidate is selected; its posterior becomes the forecast for that month.
   - Uncertainty is propagated forward (previous stddev feeds into next step).
   - Hardening applies step-cap clipping and outlier-aware variance inflation.
   - The calibrator adjusts the z-multiplier for the 95 % CI.

### 7. Output Generation

- **`write_forecast_csv()`** writes `list[ForecastPoint]` to a CSV.
- **`plot_learning_curve()`** renders a Matplotlib chart with observed points, posterior mean, and 95 % CI band, saving it as a PNG.
- The console prints a human-readable summary table.

## Evaluation CLI Workflow (`eval_cli.py`)

The evaluation harness follows a different path:

1. Loads real observations and (optionally) generates synthetic stress-test datasets (clean, noisy, outlier).
2. Builds multiple transcript scenarios (varying growth priors and risk levels).
3. Runs a **rolling-origin backtest**: for each dataset × scenario × model × horizon, the training window grows one month at a time and a prediction is made for the held-out future point.
4. Five models are compared:
   - `llambo_style:baseline` — surrogate with hardening disabled.
   - `llambo_style:hardened` — surrogate with hardening enabled.
   - `persistence` — last-value carry-forward.
   - `linear_trend` — OLS linear extrapolation.
   - `logistic_scurve` — grid-search logistic curve fit.
   - `gp_surrogate` — Gaussian Process with RBF kernel.
5. Metrics are aggregated: MAE, RMSE, CRPS, 95 % coverage, interval width, calibration error.
6. Outputs: `metrics_summary.csv`, `backtest_predictions.csv`, `ablation_comparison.csv`, calibration and benchmark PNGs, and `critical_assessment_hardened.md`.
