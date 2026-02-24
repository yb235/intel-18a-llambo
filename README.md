# Intel 18A LLAMBO Yield Forecast

This project implements a runnable, modular Python workflow for Intel 18A yield progression forecasting with a LLAMBO-style surrogate + Bayesian acquisition loop.

The model treats transcript guidance as context, uses an S-curve prior for advanced-node yield ramp, and forecasts month-by-month posterior mean plus uncertainty with a non-interactive plot output.

## What is included

- `data` ingestion for numeric observations and transcript text
- task context generation with `RibbonFET`, `PowerVia`, and S-curve assumptions
- LLAMBO-style surrogate model in a Bayesian loop with expected improvement acquisition
- month-by-month posterior forecast (mean, stddev, 95% CI)
- plotting script that saves a learning-curve chart to file
- reproducible sample data and CLI examples
- local clone of LLAMBO repository under `external/LLAMBO`

## Project layout

- `external/LLAMBO/`: cloned from `https://github.com/tennisonliu/LLAMBO`
- `src/intel_18a_llambo/`: implementation package
- `data/sample_observations.csv`: sample numeric observations
- `data/sample_transcript_q1_2026.txt`: sample management-guidance transcript

## Setup

From project root (`intel-18a-llambo`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproducible demo

### 1) Run from sample observations (Jan=64, Feb=68.5) + transcript context

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/intel18a_yield_curve.png \
  --seed 18 \
  --horizon 2026-08
```

### 2) Run with inline observations only

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-inline "Jan=64, Feb=68.5" \
  --output-csv outputs/forecast_inline.csv \
  --output-plot outputs/intel18a_yield_curve_inline.png \
  --seed 18 \
  --horizon 2026-08
```

## Assumptions

- Yield search space is constrained to `[0, 100]`.
- Management guidance can imply a target monthly improvement (for example 7-8%).
- The trajectory follows an S-curve (logistic-style progress) with slowing gains near high yield.
- LLAMBO-style behavior is represented by a context-aware surrogate that proposes candidates and scores them with an expected-improvement acquisition function.
- This is a decision-support model and not a ground-truth process simulator.

## Verification commands

```bash
python -m compileall src
PYTHONPATH=src python -m intel_18a_llambo.cli --observations-inline "Jan=64, Feb=68.5" --horizon 2026-08
```

## Quality evaluation harness

Run the hardened rolling-origin quality evaluation (horizons 1-6) with strict A/B comparison:

```bash
PYTHONPATH=src python -m intel_18a_llambo.eval_cli \
  --observations-csv data/sample_observations.csv \
  --output-dir outputs/quality_hardened \
  --max-horizon 6 \
  --seed 18 \
  --prior-weight 0.65 \
  --robust-likelihood huber \
  --huber-delta 1.75 \
  --context-drift-clip 0.02 \
  --outlier-z-clip 3.25 \
  --outlier-std-inflation 1.5 \
  --interval-calibration isotonic \
  --calibration-fallback quantile_scale \
  --calibration-min-points 10 \
  --interval-alpha 0.95
```

Equivalent entrypoint after editable install:

```bash
intel18a-yield-eval \
  --observations-csv data/sample_observations.csv \
  --output-dir outputs/quality_hardened \
  --max-horizon 6 \
  --seed 18 \
  --prior-weight 0.65 \
  --robust-likelihood huber \
  --huber-delta 1.75 \
  --context-drift-clip 0.02 \
  --outlier-z-clip 3.25 \
  --outlier-std-inflation 1.5 \
  --interval-calibration isotonic \
  --calibration-fallback quantile_scale \
  --calibration-min-points 10 \
  --interval-alpha 0.95
```

Expected outputs:

- `outputs/quality_hardened/metrics_summary.csv`
- `outputs/quality_hardened/backtest_predictions.csv`
- `outputs/quality_hardened/calibration_plot.png`
- `outputs/quality_hardened/benchmark_plot.png`
- `outputs/quality_hardened/ablation_comparison.csv`
- `outputs/quality_hardened/critical_assessment_hardened.md`

To disable hardening (keeps baseline-only LLAMBO behavior in forecast CLI while eval still runs baseline+hardened comparison):

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/intel18a_yield_curve.png \
  --seed 18 \
  --horizon 2026-08 \
  --disable-hardening
```

## Notes on LLAMBO integration

The upstream LLAMBO repository is cloned locally at `external/LLAMBO` via GitHub CLI (`gh repo clone`). This implementation uses LLAMBO design ideas (context-rich surrogate + BO loop + acquisition logic) tailored to yield forecasting.
