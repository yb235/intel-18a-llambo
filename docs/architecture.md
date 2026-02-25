# Architecture

This document describes the high-level design of the Intel 18A LLAMBO Yield Forecast system.

## Design Philosophy

The project follows a **modular pipeline** architecture. Each stage of the forecasting process is encapsulated in its own Python module with well-defined inputs and outputs. There is no framework dependency — the only runtime libraries are NumPy and Matplotlib.

## Module Map

```
src/intel_18a_llambo/
├── __init__.py              # Package entry — exports ForecastPoint, run_forecast
├── cli.py                   # Main forecast CLI (argument parsing → pipeline orchestration)
├── eval_cli.py              # Quality-evaluation CLI (rolling backtests + A/B comparison)
├── ingestion.py             # Data loading: CSV, inline, transcripts, bounds enforcement
├── context.py               # Transcript → TaskContext (NLP-lite feature extraction)
├── surrogate.py             # LLAMBO-style surrogate model + Expected Improvement acquisition
├── bayes_loop.py            # Bayesian forecast loop (observed → projected ForecastPoints)
├── hardening.py             # Robustness layer: outlier handling, calibration, heavy tails
├── evaluation.py            # Full evaluation harness: backtests, baselines, metrics, plots
├── plotting.py              # Learning-curve chart renderer
└── llambo_integration.py    # Detects vendored LLAMBO repo under external/LLAMBO/
```

## Component Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                          CLI Entry Points                            │
│  cli.py  (forecast)               eval_cli.py  (quality evaluation)  │
└────────────┬──────────────────────────────┬──────────────────────────┘
             │                              │
             ▼                              ▼
┌────────────────────┐          ┌──────────────────────┐
│   ingestion.py     │          │   evaluation.py      │
│  CSV / inline /    │          │  Rolling backtests,  │
│  transcript loading│          │  baseline models,    │
└────────┬───────────┘          │  metrics, plots,     │
         │                      │  critical assessment │
         ▼                      └──────────┬───────────┘
┌────────────────────┐                     │
│   context.py       │                     │ (calls run_forecast
│  Transcript → Task │                     │  for each fold)
│  Context features  │                     │
└────────┬───────────┘                     │
         │                                 │
         ▼                                 ▼
┌──────────────────────────────────────────────────────┐
│                    bayes_loop.py                      │
│  Iterates month-by-month, calling surrogate to pick  │
│  best growth candidate via Expected Improvement.     │
│  Produces list[ForecastPoint].                       │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────┐    ┌────────────────────┐
│       surrogate.py           │◄───│   hardening.py     │
│  LlamboStyleSurrogate        │    │  Outlier clipping, │
│  - posterior_for_candidate() │    │  robust weights,   │
│  - expected_improvement()    │    │  interval calib.   │
│  - pick_candidate_growth()   │    └────────────────────┘
└──────────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │  plotting.py   │
          │  PNG chart     │
          └────────────────┘
```

## Key Data Structures

| Structure | Module | Purpose |
|---|---|---|
| `Observation(month, yield_pct)` | `ingestion.py` | A single observed yield data point |
| `TaskContext(…)` | `context.py` | Extracted transcript features (guidance, risk, S-curve params) |
| `HardeningConfig(…)` | `hardening.py` | All tuneable robustness / calibration knobs |
| `SurrogatePosterior(mean, stddev)` | `surrogate.py` | Surrogate prediction for one candidate |
| `ForecastPoint(…)` | `bayes_loop.py` | One row of the final forecast (month, mean, CI, …) |
| `BacktestPrediction(…)` | `evaluation.py` | One prediction in the rolling-origin backtest |

## External Dependencies

| Package | Role |
|---|---|
| `numpy ≥ 1.26` | Array math, RNG, linear algebra (GP kernel) |
| `matplotlib ≥ 3.8` | Chart rendering (Agg backend, no display needed) |

The project has **no LLM runtime dependency**. The "LLAMBO-style" label refers to the *design pattern* (context-rich surrogate + Bayesian Optimization loop + acquisition function), not to a live language-model call. The upstream LLAMBO repository is cloned under `external/LLAMBO/` for reference only.
