# Quality Evaluation Harness

The evaluation subsystem (`eval_cli.py` + `evaluation.py`) provides a rigorous, reproducible framework for assessing how well the LLAMBO-style forecast performs — and whether the hardening layer actually improves things.

## Purpose

- Compare LLAMBO (baseline and hardened) against simple statistical baselines.
- Detect regressions introduced by hardening.
- Stress-test on synthetic data since real-world history is typically very short.
- Produce human-readable reports with quantitative evidence.

## Rolling-Origin Backtest

The core evaluation method is a **rolling-origin** (expanding-window) backtest:

```
Time ──────────────────────────────────────────►

  [train window]  | predict h=1,2,...,6
  [train window + 1 month]  | predict h=1,2,...,6
  [train window + 2 months]  | predict h=1,2,...,6
  ...
```

For each origin (training cutoff), the system forecasts 1 to `max_horizon` months ahead, then compares the prediction to the known held-out value.

## Models Compared

| Model | Key | Description |
|---|---|---|
| LLAMBO baseline | `llambo_style:baseline` | Surrogate + BO loop, hardening disabled |
| LLAMBO hardened | `llambo_style:hardened` | Surrogate + BO loop, hardening enabled |
| Persistence | `persistence` | Last observed value, uncertainty ∝ √horizon |
| Linear trend | `linear_trend` | OLS fit on training window |
| Logistic S-curve | `logistic_scurve` | Grid-search logistic fit (fails gracefully on short data) |
| GP surrogate | `gp_surrogate` | Gaussian Process with RBF kernel |

## Scenarios

Four transcript-derived scenarios are tested:

| Scenario | Growth Prior | Yield-Killer Context |
|---|---|---|
| `prior_7_8_with_killer` | 7-8 % | Yes (risk, variability, delay, …) |
| `prior_3_5_with_killer` | 3-5 % | Yes |
| `prior_10_12_with_killer` | 10-12 % | Yes |
| `prior_7_8_no_killer` | 7-8 % | No |

The first scenario (`prior_7_8_with_killer`) is the **primary baseline** — all models are evaluated under it. The other scenarios only test the two LLAMBO variants, to measure context sensitivity.

## Datasets

### Real

- `sample_observations` — the CSV provided in `data/sample_observations.csv` (typically only 2 points, so very limited).

### Synthetic (enabled by default)

Generated deterministically from the seed:

| Dataset | Characteristics |
|---|---|
| `synthetic_clean` | Smooth logistic ramp (36→92 %) + mild sinusoidal seasonality. 30 months. |
| `synthetic_noisy` | Clean ramp + Gaussian noise (σ = 2.4). |
| `synthetic_outlier` | Noisy ramp + three large negative shocks at months 10, 19, 24. |

These let the harness test edge cases (outlier robustness, noise sensitivity) that the short real history cannot expose.

## Metrics

| Metric | Formula / Meaning |
|---|---|
| **MAE** | Mean of |y_true − y_pred| |
| **RMSE** | √(Mean of (y_true − y_pred)²) |
| **CRPS** | Continuous Ranked Probability Score — measures full-distribution calibration under a Gaussian assumption |
| **Coverage95** | Fraction of true values that fall inside the predicted 95 % CI |
| **Mean Interval Width** | Average `ci95_high − ci95_low` — narrower is better if coverage is adequate |
| **Calibration Error** | Mean absolute deviation between nominal and empirical PIT quantiles (9 bins from 0.1 to 0.9) |

## Output Files

| File | Content |
|---|---|
| `metrics_summary.csv` | One row per (dataset, scenario, model, horizon) with all metrics |
| `backtest_predictions.csv` | Every individual prediction (dataset, origin, target, predicted, true, status) |
| `ablation_comparison.csv` | Side-by-side baseline vs. hardened for every metric, with deltas |
| `calibration_plot.png` | Nominal-vs-empirical quantile plot for each model |
| `benchmark_plot.png` | Grouped bar chart of RMSE and MAE across all models |
| `critical_assessment_hardened.md` | Auto-generated markdown with headline results, regressions, and verdict |

## Critical Assessment

The auto-generated `critical_assessment_hardened.md` includes:

1. **Data reality** — acknowledges the weakness of the real sample size.
2. **Hardening configuration** — documents every parameter used.
3. **Headline results** — RMSE, coverage, calibration for baseline vs. hardened LLAMBO.
4. **Stress findings** — how each hardening feature affected robustness.
5. **Regression check** — flags any metric where hardening made things worse.
6. **Decision verdict** — a conservative judgment on whether to adopt the hardened model.

## Running the Evaluation

See [CLI Usage](cli-usage.md) for the full flag reference. Quick start:

```bash
PYTHONPATH=src python -m intel_18a_llambo.eval_cli \
  --observations-csv data/sample_observations.csv \
  --output-dir outputs/quality_hardened \
  --max-horizon 6 \
  --seed 18
```
