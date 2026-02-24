# Critical Assessment: Intel 18A LLAMBO Quality Evaluation

## Scope and Data Reality
- Backtesting used rolling-origin monthly forecasts with horizons 1-6 months.
- `sample_observations.csv` is too short for serious validation (2 points only); synthetic stress datasets were therefore required and are explicitly labeled.
- Sample backtest usable points: 7 (statistically weak).

## Synthetic Stress Datasets Used
- `sample_observations`: Real sample dataset from project repository.
- `synthetic_clean`: Synthetic transparent stress-test: smooth logistic ramp + mild seasonality.
- `synthetic_noisy`: Synthetic transparent stress-test: clean ramp + Gaussian noise.
- `synthetic_outlier`: Synthetic transparent stress-test: noisy ramp + explicit negative outlier shocks.

## Headline Results (Baseline Scenario: prior 7-8% with yield-killer context)
- Best RMSE model on aggregated synthetic datasets: `logistic_scurve` (RMSE=5.491).
- LLAMBO-style RMSE: 5.604.
- LLAMBO 95% interval coverage: 0.757.
- LLAMBO calibration error: 0.207.

## Stress and Sensitivity Findings
- LLAMBO outlier sensitivity (RMSE clean -> outlier): 4.833 -> 6.893.
- Management-prior shifts (3-5% vs 7-8% vs 10-12%) expose whether LLAMBO predictions are guidance-dominated rather than data-dominated.
- Transcript ablation confirms context terms can materially move forecasts, which is useful for scenario analysis but increases narrative-risk exposure.
- Logistic baseline graceful failures recorded: 1 (expected on short windows or poor curvature fit).

## Failure Modes and Confidence Limits
- Extremely limited real historical data means external validity is weak; synthetic success does not imply deployment readiness.
- Coverage and calibration can degrade sharply under outliers, showing underestimation of tail risk in some regimes.
- Context-sensitive LLAMBO behavior can be brittle if transcript sentiment is noisy, biased, or strategically framed.
- GP and trend baselines can outperform context-heavy models when the underlying signal is mostly smooth and data-rich.

## Decision-Usable Verdict
- Verdict: not yet.
- This harness is suitable for model triage and robustness diagnostics, not for high-confidence capital-allocation decisions without materially longer real-world history.
