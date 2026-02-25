# Critical Assessment: Intel 18A LLAMBO Hardening Evaluation

## Scope and Data Reality
- Backtesting used rolling-origin monthly forecasts with horizons 1-6 months.
- Observed dataset(s): `enriched_monthly_panel`.
- Observed backtest usable points: 1680.
- Observed panel is materially larger than the thin baseline, but still not a substitute for true measured fab yield telemetry.

## Synthetic Stress Datasets Used
- `enriched_monthly_panel`: Observed dataset loaded from `data/processed/enriched_monthly_panel.csv`.
- `synthetic_clean`: Synthetic transparent stress-test: smooth logistic ramp + mild seasonality.
- `synthetic_noisy`: Synthetic transparent stress-test: clean ramp + Gaussian noise.
- `synthetic_outlier`: Synthetic transparent stress-test: noisy ramp + explicit negative outlier shocks.

## Hardening Configuration
- prior_weight=0.65, robust_likelihood=huber, huber_delta=1.75, student_t_df=5.0
- context_drift_clip=0.020, outlier_z_clip=3.25, outlier_std_inflation=1.50
- interval_calibration=isotonic, fallback=quantile_scale, calibration_min_points=10

## Headline Results (Baseline Scenario: prior 7-8% with yield-killer context)
- Best RMSE model on aggregated synthetic datasets: `llambo_style:hardened` (RMSE=3.904).
- LLAMBO baseline RMSE=5.604, hardened RMSE=3.904, delta=-1.700.
- LLAMBO baseline coverage95=0.757, hardened coverage95=0.915, delta=+0.158.
- LLAMBO baseline calibration_error=0.207, hardened calibration_error=0.119, delta=-0.088.
- LLAMBO outlier RMSE baseline=6.893, hardened=5.503, delta=-1.390.
- Area-factor impact (hardened with-area minus hardened no-area, RMSE): +0.000.

## Stress and Sensitivity Findings
- Scenario prior from the attached fictional case study is treated as a subjective prior and down-weighted by reliability before blending with data anchor growth.
- Prior-weight regularization and context-drift clipping reduced narrative over-dominance in high-guidance transcripts, but may under-react in genuinely fast-improving regimes.
- Outlier-aware variance inflation improved tail coverage stability in shock windows where baseline LLAMBO was over-confident.
- Isotonic/quantile interval recalibration changed uncertainty bands without changing mean dynamics, so point accuracy gains can remain small.
- Logistic baseline graceful failures recorded: 0 (expected on short windows or poor curvature fit).

## Failure Modes and Confidence Limits
- Extremely limited real historical data means external validity is weak; synthetic success does not imply deployment readiness.
- Hardening did not fully eliminate degradation under severe synthetic outliers; heavy-tail approximation helps but is not a full probabilistic model.
- Context-sensitive LLAMBO behavior can still be brittle if transcript sentiment is noisy, biased, or strategically framed.
- GP and trend baselines can outperform context-heavy models when the underlying signal is mostly smooth and data-rich.

## Regression Check
- Regressions detected: none by configured thresholds.
- Improvement classification: meaningful on selected risk metrics.

## Decision Verdict
- Verdict: conditionally improved, still limited by data.
- This harness is suitable for model triage and robustness diagnostics, not for high-confidence capital-allocation decisions without materially longer real-world history.
