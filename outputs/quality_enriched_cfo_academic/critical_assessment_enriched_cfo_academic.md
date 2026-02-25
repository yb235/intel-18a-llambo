# Critical Assessment: Enriched CFO + Academic Signals

## Scope
- Comparison key: `llambo_style:hardened`, scenario `prior_7_8_with_killer`, horizon `all`.
- Delta baseline: `outputs/quality_enriched_rerun`.
- CFO and academic/disclosed technical signals are directional priors and not direct yield telemetry.

## Metric Deltas vs `quality_enriched_rerun`
- Observed panel (`enriched_monthly_panel`):
  - mae: 5.1670 -> 5.4425 (delta +0.2755)
  - rmse: 6.7123 -> 7.2543 (delta +0.5420)
  - crps_approx: 3.9446 -> 4.2255 (delta +0.2809)
  - coverage95: 0.7238 -> 0.7143 (delta -0.0095)
  - calibration_error: 0.2238 -> 0.1984 (delta -0.0254)
- Synthetic aggregate (`__ALL_SYNTHETIC__`):
  - mae: 2.8372 -> 2.8372 (delta +0.0000)
  - rmse: 3.9037 -> 3.9037 (delta +0.0000)
  - crps_approx: 2.0823 -> 2.0823 (delta +0.0000)
  - coverage95: 0.9147 -> 0.9147 (delta +0.0000)
  - calibration_error: 0.1190 -> 0.1190 (delta +0.0000)

## Reliability Callout
- Mixed: CFO+academic enrichment changed reliability metrics, but not with a clean improvement signal.
- Reliability check components: lower observed-panel RMSE, lower calibration error, and non-degraded/stronger 95% coverage.

## Caveats
- Most observed points are proxy-derived; only the Jan/Feb 2026 anchors are hard observations in this panel.
- Subjective sources are down-weighted but still inject prior assumptions and should not be treated as factual measurements.
- Small observed sample means metric shifts can be sensitive to a few windows.
