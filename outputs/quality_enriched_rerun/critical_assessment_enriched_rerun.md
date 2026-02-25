# Critical Assessment: Enriched Rerun (Subjective Prior + Size Factor)

## What Changed
- Added provenance/source-tier classification and recorded fictional case-study PDF as `subjective_prior` (educated-guess prior only).
- Updated scenario handling: subjective priors are reliability-down-weighted before blending with data anchor growth.
- Added explicit size geometry support via `area_factor` and `effective_die_area_mm2_proxy` feature engineering in the enriched panel.
- Updated LLAMBO forecast logic to use `area_factor` in posterior mean/std behavior, plus a `hardened_no_area` ablation variant for isolation.

## Impact of Subjective-Prior Treatment
- Isolated approximation: compare `new hardened_no_area` vs `previous hardened` (keeps area disabled to isolate prior handling).
- Observed-enriched panel:
  - mae: 6.0184 vs 9.5361 (delta -3.5177)
  - rmse: 7.7195 vs 11.3984 (delta -3.6789)
  - crps_approx: 4.6932 vs 7.6961 (delta -3.0029)
  - coverage95: 0.6667 vs 0.4762 (delta +0.1905)
  - calibration_error: 0.2788 vs 0.3974 (delta -0.1186)
- Synthetic aggregate:
  - mae: 2.8372 vs 2.5456 (delta +0.2916)
  - rmse: 3.9037 vs 3.8380 (delta +0.0657)
  - crps_approx: 2.0823 vs 1.9343 (delta +0.1480)
  - coverage95: 0.9147 vs 0.9354 (delta -0.0207)
  - calibration_error: 0.1190 vs 0.0382 (delta +0.0808)

## Impact of Size/Area Factor
- Isolated comparison: `new hardened` vs `new hardened_no_area` on same rerun outputs.
- Observed-enriched panel:
  - mae: 5.1670 vs 6.0184 (delta -0.8514)
  - rmse: 6.7123 vs 7.7195 (delta -1.0072)
  - crps_approx: 3.9446 vs 4.6932 (delta -0.7486)
  - coverage95: 0.7238 vs 0.6667 (delta +0.0571)
  - calibration_error: 0.2238 vs 0.2788 (delta -0.0550)
- Synthetic aggregate:
  - mae: 2.8372 vs 2.8372 (delta +0.0000)
  - rmse: 3.9037 vs 3.9037 (delta +0.0000)
  - crps_approx: 2.0823 vs 2.0823 (delta +0.0000)
  - coverage95: 0.9147 vs 0.9147 (delta +0.0000)
  - calibration_error: 0.1190 vs 0.1190 (delta +0.0000)

## Metric Deltas vs Previous `quality_enriched`
- Baseline model for comparison: `llambo_style:hardened` in scenario `prior_7_8_with_killer`, horizon `all`.
- Observed-enriched panel:
  - mae: 9.5361 -> 5.1670 (delta -4.3691)
  - rmse: 11.3984 -> 6.7123 (delta -4.6861)
  - crps_approx: 7.6961 -> 3.9446 (delta -3.7515)
  - coverage95: 0.4762 -> 0.7238 (delta +0.2476)
  - calibration_error: 0.3974 -> 0.2238 (delta -0.1736)
- Synthetic aggregate:
  - mae: 2.5456 -> 2.8372 (delta +0.2916)
  - rmse: 3.8380 -> 3.9037 (delta +0.0657)
  - crps_approx: 1.9343 -> 2.0823 (delta +0.1480)
  - coverage95: 0.9354 -> 0.9147 (delta -0.0207)
  - calibration_error: 0.0382 -> 0.1190 (delta +0.0808)

## Caveats
- `effective_die_area_mm2_proxy` and `area_factor` are proxy features, not measured die-size telemetry.
- Proxy assumptions can bias both level and trend; directional interpretation is safer than absolute accuracy claims.
- The fictional case-study prior remains subjective and should inform scenarios, not be interpreted as observed fab truth.
