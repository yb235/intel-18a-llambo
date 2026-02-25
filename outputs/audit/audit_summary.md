# Audit Summary

## Artifact coverage
- Generated required plots: `bayesian_update_timeline.png`, `posterior_evolution.png`, `scenario_comparison.png`, `lineage_graph.svg`.
- Generated trace artifacts: `trace_matrix.csv`, `audit_dashboard.html`, `audit_summary.md`.

## Data category handling (explicit)
- Directly observed: `2` monthly anchors from `yield_source=observed_anchor` (Jan/Feb 2026 hard anchors).
- Proxy-derived: `24` monthly points from engineered proxy transforms in `build_enriched_panel.py`.
- Subjective prior: `1` trace rows tied to `subjective_prior` source tier and reliability weighting.

## Bayesian update interpretation
- Timeline reconstructs month-by-month prior -> likelihood -> posterior updates for auditability.
- Uncertainty bands are shown at 68% and 95% intervals to support audit review of confidence evolution.

## Scenario comparison inputs
- `baseline`: dataset=`sample_observations`, model_version=`baseline`, RMSE=0.3017, MAE=0.3017, coverage95=1.0000.
- `hardened`: dataset=`sample_observations`, model_version=`hardened`, RMSE=1.9000, MAE=1.9000, coverage95=1.0000.
- `enriched`: dataset=`enriched_monthly_panel`, model_version=`hardened`, RMSE=11.3984, MAE=9.5361, coverage95=0.4762.
- `enriched_rerun`: dataset=`enriched_monthly_panel`, model_version=`hardened`, RMSE=6.7123, MAE=5.1670, coverage95=0.7238.
- `enriched_cfo_academic`: dataset=`enriched_monthly_panel`, model_version=`hardened`, RMSE=7.2543, MAE=5.4425, coverage95=0.7143.

## Provenance
- Trace matrix maps each plotted datapoint to source_id/source_url/source_tier/confidence and transform_step.
- Source tiers and confidence labels are pulled from `data/raw/source_manifest.csv` where available.
