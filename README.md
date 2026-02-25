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

## Enriched dataset pipeline (reproducible)

Create enriched monthly features and run evaluation on the enriched panel:

```bash
# Optional: attempt source downloads + hashes (non-fatal if offline)
python3 scripts/ingest/fetch_sources.py

# Deterministic transform from raw source tables -> monthly panel
python3 scripts/ingest/build_enriched_panel.py

# Run evaluation on enriched panel
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python -m intel_18a_llambo.eval_cli \
  --observations-csv data/processed/enriched_monthly_panel.csv \
  --output-dir outputs/quality_enriched \
  --assessment-filename critical_assessment_enriched.md \
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

Expected enriched outputs:

- `outputs/quality_enriched/metrics_summary.csv`
- `outputs/quality_enriched/backtest_predictions.csv`
- `outputs/quality_enriched/calibration_plot.png`
- `outputs/quality_enriched/benchmark_plot.png`
- `outputs/quality_enriched/critical_assessment_enriched.md`

Important caveat:

- `data/processed/enriched_monthly_panel.csv` contains mostly proxy `yield` values derived from public indicators and milestone assumptions; only Jan/Feb 2026 are hard anchors from the original sample observations.

## Targeted rerun: subjective prior + size-factor update

Run this targeted rerun (no full rebuild) after pulling the model/provenance updates in this repo:

```bash
# Rebuild enriched panel with size/area proxy features:
python3 scripts/ingest/build_enriched_panel.py

# Rerun enriched quality evaluation into a separate output folder:
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python3 -m intel_18a_llambo.eval_cli \
  --observations-csv data/processed/enriched_monthly_panel.csv \
  --output-dir outputs/quality_enriched_rerun \
  --assessment-filename critical_assessment_enriched_rerun.md \
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

# Rebuild enriched rerun assessment with explicit deltas vs outputs/quality_enriched:
python3 - <<'PY'
from pathlib import Path
import csv

new_path = Path('outputs/quality_enriched_rerun/metrics_summary.csv')
old_path = Path('outputs/quality_enriched/metrics_summary.csv')
out_path = Path('outputs/quality_enriched_rerun/critical_assessment_enriched_rerun.md')

new_rows = list(csv.DictReader(new_path.open()))
old_rows = list(csv.DictReader(old_path.open()))
metrics = ['mae', 'rmse', 'crps_approx', 'coverage95', 'calibration_error']

def get(rows, dataset, version):
    for row in rows:
        if (
            row['dataset'] == dataset
            and row['scenario'] == 'prior_7_8_with_killer'
            and row['model'] == 'llambo_style'
            and row['model_version'] == version
            and row['horizon'] == 'all'
        ):
            return row
    return None

def val(row, metric):
    if row is None:
        return float('nan')
    raw = row.get(metric, '')
    return float(raw) if raw else float('nan')

def fmt(x):
    return 'n/a' if x != x else f'{x:.4f}'

def fmtd(x):
    return 'n/a' if x != x else f'{x:+.4f}'

obs = 'enriched_monthly_panel'
syn = '__ALL_SYNTHETIC__'
new_h = get(new_rows, obs, 'hardened')
new_hna = get(new_rows, obs, 'hardened_no_area')
old_h = get(old_rows, obs, 'hardened')
new_h_syn = get(new_rows, syn, 'hardened')
new_hna_syn = get(new_rows, syn, 'hardened_no_area')
old_h_syn = get(old_rows, syn, 'hardened')

lines = [
    '# Critical Assessment: Enriched Rerun (Subjective Prior + Size Factor)',
    '',
    '## What Changed',
    '- Added provenance/source-tier classification and registered fictional case-study PDF as `subjective_prior`.',
    '- Treated fictional case-study input as subjective scenario prior, not observed truth; applied reliability down-weighting.',
    '- Added size-factor support (`effective_die_area_mm2_proxy`, `area_factor`) in feature engineering and LLAMBO forecast logic.',
    '',
    '## Impact of Subjective-Prior Treatment',
    '- Approximation: compare `new hardened_no_area` vs `previous hardened`.',
]
for dataset, label, nrow, orow in [
    (obs, 'Observed-enriched panel', new_hna, old_h),
    (syn, 'Synthetic aggregate', new_hna_syn, old_h_syn),
]:
    lines.append(f'- {label}:')
    for metric in metrics:
        delta = val(nrow, metric) - val(orow, metric)
        lines.append(f'  - {metric}: {fmt(val(nrow, metric))} vs {fmt(val(orow, metric))} (delta {fmtd(delta)})')

lines.extend(['', '## Impact of Size/Area Factor', '- Compare `new hardened` vs `new hardened_no_area`.'])
for dataset, label, h, hna in [
    (obs, 'Observed-enriched panel', new_h, new_hna),
    (syn, 'Synthetic aggregate', new_h_syn, new_hna_syn),
]:
    lines.append(f'- {label}:')
    for metric in metrics:
        delta = val(h, metric) - val(hna, metric)
        lines.append(f'  - {metric}: {fmt(val(h, metric))} vs {fmt(val(hna, metric))} (delta {fmtd(delta)})')

lines.extend(['', '## Metric Deltas vs Previous `quality_enriched`', '- Baseline comparison key: `llambo_style:hardened`, `prior_7_8_with_killer`, horizon `all`.'])
for dataset, label, nrow, orow in [
    (obs, 'Observed-enriched panel', new_h, old_h),
    (syn, 'Synthetic aggregate', new_h_syn, old_h_syn),
]:
    lines.append(f'- {label}:')
    for metric in metrics:
        delta = val(nrow, metric) - val(orow, metric)
        lines.append(f'  - {metric}: {fmt(val(orow, metric))} -> {fmt(val(nrow, metric))} (delta {fmtd(delta)})')

lines.extend([
    '',
    '## Caveats',
    '- `effective_die_area_mm2_proxy`/`area_factor` are proxy-based assumptions when die-size telemetry is unavailable.',
    '- Interpret directional changes more strongly than absolute level accuracy.',
])

out_path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')
print(f'Wrote {out_path}')
PY
```
