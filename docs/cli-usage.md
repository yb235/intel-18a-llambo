# CLI Usage

The project provides two command-line interfaces.

---

## 1. Forecast CLI — `intel_18a_llambo.cli`

Produces a month-by-month yield forecast.

### Invocation

```bash
# Using PYTHONPATH
PYTHONPATH=src python -m intel_18a_llambo.cli [OPTIONS]

# After editable install (pip install -e .)
intel18a-yield [OPTIONS]
```

### Required Input (one of)

| Flag | Description |
|---|---|
| `--observations-csv PATH` | CSV file with `month,yield` columns |
| `--observations-inline STR` | Inline observations, e.g. `"Jan=64, Feb=68.5"` |

If neither is provided, the CLI defaults to `"Jan=64, Feb=68.5"`.

### Optional Flags

| Flag | Default | Description |
|---|---|---|
| `--transcript-files PATH [PATH …]` | *(none)* | One or more local transcript text files |
| `--horizon YYYY-MM` | `2026-08` | Forecast until this month |
| `--months-ahead N` | `6` | Minimum number of months to forecast |
| `--output-csv PATH` | `outputs/forecast.csv` | Path for the output CSV |
| `--output-plot PATH` | `outputs/intel18a_yield_curve.png` | Path for the output PNG chart |
| `--seed N` | `18` | Random seed for reproducibility |
| `--print-context` | *(off)* | Print the generated task context description |
| `--disable-hardening` | *(off)* | Use baseline LLAMBO (no hardening layer) |

### Hardening Flags

These are active only when hardening is enabled (i.e., `--disable-hardening` is **not** passed).

| Flag | Default | Description |
|---|---|---|
| `--prior-weight F` | `0.65` | Weight on context prior vs. data anchor (0–1) |
| `--robust-likelihood {none,huber,student_t}` | `huber` | Heavy-tail approximation mode |
| `--huber-delta F` | `1.75` | Huber threshold (standardized residual units) |
| `--student-t-df F` | `5.0` | Student-t degrees of freedom |
| `--context-drift-clip F` | `0.02` | Cap on transcript-induced context drift |
| `--outlier-z-clip F` | `3.25` | Outlier clipping threshold |
| `--outlier-std-inflation F` | `1.5` | Stddev inflation factor under outlier pressure |
| `--interval-calibration {none,isotonic,quantile_scale}` | `isotonic` | Predictive interval recalibration mode |
| `--calibration-fallback {none,quantile_scale}` | `quantile_scale` | Fallback when isotonic has too few points |
| `--calibration-min-points N` | `10` | Minimum residual points for calibrator |
| `--interval-alpha F` | `0.95` | Target central coverage for intervals |

### Example: Full Run with Transcript

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/intel18a_yield_curve.png \
  --seed 18 \
  --horizon 2026-08 \
  --print-context
```

### Example: Baseline (No Hardening)

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-inline "Jan=64, Feb=68.5" \
  --horizon 2026-08 \
  --disable-hardening
```

---

## 2. Evaluation CLI — `intel_18a_llambo.eval_cli`

Runs a rolling-origin backtest and quality evaluation comparing LLAMBO (baseline + hardened) against statistical baselines.

### Invocation

```bash
# Using PYTHONPATH
PYTHONPATH=src python -m intel_18a_llambo.eval_cli [OPTIONS]

# After editable install
intel18a-yield-eval [OPTIONS]
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--observations-csv PATH` | `data/sample_observations.csv` | Input CSV |
| `--transcript-files PATH [PATH …]` | *(none)* | Transcripts (accepted for parity) |
| `--output-dir PATH` | `outputs/quality` | Directory for all evaluation outputs |
| `--max-horizon N` | `6` | Maximum monthly horizon for rolling backtests |
| `--seed N` | `18` | Random seed |
| `--no-synthetic` | *(off)* | Disable synthetic stress-test datasets |
| `--disable-plots` | *(off)* | Skip plot generation |
| `--disable-hardening` | *(off)* | Disable hardened LLAMBO variant |

All hardening flags from the forecast CLI are also accepted.

### Example: Full Evaluation

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

### Expected Outputs

| File | Description |
|---|---|
| `metrics_summary.csv` | Aggregated metrics per dataset × scenario × model × horizon |
| `backtest_predictions.csv` | Every individual backtest prediction |
| `ablation_comparison.csv` | Side-by-side baseline vs. hardened LLAMBO |
| `calibration_plot.png` | Calibration curve (nominal vs. empirical quantiles) |
| `benchmark_plot.png` | RMSE / MAE bar chart across models |
| `critical_assessment_hardened.md` | Auto-generated findings, regressions, verdict |
