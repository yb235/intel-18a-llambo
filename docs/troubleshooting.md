# Troubleshooting Guide

> Common problems, error messages, and solutions for the Intel 18A LLAMBO system.

---

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Running the Forecast CLI](#running-the-forecast-cli)
3. [Running the Evaluation CLI](#running-the-evaluation-cli)
4. [Data & CSV Errors](#data--csv-errors)
5. [Plot / Matplotlib Issues](#plot--matplotlib-issues)
6. [Running Scripts](#running-scripts)
7. [Audit Visualization](#audit-visualization)
8. [Understanding Outputs](#understanding-outputs)
9. [Getting Help](#getting-help)

---

## Setup & Installation

### `ModuleNotFoundError: No module named 'intel_18a_llambo'`

**Cause:** The package is not on the Python path.

**Fix A** — Use `PYTHONPATH=src` prefix:
```bash
PYTHONPATH=src python -m intel_18a_llambo.cli --help
```

**Fix B** — Install the package in editable mode:
```bash
pip install -e .
intel18a-yield --help
```

---

### `ModuleNotFoundError: No module named 'numpy'` (or `matplotlib`)

**Cause:** Dependencies are not installed.

**Fix:**
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install "numpy>=1.26.0" "matplotlib>=3.8.0"
```

---

### `python3: command not found` or wrong Python version

**Cause:** Python 3.10+ is required.

**Check your version:**
```bash
python3 --version
```

**Fix:** Install Python 3.10 or later, or use `python` instead of `python3` depending on your OS.

---

### Virtual environment issues

If packages seem missing even after installing:

```bash
# Create and activate a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Forecast CLI

### `ValueError: Could not parse inline observations. Example: Jan=64, Feb=68.5`

**Cause:** The `--observations-inline` string uses an unrecognised format.

**Fix:** Use the format `Month=value`, separated by commas or spaces:
```bash
--observations-inline "Jan=64, Feb=68.5"
--observations-inline "2026-01=64, 2026-02=68.5"
```

Supported month tokens: `Jan`, `Feb`, `Mar`, `Apr`, `May`, `Jun`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec` (case-insensitive) or `YYYY-MM` format.

---

### `ValueError: Horizon must be YYYY-MM format.`

**Cause:** `--horizon` was given in a wrong format.

**Fix:**
```bash
--horizon 2026-08    # correct: YYYY-MM
# NOT: --horizon 08-2026 or --horizon 2026/08
```

---

### `ValueError: At least one observation is required.`

**Cause:** The observations list is empty (e.g., the CSV exists but has no data rows, or the inline string parsed no matches).

**Fix:** Check the CSV file has rows below the header, or verify the inline string format.

---

### The forecast only shows observed months, no future predictions

**Cause:** `--horizon` is set to a month that is at or before the last observation month.

**Fix:** Set `--horizon` to a future month:
```bash
--horizon 2026-08   # if last observation is Feb 2026, this gives 6 months
```

Or use `--months-ahead 6` to always forecast at least 6 months ahead regardless of the horizon.

---

### `FileNotFoundError` for CSV or transcript files

**Cause:** The path to the file does not exist or is wrong.

**Fix:**
- Check the path is relative to the current directory, not the `src/` directory
- Run the command from the project root (`intel-18a-llambo/`):

```bash
cd /path/to/intel-18a-llambo
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv ...
```

---

### Output CSV or plot is not being saved

**Cause:** The output directory does not exist and could not be created (e.g., permission error).

**Check:** The CLI auto-creates parent directories. If it fails, ensure you have write permission:
```bash
mkdir -p outputs
```

---

## Running the Evaluation CLI

### Evaluation takes a long time

**Normal behavior:** The evaluation harness runs many scenario × model version × horizon combinations on both real and synthetic datasets. For 6 horizons × 3 model versions × ~6 datasets, this can be 100+ forecast runs.

**Speed up:**
```bash
# Disable synthetic datasets (not recommended for short histories)
--no-synthetic

# Disable plots (saves matplotlib overhead)
--disable-plots

# Reduce max horizon
--max-horizon 3
```

---

### `calibration_error` is high (> 0.20)

**Explanation:** The model's stated 95% intervals are not covering actuals 95% of the time. This is expected with very short histories (< 6 months).

**Remedies:**
- Use the hardened model (default): it applies interval calibration
- Use a longer history if available: the calibrator needs ≥ `calibration-min-points` residuals
- Try `--interval-calibration quantile_scale` as a more stable alternative to isotonic

---

### `coverage95` is 0.0 or NaN in `metrics_summary.csv`

**Cause:** Too few backtest predictions to compute meaningful coverage. This happens when `max-horizon` is larger than the available history.

**Fix:** Reduce `--max-horizon` to be less than `len(observations) - 2`.

---

### `KeyError` or `csv.Error` when reading output CSVs

**Cause:** The evaluation was interrupted mid-run, leaving partial output files.

**Fix:** Delete the output directory and re-run:
```bash
rm -rf outputs/quality_hardened
PYTHONPATH=src python -m intel_18a_llambo.eval_cli ...
```

---

## Data & CSV Errors

### `KeyError: 'yield'` or `KeyError: 'month'`

**Cause:** The CSV file is missing required columns.

**Required columns:** `month` and `yield` (lowercase).

**Fix:** Check your CSV header:
```csv
month,yield
2026-01,64.0
2026-02,68.5
```

---

### Yield values look wrong after loading

**Cause:** `ensure_bounds()` clamps yield to `[0, 100]`. Values outside this range will be silently clamped.

**Check:** Ensure your yield values are percentages (0–100), not fractions (0–1).
- Correct: `64.0` (meaning 64%)
- Wrong: `0.64` (will be clamped to 0.64 instead of 64%)

---

### `area_factor` values are always 1.0

**Cause:** Your CSV does not have an `area_factor`, `effective_die_area_mm2`, or `effective_die_area_mm2_proxy` column. The default is 1.0.

**Fix:** Add one of these columns if you have die-size data, or leave it as-is if you don't (the model will work fine without it).

---

### Month parsing produces unexpected years

**Cause:** `parse_inline_observations` defaults to year 2026 for bare month names like `"Jan"`.

**Fix:** Use ISO format to specify the year explicitly:
```
--observations-inline "2025-11=58, 2025-12=61, 2026-01=64"
```

---

## Plot / Matplotlib Issues

### `cannot connect to X server` or matplotlib backend error

**Cause:** Running in a headless environment (server, CI, Docker) without a display.

**Fix:** The code already sets `matplotlib.use("Agg")` (file-based, no display needed). If you're still seeing this error, ensure no other import is changing the backend before this module loads.

**Alternative:** Set the environment variable:
```bash
MPLBACKEND=Agg PYTHONPATH=src python -m intel_18a_llambo.cli ...
```

---

### Plots are blank or have no data points

**Cause:** `observed_yield` fields are all `None` (only happens if the observations list was empty — see [ValueError above](#valueerror-at-least-one-observation-is-required)).

---

### `MPLCONFIGDIR` warnings or errors

**Cause:** Matplotlib cannot write to its default config directory.

**Fix:** Point it to a writable temp directory:
```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python -m intel_18a_llambo.eval_cli ...
```

---

## Running Scripts

### `build_enriched_panel.py` fails with `FileNotFoundError`

**Cause:** One of the raw CSV files in `data/raw/` is missing.

**Required files:**
- `data/raw/intel_quarterly_financial_signals.csv`
- `data/raw/intel_18a_milestones.csv`
- `data/raw/intel_18a_academic_signals.csv`
- `data/raw/intel_cfo_signals.csv`

**Fix:** Run from the project root:
```bash
python3 scripts/ingest/build_enriched_panel.py
```

---

### `fetch_sources.py` shows `fetch failed` for all URLs

**Cause:** The environment does not have outbound internet access. This is expected behavior — the script records failures gracefully and the pipeline continues.

**Fix:** No fix needed. The script is optional. The raw CSV files are already included in the repository. `fetch_sources.py` is only for verifying provenance hashes when internet access is available.

---

### `write_cfo_academic_assessment.py` says files not found

**Cause:** The referenced evaluation runs (`outputs/quality_enriched_rerun/` and `outputs/quality_enriched_cfo_academic/`) have not been generated yet.

**Fix:** Run the evaluation steps in order:
```bash
# Step 1: Build enriched panel
python3 scripts/ingest/build_enriched_panel.py

# Step 2: Run CFO academic evaluation
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python3 -m intel_18a_llambo.eval_cli \
  --observations-csv data/processed/enriched_monthly_panel.csv \
  --output-dir outputs/quality_enriched_cfo_academic ...

# Step 3: Write assessment
python3 scripts/eval/write_cfo_academic_assessment.py
```

---

## Audit Visualization

### `audit_viz` fails with `Warning: Missing optional file`

**Cause:** The audit visualization tool expects the enriched panel and source manifest CSV to exist.

**Fix:**
1. Run `python3 scripts/ingest/build_enriched_panel.py` first
2. Verify `data/raw/source_manifest.csv` exists (it should be in the repo)
3. Run the audit viz command from the project root

---

### Lineage graph is empty or very sparse

**Cause:** The source manifest has few entries or the panel has no `yield_source` column to connect nodes.

**Expected behavior:** The lineage graph is most useful after running the full enriched panel pipeline. With just `sample_observations.csv`, the graph will be minimal.

---

## Understanding Outputs

### `mae` in metrics_summary.csv is very high (> 10)

**Explanation:** With only 2 real observations (Jan and Feb 2026), the model has almost no history to learn from. The evaluation is running on mostly synthetic data or proxy yields. MAE of 10+ on proxied yield is expected and does not indicate a bug.

**What to compare:** Look at `baseline` vs `hardened` MAE for the *same* dataset. The improvement (hardened should be lower) is the signal, not the absolute value.

---

### The 95% CI is very wide

**Explanation:** Wide CIs are correct behavior. The model is honestly uncertain about the future, especially for longer horizons (months 4-6 ahead). A model that produces narrow CIs but wrong values is *overconfident* and worse.

**Ideal behavior:** CIs widen with each step ahead. If `coverage95 ≈ 0.95`, the width is appropriate for the actual uncertainty.

---

### Forecast mean doesn't match management's stated target

**Explanation:** The model's forecast is a *data-driven prediction*, not a projection of management guidance. The guidance is used as a *prior* blended with the data anchor. If the observed data suggests slower growth than guidance, the blended prediction will be below guidance.

**Adjust:** Lower `--prior-weight` (closer to 0) to rely more on data; raise it (closer to 1) to follow guidance more closely.

---

## Getting Help

1. **Read the docs first:** All key concepts are in `docs/model_guide.md`, `docs/data_ingestion_guide.md`, `docs/architecture.md`, and `docs/glossary.md`.

2. **Run with verbose output:**
   ```bash
   PYTHONPATH=src python -m intel_18a_llambo.cli \
     --observations-inline "Jan=64, Feb=68.5" \
     --print-context \
     --horizon 2026-08
   ```

3. **Verify your installation:**
   ```bash
   python -m compileall src
   PYTHONPATH=src python -m intel_18a_llambo.cli \
     --observations-inline "Jan=64, Feb=68.5" --horizon 2026-08
   ```

4. **Check for import errors:**
   ```python
   from intel_18a_llambo import ForecastPoint, run_forecast
   print("Import OK")
   ```

---

*Last updated: 2026-02-25*
