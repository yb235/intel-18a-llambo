# Getting Started

This guide walks you through setting up the project and running your first yield forecast.

## Prerequisites

- **Python 3.10+**
- `pip` (bundled with Python)
- (Optional) `git` — only needed if you want to inspect the vendored LLAMBO repository under `external/LLAMBO/`

## 1. Clone the Repository

```bash
git clone https://github.com/yb235/intel-18a-llambo.git
cd intel-18a-llambo
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

## 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The only runtime dependencies are **NumPy ≥ 1.26** and **Matplotlib ≥ 3.8**.

For an editable (development) install that also registers the CLI entry-points:

```bash
pip install -e .
```

## 4. Verify the Installation

```bash
python -m compileall src                 # byte-compile all modules — no errors expected
PYTHONPATH=src python -c "from intel_18a_llambo import run_forecast; print('OK')"
```

## 5. Run Your First Forecast

### Option A — From the sample CSV + transcript

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/intel18a_yield_curve.png \
  --seed 18 \
  --horizon 2026-08
```

### Option B — Inline observations only (no transcript)

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-inline "Jan=64, Feb=68.5" \
  --output-csv outputs/forecast_inline.csv \
  --output-plot outputs/intel18a_yield_curve_inline.png \
  --seed 18 \
  --horizon 2026-08
```

Both commands produce:

| Output | Description |
|---|---|
| `outputs/forecast.csv` | Month-by-month posterior mean, stddev, 95 % CI |
| `outputs/intel18a_yield_curve.png` | Learning-curve chart (observed + forecast) |

### Reading the Console Output

The CLI prints a summary table:

```
Month      Mean   StdDev   CI95
2026-01   64.00    0.00  [ 64.00,  64.00]
2026-02   68.50    0.00  [ 68.50,  68.50]
2026-03   72.14    1.58  [ 69.04,  75.24]
...
```

- **Observed months** have `StdDev = 0` and a collapsed CI — they are known data.
- **Forecast months** have non-zero uncertainty that grows the further ahead you project.

## 6. Run the Quality Evaluation (Optional)

```bash
PYTHONPATH=src python -m intel_18a_llambo.eval_cli \
  --observations-csv data/sample_observations.csv \
  --output-dir outputs/quality_hardened \
  --max-horizon 6 \
  --seed 18
```

See [Evaluation](evaluation.md) for the full flag reference and output description.

## Next Steps

- [Architecture](architecture.md) — understand how the modules fit together.
- [CLI Usage](cli-usage.md) — complete flag reference for both CLIs.
- [Configuration](configuration.md) — tune hardening, calibration, and Bayesian parameters.
