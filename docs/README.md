# Intel 18A LLAMBO Documentation

> Complete documentation for the Intel 18A yield forecasting system using LLAMBO-style Bayesian optimization.

---

## 📚 Documentation Index

| Document | Description |
|----------|-------------|
| [model_guide.md](model_guide.md) | Complete model architecture, processes, and algorithms explained for beginners |
| [data_ingestion_guide.md](data_ingestion_guide.md) | How structured and unstructured data are ingested, transformed, and fed into the model |
| [data_provenance.md](data_provenance.md) | Source tracking and reliability classification for all data inputs |
| [architecture.md](architecture.md) | Module-by-module code walkthrough — how every Python file works and how they connect |
| [api_reference.md](api_reference.md) | Complete CLI argument reference and Python API (functions, dataclasses, entry points) |
| [glossary.md](glossary.md) | Plain-English definitions of every technical term used in this project |
| [troubleshooting.md](troubleshooting.md) | Common errors, error messages, and step-by-step fixes |

---

## 🎯 Quick Reference

### What This Project Does

Forecasts Intel 18A process node yield progression using:
- **Structured data**: Financial metrics, milestone stages, yield observations
- **Unstructured data**: Management transcripts, technical disclosures
- **Bayesian optimization**: LLAMBO-style surrogate model with acquisition functions

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│  DATA INGESTION          →  FEATURE ENGINEERING             │
│  • CSV parsing              • Monthly alignment             │
│  • Text extraction          • Z-score normalization         │
│  • Signal weighting         • Proxy yield construction      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT GENERATION                                        │
│  • Guidance extraction (7-8% → 0.07-0.08)                  │
│  • Sentiment scoring (word counting)                       │
│  • S-curve parameters                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  LLAMBO SURROGATE MODEL                                    │
│  • Posterior prediction: mean = prev + headroom×growth     │
│  • S-curve dynamics: phase_gain slows near midpoint        │
│  • Area factor: die size penalty                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  ACQUISITION FUNCTION (Expected Improvement)               │
│  • Balances exploitation vs exploration                    │
│  • Picks best growth rate candidate                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  BAYESIAN LOOP                                             │
│  • Month-by-month iteration                                │
│  • Uncertainty propagation                                 │
│  • Hardening for robustness                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT                                                    │
│  • Forecast CSV (mean, stddev, CI95 per month)             │
│  • Calibration plots                                       │
│  • Evaluation metrics (MAE, RMSE, CRPS, coverage)          │
└─────────────────────────────────────────────────────────────┘
```

### Key Metrics (Quality Enriched Rerun)

| Metric | Baseline | Hardened | Improvement |
|--------|----------|----------|-------------|
| **MAE** | 9.54 | 5.17 | -4.37 |
| **RMSE** | 11.40 | 6.71 | -4.69 |
| **Coverage95** | 47.6% | 72.4% | +24.8% |
| **Calibration Error** | 0.40 | 0.22 | -0.18 |

---

## 🚀 Quick Start

### Basic Forecast

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/yield_curve.png \
  --horizon 2026-08
```

### Quality Evaluation

```bash
PYTHONPATH=src python -m intel_18a_llambo.eval_cli \
  --observations-csv data/processed/enriched_monthly_panel.csv \
  --output-dir outputs/quality_enriched \
  --max-horizon 6 \
  --prior-weight 0.65 \
  --robust-likelihood huber \
  --interval-calibration isotonic
```

---

## 📁 Project Structure

```
intel-18a-llambo/
├── docs/                          # Documentation (you are here)
│   ├── README.md                  # This index
│   ├── model_guide.md             # Model architecture
│   ├── data_ingestion_guide.md    # Data pipeline
│   ├── data_provenance.md         # Source tracking
│   ├── architecture.md            # Code structure & module guide
│   ├── api_reference.md           # CLI & Python API reference
│   ├── glossary.md                # Technical terms explained
│   └── troubleshooting.md         # Common errors and fixes
├── src/intel_18a_llambo/          # Source code
│   ├── ingestion.py               # Data loading
│   ├── context.py                 # Text → context extraction
│   ├── surrogate.py               # Prediction engine
│   ├── bayes_loop.py              # Bayesian iteration
│   ├── hardening.py               # Robustness tweaks
│   ├── evaluation.py              # Backtesting
│   ├── cli.py                     # Command-line interface
│   └── eval_cli.py                # Evaluation CLI
├── data/
│   ├── raw/                       # Source CSVs
│   ├── interim/                   # Intermediate files
│   ├── processed/                 # Enriched panel
│   └── sample_*.csv               # Example inputs
├── scripts/
│   └── ingest/
│       ├── fetch_sources.py       # Download sources
│       └── build_enriched_panel.py # Feature engineering
├── outputs/                       # Generated outputs
└── external/LLAMBO/               # Reference implementation
```

---

## 📖 Detailed Documentation

For in-depth explanations, see:

1. **[model_guide.md](model_guide.md)** - Learn how the model works:
   - Data ingestion → Context generation → Surrogate model
   - Acquisition functions → Bayesian loop → Hardening
   - Evaluation metrics → Output interpretation

2. **[data_ingestion_guide.md](data_ingestion_guide.md)** - Understand the data pipeline:
   - Structured data (CSV) → parsing → normalization
   - Unstructured data (text) → extraction → numerical signals
   - Feature engineering → proxy yield construction
   - Source tier and confidence weighting

3. **[data_provenance.md](data_provenance.md)** - Track data sources:
   - Where each data point came from
   - Reliability classifications
   - Confidence labels

4. **[architecture.md](architecture.md)** - Understand the code structure:
   - Module-by-module explanation of every Python file
   - How the files relate to each other (dependency graph)
   - Data flow from raw inputs to forecast output

5. **[api_reference.md](api_reference.md)** - Use the CLI and Python API:
   - All command-line arguments with types, defaults, and descriptions
   - Output file formats explained (CSV columns, plot descriptions)
   - Python functions and dataclasses for programmatic use

6. **[glossary.md](glossary.md)** - Look up any term:
   - Plain-English definitions for every technical concept
   - Bayesian optimization, LLAMBO, S-curve, yield, EI, and more

7. **[troubleshooting.md](troubleshooting.md)** - Fix problems fast:
   - Common error messages and their causes
   - Step-by-step fixes for setup, data, and runtime issues

---

*Last updated: 2026-02-25*
