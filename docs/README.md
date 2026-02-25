# Intel 18A LLAMBO — Documentation

Welcome to the documentation for the **Intel 18A LLAMBO Yield Forecast** project. This guide is designed for first-time users who want to understand, run, and extend the codebase.

## What Is This Project?

This project implements a **Bayesian yield-forecasting system** for Intel's 18A semiconductor process node. It uses a **LLAMBO-style surrogate model** (Language-model-based Bayesian Optimization) combined with a Bayesian acquisition loop to forecast monthly wafer yield progression — producing a mean prediction, uncertainty band, and 95 % confidence interval for each future month.

Key capabilities:

- Ingests numeric yield observations (CSV or inline) and optional management-transcript text.
- Builds a rich task context from transcript sentiment (RibbonFET/PowerVia mentions, growth guidance, risk signals).
- Runs a surrogate + Expected Improvement acquisition loop to propose the most likely yield trajectory.
- Outputs a CSV forecast, a learning-curve PNG chart, and (optionally) a full quality-evaluation report with calibration plots, benchmark comparisons, and a critical assessment.

## Table of Contents

| Document | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation, setup, and your first forecast |
| [Architecture](architecture.md) | High-level system design and module map |
| [Workflow](workflow.md) | End-to-end data flow from input to output |
| [Modules](modules.md) | Detailed explanation of every source module |
| [API Reference](api-reference.md) | Public classes, functions, and their signatures |
| [CLI Usage](cli-usage.md) | Command-line interface flags and examples |
| [Configuration](configuration.md) | All tuneable parameters (hardening, calibration, …) |
| [Evaluation](evaluation.md) | Quality-evaluation harness and benchmarking |
| [Glossary](glossary.md) | Key terms and concepts |

## Quick Links

- **Source code:** `src/intel_18a_llambo/`
- **Sample data:** `data/sample_observations.csv`, `data/sample_transcript_q1_2026.txt`
- **Outputs:** `outputs/`
- **Upstream LLAMBO repo (local clone):** `external/LLAMBO/`
