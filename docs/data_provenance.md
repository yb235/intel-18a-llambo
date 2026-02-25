# Data Provenance

## Scope
This repo now includes a reproducible enrichment pipeline that builds a monthly Intel 18A-focused panel from public Intel IR/news sources.

## Pipeline files
- `scripts/ingest/fetch_sources.py`: optional fetch + hash collection from `data/raw/source_manifest.csv`.
- `scripts/ingest/build_enriched_panel.py`: deterministic transform from raw event/quarterly signals into monthly panel.
- `data/raw/intel_quarterly_financial_signals.csv`: manually extracted quarterly financial signals.
- `data/raw/intel_18a_milestones.csv`: manually extracted dated 18A milestones.
- `data/raw/intel_18a_academic_signals.csv`: dated academic/disclosed technical discussion signals for indirect yield maturity.
- `data/raw/intel_cfo_signals.csv`: quarterly CFO guidance signals for GM trajectory and IFS profitability timing.
- `data/interim/source_hashes.csv`: fetch/hashing run log (best effort in current environment).
- `data/interim/enriched_monthly_features.csv`: expanded monthly feature matrix.
- `data/processed/enriched_monthly_panel.csv`: final model-ready dataset with `month,yield,...`.

## Source records
Canonical source inventory is in `data/raw/source_manifest.csv` with required fields:
- URL
- accessed date
- source tier classification
- confidence tier
- licensing/usage notes
- hash placeholder (populated when fetch succeeds)
- extraction method notes

### Source tiers
- `public_observed`: publicly accessible factual sources (Intel IR/newsroom/pages/PDFs).
- `subjective_prior`: narrative or fictional scenario priors used only as educated-guess priors, never as hard observed truth.
- `tooling_attempt`: metadata for failed/non-authoritative tool attempts.

## Extraction methods
- Financial values: manual extraction from Intel IR quarterly financial results tables.
- Milestones: manual extraction from Intel Newsroom/Intel press releases and Intel-hosted PDF briefing.
- Transform to monthly: quarter values are carried to each month in quarter; milestone stage is stepwise by event date.
- CFO signals: quarter-indexed guidance values are time-carried to each month and down-weighted by source tier + confidence.
- Academic/disclosed technical signals: event-indexed values use recency decay and source tier weighting before monthly alignment.

## Licensing and usage notes
- Sources are public web pages/PDFs from Intel domains.
- This repository stores derived numeric features and short attribution metadata, not full copyrighted page bodies.
- Use short quotations with attribution when reporting findings.

## Integrity notes
- In this environment, outbound network from shell tools is restricted. `fetch_sources.py` records failures instead of hard-failing the pipeline.
- NASA Terminal MCP was attempted through `mcporter` and returned `fetch failed`; this is documented in `data/raw/source_manifest.csv` under `nasa_terminal_attempt`.
- The attached fictional case-study PDF is registered as `subjective_prior` and is only used to shape scenario priors after reliability down-weighting.
- Any row marked `subjective_prior` is down-weighted before becoming a model prior feature and is never treated as hard observed truth.

## Local artifact hashes (SHA-256)
- `data/raw/intel_quarterly_financial_signals.csv`: `abfddadbaecc43c9d7de2acc26a7c59341575ced2668057eb5604b054f6062c1`
- `data/raw/intel_18a_milestones.csv`: `4f051996e384005fe82bd1c255251ff798f6402f0d5a97653447ef5c1baf53fb`
- `data/raw/intel_18a_academic_signals.csv`: `1bcd37378d73d0a8f1f1e00ed1586a36c43ad128b55a4910343a7a9211484100`
- `data/raw/intel_cfo_signals.csv`: `0456e9ce8ec7d7e6057b586abba2ddcd7e6b436d8b8886508e6c7090a29cb26d`
- `data/raw/source_manifest.csv`: `00c6046f9b5ce14fb2fdbeefd88581f3b72503d5f7f0c8bbfc583b5134a2d9fc`
- `data/interim/source_hashes.csv`: `aa008d8113a456744fd19b2dfa5080c7b8ea105198d131d9ddc63a7dd7ca641f`
- `data/processed/enriched_monthly_panel.csv`: `42a36b107d381ddd8900c71b4bf6b8960a4d8b30457562b2d5eefb851b7b2116`

## Skepticism and limits
- `yield` in `data/processed/enriched_monthly_panel.csv` is mostly a proxy target inferred from public indicators and two observed anchors (`2026-01=64.0`, `2026-02=68.5`).
- Treat this as a stress-test dataset for methodology comparison, not as measured fab yield telemetry.
- `effective_die_area_mm2_proxy` and `area_factor` are engineered proxies when true die-size telemetry is unavailable.
- `cfo_gm_signal_strength`, `ifs_profitability_timeline_score`, and `academic_yield_maturity_signal` are directional priors, not direct yield measurements.
