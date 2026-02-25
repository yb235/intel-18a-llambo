# LLAMBO Data Ingestion — Complete Deep Dive

> How fundamental (structured) and unstructured data are ingested, transformed, and fed into the model.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Structured Data Ingestion](#part-1-structured-data-ingestion)
3. [Unstructured Data Ingestion](#part-2-unstructured-data-ingestion)
4. [Feature Engineering](#part-3-feature-engineering)
5. [Source Tier & Confidence Weighting](#part-4-source-tier--confidence-weighting)
6. [Final Pipeline to Model](#part-5-final-pipeline-to-model)
7. [Methodology Assessment](#is-this-a-consistent-methodology)

---

## The Big Picture

LLAMBO ingests **two types of data**:

| Type | Examples | Format | Purpose |
|------|----------|--------|---------|
| **Structured (Fundamental)** | Yield observations, financial metrics, milestone stages | CSV tables with numbers | Direct inputs to the model |
| **Unstructured (Text)** | Management transcripts, earnings call notes, technical disclosures | Free text | Extract signals → convert to numbers |

Both are **converted to numerical features** before feeding into the Bayesian optimization model.

---

## PART 1: Structured Data Ingestion

### Source Files

| File | What It Contains | Update Frequency |
|------|------------------|------------------|
| `intel_quarterly_financial_signals.csv` | Revenue, gross margin % by quarter | Quarterly |
| `intel_18a_milestones.csv` | Process ramp stages (1-8) | Event-driven |
| `intel_cfo_signals.csv` | CFO commentary signals (-1 to +1) | Quarterly |
| `intel_18a_academic_signals.csv` | Technical maturity signals | Event-driven |

### Example: Financial Data

**Raw CSV** (`data/raw/intel_quarterly_financial_signals.csv`):
```csv
period,period_start,revenue_bil_usd,gross_margin_gaap_pct,confidence
2024Q1,2024-01-01,12.7,41.0,medium
2024Q2,2024-04-01,12.8,35.4,high
2024Q3,2024-07-01,13.3,15.0,high  ← margin crash!
2024Q4,2024-10-01,14.3,39.2,high
```

**Processing Pipeline:**

```
RAW CSV
    ↓
STEP 1: Parse to Python objects
    QuarterlyPoint(
      period_start = date(2024, 7, 1)
      gross_margin_gaap_pct = 15.0
      revenue_bil_usd = 13.3
    )
    ↓
STEP 2: Map to months (quarterly → monthly)
    July 2024  → uses Q3 2024 data
    August 2024 → uses Q3 2024 data
    September 2024 → uses Q3 2024 data
    ↓
STEP 3: Z-score normalization
    All margins: [41.0, 35.4, 15.0, 39.2, ...]
    Mean = 33.6, Std = 9.2
    15.0 → (15.0 - 33.6) / 9.2 = -2.02 z-score
    
    Interpretation: Q3 margin was 2 std devs below mean!
```

### Example: Milestone Data

**Raw CSV** (`data/raw/intel_18a_milestones.csv`):
```csv
event_date,stage_value,description
2023-07-18,1,Ramp-C test chip announced
2024-04-22,2,DoD RAMP-C phase 3
2024-08-01,3,First power-on Panther Lake
2025-04-01,5,Risk production entered
2025-10-09,7,Panther Lake in production
2026-01-01,8,Broad availability start
```

**Processing:**

```
MILESTONE PROGRESSION:
Stage 1 → Stage 2 → Stage 3 → ... → Stage 8
Jul 2023   Apr 2024   Aug 2024         Jan 2026

For any month, find HIGHEST stage reached up to that point:

January 2024:   stage_as_of(2024-01) → stage 1 → norm = 0.125
May 2024:       stage_as_of(2024-05) → stage 2 → norm = 0.250
September 2024: stage_as_of(2024-09) → stage 3 → norm = 0.375
February 2026:  stage_as_of(2026-02) → stage 8 → norm = 1.000
```

---

## PART 2: Unstructured Data Ingestion

### The Challenge

Unstructured data comes as **free text**. The model needs **numbers**, not words.

**Solution:** Extract signals using pattern matching and keyword counting.

### Source: Management Transcripts

**Example transcript** (`data/sample_transcript_q1_2026.txt`):
```
Management noted that Intel 18A is progressing with RibbonFET and 
PowerVia integration maturing in pilot lines. The team reiterated 
a guidance target of around 7-8% monthly yield improvement through 
the mid-2026 learning phase. Risk was acknowledged around process 
variability, but overall confidence improved after recent defect-
density reduction.
```

### Signal Extraction (context.py)

```
RAW TEXT
    ↓
1. GUIDANCE EXTRACTION (regex)
    Pattern: "(\d+)-(\d+)% monthly"
    Match: "7-8% monthly"
    Result: guidance_low = 0.07, guidance_high = 0.08
    ↓
2. TECHNOLOGY KEYWORD DETECTION
    Search "ribbonfet" → FOUND → ribbonfet_mentioned = True
    Search "powervia" → FOUND → powervia_mentioned = True
    ↓
3. SENTIMENT WORD COUNTING
    POSITIVE: ["confidence", "improved", "progress", "ahead", "reduction", "stable"]
    Text scan:
      - "confidence" ×1 ✓
      - "improved" ×1 ✓
      - "progressing" contains "progress" ✓
    transcript_confidence = 3
    
    RISK: ["risk", "variability", "delay", "challenge", "uncertain", "headwind"]
    Text scan:
      - "risk" ×1 ✓
      - "variability" ×1 ✓
    transcript_risk = 2
    ↓
4. S-CURVE PARAMETER ADJUSTMENT
    Base s_curve_midpoint = 78.0
    If BOTH RibbonFET AND PowerVia: s_curve_midpoint = 80.0
    
    s_curve_steepness = 0.14 + min(0.03, 0.003 × confidence)
                      = 0.14 + 0.009 = 0.149
    ↓
FINAL TaskContext OBJECT:
    TaskContext(
      guidance_growth_low = 0.07,
      guidance_growth_high = 0.08,
      guidance_growth_mid = 0.075,
      transcript_confidence = 3.0,
      transcript_risk = 2.0,
      ribbonfet_mentioned = True,
      powervia_mentioned = True,
      s_curve_midpoint = 80.0,
      s_curve_steepness = 0.149,
    )
```

### Source: CFO Commentary Signals

**Raw CSV** (`data/raw/intel_cfo_signals.csv`):
```csv
period,cfo_gm_signal_strength,ifs_profitability_timeline_score,description
2024Q3,-0.30,-0.28,"CFO guidance signaled trough-like margin pressure"
2025Q3,0.10,0.08,"CFO commentary shifted toward recovery momentum"
```

**How signals were created:**
1. Human read CFO's earnings call commentary
2. Assessed sentiment on -1 to +1 scale
3. Recorded confidence in assessment (0-1)

---

## PART 3: Feature Engineering

### Building the Enriched Monthly Panel

All sources merged into single monthly dataset:

```
DATA SOURCES
├── Financial (quarterly)
├── Milestones (events)
├── CFO Signals (quarterly)
└── Academic Signals (events)
        ↓ MERGE ↓
ENRICHED MONTHLY PANEL
(One row per month with all features aligned)
```

### The Proxy Yield Equation

Since actual yield is sparse (only Jan/Feb 2026 observed), the system constructs proxy yield for historical months:

```python
def calculate_proxy_yield(month_index, stage_norm, gm_z, rev_z, 
                          cfo_signal, ifs_signal, academic_signal,
                          disclosure, area_factor):
    # Base trend: time + process maturity
    y = 31.0 + 0.62 * month_index + 17.5 * stage_norm
    
    # Financial signals
    y += 4.2 * gm_z + 1.8 * rev_z
    
    # Commentary signals
    y += 2.6 * cfo_signal + 2.9 * ifs_signal + 3.4 * academic_signal
    
    # Disclosure confidence
    y += 1.2 * (disclosure - 0.5)
    
    # Die size penalty
    y -= 6.5 * (area_factor - 1.0)
    
    return clip(y, 20.0, 92.0)
```

### Feature Breakdown

| Feature | Source | Transformation | Range |
|---------|--------|----------------|-------|
| `month_index` | Calculated | 0, 1, 2, ... | 0-32 |
| `milestone_stage` | Milestones CSV | stage_as_of(month) | 1-8 |
| `milestone_stage_norm` | Calculated | stage / 8.0 | 0.125-1.0 |
| `gross_margin_gaap_pct` | Financial CSV | Direct | 15-41% |
| `gm_z`, `rev_z` | Calculated | Z-score | ~-2 to +2 |
| `cfo_gm_signal_strength` | CFO Signals | Weighted | -0.3 to +0.18 |
| `ifs_profitability_timeline_score` | CFO Signals | Weighted | -0.28 to +0.22 |
| `academic_yield_maturity_signal` | Academic CSV | Decay-weighted | 0 to 0.68 |
| `disclosure_confidence` | Multiple | Blended | 0.38-0.93 |
| `effective_die_area_mm2_proxy` | Calculated | Formula | 150-230 mm² |
| `area_factor` | Calculated | Normalized | 0.6-1.6 |
| `yield` | Multiple | Proxy OR anchor | 20-92% |
| `yield_source` | Calculated | Provenance tag | observed/proxy |

---

## PART 4: Source Tier & Confidence Weighting

### The Reliability Problem

Not all data is equally trustworthy:
- **Public observed** (earnings) → High reliability
- **Management guidance** → Medium (can be optimistic)
- **Subjective priors** (assumptions) → Low reliability

### Weighting System

```python
def tier_weight(source_tier: str) -> float:
    if source_tier == "public_observed":
        return 1.0      # Full weight
    if source_tier == "subjective_prior":
        return 0.35     # Down-weighted
    if source_tier == "tooling_attempt":
        return 0.1      # Barely trusted
    return 0.6

def confidence_weight(confidence: str) -> float:
    if confidence == "high":
        return 1.0
    if confidence == "medium":
        return 0.8
    if confidence == "low":
        return 0.6
    return 0.75
```

### Time Decay

Older signals are less relevant:

```python
def academic_signal_as_of(month, events):
    for event in events[-4:]:
        months_since = months_between(month, event.date)
        decay = 0.86 ** months_since  # Exponential decay
        
        w = decay * tier_weight(event.tier) * confidence_weight(event.conf)
        weighted_signal += event.value * w
```

**Example:**
- Event 3 months ago: decay = 0.86³ = 0.636
- Event 12 months ago: decay = 0.86¹² = 0.163
- → Recent events have ~4x more influence

---

## PART 5: Final Pipeline to Model

```
RAW DATA SOURCES
├── Financial CSVs (numbers)
├── Milestone CSVs (events)
├── Transcript TXT (text)
└── Signal CSVs (sentiment)
        ↓
INGESTION LAYER
├── Parse CSV → Python objects
├── Parse text → TaskContext (regex + keywords)
├── Apply tier weights
└── Apply confidence weights
        ↓
FEATURE ENGINEERING
├── Align quarterly → monthly
├── Calculate milestone stage as-of
├── Z-score normalize
├── Apply time decay
├── Blend confidence sources
├── Calculate proxy yield
└── Mark provenance
        ↓
ENRICHED MONTHLY PANEL
(data/processed/enriched_monthly_panel.csv)
        ↓
MODEL INPUT (Observation objects)
├── month, yield_pct, area_factor
├── cfo_gm_signal, ifs_profitability
├── academic_signal, disclosure_confidence
        ↓
LLAMBO MODEL
├── Surrogate uses area_factor for size penalty
├── Context affects growth drift
├── Signals influence prior blend
└── Confidence modulates uncertainty
```

---

## Is This a Consistent Methodology?

### ✅ What IS Consistent

| Aspect | Consistency |
|--------|-------------|
| **Data structure** | All sources → dataclasses → CSV |
| **Time alignment** | All mapped to monthly grain |
| **Weighting** | Same tier_weight() and confidence_weight() everywhere |
| **Decay** | Same exponential decay (0.86^months) for all events |
| **Bounds** | All signals clipped to valid ranges |
| **Provenance** | Every row tagged with yield_source |

### ⚠️ Limitations

| Issue | Why It Matters |
|-------|----------------|
| **Manual sentiment scoring** | Human-judged (-1 to +1), not algorithmic |
| **Proxy equation is arbitrary** | Coefficients are engineering guesses, not learned |
| **Text extraction is naive** | Word counting, not NLP/LLM sentiment |
| **Die area is pure proxy** | No actual data; formula-based estimate |
| **Sparse anchors** | Only 2 observed points to calibrate history |

### 🔧 Potential Improvements

1. Use LLM for text extraction instead of regex/keywords
2. Learn proxy coefficients from data
3. Add uncertainty to proxy yield (not just point estimates)
4. Cross-validate feature engineering choices
5. Document manual signal extraction more rigorously

---

## Summary

| Question | Answer |
|----------|--------|
| **How is fundamental data ingested?** | CSV → dataclass → monthly alignment → normalization |
| **How is unstructured data converted?** | Regex + keyword counting → numerical scores |
| **Are both combined consistently?** | Yes, through unified monthly panel with provenance |
| **Is methodology robust?** | Partially — weighting systematic, proxy equation heuristic |

The system is **transparent and auditable** — every data point traces to source with confidence weights. But proxy yield construction is an engineering approximation, not a learned model.

---

*Last updated: 2026-02-25*
