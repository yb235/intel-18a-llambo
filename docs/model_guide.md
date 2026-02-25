# Intel 18A LLAMBO — Complete Model Guide

> A beginner-friendly explanation of how the yield forecasting system works.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Big Picture Architecture](#big-picture-architecture)
3. [Process 1: Data Ingestion](#process-1-data-ingestion)
4. [Process 2: Context Generation](#process-2-context-generation)
5. [Process 3: Surrogate Model](#process-3-surrogate-model)
6. [Process 4: Acquisition Function](#process-4-acquisition-function)
7. [Process 5: Bayesian Loop](#process-5-bayesian-loop)
8. [Process 6: Hardening](#process-6-hardening)
9. [Process 7: Evaluation Metrics](#process-7-evaluation-metrics)
10. [Process 8: Output Interpretation](#process-8-output-interpretation)
11. [Quick Reference Summary](#quick-reference-summary)

---

## Problem Overview

### What Problem Are We Solving?

**Intel is manufacturing chips using a new process called "18A"** (their most advanced node). During the ramp-up phase, they need to predict:

> *"What will our manufacturing yield be next month? And the month after that?"*

**Yield** = percentage of chips that come out working perfectly. If you make 100 chips and 64 work → 64% yield.

This is **hard** because:
- New processes are unpredictable
- Limited data (only a few months of observations)
- Management says things like "7-8% monthly improvement" but how reliable is that?

---

## Big Picture Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT DATA                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Numeric observations:  Jan=64%, Feb=68.5%                   │
│  2. Transcript text: "Management said 7-8% monthly target..."   │
│  3. Die size info: How big is the chip? (affects defects)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   CONTEXT GENERATION                             │
│  (Turn raw text into numbers the model can use)                 │
├─────────────────────────────────────────────────────────────────┤
│  • Extract: 7-8% → guidance_growth_low=0.07, high=0.08          │
│  • Count positive words: "confidence", "improved", "progress"   │
│  • Count risk words: "risk", "variability", "challenge"         │
│  • Set S-curve parameters (more on this later)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SURROGATE MODEL                                │
│  (The "brain" that predicts next month's yield)                 │
├─────────────────────────────────────────────────────────────────┤
│  Given: current yield + growth rate + context                   │
│  Output: predicted mean + uncertainty (stddev)                  │
│                                                                  │
│  Key formula:                                                    │
│  next_yield = current + headroom × growth × phase_gain          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 ACQUISITION FUNCTION                             │
│  (Decides WHICH growth rate to try next)                        │
├─────────────────────────────────────────────────────────────────┤
│  Uses Expected Improvement (EI):                                 │
│  "Which growth rate gives the best chance of beating            │
│   our current best, while accounting for uncertainty?"          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   BAYESIAN LOOP                                  │
│  (Iterate month by month into the future)                       │
├─────────────────────────────────────────────────────────────────┤
│  For each future month:                                          │
│    1. Pick best growth rate candidate                           │
│    2. Get posterior prediction (mean + uncertainty)             │
│    3. Propagate uncertainty forward                             │
│    4. Apply hardening (robustness tweaks)                       │
│    5. Output: mean, 95% confidence interval                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT                                      │
├─────────────────────────────────────────────────────────────────┤
│  • Forecast CSV with mean, stddev, CI95 low/high per month      │
│  • Yield curve plot                                              │
│  • Calibration plot (how well do our 95% intervals cover?)      │
│  • Metrics: MAE, RMSE, coverage, calibration error              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Process 1: Data Ingestion

### What It Does
Reads your data files and converts them into Python objects the model can use.

### The Code (`src/intel_18a_llambo/ingestion.py`)

```python
@dataclass(frozen=True)
class Observation:
    month: date          # When was this measured?
    yield_pct: float     # What was the yield? (e.g., 64.0 = 64%)
    area_factor: float   # How big is the die? (normalized)
```

### Example Input File (`data/sample_observations.csv`)

```csv
month,yield
2026-01,64.0
2026-02,68.5
```

### What Happens

1. **Read CSV** → parse each row
2. **Parse month** → "2026-01" becomes `date(2026, 1, 1)`
3. **Calculate area_factor**:
   - If you have `effective_die_area_mm2=200`, it becomes `200/180 = 1.11`
   - Bigger die = more defects = harder to get high yield
4. **Sort by date** → ensure chronological order

### Why Area Factor Matters

Imagine you're baking cookies:
- **Small cookie** (area_factor=0.8) → easier to get perfect
- **Big cookie** (area_factor=1.3) → more chances to burn

Same with chips: bigger die area = more places for defects to hide.

---

## Process 2: Context Generation

### What It Does
Turns **management transcripts** (text) into **numerical signals** the model can use.

### The Code (`src/intel_18a_llambo/context.py`)

```python
@dataclass(frozen=True)
class TaskContext:
    guidance_growth_low: float      # e.g., 0.07 (7%)
    guidance_growth_high: float     # e.g., 0.08 (8%)
    guidance_growth_mid: float      # e.g., 0.075 (7.5%)
    transcript_confidence: float    # count of positive words
    transcript_risk: float          # count of risk words
    ribbonfet_mentioned: bool       # new transistor tech?
    powervia_mentioned: bool        # new power delivery?
    s_curve_midpoint: float         # where yield gains slow down
    s_curve_steepness: float        # how sharp is the S-curve
```

### Example Transcript

```
Management noted that Intel 18A is progressing with RibbonFET and 
PowerVia integration maturing in pilot lines. The team reiterated 
a guidance target of around 7-8% monthly yield improvement through 
the mid-2026 learning phase. Risk was acknowledged around process 
variability, but overall confidence improved after recent defect-
density reduction.
```

### What Gets Extracted

| Signal | How It's Computed | Value |
|--------|-------------------|-------|
| **guidance_growth_low** | Regex: "7-8%" → 7/100 | 0.07 |
| **guidance_growth_high** | Regex: "7-8%" → 8/100 | 0.08 |
| **ribbonfet_mentioned** | Search for "ribbonfet" | True |
| **powervia_mentioned** | Search for "powervia" | True |
| **transcript_confidence** | Count: "confidence", "improved", "progress", "ahead", "reduction", "stable" | 3 |
| **transcript_risk** | Count: "risk", "variability", "challenge" | 2 |
| **s_curve_midpoint** | If RibbonFET + PowerVia → 80.0, else 78.0 | 80.0 |

### Why This Matters

The model uses these signals to **adjust its predictions**:
- More confidence words → slightly more optimistic forecast
- More risk words → slightly more conservative forecast
- S-curve midpoint tells the model: "around 80% yield, improvements start slowing down"

---

## Process 3: Surrogate Model

### What It Does
This is the **core prediction engine**. Given the current state, it predicts:
1. What will next month's yield be? (mean)
2. How uncertain are we? (stddev)

### The Key Formula

```python
def posterior_for_candidate(prev_yield, growth_rate, month_index, area_factor):
    # 1. Calculate "headroom" - how much room left to improve?
    headroom = 100.0 - prev_yield  # e.g., 100 - 64 = 36
    
    # 2. S-curve phase - are we in fast-growth or slow-growth zone?
    # Logistic function: phase approaches 1.0 as yield approaches midpoint
    phase = 1 / (1 + exp(-(prev_yield - s_curve_midpoint) * steepness))
    phase_gain = 1.0 - 0.55 * phase  # Higher phase → lower gain
    
    # 3. Context adjustments from transcript
    confidence_boost = 0.0025 * transcript_confidence  # +0.75% if confidence=3
    risk_drag = 0.0020 * transcript_risk               # -0.40% if risk=2
    context_drift = confidence_boost - risk_drag       # +0.35% net
    
    # 4. Size penalty - bigger die = more defects
    size_drag = 0.03 * (area_factor - 1.0)  # -0.9% if area_factor=1.3
    
    # 5. Effective growth rate
    effective_growth = growth_rate + context_drift - size_drag
    
    # 6. FINAL PREDICTION
    mean = prev_yield + headroom * effective_growth * phase_gain
    
    # 7. Uncertainty (stddev)
    base_std = 1.7 - 0.12 * month_index  # Gets more certain over time
    stddev = base_std + 18 * abs(growth_rate - guidance_mid)  # Penalize unrealistic growth
    
    return mean, stddev
```

### Walkthrough Example

**Starting state:**
- prev_yield = 64%
- growth_rate = 7% (0.07)
- month_index = 1 (first forecast)
- area_factor = 1.0 (normal die size)
- s_curve_midpoint = 80
- guidance_mid = 7.5%

**Step-by-step:**

```
1. headroom = 100 - 64 = 36

2. phase = 1 / (1 + exp(-(64 - 80) * 0.14))
         = 1 / (1 + exp(2.24))
         = 1 / (1 + 9.4)
         = 0.096  (we're in the fast-growth zone, far from midpoint)
   
   phase_gain = 1 - 0.55 * 0.096 = 0.947  (almost full gain)

3. context_drift = 0.0025*3 - 0.0020*2 = 0.0035 (tiny boost)

4. size_drag = 0.03 * (1.0 - 1.0) = 0  (no penalty for normal size)

5. effective_growth = 0.07 + 0.0035 - 0 = 0.0735

6. mean = 64 + 36 * 0.0735 * 0.947
        = 64 + 2.51
        = 66.5%

7. base_std = 1.7 - 0.12*1 = 1.58
   growth_penalty = 18 * |0.07 - 0.075| = 0.09
   stddev = 1.58 + 0.09 = 1.67

Result: Next month prediction = 66.5% ± 1.67%
```

### The S-Curve Explained

```
Yield %
100 │                        ╭───────────
    │                    ╭───╯
 80 │                ╭───╯   ← Midpoint (gains slow down here)
    │            ╭───╯
 60 │        ╭───╯
    │    ╭───╯
 40 │╭───╯
    │╯
  0 └─────────────────────────────────→ Time
     Early    Mid-ramp    Mature
     (fast)   (slowing)   (slow)
```

- **Below 60%**: Fast improvements, lots of low-hanging fruit
- **60-80%**: Still improving, but pace is slowing
- **Above 80%**: Diminishing returns, squeezing out last few percentage points

---

## Process 4: Acquisition Function

### What It Does
Decides **which growth rate to try** for the next prediction. It balances:
- **Exploitation**: Pick growth rates that seem promising
- **Exploration**: Try uncertain growth rates to learn more

### Expected Improvement (EI) Formula

```python
def expected_improvement(mu, sigma, incumbent_best, xi=0.05):
    """
    mu: predicted mean yield for this candidate
    sigma: uncertainty (stddev) for this candidate
    incumbent_best: best yield we've seen so far
    xi: exploration parameter (small = more exploitation)
    """
    if sigma <= 0:
        return max(0, mu - incumbent_best - xi)
    
    z = (mu - incumbent_best - xi) / sigma
    
    EI = (mu - incumbent_best - xi) * norm_cdf(z) + sigma * norm_pdf(z)
    
    return EI
```

### How It Works

The model tries **61 different growth rates** (0% to 15% in 0.25% increments) and picks the one with highest EI.

**Example:**
- incumbent_best = 68.5% (best we've seen)
- Candidate A: mean=70%, sigma=2% → EI = 0.42
- Candidate B: mean=69%, sigma=4% → EI = 0.38
- Candidate C: mean=72%, sigma=6% → EI = 0.51 ← **Winner!**

Candidate C wins because:
- Higher potential upside (72%)
- Enough uncertainty that we're not overconfident

---

## Process 5: Bayesian Loop

### What It Does
Runs the surrogate model **month by month** into the future, propagating uncertainty at each step.

### The Algorithm

```python
def run_forecast(observations, context, months_ahead=6):
    # Start with observed data
    prev_mean = observations[-1].yield_pct  # 68.5%
    prev_std = 0.0
    
    for step in range(1, months_ahead + 1):
        # 1. Estimate area factor for this month
        area_factor = estimate_from_trend(prev_area_factor)
        
        # 2. Pick best growth rate using acquisition function
        growth, posterior, acq = surrogate.pick_candidate_growth(
            prev_yield=prev_mean,
            incumbent_best=best_so_far,
            month_index=step,
            area_factor=area_factor
        )
        
        # 3. Propagate uncertainty forward
        # New uncertainty = model uncertainty + carryover from previous
        propagated_std = sqrt(
            posterior.stddev**2 + (0.35 * prev_std)**2
        )
        
        # 4. Apply hardening (more on this later)
        if hardening.enabled:
            propagated_std *= outlier_multiplier * robust_multiplier
        
        # 5. Calculate 95% confidence interval
        z95 = 1.96  # for 95% CI
        ci_low = mean - z95 * propagated_std
        ci_high = mean + z95 * propagated_std
        
        # 6. Store result
        output.append(ForecastPoint(
            month=next_month,
            posterior_mean=mean,
            posterior_stddev=propagated_std,
            ci95_low=ci_low,
            ci95_high=ci_high,
            selected_growth_rate=growth,
            acquisition_value=acq
        ))
        
        # 7. Update for next iteration
        prev_mean = mean
        prev_std = propagated_std
```

### Why Uncertainty Grows Over Time

```
Month 1:  66.5% ± 1.7%
Month 2:  68.8% ± 2.1%  ← uncertainty grew
Month 3:  70.9% ± 2.5%  ← and grew more
Month 4:  72.7% ± 3.0%  ← compounding uncertainty
Month 5:  74.2% ± 3.5%
Month 6:  75.4% ± 4.1%
```

Each step adds uncertainty because we're building on previous predictions, not actual measurements.

---

## Process 6: Hardening

### What It Does
Makes the model **more robust** to outliers, weird data, and overconfidence.

### Key Hardening Options

| Parameter | What It Does | Default |
|-----------|--------------|---------|
| `prior_weight` | How much to trust prior vs data | 0.65 |
| `robust_likelihood` | "huber" or "student_t" for outlier handling | "huber" |
| `huber_delta` | Threshold for Huber loss | 1.75 |
| `context_drift_clip` | Max allowed context adjustment | 0.02 |
| `outlier_z_clip` | Z-score threshold for outliers | 3.25 |
| `outlier_std_inflation` | Inflate uncertainty for outliers | 1.5 |
| `interval_calibration` | "isotonic" to fix coverage | "isotonic" |

### Robust Likelihood (Huber)

**Normal loss function:**
```
error = predicted - actual
loss = error²  (squared error)
```

**Problem:** If one prediction is way off (error=10), loss=100 dominates everything.

**Huber loss:**
```python
def huber_loss(error, delta=1.75):
    if abs(error) <= delta:
        return 0.5 * error**2  # quadratic for small errors
    else:
        return delta * (abs(error) - 0.5 * delta)  # linear for large errors
```

**Result:** Outliers have less influence on the model.

### Interval Calibration

**Problem:** Your "95% confidence interval" might only cover 70% of actuals.

**Solution:** Isotonic regression on historical z-scores.

```python
# Historical z-scores (how many stddevs off were we?)
z_scores = [0.5, -1.2, 0.8, 2.5, -0.3, 1.9, ...]

# What z-value captures 95% of these?
calibrated_z = isotonic_regression(z_scores, percentile=95)

# If calibrated_z = 2.3 instead of 1.96, use 2.3
ci_low = mean - 2.3 * stddev
ci_high = mean + 2.3 * stddev
```

### The Result

| Metric | Baseline | Hardened | Why Better? |
|--------|----------|----------|-------------|
| **MAE** | 9.54 | 5.17 | Outliers don't dominate |
| **Coverage95** | 47.6% | 72.4% | Calibrated intervals |
| **Calibration Error** | 0.40 | 0.22 | Intervals match reality |

---

## Process 7: Evaluation Metrics

### How We Test the Model

**Rolling-origin backtesting:**

```
Data: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug]

Test 1: Train on [Jan,Feb] → predict Mar (horizon=1)
Test 2: Train on [Jan,Feb,Mar] → predict Apr (horizon=1)
Test 3: Train on [Jan,Feb,Mar,Apr] → predict May (horizon=1)
...
Test N: Train on [Jan...Jul] → predict Aug (horizon=1)

Then repeat for horizon=2,3,4,5,6
```

### Metrics Explained

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **MAE** | `mean(\|predicted - actual\|)` | Average error (easy to interpret) |
| **RMSE** | `sqrt(mean((predicted - actual)²))` | Penalizes large errors more |
| **CRPS** | Continuous Ranked Probability Score | How good is the whole distribution? |
| **Coverage95** | `% of actuals within 95% CI` | Should be ~95% |
| **Calibration Error** | `\|coverage95 - 0.95\|` | How far from ideal coverage |
| **Interval Width** | `mean(ci_high - ci_low)` | How uncertain are we? |

### CRPS Deep Dive

CRPS measures **how well the entire predicted distribution matches reality**:

```python
def crps(predicted_mean, predicted_std, actual):
    # Predicted CDF (cumulative distribution function)
    def F(x):
        return norm_cdf((x - predicted_mean) / predicted_std)
    
    # Actual CDF (step function at the observed value)
    def F_actual(x):
        return 1.0 if x >= actual else 0.0
    
    # CRPS = integral of (F - F_actual)²
    # Lower is better
```

If you predict `68% ± 3%` and actual is `70%`:
- Your distribution should have some mass at 70%
- CRPS penalizes if the distribution is too narrow or shifted

---

## Process 8: Output Interpretation

### metrics_summary.csv

```csv
dataset,scenario,model,model_version,horizon,mae,rmse,coverage95,calibration_error
enriched_panel,prior_7_8_with_killer,llambo_style,hardened,1,3.47,4.90,0.85,0.14
enriched_panel,prior_7_8_with_killer,llambo_style,hardened,2,5.05,6.04,0.68,0.19
enriched_panel,prior_7_8_with_killer,llambo_style,hardened,3,5.85,7.23,0.56,0.16
...
```

**How to read:**
- `horizon=1` → 1 month ahead predictions
- `mae=3.47` → average error of 3.47 percentage points
- `coverage95=0.85` → 85% of actuals fell within 95% CI (should be 95%)
- `calibration_error=0.14` → 14% away from ideal coverage

### backtest_predictions.csv

```csv
origin_month,target_month,horizon,y_true,y_pred_mean,y_pred_stddev,ci95_low,ci95_high
2026-01,2026-02,1,68.5,66.5,1.7,63.2,69.8
2026-02,2026-03,1,72.1,70.2,2.1,66.1,74.3
...
```

**How to read:**
- We trained on data up to `2026-01`
- Predicted `2026-02` would be `66.5% ± 1.7%`
- Actual was `68.5%` → within our CI!

### calibration_plot.png

```
|                    ╱
| Ideal           ╱
| Coverage       ╱
|     │        ╱
| 95% ├───────●━━━━━━━━━  ← Your model
|     │      ╱
|     │    ╱
|     └───────────────────→
|           Predicted Coverage
```

- **X-axis**: What coverage you claimed (e.g., 95%)
- **Y-axis**: What coverage you actually got
- **Diagonal line**: Perfect calibration
- **Your curve**: Should hug the diagonal

### benchmark_plot.png

Shows MAE/RMSE for different model versions:
- `baseline` (no hardening)
- `hardened` (with robust tweaks)
- `hardened_no_area` (without size factor)

---

## Quick Reference Summary

| Component | Purpose | Key Insight |
|-----------|---------|-------------|
| **Ingestion** | Load data | Yield % + die size → observations |
| **Context** | Parse transcripts | Text → confidence/risk scores |
| **Surrogate** | Predict next yield | Mean = prev + headroom × growth × phase |
| **Acquisition** | Pick growth rate | Maximize Expected Improvement |
| **Bayesian Loop** | Iterate forward | Propagate uncertainty month by month |
| **Hardening** | Robustness | Handle outliers, calibrate intervals |
| **Evaluation** | Test quality | Rolling backtest, CRPS, coverage |

### The Magic

The model combines:
- 📊 **Data-driven** learning from observations
- 📝 **Context-aware** signals from management guidance
- 🎯 **Bayesian optimization** to pick best growth candidates
- 🛡️ **Robust statistics** to handle messy real-world data

---

## Running the Model

### Basic Forecast

```bash
PYTHONPATH=src python -m intel_18a_llambo.cli \
  --observations-csv data/sample_observations.csv \
  --transcript-files data/sample_transcript_q1_2026.txt \
  --output-csv outputs/forecast.csv \
  --output-plot outputs/intel18a_yield_curve.png \
  --seed 18 \
  --horizon 2026-08
```

### Quality Evaluation

```bash
PYTHONPATH=src python -m intel_18a_llambo.eval_cli \
  --observations-csv data/processed/enriched_monthly_panel.csv \
  --output-dir outputs/quality_enriched \
  --max-horizon 6 \
  --seed 18 \
  --prior-weight 0.65 \
  --robust-likelihood huber \
  --interval-calibration isotonic
```

---

*Last updated: 2026-02-25*
