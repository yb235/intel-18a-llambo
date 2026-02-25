# Glossary

> Plain-English definitions of every technical term used in the Intel 18A LLAMBO project.

---

## A

**Acquisition function**
A mathematical formula that decides which candidate to try next in a Bayesian optimization loop. This project uses *Expected Improvement* (EI), which picks the growth rate that has the highest probability of beating the current best prediction. Think of it as "which option has the best risk-adjusted upside?"

**Area factor**
A unitless scale that represents the relative size of a chip die compared to a 180 mm² reference. `area_factor = die_area_mm2 / 180`. A bigger die has more surface area, increasing the chance that a random manufacturing defect lands on it. `area_factor = 1.0` means nominal size; `1.3` means 30% bigger than reference, which drags yield down.

---

## B

**Backtest / backtesting**
Testing a forecasting model on *historical* data to see how well it would have predicted the past. In this project, "rolling-origin" backtests progressively expand the training window and record predictions at each step. See also: *rolling-origin evaluation*.

**Baseline model**
The version of the LLAMBO model with *no hardening* enabled. It uses the raw surrogate predictions without outlier handling, interval calibration, or robust likelihoods. Useful as a reference to measure how much hardening improves things.

**Bayesian optimization (BO)**
A strategy for finding the best value of a function when evaluations are expensive. The key idea: use a *surrogate model* to cheaply approximate the function, then use an *acquisition function* to pick the next point to evaluate. In this project, BO is adapted for time-series: instead of optimizing a lab experiment, it chooses the next growth rate to use in a yield forecast.

**Bayesian update**
Combining prior beliefs with new evidence to form an updated (posterior) belief. Example: you believe yield will be ~67% (prior). You observe the actual is 68.5%. You update your belief toward 68.5%.

---

## C

**Calibration**
A model is well-calibrated if its stated confidence actually matches reality. A perfectly calibrated model's "95% prediction interval" should contain the actual value 95% of the time. Poor calibration means your intervals are systematically too narrow or too wide.

**Calibration error**
`|coverage95 - 0.95|` — how far the actual 95% coverage is from the ideal 0.95. Lower is better. 0.0 = perfect calibration.

**CFO signals**
Numerical estimates of the financial direction signaled by Intel's Chief Financial Officer in earnings call commentary. Values range from -1 (strong negative signal) to +1 (strong positive signal). Manually extracted and recorded in `data/raw/intel_cfo_signals.csv`.

**CI** (confidence interval / credible interval)
The range within which the true value is expected to fall with a given probability. This project produces 95% CIs: `[ci95_low, ci95_high]`. For a well-calibrated model, 95 out of 100 actual values should fall inside these bounds.

**Context drift**
The net adjustment to the growth rate that comes from transcript signals: `confidence_boost - risk_drag`. A positive context drift slightly increases the forecast (management is optimistic); negative slightly decreases it.

**CRPS** (Continuous Ranked Probability Score)
A metric that evaluates the *entire predicted distribution*, not just the mean. Lower CRPS = better. A sharp, centered prediction distribution scores better than a wide, off-center one. Measures both accuracy and sharpness.

**Coverage95**
The fraction of actual yield values that fell inside the model's 95% prediction interval during backtesting. Should be ≈ 0.95 for a well-calibrated model. Values below 0.95 mean the model is *overconfident* (intervals too narrow); above 0.95 means it is *underconfident* (intervals too wide).

---

## D

**Data anchor growth**
The empirical growth rate estimated from observed data (median of observed period-to-period yield changes). The hardened model blends this with the prior-context growth rate to avoid over-relying on management guidance alone.

**Die area**
The physical size of a single chip on the silicon wafer (measured in mm²). Larger dice produce more chips per design but are harder to manufacture without defects because the defect budget is spread over more area.

**Disclosure confidence**
A number from 0 to 1 representing how confident the system is in the source data for a given month. High disclosure confidence (0.9) means the signal is well-supported by public disclosures; low (0.2) means it is a rough estimate.

---

## E

**Effective growth rate**
The final growth rate fed into the surrogate after adjusting for context drift and die-area drag:
`effective_growth = growth_rate + context_drift - size_drag`

**EI** → see *Acquisition function* and *Expected Improvement*

**Enriched monthly panel**
The merged, feature-engineered dataset in `data/processed/enriched_monthly_panel.csv`. It combines quarterly financial data, milestone events, CFO signals, and academic signals into a single monthly table with one row per month from 2024-01 to 2026-02. The `yield` column is mostly a *proxy* (engineered estimate) except for the Jan and Feb 2026 anchor points.

**Expected Improvement (EI)**
The core acquisition function used to pick the next growth-rate candidate. EI is calculated as:
```
EI = (μ - best - ξ) × Φ(z) + σ × φ(z)
where z = (μ - best - ξ) / σ
```
`μ` = predicted mean, `σ` = predicted uncertainty, `best` = best yield seen so far, `ξ` = exploration bonus (0.05). A high EI score means the candidate is likely to beat the current best or has enough uncertainty to be worth exploring.

---

## F

**Feature engineering**
The process of transforming raw data sources into numerical inputs that a model can use. In this project it includes: z-score normalization of financial metrics, milestone stage normalization, exponential time-decay of academic signals, and die-area proxy construction.

**ForecastPoint**
The output unit of the Bayesian loop. One `ForecastPoint` per month containing: month, observed yield (if any), posterior mean, posterior stddev, 95% CI bounds, selected growth rate, acquisition value, area factor.

---

## G

**Gross margin (GM)**
Revenue minus cost of goods sold, expressed as a percentage. For Intel, a low GM (like Q3 2024's 15%) signals manufacturing trouble; a high GM (41%) signals efficiency. GM is used as a proxy for process health.

**Growth rate**
The monthly yield improvement as a fraction: `growth = (this_month - last_month) / last_month`. Example: going from 64% to 68.5% is a growth rate of `(68.5 - 64) / 64 = 0.070` (7%).

**Guidance range**
Management's stated expected improvement range, e.g. "7-8% monthly yield improvement". Parsed by regex from transcript text into `guidance_growth_low=0.07` and `guidance_growth_high=0.08`.

---

## H

**Hardening**
A collection of robustness techniques applied on top of the baseline LLAMBO model to make it perform better on limited, noisy, real-world data:
- Robust likelihoods (Huber, Student-t) to reduce the influence of outliers
- Interval calibration to fix miscalibrated 95% CIs
- Context drift clipping to prevent transcript noise from dominating
- Outlier detection and uncertainty inflation

**HardeningConfig**
The dataclass that groups all hardening parameters. Created from CLI arguments. Call `.validated()` to clamp values to valid ranges.

**Headroom**
`100 - current_yield` — how much room is left to improve. At 64% yield, headroom = 36. At 90% yield, headroom = 10. The surrogate multiplies `headroom × growth × phase_gain` to compute the increment.

**Horizon**
The forecast target month in `YYYY-MM` format (e.g., `2026-08`). The model generates monthly predictions from the last observed month up to and including the horizon month.

**Huber loss / Huber likelihood**
A robust alternative to the standard squared-error loss that down-weights outliers. For small errors (within `huber_delta`), it behaves like squared error. For large errors, it switches to linear, capping the outlier's influence.

---

## I

**IFS** (Intel Foundry Services)
Intel's contract manufacturing business. IFS profitability timeline is one of the CFO signals tracked in this project.

**Incumbent best**
The highest yield value seen so far during the Bayesian loop. The acquisition function (EI) always computes improvement relative to this value.

**Innovation scale**
The standard deviation of observed yield differences (from month to month). Used to set a maximum step size for forecast increments during hardening, preventing unrealistically large jumps.

**Interval calibration**
Post-hoc adjustment of the z-value used for CI construction, based on how well in-sample intervals covered the data. Methods: `isotonic` (uses empirical CDF inversion), `quantile_scale` (uses empirical quantile at the target alpha).

**Isotonic regression (calibration)**
In this context, refers to using a monotone CDF inversion approach on historical standardized residuals to find the z-value that achieves the target coverage. "Isotonic" means the function is forced to be non-decreasing.

---

## L

**LLAMBO**
Large Language Model Bayesian Optimization — a technique from a 2024 research paper that uses LLMs as surrogate models in Bayesian optimization. This project adapts the *design pattern* (context-rich surrogate + acquisition function loop) for yield forecasting, but does **not** use an actual LLM. The external LLAMBO reference implementation is at `external/LLAMBO`.

---

## M

**MAE** (Mean Absolute Error)
Average of `|predicted - actual|` across all backtest predictions. Measured in percentage points. MAE of 5.17 means the model is off by an average of 5.17 yield percentage points.

**Milestone stage**
A 1–8 scale tracking how far along the Intel 18A process ramp has progressed. Stage 1 = test chip announced; Stage 8 = broad production availability. Higher stage generally correlates with higher achievable yield. Normalized to `[0.125, 1.0]` as `milestone_stage_norm`.

**Model version**
One of three variants tested during evaluation:
- `baseline`: No hardening
- `hardened`: Full hardening enabled
- `hardened_no_area`: Hardening enabled but die-area effects ignored (`use_area_factor=False`)

---

## O

**Observation**
One data point: a `(month, yield_pct)` pair plus optional feature fields. The fundamental unit of input to the model.

**Outlier**
A data point whose standardized residual (z-score) exceeds the `outlier_z_clip` threshold (default 3.25). Outliers get down-weighted and their detection inflates future uncertainty estimates.

---

## P

**Phase / Phase gain**
The S-curve slowdown factor. `phase` is a logistic function value between 0 (early ramp, fast gains) and 1 (mature process, slow gains). `phase_gain = 1 - 0.55 × phase` scales down predicted improvement as yield approaches the S-curve midpoint.

**Posterior**
In Bayesian statistics, the updated belief after seeing data. Here: the model's predicted yield distribution (mean + stddev) after incorporating the latest observations.

**PowerVia**
Intel's backside power delivery technology used in the 18A process node. When mentioned in a transcript, the S-curve midpoint is raised from 78 to 80, reflecting optimism about the technology's impact on yield ceiling.

**Prior / Prior weight**
The model's initial belief before seeing data. In this project, the "prior" is primarily the context guidance extracted from transcripts. `prior_weight` controls how much the prior influences the effective growth rate vs. the data-anchor growth rate.

**Proxy yield**
An estimated yield value constructed from public indicators (financial metrics, milestone stages, signals) for months where no actual yield was observed. Used to extend the training history. All months before Jan 2026 in the enriched panel use proxy yields.

---

## R

**RibbonFET**
Intel's next-generation transistor architecture used in 18A. When mentioned in a transcript, it signals advanced process node maturity. Together with PowerVia, its mention raises the S-curve midpoint from 78 to 80.

**RMSE** (Root Mean Squared Error)
`sqrt(mean((predicted - actual)²))` — like MAE but penalizes large errors more heavily. More sensitive to occasional big misses than MAE.

**Robust likelihood**
An alternative to normal (Gaussian) likelihood that reduces the influence of outliers on parameter estimation. Options: `huber` (recommended) or `student_t` (heavier tails).

**Rolling-origin evaluation**
A backtesting strategy where the training window expands by one month at each step:
- Origin k=2: train on months 1-2, predict month 3
- Origin k=3: train on months 1-3, predict month 4
- ...and so on for all horizons

---

## S

**S-curve**
A sigmoid-shaped growth trajectory common in technology adoption and manufacturing ramp. Yield improves quickly in the early ramp (steep section), then slows as it approaches the ceiling. Parameters:
- `s_curve_midpoint`: yield level where gains slow down (78-80%)
- `s_curve_steepness`: sharpness of the inflection (0.14-0.17)

**Scenario**
A named configuration combining a guidance context with a risk modifier. Used in evaluation to test multiple "what if" situations. Example: `prior_7_8_with_killer` uses 7-8% guidance plus an added risk scenario.

**Seed (random seed)**
An integer that initializes the random number generator to a fixed state. Using the same seed always produces identical results, making runs reproducible. Default: 18.

**Source manifest**
The file `data/raw/source_manifest.csv` cataloguing all data sources: URL, access date, source tier, confidence, licensing notes, and hash.

**Source tier**
Classification of how trustworthy a data source is:
- `public_observed`: Published factual data (Intel IR, earnings) → weight 1.0
- `subjective_prior`: Narrative assumptions or fictional scenario inputs → weight 0.35
- `tooling_attempt`: Failed/partial tool retrieval → weight 0.1

**Stddev / σ (standard deviation)**
A measure of uncertainty or spread. In forecasting: larger stddev = wider confidence intervals = less certain prediction. The surrogate computes stddev based on how far the candidate growth rate is from guidance and on the forecast step number.

**Student-t**
A probability distribution with heavier tails than the normal distribution. Used as a robust alternative to Gaussian likelihood when outliers are expected. Controlled by `student_t_df` (degrees of freedom); lower df = heavier tails.

**Surrogate model**
A cheap-to-evaluate approximation of an expensive function. Here: `LlamboStyleSurrogate` approximates the yield-given-growth-rate relationship using the analytical formula instead of running a real fab process. The surrogate's output (mean + stddev) drives the acquisition function.

---

## T

**Task context** → see *TaskContext*

**TaskContext**
A frozen dataclass holding all the numerical signals extracted from management transcripts. It is the "context" that makes this a *context-aware* LLAMBO model.

**Transcript confidence**
The count of positive sentiment words (`confidence`, `improved`, `progress`, `ahead`, `reduction`, `stable`) found in the management transcript. Higher count → small upward nudge to the effective growth rate.

**Transcript risk**
The count of risk/negative sentiment words (`risk`, `variability`, `delay`, `challenge`, `uncertain`, `headwind`) found in the management transcript. Higher count → small downward drag on the effective growth rate.

---

## U

**Uncertainty propagation**
The process of combining forecast uncertainties across multiple steps:
`propagated_std = sqrt(posterior.stddev² + (0.35 × prev_std)²)`
This ensures uncertainty grows the further ahead you forecast.

---

## Y

**Yield**
In semiconductor manufacturing: the percentage of chips produced on a wafer that work correctly. `yield = (working chips / total chips) × 100`. Higher is better. A yield of 64% means 36% of chips are defective and discarded.

**Yield ceiling**
The theoretical maximum yield for a given process node, approached asymptotically as the process matures. The S-curve models this as the function approaching 100% yield at infinite time.

---

*Last updated: 2026-02-25*
