# Glossary

Key terms and concepts used throughout the codebase and documentation.

---

### Acquisition Function
A function that decides which candidate to evaluate next in a Bayesian Optimization loop. This project uses **Expected Improvement (EI)**, which balances exploiting high-mean candidates against exploring uncertain ones.

### Bayesian Optimization (BO)
An iterative framework for optimizing expensive-to-evaluate functions. At each step, a surrogate model approximates the objective, an acquisition function proposes the next query point, and the surrogate is updated with the result. In this project, the "objective" is the next month's yield.

### Calibration
The alignment between predicted uncertainty and actual outcomes. A well-calibrated model's 95 % CI should contain the true value ~95 % of the time. The project measures calibration via PIT (Probability Integral Transform) quantiles and offers isotonic or quantile-scale recalibration.

### CI95 (95 % Confidence Interval)
The range `[ci95_low, ci95_high]` within which the true yield is expected to fall with 95 % probability, assuming the posterior is approximately Gaussian.

### Context Drift
The net adjustment to the growth rate derived from transcript sentiment. Positive when confidence terms dominate; negative when risk terms dominate. Clipped by `context_drift_clip` under hardening.

### Coverage95
A metric measuring how often the true value actually falls inside the predicted 95 % CI. Ideal value: 0.95. Over-coverage (> 0.95) means the intervals are too wide; under-coverage (< 0.95) means they are too narrow.

### CRPS (Continuous Ranked Probability Score)
A strictly proper scoring rule that evaluates both the accuracy and calibration of a probabilistic forecast. Lower is better. Under a Gaussian assumption, CRPS has a closed-form expression used in `evaluation.py`.

### Expected Improvement (EI)
The acquisition function used to select the best candidate growth rate at each forecast step. EI = (μ − f* − ξ) · Φ(z) + σ · φ(z), where f* is the best yield seen so far and ξ is a small exploration bonus.

### GP (Gaussian Process)
A non-parametric probabilistic model that defines a distribution over functions. Used as one of the baseline models (`gp_surrogate`) in the evaluation harness, with an RBF (radial basis function) kernel.

### Hardening
The set of robustness features layered on top of the baseline LLAMBO surrogate: prior/data blending, robust likelihoods, context-drift clipping, outlier-aware variance inflation, and interval calibration. Can be toggled on/off via `--disable-hardening`.

### Headroom
The remaining room for yield improvement: `max(0, 100 − prev_yield)`. As yield approaches 100 %, headroom shrinks and the surrogate naturally predicts smaller absolute gains.

### Huber Loss / Weighting
A robust loss function that behaves like squared error for small residuals and like absolute error for large residuals. Controlled by `huber_delta`. Residuals with |z| ≤ δ get full weight; those with |z| > δ get weight δ/|z|.

### Innovation Scale
An estimate of the typical month-to-month change in yield, derived from the standard deviation (or absolute value) of first differences. Used to set baseline uncertainty and step-size caps.

### Intel 18A
Intel's 18-angstrom (1.8 nm class) process node. This is the semiconductor manufacturing technology whose yield ramp the project forecasts.

### Isotonic Calibration
A calibration method that builds a monotone empirical CDF from historical absolute z-scores and inverts it at the target coverage level. Ensures the CI z-multiplier matches observed residual behavior.

### LLAMBO (Language-model-based Bayesian Optimization)
A research framework by Liu et al. that uses large language models as surrogate models in Bayesian Optimization. This project adapts the *design pattern* — context-aware surrogate + BO loop + acquisition — but does **not** call an actual LLM at runtime.

### Observation
A single data point: a calendar month and its associated wafer yield percentage. Represented by the `Observation(month, yield_pct)` dataclass.

### Persistence Model
The simplest baseline: predict the last observed value for all future months. Uncertainty grows with √horizon.

### Phase Gain
The S-curve damping factor: `1 − 0.55 × phase`, where `phase` is the logistic function of yield relative to the S-curve midpoint. Near the midpoint, phase gain decreases, slowing predicted improvement.

### PowerVia
Intel's back-side power delivery technology used in 18A. Its mention in transcripts is detected and increases the S-curve midpoint, reflecting higher technology maturity.

### Prior Weight
The blending coefficient between the context-driven growth rate (prior) and the data-anchored growth rate (median historical). `prior_weight = 1.0` trusts the transcript fully; `prior_weight = 0.0` trusts only the data.

### RibbonFET
Intel's gate-all-around (GAA) transistor architecture used in 18A. Its mention in transcripts (along with PowerVia) signals advanced technology integration and raises the S-curve midpoint.

### Rolling-Origin Backtest
An evaluation method where the training window expands one time step at a time, and predictions are made for the held-out future. This avoids look-ahead bias and measures how the model would have performed in real time.

### S-Curve (Logistic Curve)
The assumed shape of yield improvement over time: slow start, rapid mid-phase improvement, and saturation near high yields. Parameterized by a midpoint (78 or 80) and steepness (0.14–0.17).

### Surrogate Model
A fast, approximate model used in place of the true (expensive) objective function. In this project, `LlamboStyleSurrogate` combines the S-curve prior, context drift, and hardening to predict yield for any candidate growth rate.

### Task Context
A structured summary of information extracted from management transcripts: growth guidance, technology mentions, sentiment scores, and derived S-curve parameters. Encapsulated in the `TaskContext` dataclass.

### Yield (Wafer Yield)
The percentage of functional chips produced from a wafer. Ranges from 0 % (no good chips) to 100 % (every chip works). In semiconductor manufacturing, yield ramp is the process of improving this percentage over time as the manufacturing process matures.
