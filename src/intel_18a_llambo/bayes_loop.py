from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import numpy as np

from .context import TaskContext
from .hardening import HardeningConfig, build_interval_calibrator, outlier_scale_multiplier, robust_weight
from .ingestion import Observation
from .surrogate import LlamboStyleSurrogate


@dataclass(frozen=True)
class ForecastPoint:
    month: date
    observed_yield: float | None
    posterior_mean: float
    posterior_stddev: float
    ci95_low: float
    ci95_high: float
    selected_growth_rate: float | None
    acquisition_value: float | None


def add_months(base: date, count: int) -> date:
    year = base.year + ((base.month - 1 + count) // 12)
    month = ((base.month - 1 + count) % 12) + 1
    return date(year, month, 1)


def month_distance(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return min(hi, max(lo, value))


def run_forecast(
    observations: list[Observation],
    context: TaskContext,
    months_ahead: int = 6,
    horizon: date | None = None,
    seed: int = 18,
    hardening: HardeningConfig | None = None,
) -> list[ForecastPoint]:
    if not observations:
        raise ValueError("At least one observation is required.")

    observations = sorted(observations, key=lambda item: item.month)
    cfg = (hardening or HardeningConfig.baseline()).validated()
    observed_yields = [item.yield_pct for item in observations]
    surrogate = LlamboStyleSurrogate(context=context, seed=seed, observed_yields=observed_yields, hardening=cfg)

    output: list[ForecastPoint] = []
    for obs in observations:
        output.append(
            ForecastPoint(
                month=obs.month,
                observed_yield=obs.yield_pct,
                posterior_mean=obs.yield_pct,
                posterior_stddev=0.0,
                ci95_low=obs.yield_pct,
                ci95_high=obs.yield_pct,
                selected_growth_rate=None,
                acquisition_value=None,
            )
        )

    last_month = observations[-1].month
    steps = months_ahead
    if horizon is not None:
        steps = max(steps, max(0, month_distance(last_month, horizon)))

    prev_mean = observations[-1].yield_pct
    prev_std = 0.0
    incumbent_best = max(item.yield_pct for item in observations)
    diffs = np.diff(np.array(observed_yields, dtype=float)) if len(observed_yields) > 1 else np.array([0.0], dtype=float)
    innovation_scale = float(max(0.8, np.std(diffs, ddof=1) if diffs.size > 1 else abs(diffs[-1])))

    hist_z_scores: list[float] = []
    if cfg.enabled and len(observations) >= 3:
        rolling_best = observations[0].yield_pct
        for idx in range(1, len(observations)):
            prev = observations[idx - 1].yield_pct
            _, hist_post, _ = surrogate.pick_candidate_growth(prev_yield=prev, incumbent_best=rolling_best, month_index=1)
            z = (observations[idx].yield_pct - hist_post.mean) / max(hist_post.stddev, 1e-6)
            hist_z_scores.append(float(z))
            rolling_best = max(rolling_best, observations[idx].yield_pct)

    calibrator = build_interval_calibrator(hist_z_scores, config=cfg, default_z=1.96)
    outlier_multiplier = outlier_scale_multiplier(hist_z_scores, config=cfg)
    robust_penalty = (
        float(np.mean([1.0 - robust_weight(z, cfg) for z in hist_z_scores]))
        if (cfg.enabled and hist_z_scores)
        else 0.0
    )
    robust_multiplier = 1.0 + 0.35 * robust_penalty

    for step in range(1, steps + 1):
        next_month = add_months(last_month, step)
        growth, posterior, acq = surrogate.pick_candidate_growth(
            prev_yield=prev_mean,
            incumbent_best=incumbent_best,
            month_index=step,
        )

        propagated_std = (posterior.stddev**2 + (0.35 * prev_std) ** 2) ** 0.5
        if cfg.enabled:
            propagated_std = max(0.15, propagated_std * outlier_multiplier * robust_multiplier)

        mean = _clip(posterior.mean)
        if cfg.enabled:
            step_cap = cfg.outlier_z_clip * innovation_scale
            delta = mean - prev_mean
            if abs(delta) > step_cap:
                mean = _clip(prev_mean + math.copysign(step_cap, delta))

        z95 = calibrator.adjusted_z(1.96)
        low = _clip(mean - z95 * propagated_std)
        high = _clip(mean + z95 * propagated_std)

        point = ForecastPoint(
            month=next_month,
            observed_yield=None,
            posterior_mean=mean,
            posterior_stddev=propagated_std,
            ci95_low=low,
            ci95_high=high,
            selected_growth_rate=growth,
            acquisition_value=acq,
        )
        output.append(point)

        prev_mean = mean
        prev_std = propagated_std
        incumbent_best = max(incumbent_best, mean)

    return output
