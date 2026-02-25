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
    area_factor: float | None


def add_months(base: date, count: int) -> date:
    year = base.year + ((base.month - 1 + count) // 12)
    month = ((base.month - 1 + count) % 12) + 1
    return date(year, month, 1)


def month_distance(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return min(hi, max(lo, value))


def _feature_prior_signal(item: Observation) -> float:
    blend = (
        0.40 * item.cfo_gm_signal_strength
        + 0.35 * item.ifs_profitability_timeline_score
        + 0.25 * item.academic_yield_maturity_signal
    )
    return float(_clip(blend * item.disclosure_confidence, -1.0, 1.0))


def run_forecast(
    observations: list[Observation],
    context: TaskContext,
    months_ahead: int = 6,
    horizon: date | None = None,
    seed: int = 18,
    hardening: HardeningConfig | None = None,
    use_area_factor: bool = True,
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
                area_factor=obs.area_factor,
            )
        )

    last_month = observations[-1].month
    steps = months_ahead
    if horizon is not None:
        steps = max(steps, max(0, month_distance(last_month, horizon)))

    prev_mean = observations[-1].yield_pct
    prev_std = 0.0
    prev_area_factor = observations[-1].area_factor
    prev_signal_prior = _feature_prior_signal(observations[-1])
    prev_disclosure = observations[-1].disclosure_confidence
    area_diffs = (
        np.diff(np.array([item.area_factor for item in observations], dtype=float))
        if len(observations) > 1
        else np.array([0.0], dtype=float)
    )
    signal_series = np.array([_feature_prior_signal(item) for item in observations], dtype=float)
    signal_diffs = np.diff(signal_series) if signal_series.size > 1 else np.array([0.0], dtype=float)
    disclosure_series = np.array([item.disclosure_confidence for item in observations], dtype=float)
    disclosure_diffs = np.diff(disclosure_series) if disclosure_series.size > 1 else np.array([0.0], dtype=float)
    area_drift = float(np.median(area_diffs)) if area_diffs.size else 0.0
    signal_drift = float(np.median(signal_diffs)) if signal_diffs.size else 0.0
    disclosure_drift = float(np.median(disclosure_diffs)) if disclosure_diffs.size else 0.0
    incumbent_best = max(item.yield_pct for item in observations)
    diffs = np.diff(np.array(observed_yields, dtype=float)) if len(observed_yields) > 1 else np.array([0.0], dtype=float)
    innovation_scale = float(max(0.8, np.std(diffs, ddof=1) if diffs.size > 1 else abs(diffs[-1])))

    hist_z_scores: list[float] = []
    if cfg.enabled and len(observations) >= 3:
        rolling_best = observations[0].yield_pct
        for idx in range(1, len(observations)):
            prev = observations[idx - 1].yield_pct
            hist_signal = _feature_prior_signal(observations[idx - 1])
            hist_disclosure = observations[idx - 1].disclosure_confidence
            hist_area = observations[idx - 1].area_factor if use_area_factor else 1.0
            hist_area = _clip(hist_area - 0.14 * hist_signal * max(0.35, hist_disclosure), 0.6, 1.6)
            _, hist_post, _ = surrogate.pick_candidate_growth(
                prev_yield=prev,
                incumbent_best=rolling_best,
                month_index=1,
                area_factor=hist_area,
            )
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
        est_area_factor = _clip(prev_area_factor + area_drift, 0.6, 1.6) if use_area_factor else 1.0
        est_signal_prior = _clip(prev_signal_prior + signal_drift, -1.0, 1.0)
        est_disclosure = _clip(prev_disclosure + disclosure_drift, 0.0, 1.0)
        signal_adjust = 0.14 * est_signal_prior * max(0.35, est_disclosure)
        effective_area_factor = _clip(est_area_factor - signal_adjust, 0.6, 1.6)
        growth, posterior, acq = surrogate.pick_candidate_growth(
            prev_yield=prev_mean,
            incumbent_best=incumbent_best,
            month_index=step,
            area_factor=effective_area_factor,
        )

        propagated_std = (posterior.stddev**2 + (0.35 * prev_std) ** 2) ** 0.5
        if cfg.enabled:
            propagated_std = max(0.15, propagated_std * outlier_multiplier * robust_multiplier)
            if est_signal_prior >= 0.0:
                propagated_std *= max(0.82, 1.0 - 0.10 * est_signal_prior * est_disclosure)
            else:
                propagated_std *= 1.0 + 0.08 * abs(est_signal_prior) * est_disclosure

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
            area_factor=effective_area_factor,
        )
        output.append(point)

        prev_mean = mean
        prev_std = propagated_std
        prev_area_factor = est_area_factor
        prev_signal_prior = est_signal_prior
        prev_disclosure = est_disclosure
        incumbent_best = max(incumbent_best, mean)

    return output
