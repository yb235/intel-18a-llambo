from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, pi, sqrt
import numpy as np

from .context import TaskContext
from .hardening import HardeningConfig, clip_value


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass(frozen=True)
class SurrogatePosterior:
    mean: float
    stddev: float


class LlamboStyleSurrogate:
    """Context-aware surrogate for next-step yield proposals.

    The design mirrors LLAMBO ideas:
    - rich task context influences candidate utility
    - surrogate predicts objective + uncertainty
    - acquisition function selects next candidate
    """

    def __init__(
        self,
        context: TaskContext,
        seed: int = 18,
        observed_yields: list[float] | None = None,
        hardening: HardeningConfig | None = None,
    ) -> None:
        self.context = context
        self.rng = np.random.default_rng(seed)
        self.hardening = (hardening or HardeningConfig.baseline()).validated()
        self.data_anchor_growth = self._estimate_data_anchor_growth(observed_yields or [])

    @staticmethod
    def _estimate_data_anchor_growth(observed_yields: list[float]) -> float:
        if len(observed_yields) < 2:
            return 0.0
        arr = np.array(observed_yields, dtype=float)
        prev = np.maximum(1.0, arr[:-1])
        growth = np.diff(arr) / prev
        if growth.size == 0:
            return 0.0
        return float(clip_value(np.median(growth), 0.0, 0.20))

    def posterior_for_candidate(self, prev_yield: float, growth_rate: float, month_index: int) -> SurrogatePosterior:
        growth_rate = max(0.0, min(0.20, growth_rate))

        headroom = max(0.0, 100.0 - prev_yield)
        phase = 1.0 / (1.0 + exp(-(prev_yield - self.context.s_curve_midpoint) * self.context.s_curve_steepness))
        phase_gain = 1.0 - 0.55 * phase

        confidence_boost = 0.0025 * self.context.transcript_confidence
        risk_drag = 0.0020 * self.context.transcript_risk
        context_drift = confidence_boost - risk_drag
        if self.hardening.enabled:
            context_drift = clip_value(context_drift, -self.hardening.context_drift_clip, self.hardening.context_drift_clip)

        effective_growth = max(0.0, growth_rate + context_drift)
        if self.hardening.enabled:
            blended_growth = (
                self.hardening.prior_weight * effective_growth
                + (1.0 - self.hardening.prior_weight) * self.data_anchor_growth
            )
            effective_growth = clip_value(blended_growth, 0.0, 0.20)
        mean = prev_yield + headroom * effective_growth * phase_gain
        mean = min(100.0, max(0.0, mean))

        base_std = 1.7 - min(1.1, 0.12 * month_index)
        growth_alignment_penalty = abs(growth_rate - self.context.guidance_growth_mid) * 18.0
        stddev = max(0.45, base_std + growth_alignment_penalty)
        if self.hardening.enabled:
            # Increase predictive spread for heavy-tail modes to avoid over-confident intervals.
            if self.hardening.robust_likelihood == "huber":
                stddev *= 1.08
            elif self.hardening.robust_likelihood == "student_t":
                stddev *= 1.15

        return SurrogatePosterior(mean=mean, stddev=stddev)

    @staticmethod
    def expected_improvement(mu: float, sigma: float, incumbent_best: float, xi: float = 0.05) -> float:
        if sigma <= 1e-9:
            return max(0.0, mu - incumbent_best - xi)
        z = (mu - incumbent_best - xi) / sigma
        return (mu - incumbent_best - xi) * norm_cdf(z) + sigma * norm_pdf(z)

    def pick_candidate_growth(self, prev_yield: float, incumbent_best: float, month_index: int) -> tuple[float, SurrogatePosterior, float]:
        grid = np.linspace(0.00, 0.15, 61)
        best_growth = float(grid[0])
        best_posterior = self.posterior_for_candidate(prev_yield, best_growth, month_index)
        best_acq = -1.0

        for growth in grid:
            posterior = self.posterior_for_candidate(prev_yield, float(growth), month_index)
            acq = self.expected_improvement(posterior.mean, posterior.stddev, incumbent_best)
            if acq > best_acq:
                best_growth = float(growth)
                best_posterior = posterior
                best_acq = acq

        return best_growth, best_posterior, best_acq
