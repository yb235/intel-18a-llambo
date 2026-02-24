from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def clip_value(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


@dataclass(frozen=True)
class HardeningConfig:
    enabled: bool = False
    prior_weight: float = 0.65
    robust_likelihood: str = "huber"
    huber_delta: float = 1.75
    student_t_df: float = 5.0
    context_drift_clip: float = 0.02
    outlier_z_clip: float = 3.25
    outlier_std_inflation: float = 1.5
    interval_calibration: str = "isotonic"
    calibration_fallback: str = "quantile_scale"
    calibration_min_points: int = 10
    interval_alpha: float = 0.95

    @staticmethod
    def baseline() -> "HardeningConfig":
        return HardeningConfig(enabled=False)

    def validated(self) -> "HardeningConfig":
        prior_weight = clip_value(self.prior_weight, 0.0, 1.0)
        huber_delta = max(0.1, self.huber_delta)
        student_t_df = max(2.1, self.student_t_df)
        context_drift_clip = max(0.0, self.context_drift_clip)
        outlier_z_clip = max(1.5, self.outlier_z_clip)
        outlier_std_inflation = max(1.0, self.outlier_std_inflation)
        calibration_min_points = max(3, self.calibration_min_points)
        interval_alpha = clip_value(self.interval_alpha, 0.5, 0.999)

        robust_likelihood = self.robust_likelihood.lower().strip()
        if robust_likelihood not in {"none", "huber", "student_t"}:
            robust_likelihood = "huber"

        interval_calibration = self.interval_calibration.lower().strip()
        if interval_calibration not in {"none", "isotonic", "quantile_scale"}:
            interval_calibration = "isotonic"

        calibration_fallback = self.calibration_fallback.lower().strip()
        if calibration_fallback not in {"none", "quantile_scale"}:
            calibration_fallback = "quantile_scale"

        return HardeningConfig(
            enabled=self.enabled,
            prior_weight=prior_weight,
            robust_likelihood=robust_likelihood,
            huber_delta=huber_delta,
            student_t_df=student_t_df,
            context_drift_clip=context_drift_clip,
            outlier_z_clip=outlier_z_clip,
            outlier_std_inflation=outlier_std_inflation,
            interval_calibration=interval_calibration,
            calibration_fallback=calibration_fallback,
            calibration_min_points=calibration_min_points,
            interval_alpha=interval_alpha,
        )


@dataclass(frozen=True)
class IntervalCalibrator:
    method: str
    z_value: float
    data_points: int

    def adjusted_z(self, default_z: float) -> float:
        if self.method == "none" or self.z_value <= 0.0:
            return default_z
        return self.z_value


def _empirical_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, clip_value(q, 0.0, 1.0)))


def _cdf_from_abs_z(abs_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_abs = np.sort(abs_z)
    n = sorted_abs.size
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    empirical = (np.arange(n, dtype=float) + 1.0) / float(n)
    return sorted_abs, empirical


def _invert_monotone_cdf(x: np.ndarray, y: np.ndarray, target: float) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    if target <= y[0]:
        return float(x[0])
    if target >= y[-1]:
        return float(x[-1])
    idx = int(np.searchsorted(y, target))
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    if abs(y1 - y0) < 1e-12:
        return float(x1)
    frac = (target - y0) / (y1 - y0)
    return float(x0 + frac * (x1 - x0))


def build_interval_calibrator(
    z_scores: list[float],
    config: HardeningConfig,
    default_z: float = 1.96,
) -> IntervalCalibrator:
    cfg = config.validated()
    if (not cfg.enabled) or cfg.interval_calibration == "none":
        return IntervalCalibrator(method="none", z_value=default_z, data_points=len(z_scores))

    clean = np.array([abs(z) for z in z_scores if math.isfinite(z)], dtype=float)
    if clean.size < cfg.calibration_min_points:
        if cfg.calibration_fallback == "quantile_scale" and clean.size >= 3:
            z_q = _empirical_quantile(clean, cfg.interval_alpha)
            return IntervalCalibrator(method="quantile_scale_fallback", z_value=float(max(default_z, z_q)), data_points=int(clean.size))
        return IntervalCalibrator(method="none", z_value=default_z, data_points=int(clean.size))

    if cfg.interval_calibration == "quantile_scale":
        z_q = _empirical_quantile(clean, cfg.interval_alpha)
        return IntervalCalibrator(method="quantile_scale", z_value=float(max(1.0, z_q)), data_points=int(clean.size))

    # Isotonic-style monotone CDF inversion using empirical CDF.
    x, y = _cdf_from_abs_z(clean)
    z_iso = _invert_monotone_cdf(x, y, cfg.interval_alpha)
    if not math.isfinite(z_iso):
        if cfg.calibration_fallback == "quantile_scale":
            z_q = _empirical_quantile(clean, cfg.interval_alpha)
            return IntervalCalibrator(method="quantile_scale_fallback", z_value=float(max(1.0, z_q)), data_points=int(clean.size))
        return IntervalCalibrator(method="none", z_value=default_z, data_points=int(clean.size))
    return IntervalCalibrator(method="isotonic", z_value=float(max(1.0, z_iso)), data_points=int(clean.size))


def robust_weight(z: float, config: HardeningConfig) -> float:
    cfg = config.validated()
    if (not cfg.enabled) or cfg.robust_likelihood == "none":
        return 1.0
    abs_z = abs(z)
    if cfg.robust_likelihood == "huber":
        if abs_z <= cfg.huber_delta:
            return 1.0
        return cfg.huber_delta / max(abs_z, 1e-9)
    # Student-t style heavy-tail down-weighting.
    nu = cfg.student_t_df
    return (nu + 1.0) / (nu + z * z)


def outlier_scale_multiplier(z_scores: list[float], config: HardeningConfig) -> float:
    cfg = config.validated()
    if (not cfg.enabled) or not z_scores:
        return 1.0
    clipped = [min(abs(z), cfg.outlier_z_clip * 2.0) for z in z_scores if math.isfinite(z)]
    if not clipped:
        return 1.0
    exceed = [z for z in clipped if z > cfg.outlier_z_clip]
    if not exceed:
        return 1.0
    severity = float(np.mean([(z - cfg.outlier_z_clip) / cfg.outlier_z_clip for z in exceed]))
    return 1.0 + severity * (cfg.outlier_std_inflation - 1.0)
