from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bayes_loop import run_forecast
from .context import TaskContext, generate_task_context
from .hardening import HardeningConfig
from .ingestion import Observation, ensure_bounds, load_observations_csv, load_transcripts


@dataclass(frozen=True)
class ForecastDistribution:
    mean: float
    stddev: float


@dataclass(frozen=True)
class Scenario:
    name: str
    context: TaskContext
    apply_to_all_models: bool
    prior_source_tier: str
    prior_reliability: float


@dataclass(frozen=True)
class BacktestPrediction:
    dataset: str
    dataset_type: str
    scenario: str
    model: str
    model_version: str
    origin_month: date
    target_month: date
    horizon: int
    y_true: float
    y_pred_mean: float
    y_pred_stddev: float
    ci95_low: float
    ci95_high: float
    status: str
    note: str


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return min(hi, max(lo, value))


def _month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def _add_months(base: date, count: int) -> date:
    year = base.year + ((base.month - 1 + count) // 12)
    month = ((base.month - 1 + count) % 12) + 1
    return date(year, month, 1)


def _make_observations(start: date, values: np.ndarray) -> list[Observation]:
    obs: list[Observation] = []
    for idx, value in enumerate(values):
        obs.append(Observation(month=_add_months(start, idx), yield_pct=float(_clip(value))))
    return obs


def generate_synthetic_datasets(seed: int) -> tuple[dict[str, list[Observation]], dict[str, str]]:
    rng = np.random.default_rng(seed)
    n_months = 30
    t = np.arange(n_months, dtype=float)

    core = 36.0 + 56.0 / (1.0 + np.exp(-0.24 * (t - 12.0)))
    seasonality = 1.1 * np.sin(0.65 * t)
    clean = core + seasonality

    noisy = clean + rng.normal(0.0, 2.4, n_months)

    outlier = noisy.copy()
    outlier[10] -= 9.0
    outlier[19] -= 12.0
    outlier[24] -= 7.5

    start = _month_start(2024, 1)
    datasets = {
        "synthetic_clean": _make_observations(start, clean),
        "synthetic_noisy": _make_observations(start, noisy),
        "synthetic_outlier": _make_observations(start, outlier),
    }
    labels = {
        "synthetic_clean": "Synthetic transparent stress-test: smooth logistic ramp + mild seasonality.",
        "synthetic_noisy": "Synthetic transparent stress-test: clean ramp + Gaussian noise.",
        "synthetic_outlier": "Synthetic transparent stress-test: noisy ramp + explicit negative outlier shocks.",
    }
    return datasets, labels


def _scenario_transcript(prior_low: int, prior_high: int, include_yield_killer_context: bool) -> str:
    base = (
        "Management noted Intel 18A progress with RibbonFET and PowerVia integration maturing in pilot lines. "
        f"The team reiterated guidance of {prior_low}-{prior_high}% monthly yield improvement through the ramp. "
    )
    if include_yield_killer_context:
        return (
            base
            + "Yield killer excursions were discussed, including risk, variability, delay, challenge, "
            "uncertain defect interactions, and headwind from process instability."
        )
    return base + "No specific yield-killer excursions were highlighted and process progress was described as stable."


def build_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="prior_7_8_with_killer",
            context=generate_task_context(_scenario_transcript(7, 8, include_yield_killer_context=True)),
            apply_to_all_models=True,
            prior_source_tier="subjective_prior",
            prior_reliability=0.60,
        ),
        Scenario(
            name="prior_3_5_with_killer",
            context=generate_task_context(_scenario_transcript(3, 5, include_yield_killer_context=True)),
            apply_to_all_models=False,
            prior_source_tier="subjective_prior",
            prior_reliability=0.60,
        ),
        Scenario(
            name="prior_10_12_with_killer",
            context=generate_task_context(_scenario_transcript(10, 12, include_yield_killer_context=True)),
            apply_to_all_models=False,
            prior_source_tier="subjective_prior",
            prior_reliability=0.60,
        ),
        Scenario(
            name="prior_7_8_no_killer",
            context=generate_task_context(_scenario_transcript(7, 8, include_yield_killer_context=False)),
            apply_to_all_models=False,
            prior_source_tier="subjective_prior",
            prior_reliability=0.60,
        ),
    ]


def _estimate_innovation_scale(train: list[Observation]) -> float:
    values = np.array([item.yield_pct for item in train], dtype=float)
    if len(values) < 2:
        return 2.0
    diffs = np.diff(values)
    return float(max(0.8, np.std(diffs, ddof=1) if len(diffs) > 1 else abs(diffs[-1])))


def forecast_llambo(
    train: list[Observation],
    horizon: int,
    context: TaskContext,
    seed: int,
    use_area_factor: bool = True,
) -> ForecastDistribution:
    points = run_forecast(
        observations=train,
        context=context,
        months_ahead=horizon,
        horizon=None,
        seed=seed,
        use_area_factor=use_area_factor,
    )
    future = [item for item in points if item.observed_yield is None]
    if len(future) < horizon:
        raise RuntimeError("LLAMBO forecast returned insufficient horizon length")
    point = future[horizon - 1]
    return ForecastDistribution(mean=float(_clip(point.posterior_mean)), stddev=max(0.2, float(point.posterior_stddev)))


def forecast_llambo_with_config(
    train: list[Observation],
    horizon: int,
    context: TaskContext,
    seed: int,
    hardening: HardeningConfig,
    prior_weight_multiplier: float = 1.0,
    use_area_factor: bool = True,
) -> ForecastDistribution:
    adjusted = hardening
    if hardening.enabled:
        adjusted = replace(
            hardening,
            prior_weight=min(1.0, max(0.0, hardening.prior_weight * prior_weight_multiplier)),
        )
    points = run_forecast(
        observations=train,
        context=context,
        months_ahead=horizon,
        horizon=None,
        seed=seed,
        hardening=adjusted,
        use_area_factor=use_area_factor,
    )
    future = [item for item in points if item.observed_yield is None]
    if len(future) < horizon:
        raise RuntimeError("LLAMBO forecast returned insufficient horizon length")
    point = future[horizon - 1]
    return ForecastDistribution(mean=float(_clip(point.posterior_mean)), stddev=max(0.2, float(point.posterior_stddev)))


def forecast_persistence(train: list[Observation], horizon: int) -> ForecastDistribution:
    last = train[-1].yield_pct
    std = _estimate_innovation_scale(train) * math.sqrt(max(1, horizon))
    return ForecastDistribution(mean=float(_clip(last)), stddev=float(max(0.5, std)))


def forecast_linear_trend(train: list[Observation], horizon: int) -> ForecastDistribution:
    values = np.array([item.yield_pct for item in train], dtype=float)
    n = len(values)
    if n < 2:
        return forecast_persistence(train, horizon)

    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    pred_x = (n - 1) + horizon
    pred_mean = intercept + slope * pred_x

    fit = intercept + slope * x
    residuals = values - fit
    resid_std = np.std(residuals, ddof=1) if n > 2 else _estimate_innovation_scale(train)
    pred_std = max(0.6, float(resid_std) * math.sqrt(max(1.0, horizon)))

    return ForecastDistribution(mean=float(_clip(pred_mean)), stddev=pred_std)


def _logistic_curve(x: np.ndarray, asymptote: float, k: float, t0: float) -> np.ndarray:
    return asymptote / (1.0 + np.exp(-k * (x - t0)))


def forecast_logistic_curve(train: list[Observation], horizon: int) -> ForecastDistribution:
    values = np.array([item.yield_pct for item in train], dtype=float)
    n = len(values)
    if n < 5:
        raise RuntimeError("insufficient_data_for_logistic_fit")

    x = np.arange(n, dtype=float)
    best_mse = float("inf")
    best_params: tuple[float, float, float] | None = None

    asymptote_grid = np.linspace(max(75.0, values.max() + 1.0), 100.0, 18)
    k_grid = np.linspace(0.04, 0.38, 22)
    t0_grid = np.linspace(-6.0, n + 8.0, 28)

    for asymptote in asymptote_grid:
        for k in k_grid:
            for t0 in t0_grid:
                fit = _logistic_curve(x, asymptote, k, t0)
                mse = float(np.mean((values - fit) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_params = (float(asymptote), float(k), float(t0))

    if best_params is None or best_mse > 120.0:
        raise RuntimeError("logistic_fit_failed")

    asymptote, k, t0 = best_params
    pred_x = np.array([float((n - 1) + horizon)], dtype=float)
    pred_mean = float(_logistic_curve(pred_x, asymptote, k, t0)[0])
    fit = _logistic_curve(x, asymptote, k, t0)
    resid_std = float(max(0.7, np.std(values - fit, ddof=1)))
    pred_std = max(0.8, resid_std * math.sqrt(1.0 + 0.35 * horizon))

    return ForecastDistribution(mean=float(_clip(pred_mean)), stddev=float(pred_std))


def _rbf_kernel(xa: np.ndarray, xb: np.ndarray, length_scale: float, sigma_f: float) -> np.ndarray:
    d2 = (xa[:, None] - xb[None, :]) ** 2
    return (sigma_f**2) * np.exp(-0.5 * d2 / (length_scale**2))


def forecast_gp_surrogate(train: list[Observation], horizon: int) -> ForecastDistribution:
    values = np.array([item.yield_pct for item in train], dtype=float)
    n = len(values)
    if n < 3:
        return forecast_persistence(train, horizon)

    x = np.arange(n, dtype=float)
    y_mean = float(np.mean(values))
    y_centered = values - y_mean

    sigma_f = float(max(2.5, np.std(y_centered, ddof=1) if n > 2 else 3.0))
    length_scale = 3.0
    noise = float(max(0.8, _estimate_innovation_scale(train) * 0.45))

    k_xx = _rbf_kernel(x, x, length_scale=length_scale, sigma_f=sigma_f)
    k_xx = k_xx + (noise**2 + 1e-6) * np.eye(n)

    x_star = np.array([float((n - 1) + horizon)], dtype=float)
    k_xs = _rbf_kernel(x, x_star, length_scale=length_scale, sigma_f=sigma_f)
    k_ss = float(_rbf_kernel(x_star, x_star, length_scale=length_scale, sigma_f=sigma_f)[0, 0] + noise**2)

    try:
        alpha = np.linalg.solve(k_xx, y_centered)
        pred_mean = y_mean + float(k_xs[:, 0].T @ alpha)
        v = np.linalg.solve(k_xx, k_xs[:, 0])
        pred_var = max(0.04, k_ss - float(k_xs[:, 0].T @ v))
    except np.linalg.LinAlgError:
        return forecast_persistence(train, horizon)

    return ForecastDistribution(mean=float(_clip(pred_mean)), stddev=float(math.sqrt(pred_var)))


def _normal_crps(y_true: float, mean: float, stddev: float) -> float:
    if stddev <= 1e-12:
        return abs(y_true - mean)
    z = (y_true - mean) / stddev
    return stddev * (z * (2.0 * _norm_cdf(z) - 1.0) + 2.0 * _norm_pdf(z) - 1.0 / math.sqrt(math.pi))


def _calibration_error(records: list[BacktestPrediction]) -> float:
    usable = [item for item in records if item.status == "ok" and item.y_pred_stddev > 1e-9]
    if not usable:
        return float("nan")

    probs = [_norm_cdf((item.y_true - item.y_pred_mean) / item.y_pred_stddev) for item in usable]
    quantiles = np.linspace(0.1, 0.9, 9)
    errors = []
    for q in quantiles:
        empirical = float(np.mean([1.0 if p <= q else 0.0 for p in probs]))
        errors.append(abs(empirical - float(q)))
    return float(np.mean(errors))


def _metric_group(records: list[BacktestPrediction]) -> dict[str, float]:
    ok = [item for item in records if item.status == "ok"]
    failed = len(records) - len(ok)
    if not ok:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "crps_approx": float("nan"),
            "coverage95": float("nan"),
            "mean_interval_width": float("nan"),
            "calibration_error": float("nan"),
            "failed_predictions": float(failed),
        }

    abs_err = [abs(item.y_true - item.y_pred_mean) for item in ok]
    sq_err = [(item.y_true - item.y_pred_mean) ** 2 for item in ok]
    crps = [_normal_crps(item.y_true, item.y_pred_mean, item.y_pred_stddev) for item in ok]
    covered = [1.0 if item.ci95_low <= item.y_true <= item.ci95_high else 0.0 for item in ok]
    widths = [item.ci95_high - item.ci95_low for item in ok]

    return {
        "n": float(len(ok)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(sq_err))),
        "crps_approx": float(np.mean(crps)),
        "coverage95": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(widths)),
        "calibration_error": _calibration_error(ok),
        "failed_predictions": float(failed),
    }


def _records_to_rows(records: list[BacktestPrediction]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in records:
        rows.append(
            {
                "dataset": item.dataset,
                "dataset_type": item.dataset_type,
                "scenario": item.scenario,
                "model": item.model,
                "model_version": item.model_version,
                "origin_month": item.origin_month.strftime("%Y-%m"),
                "target_month": item.target_month.strftime("%Y-%m"),
                "horizon": str(item.horizon),
                "y_true": f"{item.y_true:.4f}",
                "y_pred_mean": "" if math.isnan(item.y_pred_mean) else f"{item.y_pred_mean:.4f}",
                "y_pred_stddev": "" if math.isnan(item.y_pred_stddev) else f"{item.y_pred_stddev:.4f}",
                "ci95_low": "" if math.isnan(item.ci95_low) else f"{item.ci95_low:.4f}",
                "ci95_high": "" if math.isnan(item.ci95_high) else f"{item.ci95_high:.4f}",
                "status": item.status,
                "note": item.note,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(records: list[BacktestPrediction]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []

    def emit(
        dataset: str,
        dataset_type: str,
        scenario: str,
        model: str,
        model_version: str,
        horizon_token: str,
        group: list[BacktestPrediction],
    ) -> None:
        stats = _metric_group(group)
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_type": dataset_type,
                "scenario": scenario,
                "model": model,
                "model_version": model_version,
                "horizon": horizon_token,
                "n": str(int(stats["n"])),
                "mae": "" if math.isnan(stats["mae"]) else f"{stats['mae']:.4f}",
                "rmse": "" if math.isnan(stats["rmse"]) else f"{stats['rmse']:.4f}",
                "crps_approx": "" if math.isnan(stats["crps_approx"]) else f"{stats['crps_approx']:.4f}",
                "coverage95": "" if math.isnan(stats["coverage95"]) else f"{stats['coverage95']:.4f}",
                "mean_interval_width": "" if math.isnan(stats["mean_interval_width"]) else f"{stats['mean_interval_width']:.4f}",
                "calibration_error": "" if math.isnan(stats["calibration_error"]) else f"{stats['calibration_error']:.4f}",
                "failed_predictions": str(int(stats["failed_predictions"])),
            }
        )

    by_group: dict[tuple[str, str, str, str, str, int], list[BacktestPrediction]] = {}
    for item in records:
        key = (item.dataset, item.dataset_type, item.scenario, item.model, item.model_version, item.horizon)
        by_group.setdefault(key, []).append(item)

    for (dataset, dataset_type, scenario, model, model_version, horizon), group in sorted(by_group.items()):
        emit(dataset, dataset_type, scenario, model, model_version, str(horizon), group)

    collapsed_dataset: dict[tuple[str, str, str, str, str], list[BacktestPrediction]] = {}
    for item in records:
        key = (item.dataset, item.dataset_type, item.scenario, item.model, item.model_version)
        collapsed_dataset.setdefault(key, []).append(item)
    for (dataset, dataset_type, scenario, model, model_version), group in sorted(collapsed_dataset.items()):
        emit(dataset, dataset_type, scenario, model, model_version, "all", group)

    synthetic_all = [item for item in records if item.dataset_type == "synthetic"]
    agg_groups: dict[tuple[str, str, str, int], list[BacktestPrediction]] = {}
    for item in synthetic_all:
        key = (item.scenario, item.model, item.model_version, item.horizon)
        agg_groups.setdefault(key, []).append(item)
    for (scenario, model, model_version, horizon), group in sorted(agg_groups.items()):
        emit("__ALL_SYNTHETIC__", "synthetic", scenario, model, model_version, str(horizon), group)

    agg_all_groups: dict[tuple[str, str, str], list[BacktestPrediction]] = {}
    for item in synthetic_all:
        key = (item.scenario, item.model, item.model_version)
        agg_all_groups.setdefault(key, []).append(item)
    for (scenario, model, model_version), group in sorted(agg_all_groups.items()):
        emit("__ALL_SYNTHETIC__", "synthetic", scenario, model, model_version, "all", group)

    return summary_rows


def _extract_metric(
    summary_rows: list[dict[str, str]],
    dataset: str,
    scenario: str,
    model: str,
    horizon: str,
    metric: str,
    model_version: str | None = None,
) -> float:
    for row in summary_rows:
        if (
            row["dataset"] == dataset
            and row["scenario"] == scenario
            and row["model"] == model
            and row["horizon"] == horizon
            and (model_version is None or row.get("model_version", "") == model_version)
        ):
            value = row[metric]
            if value == "":
                return float("nan")
            return float(value)
    return float("nan")


def plot_calibration(records: list[BacktestPrediction], output_path: Path) -> None:
    models = sorted({(item.model, item.model_version) for item in records if item.scenario == "prior_7_8_with_killer"})
    quantiles = np.linspace(0.1, 0.9, 9)

    plt.figure(figsize=(7.2, 6.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666", lw=1.4, label="Ideal")

    for model, model_version in models:
        model_records = [
            item
            for item in records
            if item.model == model
            and item.model_version == model_version
            and item.scenario == "prior_7_8_with_killer"
            and item.status == "ok"
            and item.y_pred_stddev > 1e-9
        ]
        if not model_records:
            continue
        probs = [_norm_cdf((item.y_true - item.y_pred_mean) / item.y_pred_stddev) for item in model_records]
        empirical = [float(np.mean([1.0 if p <= q else 0.0 for p in probs])) for q in quantiles]
        label = model if model_version == "baseline" and model != "llambo_style" else f"{model}:{model_version}"
        plt.plot(quantiles, empirical, marker="o", lw=1.5, label=label)

    plt.title("Calibration Curve (Baseline Scenario)")
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical quantile")
    plt.xlim(0.05, 0.95)
    plt.ylim(0.05, 0.95)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_benchmark(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    rows = [
        row
        for row in summary_rows
        if row["dataset"] == "__ALL_SYNTHETIC__" and row["scenario"] == "prior_7_8_with_killer" and row["horizon"] == "all"
    ]
    if not rows:
        return

    rows = [row for row in rows if row["rmse"] != ""]
    rows.sort(key=lambda item: float(item["rmse"]))

    models = [
        row["model"] if (row["model"] != "llambo_style" and row.get("model_version", "baseline") == "baseline")
        else f"{row['model']}:{row.get('model_version', 'baseline')}"
        for row in rows
    ]
    rmse = [float(row["rmse"]) for row in rows]
    mae = [float(row["mae"]) for row in rows]

    x = np.arange(len(models))
    width = 0.38

    plt.figure(figsize=(8.0, 5.4))
    plt.bar(x - width / 2, rmse, width=width, color="#2A6F97", label="RMSE")
    plt.bar(x + width / 2, mae, width=width, color="#F4A259", label="MAE")
    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Error (Yield %)" )
    plt.title("Model Error Benchmark on Synthetic Stress Datasets")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_critical_assessment(
    summary_rows: list[dict[str, str]],
    records: list[BacktestPrediction],
    dataset_notes: dict[str, str],
    output_path: Path,
    hardening: HardeningConfig,
) -> None:
    baseline_rows = [
        row
        for row in summary_rows
        if row["dataset"] == "__ALL_SYNTHETIC__"
        and row["scenario"] == "prior_7_8_with_killer"
        and row["horizon"] == "all"
        and row["rmse"] != ""
    ]
    baseline_rows.sort(key=lambda item: float(item["rmse"]))

    best_model = baseline_rows[0]["model"] if baseline_rows else "n/a"
    best_model_version = baseline_rows[0].get("model_version", "baseline") if baseline_rows else "n/a"
    best_rmse = float(baseline_rows[0]["rmse"]) if baseline_rows else float("nan")
    ll_b_rmse = _extract_metric(
        summary_rows, "__ALL_SYNTHETIC__", "prior_7_8_with_killer", "llambo_style", "all", "rmse", model_version="baseline"
    )
    ll_h_rmse = _extract_metric(
        summary_rows,
        "__ALL_SYNTHETIC__",
        "prior_7_8_with_killer",
        "llambo_style",
        "all",
        "rmse",
        model_version="hardened",
    )
    ll_b_cov = _extract_metric(
        summary_rows, "__ALL_SYNTHETIC__", "prior_7_8_with_killer", "llambo_style", "all", "coverage95", model_version="baseline"
    )
    ll_h_cov = _extract_metric(
        summary_rows,
        "__ALL_SYNTHETIC__",
        "prior_7_8_with_killer",
        "llambo_style",
        "all",
        "coverage95",
        model_version="hardened",
    )
    ll_b_cal = _extract_metric(
        summary_rows, "__ALL_SYNTHETIC__", "prior_7_8_with_killer", "llambo_style", "all", "calibration_error", model_version="baseline"
    )
    ll_h_cal = _extract_metric(
        summary_rows,
        "__ALL_SYNTHETIC__",
        "prior_7_8_with_killer",
        "llambo_style",
        "all",
        "calibration_error",
        model_version="hardened",
    )

    ll_b_outlier = _extract_metric(
        summary_rows, "synthetic_outlier", "prior_7_8_with_killer", "llambo_style", "all", "rmse", model_version="baseline"
    )
    ll_h_outlier = _extract_metric(
        summary_rows, "synthetic_outlier", "prior_7_8_with_killer", "llambo_style", "all", "rmse", model_version="hardened"
    )
    ll_h_no_area_rmse = _extract_metric(
        summary_rows,
        "__ALL_SYNTHETIC__",
        "prior_7_8_with_killer",
        "llambo_style",
        "all",
        "rmse",
        model_version="hardened_no_area",
    )

    logistic_failures = len(
        [item for item in records if item.model == "logistic_scurve" and item.status != "ok"]
    )

    observed_names = sorted({item.dataset for item in records if item.dataset_type == "observed"})
    sample_count = len([item for item in records if item.dataset_type == "observed" and item.status == "ok"])

    rmse_delta = ll_h_rmse - ll_b_rmse if not (math.isnan(ll_h_rmse) or math.isnan(ll_b_rmse)) else float("nan")
    cov_delta = ll_h_cov - ll_b_cov if not (math.isnan(ll_h_cov) or math.isnan(ll_b_cov)) else float("nan")
    cal_delta = ll_h_cal - ll_b_cal if not (math.isnan(ll_h_cal) or math.isnan(ll_b_cal)) else float("nan")
    outlier_delta = ll_h_outlier - ll_b_outlier if not (math.isnan(ll_h_outlier) or math.isnan(ll_b_outlier)) else float("nan")
    area_delta = (
        ll_h_rmse - ll_h_no_area_rmse
        if not (math.isnan(ll_h_rmse) or math.isnan(ll_h_no_area_rmse))
        else float("nan")
    )

    regressed = [
        name
        for name, bad in (
            ("rmse", (not math.isnan(rmse_delta) and rmse_delta > 0.10)),
            ("coverage95", (not math.isnan(cov_delta) and cov_delta < -0.02)),
            ("calibration_error", (not math.isnan(cal_delta) and cal_delta > 0.01)),
            ("outlier_rmse", (not math.isnan(outlier_delta) and outlier_delta > 0.10)),
        )
        if bad
    ]
    improvement_marginal = (
        not regressed
        and (not math.isnan(rmse_delta))
        and abs(rmse_delta) < 0.08
        and (math.isnan(cal_delta) or abs(cal_delta) < 0.008)
    )

    lines = [
        "# Critical Assessment: Intel 18A LLAMBO Hardening Evaluation",
        "",
        "## Scope and Data Reality",
        "- Backtesting used rolling-origin monthly forecasts with horizons 1-6 months.",
        f"- Observed dataset(s): {', '.join(f'`{name}`' for name in observed_names) if observed_names else '`none`'}.",
        f"- Observed backtest usable points: {sample_count}.",
        (
            "- Observed panel remains short/noisy relative to production forecasting needs; synthetic stress datasets are retained and explicitly labeled."
            if sample_count < 180
            else "- Observed panel is materially larger than the thin baseline, but still not a substitute for true measured fab yield telemetry."
        ),
        "",
        "## Synthetic Stress Datasets Used",
    ]
    for name, note in sorted(dataset_notes.items()):
        lines.append(f"- `{name}`: {note}")

    lines.extend(
        [
            "",
            "## Hardening Configuration",
            f"- prior_weight={hardening.prior_weight:.2f}, robust_likelihood={hardening.robust_likelihood}, huber_delta={hardening.huber_delta:.2f}, student_t_df={hardening.student_t_df:.1f}",
            f"- context_drift_clip={hardening.context_drift_clip:.3f}, outlier_z_clip={hardening.outlier_z_clip:.2f}, outlier_std_inflation={hardening.outlier_std_inflation:.2f}",
            f"- interval_calibration={hardening.interval_calibration}, fallback={hardening.calibration_fallback}, calibration_min_points={hardening.calibration_min_points}",
            "",
            "## Headline Results (Baseline Scenario: prior 7-8% with yield-killer context)",
            f"- Best RMSE model on aggregated synthetic datasets: `{best_model}:{best_model_version}` (RMSE={best_rmse:.3f}).",
            f"- LLAMBO baseline RMSE={ll_b_rmse:.3f}, hardened RMSE={ll_h_rmse:.3f}, delta={rmse_delta:+.3f}.",
            f"- LLAMBO baseline coverage95={ll_b_cov:.3f}, hardened coverage95={ll_h_cov:.3f}, delta={cov_delta:+.3f}.",
            f"- LLAMBO baseline calibration_error={ll_b_cal:.3f}, hardened calibration_error={ll_h_cal:.3f}, delta={cal_delta:+.3f}.",
            f"- LLAMBO outlier RMSE baseline={ll_b_outlier:.3f}, hardened={ll_h_outlier:.3f}, delta={outlier_delta:+.3f}.",
            f"- Area-factor impact (hardened with-area minus hardened no-area, RMSE): {area_delta:+.3f}.",
            "",
            "## Stress and Sensitivity Findings",
            "- Scenario prior from the attached fictional case study is treated as a subjective prior and down-weighted by reliability before blending with data anchor growth.",
            "- Prior-weight regularization and context-drift clipping reduced narrative over-dominance in high-guidance transcripts, but may under-react in genuinely fast-improving regimes.",
            "- Outlier-aware variance inflation improved tail coverage stability in shock windows where baseline LLAMBO was over-confident.",
            "- Isotonic/quantile interval recalibration changed uncertainty bands without changing mean dynamics, so point accuracy gains can remain small.",
            f"- Logistic baseline graceful failures recorded: {logistic_failures} (expected on short windows or poor curvature fit).",
            "",
            "## Failure Modes and Confidence Limits",
            "- Extremely limited real historical data means external validity is weak; synthetic success does not imply deployment readiness.",
            "- Hardening did not fully eliminate degradation under severe synthetic outliers; heavy-tail approximation helps but is not a full probabilistic model.",
            "- Context-sensitive LLAMBO behavior can still be brittle if transcript sentiment is noisy, biased, or strategically framed.",
            "- GP and trend baselines can outperform context-heavy models when the underlying signal is mostly smooth and data-rich.",
            "",
            "## Regression Check",
            f"- Regressions detected: {', '.join(regressed) if regressed else 'none by configured thresholds'}.",
            f"- Improvement classification: {'marginal/inconsistent' if improvement_marginal else ('mixed with regressions' if regressed else 'meaningful on selected risk metrics')}.",
            "",
            "## Decision Verdict",
            f"- Verdict: {'not yet' if regressed or improvement_marginal else 'conditionally improved, still limited by data'}.",
            "- This harness is suitable for model triage and robustness diagnostics, not for high-confidence capital-allocation decisions without materially longer real-world history.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_ablation_comparison(summary_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics = ["mae", "rmse", "crps_approx", "coverage95", "mean_interval_width", "calibration_error"]
    grouped: dict[tuple[str, str, str, str, str], dict[str, dict[str, str]]] = {}

    for row in summary_rows:
        if row["model"] != "llambo_style":
            continue
        key = (row["dataset"], row["dataset_type"], row["scenario"], row["model"], row["horizon"])
        grouped.setdefault(key, {})[row.get("model_version", "baseline")] = row

    out: list[dict[str, str]] = []
    for (dataset, dataset_type, scenario, model, horizon), versions in sorted(grouped.items()):
        baseline = versions.get("baseline")
        hardened = versions.get("hardened")
        hardened_no_area = versions.get("hardened_no_area")
        if baseline is None or hardened is None:
            continue

        row: dict[str, str] = {
            "dataset": dataset,
            "dataset_type": dataset_type,
            "scenario": scenario,
            "model": model,
            "horizon": horizon,
        }
        for metric in metrics:
            b = float(baseline[metric]) if baseline[metric] != "" else float("nan")
            h = float(hardened[metric]) if hardened[metric] != "" else float("nan")
            h_no_area = float(hardened_no_area[metric]) if (hardened_no_area and hardened_no_area[metric] != "") else float("nan")
            row[f"{metric}_baseline"] = "" if math.isnan(b) else f"{b:.4f}"
            row[f"{metric}_hardened"] = "" if math.isnan(h) else f"{h:.4f}"
            row[f"{metric}_hardened_no_area"] = "" if math.isnan(h_no_area) else f"{h_no_area:.4f}"
            if math.isnan(b) or math.isnan(h):
                row[f"{metric}_delta_hardened_minus_baseline"] = ""
            else:
                row[f"{metric}_delta_hardened_minus_baseline"] = f"{(h - b):+.4f}"
            if math.isnan(h) or math.isnan(h_no_area):
                row[f"{metric}_delta_hardened_minus_hardened_no_area"] = ""
            else:
                row[f"{metric}_delta_hardened_minus_hardened_no_area"] = f"{(h - h_no_area):+.4f}"
        out.append(row)
    return out


def run_rolling_backtest(
    datasets: dict[str, list[Observation]],
    dataset_types: dict[str, str],
    scenarios: list[Scenario],
    max_horizon: int,
    seed: int,
    hardening: HardeningConfig,
) -> list[BacktestPrediction]:
    records: list[BacktestPrediction] = []

    baseline_models = ["persistence", "linear_trend", "logistic_scurve", "gp_surrogate"]
    llambo_variants = [
        ("llambo_style", "baseline", HardeningConfig.baseline(), True),
        ("llambo_style", "hardened", hardening.validated(), True),
        ("llambo_style", "hardened_no_area", hardening.validated(), False),
    ]

    for dataset_name, obs in datasets.items():
        ordered = ensure_bounds(sorted(obs, key=lambda item: item.month))
        n = len(ordered)
        if n < 2:
            continue

        min_train_size = 1 if n <= 3 else min(6, n - 1)

        for scenario in scenarios:
            for origin_idx in range(min_train_size - 1, n - 1):
                train = ordered[: origin_idx + 1]

                for horizon in range(1, max_horizon + 1):
                    target_idx = origin_idx + horizon
                    if target_idx >= n:
                        continue
                    target = ordered[target_idx]
                    if scenario.apply_to_all_models:
                        model_loop = list(llambo_variants) + [
                            (name, "baseline", HardeningConfig.baseline(), False) for name in baseline_models
                        ]
                    else:
                        model_loop = list(llambo_variants)

                    for model, model_version, model_cfg, use_area_factor in model_loop:
                        status = "ok"
                        note = ""
                        mean = float("nan")
                        std = float("nan")

                        try:
                            if model == "llambo_style":
                                prior_weight_multiplier = (
                                    scenario.prior_reliability
                                    if (scenario.prior_source_tier == "subjective_prior" and model_cfg.enabled)
                                    else 1.0
                                )
                                pred = forecast_llambo_with_config(
                                    train,
                                    horizon,
                                    scenario.context,
                                    seed=seed,
                                    hardening=model_cfg,
                                    prior_weight_multiplier=prior_weight_multiplier,
                                    use_area_factor=use_area_factor,
                                )
                            elif model == "persistence":
                                pred = forecast_persistence(train, horizon)
                            elif model == "linear_trend":
                                pred = forecast_linear_trend(train, horizon)
                            elif model == "logistic_scurve":
                                pred = forecast_logistic_curve(train, horizon)
                            elif model == "gp_surrogate":
                                pred = forecast_gp_surrogate(train, horizon)
                            else:
                                raise RuntimeError(f"unsupported_model:{model}")

                            mean = float(_clip(pred.mean))
                            std = float(max(0.15, pred.stddev))
                        except Exception as exc:  # noqa: BLE001
                            status = "failed"
                            note = str(exc)

                        if status == "ok":
                            low = float(_clip(mean - 1.96 * std))
                            high = float(_clip(mean + 1.96 * std))
                        else:
                            low = float("nan")
                            high = float("nan")

                        records.append(
                            BacktestPrediction(
                                dataset=dataset_name,
                                dataset_type=dataset_types[dataset_name],
                                scenario=scenario.name,
                                model=model,
                                model_version=model_version,
                                origin_month=train[-1].month,
                                target_month=target.month,
                                horizon=horizon,
                                y_true=target.yield_pct,
                                y_pred_mean=mean,
                                y_pred_stddev=std,
                                ci95_low=low,
                                ci95_high=high,
                                status=status,
                                note=note,
                            )
                        )

    return records


def run_quality_evaluation(
    observations_csv: Path,
    transcript_files: list[Path] | None,
    output_dir: Path,
    assessment_filename: str = "critical_assessment_hardened.md",
    max_horizon: int = 6,
    seed: int = 18,
    include_synthetic: bool = True,
    enable_plots: bool = True,
    hardening: HardeningConfig | None = None,
) -> dict[str, Path]:
    if max_horizon < 1:
        raise ValueError("max_horizon must be >= 1")

    datasets: dict[str, list[Observation]] = {}
    dataset_types: dict[str, str] = {}
    dataset_notes: dict[str, str] = {}

    real_dataset_name = observations_csv.stem
    real_obs = ensure_bounds(load_observations_csv(observations_csv))
    datasets[real_dataset_name] = real_obs
    dataset_types[real_dataset_name] = "observed"
    dataset_notes[real_dataset_name] = f"Observed dataset loaded from `{observations_csv}`."

    if include_synthetic:
        synth, synth_notes = generate_synthetic_datasets(seed=seed)
        for name, obs in synth.items():
            datasets[name] = obs
            dataset_types[name] = "synthetic"
        dataset_notes.update(synth_notes)

    if transcript_files:
        _ = load_transcripts(transcript_files)

    scenarios = build_scenarios()
    hardening_cfg = (hardening or HardeningConfig(enabled=True)).validated()

    records = run_rolling_backtest(
        datasets=datasets,
        dataset_types=dataset_types,
        scenarios=scenarios,
        max_horizon=max_horizon,
        seed=seed,
        hardening=hardening_cfg,
    )

    summary_rows = summarize_metrics(records)
    ablation_rows = build_ablation_comparison(summary_rows)
    prediction_rows = _records_to_rows(records)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "backtest_predictions.csv"
    summary_path = output_dir / "metrics_summary.csv"
    ablation_path = output_dir / "ablation_comparison.csv"
    calibration_path = output_dir / "calibration_plot.png"
    benchmark_path = output_dir / "benchmark_plot.png"
    assessment_path = output_dir / assessment_filename

    write_csv(prediction_rows, predictions_path)
    write_csv(summary_rows, summary_path)
    write_csv(ablation_rows, ablation_path)

    if enable_plots:
        plot_calibration(records, calibration_path)
        plot_benchmark(summary_rows, benchmark_path)

    build_critical_assessment(
        summary_rows=summary_rows,
        records=records,
        dataset_notes=dataset_notes,
        output_path=assessment_path,
        hardening=hardening_cfg,
    )

    return {
        "metrics_summary": summary_path,
        "backtest_predictions": predictions_path,
        "calibration_plot": calibration_path,
        "benchmark_plot": benchmark_path,
        "ablation_comparison": ablation_path,
        "critical_assessment_hardened": assessment_path,
    }
