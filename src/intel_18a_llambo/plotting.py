from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .bayes_loop import ForecastPoint


def plot_learning_curve(points: list[ForecastPoint], output_path: Path) -> None:
    months = [item.month.strftime("%Y-%m") for item in points]
    means = [item.posterior_mean for item in points]
    lows = [item.ci95_low for item in points]
    highs = [item.ci95_high for item in points]

    observed_x = [idx for idx, item in enumerate(points) if item.observed_yield is not None]
    observed_y = [item.observed_yield for item in points if item.observed_yield is not None]
    forecast_x = [idx for idx, item in enumerate(points) if item.observed_yield is None]
    forecast_y = [item.posterior_mean for item in points if item.observed_yield is None]

    plt.figure(figsize=(11, 5.5))
    plt.plot(range(len(points)), means, color="#0E7A0D", lw=2.3, label="Posterior mean")
    plt.fill_between(range(len(points)), lows, highs, color="#9ED9A0", alpha=0.35, label="95% CI")

    if observed_x:
        plt.scatter(observed_x, observed_y, color="#0F3557", s=52, zorder=3, label="Observed")
    if forecast_x:
        plt.scatter(forecast_x, forecast_y, color="#B84A00", s=42, zorder=3, label="Forecast")

    plt.title("Intel 18A Yield Learning Curve (LLAMBO-style Bayesian Forecast)")
    plt.ylabel("Yield (%)")
    plt.xlabel("Month")
    plt.ylim(0, 100)
    plt.xticks(range(len(points)), months, rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.22)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
