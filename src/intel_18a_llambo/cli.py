from __future__ import annotations

from argparse import ArgumentParser
from datetime import date
from pathlib import Path
import csv

from .bayes_loop import ForecastPoint, run_forecast
from .context import default_task_context, generate_task_context
from .hardening import HardeningConfig
from .ingestion import ensure_bounds, load_observations_csv, load_transcripts, parse_inline_observations
from .llambo_integration import detect_llambo_repo
from .plotting import plot_learning_curve


def parse_horizon(value: str) -> date:
    token = value.strip()
    if len(token) != 7 or token[4] != "-":
        raise ValueError("Horizon must be YYYY-MM format.")
    return date(int(token[:4]), int(token[5:7]), 1)


def write_forecast_csv(points: list[ForecastPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "month",
                "observed_yield",
                "posterior_mean",
                "posterior_stddev",
                "ci95_low",
                "ci95_high",
                "selected_growth_rate",
                "acquisition_value",
                "area_factor",
            ],
        )
        writer.writeheader()
        for item in points:
            writer.writerow(
                {
                    "month": item.month.strftime("%Y-%m"),
                    "observed_yield": "" if item.observed_yield is None else f"{item.observed_yield:.4f}",
                    "posterior_mean": f"{item.posterior_mean:.4f}",
                    "posterior_stddev": f"{item.posterior_stddev:.4f}",
                    "ci95_low": f"{item.ci95_low:.4f}",
                    "ci95_high": f"{item.ci95_high:.4f}",
                    "selected_growth_rate": "" if item.selected_growth_rate is None else f"{item.selected_growth_rate:.4f}",
                    "acquisition_value": "" if item.acquisition_value is None else f"{item.acquisition_value:.6f}",
                    "area_factor": "" if item.area_factor is None else f"{item.area_factor:.4f}",
                }
            )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Intel 18A LLAMBO-style Bayesian yield forecaster")
    parser.add_argument("--observations-csv", type=Path, default=None, help="CSV file with month,yield columns")
    parser.add_argument("--observations-inline", type=str, default=None, help='Inline observations, e.g. "Jan=64, Feb=68.5"')
    parser.add_argument("--transcript-files", type=Path, nargs="*", default=None, help="Optional local transcript files")
    parser.add_argument("--months-ahead", type=int, default=6, help="Minimum months to forecast")
    parser.add_argument("--horizon", type=str, default="2026-08", help="Forecast horizon in YYYY-MM")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/forecast.csv"), help="Forecast output CSV path")
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("outputs/intel18a_yield_curve.png"),
        help="Output PNG path for the forecast chart",
    )
    parser.add_argument("--seed", type=int, default=18, help="Random seed for reproducibility")
    parser.add_argument("--disable-hardening", action="store_true", help="Disable hardening layer for LLAMBO forecast")
    parser.add_argument("--prior-weight", type=float, default=0.65, help="Weight on context prior vs data anchor (0-1)")
    parser.add_argument(
        "--robust-likelihood",
        type=str,
        default="huber",
        choices=["none", "huber", "student_t"],
        help="Robust heavy-tail approximation",
    )
    parser.add_argument("--huber-delta", type=float, default=1.75, help="Huber threshold in standardized residual units")
    parser.add_argument("--student-t-df", type=float, default=5.0, help="Student-t degrees of freedom")
    parser.add_argument("--context-drift-clip", type=float, default=0.02, help="Clip on transcript context drift")
    parser.add_argument("--outlier-z-clip", type=float, default=3.25, help="Outlier clipping threshold")
    parser.add_argument("--outlier-std-inflation", type=float, default=1.5, help="Stddev inflation under outliers")
    parser.add_argument(
        "--interval-calibration",
        type=str,
        default="isotonic",
        choices=["none", "isotonic", "quantile_scale"],
        help="Predictive interval recalibration mode",
    )
    parser.add_argument(
        "--calibration-fallback",
        type=str,
        default="quantile_scale",
        choices=["none", "quantile_scale"],
        help="Calibration fallback mode",
    )
    parser.add_argument("--calibration-min-points", type=int, default=10, help="Min residual points for calibration")
    parser.add_argument("--interval-alpha", type=float, default=0.95, help="Target central coverage for intervals")
    parser.add_argument("--print-context", action="store_true", help="Print generated task context details")
    return parser


def _load_observations(args) -> list:
    if args.observations_csv is not None:
        return load_observations_csv(args.observations_csv)
    if args.observations_inline is not None:
        return parse_inline_observations(args.observations_inline)
    return parse_inline_observations("Jan=64, Feb=68.5")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    observations = ensure_bounds(_load_observations(args))

    transcript_text = ""
    if args.transcript_files:
        transcript_text = load_transcripts(args.transcript_files)
    context = generate_task_context(transcript_text) if transcript_text else default_task_context()

    project_root = Path(__file__).resolve().parents[2]
    llambo_path, llambo_commit = detect_llambo_repo(project_root)
    hardening = HardeningConfig(
        enabled=not args.disable_hardening,
        prior_weight=args.prior_weight,
        robust_likelihood=args.robust_likelihood,
        huber_delta=args.huber_delta,
        student_t_df=args.student_t_df,
        context_drift_clip=args.context_drift_clip,
        outlier_z_clip=args.outlier_z_clip,
        outlier_std_inflation=args.outlier_std_inflation,
        interval_calibration=args.interval_calibration,
        calibration_fallback=args.calibration_fallback,
        calibration_min_points=args.calibration_min_points,
        interval_alpha=args.interval_alpha,
    ).validated()

    points = run_forecast(
        observations=observations,
        context=context,
        months_ahead=max(1, args.months_ahead),
        horizon=parse_horizon(args.horizon),
        seed=args.seed,
        hardening=hardening,
    )
    write_forecast_csv(points, args.output_csv)
    plot_learning_curve(points, args.output_plot)

    if args.print_context:
        print(context.description)
    print(f"LLAMBO repo: {llambo_path} (commit={llambo_commit or 'unknown'})")
    print(f"Saved forecast CSV: {args.output_csv}")
    print(f"Saved forecast plot: {args.output_plot}")
    print("Month      Mean   StdDev   CI95")
    for item in points:
        ci = f"[{item.ci95_low:6.2f}, {item.ci95_high:6.2f}]"
        print(f"{item.month.strftime('%Y-%m')}  {item.posterior_mean:6.2f}  {item.posterior_stddev:6.2f}  {ci}")


if __name__ == "__main__":
    main()
