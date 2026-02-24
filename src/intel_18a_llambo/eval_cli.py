from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from .evaluation import run_quality_evaluation
from .hardening import HardeningConfig


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Intel 18A LLAMBO quality evaluation harness")
    parser.add_argument(
        "--observations-csv",
        type=Path,
        default=Path("data/sample_observations.csv"),
        help="CSV file with month,yield columns",
    )
    parser.add_argument(
        "--transcript-files",
        type=Path,
        nargs="*",
        default=None,
        help="Optional transcript files (accepted for parity with main CLI)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quality"),
        help="Output directory for evaluation artifacts",
    )
    parser.add_argument("--max-horizon", type=int, default=6, help="Maximum monthly horizon for rolling backtests")
    parser.add_argument("--seed", type=int, default=18, help="Random seed")
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Disable synthetic stress-test datasets (not recommended for short real histories)",
    )
    parser.add_argument("--disable-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--disable-hardening", action="store_true", help="Disable hardened LLAMBO variant")
    parser.add_argument("--prior-weight", type=float, default=0.65, help="Weight on context prior vs data anchor (0-1)")
    parser.add_argument(
        "--robust-likelihood",
        type=str,
        default="huber",
        choices=["none", "huber", "student_t"],
        help="Robust heavy-tail approximation for outlier handling",
    )
    parser.add_argument("--huber-delta", type=float, default=1.75, help="Huber threshold in standardized residual units")
    parser.add_argument("--student-t-df", type=float, default=5.0, help="Student-t degrees of freedom")
    parser.add_argument("--context-drift-clip", type=float, default=0.02, help="Absolute cap on transcript-induced drift")
    parser.add_argument("--outlier-z-clip", type=float, default=3.25, help="Outlier clipping threshold (z-like units)")
    parser.add_argument("--outlier-std-inflation", type=float, default=1.5, help="Stddev inflation factor under outlier pressure")
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
        help="Fallback when isotonic calibration has insufficient points",
    )
    parser.add_argument("--calibration-min-points", type=int, default=10, help="Minimum in-sample residual points for calibrator")
    parser.add_argument("--interval-alpha", type=float, default=0.95, help="Target central coverage for intervals")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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

    outputs = run_quality_evaluation(
        observations_csv=args.observations_csv,
        transcript_files=args.transcript_files,
        output_dir=args.output_dir,
        max_horizon=args.max_horizon,
        seed=args.seed,
        include_synthetic=not args.no_synthetic,
        enable_plots=not args.disable_plots,
        hardening=hardening,
    )

    print("Quality evaluation completed.")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
