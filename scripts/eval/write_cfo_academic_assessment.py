#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

CURRENT = Path("outputs/quality_enriched_cfo_academic/metrics_summary.csv")
PREVIOUS = Path("outputs/quality_enriched_rerun/metrics_summary.csv")
OUT = Path("outputs/quality_enriched_cfo_academic/critical_assessment_enriched_cfo_academic.md")

KEY = {
    "scenario": "prior_7_8_with_killer",
    "model": "llambo_style",
    "model_version": "hardened",
    "horizon": "all",
}

METRICS = ["mae", "rmse", "crps_approx", "coverage95", "calibration_error"]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get_row(rows: list[dict[str, str]], dataset: str) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("dataset") == dataset
            and row.get("scenario") == KEY["scenario"]
            and row.get("model") == KEY["model"]
            and row.get("model_version") == KEY["model_version"]
            and row.get("horizon") == KEY["horizon"]
        ):
            return row
    return None


def fval(row: dict[str, str] | None, metric: str) -> float:
    if row is None:
        return float("nan")
    raw = row.get(metric, "")
    return float(raw) if raw else float("nan")


def fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.4f}"


def fmtd(x: float) -> str:
    return "n/a" if x != x else f"{x:+.4f}"


def classify_reliability(obs_delta_rmse: float, obs_delta_cal: float, obs_delta_cov: float) -> str:
    improved = 0
    degraded = 0

    if obs_delta_rmse == obs_delta_rmse:
        if obs_delta_rmse < -0.05:
            improved += 1
        elif obs_delta_rmse > 0.05:
            degraded += 1

    if obs_delta_cal == obs_delta_cal:
        if obs_delta_cal < -0.01:
            improved += 1
        elif obs_delta_cal > 0.01:
            degraded += 1

    if obs_delta_cov == obs_delta_cov:
        if obs_delta_cov > 0.02:
            improved += 1
        elif obs_delta_cov < -0.02:
            degraded += 1

    if improved >= 2 and degraded == 0:
        return "Yes: CFO+academic enrichment improved observed-panel reliability on the configured thresholds."
    if degraded >= 2 and improved == 0:
        return "No: CFO+academic enrichment reduced observed-panel reliability on the configured thresholds."
    return "Mixed: CFO+academic enrichment changed reliability metrics, but not with a clean improvement signal."


def main() -> None:
    current_rows = load_rows(CURRENT)
    previous_rows = load_rows(PREVIOUS)

    obs_name = "enriched_monthly_panel"
    syn_name = "__ALL_SYNTHETIC__"

    curr_obs = get_row(current_rows, obs_name)
    prev_obs = get_row(previous_rows, obs_name)
    curr_syn = get_row(current_rows, syn_name)
    prev_syn = get_row(previous_rows, syn_name)

    obs_rmse_delta = fval(curr_obs, "rmse") - fval(prev_obs, "rmse")
    obs_cal_delta = fval(curr_obs, "calibration_error") - fval(prev_obs, "calibration_error")
    obs_cov_delta = fval(curr_obs, "coverage95") - fval(prev_obs, "coverage95")
    reliability_statement = classify_reliability(obs_rmse_delta, obs_cal_delta, obs_cov_delta)

    lines = [
        "# Critical Assessment: Enriched CFO + Academic Signals",
        "",
        "## Scope",
        "- Comparison key: `llambo_style:hardened`, scenario `prior_7_8_with_killer`, horizon `all`.",
        "- Delta baseline: `outputs/quality_enriched_rerun`.",
        "- CFO and academic/disclosed technical signals are directional priors and not direct yield telemetry.",
        "",
        "## Metric Deltas vs `quality_enriched_rerun`",
    ]

    for dataset, label, cur, prv in [
        (obs_name, "Observed panel (`enriched_monthly_panel`)", curr_obs, prev_obs),
        (syn_name, "Synthetic aggregate (`__ALL_SYNTHETIC__`)", curr_syn, prev_syn),
    ]:
        lines.append(f"- {label}:")
        for metric in METRICS:
            p = fval(prv, metric)
            c = fval(cur, metric)
            d = c - p
            lines.append(f"  - {metric}: {fmt(p)} -> {fmt(c)} (delta {fmtd(d)})")

    lines.extend(
        [
            "",
            "## Reliability Callout",
            f"- {reliability_statement}",
            "- Reliability check components: lower observed-panel RMSE, lower calibration error, and non-degraded/stronger 95% coverage.",
            "",
            "## Caveats",
            "- Most observed points are proxy-derived; only the Jan/Feb 2026 anchors are hard observations in this panel.",
            "- Subjective sources are down-weighted but still inject prior assumptions and should not be treated as factual measurements.",
            "- Small observed sample means metric shifts can be sensitive to a few windows.",
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
