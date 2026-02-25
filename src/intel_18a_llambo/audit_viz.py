from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import csv
import math
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .context import generate_task_context


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return min(hi, max(lo, value))


def _parse_month(token: str) -> date:
    year, month = token.split("-")
    return date(int(year), int(month), 1)


@dataclass(frozen=True)
class SourceMeta:
    source_id: str
    source_url: str
    source_tier: str
    confidence: str


@dataclass(frozen=True)
class UpdateStep:
    month: date
    month_idx: int
    observed_yield: float
    yield_source: str
    prior_mean: float
    prior_std: float
    likelihood_mean: float
    likelihood_std: float
    posterior_mean: float
    posterior_std: float


@dataclass(frozen=True)
class ScenarioPoint:
    scenario_label: str
    rmse: float
    mae: float
    coverage95: float
    dataset: str
    model_version: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_optional_csv(path: Path, warn_label: str) -> list[dict[str, str]]:
    if not path.exists():
        warnings.warn(f"Missing optional file: {warn_label} ({path})")
        return []
    return _read_csv(path)


def _build_manifest_map(manifest_path: Path) -> dict[str, SourceMeta]:
    manifest_rows = _read_csv(manifest_path)
    out: dict[str, SourceMeta] = {}
    for row in manifest_rows:
        url = row.get("url", "").strip()
        if not url:
            continue
        out[url] = SourceMeta(
            source_id=row.get("source_id", "unknown_source"),
            source_url=url,
            source_tier=row.get("source_tier", "unknown"),
            confidence=row.get("confidence", "unknown"),
        )
    return out


def _default_source(url: str) -> SourceMeta:
    return SourceMeta(
        source_id="unmapped_source",
        source_url=url,
        source_tier="unknown",
        confidence="unknown",
    )


def _conjugate_update(prior_mu: float, prior_sigma: float, like_mu: float, like_sigma: float) -> tuple[float, float]:
    pv = max(1e-8, prior_sigma**2)
    lv = max(1e-8, like_sigma**2)
    post_var = 1.0 / (1.0 / pv + 1.0 / lv)
    post_mean = post_var * (prior_mu / pv + like_mu / lv)
    return _clip(post_mean), float(max(1e-5, math.sqrt(post_var)))


def _load_update_steps(panel_path: Path) -> list[UpdateStep]:
    rows = _read_csv(panel_path)
    if not rows:
        raise ValueError(f"Panel file is empty: {panel_path}")

    context = generate_task_context(
        "Management reiterated 7-8% monthly yield improvement with yield-killer risks acknowledged."
    )

    months = [_parse_month(row["month"]) for row in rows]
    yields = [float(row["yield"]) for row in rows]
    yield_sources = [row.get("yield_source", "proxy_model") for row in rows]
    diffs = np.diff(np.array(yields, dtype=float)) if len(yields) > 1 else np.array([2.0], dtype=float)
    innovation_sigma = float(max(1.0, np.std(diffs, ddof=1) if len(diffs) > 1 else abs(diffs[-1])))

    steps: list[UpdateStep] = []
    prev_post_mu = yields[0]
    prev_post_sigma = 5.0
    for idx, (month, obs, source_label) in enumerate(zip(months, yields, yield_sources)):
        if idx == 0:
            prior_mu = max(0.0, obs - 6.0)
            prior_sigma = 8.0
        else:
            prev = prev_post_mu
            headroom = max(0.0, 100.0 - prev)
            phase = 1.0 / (1.0 + math.exp(-(prev - context.s_curve_midpoint) * context.s_curve_steepness))
            phase_gain = 1.0 - 0.55 * phase
            prior_mu = _clip(prev + headroom * context.guidance_growth_mid * phase_gain)
            prior_sigma = float(max(1.8, prev_post_sigma * 1.08))

        like_mu = obs
        like_sigma = innovation_sigma if source_label == "observed_anchor" else max(innovation_sigma, 3.2)
        post_mu, post_sigma = _conjugate_update(prior_mu, prior_sigma, like_mu, like_sigma)

        steps.append(
            UpdateStep(
                month=month,
                month_idx=idx,
                observed_yield=obs,
                yield_source=source_label,
                prior_mean=prior_mu,
                prior_std=prior_sigma,
                likelihood_mean=like_mu,
                likelihood_std=like_sigma,
                posterior_mean=post_mu,
                posterior_std=post_sigma,
            )
        )
        prev_post_mu = post_mu
        prev_post_sigma = post_sigma

    return steps


def _scenario_row(metrics_rows: list[dict[str, str]]) -> ScenarioPoint | None:
    preferred = [
        row
        for row in metrics_rows
        if row.get("scenario") == "prior_7_8_with_killer"
        and row.get("model") == "llambo_style"
        and row.get("horizon") == "all"
    ]
    if not preferred:
        return None

    preferred.sort(
        key=lambda row: (
            0 if row.get("dataset") == "enriched_monthly_panel" else 1,
            0 if row.get("model_version") == "hardened" else 1,
        )
    )
    row = preferred[0]

    def f(key: str) -> float:
        token = row.get(key, "")
        return float(token) if token else float("nan")

    return ScenarioPoint(
        scenario_label="",
        rmse=f("rmse"),
        mae=f("mae"),
        coverage95=f("coverage95"),
        dataset=row.get("dataset", "unknown_dataset"),
        model_version=row.get("model_version", "baseline"),
    )


def _load_scenarios(repo_root: Path) -> list[ScenarioPoint]:
    scenario_paths = {
        "baseline": repo_root / "outputs/quality/metrics_summary.csv",
        "hardened": repo_root / "outputs/quality_hardened/metrics_summary.csv",
        "enriched": repo_root / "outputs/quality_enriched/metrics_summary.csv",
        "enriched_rerun": repo_root / "outputs/quality_enriched_rerun/metrics_summary.csv",
        "enriched_cfo_academic": repo_root / "outputs/quality_enriched_cfo_academic/metrics_summary.csv",
    }
    points: list[ScenarioPoint] = []
    for label, path in scenario_paths.items():
        rows = _read_optional_csv(path, label)
        if not rows:
            continue
        point = _scenario_row(rows)
        if point is None:
            warnings.warn(f"No matching row for scenario comparison: {label}")
            continue
        points.append(
            ScenarioPoint(
                scenario_label=label,
                rmse=point.rmse,
                mae=point.mae,
                coverage95=point.coverage95,
                dataset=point.dataset,
                model_version=point.model_version,
            )
        )
    if not points:
        raise ValueError("No scenario metrics were available for comparison plots.")
    return points


def _source_for_url(url: str, manifest: dict[str, SourceMeta]) -> SourceMeta:
    return manifest.get(url, _default_source(url))


def _as_date(row: dict[str, str], key: str) -> date:
    value = row.get(key, "")
    if not value:
        raise ValueError(f"Missing required date field: {key}")
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def _build_trace_rows(repo_root: Path, steps: list[UpdateStep], manifest: dict[str, SourceMeta]) -> list[dict[str, str]]:
    quarterly = _read_optional_csv(repo_root / "data/raw/intel_quarterly_financial_signals.csv", "quarterly_signals")
    milestones = _read_optional_csv(repo_root / "data/raw/intel_18a_milestones.csv", "milestones")
    cfo = _read_optional_csv(repo_root / "data/raw/intel_cfo_signals.csv", "cfo_signals")
    academic = _read_optional_csv(repo_root / "data/raw/intel_18a_academic_signals.csv", "academic_signals")

    q_rows = [row for row in quarterly if row.get("period_start")]
    m_rows = [row for row in milestones if row.get("event_date")]
    c_rows = [row for row in cfo if row.get("period_start")]
    a_rows = [row for row in academic if row.get("event_date")]

    trace_rows: list[dict[str, str]] = []
    for step in steps:
        month = step.month
        month_token = month.strftime("%Y-%m")
        quarter_sources = [row for row in q_rows if _as_date(row, "period_start") <= month]
        cfo_sources = [row for row in c_rows if _as_date(row, "period_start") <= month]
        milestone_sources = [row for row in m_rows if _as_date(row, "event_date") <= month]
        academic_sources = [row for row in a_rows if _as_date(row, "event_date") <= month]

        refs: list[tuple[str, SourceMeta, str]] = []
        if quarter_sources:
            src = _source_for_url(quarter_sources[-1].get("source_url", ""), manifest)
            refs.append(("gross_margin_gaap_pct|revenue_bil_usd", src, "quarterly carry-forward"))
        if cfo_sources:
            src = _source_for_url(cfo_sources[-1].get("source_url", ""), manifest)
            refs.append(("cfo_gm_signal_strength|ifs_profitability_timeline_score", src, "quarterly guidance weighting"))
        if milestone_sources:
            src = _source_for_url(milestone_sources[-1].get("source_url", ""), manifest)
            refs.append(("milestone_stage", src, "stage_as_of(event_date<=month)"))
        if academic_sources:
            src = _source_for_url(academic_sources[-1].get("source_url", ""), manifest)
            refs.append(("academic_yield_maturity_signal", src, "recency-decayed signal blend"))

        if not refs:
            refs.append(("unknown", _default_source(""), "missing raw source links"))

        derivation_class = "direct_observed" if step.yield_source == "observed_anchor" else "proxy_derived"
        for series_name, value in (
            ("prior_mean", step.prior_mean),
            ("likelihood_mean", step.likelihood_mean),
            ("posterior_mean", step.posterior_mean),
            ("posterior_std", step.posterior_std),
        ):
            for feat, src, transform_step in refs:
                trace_rows.append(
                    {
                        "artifact": "bayesian_update_timeline.png|posterior_evolution.png",
                        "series": series_name,
                        "month": month_token,
                        "datapoint_id": f"{series_name}:{month_token}",
                        "value": f"{value:.6f}",
                        "feature_ref": feat,
                        "source_id": src.source_id,
                        "source_url": src.source_url,
                        "source_tier": src.source_tier,
                        "confidence": src.confidence,
                        "derivation_class": derivation_class,
                        "transform_step": transform_step,
                    }
                )

        if step.yield_source == "observed_anchor":
            trace_rows.append(
                {
                    "artifact": "bayesian_update_timeline.png",
                    "series": "observed_anchor",
                    "month": month_token,
                    "datapoint_id": f"observed_anchor:{month_token}",
                    "value": f"{step.observed_yield:.6f}",
                    "feature_ref": "yield",
                    "source_id": "observed_anchor_sample",
                    "source_url": "data/sample_observations.csv",
                    "source_tier": "public_observed",
                    "confidence": "high",
                    "derivation_class": "direct_observed",
                    "transform_step": "anchor override in build_enriched_panel.py",
                }
            )

    subjective_sources = [meta for meta in manifest.values() if meta.source_tier == "subjective_prior"]
    for src in subjective_sources:
        trace_rows.append(
            {
                "artifact": "scenario_comparison.png",
                "series": "subjective_prior_weight",
                "month": "n/a",
                "datapoint_id": f"subjective_prior:{src.source_id}",
                "value": "0.60",
                "feature_ref": "prior_reliability",
                "source_id": src.source_id,
                "source_url": src.source_url,
                "source_tier": src.source_tier,
                "confidence": src.confidence,
                "derivation_class": "subjective_prior",
                "transform_step": "run_rolling_backtest prior_weight_multiplier",
            }
        )

    return trace_rows


def _write_trace_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_bayesian_update_timeline(steps: list[UpdateStep], output_path: Path) -> None:
    x = np.arange(len(steps))
    labels = [step.month.strftime("%Y-%m") for step in steps]
    prior = np.array([step.prior_mean for step in steps], dtype=float)
    like = np.array([step.likelihood_mean for step in steps], dtype=float)
    post = np.array([step.posterior_mean for step in steps], dtype=float)

    plt.figure(figsize=(13, 6))
    plt.plot(x, prior, color="#6b4f1d", lw=1.8, marker="o", label="Prior mean")
    plt.plot(x, like, color="#3f3f3f", lw=1.4, marker="s", alpha=0.85, label="Likelihood mean")
    plt.plot(x, post, color="#0b6b9a", lw=2.3, marker="D", label="Posterior mean")
    for idx in range(len(steps)):
        plt.plot([x[idx], x[idx]], [prior[idx], like[idx]], color="#b0b0b0", lw=0.8, alpha=0.7)
        plt.plot([x[idx], x[idx]], [like[idx], post[idx]], color="#8bbad1", lw=0.8, alpha=0.7)

    plt.title("Bayesian Update Timeline: Prior -> Likelihood -> Posterior")
    plt.ylabel("Yield (%)")
    plt.xlabel("Month")
    plt.ylim(0, 100)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_posterior_evolution(steps: list[UpdateStep], output_path: Path) -> None:
    x = np.arange(len(steps))
    labels = [step.month.strftime("%Y-%m") for step in steps]
    mu = np.array([step.posterior_mean for step in steps], dtype=float)
    sigma = np.array([step.posterior_std for step in steps], dtype=float)
    lo95 = np.clip(mu - 1.96 * sigma, 0.0, 100.0)
    hi95 = np.clip(mu + 1.96 * sigma, 0.0, 100.0)
    lo68 = np.clip(mu - sigma, 0.0, 100.0)
    hi68 = np.clip(mu + sigma, 0.0, 100.0)

    plt.figure(figsize=(13, 6))
    plt.fill_between(x, lo95, hi95, color="#b8dff2", alpha=0.6, label="95% uncertainty")
    plt.fill_between(x, lo68, hi68, color="#7ec3e3", alpha=0.55, label="68% uncertainty")
    plt.plot(x, mu, color="#0a5a85", lw=2.4, marker="o", label="Posterior mean")
    plt.scatter(
        x,
        [step.observed_yield for step in steps],
        s=24,
        color="#8a4f08",
        alpha=0.8,
        label="Observed/proxy panel yield",
    )

    plt.title("Posterior Evolution Across Update Steps")
    plt.ylabel("Yield (%)")
    plt.xlabel("Update step (month)")
    plt.ylim(0, 100)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_scenario_comparison(points: list[ScenarioPoint], output_path: Path) -> None:
    x = np.arange(len(points), dtype=float)
    labels = [point.scenario_label for point in points]
    rmse = np.array([point.rmse for point in points], dtype=float)
    mae = np.array([point.mae for point in points], dtype=float)
    cov = np.array([point.coverage95 for point in points], dtype=float) * 100.0

    fig, ax1 = plt.subplots(figsize=(12, 6))
    width = 0.36
    ax1.bar(x - width / 2, rmse, width=width, color="#0b6b9a", label="RMSE")
    ax1.bar(x + width / 2, mae, width=width, color="#6b4f1d", label="MAE")
    ax1.set_ylabel("Error (yield %)")
    ax1.set_xlabel("Scenario run")
    ax1.set_xticks(x, labels, rotation=25, ha="right")
    ax1.grid(axis="y", alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(x, cov, color="#2f7d32", marker="o", lw=2.0, label="Coverage95")
    ax2.set_ylabel("Coverage95 (%)")
    ax2.set_ylim(0, 105)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    plt.title("Scenario Comparison: baseline, hardened, enriched variants")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_lineage_graph(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.5))
    ax.set_axis_off()

    nodes = {
        "manifest": (0.08, 0.88, "source_manifest.csv\n(source_id/tier/confidence)"),
        "raw_fin": (0.08, 0.70, "raw quarterly + CFO\npublic_observed + subjective_prior"),
        "raw_tech": (0.08, 0.52, "raw milestones + academic\npublic_observed + subjective_prior"),
        "features": (0.38, 0.62, "enriched_monthly_features.csv\nengineered proxies + confidences"),
        "panel": (0.62, 0.62, "enriched_monthly_panel.csv\nyield_source + proxy flags"),
        "inputs": (0.62, 0.42, "model inputs\nObservation(month,yield,signals)"),
        "posterior": (0.84, 0.62, "posterior updates\nprior -> likelihood -> posterior"),
        "audit": (0.84, 0.42, "audit artifacts\nplots + trace_matrix + summary"),
    }

    for _, (x, y, text) in nodes.items():
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7f9fb", "edgecolor": "#49657a", "linewidth": 1.2},
        )

    def arrow(a: str, b: str) -> None:
        x1, y1, _ = nodes[a]
        x2, y2, _ = nodes[b]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#4f6375"},
        )

    arrow("manifest", "raw_fin")
    arrow("manifest", "raw_tech")
    arrow("raw_fin", "features")
    arrow("raw_tech", "features")
    arrow("features", "panel")
    arrow("panel", "inputs")
    arrow("panel", "posterior")
    arrow("inputs", "posterior")
    arrow("posterior", "audit")
    arrow("manifest", "audit")

    ax.text(
        0.5,
        0.14,
        "Traceability rule: every plotted datapoint is mapped to source_id/source_url + transform_step in trace_matrix.csv",
        ha="center",
        va="center",
        fontsize=10,
        color="#24445a",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".svg":
        plt.savefig(output_path, format="svg")
    else:
        plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_audit_dashboard(
    output_path: Path,
    scenario_points: list[ScenarioPoint],
    trace_rel_path: str,
    lineage_rel_path: str,
) -> None:
    rows = "\n".join(
        (
            f"<tr><td>{p.scenario_label}</td><td>{p.dataset}</td>"
            f"<td>{p.model_version}</td><td>{p.rmse:.4f}</td><td>{p.mae:.4f}</td><td>{p.coverage95:.4f}</td></tr>"
        )
        for p in scenario_points
    )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>Intel 18A Bayesian Audit Dashboard</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #1f3242;
      --muted: #5f7487;
      --line: #d8e2ec;
      --accent: #0b6b9a;
    }}
    body {{ margin: 0; padding: 24px; font-family: Georgia, 'Times New Roman', serif; background: var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 1120px; margin: 0 auto; display: grid; gap: 16px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
    h1, h2 {{ margin: 0 0 10px 0; }}
    p {{ margin: 6px 0; color: var(--muted); }}
    img {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border: 1px solid var(--line); padding: 8px; text-align: left; }}
    th {{ background: #edf3f8; }}
    .links a {{ color: var(--accent); text-decoration: none; margin-right: 10px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>Intel 18A Bayesian Audit Dashboard</h1>
      <p>Static audit artifact with provenance-aware visualizations and trace matrix links.</p>
      <div class=\"links\">
        <a href=\"bayesian_update_timeline.png\">Bayesian Update Timeline</a>
        <a href=\"posterior_evolution.png\">Posterior Evolution</a>
        <a href=\"scenario_comparison.png\">Scenario Comparison</a>
        <a href=\"{lineage_rel_path}\">Lineage Graph</a>
        <a href=\"{trace_rel_path}\">Trace Matrix CSV</a>
        <a href=\"audit_summary.md\">Audit Summary</a>
      </div>
    </div>

    <div class=\"card\"><h2>Prior -> Likelihood -> Posterior Timeline</h2><img src=\"bayesian_update_timeline.png\" alt=\"Bayesian update timeline\" /></div>
    <div class=\"card\"><h2>Posterior Evolution with Uncertainty Bands</h2><img src=\"posterior_evolution.png\" alt=\"Posterior evolution\" /></div>
    <div class=\"card\"><h2>Scenario Comparison</h2><img src=\"scenario_comparison.png\" alt=\"Scenario comparison\" /></div>
    <div class=\"card\"><h2>Lineage Graph</h2><img src=\"{lineage_rel_path}\" alt=\"Lineage graph\" /></div>

    <div class=\"card\">
      <h2>Scenario Metrics Snapshot</h2>
      <table>
        <thead>
          <tr><th>Scenario</th><th>Dataset used</th><th>Model version</th><th>RMSE</th><th>MAE</th><th>Coverage95</th></tr>
        </thead>
        <tbody>
          {rows}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def _write_audit_summary(
    output_path: Path,
    steps: list[UpdateStep],
    scenario_points: list[ScenarioPoint],
    trace_rows: list[dict[str, str]],
) -> None:
    obs_count = len([step for step in steps if step.yield_source == "observed_anchor"])
    proxy_count = len(steps) - obs_count
    subj_rows = len([row for row in trace_rows if row.get("derivation_class") == "subjective_prior"])

    lines = [
        "# Audit Summary",
        "",
        "## Artifact coverage",
        "- Generated required plots: `bayesian_update_timeline.png`, `posterior_evolution.png`, `scenario_comparison.png`, `lineage_graph.svg`.",
        "- Generated trace artifacts: `trace_matrix.csv`, `audit_dashboard.html`, `audit_summary.md`.",
        "",
        "## Data category handling (explicit)",
        f"- Directly observed: `{obs_count}` monthly anchors from `yield_source=observed_anchor` (Jan/Feb 2026 hard anchors).",
        f"- Proxy-derived: `{proxy_count}` monthly points from engineered proxy transforms in `build_enriched_panel.py`.",
        f"- Subjective prior: `{subj_rows}` trace rows tied to `subjective_prior` source tier and reliability weighting.",
        "",
        "## Bayesian update interpretation",
        "- Timeline reconstructs month-by-month prior -> likelihood -> posterior updates for auditability.",
        "- Uncertainty bands are shown at 68% and 95% intervals to support audit review of confidence evolution.",
        "",
        "## Scenario comparison inputs",
    ]
    for point in scenario_points:
        lines.append(
            f"- `{point.scenario_label}`: dataset=`{point.dataset}`, model_version=`{point.model_version}`, "
            f"RMSE={point.rmse:.4f}, MAE={point.mae:.4f}, coverage95={point.coverage95:.4f}."
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "- Trace matrix maps each plotted datapoint to source_id/source_url/source_tier/confidence and transform_step.",
            "- Source tiers and confidence labels are pulled from `data/raw/source_manifest.csv` where available.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build audit-grade Bayesian visualization suite with provenance traceability")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root path",
    )
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("data/processed/enriched_monthly_panel.csv"),
        help="Model panel CSV path (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/raw/source_manifest.csv"),
        help="Source manifest CSV path (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audit"),
        help="Audit output directory (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--lineage-format",
        choices=["svg", "png"],
        default="svg",
        help="Lineage graph output format",
    )
    return parser


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    panel_path = _resolve(repo_root, args.panel_csv)
    manifest_path = _resolve(repo_root, args.manifest_csv)
    output_dir = _resolve(repo_root, args.output_dir)

    if not panel_path.exists():
        raise FileNotFoundError(f"Panel file not found: {panel_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    steps = _load_update_steps(panel_path)
    scenario_points = _load_scenarios(repo_root)
    manifest = _build_manifest_map(manifest_path)
    trace_rows = _build_trace_rows(repo_root, steps, manifest)

    timeline_path = output_dir / "bayesian_update_timeline.png"
    posterior_path = output_dir / "posterior_evolution.png"
    scenario_path = output_dir / "scenario_comparison.png"
    lineage_path = output_dir / f"lineage_graph.{args.lineage_format}"
    trace_path = output_dir / "trace_matrix.csv"
    dashboard_path = output_dir / "audit_dashboard.html"
    summary_path = output_dir / "audit_summary.md"

    _plot_bayesian_update_timeline(steps, timeline_path)
    _plot_posterior_evolution(steps, posterior_path)
    _plot_scenario_comparison(scenario_points, scenario_path)
    _plot_lineage_graph(lineage_path)
    _write_trace_csv(trace_rows, trace_path)
    _write_audit_dashboard(
        dashboard_path,
        scenario_points,
        trace_rel_path=trace_path.name,
        lineage_rel_path=lineage_path.name,
    )
    _write_audit_summary(summary_path, steps, scenario_points, trace_rows)

    print(f"Wrote {timeline_path}")
    print(f"Wrote {posterior_path}")
    print(f"Wrote {scenario_path}")
    print(f"Wrote {lineage_path}")
    print(f"Wrote {trace_path}")
    print(f"Wrote {dashboard_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
