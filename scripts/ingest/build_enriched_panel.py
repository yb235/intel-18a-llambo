#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_QUARTERLY = REPO_ROOT / "data/raw/intel_quarterly_financial_signals.csv"
RAW_MILESTONES = REPO_ROOT / "data/raw/intel_18a_milestones.csv"
INTERIM_OUT = REPO_ROOT / "data/interim/enriched_monthly_features.csv"
PROCESSED_OUT = REPO_ROOT / "data/processed/enriched_monthly_panel.csv"


@dataclass(frozen=True)
class QuarterlyPoint:
    period: str
    period_start: date
    gross_margin_gaap_pct: float
    revenue_bil_usd: float


@dataclass(frozen=True)
class Milestone:
    event_date: date
    stage_value: int


def parse_ymd(value: str) -> date:
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def add_months(base: date, count: int) -> date:
    year = base.year + ((base.month - 1 + count) // 12)
    month = ((base.month - 1 + count) % 12) + 1
    return date(year, month, 1)


def month_iter(start: date, end: date) -> list[date]:
    out: list[date] = []
    cur = month_start(start.year, start.month)
    while cur <= end:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def quarter_of_month(value: date) -> tuple[int, int]:
    q = ((value.month - 1) // 3) + 1
    return value.year, q


def load_quarterly() -> dict[tuple[int, int], QuarterlyPoint]:
    out: dict[tuple[int, int], QuarterlyPoint] = {}
    with RAW_QUARTERLY.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = parse_ymd(row["period_start"])
            key = quarter_of_month(start)
            out[key] = QuarterlyPoint(
                period=row["period"],
                period_start=start,
                gross_margin_gaap_pct=float(row["gross_margin_gaap_pct"]),
                revenue_bil_usd=float(row["revenue_bil_usd"]),
            )
    return out


def quarter_point_for_month(value: date, quarterly: dict[tuple[int, int], QuarterlyPoint]) -> QuarterlyPoint | None:
    direct = quarterly.get(quarter_of_month(value))
    if direct is not None:
        return direct
    candidates = [item for item in quarterly.values() if item.period_start <= value]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.period_start)[-1]


def load_milestones() -> list[Milestone]:
    out: list[Milestone] = []
    with RAW_MILESTONES.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out.append(Milestone(event_date=parse_ymd(row["event_date"]), stage_value=int(row["stage_value"])))
    return sorted(out, key=lambda item: item.event_date)


def stage_as_of(month: date, milestones: list[Milestone]) -> int:
    value = 0
    for item in milestones:
        if item.event_date <= month:
            value = max(value, item.stage_value)
    return value


def zscore(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    std = var ** 0.5
    if std == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def effective_die_area_proxy_mm2(month_index: int, stage_norm: float) -> float:
    # Proxy only: assumes effective exposed die area decreases through ramp maturation.
    raw = 226.0 - 18.0 * stage_norm - 0.72 * month_index
    return clip(raw, 150.0, 230.0)


def main() -> None:
    quarterly = load_quarterly()
    milestones = load_milestones()

    start = month_start(2023, 7)
    end = month_start(2026, 2)
    months = month_iter(start, end)

    rows: list[dict[str, str | float | int]] = []
    for idx, month in enumerate(months):
        q = quarter_point_for_month(month, quarterly)
        if q is None:
            continue

        stage = stage_as_of(month, milestones)
        stage_norm = stage / 8.0
        effective_die_area_mm2_proxy = effective_die_area_proxy_mm2(idx, stage_norm)
        area_factor = effective_die_area_mm2_proxy / 180.0
        rows.append(
            {
                "month": month.strftime("%Y-%m"),
                "month_index": idx,
                "quarter": q.period,
                "gross_margin_gaap_pct": q.gross_margin_gaap_pct,
                "revenue_bil_usd": q.revenue_bil_usd,
                "milestone_stage": stage,
                "milestone_stage_norm": stage_norm,
                "effective_die_area_mm2_proxy": effective_die_area_mm2_proxy,
                "area_factor": area_factor,
            }
        )

    gm = [float(row["gross_margin_gaap_pct"]) for row in rows]
    rev = [float(row["revenue_bil_usd"]) for row in rows]
    gm_z = zscore(gm)
    rev_z = zscore(rev)

    raw_y: list[float] = []
    for i, row in enumerate(rows):
        idx = float(row["month_index"])
        stage_norm = float(row["milestone_stage_norm"])
        area_factor = float(row["area_factor"])
        # Explicitly a proxy equation; not measured yield.
        y = 31.0 + 0.62 * idx + 17.5 * stage_norm + 4.2 * gm_z[i] + 1.8 * rev_z[i]
        y -= 6.5 * (area_factor - 1.0)
        y = clip(y, 20.0, 92.0)
        row["yield_proxy_raw"] = y
        raw_y.append(y)

    anchor_values = {"2026-01": 64.0, "2026-02": 68.5}
    raw_a = next(float(row["yield_proxy_raw"]) for row in rows if row["month"] == "2026-01")
    offset = anchor_values["2026-01"] - raw_a

    for row in rows:
        month = str(row["month"])
        y_cal = clip(float(row["yield_proxy_raw"]) + offset, 0.0, 100.0)
        if month in anchor_values:
            y_final = anchor_values[month]
            source = "observed_anchor"
        else:
            y_final = y_cal
            source = "proxy_model"
        row["yield"] = y_final
        row["yield_source"] = source
        row["proxy_assumption_flag"] = 1 if source == "proxy_model" else 0

    INTERIM_OUT.parent.mkdir(parents=True, exist_ok=True)
    with INTERIM_OUT.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "month",
            "quarter",
            "month_index",
            "gross_margin_gaap_pct",
            "revenue_bil_usd",
            "milestone_stage",
            "milestone_stage_norm",
            "effective_die_area_mm2_proxy",
            "area_factor",
            "yield_proxy_raw",
            "yield",
            "yield_source",
            "proxy_assumption_flag",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    PROCESSED_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_OUT.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "month",
            "yield",
            "yield_source",
            "proxy_assumption_flag",
            "gross_margin_gaap_pct",
            "revenue_bil_usd",
            "milestone_stage",
            "milestone_stage_norm",
            "effective_die_area_mm2_proxy",
            "area_factor",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote {INTERIM_OUT}")
    print(f"Wrote {PROCESSED_OUT}")
    print(f"Rows: {len(rows)} ({rows[0]['month']} to {rows[-1]['month']})")


if __name__ == "__main__":
    main()
