#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_QUARTERLY = REPO_ROOT / "data/raw/intel_quarterly_financial_signals.csv"
RAW_MILESTONES = REPO_ROOT / "data/raw/intel_18a_milestones.csv"
RAW_ACADEMIC = REPO_ROOT / "data/raw/intel_18a_academic_signals.csv"
RAW_CFO = REPO_ROOT / "data/raw/intel_cfo_signals.csv"
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


@dataclass(frozen=True)
class AcademicSignal:
    event_date: date
    signal_value: float
    disclosure_confidence: float
    source_tier: str
    confidence_label: str


@dataclass(frozen=True)
class CfoSignal:
    period_start: date
    cfo_gm_signal_strength: float
    ifs_profitability_timeline_score: float
    margin_pressure_recovery_signal: float
    disclosure_confidence: float
    source_tier: str
    confidence_label: str


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


def load_academic_signals() -> list[AcademicSignal]:
    out: list[AcademicSignal] = []
    with RAW_ACADEMIC.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out.append(
                AcademicSignal(
                    event_date=parse_ymd(row["event_date"]),
                    signal_value=float(row["academic_yield_maturity_signal"]),
                    disclosure_confidence=float(row["disclosure_confidence"]),
                    source_tier=row.get("source_tier", "public_observed"),
                    confidence_label=row.get("confidence", "medium"),
                )
            )
    return sorted(out, key=lambda item: item.event_date)


def load_cfo_signals() -> dict[tuple[int, int], CfoSignal]:
    out: dict[tuple[int, int], CfoSignal] = {}
    with RAW_CFO.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            start = parse_ymd(row["period_start"])
            key = quarter_of_month(start)
            out[key] = CfoSignal(
                period_start=start,
                cfo_gm_signal_strength=float(row["cfo_gm_signal_strength"]),
                ifs_profitability_timeline_score=float(row["ifs_profitability_timeline_score"]),
                margin_pressure_recovery_signal=float(row["margin_pressure_recovery_signal"]),
                disclosure_confidence=float(row["disclosure_confidence"]),
                source_tier=row.get("source_tier", "public_observed"),
                confidence_label=row.get("confidence", "medium"),
            )
    return out


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


def tier_weight(source_tier: str) -> float:
    token = (source_tier or "").strip().lower()
    if token == "public_observed":
        return 1.0
    if token == "subjective_prior":
        return 0.35
    if token == "tooling_attempt":
        return 0.1
    return 0.6


def confidence_weight(confidence: str) -> float:
    token = (confidence or "").strip().lower()
    if token == "high":
        return 1.0
    if token == "medium":
        return 0.8
    if token == "low":
        return 0.6
    return 0.75


def effective_die_area_proxy_mm2(month_index: int, stage_norm: float) -> float:
    # Proxy only: assumes effective exposed die area decreases through ramp maturation.
    raw = 226.0 - 18.0 * stage_norm - 0.72 * month_index
    return clip(raw, 150.0, 230.0)


def academic_signal_as_of(month: date, events: list[AcademicSignal]) -> tuple[float, float]:
    candidates = [item for item in events if item.event_date <= month]
    if not candidates:
        return 0.0, 0.0

    weighted_signal = 0.0
    weighted_conf = 0.0
    total_w = 0.0
    for item in candidates[-4:]:
        months_since = max(0, (month.year - item.event_date.year) * 12 + (month.month - item.event_date.month))
        decay = 0.86 ** months_since
        w = decay * tier_weight(item.source_tier) * confidence_weight(item.confidence_label)
        weighted_signal += item.signal_value * w
        weighted_conf += item.disclosure_confidence * w
        total_w += w

    if total_w <= 1e-9:
        return 0.0, 0.0
    return weighted_signal / total_w, clip(weighted_conf / total_w, 0.0, 1.0)


def cfo_signal_for_month(month: date, quarterly: dict[tuple[int, int], CfoSignal]) -> tuple[float, float, float, float]:
    direct = quarterly.get(quarter_of_month(month))
    point = direct
    if point is None:
        candidates = [item for item in quarterly.values() if item.period_start <= month]
        point = sorted(candidates, key=lambda item: item.period_start)[-1] if candidates else None
    if point is None:
        return 0.0, 0.0, 0.0, 0.0

    w = tier_weight(point.source_tier) * confidence_weight(point.confidence_label)
    cfo = point.cfo_gm_signal_strength * w
    ifs = point.ifs_profitability_timeline_score * w
    margin = point.margin_pressure_recovery_signal * w
    conf = clip(point.disclosure_confidence * w, 0.0, 1.0)
    return cfo, ifs, margin, conf


def main() -> None:
    quarterly = load_quarterly()
    milestones = load_milestones()
    academic_events = load_academic_signals()
    cfo_signals = load_cfo_signals()

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
        cfo_signal, ifs_signal, margin_signal, cfo_confidence = cfo_signal_for_month(month, cfo_signals)
        academic_signal, academic_confidence = academic_signal_as_of(month, academic_events)
        disclosure_confidence = clip(0.55 * cfo_confidence + 0.45 * academic_confidence, 0.0, 1.0)
        prior_blend = clip(0.45 * cfo_signal + 0.30 * ifs_signal + 0.25 * academic_signal, -1.0, 1.0)
        effective_die_area_mm2_proxy = effective_die_area_proxy_mm2(idx, stage_norm)
        area_factor = clip(effective_die_area_mm2_proxy / 180.0 - 0.08 * prior_blend, 0.6, 1.6)
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
                "cfo_gm_signal_strength": cfo_signal,
                "ifs_profitability_timeline_score": ifs_signal,
                "margin_pressure_recovery_signal": margin_signal,
                "academic_yield_maturity_signal": academic_signal,
                "disclosure_confidence": disclosure_confidence,
            }
        )

    gm = [float(row["gross_margin_gaap_pct"]) for row in rows]
    rev = [float(row["revenue_bil_usd"]) for row in rows]
    gm_z = zscore(gm)
    rev_z = zscore(rev)

    for i, row in enumerate(rows):
        idx = float(row["month_index"])
        stage_norm = float(row["milestone_stage_norm"])
        area_factor = float(row["area_factor"])
        cfo_signal = float(row["cfo_gm_signal_strength"])
        ifs_signal = float(row["ifs_profitability_timeline_score"])
        academic_signal = float(row["academic_yield_maturity_signal"])
        disclosure = float(row["disclosure_confidence"])
        # Explicitly a proxy equation; not measured yield.
        y = 31.0 + 0.62 * idx + 17.5 * stage_norm + 4.2 * gm_z[i] + 1.8 * rev_z[i]
        y += 2.6 * cfo_signal + 2.9 * ifs_signal + 3.4 * academic_signal
        y += 1.2 * (disclosure - 0.5)
        y -= 6.5 * (area_factor - 1.0)
        y = clip(y, 20.0, 92.0)
        row["yield_proxy_raw"] = y

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
            "cfo_gm_signal_strength",
            "ifs_profitability_timeline_score",
            "margin_pressure_recovery_signal",
            "academic_yield_maturity_signal",
            "disclosure_confidence",
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
            "cfo_gm_signal_strength",
            "ifs_profitability_timeline_score",
            "margin_pressure_recovery_signal",
            "academic_yield_maturity_signal",
            "disclosure_confidence",
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
