from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import csv
import re


MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass(frozen=True)
class Observation:
    month: date
    yield_pct: float
    area_factor: float = 1.0
    cfo_gm_signal_strength: float = 0.0
    ifs_profitability_timeline_score: float = 0.0
    academic_yield_maturity_signal: float = 0.0
    disclosure_confidence: float = 0.0


def month_from_text(value: str, default_year: int = 2026) -> date:
    token = value.strip().lower()
    if re.fullmatch(r"\d{4}-\d{2}", token):
        year, month = token.split("-")
        return date(int(year), int(month), 1)

    short = token[:3]
    if short not in MONTH_MAP:
        raise ValueError(f"Unsupported month token: {value}")
    return date(default_year, MONTH_MAP[short], 1)


def load_observations_csv(path: Path) -> list[Observation]:
    obs: list[Observation] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            month = month_from_text(row["month"])
            value = float(row["yield"])
            area_factor = 1.0
            if row.get("area_factor", "").strip():
                area_factor = float(row["area_factor"])
            elif row.get("effective_die_area_mm2", "").strip():
                # Normalize die area to a nominal reference to form a unitless scale proxy.
                area_factor = float(row["effective_die_area_mm2"]) / 180.0
            elif row.get("effective_die_area_mm2_proxy", "").strip():
                area_factor = float(row["effective_die_area_mm2_proxy"]) / 180.0
            area_factor = min(1.6, max(0.6, area_factor))
            cfo_signal = float(row.get("cfo_gm_signal_strength", "0") or 0.0)
            ifs_signal = float(row.get("ifs_profitability_timeline_score", "0") or 0.0)
            academic_signal = float(row.get("academic_yield_maturity_signal", "0") or 0.0)
            disclosure_confidence = float(row.get("disclosure_confidence", "0") or 0.0)
            obs.append(
                Observation(
                    month=month,
                    yield_pct=value,
                    area_factor=area_factor,
                    cfo_gm_signal_strength=max(-1.0, min(1.0, cfo_signal)),
                    ifs_profitability_timeline_score=max(-1.0, min(1.0, ifs_signal)),
                    academic_yield_maturity_signal=max(-1.0, min(1.0, academic_signal)),
                    disclosure_confidence=max(0.0, min(1.0, disclosure_confidence)),
                )
            )
    return sorted(obs, key=lambda item: item.month)


def parse_inline_observations(inline: str, default_year: int = 2026) -> list[Observation]:
    pattern = re.compile(r"([A-Za-z]{3,9}|\d{4}-\d{2})\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    matches = pattern.findall(inline)
    if not matches:
        raise ValueError("Could not parse inline observations. Example: Jan=64, Feb=68.5")

    obs = [
        Observation(month=month_from_text(month_token, default_year), yield_pct=float(yield_token), area_factor=1.0)
        for month_token, yield_token in matches
    ]
    return sorted(obs, key=lambda item: item.month)


def load_transcripts(paths: list[Path]) -> str:
    snippets: list[str] = []
    for path in paths:
        snippets.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(snippets).strip()


def ensure_bounds(observations: list[Observation], lo: float = 0.0, hi: float = 100.0) -> list[Observation]:
    bounded: list[Observation] = []
    for item in observations:
        bounded.append(
            Observation(
                month=item.month,
                yield_pct=min(hi, max(lo, item.yield_pct)),
                area_factor=min(1.6, max(0.6, item.area_factor)),
                cfo_gm_signal_strength=max(-1.0, min(1.0, item.cfo_gm_signal_strength)),
                ifs_profitability_timeline_score=max(-1.0, min(1.0, item.ifs_profitability_timeline_score)),
                academic_yield_maturity_signal=max(-1.0, min(1.0, item.academic_yield_maturity_signal)),
                disclosure_confidence=max(0.0, min(1.0, item.disclosure_confidence)),
            )
        )
    return bounded
