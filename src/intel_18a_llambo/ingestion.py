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
            obs.append(Observation(month=month, yield_pct=value))
    return sorted(obs, key=lambda item: item.month)


def parse_inline_observations(inline: str, default_year: int = 2026) -> list[Observation]:
    pattern = re.compile(r"([A-Za-z]{3,9}|\d{4}-\d{2})\s*=\s*([0-9]+(?:\.[0-9]+)?)")
    matches = pattern.findall(inline)
    if not matches:
        raise ValueError("Could not parse inline observations. Example: Jan=64, Feb=68.5")

    obs = [
        Observation(month=month_from_text(month_token, default_year), yield_pct=float(yield_token))
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
        bounded.append(Observation(month=item.month, yield_pct=min(hi, max(lo, item.yield_pct))))
    return bounded
