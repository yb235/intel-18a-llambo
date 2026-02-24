from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TaskContext:
    guidance_growth_low: float
    guidance_growth_high: float
    guidance_growth_mid: float
    transcript_confidence: float
    transcript_risk: float
    ribbonfet_mentioned: bool
    powervia_mentioned: bool
    s_curve_midpoint: float
    s_curve_steepness: float
    description: str


def _extract_guidance_range(text: str) -> tuple[float, float]:
    # Supports patterns like "7-8% monthly" or "7 to 8%".
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if range_match:
        lo = float(range_match.group(1)) / 100.0
        hi = float(range_match.group(2)) / 100.0
        return min(lo, hi), max(lo, hi)

    single_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:monthly|mo)", text, re.IGNORECASE)
    if single_match:
        value = float(single_match.group(1)) / 100.0
        return max(0.0, value - 0.005), min(0.20, value + 0.005)

    return 0.07, 0.08


def generate_task_context(transcript_text: str) -> TaskContext:
    lower = transcript_text.lower()
    ribbonfet_mentioned = "ribbonfet" in lower
    powervia_mentioned = "powervia" in lower

    positive_terms = ["confidence", "improved", "progress", "ahead", "reduction", "stable"]
    risk_terms = ["risk", "variability", "delay", "challenge", "uncertain", "headwind"]

    confidence_score = sum(lower.count(term) for term in positive_terms)
    risk_score = sum(lower.count(term) for term in risk_terms)

    guidance_low, guidance_high = _extract_guidance_range(transcript_text)
    guidance_mid = 0.5 * (guidance_low + guidance_high)

    s_curve_midpoint = 78.0
    if ribbonfet_mentioned and powervia_mentioned:
        s_curve_midpoint = 80.0

    s_curve_steepness = 0.14 + min(0.03, 0.003 * confidence_score)

    description = (
        "Intel 18A yield task context: RibbonFET="
        f"{ribbonfet_mentioned}, PowerVia={powervia_mentioned}, guidance={guidance_low:.1%}-{guidance_high:.1%} monthly, "
        f"S-curve midpoint={s_curve_midpoint:.1f}, steepness={s_curve_steepness:.3f}."
    )

    return TaskContext(
        guidance_growth_low=guidance_low,
        guidance_growth_high=guidance_high,
        guidance_growth_mid=guidance_mid,
        transcript_confidence=float(confidence_score),
        transcript_risk=float(risk_score),
        ribbonfet_mentioned=ribbonfet_mentioned,
        powervia_mentioned=powervia_mentioned,
        s_curve_midpoint=s_curve_midpoint,
        s_curve_steepness=s_curve_steepness,
        description=description,
    )


def default_task_context() -> TaskContext:
    return generate_task_context(
        "RibbonFET and PowerVia discussed with 7-8% monthly target and moderated confidence."
    )
