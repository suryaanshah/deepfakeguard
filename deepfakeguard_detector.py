from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DetectionReport:
    label: str
    confidence: float
    risk_score: float


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _weighted_risk(signals: Mapping[str, float], weights: Mapping[str, float]) -> float:
    weighted_total = 0.0
    weight_sum = 0.0

    for name, weight in weights.items():
        weighted_total += _clamp(signals.get(name, 0.0)) * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return weighted_total / weight_sum


def detect_deepfake_image(signals: Mapping[str, float]) -> DetectionReport:
    """Classify an image as REAL, SUSPICIOUS, or DEEPFAKE using CV signal scores.

    Expected signal keys and range [0, 1]:
    - face_warp_score
    - lighting_inconsistency_score
    - frequency_artifact_score
    - compression_mismatch_score
    """

    risk = _weighted_risk(
        signals,
        {
            "face_warp_score": 0.35,
            "lighting_inconsistency_score": 0.2,
            "frequency_artifact_score": 0.3,
            "compression_mismatch_score": 0.15,
        },
    )

    if risk >= 0.65:
        return DetectionReport("DEEPFAKE", confidence=risk, risk_score=risk)
    if risk >= 0.35:
        return DetectionReport("SUSPICIOUS", confidence=risk, risk_score=risk)
    return DetectionReport("REAL", confidence=1.0 - risk, risk_score=risk)


def detect_tampered_document(signals: Mapping[str, float]) -> DetectionReport:
    """Classify identity/government documents as ORIGINAL, SUSPICIOUS, or TAMPERED.

    Expected signal keys and range [0, 1]:
    - text_region_mismatch_score
    - font_inconsistency_score
    - edge_tamper_score
    - hologram_mismatch_score
    """

    risk = _weighted_risk(
        signals,
        {
            "text_region_mismatch_score": 0.35,
            "font_inconsistency_score": 0.25,
            "edge_tamper_score": 0.2,
            "hologram_mismatch_score": 0.2,
        },
    )

    if risk >= 0.6:
        return DetectionReport("TAMPERED", confidence=risk, risk_score=risk)
    if risk >= 0.3:
        return DetectionReport("SUSPICIOUS", confidence=risk, risk_score=risk)
    return DetectionReport("ORIGINAL", confidence=1.0 - risk, risk_score=risk)
