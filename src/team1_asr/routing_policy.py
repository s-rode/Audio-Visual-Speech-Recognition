from __future__ import annotations

from dataclasses import dataclass
import math

from team1_asr.visual_reliability import is_real_mouth_roi, optional_float


@dataclass(frozen=True)
class PolicyDecision:
    selects_avec: bool
    score: float | None
    threshold: float | None


def evaluate_policy(
    policy: dict[str, object],
    features: dict[str, object],
) -> PolicyDecision:
    validate_live_policy(policy)
    policy_type = str(policy.get("policy_type") or "")
    score = policy_score(policy, features)
    threshold = optional_float(policy.get("threshold"))
    if not is_real_mouth_roi(features) or policy_type == "audio_only":
        selects_avec = False
    elif policy_type == "roi_success":
        selects_avec = True
    elif score is None or threshold is None:
        selects_avec = False
    elif policy_type == "threshold" and str(policy.get("direction") or "higher") == "lower":
        selects_avec = score <= threshold
    else:
        selects_avec = score >= threshold
    return PolicyDecision(selects_avec, score, threshold)


def policy_score(
    policy: dict[str, object],
    features: dict[str, object],
) -> float | None:
    policy_type = str(policy.get("policy_type") or "")
    if policy_type == "learned_logistic":
        return _learned_probability(policy, features)
    feature = str(policy.get("feature") or "")
    return optional_float(features.get(feature)) if feature else None


def validate_live_policy(policy: dict[str, object]) -> None:
    audio_system = str(policy.get("audio_system") or "audio").strip()
    av_system = str(policy.get("av_system") or "audio_visual").strip()
    if not audio_system or not av_system or audio_system == av_system:
        raise ValueError("Live policy requires different audio and audio-visual system names.")

    policy_type = str(policy.get("policy_type") or "")
    if policy_type in {"audio_only", "roi_success"}:
        return
    if policy_type == "threshold":
        if not str(policy.get("feature") or ""):
            raise ValueError("Threshold policy requires a feature.")
        if optional_float(policy.get("threshold")) is None:
            raise ValueError("Threshold policy requires a numeric threshold.")
        direction = str(policy.get("direction") or "higher")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"Unsupported threshold direction: {direction}")
        return
    if policy_type == "learned_logistic":
        if optional_float(policy.get("threshold")) is None:
            raise ValueError("Learned policy requires a probability threshold.")
        if _learned_probability(policy, {}) is None:
            raise ValueError("Learned policy has invalid model parameters.")
        return
    raise ValueError(f"Policy type is not deployable in the live demo: {policy_type}")


def _learned_probability(
    policy: dict[str, object],
    features: dict[str, object],
) -> float | None:
    feature_names = policy.get("feature_names")
    medians = policy.get("feature_medians")
    means = policy.get("feature_means")
    scales = policy.get("feature_scales")
    coefficients = policy.get("coefficients")
    intercept = optional_float(policy.get("intercept"))
    if (
        not isinstance(feature_names, list)
        or not isinstance(medians, dict)
        or not isinstance(means, dict)
        or not isinstance(scales, dict)
        or not isinstance(coefficients, dict)
        or intercept is None
    ):
        return None

    logit = intercept
    for feature in feature_names:
        if not isinstance(feature, str):
            return None
        median = optional_float(medians.get(feature))
        mean = optional_float(means.get(feature))
        scale = optional_float(scales.get(feature))
        coefficient = optional_float(coefficients.get(feature))
        if (
            median is None
            or mean is None
            or scale is None
            or scale <= 0
            or coefficient is None
        ):
            return None
        value = optional_float(features.get(feature))
        logit += coefficient * ((median if value is None else value) - mean) / scale

    logit = max(-30.0, min(30.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))
