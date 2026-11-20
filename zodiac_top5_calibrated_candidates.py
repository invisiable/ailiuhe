"""Fixed, untuned TOP5 calibration experts for chronological model selection.

The24 candidate definitions are fixed. Each expert starts its own feedback
state at the first available historical draw and predicts before observing
the target draw. This module performs no parameter selection or file IO.
"""

from __future__ import annotations

import numpy as np

from zodiac_simple_smart import ZodiacSimpleSmart


BASELINE_ID = "raw_f1.0_c1.0_a1.0"
SCENARIO_WEIGHTS = {
    "normal": {
        "ultra_cold": 0.35, "anti_hot": 0.20, "gap": 0.18,
        "rotation": 0.12, "absence_penalty": 0.08,
        "diversity": 0.04, "similarity": 0.03,
    },
    "extreme_hot": {
        "ultra_cold": 0.20, "anti_hot": 0.08, "gap": 0.22,
        "rotation": 0.15, "absence_penalty": 0.05,
        "hot_momentum": 0.22, "diversity": 0.05, "similarity": 0.03,
    },
    "extreme_cold": {
        "ultra_cold": 0.45, "anti_hot": 0.25, "gap": 0.15,
        "rotation": 0.08, "absence_penalty": 0.04,
        "diversity": 0.02, "similarity": 0.01,
    },
}


def _configurations():
    configurations = []
    for normalize, feedback_values, absence_values in (
        (False, (0.0, 0.5, 1.0), (0.5, 1.0)),
        (True, (0.0, 0.05, 0.1), (0.0, 0.1)),
    ):
        prefix = "normalized" if normalize else "raw"
        for feedback in feedback_values:
            for cold_gap in (0.5, 1.0):
                for absence in absence_values:
                    configurations.append({
                        "id": f"{prefix}_f{feedback}_c{cold_gap}_a{absence}",
                        "normalize": normalize,
                        "feedback_penalty": feedback,
                        "cold_gap_multiplier": cold_gap,
                        "absence_penalty": absence,
                    })
    return configurations


def _feature_cache(animals, model):
    """Calculate prefix-only component scores once for all24 experts."""
    base = model.base_predictor
    cache = []
    for index in range(1, len(animals) + 1):
        history = animals[:index]
        components = {
            "ultra_cold": base._ultra_cold_strategy(history),
            "anti_hot": base._anti_hot_strategy(history),
            "gap": base._gap_analysis(history),
            "rotation": base._rotation_advanced(history),
            "absence_penalty": base._continuous_absence_penalty(history),
            "diversity": base._diversity_boost(history),
            "similarity": base._historical_similarity(history),
        }
        scenario = model._detect_scenario(history)
        if scenario == "extreme_hot":
            components["hot_momentum"] = model._hot_momentum_strategy(history)
        reverse = list(reversed(history))
        long_penalties = {}
        for zodiac in model.zodiacs:
            absence = reverse.index(zodiac) if zodiac in reverse else len(history)
            long_penalties[zodiac] = min(42.0, max(0, absence - 29) * 1.5)
        normalized = {}
        for name, scores in components.items():
            low, high = min(scores.values()), max(scores.values())
            normalized[name] = {
                zodiac: (score - low) / (high - low) if high > low else 0.0
                for zodiac, score in scores.items()
            }
        cache.append({
            "index": index, "scenario": scenario, "components": components,
            "normalized": normalized, "long_penalties": long_penalties,
        })
    return cache


def _stream(animals, zodiacs, features, parameters, first_index):
    misses = {zodiac: 0 for zodiac in zodiacs}
    predictions = {}
    for row in features:
        index = row["index"]
        weights = dict(SCENARIO_WEIGHTS[row["scenario"]])
        for component in ("ultra_cold", "gap"):
            weights[component] *= parameters["cold_gap_multiplier"]
        source = row["normalized"] if parameters["normalize"] else row["components"]
        if parameters["normalize"]:
            total = sum(weights.values())
            weights = {name: weight / total for name, weight in weights.items()}
        scores = {}
        for zodiac in zodiacs:
            score = 0.0
            for component, values in source.items():
                score += values.get(zodiac, 0) * weights.get(component, 0)
            feedback = min(24.0, max(0, misses[zodiac] - 3) * 2.0)
            if parameters["normalize"]:
                score -= parameters["absence_penalty"] * row["long_penalties"][zodiac] / 42.0
                score -= parameters["feedback_penalty"] * feedback / 24.0
            else:
                score -= parameters["absence_penalty"] * row["long_penalties"][zodiac]
                score -= parameters["feedback_penalty"] * feedback
            scores[zodiac] = float(score)
        prediction = sorted(zodiacs, key=lambda zodiac: -scores[zodiac])[:5]
        if index >= first_index:
            predictions[index] = prediction
        if index < len(animals):
            actual = animals[index]
            for zodiac in zodiacs:
                misses[zodiac] = misses[zodiac] + 1 if zodiac in prediction and zodiac != actual else 0
    return predictions


def build_calibrated_candidates(animals, first_index=40):
    """Return24 fixed candidates and forecasts at indices first_index..N.

    IndexN is the next-draw forecast; there is no outcome/state update for it.
    Stable zodiac order resolves score ties, matching the original TOP5 model.
    """
    if isinstance(first_index, bool) or not isinstance(first_index, (int, np.integer)):
        raise ValueError("首个预测索引必须是正整数")
    history = [str(animal).strip() for animal in animals]
    if not history or not 1 <= first_index <= len(history):
        raise ValueError("首个预测索引必须在1到历史长度之间")
    model = ZodiacSimpleSmart()
    invalid = sorted(set(history) - set(model.zodiacs))
    if invalid:
        raise ValueError(f"历史数据包含无效生肖：{'、'.join(invalid)}")
    features = _feature_cache(history, model)
    candidates = []
    for parameters in _configurations():
        identifier = parameters["id"]
        label = "原始生肖TOP5" if identifier == BASELINE_ID else (
            f"{'标准化' if parameters['normalize'] else '原量纲'}校准"
            f"（反馈{parameters['feedback_penalty']:g}、"
            f"冷间隔{parameters['cold_gap_multiplier']:g}、"
            f"长遗漏{parameters['absence_penalty']:g}）"
        )
        candidates.append({
            "id": identifier, "label": label, "family": "calibrated",
            "parameters": {name: value for name, value in parameters.items() if name != "id"},
            "predictions": _stream(history, model.zodiacs, features, parameters, int(first_index)),
        })
    return candidates
