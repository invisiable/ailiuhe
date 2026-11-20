"""Fixed TOP5 statistical and online candidates for chronological comparison.

The declarations below are fixed research candidates. This module does not
select parameters, compare holdout results, read files or modify the existing
ZodiacSimpleSmart model. Every forecast is made before its outcome is observed.
"""

from __future__ import annotations

import numpy as np

from zodiac_simple_smart import ZodiacSimpleSmart


ZODIACS = tuple(ZodiacSimpleSmart().zodiacs)
ZODIAC_INDEX = {zodiac: index for index, zodiac in enumerate(ZODIACS)}
EXPERT_NAMES = (
    "frequency20", "frequency60", "cold30", "markov60", "markov150", "markov_order2",
)


def _top5(scores):
    return sorted(range(12), key=lambda index: (-float(scores[index]), index))[:5]


def _frequency(history, window, alpha=2.0):
    counts = np.bincount(history[-window:], minlength=12).astype(float)
    return (counts + alpha) / (counts.sum() + 12 * alpha)


def _transition(history, window, order=1, alpha=12.0):
    prior = _frequency(history, 150)
    if len(history) < order:
        return prior
    state = tuple(history[-order:])
    sequence = history[-window:]
    counts = np.zeros(12)
    for index in range(order, len(sequence)):
        if tuple(sequence[index - order:index]) == state:
            counts[sequence[index]] += 1
    return (counts + alpha * prior) / (counts.sum() + alpha)


def _statistical_experts(history):
    counts = np.bincount(history[-30:], minlength=12).astype(float)
    inverse = 1.0 / (counts + 2.0)
    inverse /= inverse.sum()
    return np.array([
        _frequency(history, 20), _frequency(history, 60), inverse,
        _transition(history, 60), _transition(history, 150),
        _transition(history, 150, order=2),
    ])


def _expert_rows(animals):
    """Cache pre-draw experts and the feedback-aware baseline's full ranking."""
    baseline = ZodiacSimpleSmart()
    selected_misses = {zodiac: 0 for zodiac in ZODIACS}
    encoded = [ZODIAC_INDEX[zodiac] for zodiac in animals]
    rows = []
    for index in range(1, len(animals) + 1):
        result = baseline.predict_from_history(
            animals[:index], top_n=5, selected_miss_streaks=selected_misses,
        )
        ranking = [ZODIAC_INDEX[zodiac] for zodiac, _ in result["all_scores"]]
        rank_probability = np.empty(12)
        for position, zodiac in enumerate(ranking):
            rank_probability[zodiac] = (12 - position) / 78.0
        rows.append({
            "index": index,
            "actual": encoded[index] if index < len(encoded) else None,
            "baseline_rank_probability": rank_probability,
            "experts": _statistical_experts(encoded[:index]),
        })
        if index < len(animals):
            for zodiac in ZODIACS:
                selected_misses[zodiac] = (
                    selected_misses[zodiac] + 1
                    if zodiac in result["top5"] and zodiac != animals[index]
                    else 0
                )
    return rows


def _configurations():
    configs = [
        {
            "id": "fixed_frequency_markov", "label": "平滑频率＋转移（频率主导）",
            "family": "smoothed", "kind": "fixed", "weights": [0.4, 0.3, 0, 0, 0.3, 0],
        },
        {
            "id": "fixed_transition_heavy", "label": "平滑频率＋转移（转移主导）",
            "family": "smoothed", "kind": "fixed", "weights": [0.2, 0.2, 0, 0, 0.6, 0],
        },
        {
            "id": "fixed_hot_cold_transition", "label": "平滑冷热＋转移",
            "family": "smoothed", "kind": "fixed", "weights": [0.25, 0, 0.25, 0, 0.5, 0],
        },
        {
            "id": "fixed_multiscale_transition", "label": "多周期平滑转移",
            "family": "smoothed", "kind": "fixed", "weights": [0, 0.25, 0, 0.25, 0.25, 0.25],
        },
    ]
    for include_baseline, family, prefix in (
        (False, "online", "online_statistics"),
        (True, "baseline_online", "online_baseline_blend"),
    ):
        for eta in (0.15, 0.4):
            for decay in (0.98, 0.995):
                configs.append({
                    "id": f"{prefix}_eta{eta}_decay{decay}",
                    "label": f"{'原模型在线融合' if include_baseline else '在线统计专家'}（{eta}/{decay}）",
                    "family": family, "kind": "online", "eta": eta,
                    "decay": decay, "include_baseline": include_baseline,
                    "prior_mix": 0.2,
                })
    return configs


def _predict_candidate(config, rows, first_index):
    predictions = {}
    if config["kind"] == "online":
        prior = (
            np.array([0.5] + [0.5 / 6] * 6)
            if config["include_baseline"] else np.ones(6) / 6
        )
        log_performance = np.zeros(len(prior))
    for row in rows:
        if config["kind"] == "fixed":
            forecast = _top5(np.dot(config["weights"], row["experts"]))
        else:
            experts = (
                np.vstack([row["baseline_rank_probability"], row["experts"]])
                if config["include_baseline"] else row["experts"]
            )
            raw = prior * np.exp(log_performance - log_performance.max())
            weights = (1 - config["prior_mix"]) * raw / raw.sum() + config["prior_mix"] * prior
            forecast = _top5(weights @ experts)
        if row["index"] >= first_index:
            prediction = [ZODIACS[index] for index in forecast]
            if len(prediction) != 5 or len(set(prediction)) != 5:
                raise ValueError("在线候选必须始终预测5个不重复生肖")
            predictions[row["index"]] = prediction
        # The current outcome is applied only after its forecast has been saved.
        if config["kind"] == "online" and row["actual"] is not None:
            hits = np.array([row["actual"] in _top5(expert) for expert in experts], dtype=float)
            log_performance = (
                config["decay"] * log_performance
                + config["eta"] * (hits - 5 / 12)
            )
    return predictions


def build_online_candidates(animals, first_index=40):
    """Return the12 declared candidates, indexed through the next forecast.

    ``predictions[i]`` forecasts history outcome ``animals[i]`` using only
    ``animals[:i]``. The final key ``len(animals)`` is the next-draw forecast.
    Scores used inside expert mixtures are ranking weights, not calibrated
    probabilities. Parameter selection is intentionally left to the caller.
    """
    history = [str(animal).strip() for animal in animals]
    invalid = sorted(set(history) - set(ZODIACS))
    if invalid:
        raise ValueError(f"历史数据包含无效生肖：{'、'.join(invalid)}")
    if not history:
        raise ValueError("在线候选至少需要1条历史生肖")
    if isinstance(first_index, bool) or not isinstance(first_index, (int, np.integer)):
        raise ValueError("预测起始索引必须是整数")
    if not 1 <= first_index <= len(history):
        raise ValueError("预测起始索引必须在1到历史记录数之间")
    rows = _expert_rows(history)
    return [
        {
            "id": config["id"], "label": config["label"],
            "family": config["family"],
            "parameters": {key: value for key, value in config.items() if key not in ("id", "label", "family")},
            "predictions": _predict_candidate(config, rows, int(first_index)),
        }
        for config in _configurations()
    ]
