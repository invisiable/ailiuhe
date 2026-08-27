"""Walk-forward union of dynamic TOP15 and distilled confidence TOP4."""

from distill_top4_confidence_predictor import (
    NUM_TO_ZODIAC_2026,
    ZODIAC_NUMS_2026,
    DistillTop4ConfidencePredictor,
)
from precise_top15_predictor import PreciseTop15Predictor
from top15_dynamic_12_25 import LONG_WINDOW, SHORT_WINDOW, SWITCH_AFTER_MISSES, TOP_N
from top15_union_backtest import ordered_union, summarize_miss_streaks


DEFAULT_TEST_PERIODS = 400


def _zodiacs_to_numbers(zodiacs):
    return sorted({
        number
        for zodiac in zodiacs
        for number in ZODIAC_NUMS_2026[zodiac]
    })


def run_top15_dynamic_top4_union(numbers, test_periods=DEFAULT_TEST_PERIODS):
    """Backtest the dynamic TOP15 and distilled TOP4 number union."""
    values = [int(number) for number in numbers]
    if any(number < 1 or number > 49 for number in values):
        raise ValueError("历史号码必须在 1 到 49 之间")
    if len(values) <= LONG_WINDOW:
        raise ValueError(f"至少需要 {LONG_WINDOW + 1} 期历史数据")
    periods_requested = int(test_periods)
    if periods_requested <= 0:
        raise ValueError("回测期数必须大于 0")

    periods = min(periods_requested, len(values) - LONG_WINDOW)
    start_index = len(values) - periods
    dynamic_predictor = PreciseTop15Predictor()
    top4_predictor = DistillTop4ConfidencePredictor()
    dynamic_miss_streak = 0
    union_miss_streak = 0
    rows = []
    dynamic_hits = 0
    top4_hits = 0
    union_hits = 0
    top4_only_hits = 0

    for index in range(start_index, len(values)):
        dynamic_window = (
            LONG_WINDOW
            if dynamic_miss_streak >= SWITCH_AFTER_MISSES
            else SHORT_WINDOW
        )
        dynamic_prediction = dynamic_predictor.predict(
            values[index - dynamic_window:index], k=TOP_N
        )
        top4_zodiacs, top4_mode, _, confidence, _ = (
            top4_predictor.predict_with_details(values[:index], top_n=4)
        )
        top4_numbers = _zodiacs_to_numbers(top4_zodiacs)
        union_prediction = ordered_union(dynamic_prediction, top4_numbers)
        actual = values[index]
        dynamic_hit = actual in dynamic_prediction
        top4_hit = NUM_TO_ZODIAC_2026[actual] in top4_zodiacs
        union_hit = actual in union_prediction

        dynamic_predictor.update_performance(dynamic_prediction, actual)
        top4_predictor.record_result(top4_hit)
        dynamic_miss_streak = 0 if dynamic_hit else dynamic_miss_streak + 1
        union_miss_streak = 0 if union_hit else union_miss_streak + 1
        dynamic_hits += int(dynamic_hit)
        top4_hits += int(top4_hit)
        union_hits += int(union_hit)
        top4_only_hits += int(top4_hit and not dynamic_hit)

        rows.append({
            "index": index,
            "actual": actual,
            "dynamic_window": dynamic_window,
            "dynamic_prediction": dynamic_prediction,
            "dynamic_hit": dynamic_hit,
            "top4_zodiacs": top4_zodiacs,
            "top4_mode": top4_mode,
            "top4_confidence": confidence,
            "top4_numbers": top4_numbers,
            "top4_hit": top4_hit,
            "union_prediction": union_prediction,
            "union_count": len(union_prediction),
            "added_count": len(union_prediction) - len(dynamic_prediction),
            "union_hit": union_hit,
            "union_miss_streak": union_miss_streak,
        })

    miss_summary = summarize_miss_streaks(
        [row["union_hit"] for row in rows]
    )
    union_counts = [row["union_count"] for row in rows]
    added_counts = [row["added_count"] for row in rows]

    next_dynamic_window = (
        LONG_WINDOW
        if dynamic_miss_streak >= SWITCH_AFTER_MISSES
        else SHORT_WINDOW
    )
    next_dynamic_prediction = dynamic_predictor.predict(
        values[-next_dynamic_window:], k=TOP_N
    )
    next_top4_zodiacs, next_top4_mode, _, next_confidence, _ = (
        top4_predictor.predict_with_details(values, top_n=4)
    )
    next_top4_numbers = _zodiacs_to_numbers(next_top4_zodiacs)
    next_union_prediction = ordered_union(
        next_dynamic_prediction, next_top4_numbers
    )

    return {
        "test_periods": periods,
        "rows": rows,
        "dynamic_hits": dynamic_hits,
        "dynamic_hit_rate": dynamic_hits / periods,
        "top4_hits": top4_hits,
        "top4_hit_rate": top4_hits / periods,
        "union_hits": union_hits,
        "union_hit_rate": union_hits / periods,
        "top4_only_hits": top4_only_hits,
        "average_union_count": sum(union_counts) / periods,
        "min_union_count": min(union_counts),
        "max_union_count": max(union_counts),
        "average_added_count": sum(added_counts) / periods,
        "miss_streak_distribution": miss_summary["distribution"],
        "max_miss_streak": miss_summary["max_miss_streak"],
        "current_miss_streak": miss_summary["current_miss_streak"],
        "next_dynamic_window": next_dynamic_window,
        "next_dynamic_prediction": next_dynamic_prediction,
        "next_top4_zodiacs": next_top4_zodiacs,
        "next_top4_mode": next_top4_mode,
        "next_top4_confidence": next_confidence,
        "next_top4_numbers": next_top4_numbers,
        "next_union_prediction": next_union_prediction,
        "next_added_count": len(next_union_prediction) - len(next_dynamic_prediction),
    }
