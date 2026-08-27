"""Walk-forward TOP20 monitoring with Zodiac TOP10/TOP11 exclusion."""

from top15_union_backtest import summarize_miss_streaks
from top20_dual_horizon_predictor import Top20DualHorizonPredictor
from zodiac_top10_predictor import (
    NUM_TO_ZODIAC_2026,
    ZODIAC_CYCLE_2026,
    ZodiacTop10Predictor,
)


DEFAULT_TEST_PERIODS = 400
MIN_HISTORY = 25
TOP_N = 20


def _filter_prediction(prediction, selected_zodiacs):
    selected = set(selected_zodiacs)
    return [
        int(number)
        for number in prediction
        if NUM_TO_ZODIAC_2026[int(number)] in selected
    ]


def run_top20_zodiac_filter_backtest(numbers, test_periods=DEFAULT_TEST_PERIODS):
    """Backtest TOP20 after excluding numbers outside Zodiac TOP10/TOP11."""
    values = [int(number) for number in numbers]
    if any(number < 1 or number > 49 for number in values):
        raise ValueError("历史号码必须在 1 到 49 之间")
    if len(values) <= MIN_HISTORY:
        raise ValueError(f"至少需要 {MIN_HISTORY + 1} 期历史数据")
    periods_requested = int(test_periods)
    if periods_requested <= 0:
        raise ValueError("回测期数必须大于 0")

    periods = min(periods_requested, len(values) - MIN_HISTORY)
    start_index = len(values) - periods
    top20_predictor = Top20DualHorizonPredictor()
    zodiac_predictor = ZodiacTop10Predictor()
    rows = []
    original_hits = 0
    filtered_hits = 0
    filtered_miss_streak = 0

    for index in range(start_index, len(values)):
        history = values[:index]
        top20_prediction = top20_predictor.predict(history, top_n=TOP_N)
        selected_zodiacs = zodiac_predictor.predict(history, top_n=10)
        excluded_zodiacs = [
            zodiac for zodiac in ZODIAC_CYCLE_2026 if zodiac not in selected_zodiacs
        ]
        filtered_prediction = _filter_prediction(top20_prediction, selected_zodiacs)
        actual = values[index]
        original_hit = actual in top20_prediction
        filtered_hit = actual in filtered_prediction
        zodiac_hit = NUM_TO_ZODIAC_2026[actual] in selected_zodiacs
        zodiac_predictor.record_result(zodiac_hit)

        original_hits += int(original_hit)
        filtered_hits += int(filtered_hit)
        filtered_miss_streak = 0 if filtered_hit else filtered_miss_streak + 1
        rows.append({
            "index": index,
            "actual": actual,
            "top20_prediction": top20_prediction,
            "selected_zodiacs": selected_zodiacs,
            "excluded_zodiacs": excluded_zodiacs,
            "filtered_prediction": filtered_prediction,
            "filtered_count": len(filtered_prediction),
            "reduced_count": len(top20_prediction) - len(filtered_prediction),
            "original_hit": original_hit,
            "filtered_hit": filtered_hit,
            "filtered_miss_streak": filtered_miss_streak,
        })

    miss_summary = summarize_miss_streaks(
        [row["filtered_hit"] for row in rows]
    )
    filtered_counts = [row["filtered_count"] for row in rows]
    reduced_counts = [row["reduced_count"] for row in rows]

    next_top20 = top20_predictor.predict(values, top_n=TOP_N)
    next_selected_zodiacs = zodiac_predictor.predict(values, top_n=10)
    next_excluded_zodiacs = [
        zodiac for zodiac in ZODIAC_CYCLE_2026 if zodiac not in next_selected_zodiacs
    ]
    next_filtered = _filter_prediction(next_top20, next_selected_zodiacs)

    return {
        "test_periods": periods,
        "rows": rows,
        "original_hits": original_hits,
        "original_hit_rate": original_hits / periods,
        "filtered_hits": filtered_hits,
        "filtered_hit_rate": filtered_hits / periods,
        "lost_hits": original_hits - filtered_hits,
        "average_filtered_count": sum(filtered_counts) / periods,
        "min_filtered_count": min(filtered_counts),
        "max_filtered_count": max(filtered_counts),
        "average_reduced_count": sum(reduced_counts) / periods,
        "miss_streak_distribution": miss_summary["distribution"],
        "max_miss_streak": miss_summary["max_miss_streak"],
        "current_miss_streak": miss_summary["current_miss_streak"],
        "next": {
            "top20_prediction": next_top20,
            "selected_zodiacs": next_selected_zodiacs,
            "excluded_zodiacs": next_excluded_zodiacs,
            "filtered_prediction": next_filtered,
            "filtered_count": len(next_filtered),
            "reduced_count": len(next_top20) - len(next_filtered),
        },
    }
