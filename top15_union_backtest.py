"""Walk-forward union of dynamic 12/25 and fixed 25-draw TOP15 models."""

from precise_top15_predictor import PreciseTop15Predictor
from top15_dynamic_12_25 import LONG_WINDOW, SHORT_WINDOW, SWITCH_AFTER_MISSES, TOP_N
from zodiac_top10_predictor import (
    NUM_TO_ZODIAC_2026,
    ZODIAC_CYCLE_2026,
    ZodiacTop10Predictor,
)


DEFAULT_TEST_PERIODS = 400
OPTIMAL_WINDOW = 25


def ordered_union(primary, secondary):
    """Return a duplicate-free union while preserving prediction priority."""
    result = []
    seen = set()
    for number in list(primary) + list(secondary):
        value = int(number)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def apply_zodiac_filter(numbers, selected_zodiacs, number_to_zodiac, protected_numbers=()):
    """Remove numbers outside selected zodiacs while retaining protected picks."""
    selected = set(selected_zodiacs)
    protected = {int(number) for number in protected_numbers}
    return [
        int(number)
        for number in numbers
        if int(number) in protected or number_to_zodiac[int(number)] in selected
    ]


def build_slim_protected_union(
    dynamic_prediction,
    fixed_prediction,
    selected_zodiacs,
    number_to_zodiac,
    protected_added_count=5,
):
    """Keep dynamic picks, protect leading new fixed picks, then filter by zodiac."""
    dynamic = ordered_union(dynamic_prediction, [])
    dynamic_set = set(dynamic)
    additions = [
        number
        for number in ordered_union(fixed_prediction, [])
        if number not in dynamic_set
    ]
    protected = additions[:protected_added_count]
    selected = set(selected_zodiacs)
    filtered = [
        number
        for number in additions[protected_added_count:]
        if number_to_zodiac[number] in selected
    ]
    return dynamic + protected + filtered


def summarize_miss_streaks(hits):
    """Count miss runs by their observed length, including a trailing run."""
    distribution = {}
    miss_streak = 0
    max_miss_streak = 0

    for hit in hits:
        if hit:
            if miss_streak:
                distribution[miss_streak] = distribution.get(miss_streak, 0) + 1
                max_miss_streak = max(max_miss_streak, miss_streak)
                miss_streak = 0
        else:
            miss_streak += 1

    current_miss_streak = miss_streak
    if miss_streak:
        distribution[miss_streak] = distribution.get(miss_streak, 0) + 1
        max_miss_streak = max(max_miss_streak, miss_streak)

    return {
        "distribution": dict(sorted(distribution.items())),
        "current_miss_streak": current_miss_streak,
        "max_miss_streak": max_miss_streak,
    }


def summarize_completed_miss_probabilities(hits):
    """Return exact and cumulative probabilities for completed miss runs."""
    completed_distribution = {}
    miss_streak = 0

    for hit in hits:
        if hit:
            if miss_streak:
                completed_distribution[miss_streak] = (
                    completed_distribution.get(miss_streak, 0) + 1
                )
                miss_streak = 0
        else:
            miss_streak += 1

    completed_streaks = sum(completed_distribution.values())
    exact_probabilities = {}
    at_least_probabilities = {}
    if completed_streaks:
        max_length = max(completed_distribution)
        exact_probabilities = {
            length: completed_distribution.get(length, 0) / completed_streaks
            for length in range(1, max_length + 1)
        }
        at_least_probabilities = {
            length: sum(
                count
                for streak_length, count in completed_distribution.items()
                if streak_length >= length
            ) / completed_streaks
            for length in range(1, max_length + 1)
        }

    return {
        "completed_distribution": dict(sorted(completed_distribution.items())),
        "completed_streaks": completed_streaks,
        "current_miss_streak": miss_streak,
        "exact_probabilities": exact_probabilities,
        "at_least_probabilities": at_least_probabilities,
    }


def run_top15_union_backtest(numbers, test_periods=DEFAULT_TEST_PERIODS):
    """Backtest the prediction union without using the target period as input."""
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
    optimal_predictor = PreciseTop15Predictor()
    top10_predictor = ZodiacTop10Predictor()

    rows = []
    dynamic_hits = 0
    optimal_hits = 0
    union_hits = 0
    top10_zodiac_hits = 0
    top10_hard_hits = 0
    top10_protected_hits = 0
    dynamic_miss_streak = 0
    union_miss_streak = 0

    for index in range(start_index, len(values)):
        dynamic_window = LONG_WINDOW if dynamic_miss_streak >= SWITCH_AFTER_MISSES else SHORT_WINDOW
        dynamic_prediction = dynamic_predictor.predict(values[index - dynamic_window:index], k=TOP_N)
        optimal_prediction = optimal_predictor.predict(values[index - OPTIMAL_WINDOW:index], k=TOP_N)
        union_prediction = ordered_union(dynamic_prediction, optimal_prediction)
        top10_selected_zodiacs = top10_predictor.predict(values[:index], top_n=10)
        top10_excluded_zodiacs = [
            zodiac for zodiac in ZODIAC_CYCLE_2026 if zodiac not in top10_selected_zodiacs
        ]
        top10_hard_prediction = apply_zodiac_filter(
            union_prediction,
            top10_selected_zodiacs,
            NUM_TO_ZODIAC_2026,
        )
        top10_protected_prediction = apply_zodiac_filter(
            union_prediction,
            top10_selected_zodiacs,
            NUM_TO_ZODIAC_2026,
            protected_numbers=dynamic_prediction,
        )
        slim_prediction = build_slim_protected_union(
            dynamic_prediction,
            optimal_prediction,
            top10_selected_zodiacs,
            NUM_TO_ZODIAC_2026,
        )
        actual = values[index]

        dynamic_hit = actual in dynamic_prediction
        optimal_hit = actual in optimal_prediction
        union_hit = actual in union_prediction
        top10_zodiac_hit = NUM_TO_ZODIAC_2026[actual] in top10_selected_zodiacs
        top10_hard_hit = actual in top10_hard_prediction
        top10_protected_hit = actual in top10_protected_prediction
        slim_hit = actual in slim_prediction
        dynamic_predictor.update_performance(dynamic_prediction, actual)
        optimal_predictor.update_performance(optimal_prediction, actual)
        top10_predictor.record_result(top10_zodiac_hit)

        dynamic_hits += int(dynamic_hit)
        optimal_hits += int(optimal_hit)
        union_hits += int(union_hit)
        top10_zodiac_hits += int(top10_zodiac_hit)
        top10_hard_hits += int(top10_hard_hit)
        top10_protected_hits += int(top10_protected_hit)
        dynamic_miss_streak = 0 if dynamic_hit else dynamic_miss_streak + 1
        union_miss_streak = 0 if union_hit else union_miss_streak + 1

        rows.append({
            "index": index,
            "actual": actual,
            "dynamic_prediction": dynamic_prediction,
            "dynamic_window": dynamic_window,
            "dynamic_hit": dynamic_hit,
            "optimal_prediction": optimal_prediction,
            "optimal_hit": optimal_hit,
            "union_prediction": union_prediction,
            "union_count": len(union_prediction),
            "added_count": len(union_prediction) - len(dynamic_prediction),
            "hit": union_hit,
            "miss_streak": union_miss_streak,
            "top10_selected_zodiacs": top10_selected_zodiacs,
            "top10_excluded_zodiacs": top10_excluded_zodiacs,
            "top10_hard_prediction": top10_hard_prediction,
            "top10_hard_count": len(top10_hard_prediction),
            "top10_hard_reduction": len(union_prediction) - len(top10_hard_prediction),
            "top10_hard_hit": top10_hard_hit,
            "top10_protected_prediction": top10_protected_prediction,
            "top10_protected_count": len(top10_protected_prediction),
            "top10_protected_reduction": len(union_prediction) - len(top10_protected_prediction),
            "top10_protected_hit": top10_protected_hit,
            "slim_prediction": slim_prediction,
            "slim_count": len(slim_prediction),
            "slim_reduction": len(union_prediction) - len(slim_prediction),
            "slim_hit": slim_hit,
        })

    next_dynamic_window = LONG_WINDOW if dynamic_miss_streak >= SWITCH_AFTER_MISSES else SHORT_WINDOW
    next_dynamic_prediction = dynamic_predictor.predict(values[-next_dynamic_window:], k=TOP_N)
    next_optimal_prediction = optimal_predictor.predict(values[-OPTIMAL_WINDOW:], k=TOP_N)
    next_union_prediction = ordered_union(next_dynamic_prediction, next_optimal_prediction)
    next_top10_selected_zodiacs = top10_predictor.predict(values, top_n=10)
    next_top10_excluded_zodiacs = [
        zodiac for zodiac in ZODIAC_CYCLE_2026 if zodiac not in next_top10_selected_zodiacs
    ]
    next_top10_hard_prediction = apply_zodiac_filter(
        next_union_prediction,
        next_top10_selected_zodiacs,
        NUM_TO_ZODIAC_2026,
    )
    next_top10_protected_prediction = apply_zodiac_filter(
        next_union_prediction,
        next_top10_selected_zodiacs,
        NUM_TO_ZODIAC_2026,
        protected_numbers=next_dynamic_prediction,
    )
    next_slim_prediction = build_slim_protected_union(
        next_dynamic_prediction,
        next_optimal_prediction,
        next_top10_selected_zodiacs,
        NUM_TO_ZODIAC_2026,
    )
    added_counts = [row["added_count"] for row in rows]
    miss_summary = summarize_miss_streaks([row["hit"] for row in rows])
    miss_probabilities = summarize_completed_miss_probabilities(
        [row["hit"] for row in rows]
    )
    top10_hard_counts = [row["top10_hard_count"] for row in rows]
    top10_hard_reductions = [row["top10_hard_reduction"] for row in rows]
    top10_protected_counts = [row["top10_protected_count"] for row in rows]
    top10_protected_reductions = [row["top10_protected_reduction"] for row in rows]
    top10_hard_miss_summary = summarize_miss_streaks(
        [row["top10_hard_hit"] for row in rows]
    )
    top10_hard_miss_probabilities = summarize_completed_miss_probabilities(
        [row["top10_hard_hit"] for row in rows]
    )
    top10_protected_miss_summary = summarize_miss_streaks(
        [row["top10_protected_hit"] for row in rows]
    )
    top10_protected_miss_probabilities = summarize_completed_miss_probabilities(
        [row["top10_protected_hit"] for row in rows]
    )
    slim_counts = [row["slim_count"] for row in rows]
    slim_reductions = [row["slim_reduction"] for row in rows]
    slim_hits = [row["slim_hit"] for row in rows]
    slim_miss_summary = summarize_miss_streaks(slim_hits)
    slim_miss_probabilities = summarize_completed_miss_probabilities(slim_hits)

    return {
        "test_periods": periods,
        "rows": rows,
        "dynamic_hits": dynamic_hits,
        "dynamic_hit_rate": dynamic_hits / periods,
        "optimal_hits": optimal_hits,
        "optimal_hit_rate": optimal_hits / periods,
        "union_hits": union_hits,
        "union_hit_rate": union_hits / periods,
        "miss_streak_distribution": miss_summary["distribution"],
        "miss_streak_probabilities": miss_probabilities,
        "current_miss_streak": miss_summary["current_miss_streak"],
        "max_miss_streak": miss_summary["max_miss_streak"],
        "added_total": sum(added_counts),
        "added_average": sum(added_counts) / periods,
        "added_min": min(added_counts),
        "added_max": max(added_counts),
        "next_dynamic_window": next_dynamic_window,
        "next_dynamic_prediction": next_dynamic_prediction,
        "next_optimal_prediction": next_optimal_prediction,
        "next_union_prediction": next_union_prediction,
        "next_added_count": len(next_union_prediction) - len(next_dynamic_prediction),
        "slim_filter": {
            "hits": sum(slim_hits),
            "hit_rate": sum(slim_hits) / periods,
            "lost_hits": union_hits - sum(slim_hits),
            "average_count": sum(slim_counts) / periods,
            "min_count": min(slim_counts),
            "max_count": max(slim_counts),
            "average_reduction": sum(slim_reductions) / periods,
            "max_miss_streak": slim_miss_summary["max_miss_streak"],
            "current_miss_streak": slim_miss_summary["current_miss_streak"],
            "miss_streak_distribution": slim_miss_summary["distribution"],
            "miss_streak_probabilities": slim_miss_probabilities,
            "next_prediction": next_slim_prediction,
        },
        "top10_filter": {
            "zodiac_hits": top10_zodiac_hits,
            "zodiac_hit_rate": top10_zodiac_hits / periods,
            "hard_hits": top10_hard_hits,
            "hard_hit_rate": top10_hard_hits / periods,
            "hard_lost_hits": union_hits - top10_hard_hits,
            "hard_hit_retention_rate": top10_hard_hits / union_hits if union_hits else 0,
            "hard_average_count": sum(top10_hard_counts) / periods,
            "hard_min_count": min(top10_hard_counts),
            "hard_max_count": max(top10_hard_counts),
            "hard_average_reduction": sum(top10_hard_reductions) / periods,
            "hard_miss_streak_distribution": top10_hard_miss_summary["distribution"],
            "hard_current_miss_streak": top10_hard_miss_summary["current_miss_streak"],
            "hard_max_miss_streak": top10_hard_miss_summary["max_miss_streak"],
            "hard_miss_streak_probabilities": top10_hard_miss_probabilities,
            "protected_hits": top10_protected_hits,
            "protected_hit_rate": top10_protected_hits / periods,
            "protected_lost_hits": union_hits - top10_protected_hits,
            "protected_hit_retention_rate": top10_protected_hits / union_hits if union_hits else 0,
            "protected_average_count": sum(top10_protected_counts) / periods,
            "protected_min_count": min(top10_protected_counts),
            "protected_max_count": max(top10_protected_counts),
            "protected_average_reduction": sum(top10_protected_reductions) / periods,
            "protected_miss_streak_distribution": top10_protected_miss_summary["distribution"],
            "protected_current_miss_streak": top10_protected_miss_summary["current_miss_streak"],
            "protected_max_miss_streak": top10_protected_miss_summary["max_miss_streak"],
            "protected_miss_streak_probabilities": top10_protected_miss_probabilities,
            "next_selected_zodiacs": next_top10_selected_zodiacs,
            "next_excluded_zodiacs": next_top10_excluded_zodiacs,
            "next_hard_prediction": next_top10_hard_prediction,
            "next_protected_prediction": next_top10_protected_prediction,
        },
    }
