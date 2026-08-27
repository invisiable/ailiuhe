"""Walk-forward union of TOP20 monitoring and Zodiac v10 TOP5 numbers."""

from top15_union_backtest import ordered_union, summarize_miss_streaks
from top20_dual_horizon_predictor import Top20DualHorizonPredictor
from zodiac_simple_smart import ZodiacSimpleSmart


DEFAULT_TEST_PERIODS = 400
TOP20_COUNT = 20
ZODIAC_COUNT = 5


def _expand_zodiac_numbers(zodiacs, zodiac_numbers):
    numbers = []
    for zodiac in zodiacs:
        if zodiac not in zodiac_numbers:
            raise ValueError(f"生肖TOP5模型缺少 {zodiac} 的号码映射")
        numbers.extend(zodiac_numbers[zodiac])

    expanded = sorted({int(number) for number in numbers})
    if any(number < 1 or number > 49 for number in expanded):
        raise ValueError("生肖TOP5模型的号码映射必须在 1 到 49 之间")
    return expanded


def _find_longest_miss_streaks(rows, max_miss_streak):
    if max_miss_streak == 0:
        return []

    ranges = []
    start_position = None
    for position, row in enumerate(rows):
        if not row["union_hit"] and start_position is None:
            start_position = position
        if row["union_hit"] and start_position is not None:
            length = position - start_position
            if length == max_miss_streak:
                ranges.append({
                    "start_index": rows[start_position]["index"],
                    "end_index": rows[position - 1]["index"],
                    "length": length,
                })
            start_position = None

    if start_position is not None:
        length = len(rows) - start_position
        if length == max_miss_streak:
            ranges.append({
                "start_index": rows[start_position]["index"],
                "end_index": rows[-1]["index"],
                "length": length,
            })
    return ranges


def run_top20_zodiac_top5_union(
    numbers, animals, test_periods=DEFAULT_TEST_PERIODS
):
    """Backtest the TOP20 and Zodiac v10 TOP5 number union without lookahead."""
    values = [int(number) for number in numbers]
    zodiac_history = [str(animal).strip() for animal in animals]
    periods = int(test_periods)

    if len(values) != len(zodiac_history):
        raise ValueError("号码历史与生肖历史期数必须一致")
    if any(number < 1 or number > 49 for number in values):
        raise ValueError("历史号码必须在 1 到 49 之间")
    if periods <= 0:
        raise ValueError("回测期数必须大于 0")
    if len(values) <= periods:
        raise ValueError(f"完整回测{periods}期至少需要{periods + 1}期历史数据")

    start_index = len(values) - periods
    top20_predictor = Top20DualHorizonPredictor()
    zodiac_predictor = ZodiacSimpleSmart()
    rows = []
    top20_hits = 0
    zodiac_animal_hits = 0
    zodiac_number_hits = 0
    union_hits = 0
    zodiac_only_hits = 0
    union_miss_streak = 0

    for index in range(start_index, len(values)):
        top20_prediction = top20_predictor.predict(values[:index], top_n=TOP20_COUNT)
        zodiac_result = zodiac_predictor.predict_from_history(
            zodiac_history[:index], top_n=ZODIAC_COUNT, debug=False
        )
        top5_zodiacs = zodiac_result["top5"]
        zodiac_numbers = _expand_zodiac_numbers(
            top5_zodiacs, zodiac_predictor.zodiac_numbers
        )
        union_prediction = ordered_union(top20_prediction, zodiac_numbers)
        actual = values[index]
        actual_animal = zodiac_history[index]
        top20_hit = actual in top20_prediction
        zodiac_animal_hit = actual_animal in top5_zodiacs
        zodiac_number_hit = actual in zodiac_numbers
        union_hit = actual in union_prediction
        union_miss_streak = 0 if union_hit else union_miss_streak + 1

        top20_hits += int(top20_hit)
        zodiac_animal_hits += int(zodiac_animal_hit)
        zodiac_number_hits += int(zodiac_number_hit)
        union_hits += int(union_hit)
        zodiac_only_hits += int(zodiac_number_hit and not top20_hit)
        rows.append({
            "index": index,
            "actual": actual,
            "actual_animal": actual_animal,
            "top20_prediction": top20_prediction,
            "top20_hit": top20_hit,
            "top5_zodiacs": top5_zodiacs,
            "selected_model": zodiac_result["selected_model"],
            "zodiac_numbers": zodiac_numbers,
            "zodiac_animal_hit": zodiac_animal_hit,
            "zodiac_number_hit": zodiac_number_hit,
            "union_prediction": union_prediction,
            "union_count": len(union_prediction),
            "added_count": len(union_prediction) - len(top20_prediction),
            "union_hit": union_hit,
            "union_miss_streak": union_miss_streak,
        })

    miss_summary = summarize_miss_streaks(
        [row["union_hit"] for row in rows]
    )
    union_counts = [row["union_count"] for row in rows]
    added_counts = [row["added_count"] for row in rows]

    next_top20_prediction = top20_predictor.predict(values, top_n=TOP20_COUNT)
    next_zodiac_result = zodiac_predictor.predict_from_history(
        zodiac_history, top_n=ZODIAC_COUNT, debug=False
    )
    next_top5_zodiacs = next_zodiac_result["top5"]
    next_zodiac_numbers = _expand_zodiac_numbers(
        next_top5_zodiacs, zodiac_predictor.zodiac_numbers
    )
    next_union_prediction = ordered_union(
        next_top20_prediction, next_zodiac_numbers
    )

    return {
        "test_periods": periods,
        "rows": rows,
        "top20_hits": top20_hits,
        "top20_hit_rate": top20_hits / periods,
        "zodiac_animal_hits": zodiac_animal_hits,
        "zodiac_animal_hit_rate": zodiac_animal_hits / periods,
        "zodiac_number_hits": zodiac_number_hits,
        "zodiac_number_hit_rate": zodiac_number_hits / periods,
        "union_hits": union_hits,
        "union_hit_rate": union_hits / periods,
        "zodiac_only_hits": zodiac_only_hits,
        "average_union_count": sum(union_counts) / periods,
        "min_union_count": min(union_counts),
        "max_union_count": max(union_counts),
        "average_added_count": sum(added_counts) / periods,
        "miss_streak_distribution": miss_summary["distribution"],
        "max_miss_streak": miss_summary["max_miss_streak"],
        "current_miss_streak": miss_summary["current_miss_streak"],
        "longest_miss_streaks": _find_longest_miss_streaks(
            rows, miss_summary["max_miss_streak"]
        ),
        "zodiac_mapping": dict(zodiac_predictor.zodiac_numbers),
        "next_top20_prediction": next_top20_prediction,
        "next_top5_zodiacs": next_top5_zodiacs,
        "next_selected_model": next_zodiac_result["selected_model"],
        "next_zodiac_numbers": next_zodiac_numbers,
        "next_union_prediction": next_union_prediction,
        "next_added_count": len(next_union_prediction) - len(next_top20_prediction),
    }
