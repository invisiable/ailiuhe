"""Walk-forward audit for the feedback-aware Zodiac v10 TOP5 model."""

from top15_union_backtest import summarize_miss_streaks
from zodiac_simple_smart import ZodiacSimpleSmart


DEFAULT_TEST_PERIODS = 400
TOP_N = 5


def _find_longest_miss_streaks(rows, max_miss_streak):
    if max_miss_streak == 0:
        return []

    ranges = []
    start_position = None
    for position, row in enumerate(rows):
        if not row["hit"] and start_position is None:
            start_position = position
        if row["hit"] and start_position is not None:
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


def run_zodiac_top5_audit(
    animals, test_periods=DEFAULT_TEST_PERIODS, top_n=TOP_N
):
    """Backtest the latest periods using only information available beforehand."""
    history = [str(animal).strip() for animal in animals]
    periods = int(test_periods)
    prediction_count = int(top_n)
    if periods <= 0:
        raise ValueError("回测期数必须大于 0")
    if prediction_count < 1 or prediction_count > 12:
        raise ValueError("预测生肖数量必须在 1 到 12 之间")
    if len(history) <= periods:
        raise ValueError(f"完整回测{periods}期至少需要{periods + 1}期历史数据")

    predictor = ZodiacSimpleSmart()
    invalid = sorted(set(history) - set(predictor.zodiacs))
    if invalid:
        raise ValueError(f"历史数据包含无效生肖：{invalid}")

    start_index = len(history) - periods
    selected_miss_streaks = {zodiac: 0 for zodiac in predictor.zodiacs}
    selection_counts = {zodiac: 0 for zodiac in predictor.zodiacs}
    selected_hits = {zodiac: 0 for zodiac in predictor.zodiacs}
    actual_counts = {zodiac: 0 for zodiac in predictor.zodiacs}
    rows = []
    hits = 0
    miss_streak = 0

    for index in range(1, len(history)):
        result = predictor.predict_from_history(
            history[:index],
            top_n=prediction_count,
            debug=False,
            selected_miss_streaks=selected_miss_streaks,
        )
        prediction = result["top5"]
        actual = history[index]
        hit = actual in prediction
        penalties = result.get("feedback_penalties", {})

        for zodiac in predictor.zodiacs:
            if zodiac in prediction:
                if actual == zodiac:
                    selected_miss_streaks[zodiac] = 0
                else:
                    selected_miss_streaks[zodiac] += 1
            else:
                selected_miss_streaks[zodiac] = 0

        if index < start_index:
            continue

        hits += int(hit)
        miss_streak = 0 if hit else miss_streak + 1
        for zodiac in predictor.zodiacs:
            actual_counts[zodiac] += int(actual == zodiac)
            if zodiac in prediction:
                selection_counts[zodiac] += 1
                selected_hits[zodiac] += int(actual == zodiac)

        rows.append({
            "index": index,
            "actual": actual,
            "prediction": prediction,
            "selected_model": result["selected_model"],
            "feedback_penalties": {
                zodiac: penalty
                for zodiac, penalty in penalties.items()
                if penalty > 0
            },
            "hit": hit,
            "miss_streak": miss_streak,
        })

    miss_summary = summarize_miss_streaks([row["hit"] for row in rows])
    zodiac_stats = {}
    for zodiac in predictor.zodiacs:
        selections = selection_counts[zodiac]
        zodiac_stats[zodiac] = {
            "selections": selections,
            "selection_rate": selections / periods,
            "actual_count": actual_counts[zodiac],
            "selected_hits": selected_hits[zodiac],
            "conditional_hit_rate": (
                selected_hits[zodiac] / selections if selections else 0.0
            ),
        }

    next_result = predictor.predict_from_history(
        history,
        top_n=prediction_count,
        debug=False,
        selected_miss_streaks=selected_miss_streaks,
    )
    next_penalties = {
        zodiac: penalty
        for zodiac, penalty in next_result.get("feedback_penalties", {}).items()
        if penalty > 0
    }

    return {
        "top_n": prediction_count,
        "test_periods": periods,
        "rows": rows,
        "hits": hits,
        "hit_rate": hits / periods,
        "max_miss_streak": miss_summary["max_miss_streak"],
        "current_miss_streak": miss_summary["current_miss_streak"],
        "miss_streak_distribution": miss_summary["distribution"],
        "longest_miss_streaks": _find_longest_miss_streaks(
            rows, miss_summary["max_miss_streak"]
        ),
        "zodiac_stats": zodiac_stats,
        "next_prediction": next_result["top5"],
        "next_selected_model": next_result["selected_model"],
        "next_feedback_penalties": next_penalties,
    }
