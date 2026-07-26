"""动态 12/25 窗口的 TOP15 滚动预测。

连续未中少于 3 期时使用最近 12 期数据；达到 3 期后改用最近
25 期，命中后恢复到 12 期。回测严格按时间顺序进行，不使用
待预测期及其之后的数据。
"""

from precise_top15_predictor import PreciseTop15Predictor


SHORT_WINDOW = 12
LONG_WINDOW = 25
SWITCH_AFTER_MISSES = 3
DEFAULT_TEST_PERIODS = 400
TOP_N = 15


def _selected_window(miss_streak):
    """Return the history window selected by the current miss streak."""
    return LONG_WINDOW if miss_streak >= SWITCH_AFTER_MISSES else SHORT_WINDOW


def run_top15_dynamic_12_25(numbers, test_periods=DEFAULT_TEST_PERIODS):
    """Walk forward through history and return audit rows plus next prediction.

    Args:
        numbers: Historical results in chronological order.
        test_periods: Maximum number of trailing periods to audit.

    Returns:
        A dictionary consumed by ``LuckyNumberGUI._run_top15_dynamic_12_25``.
    """
    values = [int(number) for number in numbers]
    if any(number < 1 or number > 49 for number in values):
        raise ValueError("历史号码必须在 1 到 49 之间")
    if len(values) <= LONG_WINDOW:
        raise ValueError(f"至少需要 {LONG_WINDOW + 1} 期历史数据")
    if test_periods <= 0:
        raise ValueError("回测期数必须大于 0")

    periods = min(int(test_periods), len(values) - LONG_WINDOW)
    start_index = len(values) - periods
    predictor = PreciseTop15Predictor()

    rows = []
    hits = 0
    miss_streak = 0
    max_miss_streak = 0

    for index in range(start_index, len(values)):
        window = _selected_window(miss_streak)
        history = values[index - window:index]
        prediction = predictor.predict(history, k=TOP_N)
        actual = values[index]
        hit = actual in prediction

        predictor.update_performance(prediction, actual)
        if hit:
            hits += 1
            miss_streak = 0
        else:
            miss_streak += 1
            max_miss_streak = max(max_miss_streak, miss_streak)

        rows.append({
            "index": index,
            "actual": actual,
            "window": window,
            "hit": hit,
            "miss_streak": miss_streak,
            "prediction": prediction,
        })

    next_window = _selected_window(miss_streak)
    next_prediction = predictor.predict(values[-next_window:], k=TOP_N)

    return {
        "test_periods": periods,
        "hits": hits,
        "hit_rate": hits / periods,
        "max_miss_streak": max_miss_streak,
        "current_miss_streak": miss_streak,
        "rows": rows,
        "window": next_window,
        "next_prediction": next_prediction,
    }

