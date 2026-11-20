"""Fixed, independent TOP3 rule using only counts settled before each forecast.

The rule has no offline machine-learning fitting, searched windows or weights,
or TOP5 dependency. It still estimates frequencies online; it does not promise
a particular future hit rate.
All public functions are pure and perform no file or network I/O.
"""

from __future__ import annotations

from collections import Counter
from numbers import Integral

import pandas as pd


ZODIAC_ORDER = ("鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪")
_VALID_ZODIACS = frozenset(ZODIAC_ORDER)


def _clean_animals(history):
    if isinstance(history, (str, bytes)):
        raise ValueError("history必须是生肖序列，不能是单个字符串")
    try:
        values = list(history)
    except TypeError as error:
        raise ValueError("history必须是生肖序列") from error
    animals = []
    for value in values:
        if not isinstance(value, str) or value.strip() not in _VALID_ZODIACS:
            raise ValueError("animal必须为有效生肖，不能包含缺失值")
        animals.append(value.strip())
    return animals


class IndependentTop3Predictor:
    """Equal mixture of Laplace-smoothed marginal and first-order frequencies.

    For history length N, G(z)=(count(z)+1)/(N+12).
    For its last animal a, C(z)=(count(a->z)+1)/(count(a->*)+12).
    Score(z)=0.5*G(z)+0.5*C(z); ties follow ``ZODIAC_ORDER``.
    Every invocation uses only its supplied history and returns exactly three.
    """

    def probabilities(self, history):
        animals = _clean_animals(history)
        counts = Counter(animals)
        last = animals[-1] if animals else None
        transitions = Counter(
            current for previous, current in zip(animals, animals[1:])
            if previous == last
        )
        marginal_denominator = len(animals) + len(ZODIAC_ORDER)
        conditional_denominator = sum(transitions.values()) + len(ZODIAC_ORDER)
        result = {}
        for animal in ZODIAC_ORDER:
            marginal = (counts[animal] + 1) / marginal_denominator
            conditional = (transitions[animal] + 1) / conditional_denominator
            result[animal] = {
                "global_probability": marginal,
                "conditional_probability": conditional,
                "score": 0.5 * marginal + 0.5 * conditional,
            }
        return result

    def predict(self, history):
        probabilities = self.probabilities(history)
        return sorted(ZODIAC_ORDER, key=lambda animal: -probabilities[animal]["score"])[:3]


def _prepare_history(df, test_periods):
    if isinstance(test_periods, bool) or not isinstance(test_periods, Integral) or test_periods <= 0:
        raise ValueError("回测期数必须为正整数")
    if not isinstance(df, pd.DataFrame) or not {"date", "animal"}.issubset(df.columns):
        raise ValueError("历史数据必须包含date、animal列")
    if len(df) <= test_periods:
        raise ValueError("回测首期之前至少需要一条历史记录")
    history = df.copy(deep=True).reset_index(drop=True)
    try:
        dates = pd.to_datetime(history["date"], errors="raise").dt.normalize()
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("date列必须为有效日期") from error
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("日期必须有效、严格升序且不能重复")
    history["date"] = dates
    history["animal"] = _clean_animals(history["animal"])
    return history


def _summarize(records):
    streak = maximum = 0
    longest = []
    run_start = None
    for offset, row in enumerate(records):
        if row["hit"]:
            if streak:
                longest.append({
                    "length": streak, "from": records[run_start]["date"],
                    "through": records[offset - 1]["date"], "closed": True,
                })
            streak = 0
            run_start = None
        else:
            if not streak:
                run_start = offset
            streak += 1
            maximum = max(maximum, streak)
    if streak:
        longest.append({
            "length": streak, "from": records[run_start]["date"],
            "through": records[-1]["date"], "closed": False,
        })
    hits = sum(row["hit"] for row in records)
    return {
        "periods": len(records),
        "hits": hits,
        "misses": len(records) - hits,
        "hit_rate": hits / len(records) if records else 0.0,
        "max_miss_streak": maximum,
        "current_miss_streak": streak,
        "longest_miss_runs": [run for run in longest if run["length"] == maximum],
        "from": records[0]["date"] if records else None,
        "through": records[-1]["date"] if records else None,
    }


def build_independent_top3_report(df, test_periods=400):
    """Replay the final test_periods rows, predicting before seeing each outcome.

    Dates define chronological order; streaks count supplied records, not days.
    ``next_date`` is the next calendar day, without asserting a draw schedule.
    A replay of previously examined data is not a pristine prospective test.
    """
    history = _prepare_history(df, test_periods)
    animals = history["animal"].tolist()
    dates = history["date"]
    first_index = len(history) - int(test_periods)
    predictor = IndependentTop3Predictor()
    records = []
    for index in range(first_index, len(history)):
        scores = predictor.probabilities(animals[:index])
        prediction = sorted(ZODIAC_ORDER, key=lambda animal: -scores[animal]["score"])[:3]
        records.append({
            "index": index,
            "date": dates.iloc[index].strftime("%Y-%m-%d"),
            "history_count": index,
            "actual": animals[index],
            "prediction": prediction,
            "scores": scores,
            "hit": animals[index] in prediction,
        })
    next_scores = predictor.probabilities(animals)
    next_prediction = sorted(ZODIAC_ORDER, key=lambda animal: -next_scores[animal]["score"])[:3]
    return {
        "scheme_id": "independent_all_history_equal_marginal_markov_top3_v1",
        "scheme_label": "独立TOP3：全历史频率与一阶转移各占一半",
        "protocol": {
            "top_n": 3,
            "history": "all_prior_rows",
            "laplace_pseudocount": 1,
            "global_weight": 0.5,
            "conditional_weight": 0.5,
            "tie_order": list(ZODIAC_ORDER),
            "parameter_search": False,
            "skip_periods": False,
        },
        "total_history": len(history),
        "test_periods": int(test_periods),
        "first_test_index": first_index,
        "summary": _summarize(records),
        "blocks": [_summarize(records[offset:offset + 100]) for offset in range(0, len(records), 100)],
        "records": records,
        "next_date": (dates.iloc[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "next_prediction": next_prediction,
        "next_scores": next_scores,
        "limitations": [
            "固定规则不进行窗口或权重搜索，但仍使用历史频数做在线概率估计。",
            "历史回放不能保证未来命中率，也不能作为未接触数据的盲测。",
            "连续不中按所提供的记录计数；下一日期只是最后记录的下一自然日。",
        ],
    }
