"""固定18号双源模型的逐期无前视审计。"""

from collections import Counter

import pandas as pd

from dual_objective_top15_predictor import DualObjectiveTop15Predictor
from fixed_dual_source_top18_predictor import FixedDualSourceTop18Predictor


def _summary(rows, hit_key):
    hits = [bool(row[hit_key]) for row in rows]
    streak = 0
    maximum = 0
    distribution = Counter()
    ranges = []
    start = None
    for position, hit in enumerate(hits):
        if hit:
            if streak:
                distribution[streak] += 1
                ranges.append((start, position - 1, streak))
            streak = 0
            start = None
        else:
            if streak == 0:
                start = position
            streak += 1
            maximum = max(maximum, streak)
    if streak:
        distribution[streak] += 1
        ranges.append((start, len(rows) - 1, streak))

    longest = [
        {
            'start_index': rows[first]['index'],
            'end_index': rows[last]['index'],
            'length': length,
        }
        for first, last, length in ranges
        if length == maximum
    ]
    segments_50 = []
    for first in range(0, len(rows), 50):
        segment = hits[first:first + 50]
        segments_50.append({
            'start_period': first + 1,
            'end_period': first + len(segment),
            'hits': sum(segment),
            'periods': len(segment),
        })
    return {
        'hits': sum(hits),
        'hit_rate': sum(hits) / len(hits),
        'max_miss_streak': maximum,
        'current_miss_streak': streak,
        'miss_streak_distribution': dict(sorted(distribution.items())),
        'longest_miss_streaks': longest,
        'segments_50': segments_50,
        'first_200_hits': sum(hits[:200]),
        'last_200_hits': sum(hits[200:]),
    }


def run_fixed_dual_source_top18_audit(df, test_periods=400):
    required = {'date', 'number', 'animal'}
    if not required.issubset(df.columns):
        raise ValueError(f"数据必须包含列：{sorted(required)}")

    numbers = df['number'].astype(int).tolist()
    animals = df['animal'].astype(str).str.strip().tolist()
    periods = int(test_periods)
    if periods <= 0:
        raise ValueError('回测期数必须大于0')
    if len(df) < periods + 50:
        raise ValueError(f'完整回测{periods}期至少需要{periods + 50}期数据')

    dates = pd.to_datetime(df['date'], errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('日期必须严格按时间升序且不能重复')

    start_index = len(df) - periods
    model = FixedDualSourceTop18Predictor()
    baseline = DualObjectiveTop15Predictor()
    rows = []
    for index in range(50, len(df)):
        model_details = model.predict_with_details(
            numbers[:index], animals[:index]
        )
        baseline_details = baseline.predict_with_details(numbers[:index])
        actual_number = numbers[index]
        actual_animal = animals[index]
        model_hit = model.update(
            model_details['top18'], actual_number, actual_animal
        )
        baseline_hit = baseline.update(
            baseline_details['top15'], actual_number
        )
        if index < start_index:
            continue
        rows.append({
            'index': index,
            'date': str(df.iloc[index]['date']),
            'actual_number': actual_number,
            'actual_animal': actual_animal,
            'prediction': model_details['top18'],
            'hit': model_hit,
            'mode': model_details['mode'],
            'number_weight': model_details['number_weight'],
            'zodiac_weight': model_details['zodiac_weight'],
            'zodiac_top5': model_details['zodiac_top5'],
            'miss_streak': model.consecutive_misses,
            'baseline_prediction': baseline_details['top15'],
            'baseline_hit': baseline_hit,
            'baseline_miss_streak': baseline.consecutive_misses,
        })

    next_details = model.predict_with_details(numbers, animals)
    model_summary = _summary(rows, 'hit')
    baseline_summary = _summary(rows, 'baseline_hit')
    return {
        'periods': periods,
        'rows': rows,
        'model': model_summary,
        'baseline': baseline_summary,
        'next_prediction': next_details['top18'],
        'next_mode': next_details['mode'],
        'next_zodiac_top5': next_details['zodiac_top5'],
        'next_number_weight': next_details['number_weight'],
        'next_zodiac_weight': next_details['zodiac_weight'],
    }
