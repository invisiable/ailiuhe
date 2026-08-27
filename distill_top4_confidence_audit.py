"""蒸馏TOP4置信度模型的无前视滚动审计。"""
from collections import Counter

from distill_top4_confidence_predictor import (
    DistillTop4ConfidencePredictor,
    NUM_TO_ZODIAC_2026,
)


def _longest_miss_ranges(rows):
    ranges = []
    streak_start = None
    for position, row in enumerate(rows):
        if not row['hit'] and streak_start is None:
            streak_start = position
        if row['hit'] and streak_start is not None:
            ranges.append((streak_start, position - 1))
            streak_start = None
    if streak_start is not None:
        ranges.append((streak_start, len(rows) - 1))

    max_length = max((end - start + 1 for start, end in ranges), default=0)
    longest = [
        {
            'start_index': rows[start]['index'],
            'end_index': rows[end]['index'],
            'length': end - start + 1,
        }
        for start, end in ranges
        if end - start + 1 == max_length
    ]
    return max_length, longest, [end - start + 1 for start, end in ranges]


def _segment_stats(rows, size):
    segments = []
    for start in range(0, len(rows), size):
        segment = rows[start:start + size]
        hits = sum(row['hit'] for row in segment)
        segments.append({
            'start_period': start + 1,
            'end_period': start + len(segment),
            'periods': len(segment),
            'hits': hits,
            'hit_rate': hits / len(segment) * 100,
        })
    return segments


def run_distill_top4_confidence_audit(numbers, test_periods=400):
    """逐期预测并在开奖后更新状态，返回固定TOP4审计结果。"""
    numbers = [int(number) for number in numbers]
    if len(numbers) < 21:
        raise ValueError('至少需要21期数据')

    periods = min(int(test_periods), len(numbers) - 20)
    if periods <= 0:
        raise ValueError('test_periods必须大于0')

    start_index = len(numbers) - periods
    predictor = DistillTop4ConfidencePredictor()
    rows = []
    miss_streak = 0

    for index in range(20, len(numbers)):
        predicted, mode, scores, confidence, excluded = predictor.predict_with_details(
            numbers[:index], top_n=4
        )
        actual_zodiac = NUM_TO_ZODIAC_2026[numbers[index]]
        hit = actual_zodiac in predicted
        predictor.record_result(hit)

        if index < start_index:
            continue

        miss_streak = 0 if hit else miss_streak + 1
        rows.append({
            'index': index,
            'actual_number': numbers[index],
            'actual_zodiac': actual_zodiac,
            'prediction': predicted,
            'mode': mode,
            'scores': scores,
            'confidence': confidence,
            'excluded': excluded,
            'hit': hit,
            'miss_streak': miss_streak,
        })

    max_miss_streak, longest_ranges, miss_streaks = _longest_miss_ranges(rows)
    hits = sum(row['hit'] for row in rows)
    mode_counts = Counter(row['mode'].split('(')[0] for row in rows)
    mode_hits = Counter(
        row['mode'].split('(')[0] for row in rows if row['hit']
    )
    mode_stats = {
        mode: {
            'periods': count,
            'hits': mode_hits[mode],
            'hit_rate': mode_hits[mode] / count * 100,
        }
        for mode, count in mode_counts.items()
    }
    next_prediction, next_mode, next_scores, next_confidence, next_excluded = (
        predictor.predict_with_details(numbers, top_n=4)
    )

    return {
        'rows': rows,
        'periods': periods,
        'hits': hits,
        'hit_rate': hits / periods * 100,
        'max_miss_streak': max_miss_streak,
        'current_miss_streak': rows[-1]['miss_streak'] if rows else 0,
        'miss_streak_distribution': dict(sorted(Counter(miss_streaks).items())),
        'longest_miss_streaks': longest_ranges,
        'segments_50': _segment_stats(rows, 50),
        'segments_100': _segment_stats(rows, 100),
        'mode_stats': mode_stats,
        'next_prediction': next_prediction,
        'next_mode': next_mode,
        'next_scores': next_scores,
        'next_confidence': next_confidence,
        'next_excluded': next_excluded,
    }