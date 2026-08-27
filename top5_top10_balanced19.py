"""TOP5候选池与TOP10共识评分结合的固定19号方案。"""

from collections import Counter

import numpy as np
import pandas as pd

from dual_objective_top15_predictor import (
    NORMAL_WEIGHTS,
    DualObjectiveTop15Predictor,
)
from zodiac_simple_smart import ZodiacSimpleSmart
from zodiac_top10_predictor import ZodiacTop10Predictor, ZODIAC_CYCLE_2026


TOP_N = 19
YEAR_SWITCH_DATE = pd.Timestamp('2026-02-17')
SNAKE_YEAR_CYCLE = ['蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊', '马']
HORSE_YEAR_CYCLE = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']


def get_zodiac_number_map(target_date):
    """返回目标日期所属年份的49号生肖映射。"""
    date = pd.Timestamp(target_date)
    cycle = HORSE_YEAR_CYCLE if date >= YEAR_SWITCH_DATE else SNAKE_YEAR_CYCLE
    return {
        zodiac: [
            number
            for number in range(1, 50)
            if cycle[(number - 1) % 12] == zodiac
        ]
        for zodiac in cycle
    }


def _normalize(scores):
    values = np.asarray(scores, dtype=float)
    spread = float(values.max() - values.min())
    if spread == 0:
        return np.zeros_like(values)
    return (values - values.min()) / spread


def _summary(rows, hit_key, count_key):
    hits = [bool(row[hit_key]) for row in rows]
    counts = [int(row[count_key]) for row in rows]
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

    return {
        'hits': sum(hits),
        'hit_rate': sum(hits) / len(hits),
        'max_miss_streak': maximum,
        'current_miss_streak': streak,
        'average_count': sum(counts) / len(counts),
        'min_count': min(counts),
        'max_count': max(counts),
        'last_100_hits': sum(hits[-100:]),
        'segments_100': [
            sum(hits[start:start + 100])
            for start in range(0, len(hits), 100)
        ],
        'miss_streak_distribution': dict(sorted(distribution.items())),
        'longest_miss_streaks': [
            {
                'start_index': rows[first]['index'],
                'end_index': rows[last]['index'],
                'length': length,
            }
            for first, last, length in ranges
            if length == maximum
        ],
    }


def _update_top5_feedback(predictor, selected_misses, prediction, actual):
    selected = set(prediction)
    for zodiac in predictor.zodiacs:
        if zodiac not in selected or zodiac == actual:
            selected_misses[zodiac] = 0
        else:
            selected_misses[zodiac] += 1


def run_top5_top10_balanced19_audit(df, test_periods=400, target_count=TOP_N):
    """逐期预热并审计TOP5池内固定数量方案。"""
    required = {'date', 'number', 'animal'}
    if not required.issubset(df.columns):
        raise ValueError(f"数据必须包含列：{sorted(required)}")

    periods = int(test_periods)
    if periods <= 0 or len(df) < periods + 50:
        raise ValueError(f'完整回测{periods}期至少需要{periods + 50}期数据')
    target_count = int(target_count)
    if target_count <= 0 or target_count > 49:
        raise ValueError('目标号码数量必须在1到49之间')
    dates = pd.to_datetime(df['date'], errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('日期必须严格按时间升序且不能重复')

    numbers = df['number'].astype(int).tolist()
    animals = df['animal'].astype(str).str.strip().tolist()
    start_index = len(df) - periods
    number_predictor = DualObjectiveTop15Predictor()
    top5_predictor = ZodiacSimpleSmart()
    top10_predictor = ZodiacTop10Predictor()
    selected_misses = {zodiac: 0 for zodiac in top5_predictor.zodiacs}
    rows = []
    last_details = None

    for index in range(1, len(df)):
        top5_result = top5_predictor.predict_from_history(
            animals[:index],
            top_n=5,
            debug=False,
            selected_miss_streaks=selected_misses,
        )
        top10_scores = top10_predictor.score_from_animals(animals[:index])
        top10_indexes = np.argsort(-top10_scores, kind='stable')
        top10_ranking = [ZODIAC_CYCLE_2026[item] for item in top10_indexes]

        if index >= 50:
            features, _ = number_predictor.build_features(numbers[:index])
            normal_scores = np.tensordot(NORMAL_WEIGHTS, features, axes=(0, 0))
            active_scores = (
                features[14]
                if number_predictor.consecutive_misses >= 5
                else normal_scores
            )
            base_top15 = number_predictor._select_top15(active_scores)

            zodiac_map = get_zodiac_number_map(dates.iloc[index])
            number_to_zodiac = {
                number: zodiac
                for zodiac, values in zodiac_map.items()
                for number in values
            }
            full_numbers = sorted({
                number
                for zodiac in top5_result['top5']
                for number in zodiac_map[zodiac]
            })
            normalized_numbers = _normalize(active_scores)
            normalized_top10 = dict(zip(
                ZODIAC_CYCLE_2026,
                _normalize(top10_scores),
            ))
            combined_scores = {
                number: (
                    0.5 * normalized_numbers[number - 1]
                    + 0.5 * normalized_top10[number_to_zodiac[number]]
                )
                for number in full_numbers
            }
            prediction = sorted(
                full_numbers,
                key=lambda number: (-combined_scores[number], number),
            )[:target_count]
            actual_number = numbers[index]
            reduced_hit = actual_number in prediction
            full_hit = actual_number in full_numbers
            last_details = {
                'prediction': prediction,
                'full_numbers': full_numbers,
                'top5': list(top5_result['top5']),
                'top10_ranking': top10_ranking,
            }

            if index >= start_index:
                rows.append({
                    'index': index,
                    'date': str(df.iloc[index]['date']),
                    'actual_number': actual_number,
                    'actual_animal': animals[index],
                    'prediction': prediction,
                    'prediction_count': len(prediction),
                    'hit': reduced_hit,
                    'full_top5_numbers': full_numbers,
                    'full_top5_count': len(full_numbers),
                    'full_top5_hit': full_hit,
                    'top5': list(top5_result['top5']),
                    'top10': top10_ranking[:10],
                })
            number_predictor.update(base_top15, actual_number)

        actual_animal = animals[index]
        _update_top5_feedback(
            top5_predictor,
            selected_misses,
            top5_result['top5'],
            actual_animal,
        )
        top10_predictor.record_result(actual_animal in top10_ranking[:10])

    next_date = dates.iloc[-1] + pd.Timedelta(days=1)
    top5_result = top5_predictor.predict_from_history(
        animals,
        top_n=5,
        debug=False,
        selected_miss_streaks=selected_misses,
    )
    top10_scores = top10_predictor.score_from_animals(animals)
    features, _ = number_predictor.build_features(numbers)
    normal_scores = np.tensordot(NORMAL_WEIGHTS, features, axes=(0, 0))
    active_scores = (
        features[14]
        if number_predictor.consecutive_misses >= 5
        else normal_scores
    )
    zodiac_map = get_zodiac_number_map(next_date)
    number_to_zodiac = {
        number: zodiac
        for zodiac, values in zodiac_map.items()
        for number in values
    }
    full_numbers = sorted({
        number
        for zodiac in top5_result['top5']
        for number in zodiac_map[zodiac]
    })
    normalized_top10 = dict(zip(ZODIAC_CYCLE_2026, _normalize(top10_scores)))
    normalized_numbers = _normalize(active_scores)
    next_prediction = sorted(
        full_numbers,
        key=lambda number: (
            -(
                0.5 * normalized_numbers[number - 1]
                + 0.5 * normalized_top10[number_to_zodiac[number]]
            ),
            number,
        ),
    )[:target_count]

    return {
        'periods': periods,
        'target_count': target_count,
        'rows': rows,
        'reduced': _summary(rows, 'hit', 'prediction_count'),
        'full_top5': _summary(rows, 'full_top5_hit', 'full_top5_count'),
        'next_prediction': next_prediction,
        'next_top5': list(top5_result['top5']),
        'next_top10': [
            ZODIAC_CYCLE_2026[index]
            for index in np.argsort(-top10_scores, kind='stable')[:10]
        ],
        'next_target_date': next_date.strftime('%Y-%m-%d'),
    }