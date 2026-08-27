import unittest
from unittest.mock import patch

import pandas as pd

from top15_union_backtest import (
    apply_zodiac_filter,
    build_slim_protected_union,
    ordered_union,
    run_top15_union_backtest,
    summarize_completed_miss_probabilities,
    summarize_miss_streaks,
)


class FakePredictor:
    instances = []

    def __init__(self):
        self.predict_calls = []
        self.update_calls = []
        FakePredictor.instances.append(self)

    def predict(self, history, k=15):
        values = [int(number) for number in history]
        self.predict_calls.append(values)
        offset = len(FakePredictor.instances)
        return [((len(values) + offset + step - 1) % 49) + 1 for step in range(k)]

    def update_performance(self, prediction, actual):
        self.update_calls.append((list(prediction), int(actual)))


class FakeZodiacPredictor:
    instances = []

    def __init__(self):
        self.predict_calls = []
        self.top_n_calls = []
        self.record_calls = []
        FakeZodiacPredictor.instances.append(self)

    def predict(self, history, top_n=10):
        self.predict_calls.append([int(number) for number in history])
        self.top_n_calls.append(top_n)
        return ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡']

    def record_result(self, hit):
        self.record_calls.append(bool(hit))


class FakeTop10Predictor(FakeZodiacPredictor):
    instances = []

    def __init__(self):
        super().__init__()
        FakeZodiacPredictor.instances.pop()
        FakeTop10Predictor.instances.append(self)


class Top15UnionBacktestTest(unittest.TestCase):
    def setUp(self):
        FakePredictor.instances = []
        FakeZodiacPredictor.instances = []
        FakeTop10Predictor.instances = []

    def test_ordered_union_preserves_dynamic_order_and_appends_new_numbers(self):
        self.assertEqual(ordered_union([8, 3, 21, 5], [3, 13, 8, 34]), [8, 3, 21, 5, 13, 34])

    def test_zodiac_filter_can_protect_dynamic_numbers(self):
        union = [1, 2, 3, 4, 5]
        number_to_zodiac = {1: '马', 2: '蛇', 3: '龙', 4: '兔', 5: '虎'}
        selected_zodiacs = ['马', '龙', '虎']

        self.assertEqual(
            apply_zodiac_filter(union, selected_zodiacs, number_to_zodiac),
            [1, 3, 5],
        )
        self.assertEqual(
            apply_zodiac_filter(
                union,
                selected_zodiacs,
                number_to_zodiac,
                protected_numbers=[1, 2, 3],
            ),
            [1, 2, 3, 5],
        )

    def test_slim_union_protects_first_five_new_fixed_numbers(self):
        number_to_zodiac = {
            1: '鼠', 2: '牛', 3: '虎', 4: '兔', 5: '龙',
            6: '蛇', 7: '马', 8: '羊', 9: '猴', 10: '鸡',
        }

        prediction = build_slim_protected_union(
            dynamic_prediction=[1, 2, 3],
            fixed_prediction=[2, 4, 5, 6, 7, 8, 9, 10],
            selected_zodiacs=['猴'],
            number_to_zodiac=number_to_zodiac,
        )

        self.assertEqual(prediction, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_summarizes_completed_and_trailing_miss_streaks(self):
        self.assertEqual(
            summarize_miss_streaks([False, False, True, False, True, False, False]),
            {
                'distribution': {1: 1, 2: 2},
                'current_miss_streak': 2,
                'max_miss_streak': 2,
            },
        )

    def test_summarizes_no_misses(self):
        self.assertEqual(
            summarize_miss_streaks([True, True]),
            {
                'distribution': {},
                'current_miss_streak': 0,
                'max_miss_streak': 0,
            },
        )

    def test_summarizes_completed_miss_streak_probabilities(self):
        summary = summarize_completed_miss_probabilities(
            [False, True, False, False, True, False, False, False]
        )
        self.assertEqual(summary['completed_distribution'], {1: 1, 2: 1})
        self.assertEqual(summary['completed_streaks'], 2)
        self.assertEqual(summary['current_miss_streak'], 3)
        self.assertEqual(summary['exact_probabilities'], {1: 0.5, 2: 0.5})
        self.assertEqual(summary['at_least_probabilities'], {1: 1.0, 2: 0.5})

    def test_exact_probabilities_include_zero_frequency_lengths(self):
        summary = summarize_completed_miss_probabilities(
            [False, True, False, False, False, True]
        )
        self.assertEqual(
            summary['exact_probabilities'],
            {1: 0.5, 2: 0.0, 3: 0.5},
        )

    def test_rejects_test_periods_that_truncate_to_zero(self):
        numbers = [((index * 7) % 49) + 1 for index in range(30)]
        with self.assertRaisesRegex(ValueError, "回测期数必须大于 0"):
            run_top15_union_backtest(numbers, test_periods=0.5)

    @patch('top15_union_backtest.PreciseTop15Predictor', FakePredictor)
    def test_backtest_union_and_totals_are_consistent(self):
        numbers = [((index * 7) % 49) + 1 for index in range(70)]
        result = run_top15_union_backtest(numbers, test_periods=20)

        self.assertEqual(result['test_periods'], 20)
        self.assertEqual(len(result['rows']), 20)
        self.assertEqual(len(FakePredictor.instances), 2)

        for row in result['rows']:
            expected_union = ordered_union(row['dynamic_prediction'], row['optimal_prediction'])
            self.assertEqual(row['union_prediction'], expected_union)
            self.assertEqual(row['added_count'], len(expected_union) - len(row['dynamic_prediction']))
            self.assertEqual(row['hit'], row['actual'] in expected_union)
            self.assertIn(row['dynamic_window'], (12, 25))
            self.assertEqual(row['actual'], numbers[row['index']])
            self.assertTrue(set(row['top10_hard_prediction']).issubset(expected_union))
            self.assertTrue(set(row['top10_protected_prediction']).issubset(expected_union))
            self.assertTrue(set(row['dynamic_prediction']).issubset(row['top10_protected_prediction']))
            self.assertTrue(set(row['dynamic_prediction']).issubset(row['slim_prediction']))
            self.assertTrue(set(row['slim_prediction']).issubset(expected_union))
            self.assertEqual(row['slim_hit'], row['actual'] in row['slim_prediction'])

        added_counts = [row['added_count'] for row in result['rows']]
        self.assertEqual(result['added_total'], sum(added_counts))
        self.assertAlmostEqual(result['added_average'], sum(added_counts) / len(added_counts))
        self.assertEqual(result['added_min'], min(added_counts))
        self.assertEqual(result['added_max'], max(added_counts))
        self.assertEqual(result['union_hits'], sum(row['hit'] for row in result['rows']))
        self.assertEqual(result['dynamic_hits'], sum(row['actual'] in row['dynamic_prediction'] for row in result['rows']))
        self.assertEqual(result['optimal_hits'], sum(row['actual'] in row['optimal_prediction'] for row in result['rows']))
        top10_filter = result['top10_filter']
        self.assertEqual(
            top10_filter['hard_lost_hits'],
            result['union_hits'] - top10_filter['hard_hits'],
        )
        self.assertEqual(
            top10_filter['protected_lost_hits'],
            result['union_hits'] - top10_filter['protected_hits'],
        )
        self.assertGreaterEqual(top10_filter['hard_max_miss_streak'], 0)
        self.assertGreaterEqual(top10_filter['protected_max_miss_streak'], 0)
        self.assertTrue(set(top10_filter['next_hard_prediction']).issubset(result['next_union_prediction']))
        self.assertEqual(
            result['next_union_prediction'],
            ordered_union(result['next_dynamic_prediction'], result['next_optimal_prediction']),
        )
        self.assertEqual(
            result['slim_filter']['hits'],
            sum(row['slim_hit'] for row in result['rows']),
        )
        self.assertEqual(
            result['slim_filter']['average_count'],
            sum(row['slim_count'] for row in result['rows']) / result['test_periods'],
        )
        self.assertTrue(
            set(result['slim_filter']['next_prediction']).issubset(
                result['next_union_prediction']
            )
        )

    @patch('top15_union_backtest.PreciseTop15Predictor', FakePredictor)
    @patch('top15_union_backtest.ZodiacTop10Predictor', FakeTop10Predictor)
    def test_each_prediction_uses_only_history_before_the_actual(self):
        numbers = [((index * 11) % 49) + 1 for index in range(60)]
        result = run_top15_union_backtest(numbers, test_periods=10)
        dynamic_predictor, optimal_predictor = FakePredictor.instances
        top10_predictor = FakeTop10Predictor.instances[0]

        for call_index, row in enumerate(result['rows']):
            index = row['index']
            dynamic_window = row['dynamic_window']
            self.assertEqual(dynamic_predictor.predict_calls[call_index], numbers[index - dynamic_window:index])
            self.assertEqual(optimal_predictor.predict_calls[call_index], numbers[index - 25:index])
            self.assertEqual(dynamic_predictor.update_calls[call_index][1], numbers[index])
            self.assertEqual(optimal_predictor.update_calls[call_index][1], numbers[index])
            self.assertEqual(top10_predictor.predict_calls[call_index], numbers[:index])

        self.assertEqual(dynamic_predictor.predict_calls[-1], numbers[-result['next_dynamic_window']:])
        self.assertEqual(optimal_predictor.predict_calls[-1], numbers[-25:])
        self.assertEqual(top10_predictor.predict_calls[-1], numbers)
        self.assertEqual(len(top10_predictor.record_calls), result['test_periods'])
        self.assertTrue(all(top_n == 10 for top_n in top10_predictor.top_n_calls))

    def test_real_400_period_slim_filter_preserves_union_constraints(self):
        numbers = pd.read_csv(
            'data/lucky_numbers.csv',
            encoding='utf-8-sig',
        )['number'].astype(int).tolist()

        result = run_top15_union_backtest(numbers, test_periods=400)
        slim = result['slim_filter']

        self.assertEqual(result['union_hits'], 181)
        self.assertEqual(result['max_miss_streak'], 9)
        self.assertEqual(slim['hits'], 181)
        self.assertEqual(slim['max_miss_streak'], 9)
        self.assertLess(slim['average_count'], 18.765)


if __name__ == '__main__':
    unittest.main()
