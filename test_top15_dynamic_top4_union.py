import unittest
from unittest.mock import patch

from distill_top4_confidence_predictor import NUM_TO_ZODIAC_2026, ZODIAC_NUMS_2026
from top15_dynamic_top4_union import run_top15_dynamic_top4_union


class FakeDynamicPredictor:
    instances = []

    def __init__(self):
        self.predict_calls = []
        self.update_calls = []
        FakeDynamicPredictor.instances.append(self)

    def predict(self, history, k=15):
        self.predict_calls.append(([int(number) for number in history], k))
        return list(range(1, 16))

    def update_performance(self, prediction, actual):
        self.update_calls.append((list(prediction), int(actual)))


class FakeTop4Predictor:
    instances = []

    def __init__(self):
        self.predict_calls = []
        self.record_calls = []
        FakeTop4Predictor.instances.append(self)

    def predict_with_details(self, history, top_n=4):
        self.predict_calls.append(([int(number) for number in history], top_n))
        zodiacs = ['马', '蛇', '龙', '兔']
        return zodiacs, '测试模式', {}, 0.1, []

    def record_result(self, hit):
        self.record_calls.append(bool(hit))


class Top15DynamicTop4UnionTest(unittest.TestCase):
    def setUp(self):
        FakeDynamicPredictor.instances = []
        FakeTop4Predictor.instances = []

    @patch('top15_dynamic_top4_union.PreciseTop15Predictor', FakeDynamicPredictor)
    @patch('top15_dynamic_top4_union.DistillTop4ConfidencePredictor', FakeTop4Predictor)
    def test_predictions_only_use_prior_history_and_update_independently(self):
        numbers = [((index * 7) % 49) + 1 for index in range(35)]
        result = run_top15_dynamic_top4_union(numbers, test_periods=5)
        dynamic = FakeDynamicPredictor.instances[0]
        top4 = FakeTop4Predictor.instances[0]

        for offset, row in enumerate(result['rows']):
            index = row['index']
            window = row['dynamic_window']
            self.assertEqual(dynamic.predict_calls[offset], (numbers[index - window:index], 15))
            self.assertEqual(top4.predict_calls[offset], (numbers[:index], 4))
            self.assertEqual(dynamic.update_calls[offset][1], numbers[index])
            expected_top4_hit = NUM_TO_ZODIAC_2026[numbers[index]] in row['top4_zodiacs']
            self.assertEqual(top4.record_calls[offset], expected_top4_hit)

        self.assertEqual(dynamic.predict_calls[-1], (numbers[-result['next_dynamic_window']:], 15))
        self.assertEqual(top4.predict_calls[-1], (numbers, 4))

    @patch('top15_dynamic_top4_union.PreciseTop15Predictor', FakeDynamicPredictor)
    @patch('top15_dynamic_top4_union.DistillTop4ConfidencePredictor', FakeTop4Predictor)
    def test_union_rows_and_summary_are_consistent(self):
        numbers = [((index * 11) % 49) + 1 for index in range(35)]
        result = run_top15_dynamic_top4_union(numbers, test_periods=5)
        rows = result['rows']

        for row in rows:
            expected_top4_numbers = sorted({
                number
                for zodiac in row['top4_zodiacs']
                for number in ZODIAC_NUMS_2026[zodiac]
            })
            expected_union = list(row['dynamic_prediction'])
            expected_union.extend(number for number in expected_top4_numbers if number not in expected_union)
            self.assertEqual(row['top4_numbers'], expected_top4_numbers)
            self.assertEqual(row['union_prediction'], expected_union)
            self.assertEqual(row['added_count'], len(expected_union) - 15)
            self.assertEqual(row['union_hit'], row['actual'] in expected_union)

        self.assertEqual(result['dynamic_hits'], sum(row['dynamic_hit'] for row in rows))
        self.assertEqual(result['top4_hits'], sum(row['top4_hit'] for row in rows))
        self.assertEqual(result['union_hits'], sum(row['union_hit'] for row in rows))
        self.assertEqual(result['max_miss_streak'], max(row['union_miss_streak'] for row in rows))
        self.assertEqual(
            sum(length * count for length, count in result['miss_streak_distribution'].items()),
            len(rows) - result['union_hits'],
        )
        self.assertEqual(len(result['next_union_prediction']), len(set(result['next_union_prediction'])))


if __name__ == '__main__':
    unittest.main()
