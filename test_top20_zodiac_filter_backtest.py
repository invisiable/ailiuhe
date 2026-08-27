import unittest
from unittest.mock import patch

from top20_zodiac_filter_backtest import run_top20_zodiac_filter_backtest
from zodiac_top10_predictor import (
    NUM_TO_ZODIAC_2026,
    ZODIAC_CYCLE_2026,
    ZodiacTop10Predictor,
)


class FakeTop20Predictor:
    instances = []

    def __init__(self):
        self.calls = []
        FakeTop20Predictor.instances.append(self)

    def predict(self, history, top_n=20):
        self.calls.append(([int(number) for number in history], top_n))
        return list(range(1, 21))


class FakeZodiacPredictor:
    instances = []

    def __init__(self):
        self.events = []
        FakeZodiacPredictor.instances.append(self)

    def predict(self, history, top_n=10):
        self.events.append(('predict', [int(number) for number in history], top_n))
        return ZODIAC_CYCLE_2026[:10]

    def record_result(self, hit):
        self.events.append(('record', bool(hit)))


class Top20ZodiacFilterBacktestTest(unittest.TestCase):
    def setUp(self):
        FakeTop20Predictor.instances = []
        FakeZodiacPredictor.instances = []

    def test_zodiac_top10_expands_to_top11_after_three_misses(self):
        predictor = ZodiacTop10Predictor()
        history = list(range(1, 50))
        self.assertEqual(len(predictor.predict(history, top_n=10)), 10)
        predictor.record_result(False)
        predictor.record_result(False)
        predictor.record_result(False)
        self.assertEqual(len(predictor.predict(history, top_n=10)), 11)

    @patch('top20_zodiac_filter_backtest.ZodiacTop10Predictor', FakeZodiacPredictor)
    @patch('top20_zodiac_filter_backtest.Top20DualHorizonPredictor', FakeTop20Predictor)
    def test_uses_only_prior_history_and_updates_zodiac_after_prediction(self):
        numbers = [((index * 7) % 49) + 1 for index in range(35)]
        result = run_top20_zodiac_filter_backtest(numbers, test_periods=5)
        top20_predictor = FakeTop20Predictor.instances[0]
        zodiac_predictor = FakeZodiacPredictor.instances[0]

        for offset, row in enumerate(result['rows']):
            index = row['index']
            self.assertEqual(top20_predictor.calls[offset], (numbers[:index], 20))
            self.assertEqual(zodiac_predictor.events[offset * 2], ('predict', numbers[:index], 10))
            expected_zodiac_hit = NUM_TO_ZODIAC_2026[numbers[index]] in row['selected_zodiacs']
            self.assertEqual(zodiac_predictor.events[offset * 2 + 1], ('record', expected_zodiac_hit))

        self.assertEqual(top20_predictor.calls[-1], (numbers, 20))
        self.assertEqual(zodiac_predictor.events[-1], ('predict', numbers, 10))

    @patch('top20_zodiac_filter_backtest.ZodiacTop10Predictor', FakeZodiacPredictor)
    @patch('top20_zodiac_filter_backtest.Top20DualHorizonPredictor', FakeTop20Predictor)
    def test_rows_and_summary_are_consistent(self):
        numbers = [((index * 11) % 49) + 1 for index in range(35)]
        result = run_top20_zodiac_filter_backtest(numbers, test_periods=5)
        rows = result['rows']
        self.assertEqual(len(rows), 5)

        running_misses = 0
        for row in rows:
            expected_filtered = [
                number for number in row['top20_prediction']
                if NUM_TO_ZODIAC_2026[number] in row['selected_zodiacs']
            ]
            self.assertEqual(row['filtered_prediction'], expected_filtered)
            self.assertEqual(row['reduced_count'], 20 - len(expected_filtered))
            self.assertEqual(row['original_hit'], row['actual'] in row['top20_prediction'])
            self.assertEqual(row['filtered_hit'], row['actual'] in expected_filtered)
            running_misses = 0 if row['filtered_hit'] else running_misses + 1
            self.assertEqual(row['filtered_miss_streak'], running_misses)

        self.assertEqual(result['original_hits'], sum(row['original_hit'] for row in rows))
        self.assertEqual(result['filtered_hits'], sum(row['filtered_hit'] for row in rows))
        self.assertEqual(result['max_miss_streak'], max(row['filtered_miss_streak'] for row in rows))
        self.assertEqual(result['current_miss_streak'], rows[-1]['filtered_miss_streak'])
        self.assertEqual(
            sum(length * count for length, count in result['miss_streak_distribution'].items()),
            len(rows) - result['filtered_hits'],
        )
        self.assertEqual(
            result['next']['excluded_zodiacs'],
            [zodiac for zodiac in ZODIAC_CYCLE_2026 if zodiac not in result['next']['selected_zodiacs']],
        )


if __name__ == '__main__':
    unittest.main()
