import unittest
from unittest.mock import patch

import numpy as np

from distill_top4_confidence_audit import run_distill_top4_confidence_audit
from distill_top4_confidence_predictor import DistillTop4ConfidencePredictor


class FakeDistillPredictor:
    instances = []

    def __init__(self):
        self.calls = []
        self.results = []
        self.consecutive_miss = 0
        FakeDistillPredictor.instances.append(self)

    def predict_with_details(self, numbers, top_n=4):
        self.calls.append((list(numbers), top_n, self.consecutive_miss))
        predicted = ['马', '蛇', '龙', '兔']
        return predicted, '测试模式', {}, 0.02, ['虎', '牛', '鼠']

    def record_result(self, hit):
        self.results.append(bool(hit))
        self.consecutive_miss = 0 if hit else self.consecutive_miss + 1


class DistillTop4ConfidenceOptimizationTest(unittest.TestCase):
    def setUp(self):
        FakeDistillPredictor.instances = []

    def test_gap_score_is_capped_at_eighteen_periods(self):
        predictor = DistillTop4ConfidencePredictor()
        animals = ['马'] + ['蛇'] * 40

        scores = predictor._gap_scores(animals)

        self.assertAlmostEqual(scores[0], 18 / 12)
        self.assertTrue(np.all(scores <= 18 / 12))

    def test_only_three_misses_uses_rescue_branch(self):
        predictor = DistillTop4ConfidencePredictor()
        numbers = list(range(1, 50))

        predictor.consecutive_miss = 2
        _, mode_two, _, _, _ = predictor.predict_with_details(numbers, top_n=4)
        predictor.consecutive_miss = 3
        _, mode_three, _, _, _ = predictor.predict_with_details(numbers, top_n=4)
        predictor.consecutive_miss = 4
        _, mode_four, _, _, _ = predictor.predict_with_details(numbers, top_n=4)

        self.assertNotIn('连败', mode_two)
        self.assertIn('连败3期→反miss', mode_three)
        self.assertNotIn('连败', mode_four)

    def test_predictor_always_returns_fixed_top4(self):
        predictor = DistillTop4ConfidencePredictor()

        predicted = predictor.predict(list(range(1, 50)), top_n=1)

        self.assertEqual(len(predicted), 4)
        self.assertEqual(len(set(predicted)), 4)

    @patch(
        'distill_top4_confidence_audit.DistillTop4ConfidencePredictor',
        FakeDistillPredictor,
    )
    def test_audit_uses_prior_numbers_and_reports_longest_ranges(self):
        numbers = [1] * 30 + [1, 5, 6, 2, 9, 10]

        result = run_distill_top4_confidence_audit(numbers, test_periods=6)
        predictor = FakeDistillPredictor.instances[0]

        self.assertEqual(len(result['rows']), 6)
        for row in result['rows']:
            call = predictor.calls[row['index'] - 20]
            self.assertEqual(call[0], numbers[:row['index']])
            self.assertEqual(call[1], 4)
        self.assertEqual(result['hits'], 2)
        self.assertTrue(all(len(row['prediction']) == 4 for row in result['rows']))
        self.assertEqual(result['max_miss_streak'], 2)
        self.assertEqual(
            result['longest_miss_streaks'],
            [
                {'start_index': 31, 'end_index': 32, 'length': 2},
                {'start_index': 34, 'end_index': 35, 'length': 2},
            ],
        )
        self.assertEqual(len(FakeDistillPredictor.instances), 1)
        self.assertEqual(predictor.calls[-1][0], numbers)
        self.assertEqual(predictor.calls[-1][2], 2)

    @patch(
        'distill_top4_confidence_audit.DistillTop4ConfidencePredictor',
        FakeDistillPredictor,
    )
    def test_audit_prewarms_state_before_display_window(self):
        numbers = [1] * 20 + [5] * 10 + [1] * 6

        run_distill_top4_confidence_audit(numbers, test_periods=6)
        predictor = FakeDistillPredictor.instances[0]

        first_display_call = predictor.calls[10]
        self.assertEqual(first_display_call[0], numbers[:30])
        self.assertEqual(first_display_call[2], 10)


if __name__ == '__main__':
    unittest.main()
