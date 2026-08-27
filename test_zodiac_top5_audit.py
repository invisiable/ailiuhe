import unittest
from unittest.mock import patch

from zodiac_simple_smart import ZodiacSimpleSmart
from zodiac_top5_audit import run_zodiac_top5_audit


class FakeZodiacPredictor:
    instances = []

    def __init__(self):
        self.zodiacs = ['鸡', '狗', '猪', '鼠', '牛', '虎', '羊']
        self.calls = []
        FakeZodiacPredictor.instances.append(self)

    def predict_from_history(
        self, animals, top_n=5, debug=False, selected_miss_streaks=None
    ):
        self.calls.append((
            list(animals), top_n, debug, dict(selected_miss_streaks or {})
        ))
        selected = ['鸡', '狗', '猪', '鼠', '牛']
        if (selected_miss_streaks or {}).get('鸡', 0) >= 4:
            selected = ['狗', '猪', '鼠', '牛', '虎']
        return {
            'top5': selected,
            'selected_model': '反馈测试模型',
            'feedback_penalties': {
                zodiac: max(0, streak - 3) * 1.5
                for zodiac, streak in (selected_miss_streaks or {}).items()
            },
        }


class ZodiacTop5AuditTest(unittest.TestCase):
    def setUp(self):
        FakeZodiacPredictor.instances = []

    def test_real_model_caps_long_absence_and_selection_feedback(self):
        predictor = ZodiacSimpleSmart()
        history = ['鸡'] + ['鼠'] * 31

        result = predictor.predict_from_history(
            history,
            top_n=5,
            selected_miss_streaks={'鸡': 4},
        )

        self.assertEqual(result['feedback_penalties']['鸡'], 2.0)
        self.assertEqual(result['long_absence_penalties']['鸡'], 3.0)

        capped = predictor.predict_from_history(
            ['鸡'] + ['鼠'] * 100,
            top_n=5,
            selected_miss_streaks={'鸡': 100},
        )
        self.assertEqual(capped['feedback_penalties']['鸡'], 24.0)
        self.assertEqual(capped['long_absence_penalties']['鸡'], 42.0)

    @patch('zodiac_top5_audit.ZodiacSimpleSmart', FakeZodiacPredictor)
    def test_walk_forward_feedback_uses_only_previous_results(self):
        animals = ['羊'] * 30 + ['狗', '狗', '狗', '狗', '鸡', '羊']

        result = run_zodiac_top5_audit(animals, test_periods=6)
        predictor = FakeZodiacPredictor.instances[0]

        self.assertEqual(len(result['rows']), 6)
        for call_index, call in enumerate(predictor.calls[:-1], start=1):
            self.assertEqual(call[0], animals[:call_index])
            self.assertEqual(call[1:3], (5, False))
        for row in result['rows']:
            index = row['index']
            self.assertEqual(predictor.calls[index - 1][0], animals[:index])
            if predictor.calls[index - 1][3].get('鸡', 0) >= 4:
                self.assertNotIn('鸡', row['prediction'])
        self.assertEqual(predictor.calls[-1][0], animals)

    @patch('zodiac_top5_audit.ZodiacSimpleSmart', FakeZodiacPredictor)
    def test_reports_predictions_and_longest_miss_ranges(self):
        animals = ['羊'] * 30 + ['狗', '羊', '羊', '狗', '羊', '羊']

        result = run_zodiac_top5_audit(animals, test_periods=6)

        self.assertEqual(result['hits'], 2)
        self.assertEqual(result['max_miss_streak'], 2)
        self.assertEqual(result['current_miss_streak'], 2)
        self.assertEqual(result['miss_streak_distribution'], {2: 2})
        self.assertEqual(
            result['longest_miss_streaks'],
            [
                {'start_index': 31, 'end_index': 32, 'length': 2},
                {'start_index': 34, 'end_index': 35, 'length': 2},
            ],
        )
        self.assertEqual(
            [row['miss_streak'] for row in result['rows']],
            [0, 1, 2, 0, 1, 2],
        )


if __name__ == '__main__':
    unittest.main()
