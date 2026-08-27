import unittest
from unittest.mock import patch

from top20_zodiac_top5_union import run_top20_zodiac_top5_union


class FakeTop20Predictor:
    instances = []

    def __init__(self):
        self.calls = []
        FakeTop20Predictor.instances.append(self)

    def predict(self, history, top_n=20):
        self.calls.append(([int(number) for number in history], top_n))
        return list(range(1, 21))


class FakeZodiacTop5Predictor:
    instances = []

    def __init__(self):
        self.calls = []
        self.zodiac_numbers = {
            '甲': [20, 21],
            '乙': [22],
            '丙': [23],
            '丁': [24],
            '戊': [25],
        }
        FakeZodiacTop5Predictor.instances.append(self)

    def predict_from_history(self, animals, top_n=5, debug=False):
        self.calls.append((list(animals), top_n, debug))
        return {
            'top5': ['甲', '乙', '丙', '丁', '戊'],
            'selected_model': '假TOP5模型',
        }


class Top20ZodiacTop5UnionTest(unittest.TestCase):
    def setUp(self):
        FakeTop20Predictor.instances = []
        FakeZodiacTop5Predictor.instances = []

    @patch('top20_zodiac_top5_union.Top20DualHorizonPredictor', FakeTop20Predictor)
    @patch('top20_zodiac_top5_union.ZodiacSimpleSmart', FakeZodiacTop5Predictor)
    def test_uses_prior_history_and_predictor_owned_mapping(self):
        numbers = [((index * 7) % 49) + 1 for index in range(430)]
        animals = [f'历史{index}' for index in range(430)]

        result = run_top20_zodiac_top5_union(numbers, animals, test_periods=400)
        top20 = FakeTop20Predictor.instances[0]
        zodiac = FakeZodiacTop5Predictor.instances[0]

        self.assertEqual(len(result['rows']), 400)
        self.assertEqual(len(top20.calls), 401)
        self.assertEqual(len(zodiac.calls), 401)
        for offset, row in enumerate(result['rows']):
            index = row['index']
            self.assertEqual(top20.calls[offset], (numbers[:index], 20))
            self.assertEqual(zodiac.calls[offset], (animals[:index], 5, False))
            self.assertEqual(row['zodiac_numbers'], [20, 21, 22, 23, 24, 25])
            self.assertEqual(row['union_prediction'], list(range(1, 26)))
            self.assertEqual(row['added_count'], 5)
            self.assertEqual(row['union_hit'], row['actual'] in row['union_prediction'])

        self.assertEqual(top20.calls[-1], (numbers, 20))
        self.assertEqual(zodiac.calls[-1], (animals, 5, False))
        self.assertEqual(result['next_union_prediction'], list(range(1, 26)))

    @patch('top20_zodiac_top5_union.Top20DualHorizonPredictor', FakeTop20Predictor)
    @patch('top20_zodiac_top5_union.ZodiacSimpleSmart', FakeZodiacTop5Predictor)
    def test_reports_all_longest_miss_streak_ranges(self):
        evaluated = [1] * 400
        evaluated[10:14] = [49] * 4
        evaluated[100:104] = [48] * 4
        numbers = [1] * 30 + evaluated
        animals = ['甲'] * len(numbers)

        result = run_top20_zodiac_top5_union(numbers, animals, test_periods=400)

        self.assertEqual(result['union_hits'], 392)
        self.assertEqual(result['max_miss_streak'], 4)
        self.assertEqual(result['current_miss_streak'], 0)
        self.assertEqual(result['miss_streak_distribution'], {4: 2})
        self.assertEqual(
            result['longest_miss_streaks'],
            [
                {'start_index': 40, 'end_index': 43, 'length': 4},
                {'start_index': 130, 'end_index': 133, 'length': 4},
            ],
        )
        self.assertEqual(
            [row['union_miss_streak'] for row in result['rows'][9:15]],
            [0, 1, 2, 3, 4, 0],
        )


if __name__ == '__main__':
    unittest.main()
