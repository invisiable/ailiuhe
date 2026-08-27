import unittest

import pandas as pd

from top5_top10_balanced19 import (
    get_zodiac_number_map,
    run_top5_top10_balanced19_audit,
)
from zodiac_top10_predictor import ZodiacTop10Predictor


class Top5Top10Balanced19Test(unittest.TestCase):
    def test_mapping_switches_on_2026_new_year_boundary(self):
        snake_year = get_zodiac_number_map('2026-02-16')
        horse_year = get_zodiac_number_map('2026-02-17')

        self.assertEqual(snake_year['蛇'], [1, 13, 25, 37, 49])
        self.assertEqual(snake_year['马'], [12, 24, 36, 48])
        self.assertEqual(horse_year['马'], [1, 13, 25, 37, 49])
        self.assertEqual(horse_year['蛇'], [2, 14, 26, 38])
        for mapping in (snake_year, horse_year):
            numbers = [number for values in mapping.values() for number in values]
            self.assertEqual(sorted(numbers), list(range(1, 50)))
            self.assertEqual(len(numbers), len(set(numbers)))

    def test_top10_can_predict_directly_from_animal_history(self):
        predictor = ZodiacTop10Predictor()
        animals = ['鼠', '牛', '虎', '兔', '龙'] * 10

        prediction = predictor.predict_from_animals(animals, top_n=10)

        self.assertEqual(len(prediction), 10)
        self.assertEqual(len(set(prediction)), 10)
        self.assertTrue(set(prediction).issubset(set(predictor.zodiac_cycle)))

    def test_real_400_period_balanced_regression(self):
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

        result = run_top5_top10_balanced19_audit(df, test_periods=400)

        self.assertEqual(result['reduced']['hits'], 198)
        self.assertEqual(result['reduced']['max_miss_streak'], 5)
        self.assertEqual(result['reduced']['average_count'], 19.0)
        self.assertEqual(result['reduced']['last_100_hits'], 53)
        self.assertEqual(result['full_top5']['hits'], 204)
        self.assertEqual(result['full_top5']['max_miss_streak'], 5)
        self.assertAlmostEqual(result['full_top5']['average_count'], 20.2425)
        self.assertTrue(
            all(len(row['prediction']) == 19 for row in result['rows'])
        )
        self.assertTrue(
            all(set(row['prediction']).issubset(row['full_top5_numbers'])
                for row in result['rows'])
        )

    def test_real_400_period_fixed18_meets_union_constraints(self):
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

        result = run_top5_top10_balanced19_audit(
            df,
            test_periods=400,
            target_count=18,
        )

        self.assertEqual(result['reduced']['hits'], 188)
        self.assertEqual(result['reduced']['max_miss_streak'], 6)
        self.assertEqual(result['reduced']['average_count'], 18.0)
        self.assertEqual(result['reduced']['last_100_hits'], 50)
        self.assertTrue(
            all(len(row['prediction']) == 18 for row in result['rows'])
        )


if __name__ == '__main__':
    unittest.main()
