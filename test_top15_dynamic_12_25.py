import csv
import unittest

from top15_dynamic_12_25 import run_top15_dynamic_12_25


class Top15Dynamic1225Test(unittest.TestCase):
    def test_matches_miniprogram_legacy_algorithm(self):
        with open('data/lucky_numbers.csv', encoding='utf-8-sig', newline='') as data_file:
            numbers = [int(row['number']) for row in csv.DictReader(data_file)]
        result = run_top15_dynamic_12_25(numbers)

        self.assertEqual(result['hits'], 147)
        self.assertEqual(result['test_periods'], 400)
        self.assertEqual(result['max_miss_streak'], 9)
        self.assertEqual(result['current_miss_streak'], 2)
        self.assertEqual(result['window'], 12)
        self.assertEqual(
            result['next_prediction'],
            [7, 8, 1, 2, 4, 6, 39, 42, 9, 10, 43, 41, 26, 44, 36],
        )


if __name__ == '__main__':
    unittest.main()