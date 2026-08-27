import unittest

import numpy as np

from fused_top20_predictor import (
    _select_top_k,
    find_longest_miss_streaks,
    format_fused_top20_details,
)


class FusedTop20DetailsTest(unittest.TestCase):
    def test_select_top_k_uses_score_then_index_for_boundary_ties(self):
        scores = np.array(
            [100.0 - index for index in range(19)] + [0.0] * 30
        )

        self.assertEqual(
            _select_top_k(scores, top_n=20),
            list(range(1, 21)),
        )

    def test_finds_all_tied_longest_miss_ranges(self):
        records = [
            {'index': 10, 'date': '2026/1/1', 'hit': False},
            {'index': 11, 'date': '2026/1/2', 'hit': False},
            {'index': 12, 'date': '2026/1/3', 'hit': True},
            {'index': 13, 'date': '2026/1/4', 'hit': False},
            {'index': 14, 'date': '2026/1/5', 'hit': False},
        ]

        self.assertEqual(
            find_longest_miss_streaks(records),
            [
                {
                    'start_index': 10,
                    'end_index': 11,
                    'start_date': '2026/1/1',
                    'end_date': '2026/1/2',
                    'length': 2,
                },
                {
                    'start_index': 13,
                    'end_index': 14,
                    'start_date': '2026/1/4',
                    'end_date': '2026/1/5',
                    'length': 2,
                },
            ],
        )

    def test_formats_every_prediction_row_and_longest_ranges(self):
        report = {
            'overall': {'hits': 1, 'periods': 2, 'hit_rate': 50.0,
                        'max_miss_streak': 1, 'ending_miss_streak': 0},
            'longest_miss_streaks': [
                {'start_date': '2026/1/1', 'end_date': '2026/1/1', 'length': 1}
            ],
            'records': [
                {'date': '2026/1/1', 'actual': 7, 'prediction': [1, 2],
                 'hit': False, 'overlap': 10, 'dual_mode': '常态',
                 'dynamic_window': 12},
                {'date': '2026/1/2', 'actual': 2, 'prediction': [1, 2],
                 'hit': True, 'overlap': 11, 'dual_mode': '常态',
                 'dynamic_window': 25},
            ],
        }

        text = format_fused_top20_details(report)

        self.assertIn('最长连续不中：1期', text)
        self.assertIn('2026/1/1 至 2026/1/1', text)
        self.assertIn('001 2026/1/1', text)
        self.assertIn('002 2026/1/2', text)
        self.assertEqual(text.count('预测='), 2)


if __name__ == '__main__':
    unittest.main()
