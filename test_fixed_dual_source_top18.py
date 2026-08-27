import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fixed_dual_source_top18_predictor import FixedDualSourceTop18Predictor
from fixed_dual_source_top18_audit import run_fixed_dual_source_top18_audit


class FixedDualSourceTop18Test(unittest.TestCase):
    def test_normalize_handles_constant_scores(self):
        normalized = FixedDualSourceTop18Predictor._normalize(np.ones(49))

        self.assertTrue(np.all(np.isfinite(normalized)))
        self.assertTrue(np.all(normalized == 0))

    def test_always_selects_eighteen_unique_numbers(self):
        scores = np.arange(49, dtype=float)

        prediction = FixedDualSourceTop18Predictor._select_top_n(scores, 18)

        self.assertEqual(len(prediction), 18)
        self.assertEqual(len(set(prediction)), 18)
        self.assertTrue(all(1 <= number <= 49 for number in prediction))

    def test_cold_rescue_starts_after_five_misses_and_hit_restores_normal(self):
        predictor = FixedDualSourceTop18Predictor()
        predictor.consecutive_misses = 4
        self.assertEqual(
            predictor._current_blend(), (1.0, 0.0, '风险优先固定18')
        )

        predictor.consecutive_misses = 5
        self.assertEqual(
            predictor._current_blend(), (1.0, 0.0, '连不中5期冷号救援')
        )

        predictor.consecutive_misses = 6
        predictor._update_model_streak(True)
        self.assertEqual(predictor.consecutive_misses, 0)
        self.assertEqual(
            predictor._current_blend(), (1.0, 0.0, '风险优先固定18')
        )

    def test_real_400_period_regression(self):
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

        result = run_fixed_dual_source_top18_audit(df, test_periods=400)

        self.assertEqual(result['model']['hits'], 166)
        self.assertEqual(result['model']['max_miss_streak'], 7)
        self.assertEqual(result['baseline']['hits'], 152)
        self.assertEqual(result['baseline']['max_miss_streak'], 10)
        self.assertEqual(result['model']['last_200_hits'], 85)
        self.assertEqual(result['baseline']['last_200_hits'], 76)
        self.assertTrue(
            all(len(row['prediction']) == 18 for row in result['rows'])
        )

    def test_audit_rejects_reverse_dates(self):
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig').tail(60)

        with self.assertRaisesRegex(ValueError, '日期必须严格按时间升序'):
            run_fixed_dual_source_top18_audit(
                df.iloc[::-1].reset_index(drop=True), test_periods=10
            )

    def test_audit_requires_full_requested_window(self):
        df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig').head(100)

        with self.assertRaisesRegex(ValueError, '至少需要150期数据'):
            run_fixed_dual_source_top18_audit(df, test_periods=100)


if __name__ == '__main__':
    unittest.main()
