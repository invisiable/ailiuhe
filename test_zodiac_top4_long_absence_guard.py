import unittest

from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from retrained_zodiac_predictor import RetrainedZodiacPredictor


class ZodiacTop4LongAbsenceGuardTest(unittest.TestCase):
    def test_excludes_any_zodiac_absent_for_32_periods_from_top4(self):
        predictor = RetrainedZodiacPredictor()
        for target in predictor.zodiacs:
            recent = [
                zodiac for zodiac in predictor.zodiacs
                if zodiac != target
            ][:4]
            animals = [target] + (recent * 8)

            result = predictor.predict_from_history(animals, top_n=4)

            self.assertIn(target, result['long_absence_excluded'])
            self.assertNotIn(target, result['top4'])
            self.assertEqual(len(result['top4']), 4)

    def test_default_recommended_strategy_uses_primary_only(self):
        strategy = RecommendedZodiacTop4Strategy()

        self.assertIsNone(strategy.backup_predictor)
        self.assertEqual(strategy.current_model, 'primary')


if __name__ == '__main__':
    unittest.main()
