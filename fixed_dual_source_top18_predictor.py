"""固定18号风险优先预测器。"""

import numpy as np

from dual_objective_top15_predictor import (
    NORMAL_WEIGHTS,
    DualObjectiveTop15Predictor,
)
from zodiac_simple_smart import ZodiacSimpleSmart


TOP_N = 18
RESCUE_THRESHOLD = 5


class FixedDualSourceTop18Predictor:
    """固定18号风险优先模型，生肖TOP5仅作为旁路监测。"""

    def __init__(self):
        self.number_predictor = DualObjectiveTop15Predictor()
        self.zodiac_predictor = ZodiacSimpleSmart()
        self.selected_miss_streaks = {
            zodiac: 0 for zodiac in self.zodiac_predictor.zodiacs
        }
        self.consecutive_misses = 0
        self._last_number_scores = None
        self._last_number_feedback_scores = None
        self._last_zodiac_top5 = None

    @staticmethod
    def _normalize(scores):
        values = np.asarray(scores, dtype=float)
        spread = float(values.max() - values.min())
        if spread == 0:
            return np.zeros_like(values)
        return (values - values.min()) / spread

    @staticmethod
    def _select_top_n(scores, top_n=TOP_N):
        count = int(top_n)
        indexes = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )[:count]
        return [int(index) + 1 for index in indexes]

    def _current_blend(self):
        if self.consecutive_misses >= RESCUE_THRESHOLD:
            return 1.0, 0.0, '连不中5期冷号救援'
        return 1.0, 0.0, '风险优先固定18'

    def _build_number_scores(self, numbers):
        features, _ = self.number_predictor.build_features(numbers)
        normal_scores = np.tensordot(NORMAL_WEIGHTS, features, axes=(0, 0))
        rescue_scores = features[14]
        output_scores = (
            rescue_scores if self.consecutive_misses >= 5 else normal_scores
        )
        feedback_scores = (
            rescue_scores
            if self.number_predictor.consecutive_misses >= 5
            else normal_scores
        )
        return output_scores, feedback_scores

    def _build_zodiac_scores(self, animals):
        result = self.zodiac_predictor.predict_from_history(
            animals,
            top_n=5,
            debug=False,
            selected_miss_streaks=self.selected_miss_streaks,
        )
        ranking = [zodiac for zodiac, _ in result['all_scores']]
        rank_scores = {
            zodiac: 1.0 - rank / 11.0
            for rank, zodiac in enumerate(ranking)
        }
        number_to_zodiac = {
            int(number): zodiac
            for zodiac, zodiac_numbers in self.zodiac_predictor.zodiac_numbers.items()
            for number in zodiac_numbers
        }
        scores = np.array([
            rank_scores[number_to_zodiac[number]]
            for number in range(1, 50)
        ])
        return scores, result

    def predict_with_details(self, numbers, animals):
        number_history = [int(number) for number in numbers]
        animal_history = [str(animal).strip() for animal in animals]
        if len(number_history) != len(animal_history):
            raise ValueError('号码与生肖历史长度必须一致')
        if len(number_history) < 50:
            raise ValueError('固定18号双源模型至少需要50期历史')

        number_scores, feedback_scores = self._build_number_scores(number_history)
        zodiac_scores, zodiac_result = self._build_zodiac_scores(animal_history)
        number_weight, zodiac_weight, mode = self._current_blend()
        fused_scores = (
            number_weight * self._normalize(number_scores)
            + zodiac_weight * self._normalize(zodiac_scores)
        )
        prediction = self._select_top_n(fused_scores, TOP_N)
        self._last_number_scores = number_scores
        self._last_number_feedback_scores = feedback_scores
        self._last_zodiac_top5 = list(zodiac_result['top5'])
        return {
            'top18': prediction,
            'mode': mode,
            'number_weight': number_weight,
            'zodiac_weight': zodiac_weight,
            'consecutive_misses': self.consecutive_misses,
            'zodiac_top5': list(zodiac_result['top5']),
            'zodiac_ranking': [zodiac for zodiac, _ in zodiac_result['all_scores']],
            'selected_model': zodiac_result['selected_model'],
            'scores': {
                number: float(fused_scores[number - 1])
                for number in prediction
            },
        }

    def predict(self, numbers, animals):
        return self.predict_with_details(numbers, animals)['top18']

    def _update_model_streak(self, hit):
        self.consecutive_misses = 0 if hit else self.consecutive_misses + 1

    def update(self, prediction, actual_number, actual_animal):
        if (
            self._last_number_feedback_scores is None
            or self._last_zodiac_top5 is None
        ):
            raise RuntimeError('必须先预测再更新结果')

        actual_number = int(actual_number)
        actual_animal = str(actual_animal).strip()
        base_top15 = self._select_top_n(self._last_number_feedback_scores, 15)
        self.number_predictor.update(base_top15, actual_number)

        selected = set(self._last_zodiac_top5)
        for zodiac in self.zodiac_predictor.zodiacs:
            if zodiac not in selected or zodiac == actual_animal:
                self.selected_miss_streaks[zodiac] = 0
            else:
                self.selected_miss_streaks[zodiac] += 1

        hit = actual_number in prediction
        self._update_model_streak(hit)
        return hit
