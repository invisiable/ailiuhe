from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd

from distilled_top15_predictor import DistilledTop15Predictor


class Top15AdvancedModel:
    """Strict TOP15 betting model focused on shorter miss streaks."""

    def __init__(
        self,
        display_count: int = 15,
        normal_count: int = 15,
        emergency_count: int = 15,
        miss_trigger: int = 1,
        skip_periods: int = 5,
    ) -> None:
        self.display_count = display_count
        self.normal_count = normal_count
        self.emergency_count = emergency_count
        self.miss_trigger = miss_trigger
        self.skip_periods = skip_periods
        self.reset_state()

    def reset_state(self) -> None:
        self.predictor = DistilledTop15Predictor()
        self.consecutive_bet_misses = 0
        self.skip_remaining = 0

    def _to_list(self, numbers) -> List[int]:
        if hasattr(numbers, "tolist"):
            return [int(x) for x in numbers.tolist()]
        return [int(x) for x in numbers]

    def _predict_top15(self, history_numbers) -> List[int]:
        history = self._to_list(history_numbers)
        return [int(x) for x in self.predictor.predict(history)[: self.display_count]]

    def get_period_action(self, history_numbers) -> Dict:
        top15 = self._predict_top15(history_numbers)

        if self.skip_remaining > 0:
            return {
                "mode": "skip",
                "mode_label": "\u89c2\u5bdf",
                "action_label": "\u8df3\u8fc7",
                "should_bet": False,
                "display_numbers": top15,
                "active_numbers": [],
                "active_count": 0,
                "reason": (
                    f"\u4e0a\u6b21\u672a\u4e2d\u540e\u8fdb\u5165\u89c2\u5bdf\u6a21\u5f0f\uff0c"
                    f"\u5269\u4f59 {self.skip_remaining} \u671f"
                ),
            }

        return {
            "mode": "normal",
            "mode_label": "\u6b63\u5e38",
            "action_label": "\u6295\u6ce8",
            "should_bet": True,
            "display_numbers": top15,
            "active_numbers": top15,
            "active_count": len(top15),
            "reason": f"\u4e25\u683cTOP15\u6a21\u5f0f\uff0c\u5b9e\u9645\u6295\u6ce8\u53f7\u7801\u6570={len(top15)}",
        }

    def settle_period(self, action: Dict, actual_number: int) -> Dict:
        actual = int(actual_number)
        before_streak = self.consecutive_bet_misses
        reference_prediction = action["display_numbers"]
        reference_hit = actual in reference_prediction

        # Keep predictor state aligned on every period, including skipped ones.
        self.predictor.update(reference_prediction, actual)

        if action["mode"] == "skip":
            self.skip_remaining = max(0, self.skip_remaining - 1)
            return {
                "hit": False,
                "reference_hit": reference_hit,
                "bet_miss_streak_before": before_streak,
                "bet_miss_streak_after": before_streak,
                "skip_remaining_after": self.skip_remaining,
            }

        hit = actual in action["active_numbers"]
        if hit:
            self.consecutive_bet_misses = 0
            self.skip_remaining = 0
        else:
            self.consecutive_bet_misses += 1
            if self.consecutive_bet_misses >= self.miss_trigger:
                self.skip_remaining = self.skip_periods

        return {
            "hit": hit,
            "reference_hit": reference_hit,
            "bet_miss_streak_before": before_streak,
            "bet_miss_streak_after": self.consecutive_bet_misses,
            "skip_remaining_after": self.skip_remaining,
        }

    def backtest_recent(self, df: pd.DataFrame, test_periods: int = 400) -> Dict:
        if len(df) < 60:
            raise ValueError("\u0054\u004f\u0050\u0031\u0035\u9ad8\u7ea7\u6a21\u578b\u81f3\u5c11\u9700\u8981\u0036\u0030\u671f\u6570\u636e")

        periods = min(test_periods, len(df) - 50)
        start_idx = len(df) - periods
        numbers = df["number"].astype(int).tolist()

        self.reset_state()

        details = []
        bet_periods = 0
        skip_periods = 0
        hit_count = 0
        skipped_reference_hits = 0
        streak_distribution = Counter()
        current_miss_segment = 0
        max_bet_miss = 0

        for i in range(start_idx, len(df)):
            history_numbers = numbers[:i]
            actual = numbers[i]
            action = self.get_period_action(history_numbers)
            settle = self.settle_period(action, actual)

            if action["should_bet"]:
                bet_periods += 1
                if settle["hit"]:
                    hit_count += 1
                    if current_miss_segment > 0:
                        streak_distribution[current_miss_segment] += 1
                        current_miss_segment = 0
                else:
                    current_miss_segment += 1
                    max_bet_miss = max(max_bet_miss, current_miss_segment)
            else:
                skip_periods += 1
                if settle["reference_hit"]:
                    skipped_reference_hits += 1

            details.append(
                {
                    "seq": i - start_idx + 1,
                    "date": str(df.iloc[i]["date"]),
                    "actual": actual,
                    "mode": action["mode"],
                    "mode_label": action["mode_label"],
                    "action_label": action["action_label"],
                    "active_count": action["active_count"],
                    "display_numbers": action["display_numbers"],
                    "active_numbers": action["active_numbers"],
                    "hit": settle["hit"],
                    "reference_hit": settle["reference_hit"],
                    "bet_miss_streak_before": settle["bet_miss_streak_before"],
                    "bet_miss_streak_after": settle["bet_miss_streak_after"],
                    "skip_remaining_after": settle["skip_remaining_after"],
                    "reason": action["reason"],
                }
            )

        if current_miss_segment > 0:
            streak_distribution[current_miss_segment] += 1

        miss_segment_count = sum(streak_distribution.values())
        within_3_segments = sum(
            count for length, count in streak_distribution.items() if length <= 3
        )
        within_5_segments = sum(
            count for length, count in streak_distribution.items() if length <= 5
        )

        next_action = self.get_period_action(numbers)

        return {
            "details": details,
            "summary": {
                "total_periods": periods,
                "bet_periods": bet_periods,
                "skip_periods": skip_periods,
                "hit_count": hit_count,
                "skipped_reference_hits": skipped_reference_hits,
                "bet_hit_rate": (hit_count / bet_periods * 100) if bet_periods else 0.0,
                "coverage_rate": (bet_periods / periods * 100) if periods else 0.0,
                "max_bet_miss": max_bet_miss,
                "streak_distribution": dict(sorted(streak_distribution.items())),
                "miss_segment_count": miss_segment_count,
                "within_3_ratio": (
                    within_3_segments / miss_segment_count * 100
                    if miss_segment_count
                    else 0.0
                ),
                "within_5_ratio": (
                    within_5_segments / miss_segment_count * 100
                    if miss_segment_count
                    else 0.0
                ),
            },
            "next_action": next_action,
        }
