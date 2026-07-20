"""TOP3 tail-digit strategy with a fixed three-draw prediction cycle.

The model intentionally keeps one group of three tails for up to three draws.
It starts a new group immediately after a hit, or after the third miss.  This
is a risk-control/record-keeping rule; it does not and cannot guarantee a hit.
"""

from collections import Counter, defaultdict

from tail_digit_top3_predictor import TailDigitTop3Predictor, number_to_tail


# Fixed signal blend selected once through four rolling validation segments.
# It is never switched in response to a miss streak.
TREND_TRANSITION_WEIGHTS = {
    "frequency": 0.40,
    "cold_rebound": 0.18,
    "trend": 0.02,
    "cycle": 0.27,
    "adjacent": 0.11,
    "gap_pattern": 0.01,
}

# A small, fixed calibration after a hit.  It is a score adjustment, not an
# exclusion: the tail can still remain in TOP3 when other signals support it.
HIT_TAIL_PENALTY = 0.02

class ThreePeriodTailTop3Predictor:
    """Stateful TOP3 predictor: one prediction group is valid for 1--3 draws."""

    def __init__(self, selection_window=90):
        self.selection_window = selection_window
        self.base = TailDigitTop3Predictor()
        self.active_prediction = None
        self.active_scores = None
        self.active_model = None
        self.draws_used = 0
        self.cycle_number = 0
        self.history_preds = []
        self.hit_records = []
        self.previous_hit_tail = None

    def _scores(self, numbers, weights):
        tails = [number_to_tail(n) for n in numbers]
        components = {
            "frequency": self.base._frequency_analysis(tails),
            "cold_rebound": self.base._cold_rebound_analysis(tails),
            "trend": self.base._trend_momentum_analysis(tails),
            "cycle": self.base._cycle_analysis(tails),
            "adjacent": self.base._adjacent_analysis(tails),
            "gap_pattern": self.base._gap_pattern_analysis(tails),
        }
        return {
            digit: sum(weights[name] * components[name].get(digit, 0.0)
                       for name in weights)
            for digit in range(10)
        }

    def _select_weights(self, numbers):
        """Return the fixed blend; no miss-streak driven model switching."""
        return "稳健频率+周期（固定）", TREND_TRANSITION_WEIGHTS

    def predict_with_details(self, numbers):
        """Return (tails, scores, status) for the next draw."""
        if self.active_prediction is None:
            model_name, weights = self._select_weights(numbers)
            scores = self._scores(numbers, weights)
            if self.previous_hit_tail is not None:
                scores[self.previous_hit_tail] -= HIT_TAIL_PENALTY
            self.active_prediction = sorted(scores, key=scores.get, reverse=True)[:3]
            self.active_scores = scores
            self.active_model = model_name
            self.draws_used = 0
            self.cycle_number += 1
            action = "new_cycle"
        else:
            action = "hold"

        status = {
            "cycle": self.cycle_number,
            "draw": self.draws_used + 1,
            "action": action,
            "model": self.active_model,
        }
        return list(self.active_prediction), dict(self.active_scores), status

    def record_result(self, predicted, hit, actual_tail=None):
        """Record one settled draw and reset only on hit or on draw three."""
        if self.active_prediction is None:
            raise RuntimeError("Call predict_with_details before record_result.")
        if list(predicted) != self.active_prediction:
            raise ValueError("Result does not match the active three-period prediction.")

        self.history_preds.append(list(predicted))
        self.hit_records.append(bool(hit))
        self.draws_used += 1
        reset_reason = None
        if hit:
            reset_reason = "hit"
            self.previous_hit_tail = actual_tail
        elif self.draws_used >= 3:
            reset_reason = "three_misses"
            self.previous_hit_tail = None
        else:
            self.previous_hit_tail = None

        if reset_reason:
            self.active_prediction = None
            self.active_scores = None
            self.active_model = None
            self.draws_used = 0
        return reset_reason


def validate_three_period_top3(numbers, dates=None, test_periods=100, end_index=None):
    """Walk-forward validation with full per-draw and per-cycle conclusions."""
    end_index = len(numbers) if end_index is None else min(end_index, len(numbers))
    test_periods = min(test_periods, end_index - 60)
    if test_periods <= 0:
        raise ValueError("At least 61 historical draws are required.")

    start = end_index - test_periods
    predictor = ThreePeriodTailTop3Predictor()
    rows, cycles = [], []
    cycle_rows = []

    for index in range(start, end_index):
        predicted, scores, status = predictor.predict_with_details(numbers[:index])
        actual = numbers[index]
        tail = number_to_tail(actual)
        hit = tail in predicted
        reset_reason = predictor.record_result(predicted, hit, actual_tail=tail)
        row = {
            "period": index - start + 1,
            "date": dates[index] if dates is not None else str(index),
            "actual": actual,
            "tail": tail,
            "predicted": predicted,
            "hit": hit,
            "cycle": status["cycle"],
            "cycle_draw": status["draw"],
            "model": status["model"],
            "reset_reason": reset_reason,
            "scores": scores,
        }
        rows.append(row)
        cycle_rows.append(row)
        if reset_reason:
            cycles.append({
                "cycle": status["cycle"],
                "predicted": predicted,
                "draws": len(cycle_rows),
                "hit": hit,
                "reason": reset_reason,
                "start_period": cycle_rows[0]["period"],
                "end_period": cycle_rows[-1]["period"],
            })
            cycle_rows = []

    if cycle_rows:  # The latest cycle may still be in progress at the cutoff.
        cycles.append({
            "cycle": cycle_rows[0]["cycle"], "predicted": cycle_rows[0]["predicted"],
            "draws": len(cycle_rows), "hit": any(r["hit"] for r in cycle_rows),
            "reason": "in_progress", "start_period": cycle_rows[0]["period"],
            "end_period": cycle_rows[-1]["period"],
        })

    settled_cycles = [c for c in cycles if c["reason"] != "in_progress"]
    hits = sum(row["hit"] for row in rows)
    cycle_hits = sum(c["hit"] for c in settled_cycles)
    model_counts = Counter(row["model"] for row in rows if row["cycle_draw"] == 1)
    max_miss = 0
    current_miss = 0
    for row in rows:
        current_miss = 0 if row["hit"] else current_miss + 1
        max_miss = max(max_miss, current_miss)

    return {
        "rows": rows,
        "cycles": cycles,
        "total": len(rows),
        "hits": hits,
        "hit_rate": hits / len(rows) * 100,
        "settled_cycles": len(settled_cycles),
        "cycle_hits": cycle_hits,
        "cycle_hit_rate": (cycle_hits / len(settled_cycles) * 100) if settled_cycles else 0.0,
        "three_miss_cycles": sum(c["reason"] == "three_misses" for c in settled_cycles),
        "max_miss": max_miss,
        "model_counts": dict(model_counts),
        "predictor": predictor,
    }
