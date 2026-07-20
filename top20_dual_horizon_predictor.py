"""TOP20 dual-horizon rank-fusion model, separate from the TOP15 policy."""

from precise_top15_predictor import PreciseTop15Predictor


class Top20DualHorizonPredictor:
    """Fuse independent short- and medium-horizon complete rankings."""

    SHORT_WINDOW = 6
    MEDIUM_WINDOW = 25

    def predict(self, history, top_n=20):
        scores = {number: 0.0 for number in range(1, 50)}
        for window in (self.SHORT_WINDOW, self.MEDIUM_WINDOW):
            ranking = PreciseTop15Predictor().predict(history[-window:], k=49)
            for position, number in enumerate(ranking):
                scores[number] += 49 - position
        return sorted(scores, key=scores.get, reverse=True)[:top_n]
