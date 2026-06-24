"""
生肖TOP10预测器 - 基于TOP9模型，扩展到TOP10提升覆盖率
============================================
策略与TOP9相同（相同权重），默认返回TOP10生肖。
反miss机制：
1. 正常模式: cold15(0.20) + cold30(0.05) + MK150(0.50) + gap(0.10) + hot30(0.15) → TOP10
2. 反miss L1 (连miss>=2): blend 25% 额外热号 → TOP10
3. 反miss L2 (连miss>=3): blend + 扩展TOP11

随机基线: 83.3% (TOP10/12)
"""
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE_2026)}
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_NUMS_2026 = {}
for z in ZODIAC_CYCLE_2026:
    ZODIAC_NUMS_2026[z] = sorted([n for n in range(1, 50) if NUM_TO_ZODIAC_2026[n] == z])


class ZodiacTop10Predictor:
    """
    生肖TOP10预测器
    与TOP9相同的多维度权重，默认选TOP10生肖
    """

    def __init__(self):
        self.name = "生肖TOP10"
        self.consecutive_miss = 0

        # 与TOP9相同权重
        self.weights = {
            'cold15': 0.20,
            'cold30': 0.05,
            'mk150': 0.50,
            'gap': 0.10,
            'hot30': 0.15,
        }

        # 反miss配置
        self.blend_threshold = 2      # 连miss>=2时blend额外热号
        self.expand_threshold = 3     # 连miss>=3时扩展到TOP11
        self.hot_blend_ratio = 0.25
        self.hot_window = 30

    def _cold_scores(self, animals, window):
        w = min(window, len(animals))
        freq = Counter(animals[-w:])
        mx = max(freq.values()) if freq else 1
        return np.array([1.0 - freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

    def _hot_scores(self, animals, window):
        w = min(window, len(animals))
        freq = Counter(animals[-w:])
        mx = max(freq.values()) if freq else 1
        return np.array([freq.get(z, 0) / max(mx, 1) for z in ZODIAC_CYCLE_2026])

    def _markov_scores(self, animals, window=None, laplace=1.0):
        probs = np.ones(12) / 12
        h = animals[-window:] if window and len(animals) > window else animals
        if len(h) < 2:
            return probs
        trans = {}
        for k in range(1, len(h)):
            p, c = h[k - 1], h[k]
            if p not in trans:
                trans[p] = Counter()
            trans[p][c] += 1
        state = animals[-1]
        if state in trans:
            total = sum(trans[state].values()) + laplace * 12
            for zi, z in enumerate(ZODIAC_CYCLE_2026):
                probs[zi] = (trans[state].get(z, 0) + laplace) / total
        return probs

    def _gap_scores(self, animals):
        scores = []
        for z in ZODIAC_CYCLE_2026:
            last = -1
            for j in range(len(animals) - 1, -1, -1):
                if animals[j] == z:
                    last = j
                    break
            gap = (len(animals) - 1 - last) if last >= 0 else len(animals)
            scores.append(gap / 12.0)
        return np.array(scores)

    def _base_scores(self, animals):
        cold15 = self._cold_scores(animals, 15)
        cold30 = self._cold_scores(animals, 30)
        mk150 = self._markov_scores(animals, window=150)
        gap = self._gap_scores(animals)
        hot30 = self._hot_scores(animals, self.hot_window)
        return (self.weights['cold15'] * cold15 +
                self.weights['cold30'] * cold30 +
                self.weights['mk150'] * mk150 +
                self.weights['gap'] * gap +
                self.weights['hot30'] * hot30)

    def predict(self, numbers, top_n=10):
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        scores = self._base_scores(animals)
        if self.consecutive_miss >= self.blend_threshold:
            hot = self._hot_scores(animals, self.hot_window)
            scores = (1 - self.hot_blend_ratio) * scores + self.hot_blend_ratio * hot
        sorted_idx = np.argsort(-scores)
        if self.consecutive_miss >= self.expand_threshold:
            return [ZODIAC_CYCLE_2026[i] for i in sorted_idx[:top_n + 1]]
        return [ZODIAC_CYCLE_2026[i] for i in sorted_idx[:top_n]]

    def predict_with_details(self, numbers, top_n=10):
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        base = self._base_scores(animals)
        hot = self._hot_scores(animals, self.hot_window)

        if self.consecutive_miss >= self.expand_threshold:
            mode = f"反miss-L2(扩展TOP11, 连miss={self.consecutive_miss})"
            combined = (1 - self.hot_blend_ratio) * base + self.hot_blend_ratio * hot
            sorted_idx = np.argsort(-combined)
            top_idx = list(sorted_idx[:top_n + 1])
        elif self.consecutive_miss >= self.blend_threshold:
            mode = f"反miss-L1(blend热号, 连miss={self.consecutive_miss})"
            combined = (1 - self.hot_blend_ratio) * base + self.hot_blend_ratio * hot
            sorted_idx = np.argsort(-combined)
            top_idx = list(sorted_idx[:top_n])
        else:
            mode = "正常(多维度组合)"
            combined = base
            sorted_idx = np.argsort(-combined)
            top_idx = list(sorted_idx[:top_n])

        predictions = [ZODIAC_CYCLE_2026[i] for i in top_idx]
        scores = {}
        for i in top_idx:
            z = ZODIAC_CYCLE_2026[i]
            scores[z] = {
                'base': float(base[i]),
                'hot': float(hot[i]),
                'combined': float(combined[i]) if self.consecutive_miss >= self.blend_threshold else float(base[i]),
            }
        return predictions, mode, scores

    def record_result(self, hit):
        if hit:
            self.consecutive_miss = 0
        else:
            self.consecutive_miss += 1

    def reset(self):
        self.consecutive_miss = 0

    def get_zodiac_numbers(self, zodiac):
        return ZODIAC_NUMS_2026.get(zodiac, [])

    def get_all_predicted_numbers(self, numbers, top_n=10):
        predicted = self.predict(numbers, top_n)
        result = {}
        for z in predicted:
            result[z] = ZODIAC_NUMS_2026[z]
        return result
