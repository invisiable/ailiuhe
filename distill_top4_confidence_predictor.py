"""
硬蒸馏TOP9→TOP4 方案E: 置信度分层预测器
============================================
Stage1: TOP9排除3个生肖 (85%命中率过滤)
Stage2: 根据TOP9置信度分层选择策略
  - 高置信(TOP9第9名与第10名分差≥0.05): MK80选TOP4
  - 低置信(分差<0.05): v3静态选TOP4

固定TOP4, 纯生肖版本
"""
import numpy as np
from collections import Counter

ZODIAC_CYCLE_2026 = ['马', '蛇', '龙', '兔', '虎', '牛', '鼠', '猪', '狗', '鸡', '猴', '羊']
Z_IDX = {z: i for i, z in enumerate(ZODIAC_CYCLE_2026)}
NUM_TO_ZODIAC_2026 = {n: ZODIAC_CYCLE_2026[(n - 1) % 12] for n in range(1, 50)}
ZODIAC_NUMS_2026 = {}
for z in ZODIAC_CYCLE_2026:
    ZODIAC_NUMS_2026[z] = sorted([n for n in range(1, 50) if NUM_TO_ZODIAC_2026[n] == z])


class DistillTop4ConfidencePredictor:
    """
    硬蒸馏TOP9→TOP4 方案E: 置信度分层
    高置信TOP9 → MK80选4个
    低置信TOP9 → v3选4个
    """

    def __init__(self):
        self.name = "蒸馏TOP4_置信度分层"
        self.confidence_threshold = 0.05

        # Stage1: TOP9权重
        self.s1_weights = {
            'cold15': 0.20, 'cold30': 0.05,
            'mk150': 0.50, 'gap': 0.10, 'hot30': 0.15,
        }
        self.v3_weights = {'cold15': 0.30, 'cold30': 0.10, 'mk150': 0.60}

    # ---- 评分函数 ----
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

    def _stage1_scores(self, animals):
        return (self.s1_weights['cold15'] * self._cold_scores(animals, 15) +
                self.s1_weights['cold30'] * self._cold_scores(animals, 30) +
                self.s1_weights['mk150'] * self._markov_scores(animals, 150) +
                self.s1_weights['gap'] * self._gap_scores(animals) +
                self.s1_weights['hot30'] * self._hot_scores(animals, 30))

    def _s2_mk80(self, animals):
        return self._markov_scores(animals, 80)

    def _s2_v3(self, animals):
        return (self.v3_weights['cold15'] * self._cold_scores(animals, 15) +
                self.v3_weights['cold30'] * self._cold_scores(animals, 30) +
                self.v3_weights['mk150'] * self._markov_scores(animals, 150))

    def predict(self, numbers, top_n=4):
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        s1 = self._stage1_scores(animals)
        sorted_s1 = np.sort(s1)[::-1]
        confidence = sorted_s1[8] - sorted_s1[9] if len(sorted_s1) > 9 else 0
        top9_idx = np.argsort(-s1)[:9]
        if confidence >= self.confidence_threshold:
            s2 = self._s2_mk80(animals)
        else:
            s2 = self._s2_v3(animals)
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:4]
        top_idx = top9_idx[top_in9]
        return [ZODIAC_CYCLE_2026[i] for i in top_idx]

    def predict_with_details(self, numbers, top_n=4):
        animals = [NUM_TO_ZODIAC_2026[n] for n in numbers]
        s1 = self._stage1_scores(animals)
        sorted_s1 = np.sort(s1)[::-1]
        confidence = sorted_s1[8] - sorted_s1[9] if len(sorted_s1) > 9 else 0
        top9_idx = np.argsort(-s1)[:9]
        if confidence >= self.confidence_threshold:
            s2 = self._s2_mk80(animals)
            mode = f"高置信({confidence:.3f}≥{self.confidence_threshold})→MK80"
        else:
            s2 = self._s2_v3(animals)
            mode = f"低置信({confidence:.3f}<{self.confidence_threshold})→v3"
        s2_in9 = s2[top9_idx]
        top_in9 = np.argsort(-s2_in9)[:4]
        top_idx = top9_idx[top_in9]
        predictions = [ZODIAC_CYCLE_2026[i] for i in top_idx]
        excluded_idx = [i for i in range(12) if i not in set(top9_idx)]
        excluded = [ZODIAC_CYCLE_2026[i] for i in excluded_idx]
        hot = self._hot_scores(animals, 30)
        scores = {}
        for i in top_idx:
            z = ZODIAC_CYCLE_2026[i]
            scores[z] = {
                's1': float(s1[i]),
                's2': float(s2[i]),
                'hot': float(hot[i]),
            }
        return predictions, mode, scores, confidence, excluded

    def record_result(self, hit):
        pass

    def reset(self):
        pass

    def get_zodiac_numbers(self, zodiac):
        return ZODIAC_NUMS_2026.get(zodiac, [])

    def get_all_predicted_numbers(self, numbers, top_n=4):
        predicted = self.predict(numbers, top_n)
        return {z: ZODIAC_NUMS_2026[z] for z in predicted}
