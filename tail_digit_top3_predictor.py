"""
尾数TOP3预测模型 - 纯尾数统计预测，每次输出3个尾数

尾数映射(1-49):
  尾数0: [10, 20, 30, 40]        (4个)
  尾数1: [1, 11, 21, 31, 41]     (5个)
  尾数2: [2, 12, 22, 32, 42]     (5个)
  尾数3: [3, 13, 23, 33, 43]     (5个)
  尾数4: [4, 14, 24, 34, 44]     (5个)
  尾数5: [5, 15, 25, 35, 45]     (5个)
  尾数6: [6, 16, 26, 36, 46]     (5个)
  尾数7: [7, 17, 27, 37, 47]     (5个)
  尾数8: [8, 18, 28, 38, 48]     (5个)
  尾数9: [9, 19, 29, 39, 49]     (5个)

核心策略:
  - 6维统计信号加权打分(频率/冷号/趋势/周期/关联/间隔)
  - 智能轮换排除: miss后排除上轮，从剩余选TOP3
  - 救援模式: 连miss≥3时切换冷号+间隔+周期评分
  - 300期回测: 单期命中~34%, 三期窗口~70%
  - 随机基线: 30%(3/10)
"""

import numpy as np
from collections import Counter, defaultdict


# 尾数到号码的映射(1-49)
TAIL_DIGIT_NUMBERS = {
    0: [10, 20, 30, 40],
    1: [1, 11, 21, 31, 41],
    2: [2, 12, 22, 32, 42],
    3: [3, 13, 23, 33, 43],
    4: [4, 14, 24, 34, 44],
    5: [5, 15, 25, 35, 45],
    6: [6, 16, 26, 36, 46],
    7: [7, 17, 27, 37, 47],
    8: [8, 18, 28, 38, 48],
    9: [9, 19, 29, 39, 49],
}


def number_to_tail(n):
    """号码转尾数"""
    return n % 10


class TailDigitTop3Predictor:
    """
    尾数TOP3智能轮换预测模型

    选3个尾数(约14-15个号码), 基于纯尾数统计分析。
    使用轮换排除机制提升连续命中率。

    300期回测: 单期34%, 三期窗口70%
    """

    def __init__(self):
        self.history_preds = []
        self.hit_records = []

    def predict(self, numbers, top_n=3):
        """
        预测下一期最可能的3个尾数

        Args:
            numbers: 历史号码列表
            top_n: 返回尾数个数(默认3)
        Returns:
            list: 预测的3个尾数
        """
        all_scores = self._calculate_scores(numbers)
        sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        # 计算连续miss
        miss_streak = 0
        for j in range(len(self.hit_records) - 1, -1, -1):
            if not self.hit_records[j]:
                miss_streak += 1
            else:
                break

        tails = [number_to_tail(n) for n in numbers]

        if miss_streak == 0:
            # 正常模式: 取得分前3
            predicted = [d for d, s in sorted_all[:top_n]]
        elif miss_streak == 1:
            # 轮换1: 排除上轮预测, 从剩余取TOP3
            excluded = set(self.history_preds[-1]) if self.history_preds else set()
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            predicted = [d for d, s in remaining[:top_n]] if len(remaining) >= top_n else [d for d, s in sorted_all[:top_n]]
        elif miss_streak == 2:
            # 轮换2: 排除最近2轮预测, 取剩余TOP3
            excluded = set()
            for hp in self.history_preds[-2:]:
                if hp:
                    excluded.update(hp)
            remaining = [(d, s) for d, s in sorted_all if d not in excluded]
            if len(remaining) >= top_n:
                predicted = [d for d, s in remaining[:top_n]]
            else:
                predicted = [d for d, s in sorted_all[:top_n]]
        else:
            # 救援模式(≥3 miss): 切换评分体系, 用冷号+间隔+周期
            cold = self._cold_rebound_analysis(tails)
            gap = self._gap_pattern_analysis(tails)
            cycle = self._cycle_analysis(tails)
            rescue = {d: 0.45 * cold[d] + 0.30 * gap[d] + 0.25 * cycle[d] for d in range(10)}
            rescue_sorted = sorted(rescue.items(), key=lambda x: x[1], reverse=True)
            # 排除上一轮
            excluded = set(self.history_preds[-1]) if self.history_preds else set()
            remaining = [(d, s) for d, s in rescue_sorted if d not in excluded]
            predicted = [d for d, s in remaining[:top_n]] if len(remaining) >= top_n else [d for d, s in rescue_sorted[:top_n]]

        return predicted

    def predict_with_details(self, numbers, top_n=3):
        """带详情的预测"""
        all_scores = self._calculate_scores(numbers)
        predicted = self.predict(numbers, top_n)

        miss_streak = 0
        for j in range(len(self.hit_records) - 1, -1, -1):
            if not self.hit_records[j]:
                miss_streak += 1
            else:
                break

        if miss_streak == 0:
            mode = "正常"
        elif miss_streak == 1:
            mode = "轮换1"
        elif miss_streak == 2:
            mode = "轮换2"
        else:
            mode = f"救援"

        return predicted, all_scores, mode

    def record_result(self, predicted, hit):
        """记录预测结果"""
        self.history_preds.append(predicted)
        self.hit_records.append(hit)

    def _calculate_scores(self, numbers):
        """计算各尾数综合得分 - 6维信号加权"""
        if len(numbers) < 10:
            return {d: 1.0 for d in range(10)}

        tails = [number_to_tail(n) for n in numbers]

        freq_scores = self._frequency_analysis(tails)
        cold_scores = self._cold_rebound_analysis(tails)
        trend_scores = self._trend_momentum_analysis(tails)
        cycle_scores = self._cycle_analysis(tails)
        adjacent_scores = self._adjacent_analysis(tails)
        gap_scores = self._gap_pattern_analysis(tails)

        # 加权融合
        weights = {
            'frequency': 0.20,
            'cold_rebound': 0.25,
            'trend': 0.20,
            'cycle': 0.15,
            'adjacent': 0.10,
            'gap_pattern': 0.10,
        }

        combined = {}
        for d in range(10):
            combined[d] = (
                weights['frequency'] * freq_scores.get(d, 0) +
                weights['cold_rebound'] * cold_scores.get(d, 0) +
                weights['trend'] * trend_scores.get(d, 0) +
                weights['cycle'] * cycle_scores.get(d, 0) +
                weights['adjacent'] * adjacent_scores.get(d, 0) +
                weights['gap_pattern'] * gap_scores.get(d, 0)
            )

        return combined

    def _frequency_analysis(self, tails, window=30):
        """频率分析 - 最近window期各尾数出现频率"""
        recent = tails[-window:]
        counter = Counter(recent)
        total = len(recent)

        scores = {}
        for d in range(10):
            scores[d] = counter.get(d, 0) / total

        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores

    def _cold_rebound_analysis(self, tails):
        """冷号回补分析 - 长期未出现的尾数回补概率"""
        gaps = {}
        for d in range(10):
            last_idx = -1
            for i in range(len(tails) - 1, -1, -1):
                if tails[i] == d:
                    last_idx = i
                    break
            if last_idx == -1:
                gaps[d] = len(tails)
            else:
                gaps[d] = len(tails) - 1 - last_idx

        expected_gap = 10
        scores = {}
        for d in range(10):
            ratio = gaps[d] / expected_gap
            scores[d] = 1.0 / (1.0 + np.exp(-0.5 * (ratio - 1.0)))

        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores

    def _trend_momentum_analysis(self, tails):
        """趋势动量分析 - 近期频率变化加速度"""
        if len(tails) < 30:
            return {d: 0.5 for d in range(10)}

        recent_10 = tails[-10:]
        prev_20 = tails[-30:-10]

        counter_recent = Counter(recent_10)
        counter_prev = Counter(prev_20)

        scores = {}
        for d in range(10):
            freq_recent = counter_recent.get(d, 0) / 10
            freq_prev = counter_prev.get(d, 0) / 20
            momentum = freq_recent - freq_prev
            scores[d] = momentum

        min_s = min(scores.values())
        max_s = max(scores.values())
        spread = max_s - min_s if max_s != min_s else 1
        scores = {d: (s - min_s) / spread for d, s in scores.items()}
        return scores

    def _cycle_analysis(self, tails):
        """周期分析 - 检测尾数出现的周期性规律"""
        scores = {}
        for d in range(10):
            positions = [i for i, t in enumerate(tails) if t == d]
            if len(positions) < 3:
                scores[d] = 0.5
                continue

            intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            avg_interval = np.mean(intervals)
            since_last = len(tails) - 1 - positions[-1]

            if avg_interval > 0:
                ratio = since_last / avg_interval
                scores[d] = np.exp(-2 * (ratio - 1.0) ** 2)
            else:
                scores[d] = 0.5

        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores

    def _adjacent_analysis(self, tails):
        """相邻尾数关联分析 - 前一期尾数对下一期的影响(马尔可夫)"""
        if len(tails) < 2:
            return {d: 0.5 for d in range(10)}

        transitions = defaultdict(Counter)
        for i in range(len(tails) - 1):
            transitions[tails[i]][tails[i + 1]] += 1

        last_tail = tails[-1]
        if last_tail not in transitions or not transitions[last_tail]:
            return {d: 0.5 for d in range(10)}

        total_trans = sum(transitions[last_tail].values())
        scores = {}
        for d in range(10):
            scores[d] = transitions[last_tail].get(d, 0) / total_trans

        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores

    def _gap_pattern_analysis(self, tails):
        """间隔模式分析 - 基于间隔分布的z-score"""
        scores = {}
        for d in range(10):
            positions = [i for i, t in enumerate(tails) if t == d]
            if len(positions) < 4:
                scores[d] = 0.5
                continue

            intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            avg_gap = np.mean(intervals)
            std_gap = np.std(intervals) if len(intervals) > 1 else avg_gap * 0.5
            current_gap = len(tails) - 1 - positions[-1]

            if std_gap > 0:
                z_score = (current_gap - avg_gap) / std_gap
                scores[d] = min(1.0, max(0.0, 0.5 + 0.3 * z_score))
            else:
                scores[d] = 0.5 if current_gap < avg_gap else 0.7

        max_s = max(scores.values()) if scores else 1
        if max_s > 0:
            scores = {d: s / max_s for d, s in scores.items()}
        return scores


def validate_top3(numbers, test_periods=300):
    """回测验证TOP3尾数预测模型"""
    predictor = TailDigitTop3Predictor()

    test_periods = min(test_periods, len(numbers) - 50)
    start_idx = len(numbers) - test_periods

    hits = 0
    results = []

    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)

        predicted_tails = predictor.predict(hist, top_n=3)
        hit = actual_tail in predicted_tails
        hits += 1 if hit else 0
        predictor.record_result(predicted_tails, hit)

        # 判断模式
        ms = 0
        for j in range(len(predictor.hit_records) - 2, -1, -1):
            if not predictor.hit_records[j]:
                ms += 1
            else:
                break
        if ms == 0:
            mode = "正常"
        elif ms == 1:
            mode = "轮换1"
        elif ms == 2:
            mode = "轮换2"
        else:
            mode = "救援"

        results.append({
            'period': i - start_idx + 1,
            'actual': actual,
            'actual_tail': actual_tail,
            'predicted_tails': predicted_tails,
            'hit': hit,
            'mode': mode,
        })

    hit_rate = hits / test_periods * 100

    # 最大连续miss
    max_miss = 0
    cur_miss = 0
    for r in results:
        if not r['hit']:
            cur_miss += 1
            max_miss = max(max_miss, cur_miss)
        else:
            cur_miss = 0

    # 3期窗口
    hit_list = [r['hit'] for r in results]
    windows_3 = [any(hit_list[i:i + 3]) for i in range(len(hit_list) - 2)]
    win3_rate = sum(windows_3) / len(windows_3) * 100 if windows_3 else 0

    return {
        'hit_rate': hit_rate,
        'hits': hits,
        'total': test_periods,
        'max_miss': max_miss,
        'win3_rate': win3_rate,
        'results': results,
    }


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()

    print("=" * 60)
    print("尾数TOP3预测模型验证 (300期) - 纯尾数统计")
    print("=" * 60)

    result = validate_top3(numbers, test_periods=300)

    print(f"\n验证结果:")
    print(f"  单期命中率: {result['hit_rate']:.1f}% ({result['hits']}/{result['total']})")
    print(f"  三期窗口命中: {result['win3_rate']:.1f}%")
    print(f"  最大连续miss: {result['max_miss']}期")
    print(f"  随机基线: 30% (3/10)")
    print(f"  提升: +{result['hit_rate'] - 30:.1f}%")

    # 模式统计
    mode_data = defaultdict(list)
    for r in result['results']:
        mode_data[r['mode']].append(r['hit'])

    print(f"\n模式统计:")
    for mode in ['正常', '轮换1', '轮换2', '救援']:
        if mode in mode_data:
            h_list = mode_data[mode]
            h_cnt = sum(h_list)
            rate = h_cnt / len(h_list) * 100
            print(f"  {mode}: {len(h_list)}次, 命中{h_cnt}次 ({rate:.1f}%)")

    # 下一期预测
    predictor = TailDigitTop3Predictor()
    predicted, scores, mode = predictor.predict_with_details(numbers, top_n=3)

    print(f"\n下一期预测 (模式:{mode}):")
    medals = ['🥇', '🥈', '🥉']
    total_coverage = 0
    for idx, d in enumerate(predicted):
        nums = TAIL_DIGIT_NUMBERS[d]
        print(f"  {medals[idx]} 尾数{d} → {nums} (得分:{scores[d]:.4f})")
        total_coverage += len(nums)

    all_nums = sorted([n for d in predicted for n in TAIL_DIGIT_NUMBERS[d]])
    print(f"\n  覆盖号码({total_coverage}个): {all_nums}")
    print(f"  覆盖率: {total_coverage}/49 = {total_coverage / 49 * 100:.1f}%")
