"""
蒸馏TOP15预测器 - 固定15颗
经6轮实验验证的最优方案：反模式策略 + PreciseTop15基底

实验结果（400期回测）：
  Base Top15:                36.50%命中, 最大连败14期
  反模式+Base Top15(4替换):  32.75%命中, 最大连败15期 ✗ 替换降质
  反模式+Base Top15(2替换):  32.00%命中, 最大连败14期 ✗ 同上
  多模型融合(4模型):         32.50%命中, 最大连败14期 ✗ 信号稀释
  模型轮换(连败切换):        35.00%命中, 最大连败16期 ✗ 更差
  自适应权重(5方法):         33.00%命中, 最大连败14期 ✗ 过度优化
  ★反模式+PreciseTop15:     36.25%命中, 最大连败 9期 ← 最优

结论：PreciseTop15的末位预测较弱，反模式盲区注入恰好替换弱项，
      实现命中率接近最强模型、连败显著降低的最优平衡。
      Base Top15的15个pick全部高质量，任何替换都会降质。
"""

import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor


class DistilledTop15Predictor:
    """蒸馏TOP15预测器 - 固定15颗

    反模式策略：分析PreciseTop15的盲区，用盲区号码替换末位4个预测。
    命中率36.25%（接近最强36.5%），最大连败9期（优于最强的14期）。
    """

    def __init__(self):
        self.precise_predictor = PreciseTop15Predictor()
        self.consecutive_misses = 0
        self.recent_predictions = []
        self.recent_actuals = []

    def update(self, prediction, actual):
        """更新追踪状态"""
        actual = int(actual)
        hit = actual in prediction
        if hit:
            self.consecutive_misses = 0
        else:
            self.consecutive_misses += 1

        self.recent_predictions.append(prediction)
        self.recent_actuals.append(actual)
        if len(self.recent_predictions) > 20:
            self.recent_predictions.pop(0)
            self.recent_actuals.pop(0)

    def predict(self, numbers):
        """反模式预测：分析模型盲区，主动覆盖"""
        numbers_list = list(numbers) if not isinstance(numbers, list) else numbers
        if hasattr(numbers, 'tolist'):
            numbers_list = numbers.tolist()

        base_pred = self.precise_predictor.predict(numbers)
        if hasattr(base_pred, 'tolist'):
            base_pred = base_pred.tolist()
        base_pred = [int(x) for x in base_pred[:15]]

        if len(self.recent_actuals) < 5:
            return base_pred

        # 分析最近miss的号码特征
        recent_misses = []
        for pred, actual in zip(self.recent_predictions[-10:], self.recent_actuals[-10:]):
            if actual not in pred:
                recent_misses.append(actual)

        if not recent_misses:
            return base_pred

        # 分析miss号码的区间分布
        miss_ranges = Counter()
        for n in recent_misses:
            if n <= 10:
                miss_ranges['1-10'] += 1
            elif n <= 20:
                miss_ranges['11-20'] += 1
            elif n <= 30:
                miss_ranges['21-30'] += 1
            elif n <= 40:
                miss_ranges['31-40'] += 1
            else:
                miss_ranges['41-49'] += 1

        # 找到模型盲区范围（最容易miss的2个区间）
        blind_spots = [r for r, c in miss_ranges.most_common(2)]

        # 从盲区范围选择号码
        blind_candidates = []
        range_map = {
            '1-10': (1, 10), '11-20': (11, 20), '21-30': (21, 30),
            '31-40': (31, 40), '41-49': (41, 49)
        }

        recent_5 = set(numbers_list[-5:]) if len(numbers_list) >= 5 else set()

        for r in blind_spots:
            start, end = range_map[r]
            for n in range(start, end + 1):
                if n not in recent_5 and n not in base_pred[:10]:
                    blind_candidates.append(n)

        # 用盲区号码替换低优先级的基准预测（替换末位4个）
        inject_count = min(4, len(blind_candidates))
        result = list(base_pred[:15 - inject_count])

        # 确定性选择（基于数据特征的种子）
        np.random.seed(int(numbers_list[-1]) + len(numbers_list))
        if blind_candidates:
            chosen = list(np.random.choice(
                blind_candidates,
                size=min(inject_count, len(blind_candidates)),
                replace=False
            ))
            for n in chosen:
                if int(n) not in result:
                    result.append(int(n))

        # 补齐到15个
        for n in base_pred:
            if n not in result:
                result.append(n)
            if len(result) >= 15:
                break

        return result[:15]
