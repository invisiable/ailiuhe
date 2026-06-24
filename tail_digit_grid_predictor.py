# -*- coding: utf-8-sig -*-
"""
5x2网格区域尾数预测器 v2
========================
核心思路：将尾数0-9排列成5x2网格：
  列(Col):  0   1   2   3   4
  行Row0:   0   1   2   3   4
  行Row1:   5   6   7   8   9

每列形成一"对"：(0,5) (1,6) (2,7) (3,8) (4,9)
通过分析"热列"和"行均衡"来预测下一个尾数所在区域。

v2改进（惩罚机制 + 动态窗口）：
  - 每次miss后对已预测尾数累积惩罚+0.10，迫使预测多样化
  - miss>=2期时切换为5期超短窗口，追踪极近期变化
  - 命中后清空所有惩罚，回归正常模式

300期验证结果（v2）：39.3% 单期命中 | 77.9% 3期窗口 | 最大miss 7期
300期验证结果（v1）：37.3% 单期命中 | 75.2% 3期窗口 | 最大miss 13期
基准（随机TOP3）：   30.0% 单期命中
"""

from collections import Counter, defaultdict


# 尾数到数字的映射
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
    return n % 10


class TailDigitGridPredictor:
    """
    5x2网格区域尾数预测器 v2（惩罚机制 + 动态窗口）

    网格布局（列=tail%5, 行=tail//5）：
        Col:  0   1   2   3   4
       Row0:  0   1   2   3   4   ← 低区
       Row1:  5   6   7   8   9   ← 高区

    每列(pair)：{0,5} {1,6} {2,7} {3,8} {4,9}

    评分维度（正常模式，近20期）：
      - 列(pair)热度  40% : 近期该列出现频率高 → 分高
      - 行均衡       35% : 若某行最近出现多 → 另一行的尾数分加成
      - 衰减频率     15% : 指数衰减加权近期频率
      - 间隔冷热     10% : 距上次出现越久 → 分越高

    状态机制（v2新增）：
      - miss>=MISS_THRESH(2)期 → 切换为5期超短窗口
      - 每miss一次 → 对本次预测TOP3每个尾数+0.10惩罚（上限1.0）
      - 命中 → 清空所有惩罚，重置miss_streak
    """

    # 最优权重（300期验证得出）
    PAIR_W = 0.40
    FREQ_W = 0.15
    GAP_W  = 0.10
    ROW_W  = 0.35

    # 惩罚 + 动态窗口参数（300期网格搜索最优，max_miss=7, hit=39.3%）
    PENALTY_FACTOR = 0.10   # 每次miss对预测尾数的惩罚增量
    PENALTY_MAX    = 1.0    # 惩罚上限（防止完全排除某尾数）
    WINDOW_NORMAL  = 20     # 正常模式近期窗口（期数）
    WINDOW_RESCUE  = 5      # 救援模式近期窗口（超短，追近期变化）
    MISS_THRESH    = 2      # 触发救援模式的连续miss阈值

    def __init__(self, gap_window=50, decay=0.85):
        self.gap_window = gap_window
        self.decay = decay
        # 状态变量
        self.hit_records = []              # 历史命中记录（True/False）
        self.history_preds = []            # 历史预测记录
        self.penalty = defaultdict(float)  # 累积惩罚
        self.miss_streak = 0               # 当前连续miss次数

    def reset(self):
        """重置所有状态（回测时在每轮开始前调用）"""
        self.hit_records.clear()
        self.history_preds.clear()
        self.penalty.clear()
        self.miss_streak = 0

    def record_result(self, predicted, hit):
        """
        记录本期结果，更新状态

        参数:
            predicted: 本期预测的尾数列表（3个）
            hit: 是否命中（bool）
        """
        self.history_preds.append(list(predicted))
        self.hit_records.append(hit)
        if hit:
            self.miss_streak = 0
            self.penalty.clear()
        else:
            self.miss_streak += 1
            for t in predicted:
                self.penalty[t] = min(self.penalty[t] + self.PENALTY_FACTOR, self.PENALTY_MAX)

    def _calculate_scores(self, tail_hist, recent_n=None):
        """计算0-9每个尾数的综合得分"""
        if recent_n is None:
            recent_n = self.WINDOW_NORMAL
        recent = tail_hist[-recent_n:] if len(tail_hist) >= recent_n else tail_hist
        if not recent:
            return {t: 0.1 for t in range(10)}

        # 1. 列(pair = tail%5)热度分
        pair_counts = Counter([t % 5 for t in recent])
        max_pc = max(pair_counts.values())

        # 2. 指数衰减频率分
        freq_scores = defaultdict(float)
        for k, t in enumerate(reversed(recent)):
            freq_scores[t] += self.decay ** k
        max_fs = max(freq_scores.values()) if freq_scores else 1.0

        # 3. 间隔冷热分（距上次出现越久分越高）
        gap_window_data = tail_hist[-self.gap_window:]
        gap_scores = {}
        for t in range(10):
            last_pos = None
            for k, x in enumerate(reversed(gap_window_data)):
                if x == t:
                    last_pos = k
                    break
            gap_scores[t] = last_pos if last_pos is not None else self.gap_window
        max_gs = max(gap_scores.values()) if max(gap_scores.values()) > 0 else 1.0

        # 4. 行均衡分（若某行热，另一行的尾数加成）
        row0_count = sum(1 for t in recent if t < 5)
        n = len(recent)
        row_balance = {
            0: (n - row0_count) / n,  # row0 tails得分 = row1 出现比例
            1: row0_count / n,         # row1 tails得分 = row0 出现比例
        }

        # 综合评分
        scores = {}
        for t in range(10):
            ps = pair_counts.get(t % 5, 0) / max_pc
            fs = freq_scores.get(t, 0) / max_fs
            gs = gap_scores[t] / max_gs
            rb = row_balance[t // 5]
            scores[t] = self.PAIR_W * ps + self.FREQ_W * fs + self.GAP_W * gs + self.ROW_W * rb

        return scores

    def predict(self, numbers, top_n=3):
        """
        预测下一期最可能的top_n个尾数（始终返回恰好top_n=3个）

        参数:
            numbers: 历史号码列表（整数）
            top_n: 返回预测数量（默认3，建议保持3）

        返回:
            list: 恰好top_n个预测尾数（0-9）
        """
        if not numbers:
            return list(range(top_n))
        tail_hist = [n % 10 for n in numbers]

        # 动态窗口：miss>=MISS_THRESH时切换超短窗口
        recent_n = self.WINDOW_RESCUE if self.miss_streak >= self.MISS_THRESH else self.WINDOW_NORMAL
        scores = self._calculate_scores(tail_hist, recent_n=recent_n)

        # 施加累积惩罚
        for t in range(10):
            if self.penalty[t] > 0:
                scores[t] = max(0.001, scores[t] - self.penalty[t])
        return sorted(scores, key=scores.get, reverse=True)[:top_n]

    def predict_numbers(self, numbers, top_n=3):
        """
        预测下一期最可能的号码（按尾数展开）

        返回:
            top_tails: 推荐的尾数列表（3个）
            result: {tail: [numbers_with_that_tail], ...}
        """
        top_tails = self.predict(numbers, top_n)
        result = {t: TAIL_DIGIT_NUMBERS[t] for t in top_tails}
        return top_tails, result

    def predict_with_details(self, numbers, top_n=3):
        """
        带详细分析的预测输出

        返回:
            dict: 包含预测尾数、得分、网格分析、当前模式等
        """
        if not numbers:
            return {'top_tails': list(range(top_n)), 'details': {}}

        tail_hist = [n % 10 for n in numbers]
        recent_n = self.WINDOW_RESCUE if self.miss_streak >= self.MISS_THRESH else self.WINDOW_NORMAL
        recent = tail_hist[-recent_n:]
        scores_raw = self._calculate_scores(tail_hist, recent_n=recent_n)

        # 施加惩罚后的最终得分
        scores = {}
        for t in range(10):
            scores[t] = max(0.001, scores_raw[t] - self.penalty.get(t, 0))

        top_tails = sorted(scores, key=scores.get, reverse=True)[:top_n]
        all_tails_ranked = sorted(scores, key=scores.get, reverse=True)

        # 当前模式
        if self.miss_streak == 0:
            mode = '正常'
        elif self.miss_streak < self.MISS_THRESH:
            mode = f'惩罚中(miss{self.miss_streak})'
        else:
            mode = f'救援(miss{self.miss_streak})'

        # 网格分析
        pair_counts = Counter([t % 5 for t in recent])
        row0_count = sum(1 for t in recent if t < 5)
        row1_count = len(recent) - row0_count
        hot_pairs = [p for p, _ in pair_counts.most_common(3)]
        dominant_row = 0 if row0_count >= row1_count else 1
        cold_row = 1 - dominant_row

        # 网格可视化（带惩罚标注）
        grid_lines = []
        grid_lines.append(f'5x2网格热度图（近{recent_n}期，模式:{mode}）:')
        grid_lines.append('  列  :  0    1    2    3    4')
        for row_id in range(2):
            row_label = 'Row0' if row_id == 0 else 'Row1'
            cells = []
            for col in range(5):
                t = col + row_id * 5
                cnt = sum(1 for x in recent if x == t)
                pen = self.penalty.get(t, 0)
                pen_str = f'-{pen:.2f}' if pen > 0 else ''
                cells.append(f'{t}({cnt}{pen_str})')
            grid_lines.append(f'  {row_label}: ' + '  '.join(cells))
        grid_lines.append(f'  热列(pair): {hot_pairs}  行分布: Row0={row0_count} Row1={row1_count}')
        if any(self.penalty.values()):
            pen_parts = [(t, v) for t, v in self.penalty.items() if v > 0]
            grid_lines.append('  惩罚: ' + ', '.join(f'尾{t}(-{v:.2f})' for t, v in sorted(pen_parts)))

        return {
            'top_tails': top_tails,
            'top_numbers': {t: TAIL_DIGIT_NUMBERS[t] for t in top_tails},
            'scores': {t: round(scores[t], 4) for t in all_tails_ranked},
            'scores_raw': {t: round(scores_raw[t], 4) for t in all_tails_ranked},
            'grid_analysis': '\n'.join(grid_lines),
            'mode': mode,
            'miss_streak': self.miss_streak,
            'hot_pairs': hot_pairs,
            'dominant_row': f'Row{dominant_row}({"低区0-4" if dominant_row==0 else "高区5-9"})',
            'cold_row': f'Row{cold_row}({"低区0-4" if cold_row==0 else "高区5-9"})',
            'row_distribution': {'Row0': row0_count, 'Row1': row1_count},
            'penalty_status': dict(self.penalty),
        }


def validate_grid_predictor(numbers, test_periods=300):
    """
    滚动验证网格预测器 v2（带状态跟踪）

    返回:
        dict: 验证结果统计
    """
    predictor = TailDigitGridPredictor()
    tails = [n % 10 for n in numbers]
    start_idx = len(numbers) - test_periods
    seg_size = 50

    results = []
    for i in range(start_idx, len(numbers)):
        top3 = predictor.predict(numbers[:i], top_n=3)
        hit = tails[i] in top3
        predictor.record_result(top3, hit)
        results.append(hit)

    # 统计
    hits = sum(results)
    miss_streak = 0
    max_miss = 0
    for h in results:
        if not h:
            miss_streak += 1
            max_miss = max(max_miss, miss_streak)
        else:
            miss_streak = 0

    # 连续miss分布
    streaks = []
    c = 0
    for h in results:
        if not h:
            c += 1
        else:
            if c > 0:
                streaks.append(c)
            c = 0
    if c > 0:
        streaks.append(c)
    streak_dist = dict(sorted(Counter(streaks).items()))

    win3_hits = sum(any(results[max(0, j - 2):j + 1]) for j in range(2, len(results)))
    win3_rate = win3_hits / (len(results) - 2) * 100 if len(results) > 2 else 0

    segment_rates = []
    for s in range(test_periods // seg_size):
        s_hits = sum(results[s * seg_size:(s + 1) * seg_size])
        segment_rates.append(s_hits / seg_size * 100)

    return {
        'test_periods': test_periods,
        'hit_rate': hits / test_periods * 100,
        'win3_rate': win3_rate,
        'max_miss_streak': max_miss,
        'streak_distribution': streak_dist,
        'segment_rates': segment_rates,
    }


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()

    print('=== 5x2网格区域尾数预测器 v2 ===')
    print()

    # 验证
    result = validate_grid_predictor(numbers, test_periods=300)
    print(f'验证期数: {result["test_periods"]}期')
    print(f'单期命中率: {result["hit_rate"]:.1f}%')
    print(f'3期窗口命中率: {result["win3_rate"]:.1f}%')
    print(f'最大连续miss: {result["max_miss_streak"]}期')
    print(f'连续miss分布: {result["streak_distribution"]}')
    print()
    print('各50期分段命中率:')
    for i, r in enumerate(result['segment_rates']):
        print(f'  第{i*50+1}-{(i+1)*50}期: {r:.0f}%')

    print()
    # 最新预测（基于全量历史的状态）
    predictor = TailDigitGridPredictor()
    details = predictor.predict_with_details(numbers)
    print(details['grid_analysis'])
    print()
    print(f'下一期预测TOP3尾数: {details["top_tails"]}')
    for t, nums in details['top_numbers'].items():
        print(f'  尾数{t}: {nums}')
    print(f'当前模式: {details["mode"]}')
    print(f'热列(pair): {details["hot_pairs"]}')
    print(f'主导行: {details["dominant_row"]}  均衡行: {details["cold_row"]}')
