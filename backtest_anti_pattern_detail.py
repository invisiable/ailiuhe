"""
策略5:反模式 - 详细400期回测
列出所有连续不中情况
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from top15_predictor import Top15Predictor
from precise_top15_predictor import PreciseTop15Predictor
from smart_top15_predictor import SmartTop15Predictor


class AntiPatternPredictor:
    """反模式策略预测器"""
    
    def __init__(self):
        self.precise_predictor = PreciseTop15Predictor()
        self.consecutive_misses = 0
        self.recent_predictions = []
        self.recent_actuals = []
    
    def update(self, prediction, actual):
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
        base_pred = self.precise_predictor.predict(numbers)
        
        if len(self.recent_actuals) < 5:
            return base_pred[:15]
        
        # 分析最近miss的号码特征
        recent_misses = []
        for pred, actual in zip(self.recent_predictions[-10:], self.recent_actuals[-10:]):
            if actual not in pred:
                recent_misses.append(actual)
        
        if not recent_misses:
            return base_pred[:15]
        
        # 分析miss号码的区间分布
        miss_ranges = Counter()
        for n in recent_misses:
            if n <= 10: miss_ranges['1-10'] += 1
            elif n <= 20: miss_ranges['11-20'] += 1
            elif n <= 30: miss_ranges['21-30'] += 1
            elif n <= 40: miss_ranges['31-40'] += 1
            else: miss_ranges['41-49'] += 1
        
        # 找到模型盲区范围（最容易miss的2个区间）
        blind_spots = [r for r, c in miss_ranges.most_common(2)]
        
        # 从盲区范围选择号码
        blind_candidates = []
        range_map = {'1-10': (1,10), '11-20': (11,20), '21-30': (21,30), 
                     '31-40': (31,40), '41-49': (41,49)}
        
        recent_5 = set(numbers_list[-5:]) if len(numbers_list) >= 5 else set()
        
        for r in blind_spots:
            start, end = range_map[r]
            for n in range(start, end+1):
                if n not in recent_5 and n not in base_pred[:10]:
                    blind_candidates.append(n)
        
        # 用盲区号码替换低优先级的基础预测
        inject_count = min(4, len(blind_candidates))
        result = base_pred[:15 - inject_count]
        
        # 确定性选择（基于数据特征）
        np.random.seed(int(numbers_list[-1]) + len(numbers_list))
        if blind_candidates:
            chosen = list(np.random.choice(blind_candidates, 
                                           size=min(inject_count, len(blind_candidates)), 
                                           replace=False))
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


def backtest_anti_pattern_400():
    print("=" * 80)
    print("策略5:反模式预测器 - 最近400期详细回测")
    print("=" * 80)

    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)
    test_periods = 400
    start_idx = total - test_periods

    predictor = AntiPatternPredictor()
    # 对比用基准
    baseline = PreciseTop15Predictor()

    hits = []
    baseline_hits = []
    details = []

    for i in range(start_idx, total):
        train_data = numbers[:i]
        actual = int(numbers[i])

        # 反模式预测
        top15 = predictor.predict(train_data)
        is_hit = actual in top15
        predictor.update(top15, actual)

        # 基准预测
        base_top15 = baseline.predict(train_data)
        base_hit = actual in base_top15
        baseline.update_performance(base_top15, actual)

        rank = top15.index(actual) + 1 if is_hit else -1
        hits.append(is_hit)
        baseline_hits.append(base_hit)

        details.append({
            'seq': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'hit': is_hit,
            'base_hit': base_hit,
            'rank': rank,
            'top15': top15
        })

        if (i - start_idx + 1) % 100 == 0:
            done = i - start_idx + 1
            hit_count = sum(hits)
            base_count = sum(baseline_hits)
            print(f"  进度: {done}/{test_periods}  反模式:{hit_count}/{done}={hit_count/done*100:.1f}%  基准:{base_count}/{done}={base_count/done*100:.1f}%")

    # ========== 总体统计 ==========
    total_hits = sum(hits)
    base_total = sum(baseline_hits)
    print(f"\n{'=' * 80}")
    print(f"总体统计对比")
    print(f"{'=' * 80}")
    print(f"{'指标':<16} {'反模式':>10} {'基准(Precise)':>14} {'差异':>10}")
    print("-" * 55)
    print(f"{'命中次数':<16} {total_hits:>10} {base_total:>14} {total_hits-base_total:>+10}")
    print(f"{'命中率':<16} {total_hits/test_periods*100:>9.2f}% {base_total/test_periods*100:>13.2f}% {(total_hits-base_total)/test_periods*100:>+9.2f}%")
    print(f"{'理论随机':<16} {'':>10} {15/49*100:>13.2f}%")

    # ========== 分段统计 ==========
    print(f"\n{'=' * 80}")
    print(f"分段命中率对比 (每50期)")
    print(f"{'=' * 80}")
    print(f"{'区间':<20} {'反模式':>10} {'基准':>10} {'差异':>8}")
    print("-" * 55)
    for seg_start in range(0, test_periods, 50):
        seg_end = min(seg_start + 50, test_periods)
        seg_hits = sum(hits[seg_start:seg_end])
        seg_base = sum(baseline_hits[seg_start:seg_end])
        seg_total = seg_end - seg_start
        start_date = details[seg_start]['date']
        end_date = details[seg_end - 1]['date']
        print(f"  {seg_start+1:>3}-{seg_end:>3}期 ({start_date[:7]}~{end_date[:7]})  "
              f"{seg_hits/seg_total*100:>5.1f}%  {seg_base/seg_total*100:>5.1f}%  "
              f"{(seg_hits-seg_base)/seg_total*100:>+5.1f}%")

    # ========== 连续不中分析 ==========
    print(f"\n{'=' * 80}")
    print(f"连续不中分析")
    print(f"{'=' * 80}")

    # 反模式连续不中
    streaks = []
    streak_start = None
    for i, d in enumerate(details):
        if not d['hit']:
            if streak_start is None:
                streak_start = i
        else:
            if streak_start is not None:
                length = i - streak_start
                streaks.append({
                    'start_seq': details[streak_start]['seq'],
                    'end_seq': details[i - 1]['seq'],
                    'length': length,
                    'start_date': details[streak_start]['date'],
                    'end_date': details[i - 1]['date'],
                    'start_idx': streak_start,
                    'end_idx': i - 1
                })
                streak_start = None
    if streak_start is not None:
        length = len(details) - streak_start
        streaks.append({
            'start_seq': details[streak_start]['seq'],
            'end_seq': details[-1]['seq'],
            'length': length,
            'start_date': details[streak_start]['date'],
            'end_date': details[-1]['date'],
            'start_idx': streak_start,
            'end_idx': len(details) - 1
        })

    # 基准连续不中
    base_streaks = []
    streak_start = None
    for i, d in enumerate(details):
        if not d['base_hit']:
            if streak_start is None:
                streak_start = i
        else:
            if streak_start is not None:
                base_streaks.append(i - streak_start)
                streak_start = None
    if streak_start is not None:
        base_streaks.append(len(details) - streak_start)

    streak_lengths = [s['length'] for s in streaks]
    max_streak = max(streak_lengths) if streak_lengths else 0
    avg_streak = sum(streak_lengths) / len(streak_lengths) if streak_lengths else 0
    base_max = max(base_streaks) if base_streaks else 0
    base_avg = sum(base_streaks) / len(base_streaks) if base_streaks else 0

    print(f"{'指标':<20} {'反模式':>10} {'基准':>10} {'改善':>10}")
    print("-" * 55)
    print(f"{'连续不中段数':<20} {len(streaks):>10} {len(base_streaks):>10}")
    print(f"{'最长连续不中':<20} {max_streak:>10} {base_max:>10} {base_max-max_streak:>+10}")
    print(f"{'平均连续不中':<20} {avg_streak:>9.2f} {base_avg:>9.2f} {base_avg-avg_streak:>+9.2f}")
    print(f"{'≥3期连败次数':<20} {sum(1 for s in streak_lengths if s>=3):>10} {sum(1 for s in base_streaks if s>=3):>10} {sum(1 for s in base_streaks if s>=3)-sum(1 for s in streak_lengths if s>=3):>+10}")
    print(f"{'≥5期连败次数':<20} {sum(1 for s in streak_lengths if s>=5):>10} {sum(1 for s in base_streaks if s>=5):>10} {sum(1 for s in base_streaks if s>=5)-sum(1 for s in streak_lengths if s>=5):>+10}")
    print(f"{'≥7期连败次数':<20} {sum(1 for s in streak_lengths if s>=7):>10} {sum(1 for s in base_streaks if s>=7):>10} {sum(1 for s in base_streaks if s>=7)-sum(1 for s in streak_lengths if s>=7):>+10}")

    # 连续不中长度分布
    print(f"\n连续不中长度分布:")
    length_dist = Counter(streak_lengths)
    base_dist = Counter(base_streaks)
    all_lengths = sorted(set(streak_lengths + base_streaks))
    print(f"{'长度':<6} {'反模式':>8} {'基准':>8}")
    print("-" * 25)
    for length in all_lengths:
        print(f"  {length:>2}期  {length_dist.get(length, 0):>6}次  {base_dist.get(length, 0):>6}次")

    # ========== 所有连续不中段明细 ==========
    print(f"\n{'=' * 80}")
    print(f"所有连续不中段明细 (按时间顺序)")
    print(f"{'=' * 80}")
    print(f"{'序号':>4}  {'起始':>4}  {'结束':>4}  {'长度':>4}  {'起始日期':<12}  {'结束日期':<12}  实际号码")
    print("-" * 100)

    for idx, s in enumerate(streaks, 1):
        miss_nums = [details[j]['actual'] for j in range(s['start_idx'], s['end_idx'] + 1)]
        miss_str = ','.join(str(n) for n in miss_nums)
        print(f"  {idx:>3}  {s['start_seq']:>4}  {s['end_seq']:>4}  {s['length']:>4}  "
              f"{s['start_date']:<12}  {s['end_date']:<12}  {miss_str}")

    # ========== 重点：≥5期连续不中 ==========
    long_streaks = [s for s in streaks if s['length'] >= 5]
    print(f"\n{'=' * 80}")
    print(f"重点关注：连续不中 ≥ 5期 ({len(long_streaks)}次)")
    print(f"{'=' * 80}")
    for idx, s in enumerate(long_streaks, 1):
        print(f"\n--- 第{idx}段: 连续{s['length']}期不中 (第{s['start_seq']}-{s['end_seq']}期, {s['start_date']}~{s['end_date']}) ---")
        for j in range(s['start_idx'], s['end_idx'] + 1):
            d = details[j]
            base_mark = "✓基准中" if d['base_hit'] else "✗基准也没中"
            print(f"  第{d['seq']}期({d['date']}): 实际={d['actual']:>2}, "
                  f"TOP15={d['top15'][:8]}..., {base_mark}")

    # ========== 反模式独有命中 vs 独有miss ==========
    print(f"\n{'=' * 80}")
    print(f"策略差异分析")
    print(f"{'=' * 80}")
    only_anti_hit = sum(1 for h, bh in zip(hits, baseline_hits) if h and not bh)
    only_base_hit = sum(1 for h, bh in zip(hits, baseline_hits) if not h and bh)
    both_hit = sum(1 for h, bh in zip(hits, baseline_hits) if h and bh)
    both_miss = sum(1 for h, bh in zip(hits, baseline_hits) if not h and not bh)

    print(f"  两者都命中:     {both_hit}次")
    print(f"  反模式独有命中: {only_anti_hit}次 ⭐(策略贡献)")
    print(f"  基准独有命中:   {only_base_hit}次 (策略代价)")
    print(f"  两者都未中:     {both_miss}次")
    print(f"  净收益:         {only_anti_hit - only_base_hit:+d}次")

    print(f"\n回测完成。")


if __name__ == '__main__':
    backtest_anti_pattern_400()
