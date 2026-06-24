"""
蒸馏TOP15 vs 基准TOP15 - 最近400期回测对比
固定15颗，不扩展
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from distilled_top15_predictor import DistilledTop15Predictor
from top15_predictor import Top15Predictor


def backtest(test_periods=400):
    print("=" * 80)
    print(f"蒸馏TOP15 vs 基准TOP15 - 最近{test_periods}期回测")
    print("=" * 80)

    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)

    if total < test_periods + 30:
        test_periods = total - 30
        print(f"数据不足，调整为{test_periods}期")

    distilled = DistilledTop15Predictor()
    baseline = Top15Predictor()

    start_idx = total - test_periods

    hits_d = []  # 蒸馏
    hits_b = []  # 基准
    details = []

    for i in range(start_idx, total):
        train_data = numbers[:i]
        actual = int(numbers[i])
        date = df.iloc[i]['date']

        # 蒸馏预测
        pred_d = distilled.predict(train_data)
        hit_d = actual in pred_d

        # 基准预测
        pred_b = baseline.predict(train_data)
        hit_b = actual in pred_b

        hits_d.append(hit_d)
        hits_b.append(hit_b)

        # 更新蒸馏模型状态
        distilled.update(pred_d, actual)

        details.append({
            'seq': i - start_idx + 1,
            'date': date,
            'actual': actual,
            'hit_d': hit_d,
            'hit_b': hit_b,
            'pred_d': pred_d,
            'pred_b': pred_b,
        })

        if (i - start_idx + 1) % 100 == 0:
            done = i - start_idx + 1
            hd = sum(hits_d)
            hb = sum(hits_b)
            print(f"  进度: {done}/{test_periods}  "
                  f"蒸馏:{hd}/{done}={hd/done*100:.1f}%  "
                  f"基准:{hb}/{done}={hb/done*100:.1f}%")

    # ========== 总体统计 ==========
    total_d = sum(hits_d)
    total_b = sum(hits_b)
    rate_d = total_d / test_periods * 100
    rate_b = total_b / test_periods * 100

    print(f"\n{'=' * 80}")
    print(f"总体统计对比")
    print(f"{'=' * 80}")
    print(f"{'指标':<20}  {'蒸馏TOP15':>10}  {'基准TOP15':>10}  {'差异':>10}")
    print("-" * 60)
    print(f"{'命中次数':<20}  {total_d:>10}  {total_b:>10}  {total_d - total_b:>+10}")
    print(f"{'命中率':<20}  {rate_d:>9.2f}%  {rate_b:>9.2f}%  {rate_d - rate_b:>+9.2f}%")
    print(f"{'理论随机':<20}  {'':>10}  {15/49*100:>9.2f}%")

    # ========== 分段对比 ==========
    print(f"\n{'=' * 80}")
    print(f"分段命中率对比 (每50期)")
    print(f"{'=' * 80}")
    print(f"{'区间':<28}  {'蒸馏':>8}  {'基准':>8}  {'差异':>8}")
    print("-" * 60)
    for seg_start in range(0, test_periods, 50):
        seg_end = min(seg_start + 50, test_periods)
        seg_d = sum(hits_d[seg_start:seg_end])
        seg_b = sum(hits_b[seg_start:seg_end])
        seg_total = seg_end - seg_start
        rd = seg_d / seg_total * 100
        rb = seg_b / seg_total * 100
        sd = details[seg_start]['date']
        ed = details[seg_end - 1]['date']
        print(f"  {seg_start+1:>3}-{seg_end:>3}期 ({sd}~{ed})  "
              f"{rd:>6.1f}%  {rb:>6.1f}%  {rd - rb:>+6.1f}%")

    # ========== 连续不中分析 ==========
    def calc_streaks(hit_list):
        streaks = []
        streak_start = None
        for i, hit in enumerate(hit_list):
            if not hit:
                if streak_start is None:
                    streak_start = i
            else:
                if streak_start is not None:
                    length = i - streak_start
                    streaks.append({
                        'start': streak_start,
                        'end': i - 1,
                        'length': length,
                        'start_date': details[streak_start]['date'],
                        'end_date': details[i - 1]['date'],
                        'numbers': [details[j]['actual'] for j in range(streak_start, i)]
                    })
                    streak_start = None
        if streak_start is not None:
            length = len(hit_list) - streak_start
            streaks.append({
                'start': streak_start,
                'end': len(hit_list) - 1,
                'length': length,
                'start_date': details[streak_start]['date'],
                'end_date': details[len(hit_list) - 1]['date'],
                'numbers': [details[j]['actual'] for j in range(streak_start, len(hit_list))]
            })
        return streaks

    streaks_d = calc_streaks(hits_d)
    streaks_b = calc_streaks(hits_b)

    max_d = max(s['length'] for s in streaks_d) if streaks_d else 0
    max_b = max(s['length'] for s in streaks_b) if streaks_b else 0
    avg_d = sum(s['length'] for s in streaks_d) / len(streaks_d) if streaks_d else 0
    avg_b = sum(s['length'] for s in streaks_b) / len(streaks_b) if streaks_b else 0

    ge3_d = sum(1 for s in streaks_d if s['length'] >= 3)
    ge3_b = sum(1 for s in streaks_b if s['length'] >= 3)
    ge5_d = sum(1 for s in streaks_d if s['length'] >= 5)
    ge5_b = sum(1 for s in streaks_b if s['length'] >= 5)
    ge7_d = sum(1 for s in streaks_d if s['length'] >= 7)
    ge7_b = sum(1 for s in streaks_b if s['length'] >= 7)

    print(f"\n{'=' * 80}")
    print(f"连续不中分析")
    print(f"{'=' * 80}")
    print(f"{'指标':<24}  {'蒸馏':>8}  {'基准':>8}  {'改善':>8}")
    print("-" * 56)
    print(f"{'连续不中段数':<24}  {len(streaks_d):>8}  {len(streaks_b):>8}")
    print(f"{'最长连续不中':<24}  {max_d:>8}  {max_b:>8}  {max_b - max_d:>+8}")
    print(f"{'平均连续不中':<24}  {avg_d:>8.2f}  {avg_b:>8.2f}  {avg_b - avg_d:>+8.2f}")
    print(f"{'≥3期连败次数':<24}  {ge3_d:>8}  {ge3_b:>8}  {ge3_b - ge3_d:>+8}")
    print(f"{'≥5期连败次数':<24}  {ge5_d:>8}  {ge5_b:>8}  {ge5_b - ge5_d:>+8}")
    print(f"{'≥7期连败次数':<24}  {ge7_d:>8}  {ge7_b:>8}  {ge7_b - ge7_d:>+8}")

    # 连续不中长度分布
    all_lengths = set([s['length'] for s in streaks_d] + [s['length'] for s in streaks_b])
    dist_d = Counter(s['length'] for s in streaks_d)
    dist_b = Counter(s['length'] for s in streaks_b)

    print(f"\n连续不中长度分布:")
    print(f"{'长度':>6}  {'蒸馏':>8}  {'基准':>8}")
    print("-" * 28)
    for length in sorted(all_lengths):
        print(f"  {length:>2}期  {dist_d.get(length, 0):>6}次  {dist_b.get(length, 0):>6}次")

    # ========== 所有连续不中段明细 ==========
    print(f"\n{'=' * 80}")
    print(f"蒸馏模型 - 所有连续不中段明细 (按时间顺序)")
    print(f"{'=' * 80}")
    print(f"{'序号':>4}  {'起始':>6}  {'结束':>6}  {'长度':>4}  {'起始日期':<14}  {'结束日期':<14}  实际号码")
    print("-" * 90)
    for idx, s in enumerate(streaks_d, 1):
        nums = ','.join(str(n) for n in s['numbers'])
        print(f"  {idx:>3}  {s['start']+1:>6}  {s['end']+1:>6}  {s['length']:>4}  "
              f"{s['start_date']:<14}  {s['end_date']:<14}  {nums}")

    # ========== ≥5期详细 ==========
    long_d = [s for s in streaks_d if s['length'] >= 5]
    print(f"\n{'=' * 80}")
    print(f"重点关注：连续不中 ≥ 5期 ({len(long_d)}次)")
    print(f"{'=' * 80}")
    for idx, s in enumerate(long_d, 1):
        print(f"\n--- 第{idx}段: 连续{s['length']}期不中 "
              f"(第{s['start']+1}-{s['end']+1}期, {s['start_date']}~{s['end_date']}) ---")
        for j in range(s['start'], s['end'] + 1):
            d = details[j]
            pred_str = str(d['pred_d'][:8]) + '...'
            b_status = "✓基准中" if d['hit_b'] else "✗基准也没中"
            print(f"  第{d['seq']}期({d['date']}): 实际={d['actual']:>2}, "
                  f"TOP15={pred_str}, {b_status}")

    # ========== 策略差异分析 ==========
    both_hit = sum(1 for i in range(test_periods) if hits_d[i] and hits_b[i])
    only_d = sum(1 for i in range(test_periods) if hits_d[i] and not hits_b[i])
    only_b = sum(1 for i in range(test_periods) if not hits_d[i] and hits_b[i])
    both_miss = sum(1 for i in range(test_periods) if not hits_d[i] and not hits_b[i])

    print(f"\n{'=' * 80}")
    print(f"策略差异分析")
    print(f"{'=' * 80}")
    print(f"  两者都命中:     {both_hit}次")
    print(f"  蒸馏独有命中:   {only_d}次 ⭐(策略贡献)")
    print(f"  基准独有命中:   {only_b}次 (策略代价)")
    print(f"  两者都未中:     {both_miss}次")
    print(f"  净收益:         {only_d - only_b:+}次")

    print(f"\n回测完成。")


if __name__ == '__main__':
    backtest()
