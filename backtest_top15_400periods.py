"""
TOP15蒸馏模型 - 最近400期回测
分析命中情况和连续不中期数
"""

import pandas as pd
import numpy as np
from collections import Counter
from top15_predictor import Top15Predictor


def backtest_top15(test_periods=400):
    print("=" * 80)
    print(f"TOP15蒸馏模型 - 最近{test_periods}期回测")
    print("=" * 80)

    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)

    if total < test_periods + 30:
        test_periods = total - 30
        print(f"数据不足，调整为回测{test_periods}期")

    predictor = Top15Predictor()

    hits = []
    details = []

    start_idx = total - test_periods

    for i in range(start_idx, total):
        train_data = numbers[:i]
        actual = int(numbers[i])

        top15 = predictor.predict(train_data)

        is_hit = actual in top15
        rank = top15.index(actual) + 1 if is_hit else -1
        hits.append(is_hit)

        details.append({
            'seq': i - start_idx + 1,
            'period_idx': i + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'hit': is_hit,
            'rank': rank,
            'top15': top15
        })

        if (i - start_idx + 1) % 50 == 0:
            done = i - start_idx + 1
            hit_count = sum(hits)
            print(f"  进度: {done}/{test_periods}  当前命中率: {hit_count}/{done} = {hit_count/done*100:.1f}%")

    # ========== 总体统计 ==========
    total_hits = sum(hits)
    hit_rate = total_hits / test_periods * 100
    print(f"\n{'=' * 80}")
    print(f"总体统计")
    print(f"{'=' * 80}")
    print(f"回测期数: {test_periods}")
    print(f"命中次数: {total_hits}")
    print(f"未中次数: {test_periods - total_hits}")
    print(f"命中率:   {hit_rate:.2f}%")
    print(f"理论随机: {15/49*100:.2f}%")

    # ========== 分段统计 ==========
    print(f"\n{'=' * 80}")
    print(f"分段命中率 (每50期)")
    print(f"{'=' * 80}")
    for seg_start in range(0, test_periods, 50):
        seg_end = min(seg_start + 50, test_periods)
        seg_hits = sum(hits[seg_start:seg_end])
        seg_total = seg_end - seg_start
        seg_rate = seg_hits / seg_total * 100
        start_date = details[seg_start]['date']
        end_date = details[seg_end - 1]['date']
        print(f"  第{seg_start+1:>3}-{seg_end:>3}期 ({start_date}~{end_date}): "
              f"{seg_hits:>2}/{seg_total} = {seg_rate:>5.1f}%")

    # ========== 连续不中分析 ==========
    print(f"\n{'=' * 80}")
    print(f"连续不中分析")
    print(f"{'=' * 80}")

    streaks = []  # (start_seq, end_seq, length, start_date, end_date)
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

    # 处理末尾未中
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

    if streaks:
        # 按长度排序显示
        streaks_sorted = sorted(streaks, key=lambda x: x['length'], reverse=True)
        max_streak = streaks_sorted[0]['length']
        avg_streak = sum(s['length'] for s in streaks) / len(streaks)

        print(f"连续不中次数:     {len(streaks)}次")
        print(f"最长连续不中:     {max_streak}期")
        print(f"平均连续不中:     {avg_streak:.2f}期")

        # 连续不中长度分布
        streak_lengths = [s['length'] for s in streaks]
        length_dist = Counter(streak_lengths)
        print(f"\n连续不中长度分布:")
        for length in sorted(length_dist.keys()):
            count = length_dist[length]
            bar = '█' * count
            print(f"  {length:>2}期不中: {count:>3}次 {bar}")

        # 列出所有连续不中段（按时间顺序）
        print(f"\n{'=' * 80}")
        print(f"所有连续不中段明细 (按时间顺序)")
        print(f"{'=' * 80}")
        print(f"{'序号':>4}  {'起始序号':>6}  {'结束序号':>6}  {'长度':>4}  {'起始日期':<12}  {'结束日期':<12}  详情")
        print("-" * 100)

        for idx, s in enumerate(streaks, 1):
            # 获取这段连续不中的实际号码
            miss_nums = [details[j]['actual'] for j in range(s['start_idx'], s['end_idx'] + 1)]
            miss_str = ','.join(str(n) for n in miss_nums)
            print(f"  {idx:>3}  {s['start_seq']:>6}  {s['end_seq']:>6}  {s['length']:>4}  "
                  f"{s['start_date']:<12}  {s['end_date']:<12}  实际号码: {miss_str}")

        # 重点关注：超过3期的连续不中
        long_streaks = [s for s in streaks if s['length'] >= 3]
        print(f"\n{'=' * 80}")
        print(f"重点关注：连续不中 ≥ 3期 ({len(long_streaks)}次)")
        print(f"{'=' * 80}")
        for idx, s in enumerate(long_streaks, 1):
            miss_details = []
            for j in range(s['start_idx'], s['end_idx'] + 1):
                d = details[j]
                miss_details.append(f"  第{d['seq']}期({d['date']}): 实际={d['actual']}, TOP15={d['top15']}")
            print(f"\n--- 第{idx}段: 连续{s['length']}期不中 (序号{s['start_seq']}-{s['end_seq']}) ---")
            for line in miss_details:
                print(line)
    else:
        print("无连续不中记录（全部命中）")

    # ========== 命中排名分布 ==========
    print(f"\n{'=' * 80}")
    print(f"命中时的排名分布")
    print(f"{'=' * 80}")
    hit_ranks = [d['rank'] for d in details if d['hit']]
    if hit_ranks:
        rank_dist = Counter(hit_ranks)
        for rank in sorted(rank_dist.keys()):
            count = rank_dist[rank]
            bar = '█' * count
            print(f"  第{rank:>2}名命中: {count:>3}次 {bar}")

    print(f"\n回测完成。")


if __name__ == '__main__':
    backtest_top15(400)
