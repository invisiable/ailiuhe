"""
最优智能投注策略 - 最近400期回测
分析命中情况、连续不中期数和投注表现
"""

import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor
from optimal_betting_strategy_v4 import OptimalBettingStrategyV4


def backtest_optimal_smart_400():
    print("=" * 80)
    print("最优智能投注策略(v4.1) - 最近400期回测")
    print("=" * 80)

    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    total = len(df)
    test_periods = 400

    if total < test_periods + 30:
        test_periods = total - 30
        print(f"数据不足，调整为回测{test_periods}期")

    predictor = PreciseTop15Predictor()
    strategy = OptimalBettingStrategyV4(base_bet=15, win_reward=47, max_multiplier=10)

    start_idx = total - test_periods
    hits = []
    details = []

    for i in range(start_idx, total):
        train_data = numbers[:i]
        actual = int(numbers[i])

        top15 = predictor.predict(train_data)
        is_hit = actual in top15

        # 更新predictor性能追踪
        predictor.update_performance(top15, actual)

        # 处理投注策略
        period_result = strategy.process_period(is_hit)

        rank = top15.index(actual) + 1 if is_hit else -1
        hits.append(is_hit)

        details.append({
            'seq': i - start_idx + 1,
            'period_idx': i + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'hit': is_hit,
            'rank': rank,
            'top15': top15,
            'paused': period_result['paused'],
            'multiplier': period_result['multiplier'],
            'bet': period_result['bet'],
            'profit': period_result['profit'],
            'balance': period_result['balance'],
            'consecutive_losses': period_result['consecutive_losses'],
        })

        if (i - start_idx + 1) % 50 == 0:
            done = i - start_idx + 1
            hit_count = sum(hits)
            print(f"  进度: {done}/{test_periods}  命中率: {hit_count}/{done} = {hit_count/done*100:.1f}%  余额: {period_result['balance']:+.0f}元")

    # ========== 总体统计 ==========
    stats = strategy.get_statistics()
    total_hits = sum(hits)
    hit_rate = total_hits / test_periods * 100

    print(f"\n{'=' * 80}")
    print(f"总体统计")
    print(f"{'=' * 80}")
    print(f"回测期数:       {test_periods}")
    print(f"命中次数:       {total_hits}")
    print(f"未中次数:       {test_periods - total_hits}")
    print(f"命中率(预测):   {hit_rate:.2f}%")
    print(f"理论随机:       {15/49*100:.2f}%")
    print(f"\n投注统计:")
    print(f"  实际投注期数: {stats['bet_periods']}")
    print(f"  暂停期数:     {stats['pause_periods']}")
    print(f"  暂停中命中:   {stats['paused_hit_count']}次")
    print(f"  投注命中率:   {stats['hit_rate']*100:.2f}%")
    print(f"  总投注额:     {stats['total_bet']:.0f}元")
    print(f"  总奖金:       {stats['total_win']:.0f}元")
    print(f"  净利润:       {stats['balance']:+.0f}元")
    print(f"  最大回撤:     {stats['max_drawdown']:.0f}元")
    print(f"  ROI:          {stats['roi']:+.2f}%")

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
        seg_balance = details[seg_end - 1]['balance']
        print(f"  第{seg_start+1:>3}-{seg_end:>3}期 ({start_date}~{end_date}): "
              f"{seg_hits:>2}/{seg_total} = {seg_rate:>5.1f}%  余额:{seg_balance:>+8.0f}元")

    # ========== 连续不中分析 ==========
    print(f"\n{'=' * 80}")
    print(f"连续不中分析（包含暂停期）")
    print(f"{'=' * 80}")

    # 分析连续不中（不管是否暂停，只看预测是否命中）
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

    if streaks:
        streaks_sorted = sorted(streaks, key=lambda x: x['length'], reverse=True)
        max_streak = streaks_sorted[0]['length']
        avg_streak = sum(s['length'] for s in streaks) / len(streaks)

        print(f"连续不中次数:     {len(streaks)}次")
        print(f"最长连续不中:     {max_streak}期")
        print(f"平均连续不中:     {avg_streak:.2f}期")

        # 长度分布
        streak_lengths = [s['length'] for s in streaks]
        length_dist = Counter(streak_lengths)
        print(f"\n连续不中长度分布:")
        for length in sorted(length_dist.keys()):
            count = length_dist[length]
            bar = '█' * count
            print(f"  {length:>2}期不中: {count:>3}次 {bar}")

        # 列出所有连续不中段
        print(f"\n{'=' * 80}")
        print(f"所有连续不中段明细 (按时间顺序)")
        print(f"{'=' * 80}")
        print(f"{'序号':>4}  {'起始':>4}  {'结束':>4}  {'长度':>4}  {'起始日期':<12}  {'结束日期':<12}  {'亏损额':>8}  实际号码")
        print("-" * 110)

        for idx, s in enumerate(streaks, 1):
            miss_nums = [details[j]['actual'] for j in range(s['start_idx'], s['end_idx'] + 1)]
            miss_str = ','.join(str(n) for n in miss_nums)
            # 计算这段不中的亏损
            loss = sum(details[j]['profit'] for j in range(s['start_idx'], s['end_idx'] + 1))
            print(f"  {idx:>3}  {s['start_seq']:>4}  {s['end_seq']:>4}  {s['length']:>4}  "
                  f"{s['start_date']:<12}  {s['end_date']:<12}  {loss:>+8.1f}  {miss_str}")

        # 重点：>=5期的连续不中
        long_streaks = [s for s in streaks if s['length'] >= 5]
        print(f"\n{'=' * 80}")
        print(f"重点关注：连续不中 ≥ 5期 ({len(long_streaks)}次)")
        print(f"{'=' * 80}")
        for idx, s in enumerate(long_streaks, 1):
            miss_details = []
            total_loss = 0
            for j in range(s['start_idx'], s['end_idx'] + 1):
                d = details[j]
                status = "暂停" if d['paused'] else f"{d['multiplier']}倍"
                loss_str = f"亏{-d['profit']:.0f}" if d['profit'] < 0 else "暂停"
                total_loss += d['profit']
                miss_details.append(
                    f"  第{d['seq']}期({d['date']}): 实际={d['actual']:>2}, "
                    f"投注={status:<6}, {loss_str}")
            print(f"\n--- 第{idx}段: 连续{s['length']}期不中 ({s['start_date']}~{s['end_date']}) 累计亏损:{total_loss:+.0f}元 ---")
            for line in miss_details:
                print(line)

    # ========== 仅计算实际投注期的连续不中 ==========
    print(f"\n{'=' * 80}")
    print(f"实际投注的连续不中分析（排除暂停期）")
    print(f"{'=' * 80}")

    bet_streaks = []
    bet_streak_start = None
    bet_details_filtered = [(i, d) for i, d in enumerate(details) if not d['paused']]

    for idx_in_list, (orig_idx, d) in enumerate(bet_details_filtered):
        if not d['hit']:
            if bet_streak_start is None:
                bet_streak_start = idx_in_list
        else:
            if bet_streak_start is not None:
                length = idx_in_list - bet_streak_start
                start_d = bet_details_filtered[bet_streak_start][1]
                end_d = bet_details_filtered[idx_in_list - 1][1]
                loss = sum(bet_details_filtered[j][1]['profit'] for j in range(bet_streak_start, idx_in_list))
                bet_streaks.append({
                    'length': length,
                    'start_date': start_d['date'],
                    'end_date': end_d['date'],
                    'loss': loss,
                    'start_idx': bet_streak_start,
                    'end_idx': idx_in_list - 1
                })
                bet_streak_start = None

    if bet_streak_start is not None:
        length = len(bet_details_filtered) - bet_streak_start
        start_d = bet_details_filtered[bet_streak_start][1]
        end_d = bet_details_filtered[-1][1]
        loss = sum(bet_details_filtered[j][1]['profit'] for j in range(bet_streak_start, len(bet_details_filtered)))
        bet_streaks.append({
            'length': length,
            'start_date': start_d['date'],
            'end_date': end_d['date'],
            'loss': loss,
            'start_idx': bet_streak_start,
            'end_idx': len(bet_details_filtered) - 1
        })

    if bet_streaks:
        bet_streaks_sorted = sorted(bet_streaks, key=lambda x: x['length'], reverse=True)
        max_bet_streak = bet_streaks_sorted[0]['length']
        avg_bet_streak = sum(s['length'] for s in bet_streaks) / len(bet_streaks)

        print(f"实际投注连续不中次数: {len(bet_streaks)}次")
        print(f"最长投注连续不中:     {max_bet_streak}期")
        print(f"平均投注连续不中:     {avg_bet_streak:.2f}期")

        # 长度分布
        bet_streak_lengths = [s['length'] for s in bet_streaks]
        bet_length_dist = Counter(bet_streak_lengths)
        print(f"\n投注连续不中长度分布:")
        for length in sorted(bet_length_dist.keys()):
            count = bet_length_dist[length]
            bar = '█' * count
            print(f"  {length:>2}期不中: {count:>3}次 {bar}")

        # 重点：>=5期的
        long_bet_streaks = [s for s in bet_streaks if s['length'] >= 5]
        if long_bet_streaks:
            print(f"\n实际投注连续不中 ≥5期 ({len(long_bet_streaks)}次):")
            for idx, s in enumerate(sorted(long_bet_streaks, key=lambda x: x['length'], reverse=True), 1):
                print(f"  {idx}. 连续{s['length']}期不中 ({s['start_date']}~{s['end_date']}) 亏损:{s['loss']:+.0f}元")
                # 详细展示
                for j in range(s['start_idx'], s['end_idx'] + 1):
                    _, d = bet_details_filtered[j]
                    print(f"     第{d['seq']}期({d['date']}): 实际={d['actual']:>2}, "
                          f"倍数={d['multiplier']}, 亏损={d['profit']:+.1f}元")

    # ========== 倍数使用统计 ==========
    print(f"\n{'=' * 80}")
    print(f"倍数使用统计")
    print(f"{'=' * 80}")
    bet_periods_data = [d for d in details if not d['paused']]
    mult_dist = Counter(d['multiplier'] for d in bet_periods_data)
    for mult in sorted(mult_dist.keys()):
        count = mult_dist[mult]
        hit_at_mult = sum(1 for d in bet_periods_data if d['multiplier'] == mult and d['hit'])
        rate = hit_at_mult / count * 100 if count > 0 else 0
        print(f"  {mult:>5.1f}倍: {count:>3}次, 命中{hit_at_mult}次({rate:.1f}%)")

    # 达到最大倍数10的情况
    max_mult_periods = [d for d in bet_periods_data if d['multiplier'] >= 10]
    print(f"\n触及最大倍数(≥10): {len(max_mult_periods)}次")

    print(f"\n回测完成。")


if __name__ == '__main__':
    backtest_optimal_smart_400()
