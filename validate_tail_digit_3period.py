"""
尾数预测模型 - 三期内中一期策略分析
自适应策略: 正常4组, miss 1期扩6组, miss 2期扩8组
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tail_digit_predictor import TailDigitPredictor, TAIL_DIGIT_NUMBERS, number_to_tail


def run_analysis():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()

    test_periods = min(300, len(df) - 50)
    start_idx = len(df) - test_periods

    print('=' * 85)
    print('🔢 尾数预测 - 三期内中一期 自适应策略分析')
    print('=' * 85)
    print()
    print('策略说明:')
    print('  • 正常模式(N=4): 预测4组尾数, 覆盖约20个号码')
    print('  • 追击模式(N=6): 连miss 1期后扩展到6组, 覆盖约29个号码')
    print('  • 强追模式(N=8): 连miss 2期后扩展到8组, 覆盖约39个号码')
    print('  • 目标: 确保三期内至少命中一期')
    print()

    predictor = TailDigitPredictor()
    hits = []
    all_results = []

    for i in range(start_idx, len(df)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)

        recent_misses = 0
        for j in range(len(hits) - 1, -1, -1):
            if not hits[j]:
                recent_misses += 1
            else:
                break

        if recent_misses >= 2:
            top_n = 8
        elif recent_misses >= 1:
            top_n = 6
        else:
            top_n = 4

        predicted = predictor.predict(hist, top_n=top_n)
        hit = actual_tail in predicted
        hits.append(hit)

        coverage = sum(len(TAIL_DIGIT_NUMBERS[d]) for d in predicted)
        mode = f'N={top_n}'

        all_results.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'tail': actual_tail,
            'predicted': predicted,
            'hit': hit,
            'mode': mode,
            'coverage': coverage,
        })

    # Print all 300 periods
    print(f"{'期号':>4} {'日期':>12} {'号码':>4} {'尾数':>4} {'预测尾数':>24} {'模式':>6} {'覆盖':>4} {'结果':>4} {'3期窗口':>8}")
    print('-' * 85)

    for i, r in enumerate(all_results):
        mark = '✅' if r['hit'] else '❌'
        pred_str = ','.join([str(d) for d in r['predicted']])

        # 3-period window status
        if i >= 2:
            w3 = any(hits[i - 2:i + 1])
            w3_str = '✅' if w3 else '💀'
        else:
            w3_str = '--'

        print(f"{r['period']:>4} {r['date']:>12} {r['actual']:>4} {r['tail']:>4} {pred_str:>24} {r['mode']:>6} {r['coverage']:>4} {mark:>4} {w3_str:>8}")

    # Summary
    print()
    print('=' * 85)
    print('📊 汇总统计')
    print('=' * 85)
    total_hits = sum(hits)
    hit_rate = total_hits / len(hits) * 100
    print(f'单期命中: {total_hits}/{len(hits)} = {hit_rate:.1f}%')
    print(f'随机基线: 约52% (自适应4-6-8加权平均)')

    windows_3 = [any(hits[i:i + 3]) for i in range(len(hits) - 2)]
    win3_rate = sum(windows_3) / len(windows_3) * 100
    fail3 = len(windows_3) - sum(windows_3)
    print(f'\n三期窗口命中: {sum(windows_3)}/{len(windows_3)} = {win3_rate:.1f}%')
    print(f'三期全miss次数: {fail3}次')

    max_miss = 0
    cur = 0
    for h in hits:
        if not h:
            cur += 1
            max_miss = max(max_miss, cur)
        else:
            cur = 0
    print(f'最大连续miss: {max_miss}期')

    # Miss streak distribution
    streaks = []
    c = 0
    for h in hits:
        if not h:
            c += 1
        else:
            if c > 0:
                streaks.append(c)
            c = 0
    if c > 0:
        streaks.append(c)

    print(f'\n连续miss分布:')
    streak_counter = Counter(streaks)
    for length in sorted(streak_counter.keys()):
        pct = streak_counter[length] / len(streaks) * 100 if streaks else 0
        print(f'  {length}期miss: {streak_counter[length]}次 ({pct:.0f}%)')

    # Mode distribution
    print(f'\n模式使用统计:')
    mode_hits = defaultdict(list)
    for r in all_results:
        mode_hits[r['mode']].append(r['hit'])

    for mode in sorted(mode_hits.keys()):
        h_list = mode_hits[mode]
        h_cnt = sum(h_list)
        total_m = len(h_list)
        rate_m = h_cnt / total_m * 100
        print(f'  {mode}: {total_m}次使用, 命中{h_cnt}次 ({rate_m:.1f}%)')

    # 分段统计
    print(f'\n分段统计(每50期):')
    seg_size = 50
    for s in range(test_periods // seg_size):
        seg_hits = sum(hits[s * seg_size:(s + 1) * seg_size])
        seg_rate = seg_hits / seg_size * 100
        seg_w3 = [any(hits[i:i + 3]) for i in range(s * seg_size, min((s + 1) * seg_size - 2, len(hits) - 2))]
        seg_w3_rate = sum(seg_w3) / len(seg_w3) * 100 if seg_w3 else 0
        bar = '█' * int(seg_w3_rate / 5) + '░' * (20 - int(seg_w3_rate / 5))
        print(f'  {s * seg_size + 1:>3}-{(s + 1) * seg_size:>3}: 命中{seg_hits}/{seg_size}={seg_rate:.0f}% | 3期窗口{seg_w3_rate:.0f}% {bar}')

    # 3-period failure details
    if fail3 > 0:
        print(f'\n三期全miss详情 (共{fail3}次):')
        for i in range(len(hits) - 2):
            if not any(hits[i:i + 3]):
                r0 = all_results[i]
                r1 = all_results[i + 1]
                r2 = all_results[i + 2]
                print(f'  第{r0["period"]}-{r2["period"]}期: '
                      f'实际尾数={r0["tail"]},{r1["tail"]},{r2["tail"]} | '
                      f'日期={r0["date"]}~{r2["date"]}')

    # Next prediction
    print(f'\n{"=" * 85}')
    print('🔮 下一期预测')
    print('=' * 85)
    
    # Determine current miss streak
    current_misses = 0
    # We don't know if current period hit, so base on the last known result
    # In practice, user would update with the latest result
    
    predicted_4, scores_4, _ = predictor.predict_with_details(numbers, top_n=4)
    predicted_6, scores_6, _ = predictor.predict_with_details(numbers, top_n=6)
    predicted_8, scores_8, _ = predictor.predict_with_details(numbers, top_n=8)
    
    print(f'\n正常模式 (N=4) - 命中后使用:')
    for d in predicted_4:
        nums = TAIL_DIGIT_NUMBERS[d]
        print(f'  🎯 尾数{d} → {nums} (得分:{scores_4[d]:.4f})')
    all_nums_4 = sorted([n for d in predicted_4 for n in TAIL_DIGIT_NUMBERS[d]])
    print(f'  覆盖号码({len(all_nums_4)}个): {all_nums_4}')
    
    print(f'\n追击模式 (N=6) - miss 1期后使用:')
    for d in predicted_6:
        nums = TAIL_DIGIT_NUMBERS[d]
        print(f'  🎯 尾数{d} → {nums} (得分:{scores_6[d]:.4f})')
    all_nums_6 = sorted([n for d in predicted_6 for n in TAIL_DIGIT_NUMBERS[d]])
    print(f'  覆盖号码({len(all_nums_6)}个): {all_nums_6}')
    
    print(f'\n强追模式 (N=8) - 连miss 2期后使用:')
    for d in predicted_8:
        nums = TAIL_DIGIT_NUMBERS[d]
        print(f'  🎯 尾数{d} → {nums} (得分:{scores_8[d]:.4f})')
    all_nums_8 = sorted([n for d in predicted_8 for n in TAIL_DIGIT_NUMBERS[d]])
    print(f'  覆盖号码({len(all_nums_8)}个): {all_nums_8}')


if __name__ == '__main__':
    run_analysis()
