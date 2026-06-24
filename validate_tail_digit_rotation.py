"""
尾数预测模型 - 三期内中一期策略分析
智能轮换模型: 固定4组尾数, 通过轮换和评分切换提高3期窗口命中率
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tail_digit_predictor import TailDigitPredictor, TailDigitRotationPredictor, TAIL_DIGIT_NUMBERS, number_to_tail


def run_analysis():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()

    test_periods = min(300, len(df) - 50)
    start_idx = len(df) - test_periods

    print('=' * 85)
    print('🔢 尾数预测 - 智能轮换模型 (固定4组尾数, 三期内中一期)')
    print('=' * 85)
    print()
    print('模型说明:')
    print('  • 固定每次预测4组尾数, 覆盖约19-20个号码, 不扩大范围')
    print('  • 正常模式: 6维统计信号(频率+冷号+趋势+周期+关联+间隔)加权打分取TOP4')
    print('  • 轮换模式: miss后排除上轮预测尾数, 从剩余中选得分最高的4组')
    print('  • 救援模式: 连miss≥3时切换到冷号回补+间隔+周期评分体系')
    print('  • 核心逻辑: 通过轮换覆盖不同尾数区间, 最大化3期窗口命中')
    print()

    predictor = TailDigitRotationPredictor()
    hits = []
    all_results = []

    for i in range(start_idx, len(df)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)

        predicted = predictor.predict(hist, top_n=4)
        hit = actual_tail in predicted
        hits.append(hit)
        predictor.record_result(predicted, hit)

        # 确定模式
        ms = 0
        for j in range(len(hits) - 2, -1, -1):
            if not hits[j]:
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

        all_results.append({
            'period': i - start_idx + 1,
            'date': df.iloc[i]['date'],
            'actual': actual,
            'tail': actual_tail,
            'predicted': predicted,
            'hit': hit,
            'mode': mode,
        })

    # Print all 300 periods
    print(f"{'期号':>4} {'日期':>12} {'号码':>4} {'尾数':>4} {'预测尾数':>16} {'模式':>6} {'结果':>4} {'3期窗口':>8}")
    print('-' * 72)

    for i, r in enumerate(all_results):
        mark = '✅' if r['hit'] else '❌'
        pred_str = ','.join([str(d) for d in r['predicted']])

        if i >= 2:
            w3 = any(hits[i - 2:i + 1])
            w3_str = '✅' if w3 else '💀'
        else:
            w3_str = '--'

        print(f"{r['period']:>4} {r['date']:>12} {r['actual']:>4} {r['tail']:>4} {pred_str:>16} {r['mode']:>6} {mark:>4} {w3_str:>8}")

    # Summary
    print()
    print('=' * 85)
    print('📊 汇总统计')
    print('=' * 85)
    total_hits = sum(hits)
    hit_rate = total_hits / len(hits) * 100
    print(f'单期命中: {total_hits}/{len(hits)} = {hit_rate:.1f}%')
    print(f'随机基线: 40.0% (4/10), 提升: +{hit_rate - 40:.1f}%')

    windows_3 = [any(hits[i:i + 3]) for i in range(len(hits) - 2)]
    win3_rate = sum(windows_3) / len(windows_3) * 100
    fail3 = len(windows_3) - sum(windows_3)
    print(f'\n⭐ 三期窗口命中: {sum(windows_3)}/{len(windows_3)} = {win3_rate:.1f}%')
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
    print(f'\n模式统计:')
    mode_hits = defaultdict(list)
    for r in all_results:
        mode_hits[r['mode']].append(r['hit'])

    for mode in ['正常', '轮换1', '轮换2', '救援']:
        if mode in mode_hits:
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
        print(f'  {s * seg_size + 1:>3}-{(s + 1) * seg_size:>3}: 命中{seg_rate:.0f}% | 3期窗口{seg_w3_rate:.0f}% {bar}')

    # 3-period failure details
    if fail3 > 0:
        print(f'\n三期全miss详情:')
        shown = set()
        for i in range(len(hits) - 2):
            if not any(hits[i:i + 3]):
                key = (i, i+2)
                if key not in shown:
                    r0 = all_results[i]
                    r2 = all_results[i + 2]
                    print(f'  第{r0["period"]}-{r2["period"]}期: '
                          f'尾数={all_results[i]["tail"]},{all_results[i+1]["tail"]},{all_results[i+2]["tail"]}')
                    shown.add(key)

    # Next prediction
    print(f'\n{"=" * 85}')
    print('🔮 下一期预测')
    print('=' * 85)
    
    predicted, scores, mode = predictor.predict_with_details(numbers, top_n=4)
    print(f'模式: {mode}')
    print()
    for d in predicted:
        nums = TAIL_DIGIT_NUMBERS[d]
        print(f'  🎯 尾数{d} → {nums} (得分:{scores[d]:.4f})')
    all_nums = sorted([n for d in predicted for n in TAIL_DIGIT_NUMBERS[d]])
    print(f'\n  覆盖号码({len(all_nums)}个): {all_nums}')


if __name__ == '__main__':
    run_analysis()
