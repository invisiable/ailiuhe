"""
尾数预测 - 固定窗口策略对比测试
新思路: 预测一组尾号，固定使用3期，中或不中都重新预测
对比当前: 每期都重新预测
"""
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tail_digit_predictor import TailDigitPredictor, TailDigitRotationPredictor, TAIL_DIGIT_NUMBERS, number_to_tail


def strategy_fixed_window(numbers, window_size=3):
    """
    固定窗口策略: 预测一组尾号, 连续使用window_size期
    无论中不中, 到期后重新预测
    """
    predictor = TailDigitPredictor()
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    hits = []
    windows = []  # 每个窗口的结果
    current_pred = None
    window_count = 0
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        
        # 是否需要重新预测
        if current_pred is None or window_count >= window_size:
            current_pred = predictor.predict(hist, top_n=4)
            window_count = 0
        
        hit = actual_tail in current_pred
        hits.append(hit)
        window_count += 1
    
    return hits


def strategy_fixed_window_reset_on_hit(numbers, window_size=3):
    """
    固定窗口+中即重置: 预测一组尾号, 最多使用window_size期
    中了重新预测, 不中也最多用3期就重新预测
    """
    predictor = TailDigitPredictor()
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    hits = []
    current_pred = None
    window_count = 0
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        
        # 重新预测条件: 无预测 / 窗口到期 / 上期命中
        if current_pred is None or window_count >= window_size or (hits and hits[-1]):
            current_pred = predictor.predict(hist, top_n=4)
            window_count = 0
        
        hit = actual_tail in current_pred
        hits.append(hit)
        window_count += 1
    
    return hits


def strategy_repredict_every_period(numbers):
    """当前策略: 每期重新预测"""
    predictor = TailDigitPredictor()
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    hits = []
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        predicted = predictor.predict(hist, top_n=4)
        hit = actual_tail in predicted
        hits.append(hit)
    
    return hits


def strategy_rotation_every_period(numbers):
    """当前轮换策略: 每期重新预测+轮换"""
    predictor = TailDigitRotationPredictor()
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    hits = []
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        predicted = predictor.predict(hist, top_n=4)
        hit = actual_tail in predicted
        hits.append(hit)
        predictor.record_result(predicted, hit)
    
    return hits


def strategy_fixed_window_rotation(numbers, window_size=3):
    """
    新方案: 固定窗口+智能预测
    每个窗口开始时重新预测一组, 窗口内固定使用
    利用窗口间的历史信息调整预测策略
    """
    predictor = TailDigitPredictor()
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    hits = []
    current_pred = None
    window_count = 0
    window_results = []  # 每个窗口的历史: [(predicted, hit_in_window)]
    last_window_preds = []
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        
        # 需要重新预测
        if current_pred is None or window_count >= window_size:
            # 根据上一窗口结果决定策略
            all_scores = predictor._calculate_scores(hist)
            sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            if last_window_preds and not any(h for _, h in window_results[-window_size:] if window_results):
                # 上一窗口全miss, 排除上一窗口的预测
                excluded = set(last_window_preds)
                remaining = [(d, s) for d, s in sorted_all if d not in excluded]
                if len(remaining) >= 4:
                    current_pred = [d for d, s in remaining[:4]]
                else:
                    current_pred = [d for d, s in sorted_all[:4]]
            else:
                current_pred = [d for d, s in sorted_all[:4]]
            
            last_window_preds = current_pred[:]
            window_count = 0
        
        hit = actual_tail in current_pred
        hits.append(hit)
        window_results.append((current_pred, hit))
        window_count += 1
    
    return hits


def strategy_fixed_window_cold_rotation(numbers, window_size=3):
    """
    固定窗口+冷号轮换: 窗口全miss后切换到冷号体系
    """
    test_periods = min(300, len(numbers) - 50)
    start_idx = len(numbers) - test_periods
    
    predictor = TailDigitPredictor()
    hits = []
    current_pred = None
    window_count = 0
    prev_window_hit = True  # 上一窗口是否命中
    prev_preds = []
    
    for i in range(start_idx, len(numbers)):
        hist = numbers[:i]
        actual = numbers[i]
        actual_tail = number_to_tail(actual)
        hist_tails = [number_to_tail(n) for n in hist]
        
        if current_pred is None or window_count >= window_size:
            all_scores = predictor._calculate_scores(hist)
            sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            if not prev_window_hit and prev_preds:
                # 上一窗口miss, 用冷号+排除策略
                cold = predictor._cold_rebound_analysis(hist_tails)
                gap = predictor._gap_pattern_analysis(hist_tails)
                cycle = predictor._cycle_analysis(hist_tails)
                rescue = {d: 0.45 * cold[d] + 0.30 * gap[d] + 0.25 * cycle[d] for d in range(10)}
                rescue_sorted = sorted(rescue.items(), key=lambda x: x[1], reverse=True)
                excluded = set(prev_preds)
                remaining = [(d, s) for d, s in rescue_sorted if d not in excluded]
                if len(remaining) >= 4:
                    current_pred = [d for d, s in remaining[:4]]
                else:
                    current_pred = [d for d, s in rescue_sorted[:4]]
            else:
                current_pred = [d for d, s in sorted_all[:4]]
            
            prev_preds = current_pred[:]
            prev_window_hit = False
            window_count = 0
        
        hit = actual_tail in current_pred
        hits.append(hit)
        if hit:
            prev_window_hit = True
        window_count += 1
    
    return hits


def calc_stats(hits, name):
    """计算统计指标"""
    total = len(hits)
    hit_count = sum(hits)
    hit_rate = hit_count / total * 100
    
    # 3-period windows
    windows_3 = [any(hits[i:i + 3]) for i in range(total - 2)]
    win3_rate = sum(windows_3) / len(windows_3) * 100
    fail3 = len(windows_3) - sum(windows_3)
    
    # Max miss
    max_miss = 0
    cur = 0
    for h in hits:
        if not h:
            cur += 1
            max_miss = max(max_miss, cur)
        else:
            cur = 0
    
    return {
        'name': name,
        'hit_rate': hit_rate,
        'win3_rate': win3_rate,
        'fail3': fail3,
        'max_miss': max_miss,
        'total': total,
        'hits': hit_count,
    }


def main():
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values.tolist()
    
    print('=' * 90)
    print('🔢 尾数预测 - 固定窗口策略对比测试')
    print('=' * 90)
    print()
    print('新思路: 预测一组尾号, 固定使用3期, 到期(无论中不中)重新预测')
    print('对比: 每期重新预测 vs 固定3期窗口')
    print()
    
    # 运行各策略
    strategies = []
    
    print('正在测试各策略...')
    
    # 策略1: 每期重新预测(基线)
    hits1 = strategy_repredict_every_period(numbers)
    strategies.append(calc_stats(hits1, '每期重新预测(基线)'))
    
    # 策略2: 每期轮换预测(当前最优)
    hits2 = strategy_rotation_every_period(numbers)
    strategies.append(calc_stats(hits2, '每期轮换预测(当前)'))
    
    # 策略3: 固定3期窗口
    hits3 = strategy_fixed_window(numbers, window_size=3)
    strategies.append(calc_stats(hits3, '固定3期窗口'))
    
    # 策略4: 固定3期+中即重置
    hits4 = strategy_fixed_window_reset_on_hit(numbers, window_size=3)
    strategies.append(calc_stats(hits4, '固定3期+中即重置'))
    
    # 策略5: 固定3期+miss后排除轮换
    hits5 = strategy_fixed_window_rotation(numbers, window_size=3)
    strategies.append(calc_stats(hits5, '固定3期+排除轮换'))
    
    # 策略6: 固定3期+冷号救援
    hits6 = strategy_fixed_window_cold_rotation(numbers, window_size=3)
    strategies.append(calc_stats(hits6, '固定3期+冷号救援'))
    
    # 策略7: 固定2期窗口
    hits7 = strategy_fixed_window(numbers, window_size=2)
    strategies.append(calc_stats(hits7, '固定2期窗口'))
    
    # 策略8: 固定2期+中即重置
    hits8 = strategy_fixed_window_reset_on_hit(numbers, window_size=2)
    strategies.append(calc_stats(hits8, '固定2期+中即重置'))
    
    # 输出对比表
    print()
    print('=' * 90)
    print(f'{"策略":>22} {"单期命中":>10} {"3期窗口":>10} {"3期miss":>8} {"最大miss":>8} {"评价":>8}')
    print('-' * 90)
    
    best_w3 = max(s['win3_rate'] for s in strategies)
    
    for s in strategies:
        mark = ' ⭐' if s['win3_rate'] == best_w3 else ''
        eval_str = '最优' if s['win3_rate'] == best_w3 else ''
        print(f"{s['name']:>22} {s['hit_rate']:>8.1f}% {s['win3_rate']:>8.1f}% {s['fail3']:>8} {s['max_miss']:>8} {eval_str:>6}{mark}")
    
    print()
    print('=' * 90)
    print('📊 详细分析')
    print('=' * 90)
    
    # 对比 "固定窗口" vs "每期预测"
    baseline = strategies[0]
    print(f'\n基线 (每期重新预测): 单期{baseline["hit_rate"]:.1f}%, 3期窗口{baseline["win3_rate"]:.1f}%')
    print()
    
    for s in strategies[2:]:
        diff_hit = s['hit_rate'] - baseline['hit_rate']
        diff_w3 = s['win3_rate'] - baseline['win3_rate']
        arrow_hit = '↑' if diff_hit > 0 else '↓'
        arrow_w3 = '↑' if diff_w3 > 0 else '↓'
        print(f"  {s['name']}: 单期{arrow_hit}{abs(diff_hit):.1f}%, 3期窗口{arrow_w3}{abs(diff_w3):.1f}%")
    
    # 分段对比最优策略
    print()
    print('=' * 90)
    print('📈 最优策略分段表现 (每50期)')
    print('=' * 90)
    
    # 找出3期窗口最高的固定窗口策略
    fixed_strategies = {
        '固定3期窗口': hits3,
        '固定3期+中即重置': hits4,
        '固定3期+排除轮换': hits5,
        '固定3期+冷号救援': hits6,
    }
    
    # 对比每个分段
    seg_size = 50
    n_segs = len(hits1) // seg_size
    
    print(f"\n{'段':>4} {'每期预测':>10} {'轮换预测':>10} {'固定3期':>10} {'中即重置':>10} {'排除轮换':>10} {'冷号救援':>10}")
    print('-' * 80)
    
    all_hits_lists = [hits1, hits2, hits3, hits4, hits5, hits6]
    all_names_short = ['每期', '轮换', '固定3', '中重置', '排除轮', '冷号']
    
    for seg in range(n_segs):
        s_start = seg * seg_size
        s_end = (seg + 1) * seg_size
        row = f"{s_start+1:>3}-{s_end}"
        for h_list in all_hits_lists:
            seg_w3 = [any(h_list[i:i + 3]) for i in range(s_start, min(s_end - 2, len(h_list) - 2))]
            rate = sum(seg_w3) / len(seg_w3) * 100 if seg_w3 else 0
            row += f" {rate:>8.0f}%"
        print(row)
    
    # 结论
    print()
    print('=' * 90)
    print('💡 结论')
    print('=' * 90)
    
    fixed3_stats = strategies[2]
    rotation_stats = strategies[1]
    
    if fixed3_stats['win3_rate'] > rotation_stats['win3_rate']:
        print(f'\n✅ 固定3期窗口策略 优于 每期轮换策略!')
        print(f'   3期窗口命中: {fixed3_stats["win3_rate"]:.1f}% vs {rotation_stats["win3_rate"]:.1f}%')
        print(f'   提升: +{fixed3_stats["win3_rate"] - rotation_stats["win3_rate"]:.1f}%')
    else:
        # 找固定策略中最好的
        best_fixed = max(strategies[2:], key=lambda x: x['win3_rate'])
        if best_fixed['win3_rate'] > rotation_stats['win3_rate']:
            print(f'\n✅ {best_fixed["name"]} 优于 当前轮换策略!')
            print(f'   3期窗口命中: {best_fixed["win3_rate"]:.1f}% vs {rotation_stats["win3_rate"]:.1f}%')
        else:
            diff = rotation_stats['win3_rate'] - best_fixed['win3_rate']
            print(f'\n⚠️ 固定窗口方案中最好的: {best_fixed["name"]}')
            print(f'   3期窗口: {best_fixed["win3_rate"]:.1f}% vs 当前轮换{rotation_stats["win3_rate"]:.1f}%')
            if diff < 2:
                print(f'   差距较小({diff:.1f}%), 固定窗口方案的优势是简单且max_miss更低')
            else:
                print(f'   当前轮换策略仍然更优({diff:.1f}%差距)')
    
    print(f'\n  各策略最大连续miss对比:')
    for s in strategies:
        bar = '█' * s['max_miss'] + '░' * (15 - s['max_miss'])
        print(f"    {s['name']:>22}: {s['max_miss']}期 {bar}")


if __name__ == '__main__':
    main()
