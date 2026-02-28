"""
对比自适应混合与精准TOP15的连续失败情况
"""

import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor
from advanced_hybrid_top15_predictor import AdaptiveHybridTop15Predictor

def analyze_consecutive_misses(test_periods=200):
    """分析连续失败情况"""
    print("="*80)
    print("连续失败分析：自适应混合 vs 精准TOP15")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv')
    numbers_all = df['number'].tolist()
    
    total = len(numbers_all)
    start = total - test_periods
    
    predictors = {
        '精准TOP15': PreciseTop15Predictor(),
        '自适应混合': AdaptiveHybridTop15Predictor()
    }
    
    results = {}
    
    for name, predictor in predictors.items():
        print(f"\n测试: {name}")
        print("-" * 40)
        
        hits = []
        consecutive_misses = []
        current_miss_streak = 0
        all_miss_streaks = []  # 记录所有连续失败段
        
        for i in range(start, total):
            history = numbers_all[:i]
            actual = numbers_all[i]
            preds = predictor.predict(history)
            
            hit = actual in preds
            hits.append(1 if hit else 0)
            
            if hit:
                if current_miss_streak > 0:
                    all_miss_streaks.append(current_miss_streak)
                current_miss_streak = 0
            else:
                current_miss_streak += 1
            
            consecutive_misses.append(current_miss_streak)
        
        # 最后一段失败
        if current_miss_streak > 0:
            all_miss_streaks.append(current_miss_streak)
        
        # 统计
        hit_count = sum(hits)
        hit_rate = hit_count / test_periods * 100
        max_consecutive_miss = max(consecutive_misses) if consecutive_misses else 0
        
        # 计算连续失败的分布
        miss_distribution = Counter(all_miss_streaks)
        
        results[name] = {
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'max_consecutive_miss': max_consecutive_miss,
            'avg_miss_streak': np.mean(all_miss_streaks) if all_miss_streaks else 0,
            'miss_streaks': all_miss_streaks,
            'miss_distribution': miss_distribution,
            'consecutive_misses': consecutive_misses
        }
        
        print(f"  命中次数: {hit_count}/{test_periods} ({hit_rate:.2f}%)")
        print(f"  最大连续失败: {max_consecutive_miss}期")
        print(f"  平均连续失败: {np.mean(all_miss_streaks):.2f}期")
        print(f"  连续失败段数: {len(all_miss_streaks)}次")
        
        # 分布统计
        print(f"\n  连续失败分布:")
        for streak_len in sorted(miss_distribution.keys()):
            count = miss_distribution[streak_len]
            print(f"    {streak_len}期连续失败: {count}次")
    
    # 详细对比
    print("\n" + "="*80)
    print("详细对比分析")
    print("="*80)
    
    r1 = results['精准TOP15']
    r2 = results['自适应混合']
    
    print(f"\n{'指标':<20} {'精准TOP15':<15} {'自适应混合':<15} {'差异':<15}")
    print("-" * 60)
    
    # 命中率
    diff_hit = r2['hit_rate'] - r1['hit_rate']
    print(f"{'命中率':<20} {r1['hit_rate']:<14.2f}% {r2['hit_rate']:<14.2f}% {diff_hit:+.2f}%")
    
    # 最大连续失败
    diff_max = r2['max_consecutive_miss'] - r1['max_consecutive_miss']
    print(f"{'最大连续失败':<20} {r1['max_consecutive_miss']:<14}期 {r2['max_consecutive_miss']:<14}期 {diff_max:+d}期")
    
    # 平均连续失败
    diff_avg = r2['avg_miss_streak'] - r1['avg_miss_streak']
    print(f"{'平均连续失败':<20} {r1['avg_miss_streak']:<14.2f}期 {r2['avg_miss_streak']:<14.2f}期 {diff_avg:+.2f}期")
    
    # 连续失败段数
    diff_count = len(r2['miss_streaks']) - len(r1['miss_streaks'])
    print(f"{'连续失败段数':<20} {len(r1['miss_streaks']):<14}次 {len(r2['miss_streaks']):<14}次 {diff_count:+d}次")
    
    # 分析长连败的频率
    print("\n" + "="*80)
    print("长连败(≥5期)分析")
    print("="*80)
    
    for name, data in results.items():
        long_streaks = [s for s in data['miss_streaks'] if s >= 5]
        very_long = [s for s in data['miss_streaks'] if s >= 7]
        extreme = [s for s in data['miss_streaks'] if s >= 9]
        
        print(f"\n{name}:")
        print(f"  ≥5期连续失败: {len(long_streaks)}次")
        print(f"  ≥7期连续失败: {len(very_long)}次")
        print(f"  ≥9期连续失败: {len(extreme)}次")
        
        if long_streaks:
            print(f"  长连败详情: {sorted(long_streaks, reverse=True)}")
    
    # 风险评估
    print("\n" + "="*80)
    print("风险评估")
    print("="*80)
    
    # Fibonacci倍投下的最大投入
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    
    for name, data in results.items():
        max_miss = data['max_consecutive_miss']
        
        # 计算最大单次投入
        if max_miss < len(fib):
            max_bet_multiplier = fib[max_miss]
        else:
            max_bet_multiplier = fib[-1]
        
        max_single_bet = 15 * max_bet_multiplier
        
        # 计算最大累计投入（连续失败期间）
        cumulative_bet = sum(15 * fib[min(i, len(fib)-1)] for i in range(max_miss))
        
        print(f"\n{name} (最大连续{max_miss}期失败):")
        print(f"  最高单期投入: {max_single_bet}元 (倍数{max_bet_multiplier})")
        print(f"  最大累计亏损: {cumulative_bet}元")
    
    # 结论
    print("\n" + "="*80)
    print("💡 结论")
    print("="*80)
    
    if r2['max_consecutive_miss'] < r1['max_consecutive_miss']:
        print(f"\n✅ 自适应混合降低了最大连续失败:")
        print(f"   {r1['max_consecutive_miss']}期 → {r2['max_consecutive_miss']}期 (减少{r1['max_consecutive_miss'] - r2['max_consecutive_miss']}期)")
    elif r2['max_consecutive_miss'] == r1['max_consecutive_miss']:
        print(f"\n⚖️ 最大连续失败相同，都是{r1['max_consecutive_miss']}期")
    else:
        print(f"\n⚠️ 自适应混合最大连续失败更高:")
        print(f"   {r1['max_consecutive_miss']}期 → {r2['max_consecutive_miss']}期")
    
    if r2['avg_miss_streak'] < r1['avg_miss_streak']:
        print(f"\n✅ 自适应混合降低了平均连续失败:")
        print(f"   {r1['avg_miss_streak']:.2f}期 → {r2['avg_miss_streak']:.2f}期")
    
    # 综合建议
    long_streaks_1 = len([s for s in r1['miss_streaks'] if s >= 5])
    long_streaks_2 = len([s for s in r2['miss_streaks'] if s >= 5])
    
    print(f"\n📊 长连败(≥5期)频率对比:")
    print(f"   精准TOP15: {long_streaks_1}次")
    print(f"   自适应混合: {long_streaks_2}次")
    
    if long_streaks_2 < long_streaks_1:
        print(f"   ✅ 自适应混合减少了{long_streaks_1 - long_streaks_2}次长连败")
    elif long_streaks_2 > long_streaks_1:
        print(f"   ⚠️ 自适应混合增加了{long_streaks_2 - long_streaks_1}次长连败")
    else:
        print(f"   ⚖️ 长连败次数相同")
    
    return results

if __name__ == "__main__":
    results = analyze_consecutive_misses(test_periods=200)
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
