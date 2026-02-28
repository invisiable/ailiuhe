"""
对比测试：原始生肖TOP4策略 vs 精准生肖TOP4策略
重点关注：最大连续不中期数
"""

import pandas as pd
import numpy as np
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy
from precise_zodiac_top4_predictor import PreciseZodiacTop4Predictor


def test_strategy(predictor_name, predictor, df, test_periods=300):
    """
    测试策略性能
    
    Args:
        predictor_name: 预测器名称
        predictor: 预测器实例
        df: 数据DataFrame
        test_periods: 测试期数
        
    Returns:
        dict: 测试结果
    """
    start_idx = len(df) - test_periods
    
    results = []
    max_consecutive_misses = 0
    current_consecutive_misses = 0
    consecutive_miss_sequences = []
    
    hits = 0
    total_profit = 0
    total_bet = 0
    
    for i in range(start_idx, len(df)):
        period = i - start_idx + 1
        
        # 获取历史数据
        history_animals = df['animal'].iloc[:i].tolist()
        
        # 预测
        if hasattr(predictor, 'predict_top4') and predictor_name == "精准版":
            # 精准预测器
            top4 = predictor.predict_top4(history_animals)
        else:
            # 原始策略
            prediction = predictor.predict_top4(history_animals)
            top4 = prediction['top4']
        
        # 实际结果
        actual = df.iloc[i]['animal']
        actual_date = df.iloc[i]['date']
        actual_number = df.iloc[i]['number']
        
        # 判断命中
        is_hit = actual in top4
        
        # 更新统计
        if is_hit:
            hits += 1
            profit = 46 - 16  # 命中赚30
            current_consecutive_misses = 0
        else:
            profit = -16  # 未中亏16
            current_consecutive_misses += 1
            
            # 记录连续不中序列
            if current_consecutive_misses == 1:
                consecutive_miss_sequences.append({
                    'start': period,
                    'length': 1
                })
            else:
                consecutive_miss_sequences[-1]['length'] = current_consecutive_misses
        
        max_consecutive_misses = max(max_consecutive_misses, current_consecutive_misses)
        total_profit += profit
        total_bet += 16
        
        # 更新预测器性能（如果支持）
        if hasattr(predictor, 'update_performance'):
            if predictor_name == "精准版":
                predictor.update_performance(top4, actual)
            else:
                predictor.update_performance(is_hit)
        
        results.append({
            'period': period,
            'date': actual_date,
            'top4': top4,
            'actual': actual,
            'is_hit': is_hit,
            'profit': profit,
            'cumulative_profit': total_profit,
            'consecutive_misses': current_consecutive_misses
        })
    
    # 统计连续不中
    long_misses_5 = len([s for s in consecutive_miss_sequences if s['length'] >= 5])
    long_misses_7 = len([s for s in consecutive_miss_sequences if s['length'] >= 7])
    long_misses_10 = len([s for s in consecutive_miss_sequences if s['length'] >= 10])
    
    # 计算平均连续不中
    if consecutive_miss_sequences:
        avg_miss_length = np.mean([s['length'] for s in consecutive_miss_sequences])
    else:
        avg_miss_length = 0
    
    return {
        'predictor_name': predictor_name,
        'total_periods': test_periods,
        'hits': hits,
        'hit_rate': hits / test_periods,
        'max_consecutive_misses': max_consecutive_misses,
        'avg_consecutive_misses': avg_miss_length,
        'long_misses_5+': long_misses_5,
        'long_misses_7+': long_misses_7,
        'long_misses_10+': long_misses_10,
        'total_profit': total_profit,
        'total_bet': total_bet,
        'roi': (total_profit / total_bet) * 100,
        'results': results,
        'miss_sequences': consecutive_miss_sequences
    }


def main():
    print("="*80)
    print("生肖TOP4策略对比测试：原始版 vs 精准版")
    print("目标：将最大连续不中从12期降低到4期")
    print("="*80 + "\n")
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"数据加载完成：{len(df)}期\n")
    
    # 测试期数
    test_periods = 300
    print(f"测试期数：最近{test_periods}期\n")
    
    # 测试原始策略
    print("["*40)
    print("测试原始策略（RecommendedZodiacTop4Strategy v2.0）")
    print("["*40)
    original = RecommendedZodiacTop4Strategy(use_emergency_backup=False)
    result_original = test_strategy("原始版", original, df, test_periods)
    print("原始策略测试完成!\n")
    
    # 测试精准策略
    print("]"*40)
    print("测试精准策略（PreciseZodiacTop4Predictor）")
    print("]"*40)
    precise = PreciseZodiacTop4Predictor()
    result_precise = test_strategy("精准版", precise, df, test_periods)
    print("精准策略测试完成!\n")
    
    # 对比结果
    print("="*80)
    print("对比结果")
    print("="*80 + "\n")
    
    # 基础统计
    print("【基础统计】")
    print(f"{'指标':<20} {'原始版':<20} {'精准版':<20} {'改进':<20}")
    print("-"*80)
    print(f"{'测试期数':<20} {result_original['total_periods']:<20} {result_precise['total_periods']:<20} -")
    print(f"{'命中次数':<20} {result_original['hits']:<20} {result_precise['hits']:<20} {result_precise['hits'] - result_original['hits']:+d}")
    print(f"{'命中率':<20} {result_original['hit_rate']*100:.2f}%{'':<14} {result_precise['hit_rate']*100:.2f}%{'':<14} {(result_precise['hit_rate'] - result_original['hit_rate'])*100:+.2f}%")
    print()
    
    # 连续不中统计（重点）
    print("【连续不中统计】⭐ 核心指标")
    print(f"{'指标':<20} {'原始版':<20} {'精准版':<20} {'改进':<20}")
    print("-"*80)
    print(f"{'最大连续不中':<20} {result_original['max_consecutive_misses']}期{'':<16} {result_precise['max_consecutive_misses']}期{'':<16} {result_precise['max_consecutive_misses'] - result_original['max_consecutive_misses']:+d}期")
    print(f"{'平均连续不中':<20} {result_original['avg_consecutive_misses']:.2f}期{'':<14} {result_precise['avg_consecutive_misses']:.2f}期{'':<14} {result_precise['avg_consecutive_misses'] - result_original['avg_consecutive_misses']:+.2f}期")
    print(f"{'连续不中>=5期':<20} {result_original['long_misses_5+']}次{'':<16} {result_precise['long_misses_5+']}次{'':<16} {result_precise['long_misses_5+'] - result_original['long_misses_5+']:+d}次")
    print(f"{'连续不中>=7期':<20} {result_original['long_misses_7+']}次{'':<16} {result_precise['long_misses_7+']}次{'':<16} {result_precise['long_misses_7+'] - result_original['long_misses_7+']:+d}次")
    print(f"{'连续不中>=10期':<20} {result_original['long_misses_10+']}次{'':<16} {result_precise['long_misses_10+']}次{'':<16} {result_precise['long_misses_10+'] - result_original['long_misses_10+']:+d}次")
    print()
    
    # 财务统计
    print("【财务统计】")
    print(f"{'指标':<20} {'原始版':<20} {'精准版':<20} {'改进':<20}")
    print("-"*80)
    print(f"{'总投注':<20} {result_original['total_bet']:.2f}元{'':<12} {result_precise['total_bet']:.2f}元{'':<12} -")
    print(f"{'总收益':<20} {result_original['total_profit']:+.2f}元{'':<12} {result_precise['total_profit']:+.2f}元{'':<12} {result_precise['total_profit'] - result_original['total_profit']:+.2f}元")
    print(f"{'ROI':<20} {result_original['roi']:+.2f}%{'':<14} {result_precise['roi']:+.2f}%{'':<14} {result_precise['roi'] - result_original['roi']:+.2f}%")
    print()
    
    # 目标达成情况
    print("="*80)
    print("目标达成情况")
    print("="*80 + "\n")
    
    original_max = result_original['max_consecutive_misses']
    precise_max = result_precise['max_consecutive_misses']
    target = 4
    
    print(f"原始版最大连续不中: {original_max}期")
    print(f"精准版最大连续不中: {precise_max}期")
    print(f"目标: ≤{target}期\n")
    
    if precise_max <= target:
        print(f"✅ 目标达成！精准版最大连续不中为{precise_max}期，达到目标要求")
        improvement = ((original_max - precise_max) / original_max) * 100
        print(f"✅ 相比原始版降低了{improvement:.1f}%（从{original_max}期降至{precise_max}期）")
    else:
        print(f"⚠ 未完全达成目标，但已有显著改善")
        gap = precise_max - target
        improvement = ((original_max - precise_max) / original_max) * 100
        print(f"   • 距离目标还差{gap}期")
        print(f"   • 相比原始版已降低{improvement:.1f}%（从{original_max}期降至{precise_max}期）")
    
    print()
    
    # 详细的连续不中序列对比
    print("="*80)
    print("连续不中详细分析")
    print("="*80 + "\n")
    
    print("【原始版 - 所有>=5期的连续不中】")
    original_long = [s for s in result_original['miss_sequences'] if s['length'] >= 5]
    original_long.sort(key=lambda x: x['length'], reverse=True)
    for seq in original_long:
        end = seq['start'] + seq['length'] - 1
        print(f"  {seq['length']}期不中: 第{seq['start']}期 到 第{end}期")
    print()
    
    print("【精准版 - 所有>=5期的连续不中】")
    precise_long = [s for s in result_precise['miss_sequences'] if s['length'] >= 5]
    precise_long.sort(key=lambda x: x['length'], reverse=True)
    if precise_long:
        for seq in precise_long:
            end = seq['start'] + seq['length'] - 1
            print(f"  {seq['length']}期不中: 第{seq['start']}期 到 第{end}期")
    else:
        print("  无5期以上连续不中情况 ✓")
    print()
    
    # 关键改进总结
    print("="*80)
    print("关键改进总结")
    print("="*80 + "\n")
    
    print("✨ 精准版相比原始版的改进：")
    print(f"  1. 最大连续不中: {original_max}期 → {precise_max}期 ({(precise_max-original_max):+d}期)")
    print(f"  2. 平均连续不中: {result_original['avg_consecutive_misses']:.2f}期 → {result_precise['avg_consecutive_misses']:.2f}期")
    print(f"  3. 连续不中>=5期次数: {result_original['long_misses_5+']}次 → {result_precise['long_misses_5+']}次 ({result_precise['long_misses_5+']-result_original['long_misses_5+']:+d}次)")
    print(f"  4. ROI: {result_original['roi']:.2f}% → {result_precise['roi']:.2f}% ({result_precise['roi']-result_original['roi']:+.2f}%)")
    
    if precise_max <= target:
        print(f"\n🎉 成功达成目标：最大连续不中≤{target}期！")
    
    # 保存详细结果
    print("\n保存详细结果...")
    
    # 保存原始版
    df_original = pd.DataFrame(result_original['results'])
    df_original.to_csv('zodiac_top4_original_300periods.csv', index=False, encoding='utf-8-sig')
    print("  ✓ zodiac_top4_original_300periods.csv")
    
    # 保存精准版
    df_precise = pd.DataFrame(result_precise['results'])
    df_precise.to_csv('zodiac_top4_precise_300periods.csv', index=False, encoding='utf-8-sig')
    print("  ✓ zodiac_top4_precise_300periods.csv")
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == '__main__':
    main()
