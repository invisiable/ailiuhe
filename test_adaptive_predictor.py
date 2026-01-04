"""
测试自适应预测器在不同周期的表现
对比固定权重 vs 动态权重
"""

import pandas as pd
from zodiac_adaptive_predictor import ZodiacAdaptivePredictor
from zodiac_super_predictor import ZodiacSuperPredictor

def test_adaptive_vs_fixed(n_periods=100):
    """对比测试"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    adaptive_predictor = ZodiacAdaptivePredictor()
    fixed_predictor = ZodiacSuperPredictor()
    
    print('='*100)
    print(f'自适应预测器 vs 固定权重预测器 - {n_periods}期对比测试')
    print('='*100)
    print()
    
    # 统计
    adaptive_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    fixed_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    cycle_stats = {
        'hot_cycle_count': 0,
        'cold_cycle_count': 0,
        'hot_cycle_hits': 0,
        'cold_cycle_hits': 0
    }
    
    for i in range(total - n_periods, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        
        # 自适应预测
        adaptive_result = adaptive_predictor.predict_with_adaptive_weights(animals, top_n=5)
        adaptive_top5 = adaptive_result['top5']
        
        # 固定权重预测
        strategies_scores = {
            'ultra_cold': fixed_predictor._ultra_cold_strategy(animals),
            'anti_hot': fixed_predictor._anti_hot_strategy(animals),
            'gap': fixed_predictor._gap_analysis(animals),
            'rotation': fixed_predictor._rotation_advanced(animals),
            'absence_penalty': fixed_predictor._continuous_absence_penalty(animals),
            'diversity': fixed_predictor._diversity_boost(animals),
            'similarity': fixed_predictor._historical_similarity(animals)
        }
        
        final_scores = {}
        for zodiac in fixed_predictor.zodiacs:
            score = 0.0
            score += strategies_scores['ultra_cold'].get(zodiac, 0) * 0.35
            score += strategies_scores['anti_hot'].get(zodiac, 0) * 0.20
            score += strategies_scores['gap'].get(zodiac, 0) * 0.18
            score += strategies_scores['rotation'].get(zodiac, 0) * 0.12
            score += strategies_scores['absence_penalty'].get(zodiac, 0) * 0.08
            score += strategies_scores['diversity'].get(zodiac, 0) * 0.04
            score += strategies_scores['similarity'].get(zodiac, 0) * 0.03
            final_scores[zodiac] = score
        
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        fixed_top5 = [z for z, s in sorted_zodiacs[:5]]
        
        # 统计自适应命中
        if actual in adaptive_top5:
            rank = adaptive_top5.index(actual) + 1
            if rank == 1:
                adaptive_hits['TOP1'] += 1
                adaptive_hits['TOP2'] += 1
                adaptive_hits['TOP3'] += 1
                adaptive_hits['TOP5'] += 1
            elif rank == 2:
                adaptive_hits['TOP2'] += 1
                adaptive_hits['TOP3'] += 1
                adaptive_hits['TOP5'] += 1
            elif rank == 3:
                adaptive_hits['TOP3'] += 1
                adaptive_hits['TOP5'] += 1
            else:
                adaptive_hits['TOP5'] += 1
        
        # 统计固定权重命中
        if actual in fixed_top5:
            rank = fixed_top5.index(actual) + 1
            if rank == 1:
                fixed_hits['TOP1'] += 1
                fixed_hits['TOP2'] += 1
                fixed_hits['TOP3'] += 1
                fixed_hits['TOP5'] += 1
            elif rank == 2:
                fixed_hits['TOP2'] += 1
                fixed_hits['TOP3'] += 1
                fixed_hits['TOP5'] += 1
            elif rank == 3:
                fixed_hits['TOP3'] += 1
                fixed_hits['TOP5'] += 1
            else:
                fixed_hits['TOP5'] += 1
        
        # 周期统计
        if adaptive_result['is_hot_cycle']:
            cycle_stats['hot_cycle_count'] += 1
            if actual in adaptive_top5:
                cycle_stats['hot_cycle_hits'] += 1
        else:
            cycle_stats['cold_cycle_count'] += 1
            if actual in adaptive_top5:
                cycle_stats['cold_cycle_hits'] += 1
    
    # 显示结果
    print(f"{'='*100}")
    print(f"整体命中率对比（{n_periods}期）")
    print(f"{'='*100}")
    print(f"{'指标':<15} {'自适应权重':<20} {'固定权重':<20} {'提升'}")
    print('-'*100)
    
    for metric in ['TOP1', 'TOP2', 'TOP3', 'TOP5']:
        adaptive_rate = adaptive_hits[metric] / n_periods * 100
        fixed_rate = fixed_hits[metric] / n_periods * 100
        improvement = adaptive_rate - fixed_rate
        
        print(f"{metric:<15} {adaptive_hits[metric]:3d}/'{n_periods} ({adaptive_rate:5.1f}%)     "
              f"{fixed_hits[metric]:3d}/{n_periods} ({fixed_rate:5.1f}%)     "
              f"{improvement:+5.1f}%")
    
    print()
    print(f"{'='*100}")
    print(f"周期分析")
    print(f"{'='*100}")
    print(f"  热门周期: {cycle_stats['hot_cycle_count']}/{n_periods} ({cycle_stats['hot_cycle_count']/n_periods*100:.1f}%)")
    print(f"  冷门周期: {cycle_stats['cold_cycle_count']}/{n_periods} ({cycle_stats['cold_cycle_count']/n_periods*100:.1f}%)")
    
    if cycle_stats['hot_cycle_count'] > 0:
        hot_rate = cycle_stats['hot_cycle_hits'] / cycle_stats['hot_cycle_count'] * 100
        print(f"\n  热门周期命中率: {cycle_stats['hot_cycle_hits']}/{cycle_stats['hot_cycle_count']} ({hot_rate:.1f}%)")
    
    if cycle_stats['cold_cycle_count'] > 0:
        cold_rate = cycle_stats['cold_cycle_hits'] / cycle_stats['cold_cycle_count'] * 100
        print(f"  冷门周期命中率: {cycle_stats['cold_cycle_hits']}/{cycle_stats['cold_cycle_count']} ({cold_rate:.1f}%)")
    
    print(f"{'='*100}")
    
    return {
        'adaptive': adaptive_hits,
        'fixed': fixed_hits,
        'cycles': cycle_stats
    }

def test_recent_10_detailed():
    """详细测试最近10期"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    predictor = ZodiacAdaptivePredictor()
    
    print('\n' + '='*100)
    print('最近10期详细分析（自适应策略）')
    print('='*100)
    print()
    
    hits = 0
    
    for i in range(total - 10, total):
        period_num = i + 1
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        
        result = predictor.predict_with_adaptive_weights(animals, top_n=5, debug=False)
        
        hit = actual in result['top5']
        if hit:
            hits += 1
            rank = result['top5'].index(actual) + 1
        
        print(f"第{period_num}期: 实际={actual:4s}, 预测TOP5={', '.join(result['top5'])}, "
              f"周期={result['cycle_type']:6s}, "
              f"{'[OK]命中第' + str(rank) + '位' if hit else '[X]未中'}")
    
    print()
    print(f"最近10期命中率: {hits}/10 = {hits/10*100:.1f}%")
    print('='*100)

if __name__ == '__main__':
    # 测试最近10期
    test_recent_10_detailed()
    
    print('\n\n')
    
    # 测试最近10期对比
    print('='*100)
    print('最近10期对比')
    print('='*100)
    test_adaptive_vs_fixed(n_periods=10)
    
    print('\n\n')
    
    # 测试100期对比
    print('='*100)
    print('100期对比')
    print('='*100)
    test_adaptive_vs_fixed(n_periods=100)
