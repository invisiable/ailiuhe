"""
测试最终优化版预测器（v7.0）
对比v5.0固定冷门策略
"""

import pandas as pd
from zodiac_final_predictor import ZodiacFinalPredictor
from zodiac_super_predictor import ZodiacSuperPredictor

def test_final_vs_v5(n_periods=100):
    """对比测试"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    final_predictor = ZodiacFinalPredictor()
    v5_predictor = ZodiacSuperPredictor()
    
    print('='*100)
    print(f'v7.0最终版 vs v5.0冷门版 - {n_periods}期对比测试')
    print('='*100)
    print()
    
    # 统计
    final_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    v5_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    for i in range(total - n_periods, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        
        # v7.0预测
        final_top5, _ = final_predictor.predict_from_history(animals, top_n=5)
        
        # v5.0预测
        strategies_scores = {
            'ultra_cold': v5_predictor._ultra_cold_strategy(animals),
            'anti_hot': v5_predictor._anti_hot_strategy(animals),
            'gap': v5_predictor._gap_analysis(animals),
            'rotation': v5_predictor._rotation_advanced(animals),
            'absence_penalty': v5_predictor._continuous_absence_penalty(animals),
            'diversity': v5_predictor._diversity_boost(animals),
            'similarity': v5_predictor._historical_similarity(animals)
        }
        
        final_scores = {}
        for zodiac in v5_predictor.zodiacs:
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
        v5_top5 = [z for z, s in sorted_zodiacs[:5]]
        
        # 统计v7.0命中
        if actual in final_top5:
            rank = final_top5.index(actual) + 1
            if rank == 1:
                final_hits['TOP1'] += 1
                final_hits['TOP2'] += 1
                final_hits['TOP3'] += 1
                final_hits['TOP5'] += 1
            elif rank == 2:
                final_hits['TOP2'] += 1
                final_hits['TOP3'] += 1
                final_hits['TOP5'] += 1
            elif rank == 3:
                final_hits['TOP3'] += 1
                final_hits['TOP5'] += 1
            else:
                final_hits['TOP5'] += 1
        
        # 统计v5.0命中
        if actual in v5_top5:
            rank = v5_top5.index(actual) + 1
            if rank == 1:
                v5_hits['TOP1'] += 1
                v5_hits['TOP2'] += 1
                v5_hits['TOP3'] += 1
                v5_hits['TOP5'] += 1
            elif rank == 2:
                v5_hits['TOP2'] += 1
                v5_hits['TOP3'] += 1
                v5_hits['TOP5'] += 1
            elif rank == 3:
                v5_hits['TOP3'] += 1
                v5_hits['TOP5'] += 1
            else:
                v5_hits['TOP5'] += 1
    
    # 显示结果
    print(f"{'='*100}")
    print(f"整体命中率对比（{n_periods}期）")
    print(f"{'='*100}")
    print(f"{'指标':<15} {'v7.0最终版':<25} {'v5.0冷门版':<25} {'提升'}")
    print('-'*100)
    
    for metric in ['TOP1', 'TOP2', 'TOP3', 'TOP5']:
        final_rate = final_hits[metric] / n_periods * 100
        v5_rate = v5_hits[metric] / n_periods * 100
        improvement = final_rate - v5_rate
        
        symbol = '[UP]' if improvement > 0 else '[--]' if improvement == 0 else '[DN]'
        
        print(f"{metric:<15} {final_hits[metric]:3d}/{n_periods} ({final_rate:5.1f}%)         "
              f"{v5_hits[metric]:3d}/{n_periods} ({v5_rate:5.1f}%)         "
              f"{symbol} {improvement:+5.1f}%")
    
    print(f"{'='*100}")
    
    return final_hits, v5_hits

def test_recent_10_detailed():
    """详细测试最近10期"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    predictor = ZodiacFinalPredictor()
    
    print('\n' + '='*100)
    print('最近10期详细分析（v7.0最终版）')
    print('='*100)
    print()
    
    hits = 0
    
    for i in range(total - 10, total):
        period_num = i + 1
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        
        top5, _ = predictor.predict_from_history(animals, top_n=5, debug=False)
        
        hit = actual in top5
        if hit:
            hits += 1
            rank = top5.index(actual) + 1
        
        print(f"第{period_num}期: 实际={actual:4s}, 预测TOP5={', '.join(top5)}, "
              f"{'[OK]命中第' + str(rank) + '位' if hit else '[X]未中'}")
    
    print()
    print(f"最近10期命中率: {hits}/10 = {hits/10*100:.1f}%")
    print('='*100)

if __name__ == '__main__':
    # 测试最近10期详细
    test_recent_10_detailed()
    
    print('\n\n')
    
    # 测试最近10期对比
    print('='*100)
    print('最近10期对比')
    print('='*100)
    final_10, v5_10 = test_final_vs_v5(n_periods=10)
    
    print('\n\n')
    
    # 测试100期对比
    print('='*100)
    print('100期对比')
    print('='*100)
    final_100, v5_100 = test_final_vs_v5(n_periods=100)
    
    print('\n\n')
    print('='*100)
    print('总结')
    print('='*100)
    print()
    print(f"v7.0 vs v5.0 改进：")
    print(f"  最近10期: {final_10['TOP5']}/10 ({final_10['TOP5']/10*100:.0f}%) vs {v5_10['TOP5']}/10 ({v5_10['TOP5']/10*100:.0f}%) "
          f"= {(final_10['TOP5']-v5_10['TOP5'])/10*100:+.0f}%")
    print(f"  100期整体: {final_100['TOP5']}/100 ({final_100['TOP5']:.0f}%) vs {v5_100['TOP5']}/100 ({v5_100['TOP5']:.0f}%) "
          f"= {final_100['TOP5']-v5_100['TOP5']:+.0f}%")
    print()
    
    if final_100['TOP5'] > v5_100['TOP5']:
        print("  [OK] v7.0在保持100期高命中率的同时，提升了最近期表现")
    elif final_10['TOP5'] > v5_10['TOP5']:
        print("  [OK] v7.0改善了最近期表现")
    else:
        print("  [!] v7.0未能改善表现，需要继续优化")
