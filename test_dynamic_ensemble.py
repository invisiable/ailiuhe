"""
测试动态集成预测器v8.1
"""

import pandas as pd
from zodiac_dynamic_ensemble import ZodiacDynamicEnsemble
from zodiac_super_predictor import ZodiacSuperPredictor

def test_dynamic_ensemble(n_periods=100):
    """测试动态集成"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    dynamic_predictor = ZodiacDynamicEnsemble()
    v5_predictor = ZodiacSuperPredictor()
    
    print('='*100)
    print(f'动态集成v8.1 vs v5.0基准 - {n_periods}期对比')
    print('='*100)
    print()
    
    dynamic_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    v5_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    mode_stats = {'基准模式(v5.0)': 0, '混合模式(70%基准+30%惯性)': 0}
    
    def count_hit(top5, actual, hit_dict):
        if actual in top5:
            rank = top5.index(actual) + 1
            if rank == 1:
                hit_dict['TOP1'] += 1
                hit_dict['TOP2'] += 1
                hit_dict['TOP3'] += 1
                hit_dict['TOP5'] += 1
            elif rank == 2:
                hit_dict['TOP2'] += 1
                hit_dict['TOP3'] += 1
                hit_dict['TOP5'] += 1
            elif rank == 3:
                hit_dict['TOP3'] += 1
                hit_dict['TOP5'] += 1
            else:
                hit_dict['TOP5'] += 1
    
    for i in range(total - n_periods, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        
        # 动态预测
        dynamic_top5, _, mode = dynamic_predictor.predict_from_history(animals, top_n=5)
        mode_stats[mode] = mode_stats.get(mode, 0) + 1
        
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
        
        count_hit(dynamic_top5, actual, dynamic_hits)
        count_hit(v5_top5, actual, v5_hits)
    
    # 显示结果
    print(f"{'='*100}")
    print(f"命中率对比（{n_periods}期）")
    print(f"{'='*100}")
    print(f"{'指标':<15} {'v8.1动态集成':<25} {'v5.0基准':<25} {'提升'}")
    print('-'*100)
    
    for metric in ['TOP1', 'TOP2', 'TOP3', 'TOP5']:
        dynamic_rate = dynamic_hits[metric] / n_periods * 100
        v5_rate = v5_hits[metric] / n_periods * 100
        improvement = dynamic_rate - v5_rate
        symbol = '[UP]' if improvement > 0 else '[==]' if improvement == 0 else '[DN]'
        
        print(f"{metric:<15} {dynamic_hits[metric]:3d}/{n_periods} ({dynamic_rate:5.1f}%)         "
              f"{v5_hits[metric]:3d}/{n_periods} ({v5_rate:5.1f}%)         "
              f"{symbol} {improvement:+5.1f}%")
    
    print(f"{'='*100}")
    print(f"\n模式使用统计（{n_periods}期）:")
    for mode, count in mode_stats.items():
        print(f"  {mode}: {count}/{n_periods} ({count/n_periods*100:.1f}%)")
    print()
    
    return dynamic_hits, v5_hits

def test_recent_20():
    """测试最近20期"""
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    predictor = ZodiacDynamicEnsemble()
    
    print('\n' + '='*100)
    print('最近20期详细分析（v8.1动态集成）')
    print('='*100)
    print()
    
    hits = 0
    for i in range(total - 20, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        top5, _, mode = predictor.predict_from_history(animals, top_n=5, debug=False)
        
        hit = actual in top5
        if hit:
            hits += 1
            rank = top5.index(actual) + 1
        
        mode_short = "基准" if "基准" in mode else "混合"
        print(f"第{i+1:3d}期: 实际={actual:4s}, 预测={', '.join(top5)}, "
              f"模式={mode_short:4s}, {'[OK]第'+str(rank)+'位' if hit else '[X]未中'}")
    
    print()
    print(f"最近20期命中率: {hits}/20 = {hits/20*100:.1f}%")
    print('='*100)

if __name__ == '__main__':
    # 测试最近20期
    test_recent_20()
    
    print('\n\n')
    
    # 测试各期段
    for n in [10, 50, 100]:
        print('='*100)
        print(f'{n}期对比')
        print('='*100)
        test_dynamic_ensemble(n_periods=n)
        print('\n\n')
    
    print('='*100)
    print('总结')
    print('='*100)
    print("\n动态集成v8.1特点:")
    print("  1. 默认使用v5.0最优配置（52%命中率）")
    print("  2. 检测到热门惯性时，混合30%惯性权重")
    print("  3. 在保持整体稳定性的同时，适应局部特征变化")
