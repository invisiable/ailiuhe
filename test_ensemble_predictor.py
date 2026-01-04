"""
测试集成预测器v8.0
对比单一模型v5.0和集成效果
"""

import pandas as pd
from zodiac_ensemble_predictor import ZodiacEnsemblePredictor
from zodiac_super_predictor import ZodiacSuperPredictor

def test_ensemble_detailed(n_periods=100):
    """详细测试集成预测器"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    ensemble_predictor = ZodiacEnsemblePredictor()
    v5_predictor = ZodiacSuperPredictor()
    
    print('='*100)
    print(f'集成预测器v8.0 详细测试 - {n_periods}期')
    print('='*100)
    print()
    
    # 统计各模型命中
    ensemble_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    model_a_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    model_b_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    model_c_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    v5_hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    def count_hit(top5, actual, hit_dict):
        """统计命中"""
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
        
        # 集成预测
        ensemble_result = ensemble_predictor.predict_from_history(animals, top_n=5)
        ensemble_top5 = ensemble_result['ensemble_top5']
        model_a_top5 = ensemble_result['model_a_top5']
        model_b_top5 = ensemble_result['model_b_top5']
        model_c_top5 = ensemble_result['model_c_top5']
        
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
        
        # 统计命中
        count_hit(ensemble_top5, actual, ensemble_hits)
        count_hit(model_a_top5, actual, model_a_hits)
        count_hit(model_b_top5, actual, model_b_hits)
        count_hit(model_c_top5, actual, model_c_hits)
        count_hit(v5_top5, actual, v5_hits)
    
    # 显示结果
    print(f"{'='*100}")
    print(f"命中率对比（{n_periods}期）")
    print(f"{'='*100}")
    print(f"{'模型':<20} {'TOP1':<12} {'TOP2':<12} {'TOP3':<12} {'TOP5':<12}")
    print('-'*100)
    
    models = [
        ('集成v8.0', ensemble_hits),
        ('模型A-冷门', model_a_hits),
        ('模型B-惯性', model_b_hits),
        ('模型C-间隔', model_c_hits),
        ('v5.0基准', v5_hits)
    ]
    
    for name, hits in models:
        print(f"{name:<20} "
              f"{hits['TOP1']:2d}/{n_periods} ({hits['TOP1']/n_periods*100:5.1f}%)  "
              f"{hits['TOP2']:2d}/{n_periods} ({hits['TOP2']/n_periods*100:5.1f}%)  "
              f"{hits['TOP3']:2d}/{n_periods} ({hits['TOP3']/n_periods*100:5.1f}%)  "
              f"{hits['TOP5']:2d}/{n_periods} ({hits['TOP5']/n_periods*100:5.1f}%)")
    
    print(f"{'='*100}")
    print()
    
    # 对比v5.0
    print("集成v8.0 vs v5.0基准:")
    for metric in ['TOP1', 'TOP2', 'TOP3', 'TOP5']:
        improvement = ensemble_hits[metric] - v5_hits[metric]
        symbol = '[UP]' if improvement > 0 else '[==]' if improvement == 0 else '[DN]'
        print(f"  {metric}: {symbol} {improvement:+2d} ({improvement/n_periods*100:+5.1f}%)")
    
    print()
    
    return ensemble_hits, v5_hits

def test_recent_periods_detailed():
    """详细测试最近期表现"""
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    predictor = ZodiacEnsemblePredictor()
    
    print('\n' + '='*100)
    print('最近20期详细分析（集成v8.0）')
    print('='*100)
    print()
    
    hits = 0
    
    for i in range(total - 20, total):
        period_num = i + 1
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        actual = str(df['animal'].values[i]).strip()
        
        result = predictor.predict_from_history(animals, top_n=5, debug=False)
        ensemble_top5 = result['ensemble_top5']
        
        hit = actual in ensemble_top5
        if hit:
            hits += 1
            rank = ensemble_top5.index(actual) + 1
        
        # 显示各模型的预测
        consensus = sum([
            actual in result['model_a_top5'],
            actual in result['model_b_top5'],
            actual in result['model_c_top5']
        ])
        
        status = f"[OK]第{rank}位" if hit else "[X]未中"
        consensus_str = f"({consensus}/3模型一致)" if hit else f"({consensus}/3模型)"
        
        print(f"第{period_num:3d}期: 实际={actual:4s}, 集成预测={', '.join(ensemble_top5)}, {status} {consensus_str}")
    
    print()
    print(f"最近20期命中率: {hits}/20 = {hits/20*100:.1f}%")
    print('='*100)
    
    # 分段统计
    print()
    print("分段统计:")
    
    segments = [
        (total-20, total-10, "最近11-20期"),
        (total-10, total, "最近1-10期")
    ]
    
    for start, end, label in segments:
        seg_hits = 0
        seg_total = end - start
        
        for i in range(start, end):
            animals = [str(a).strip() for a in df['animal'].values[:i]]
            actual = str(df['animal'].values[i]).strip()
            result = predictor.predict_from_history(animals, top_n=5, debug=False)
            
            if actual in result['ensemble_top5']:
                seg_hits += 1
        
        print(f"  {label}: {seg_hits}/{seg_total} = {seg_hits/seg_total*100:.1f}%")

if __name__ == '__main__':
    # 测试最近20期
    test_recent_periods_detailed()
    
    print('\n\n')
    
    # 测试最近10期对比
    print('='*100)
    print('最近10期对比')
    print('='*100)
    ensemble_10, v5_10 = test_ensemble_detailed(n_periods=10)
    
    print('\n\n')
    
    # 测试最近50期对比
    print('='*100)
    print('最近50期对比')
    print('='*100)
    ensemble_50, v5_50 = test_ensemble_detailed(n_periods=50)
    
    print('\n\n')
    
    # 测试100期对比
    print('='*100)
    print('100期对比')
    print('='*100)
    ensemble_100, v5_100 = test_ensemble_detailed(n_periods=100)
    
    print('\n\n')
    print('='*100)
    print('综合总结')
    print('='*100)
    print()
    print("集成v8.0 vs v5.0基准 - TOP5命中率对比:")
    print(f"  最近10期:  {ensemble_10['TOP5']}/10  ({ensemble_10['TOP5']/10*100:5.1f}%) vs "
          f"{v5_10['TOP5']}/10  ({v5_10['TOP5']/10*100:5.1f}%) = {(ensemble_10['TOP5']-v5_10['TOP5'])/10*100:+5.1f}%")
    print(f"  最近50期:  {ensemble_50['TOP5']}/50  ({ensemble_50['TOP5']/50*100:5.1f}%) vs "
          f"{v5_50['TOP5']}/50  ({v5_50['TOP5']/50*100:5.1f}%) = {(ensemble_50['TOP5']-v5_50['TOP5'])/50*100:+5.1f}%")
    print(f"  100期整体: {ensemble_100['TOP5']}/100 ({ensemble_100['TOP5']:.0f}%) vs "
          f"{v5_100['TOP5']}/100 ({v5_100['TOP5']:.0f}%) = {ensemble_100['TOP5']-v5_100['TOP5']:+.0f}%")
    print()
    
    if ensemble_100['TOP5'] >= v5_100['TOP5'] and ensemble_10['TOP5'] > v5_10['TOP5']:
        print("  [OK] 集成模型在保持整体性能的同时，显著提升了最近期表现！")
    elif ensemble_100['TOP5'] > v5_100['TOP5']:
        print("  [OK] 集成模型整体性能优于基准！")
    elif ensemble_10['TOP5'] > v5_10['TOP5']:
        print("  [~] 集成模型改善了最近期表现，但整体性能略有下降")
    else:
        print("  [!] 集成模型未能改善性能")
    
    print('='*100)
