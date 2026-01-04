"""
精细化策略优化 - 第二轮10次迭代
基于第一轮结果，在更大样本上测试
"""

import pandas as pd
import numpy as np
from zodiac_super_predictor import ZodiacSuperPredictor

def test_config_detailed(ultra_cold, anti_hot, gap, rotation, absence, hot_boost, n_periods=100):
    """详细测试配置"""
    predictor = ZodiacSuperPredictor()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    for i in range(total - n_periods, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        
        strategies_scores = {
            'ultra_cold': predictor._ultra_cold_strategy(animals),
            'anti_hot': predictor._anti_hot_strategy(animals),
            'gap': predictor._gap_analysis(animals),
            'rotation': predictor._rotation_advanced(animals),
            'absence_penalty': predictor._continuous_absence_penalty(animals),
            'diversity': predictor._diversity_boost(animals),
            'similarity': predictor._historical_similarity(animals)
        }
        
        # 热门提升策略
        hot_boost_scores = {}
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        for zodiac in predictor.zodiacs:
            count = recent_10.count(zodiac)
            if count >= 3:
                hot_boost_scores[zodiac] = 8.0
            elif count == 2:
                hot_boost_scores[zodiac] = 5.0
            elif count == 1:
                hot_boost_scores[zodiac] = 2.0
            else:
                hot_boost_scores[zodiac] = 0.0
        
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            score += strategies_scores['ultra_cold'].get(zodiac, 0) * ultra_cold
            score += strategies_scores['anti_hot'].get(zodiac, 0) * anti_hot
            score += strategies_scores['gap'].get(zodiac, 0) * gap
            score += strategies_scores['rotation'].get(zodiac, 0) * rotation
            score += strategies_scores['absence_penalty'].get(zodiac, 0) * absence
            score += strategies_scores['diversity'].get(zodiac, 0) * 0.04
            score += strategies_scores['similarity'].get(zodiac, 0) * 0.03
            score += hot_boost_scores.get(zodiac, 0) * hot_boost
            final_scores[zodiac] = score
        
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top5 = [z for z, s in sorted_zodiacs[:5]]
        
        actual = str(df['animal'].values[i]).strip()
        
        if actual in top5:
            rank = top5.index(actual) + 1
            if rank == 1:
                hits['TOP1'] += 1
                hits['TOP2'] += 1
                hits['TOP3'] += 1
                hits['TOP5'] += 1
            elif rank == 2:
                hits['TOP2'] += 1
                hits['TOP3'] += 1
                hits['TOP5'] += 1
            elif rank == 3:
                hits['TOP3'] += 1
                hits['TOP5'] += 1
            else:
                hits['TOP5'] += 1
    
    return {
        'TOP1': hits['TOP1'] / n_periods,
        'TOP2': hits['TOP2'] / n_periods,
        'TOP3': hits['TOP3'] / n_periods,
        'TOP5': hits['TOP5'] / n_periods
    }

def second_round_10_iterations():
    """第二轮10次精细化迭代"""
    
    print('='*90)
    print('第二轮精细化优化 - 100期验证')
    print('基于第一轮结果，微调权重配置')
    print('='*90)
    print()
    
    # 第二轮配置（基于第一轮最佳结果微调）
    configs = [
        # 配置1: 当前最佳（基准）
        {'name': '配置1-当前最佳', 'ultra_cold': 0.35, 'anti_hot': 0.20, 'gap': 0.18, 'rotation': 0.12, 'absence': 0.08, 'hot_boost': 0.00},
        
        # 配置2: 微调冷门+5%
        {'name': '配置2-强冷门', 'ultra_cold': 0.40, 'anti_hot': 0.18, 'gap': 0.16, 'rotation': 0.12, 'absence': 0.08, 'hot_boost': 0.00},
        
        # 配置3: 强化轮转
        {'name': '配置3-强轮转', 'ultra_cold': 0.30, 'anti_hot': 0.20, 'gap': 0.18, 'rotation': 0.20, 'absence': 0.05, 'hot_boost': 0.00},
        
        # 配置4: 强化间隔
        {'name': '配置4-强间隔', 'ultra_cold': 0.28, 'anti_hot': 0.18, 'gap': 0.25, 'rotation': 0.15, 'absence': 0.07, 'hot_boost': 0.00},
        
        # 配置5: 强化惩罚
        {'name': '配置5-强惩罚', 'ultra_cold': 0.30, 'anti_hot': 0.18, 'gap': 0.18, 'rotation': 0.12, 'absence': 0.15, 'hot_boost': 0.00},
        
        # 配置6: 平衡优化
        {'name': '配置6-全平衡', 'ultra_cold': 0.25, 'anti_hot': 0.20, 'gap': 0.20, 'rotation': 0.18, 'absence': 0.10, 'hot_boost': 0.00},
        
        # 配置7: 降冷门+强gap
        {'name': '配置7-冷30+gap22', 'ultra_cold': 0.30, 'anti_hot': 0.18, 'gap': 0.22, 'rotation': 0.15, 'absence': 0.08, 'hot_boost': 0.00},
        
        # 配置8: 极致冷门
        {'name': '配置8-极冷门', 'ultra_cold': 0.45, 'anti_hot': 0.15, 'gap': 0.15, 'rotation': 0.12, 'absence': 0.06, 'hot_boost': 0.00},
        
        # 配置9: 中冷+强轮转gap
        {'name': '配置9-冷28混合', 'ultra_cold': 0.28, 'anti_hot': 0.18, 'gap': 0.20, 'rotation': 0.18, 'absence': 0.09, 'hot_boost': 0.00},
        
        # 配置10: 试验热门策略
        {'name': '配置10-微热门', 'ultra_cold': 0.30, 'anti_hot': 0.15, 'gap': 0.18, 'rotation': 0.15, 'absence': 0.08, 'hot_boost': 0.07}
    ]
    
    results = []
    
    for idx, cfg in enumerate(configs, 1):
        print(f"{'='*90}")
        print(f"测试 {idx}/10: {cfg['name']}")
        print(f"{'='*90}")
        print(f"  权重配置:")
        print(f"    ultra_cold:   {cfg['ultra_cold']*100:5.1f}%")
        print(f"    anti_hot:     {cfg['anti_hot']*100:5.1f}%")
        print(f"    gap:          {cfg['gap']*100:5.1f}%")
        print(f"    rotation:     {cfg['rotation']*100:5.1f}%")
        print(f"    absence:      {cfg['absence']*100:5.1f}%")
        print(f"    hot_boost:    {cfg['hot_boost']*100:5.1f}%")
        print(f"    其他(div+sim):  7.0%")
        
        # 100期测试
        print(f"\n  正在100期验证...")
        rates = test_config_detailed(
            cfg['ultra_cold'], cfg['anti_hot'], cfg['gap'],
            cfg['rotation'], cfg['absence'], cfg['hot_boost'],
            n_periods=100
        )
        
        print(f"  结果:")
        print(f"    TOP1: {rates['TOP1']*100:5.1f}% (理论8.3%)")
        print(f"    TOP2: {rates['TOP2']*100:5.1f}% (理论16.7%)")
        print(f"    TOP3: {rates['TOP3']*100:5.1f}% (理论25.0%)")
        print(f"    TOP5: {rates['TOP5']*100:5.1f}% (理论41.7%) {'🎯' if rates['TOP5'] >= 0.60 else '⭐' if rates['TOP5'] >= 0.55 else '★' if rates['TOP5'] >= 0.50 else '☆'}")
        print()
        
        results.append({
            'config': cfg,
            'rates': rates
        })
    
    # 排序
    results.sort(key=lambda x: x['rates']['TOP5'], reverse=True)
    
    # 显示排名
    print(f"{'='*90}")
    print("100期验证结果排名")
    print(f"{'='*90}")
    print()
    print(f"{'排名':<6} {'配置名称':<20} {'TOP1':<8} {'TOP2':<8} {'TOP3':<8} {'TOP5':<8} {'状态'}")
    print('-'*90)
    
    for rank, r in enumerate(results, 1):
        rates = r['rates']
        status = '🎯' if rates['TOP5'] >= 0.60 else '⭐' if rates['TOP5'] >= 0.55 else '★' if rates['TOP5'] >= 0.50 else '☆'
        print(f"{rank:<6} {r['config']['name']:<20} {rates['TOP1']*100:>5.1f}%  {rates['TOP2']*100:>5.1f}%  "
              f"{rates['TOP3']*100:>5.1f}%  {rates['TOP5']*100:>5.1f}%  {status}")
    
    # 最佳配置
    best = results[0]
    print(f"\n{'='*90}")
    print("🏆 最佳配置（100期验证）")
    print(f"{'='*90}")
    print(f"  配置: {best['config']['name']}")
    print(f"\n  性能:")
    print(f"    TOP1命中率: {best['rates']['TOP1']*100:5.1f}% (超理论{(best['rates']['TOP1']-0.083)*100:+5.1f}%)")
    print(f"    TOP2命中率: {best['rates']['TOP2']*100:5.1f}% (超理论{(best['rates']['TOP2']-0.167)*100:+5.1f}%)")
    print(f"    TOP3命中率: {best['rates']['TOP3']*100:5.1f}% (超理论{(best['rates']['TOP3']-0.250)*100:+5.1f}%)")
    print(f"    TOP5命中率: {best['rates']['TOP5']*100:5.1f}% (超理论{(best['rates']['TOP5']-0.417)*100:+5.1f}%)")
    print(f"\n  权重配置:")
    print(f"    ultra_cold:        {best['config']['ultra_cold']*100:5.1f}%")
    print(f"    anti_hot:          {best['config']['anti_hot']*100:5.1f}%")
    print(f"    gap:               {best['config']['gap']*100:5.1f}%")
    print(f"    rotation:          {best['config']['rotation']*100:5.1f}%")
    print(f"    absence_penalty:   {best['config']['absence']*100:5.1f}%")
    print(f"    hot_boost:         {best['config']['hot_boost']*100:5.1f}%")
    print(f"    diversity+similar:  7.0%")
    print()
    
    if best['rates']['TOP5'] >= 0.60:
        print("  ✅ 达到60%目标！")
    elif best['rates']['TOP5'] >= 0.55:
        print(f"  ⚠️ 接近目标，距60%还差: {(0.60 - best['rates']['TOP5'])*100:.1f}%")
    else:
        print(f"  ⚠️ 距离60%目标还差: {(0.60 - best['rates']['TOP5'])*100:.1f}%")
    
    print(f"{'='*90}")
    
    return results

if __name__ == '__main__':
    results = second_round_10_iterations()
