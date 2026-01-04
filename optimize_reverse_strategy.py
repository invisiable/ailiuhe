"""
反向策略优化 - 降低长期不出现，提升热门权重
目标：通过10次遍历验证，寻找60%命中率配置
"""

import pandas as pd
import numpy as np
from zodiac_super_predictor import ZodiacSuperPredictor
import itertools

def test_reverse_strategy(ultra_cold_weight, anti_hot_weight, hot_boost_weight, n_periods=50):
    """
    测试反向策略配置
    
    参数:
    - ultra_cold_weight: 冷门权重（降低）
    - anti_hot_weight: 避热权重（降低，甚至负值表示鼓励热门）
    - hot_boost_weight: 热门提升权重（新策略）
    """
    predictor = ZodiacSuperPredictor()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    for i in range(total - n_periods, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        
        # 获取策略评分
        strategies_scores = {
            'ultra_cold': predictor._ultra_cold_strategy(animals),
            'anti_hot': predictor._anti_hot_strategy(animals),
            'gap': predictor._gap_analysis(animals),
            'rotation': predictor._rotation_advanced(animals),
            'absence_penalty': predictor._continuous_absence_penalty(animals),
            'diversity': predictor._diversity_boost(animals),
            'similarity': predictor._historical_similarity(animals)
        }
        
        # 新增：热门提升策略
        hot_boost_scores = {}
        recent_10 = animals[-10:] if len(animals) >= 10 else animals
        for zodiac in predictor.zodiacs:
            count = recent_10.count(zodiac)
            # 热门生肖加分
            if count >= 3:
                hot_boost_scores[zodiac] = 8.0
            elif count == 2:
                hot_boost_scores[zodiac] = 5.0
            elif count == 1:
                hot_boost_scores[zodiac] = 2.0
            else:
                hot_boost_scores[zodiac] = 0.0
        
        # 应用权重配置
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            score += strategies_scores['ultra_cold'].get(zodiac, 0) * ultra_cold_weight
            score += strategies_scores['anti_hot'].get(zodiac, 0) * anti_hot_weight
            score += strategies_scores['gap'].get(zodiac, 0) * 0.20
            score += strategies_scores['rotation'].get(zodiac, 0) * 0.15
            score += strategies_scores['absence_penalty'].get(zodiac, 0) * 0.12
            score += strategies_scores['diversity'].get(zodiac, 0) * 0.03
            score += strategies_scores['similarity'].get(zodiac, 0) * 0.02
            score += hot_boost_scores.get(zodiac, 0) * hot_boost_weight
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
    
    return hits['TOP5'] / n_periods

def grid_search_10_iterations():
    """10次遍历搜索最优配置"""
    
    print('='*80)
    print('反向策略优化 - 10次遍历搜索')
    print('目标: 降低长期不出现生肖权重，提升热门生肖权重，达到60%命中率')
    print('='*80)
    print()
    
    # 定义搜索空间（10次遍历）
    iterations = [
        # 迭代1: 基准（当前配置）
        {'name': '迭代1-基准', 'ultra_cold': 0.35, 'anti_hot': 0.20, 'hot_boost': 0.00},
        
        # 迭代2: 降低冷门，减少避热
        {'name': '迭代2-温和', 'ultra_cold': 0.25, 'anti_hot': 0.10, 'hot_boost': 0.08},
        
        # 迭代3: 进一步降低冷门，增加热门
        {'name': '迭代3-激进', 'ultra_cold': 0.15, 'anti_hot': 0.05, 'hot_boost': 0.15},
        
        # 迭代4: 极端反转（鼓励热门）
        {'name': '迭代4-反转', 'ultra_cold': 0.10, 'anti_hot': -0.10, 'hot_boost': 0.25},
        
        # 迭代5: 完全热门导向
        {'name': '迭代5-热门', 'ultra_cold': 0.05, 'anti_hot': -0.15, 'hot_boost': 0.30},
        
        # 迭代6: 平衡策略
        {'name': '迭代6-平衡', 'ultra_cold': 0.20, 'anti_hot': 0.00, 'hot_boost': 0.15},
        
        # 迭代7: 中度热门
        {'name': '迭代7-中热', 'ultra_cold': 0.18, 'anti_hot': 0.02, 'hot_boost': 0.18},
        
        # 迭代8: 零冷门策略
        {'name': '迭代8-零冷', 'ultra_cold': 0.00, 'anti_hot': -0.20, 'hot_boost': 0.35},
        
        # 迭代9: 微冷门+强热门
        {'name': '迭代9-混合', 'ultra_cold': 0.12, 'anti_hot': -0.05, 'hot_boost': 0.22},
        
        # 迭代10: 极致热门
        {'name': '迭代10-极热', 'ultra_cold': 0.00, 'anti_hot': -0.25, 'hot_boost': 0.40}
    ]
    
    results = []
    
    for idx, config in enumerate(iterations, 1):
        print(f"\n{'='*80}")
        print(f"{config['name']}")
        print(f"{'='*80}")
        print(f"  ultra_cold权重: {config['ultra_cold']*100:5.1f}% (冷门)")
        print(f"  anti_hot权重:   {config['anti_hot']*100:5.1f}% (避热，负值=鼓励热门)")
        print(f"  hot_boost权重:  {config['hot_boost']*100:5.1f}% (热门提升)")
        print(f"  其他策略:       48% (gap+rotation+penalty+diversity+similarity)")
        print()
        
        # 在50期数据上测试
        rate_50 = test_reverse_strategy(
            config['ultra_cold'],
            config['anti_hot'],
            config['hot_boost'],
            n_periods=50
        )
        
        # 在100期数据上测试（如果50期效果好）
        if rate_50 >= 0.50:
            print(f"  50期测试: {rate_50*100:.1f}% - 效果良好，扩展到100期验证...")
            rate_100 = test_reverse_strategy(
                config['ultra_cold'],
                config['anti_hot'],
                config['hot_boost'],
                n_periods=100
            )
            print(f"  100期验证: {rate_100*100:.1f}%")
            results.append({
                'config': config,
                'rate_50': rate_50,
                'rate_100': rate_100,
                'best_rate': rate_100
            })
        else:
            print(f"  50期测试: {rate_50*100:.1f}%")
            results.append({
                'config': config,
                'rate_50': rate_50,
                'rate_100': None,
                'best_rate': rate_50
            })
    
    # 排序结果
    results.sort(key=lambda x: x['best_rate'], reverse=True)
    
    # 显示结果
    print(f"\n{'='*80}")
    print("10次迭代结果排名")
    print(f"{'='*80}")
    print()
    print(f"{'排名':<6} {'配置名称':<18} {'50期':<10} {'100期':<10} {'冷门':<8} {'避热':<8} {'热门':<8} {'状态'}")
    print('-'*80)
    
    for rank, r in enumerate(results, 1):
        cfg = r['config']
        status = '🎯' if r['best_rate'] >= 0.60 else '⭐' if r['best_rate'] >= 0.55 else '★' if r['best_rate'] >= 0.50 else '☆'
        rate_100_str = f"{r['rate_100']*100:.1f}%" if r['rate_100'] else "-"
        
        print(f"{rank:<6} {cfg['name']:<18} {r['rate_50']*100:>6.1f}%   {rate_100_str:>8} "
              f"{cfg['ultra_cold']*100:>5.0f}%  {cfg['anti_hot']*100:>6.0f}%  {cfg['hot_boost']*100:>5.0f}%  {status}")
    
    # 最佳配置
    best = results[0]
    print(f"\n{'='*80}")
    print("🏆 最佳配置")
    print(f"{'='*80}")
    print(f"  配置名称: {best['config']['name']}")
    print(f"  TOP5命中率: {best['best_rate']*100:.1f}%")
    print(f"  权重配置:")
    print(f"    - ultra_cold (冷门):        {best['config']['ultra_cold']*100:5.1f}%")
    print(f"    - anti_hot (避热):          {best['config']['anti_hot']*100:5.1f}%")
    print(f"    - hot_boost (热门提升):     {best['config']['hot_boost']*100:5.1f}%")
    print(f"    - gap (间隔):               20.0%")
    print(f"    - rotation (轮转):          15.0%")
    print(f"    - absence_penalty (惩罚):   12.0%")
    print(f"    - diversity (多样性):        3.0%")
    print(f"    - similarity (历史):         2.0%")
    print()
    
    if best['best_rate'] >= 0.60:
        print("  ✅ 达到60%目标！")
    else:
        print(f"  ⚠️ 距离60%目标还差: {(0.60 - best['best_rate'])*100:.1f}%")
    
    print(f"{'='*80}")
    
    return results

if __name__ == '__main__':
    results = grid_search_10_iterations()
