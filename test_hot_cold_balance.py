"""
测试冷热平衡优化 - 寻找50%命中率配置
降低冷门权重，调整热门策略
"""

import pandas as pd
import numpy as np
from zodiac_super_predictor import ZodiacSuperPredictor

def test_configuration(config_name, weights, n_periods=50):
    """测试特定权重配置"""
    predictor = ZodiacSuperPredictor()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    for i in range(total - n_periods, total):
        animals = [str(a).strip() for a in df['animal'].values[:i]]
        
        # 手动应用权重配置
        strategies_scores = {
            'ultra_cold': predictor._ultra_cold_strategy(animals),
            'anti_hot': predictor._anti_hot_strategy(animals),
            'gap': predictor._gap_analysis(animals),
            'rotation': predictor._rotation_advanced(animals),
            'absence_penalty': predictor._continuous_absence_penalty(animals),
            'diversity': predictor._diversity_boost(animals),
            'similarity': predictor._historical_similarity(animals)
        }
        
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            for strategy_name, weight in weights.items():
                score += strategies_scores[strategy_name].get(zodiac, 0) * weight
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
    
    top5_rate = hits['TOP5'] / n_periods
    
    print(f"\n{'='*70}")
    print(f"配置: {config_name}")
    print(f"{'='*70}")
    print(f"\n权重配置:")
    for strategy, weight in weights.items():
        if weight > 0:
            print(f"  {strategy:20s}: {weight:5.1%}")
    
    print(f"\n命中率:")
    print(f"  TOP1: {hits['TOP1']:2d}/{n_periods} = {hits['TOP1']/n_periods*100:5.1f}%")
    print(f"  TOP2: {hits['TOP2']:2d}/{n_periods} = {hits['TOP2']/n_periods*100:5.1f}%")
    print(f"  TOP3: {hits['TOP3']:2d}/{n_periods} = {hits['TOP3']/n_periods*100:5.1f}%")
    print(f"  TOP5: {hits['TOP5']:2d}/{n_periods} = {hits['TOP5']/n_periods*100:5.1f}% {'⭐' if top5_rate >= 0.50 else '★' if top5_rate >= 0.45 else ''}")
    
    return top5_rate

if __name__ == '__main__':
    print("\n" + "="*70)
    print("冷热平衡优化测试 - 目标: 50%命中率")
    print("="*70)
    
    # 配置1: 当前激进型（基准）42%
    config1 = {
        'ultra_cold': 0.35,
        'anti_hot': 0.20,
        'gap': 0.18,
        'rotation': 0.12,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置2: 降低冷门30%
    config2 = {
        'ultra_cold': 0.30,
        'anti_hot': 0.20,
        'gap': 0.20,
        'rotation': 0.15,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置3: 进一步降低冷门25%，提升轮转
    config3 = {
        'ultra_cold': 0.25,
        'anti_hot': 0.20,
        'gap': 0.22,
        'rotation': 0.18,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置4: 冷门20%，强化间隔和轮转
    config4 = {
        'ultra_cold': 0.20,
        'anti_hot': 0.20,
        'gap': 0.25,
        'rotation': 0.20,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置5: 极低冷门15%，主要靠间隔
    config5 = {
        'ultra_cold': 0.15,
        'anti_hot': 0.18,
        'gap': 0.28,
        'rotation': 0.22,
        'absence_penalty': 0.10,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置6: 均衡型（所有策略接近）
    config6 = {
        'ultra_cold': 0.22,
        'anti_hot': 0.22,
        'gap': 0.22,
        'rotation': 0.18,
        'absence_penalty': 0.10,
        'diversity': 0.04,
        'similarity': 0.02
    }
    
    # 配置7: 降低冷门25%，同时降低anti_hot（不过度回避热门）
    config7 = {
        'ultra_cold': 0.25,
        'anti_hot': 0.15,
        'gap': 0.25,
        'rotation': 0.20,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置8: 冷门20%，热门回避降至10%
    config8 = {
        'ultra_cold': 0.20,
        'anti_hot': 0.10,
        'gap': 0.28,
        'rotation': 0.25,
        'absence_penalty': 0.10,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    configs = {
        '基准-激进型35%冷门': config1,
        '降低冷门至30%': config2,
        '降低冷门至25%': config3,
        '降低冷门至20%': config4,
        '极低冷门15%': config5,
        '均衡型22%': config6,
        '冷门25%+热门回避15%': config7,
        '冷门20%+热门回避10%': config8,
    }
    
    results = {}
    for name, config in configs.items():
        rate = test_configuration(name, config, n_periods=50)
        results[name] = rate
    
    # 总结
    print(f"\n{'='*70}")
    print("最终排名（按TOP5命中率）")
    print(f"{'='*70}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, rate) in enumerate(sorted_results, 1):
        star = "🎯" if rate >= 0.50 else "⭐" if rate >= 0.45 else "★" if rate >= 0.42 else "☆"
        print(f"{rank}. {name:30s} - {rate*100:5.1f}% {star}")
    
    best_name, best_rate = sorted_results[0]
    print(f"\n{'='*70}")
    print(f"🏆 最佳配置: {best_name}")
    print(f"   命中率: {best_rate*100:.1f}%")
    if best_rate >= 0.50:
        print(f"   ✅ 达到50%目标！")
    else:
        print(f"   距离目标还差: {(0.50 - best_rate)*100:.1f}%")
    print(f"{'='*70}")
