"""
测试不同权重配置对预测效果的影响
"""

import pandas as pd
import numpy as np
from zodiac_super_predictor import ZodiacSuperPredictor

def test_configuration(config_name, weights, n_periods=30):
    """测试特定权重配置"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}")
    
    # 创建预测器实例
    predictor = ZodiacSuperPredictor()
    
    # 读取数据
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    
    # 验证最近N期
    for i in range(total - n_periods, total):
        # 使用前i期数据进行预测
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
        
        # 加权融合
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            for strategy_name, weight in weights.items():
                score += strategies_scores[strategy_name].get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        # 排序
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top5 = [z for z, s in sorted_zodiacs[:5]]
        
        # 实际结果
        actual = str(df['animal'].values[i]).strip()
        
        # 统计命中
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
    
    # 输出结果
    print(f"\n权重配置:")
    for strategy, weight in weights.items():
        print(f"  {strategy:20s}: {weight:.2%}")
    
    print(f"\n命中率统计:")
    print(f"  TOP1: {hits['TOP1']}/{n_periods} = {hits['TOP1']/n_periods*100:.1f}%")
    print(f"  TOP2: {hits['TOP2']}/{n_periods} = {hits['TOP2']/n_periods*100:.1f}%")
    print(f"  TOP3: {hits['TOP3']}/{n_periods} = {hits['TOP3']/n_periods*100:.1f}%")
    print(f"  TOP5: {hits['TOP5']}/{n_periods} = {hits['TOP5']/n_periods*100:.1f}% ⭐")
    
    return hits['TOP5'] / n_periods

if __name__ == '__main__':
    # 配置1: 原始配置（v1）
    config1 = {
        'ultra_cold': 0.35,
        'anti_hot': 0.25,
        'gap': 0.20,
        'rotation': 0.12,
        'absence_penalty': 0.00,
        'diversity': 0.05,
        'similarity': 0.03
    }
    
    # 配置2: 当前配置（v4 - 加入惩罚机制）
    config2 = {
        'ultra_cold': 0.28,
        'anti_hot': 0.25,
        'gap': 0.18,
        'rotation': 0.14,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置3: 保守型（更少冷门）
    config3 = {
        'ultra_cold': 0.20,
        'anti_hot': 0.30,
        'gap': 0.20,
        'rotation': 0.15,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置4: 激进型（更多冷门）
    config4 = {
        'ultra_cold': 0.35,
        'anti_hot': 0.20,
        'gap': 0.18,
        'rotation': 0.12,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置5: 平衡型
    config5 = {
        'ultra_cold': 0.25,
        'anti_hot': 0.25,
        'gap': 0.20,
        'rotation': 0.15,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置6: 轮转优先型
    config6 = {
        'ultra_cold': 0.22,
        'anti_hot': 0.22,
        'gap': 0.18,
        'rotation': 0.20,
        'absence_penalty': 0.10,
        'diversity': 0.05,
        'similarity': 0.03
    }
    
    # 测试所有配置
    configs = {
        '原始v1（无惩罚）': config1,
        '当前v4（微调+惩罚）': config2,
        '保守型（少冷门）': config3,
        '激进型（多冷门）': config4,
        '平衡型': config5,
        '轮转优先型': config6
    }
    
    results = {}
    for name, config in configs.items():
        rate = test_configuration(name, config, n_periods=50)
        results[name] = rate
    
    # 总结
    print(f"\n{'='*60}")
    print("最终排名（按TOP5命中率）")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, rate) in enumerate(sorted_results, 1):
        star = "★" if rate >= 0.45 else "☆"
        print(f"{rank}. {name:25s} - {rate*100:.1f}% {star}")
    
    print(f"\n推荐配置: {sorted_results[0][0]}")
