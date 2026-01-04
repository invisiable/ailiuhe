"""
在100期大数据集上测试，寻找最优配置
"""

import pandas as pd
from zodiac_super_predictor import ZodiacSuperPredictor

def validate_on_large_dataset(config_name, weights, n_periods=100):
    """在大数据集上验证"""
    predictor = ZodiacSuperPredictor()
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    if total < n_periods + 50:
        print(f"数据不足，仅有{total}期")
        n_periods = min(100, total - 50)
    
    hits = {'TOP1': 0, 'TOP2': 0, 'TOP3': 0, 'TOP5': 0}
    details = []
    
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
        
        final_scores = {}
        for zodiac in predictor.zodiacs:
            score = 0.0
            for strategy_name, weight in weights.items():
                score += strategies_scores[strategy_name].get(zodiac, 0) * weight
            final_scores[zodiac] = score
        
        sorted_zodiacs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top5 = [z for z, s in sorted_zodiacs[:5]]
        
        actual = str(df['animal'].values[i]).strip()
        period = i + 1
        
        hit = False
        if actual in top5:
            hit = True
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
        
        details.append((period, actual, top5, hit))
    
    top5_rate = hits['TOP5'] / n_periods
    
    print(f"\n{'='*70}")
    print(f"配置: {config_name} | 验证期数: {n_periods}")
    print(f"{'='*70}")
    
    print(f"\n命中率统计:")
    print(f"  TOP1: {hits['TOP1']:3d}/{n_periods} = {hits['TOP1']/n_periods*100:5.1f}% (理论 8.3%)")
    print(f"  TOP2: {hits['TOP2']:3d}/{n_periods} = {hits['TOP2']/n_periods*100:5.1f}% (理论16.7%)")
    print(f"  TOP3: {hits['TOP3']:3d}/{n_periods} = {hits['TOP3']/n_periods*100:5.1f}% (理论25.0%)")
    print(f"  TOP5: {hits['TOP5']:3d}/{n_periods} = {hits['TOP5']/n_periods*100:5.1f}% (理论41.7%) {'🎯' if top5_rate >= 0.50 else '⭐' if top5_rate >= 0.45 else ''}")
    
    # 计算提升幅度
    improvements = {
        'TOP1': hits['TOP1']/n_periods - 0.083,
        'TOP2': hits['TOP2']/n_periods - 0.167,
        'TOP3': hits['TOP3']/n_periods - 0.250,
        'TOP5': hits['TOP5']/n_periods - 0.417
    }
    
    print(f"\n提升幅度:")
    for key, val in improvements.items():
        sign = '+' if val >= 0 else ''
        print(f"  {key}: {sign}{val*100:.1f}%")
    
    # 连续命中分析
    max_streak = 0
    current_streak = 0
    for _, _, _, hit in details:
        if hit:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    recent_10_hits = sum(1 for _, _, _, hit in details[-10:] if hit)
    
    print(f"\n连续命中:")
    print(f"  最长连续: {max_streak}期")
    print(f"  最近10期: {recent_10_hits}/10 = {recent_10_hits*10}%")
    
    return top5_rate, hits, details

if __name__ == '__main__':
    print("\n" + "="*70)
    print("大数据集验证（100期）- 寻找50%命中率配置")
    print("="*70)
    
    # 测试最有潜力的几种配置
    
    # 配置1: 当前最佳（35%冷门）
    config1 = {
        'ultra_cold': 0.35,
        'anti_hot': 0.20,
        'gap': 0.18,
        'rotation': 0.12,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置2: 稍微降低冷门至32%，提升轮转
    config2 = {
        'ultra_cold': 0.32,
        'anti_hot': 0.20,
        'gap': 0.18,
        'rotation': 0.15,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置3: 冷门30%，平衡型
    config3 = {
        'ultra_cold': 0.30,
        'anti_hot': 0.20,
        'gap': 0.20,
        'rotation': 0.15,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.03
    }
    
    # 配置4: 尝试提高absence_penalty
    config4 = {
        'ultra_cold': 0.30,
        'anti_hot': 0.18,
        'gap': 0.20,
        'rotation': 0.15,
        'absence_penalty': 0.12,
        'diversity': 0.03,
        'similarity': 0.02
    }
    
    # 配置5: 强化gap和rotation
    config5 = {
        'ultra_cold': 0.28,
        'anti_hot': 0.18,
        'gap': 0.22,
        'rotation': 0.18,
        'absence_penalty': 0.08,
        'diversity': 0.04,
        'similarity': 0.02
    }
    
    configs = [
        ('当前最佳35%冷门', config1),
        ('冷门32%+轮转15%', config2),
        ('冷门30%平衡型', config3),
        ('冷门30%+强惩罚12%', config4),
        ('冷门28%+强gap22%', config5),
    ]
    
    results = []
    for name, config in configs:
        rate, hits, details = validate_on_large_dataset(name, config, n_periods=100)
        results.append((name, rate, hits))
    
    # 最终排名
    print(f"\n{'='*70}")
    print("100期验证 - 最终排名")
    print(f"{'='*70}")
    
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for rank, (name, rate, hits) in enumerate(sorted_results, 1):
        star = "🎯" if rate >= 0.50 else "⭐" if rate >= 0.45 else "★" if rate >= 0.42 else "☆"
        print(f"{rank}. {name:25s} - {rate*100:5.1f}% (TOP5: {hits['TOP5']}/100) {star}")
    
    best_name, best_rate, best_hits = sorted_results[0]
    print(f"\n{'='*70}")
    print(f"🏆 100期最佳: {best_name}")
    print(f"   TOP5命中: {best_hits['TOP5']}/100 = {best_rate*100:.1f}%")
    if best_rate >= 0.50:
        print(f"   ✅ 达到50%目标！")
    else:
        print(f"   距离目标: {(0.50 - best_rate)*100:.1f}%")
    print(f"{'='*70}")
