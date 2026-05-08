"""测试增强版PreciseTop15 - 使用全历史数据 + 多种优化"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

def enhanced_predict(all_numbers, current_idx, k=15):
    """增强版预测 - 使用全历史 + PreciseTop15核心方法"""
    predictor = PreciseTop15Predictor()
    
    # 使用25期窗口调用原始方法获取基础得分
    window25 = all_numbers[max(0, current_idx-25):current_idx]
    pattern = predictor.analyze_pattern(window25)
    if len(window25) >= 50:
        pattern['recent_50'] = window25[-50:]
    else:
        pattern['recent_50'] = window25
    
    # 获取PreciseTop15的各方法结果（扩大候选到30）
    base_k = 30
    precise_freq = predictor.method_precision_frequency(pattern, base_k)
    zone_dynamic = predictor.method_zone_dynamic(pattern, base_k)
    gap_analysis = predictor.method_gap_analysis(pattern, base_k)
    avoid_misses = predictor.method_avoid_recent_misses(pattern, base_k)
    
    # 原始PreciseTop15评分
    base_scores = {}
    for candidates, weight in [(precise_freq, 0.40), (zone_dynamic, 0.25), 
                                (gap_analysis, 0.20), (avoid_misses, 0.15)]:
        for rank, num in enumerate(candidates):
            score = weight * (1.0 - rank / len(candidates))
            base_scores[num] = base_scores.get(num, 0) + score
    
    # === 新增：全历史分析 ===
    full_history = all_numbers[:current_idx]
    recent3 = set(all_numbers[max(0,current_idx-3):current_idx])
    recent5 = set(all_numbers[max(0,current_idx-5):current_idx])
    
    # 1. 全历史间隔分析（关键改进）
    gap_scores = {}
    for n in range(1, 50):
        gap = len(full_history) + 1
        for j in range(len(full_history)-1, max(0, len(full_history)-200), -1):
            if full_history[j] == n:
                gap = len(full_history) - j
                break
        
        # 间隔得分 - 基于验证数据
        if gap <= 3:
            gap_scores[n] = 0.0
        elif gap <= 5:
            gap_scores[n] = 0.05
        elif gap <= 10:
            gap_scores[n] = 0.30
        elif gap <= 15:
            gap_scores[n] = 0.30
        elif gap <= 20:
            gap_scores[n] = 0.25
        elif gap <= 30:
            gap_scores[n] = 0.18
        elif gap <= 50:
            gap_scores[n] = 0.08
        else:
            gap_scores[n] = 0.03
    
    # 2. 全历史频率偏差（回补效应）
    overall_freq = Counter(full_history)
    expected = len(full_history) / 49.0
    recovery_scores = {}
    for n in range(1, 50):
        actual_f = overall_freq.get(n, 0)
        deficit = expected - actual_f
        if deficit > 0:
            recovery_scores[n] = min(deficit / expected * 0.15, 0.15)
        else:
            recovery_scores[n] = 0.0
    
    # 3. 区域平衡修正（31-40区加强）
    zone_fix = {}
    for n in range(1, 50):
        if 31 <= n <= 40:
            zone_fix[n] = 0.08  # 补偿31-40区
        elif 1 <= n <= 10:
            zone_fix[n] = -0.02  # 1-10区已经很好
        else:
            zone_fix[n] = 0.0
    
    # === 综合评分 ===
    final_scores = {}
    for n in range(1, 50):
        # 基础分 (60%) + 间隔分 (25%) + 回补分 (10%) + 区域修正 (5%)
        base = base_scores.get(n, 0)
        gap = gap_scores.get(n, 0)
        recovery = recovery_scores.get(n, 0)
        zone = zone_fix.get(n, 0)
        
        final_scores[n] = base * 0.60 + gap * 0.25 + recovery * 0.10 + zone * 0.05
        
        # 最近3期强惩罚
        if n in recent3:
            final_scores[n] *= 0.15
        elif n in recent5:
            final_scores[n] *= 0.35
    
    sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:k]]

# Test with different configurations
print("=== 增强版PreciseTop15 测试 ===\n")

# Different k values
for k in [15, 16, 17, 18, 19, 20, 22]:
    hits = 0
    for i in range(start_idx, len(df)):
        preds = enhanced_predict(numbers, i, k)
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    random_rate = k / 49 * 100
    skill = rate - random_rate
    marker = " ⭐⭐⭐" if rate >= 50 else (" ⭐" if rate >= 45 else "")
    print(f"  Top{k:2d}: {hits}/{TEST_PERIODS} = {rate:.1f}% (随机:{random_rate:.1f}%, 技巧:+{skill:.1f}%){marker}")

# Now test different weight combinations for k=15
print(f"\n=== 权重组合搜索 (k=15) ===")
best_rate = 0
best_config = None
configs = [
    (0.50, 0.30, 0.15, 0.05, "base50-gap30-rec15-zone5"),
    (0.60, 0.25, 0.10, 0.05, "base60-gap25-rec10-zone5"),
    (0.70, 0.20, 0.05, 0.05, "base70-gap20-rec5-zone5"),
    (0.40, 0.35, 0.15, 0.10, "base40-gap35-rec15-zone10"),
    (0.55, 0.30, 0.10, 0.05, "base55-gap30-rec10-zone5"),
    (0.45, 0.35, 0.10, 0.10, "base45-gap35-rec10-zone10"),
    (0.50, 0.25, 0.15, 0.10, "base50-gap25-rec15-zone10"),
]

for w_base, w_gap, w_rec, w_zone, name in configs:
    hits = 0
    for i in range(start_idx, len(df)):
        predictor = PreciseTop15Predictor()
        window25 = numbers[max(0, i-25):i]
        pattern = predictor.analyze_pattern(window25)
        pattern['recent_50'] = window25
        
        base_k = 30
        methods = [
            (predictor.method_precision_frequency(pattern, base_k), 0.40),
            (predictor.method_zone_dynamic(pattern, base_k), 0.25),
            (predictor.method_gap_analysis(pattern, base_k), 0.20),
            (predictor.method_avoid_recent_misses(pattern, base_k), 0.15)
        ]
        base_scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                base_scores[num] = base_scores.get(num, 0) + score
        
        full_history = numbers[:i]
        recent3 = set(numbers[max(0,i-3):i])
        recent5 = set(numbers[max(0,i-5):i])
        
        gap_scores = {}
        for n in range(1, 50):
            gap = len(full_history) + 1
            for j in range(len(full_history)-1, max(0, len(full_history)-200), -1):
                if full_history[j] == n:
                    gap = len(full_history) - j
                    break
            if gap <= 3: gap_scores[n] = 0.0
            elif gap <= 5: gap_scores[n] = 0.05
            elif gap <= 10: gap_scores[n] = 0.30
            elif gap <= 15: gap_scores[n] = 0.30
            elif gap <= 20: gap_scores[n] = 0.25
            elif gap <= 30: gap_scores[n] = 0.18
            elif gap <= 50: gap_scores[n] = 0.08
            else: gap_scores[n] = 0.03
        
        overall_freq = Counter(full_history)
        expected = len(full_history) / 49.0
        recovery_scores = {}
        for n in range(1, 50):
            actual_f = overall_freq.get(n, 0)
            deficit = expected - actual_f
            recovery_scores[n] = min(deficit / expected * 0.15, 0.15) if deficit > 0 else 0.0
        
        zone_fix = {}
        for n in range(1, 50):
            if 31 <= n <= 40: zone_fix[n] = 0.08
            elif 1 <= n <= 10: zone_fix[n] = -0.02
            else: zone_fix[n] = 0.0
        
        final_scores = {}
        for n in range(1, 50):
            base = base_scores.get(n, 0)
            gap = gap_scores.get(n, 0)
            recovery = recovery_scores.get(n, 0)
            zone = zone_fix.get(n, 0)
            final_scores[n] = base * w_base + gap * w_gap + recovery * w_rec + zone * w_zone
            if n in recent3: final_scores[n] *= 0.15
            elif n in recent5: final_scores[n] *= 0.35
        
        sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        preds = [n for n, _ in sorted_nums[:15]]
        actual = numbers[i]
        if actual in preds:
            hits += 1
    
    rate = hits / TEST_PERIODS * 100
    if rate > best_rate:
        best_rate = rate
        best_config = name
    print(f"  {name:35s}: {hits}/{TEST_PERIODS} = {rate:.1f}%")

print(f"\n最优配置: {best_config} = {best_rate:.1f}%")

# Compare: original PreciseTop15
print(f"\n=== 对比基准 ===")
predictor = PreciseTop15Predictor()
hits = 0
for i in range(start_idx, len(df)):
    lo = max(0, i - 25)
    train = numbers[lo:i]
    preds = predictor.predict(train)
    actual = numbers[i]
    if actual in preds:
        hits += 1
print(f"  原始PreciseTop15: {hits}/{TEST_PERIODS} = {hits/TEST_PERIODS*100:.1f}%")
