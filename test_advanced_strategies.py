"""测试多种高级策略组合寻找50%命中率方案"""
import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

def strategy_time_decay(numbers_all, current_idx, k=15, alpha=0.05):
    """时间衰减频率：全历史，指数衰减"""
    scores = {}
    for n in range(1, 50):
        scores[n] = 0.0
    
    for j in range(current_idx):
        age = current_idx - j
        weight = np.exp(-alpha * age)
        scores[numbers_all[j]] += weight
    
    # 最近3期惩罚
    for n in set(numbers_all[max(0,current_idx-3):current_idx]):
        scores[n] *= 0.3
    
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:k]]

def strategy_zone_balanced(numbers_all, current_idx, k=15, window=25):
    """区域平衡 + 频率"""
    train = numbers_all[max(0, current_idx-window):current_idx]
    freq = Counter(train)
    
    # Zone allocation: 3-3-3-3-3
    zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
    per_zone = k // len(zones)  # 3 per zone
    
    result = []
    for lo, hi in zones:
        zone_scores = {}
        for n in range(lo, hi+1):
            f = freq.get(n, 0)
            # Time since last appearance
            gap = window + 1
            for j in range(len(train)-1, -1, -1):
                if train[j] == n:
                    gap = len(train) - j
                    break
            
            # Sweet spot: gap 5-15
            if 5 <= gap <= 15:
                gap_score = 2.0
            elif 15 < gap <= 25:
                gap_score = 1.5
            elif gap > 25:
                gap_score = 0.8
            else:
                gap_score = 0.3
            
            zone_scores[n] = (f * 0.4 + gap_score) * (0.3 if n in set(train[-3:]) else 1.0)
        
        sorted_zone = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
        result.extend([n for n, _ in sorted_zone[:per_zone]])
    
    return result[:k]

def strategy_multi_window_hybrid(numbers_all, current_idx, k=15):
    """多窗口混合: 短窗口(热号) + 长窗口(温号) + 全历史(冷号恢复)"""
    scores = {}
    for n in range(1, 50):
        scores[n] = 0.0
    
    # 短窗口 (10期) - 热号
    short = numbers_all[max(0, current_idx-10):current_idx]
    freq_short = Counter(short)
    for n, f in freq_short.items():
        scores[n] += f * 1.5
    
    # 中窗口 (25期) - 温号
    medium = numbers_all[max(0, current_idx-25):current_idx]
    freq_med = Counter(medium)
    for n, f in freq_med.items():
        scores[n] += f * 0.8
    
    # 长窗口 (100期) - 冷号恢复
    long_win = numbers_all[max(0, current_idx-100):current_idx]
    freq_long = Counter(long_win)
    for n in range(1, 50):
        if n not in freq_med:  # 25期内未出现
            long_f = freq_long.get(n, 0)
            if long_f > 0:
                scores[n] += long_f * 0.6  # 100期内出现过的冷号
    
    # 间隔分析
    for n in range(1, 50):
        gap = 200
        for j in range(current_idx-1, max(0, current_idx-100), -1):
            if numbers_all[j] == n:
                gap = current_idx - j
                break
        if 6 <= gap <= 20:
            scores[n] *= 1.5
        elif gap <= 3:
            scores[n] *= 0.2
    
    # 最近3期强惩罚
    for n in set(numbers_all[max(0,current_idx-3):current_idx]):
        scores[n] *= 0.2
    
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:k]]

def strategy_gap_optimized(numbers_all, current_idx, k=15):
    """间隔优化策略: 基于间隔分布的最优选择"""
    scores = {}
    
    for n in range(1, 50):
        # 计算间隔
        gap = 200
        for j in range(current_idx-1, max(0, current_idx-200), -1):
            if numbers_all[j] == n:
                gap = current_idx - j
                break
        
        if gap <= 3:
            scores[n] = 0.0
        elif gap <= 5:
            scores[n] = 0.1
        elif gap <= 10:
            scores[n] = 3.0
        elif gap <= 15:
            scores[n] = 3.0
        elif gap <= 20:
            scores[n] = 2.5
        elif gap <= 30:
            scores[n] = 1.8
        elif gap <= 50:
            scores[n] = 0.5
        else:
            scores[n] = 0.3
        
        # 频率加成 (25期)
        train = numbers_all[max(0, current_idx-25):current_idx]
        freq = Counter(train)
        if freq.get(n, 0) > 0:
            scores[n] += freq[n] * 0.3
    
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:k]]

def strategy_hot_cold_split(numbers_all, current_idx, k=15, hot_slots=10, cold_slots=5):
    """热冷分离策略: 分别选热号和冷号"""
    train25 = numbers_all[max(0, current_idx-25):current_idx]
    train100 = numbers_all[max(0, current_idx-100):current_idx]
    freq25 = Counter(train25)
    freq100 = Counter(train100)
    
    recent3 = set(numbers_all[max(0,current_idx-3):current_idx])
    
    # 热号: 25期内出现过, 间隔5-20
    hot_scores = {}
    for n in range(1, 50):
        if freq25.get(n, 0) > 0 and n not in recent3:
            gap = 25
            for j in range(len(train25)-1, -1, -1):
                if train25[j] == n:
                    gap = len(train25) - j
                    break
            if 4 <= gap <= 20:
                hot_scores[n] = freq25[n] * 1.0 + (1.0 if 6 <= gap <= 15 else 0.3)
    
    hot_sorted = sorted(hot_scores.items(), key=lambda x: x[1], reverse=True)
    hot_picks = [n for n, _ in hot_sorted[:hot_slots]]
    
    # 冷号: 25期内未出现, 但100期内出现过, 按频率和区域
    cold_scores = {}
    for n in range(1, 50):
        if n not in set(train25) and n not in recent3:
            base = freq100.get(n, 0) * 0.5
            # Zone balance for cold: favor underrepresented zone 31-40
            if 31 <= n <= 40:
                base *= 1.5
            cold_scores[n] = base + 0.1
    
    cold_sorted = sorted(cold_scores.items(), key=lambda x: x[1], reverse=True)
    cold_picks = [n for n, _ in cold_sorted[:cold_slots]]
    
    return (hot_picks + cold_picks)[:k]

def strategy_ensemble_rank_fusion(numbers_all, current_idx, k=15):
    """排名融合: 多策略排名加权"""
    strategies = [
        strategy_gap_optimized(numbers_all, current_idx, 25),
        strategy_multi_window_hybrid(numbers_all, current_idx, 25),
        strategy_time_decay(numbers_all, current_idx, 25, alpha=0.03),
    ]
    
    scores = {}
    for strat_result in strategies:
        for rank, n in enumerate(strat_result):
            score = 1.0 - rank / len(strat_result)
            scores[n] = scores.get(n, 0) + score
    
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:k]]

# Test all strategies
strategies = {
    'time_decay_0.03': lambda nums, idx: strategy_time_decay(nums, idx, 15, 0.03),
    'time_decay_0.05': lambda nums, idx: strategy_time_decay(nums, idx, 15, 0.05),
    'time_decay_0.08': lambda nums, idx: strategy_time_decay(nums, idx, 15, 0.08),
    'time_decay_0.10': lambda nums, idx: strategy_time_decay(nums, idx, 15, 0.10),
    'zone_balanced': lambda nums, idx: strategy_zone_balanced(nums, idx, 15),
    'multi_window': lambda nums, idx: strategy_multi_window_hybrid(nums, idx, 15),
    'gap_optimized': lambda nums, idx: strategy_gap_optimized(nums, idx, 15),
    'hot10_cold5': lambda nums, idx: strategy_hot_cold_split(nums, idx, 15, 10, 5),
    'hot8_cold7': lambda nums, idx: strategy_hot_cold_split(nums, idx, 15, 8, 7),
    'hot12_cold3': lambda nums, idx: strategy_hot_cold_split(nums, idx, 15, 12, 3),
    'rank_fusion': lambda nums, idx: strategy_ensemble_rank_fusion(nums, idx, 15),
}

print("=== 多策略300期命中率测试 ===\n")
results = {}
for name, func in strategies.items():
    hits = 0
    for i in range(start_idx, len(df)):
        preds = func(numbers, i)
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    results[name] = rate
    marker = " ⭐" if rate >= 45 else (" ✅" if rate >= 40 else "")
    print(f"  {name:25s}: {hits}/{TEST_PERIODS} = {rate:.1f}%{marker}")

# Find best and test variations
best_name = max(results, key=results.get)
print(f"\n最优策略: {best_name} = {results[best_name]:.1f}%")

# Test gap_optimized with different k values
print(f"\n=== 最优策略扩展不同k值 ===")
for k in [15, 16, 17, 18, 19, 20]:
    hits = 0
    for i in range(start_idx, len(df)):
        preds = strategy_gap_optimized(numbers, i, k)
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    print(f"  Top{k}: {hits}/{TEST_PERIODS} = {rate:.1f}%")

# Deep analysis of gap_optimized misses
print(f"\n=== gap_optimized 失败分析 ===")
miss_gaps = []
hit_gaps = []
for i in range(start_idx, len(df)):
    preds = strategy_gap_optimized(numbers, i, 15)
    actual = numbers[i]
    gap = 200
    for j in range(i-1, max(0, i-200), -1):
        if numbers[j] == actual:
            gap = i - j
            break
    if actual in preds:
        hit_gaps.append(gap)
    else:
        miss_gaps.append(gap)

print(f"命中时平均间隔: {np.mean(hit_gaps):.1f}")
print(f"未中时平均间隔: {np.mean(miss_gaps):.1f}")
print(f"未中时间隔分布:")
gap_bins = [(0,5), (6,10), (11,15), (16,20), (21,30), (31,50), (51,100), (101,200)]
for lo, hi in gap_bins:
    count = sum(1 for g in miss_gaps if lo <= g <= hi)
    if count > 0:
        print(f"  间隔{lo}-{hi}: {count}次")
