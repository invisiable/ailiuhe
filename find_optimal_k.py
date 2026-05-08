"""找到达到50%命中率的最优方案"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

# 1. 精确搜索 PreciseTop15 在 k=18~25 的命中率
print("=== PreciseTop15 精确搜索 (目标50%) ===\n")
predictor = PreciseTop15Predictor()
for k in range(15, 26):
    hits = 0
    for i in range(start_idx, len(df)):
        lo = max(0, i - 25)
        train = numbers[lo:i]
        
        pattern = predictor.analyze_pattern(train)
        pattern['recent_50'] = train
        
        base_k = max(22, k + 5)
        methods = [
            (predictor.method_precision_frequency(pattern, base_k), 0.40),
            (predictor.method_zone_dynamic(pattern, base_k), 0.25),
            (predictor.method_gap_analysis(pattern, base_k), 0.20),
            (predictor.method_avoid_recent_misses(pattern, base_k), 0.15)
        ]
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        preds = [num for num, _ in final[:k]]
        actual = numbers[i]
        if actual in preds:
            hits += 1
    
    rate = hits / TEST_PERIODS * 100
    random_rate = k / 49 * 100
    skill = rate - random_rate
    marker = " ✅ 达标!" if rate >= 50 else ""
    print(f"  Top{k:2d}: {hits}/{TEST_PERIODS} = {rate:.1f}%  (随机{random_rate:.1f}%, 技巧+{skill:.1f}%){marker}")

# 2. 测试不同窗口对 Top20 和 Top22 的影响
print(f"\n=== 不同窗口期对 Top20/22 的影响 ===")
for k in [20, 22]:
    best_window = 25
    best_rate = 0
    for window in [15, 20, 25, 30, 40, 50]:
        predictor = PreciseTop15Predictor()
        hits = 0
        for i in range(start_idx, len(df)):
            lo = max(0, i - window)
            train = numbers[lo:i]
            
            pattern = predictor.analyze_pattern(train)
            pattern['recent_50'] = train if len(train) <= 50 else train[-50:]
            
            base_k = k + 8
            methods = [
                (predictor.method_precision_frequency(pattern, base_k), 0.40),
                (predictor.method_zone_dynamic(pattern, base_k), 0.25),
                (predictor.method_gap_analysis(pattern, base_k), 0.20),
                (predictor.method_avoid_recent_misses(pattern, base_k), 0.15)
            ]
            scores = {}
            for candidates, weight in methods:
                for rank, num in enumerate(candidates):
                    score = weight * (1.0 - rank / len(candidates))
                    scores[num] = scores.get(num, 0) + score
            
            final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            preds = [num for num, _ in final[:k]]
            actual = numbers[i]
            if actual in preds:
                hits += 1
        
        rate = hits / TEST_PERIODS * 100
        if rate > best_rate:
            best_rate = rate
            best_window = window
        print(f"  Top{k} 窗口{window:2d}: {hits}/300 = {rate:.1f}%")
    print(f"  → Top{k} 最优窗口: {best_window}期 = {best_rate:.1f}%\n")

# 3. 经济性分析: Top15 vs Top20 vs Top22 的投注效益
print(f"\n=== 投注经济性对比 (base_bet=15, win=47) ===")
base_bet = 15
win_reward = 47
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
max_mul = 10

for k in [15, 18, 20, 22]:
    predictor = PreciseTop15Predictor()
    fib_idx = 0
    total_bet = 0
    total_win = 0
    balance = 0
    min_balance = 0
    
    for i in range(start_idx, len(df)):
        lo = max(0, i - 25)
        train = numbers[lo:i]
        
        pattern = predictor.analyze_pattern(train)
        pattern['recent_50'] = train
        base_k = max(22, k + 5)
        methods = [
            (predictor.method_precision_frequency(pattern, base_k), 0.40),
            (predictor.method_zone_dynamic(pattern, base_k), 0.25),
            (predictor.method_gap_analysis(pattern, base_k), 0.20),
            (predictor.method_avoid_recent_misses(pattern, base_k), 0.15)
        ]
        scores = {}
        for candidates, weight in methods:
            for rank, num in enumerate(candidates):
                score = weight * (1.0 - rank / len(candidates))
                scores[num] = scores.get(num, 0) + score
        final = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        preds = [num for num, _ in final[:k]]
        actual = numbers[i]
        hit = actual in preds
        
        mul = min(fib[min(fib_idx, len(fib)-1)], max_mul)
        bet = base_bet * mul
        total_bet += bet
        
        if hit:
            win = win_reward * mul
            total_win += win
            balance += (win - bet)
            fib_idx = 0
        else:
            balance -= bet
            fib_idx += 1
        
        if balance < min_balance:
            min_balance = balance
    
    hit_rate = sum(1 for i2 in range(start_idx, len(df)) 
                   if numbers[i2] in [n for n, _ in sorted(
                       {n: 0 for n in range(1,50)}.items())[:k]]) / TEST_PERIODS * 100
    
    roi = (total_win - total_bet) / total_bet * 100 if total_bet > 0 else 0
    print(f"\n  Top{k}:")
    print(f"    净利润: {balance:.0f}元")
    print(f"    总投入: {total_bet:.0f}元")
    print(f"    ROI: {roi:.1f}%")
    print(f"    最大回撤: {abs(min_balance):.0f}元")
