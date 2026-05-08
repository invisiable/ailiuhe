"""分析TOP15预测的失败模式和改进机会"""
import pandas as pd
import numpy as np
from collections import Counter
from precise_top15_predictor import PreciseTop15Predictor
from top15_statistical_predictor import Top15StatisticalPredictor
from top15_zodiac_enhanced_v2 import Top15ZodiacEnhancedV2
from ensemble_top15_predictor import EnsembleTop15Predictor
from top15_predictor import Top15Predictor

df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].values

TRAIN_WINDOW = 25
TEST_PERIODS = 300
start_idx = len(df) - TEST_PERIODS

# Collect predictions from all models for each period
all_models = {
    'precise': PreciseTop15Predictor(),
    'statistical': Top15StatisticalPredictor(),
    'zodiac': Top15ZodiacEnhancedV2(),
    'ensemble': EnsembleTop15Predictor(),
    'base': Top15Predictor(),
}

period_data = []
for i in range(start_idx, len(df)):
    lo = max(0, i - TRAIN_WINDOW)
    train = numbers[lo:i]
    actual = numbers[i]
    
    all_preds = {}
    all_nums = Counter()
    for name, pred in all_models.items():
        try:
            p15 = pred.predict(train)
        except:
            p15 = pred.predict_top20(train)[:15]
        all_preds[name] = p15
        for n in p15:
            all_nums[n] += 1
    
    # Also get top20 predictions
    for name in ['statistical', 'zodiac', 'ensemble']:
        try:
            p20 = all_models[name].predict_top20(train)
            for n in p20:
                all_nums[n] += 0.5  # top20 extras get half weight
        except:
            pass
    
    # Count how many models predicted each number
    model_votes = {}
    for name, preds in all_preds.items():
        for n in preds:
            model_votes[n] = model_votes.get(n, 0) + 1
    
    # Test: union vote approach - pick top 15 by vote count
    sorted_by_votes = sorted(all_nums.items(), key=lambda x: x[1], reverse=True)
    union_top15 = [n for n, _ in sorted_by_votes[:15]]
    union_top20 = [n for n, _ in sorted_by_votes[:20]]
    
    period_data.append({
        'actual': actual,
        'precise_hit': actual in all_preds['precise'],
        'union15_hit': actual in union_top15,
        'union20_hit': actual in union_top20,
        'actual_votes': model_votes.get(actual, 0),
        'actual_all_score': all_nums.get(actual, 0),
    })

# Summary
precise_hits = sum(1 for p in period_data if p['precise_hit'])
union15_hits = sum(1 for p in period_data if p['union15_hit'])
union20_hits = sum(1 for p in period_data if p['union20_hit'])

print(f"=== 300期命中率对比 ===")
print(f"PreciseTop15 (当前): {precise_hits}/300 = {precise_hits/300*100:.1f}%")
print(f"多模型投票Top15:     {union15_hits}/300 = {union15_hits/300*100:.1f}%")
print(f"多模型投票Top20:     {union20_hits}/300 = {union20_hits/300*100:.1f}%")

# Analyze: when actual number gets 0 votes from all models
vote_dist = Counter(p['actual_votes'] for p in period_data)
print(f"\n=== 实际号码的模型投票分布 ===")
for v in sorted(vote_dist.keys()):
    count = vote_dist[v]
    print(f"  {v}个模型预测到: {count}次 ({count/300*100:.1f}%)")

# Zone analysis - which zones are hardest to predict
zones = {'1-10': (1,10), '11-20': (11,20), '21-30': (21,30), '31-40': (31,40), '41-49': (41,49)}
print(f"\n=== 各区域命中率 (PreciseTop15) ===")
for zname, (lo, hi) in zones.items():
    zone_periods = [p for p in period_data if lo <= p['actual'] <= hi]
    if zone_periods:
        zhits = sum(1 for p in zone_periods if p['precise_hit'])
        print(f"  {zname}: {zhits}/{len(zone_periods)} = {zhits/len(zone_periods)*100:.1f}%  (出现{len(zone_periods)}次)")

# Analyze the 5 most recent numbers pattern
print(f"\n=== 最近5期惩罚分析 ===")
# How often does the actual number appear in the last 5 periods?
recent_in5 = 0
for i in range(start_idx, len(df)):
    actual = numbers[i]
    recent5 = set(numbers[max(0,i-5):i])
    if actual in recent5:
        recent_in5 += 1
print(f"  实际号码在最近5期中出现过: {recent_in5}/300 = {recent_in5/300*100:.1f}%")

# Gap analysis - what gap values are most common for actual numbers
print(f"\n=== 实际号码的间隔分布 ===")
gap_hits = {'precise_hit': Counter(), 'precise_miss': Counter()}
for i in range(start_idx, len(df)):
    actual = numbers[i]
    # Find gap
    gap = None
    for j in range(i-1, max(0, i-60), -1):
        if numbers[j] == actual:
            gap = i - j
            break
    if gap is None:
        gap = 60  # not found in last 60
    
    key = 'precise_hit' if period_data[i-start_idx]['precise_hit'] else 'precise_miss'
    if gap <= 5:
        gap_hits[key]['0-5'] += 1
    elif gap <= 10:
        gap_hits[key]['6-10'] += 1
    elif gap <= 15:
        gap_hits[key]['11-15'] += 1
    elif gap <= 20:
        gap_hits[key]['16-20'] += 1
    elif gap <= 30:
        gap_hits[key]['21-30'] += 1
    else:
        gap_hits[key]['30+'] += 1

for gap_range in ['0-5', '6-10', '11-15', '16-20', '21-30', '30+']:
    h = gap_hits['precise_hit'].get(gap_range, 0)
    m = gap_hits['precise_miss'].get(gap_range, 0)
    total = h + m
    rate = h/total*100 if total > 0 else 0
    print(f"  间隔{gap_range:>5s}: 命中{h:3d} 未中{m:3d} 命中率{rate:.1f}% (共{total}次)")

# Test: what if we use different number of predictions?
print(f"\n=== PreciseTop15 扩展到不同数量 ===")
predictor = PreciseTop15Predictor()
for k in [15, 17, 18, 20, 22, 25]:
    hits = 0
    for i in range(start_idx, len(df)):
        lo = max(0, i - TRAIN_WINDOW)
        train = numbers[lo:i]
        pattern = predictor.analyze_pattern(train)
        if len(train) >= 50:
            pattern['recent_50'] = train[-50:]
        else:
            pattern['recent_50'] = train
        
        base_k = max(22, k+5)
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
    print(f"  Top{k:2d}: 命中 {hits}/300 = {rate:.1f}%  (理论随机: {k/49*100:.1f}%)")

# Test: Markov chain approach
print(f"\n=== Markov链转移概率方法 ===")
for window in [25, 50, 100]:
    hits = 0
    for i in range(start_idx, len(df)):
        lo = max(0, i - window)
        train = numbers[lo:i]
        
        if len(train) < 3:
            continue
        
        # Build transition counts from last number
        last_num = train[-1]
        transition = Counter()
        for j in range(len(train)-1):
            transition[train[j+1]] += 1
        
        # Build conditional transition from last number
        cond_transition = Counter()
        for j in range(len(train)-1):
            if train[j] == last_num:
                cond_transition[train[j+1]] += 1
        
        # Combine: general frequency + conditional + gap
        scores = {}
        for n in range(1, 50):
            scores[n] = transition.get(n, 0) * 0.3
            scores[n] += cond_transition.get(n, 0) * 2.0
            # Penalize recent
            if n in set(train[-3:]):
                scores[n] *= 0.3
        
        top15 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        preds = [n for n, _ in top15]
        actual = numbers[i]
        if actual in preds:
            hits += 1
    rate = hits / TEST_PERIODS * 100
    print(f"  窗口{window:3d}: 命中 {hits}/300 = {rate:.1f}%")
