# -*- coding: utf-8 -*-
"""
使用实际预测器搜索ROI >= 20%的参数组合
"""

import pandas as pd
import numpy as np
from itertools import product
import sys
sys.path.insert(0, '.')

# 使用实际的预测器
from top15_statistical_predictor import Top15StatisticalPredictor

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
dates = df['date'].tolist()

# 测试最近300期
test_periods = 300
test_start = len(numbers) - test_periods

print("=" * 70)
print("使用实际预测器搜索ROI >= 20% 的参数组合")
print("=" * 70)
print(f"数据: {len(numbers)}期, 测试: {test_periods}期")
print(f"时间范围: {dates[test_start]} ~ {dates[-1]}")
print()

# 预先计算所有期的预测结果
print("预计算所有期预测结果...")
predictor = Top15StatisticalPredictor()
predictions = []
hits = []

for i in range(test_start, len(numbers)):
    history = numbers[:i]
    actual = numbers[i]
    pred = predictor.predict(history)
    hit = actual in pred
    predictions.append(pred)
    hits.append(hit)

base_hit_rate = sum(hits) / len(hits) * 100
print(f"基础命中率: {base_hit_rate:.1f}% ({sum(hits)}/{len(hits)})")
print()

# Fibonacci序列
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def get_fib_multiplier(consecutive_losses, max_mult=10):
    if consecutive_losses >= len(FIB):
        mult = FIB[-1]
    else:
        mult = FIB[consecutive_losses]
    return min(mult, max_mult)

def run_backtest(lookback, good_thresh, bad_thresh, boost_mult, reduce_mult, max_mult=10):
    """运行回测"""
    results = []
    consecutive_losses = 0
    recent_results = []
    
    for idx, hit in enumerate(hits):
        # 获取基础Fibonacci倍数
        base_mult = get_fib_multiplier(consecutive_losses, max_mult)
        
        # 动态调整
        if len(recent_results) >= lookback:
            recent_window = recent_results[-lookback:]
            recent_hit_rate = sum(recent_window) / len(recent_window)
            
            if recent_hit_rate >= good_thresh:
                final_mult = min(base_mult * boost_mult, max_mult)
            elif recent_hit_rate <= bad_thresh:
                final_mult = max(base_mult * reduce_mult, 1.0)
            else:
                final_mult = base_mult
        else:
            final_mult = base_mult
        
        # 计算盈亏
        bet = 15 * final_mult
        if hit:
            profit = 30 * final_mult
            consecutive_losses = 0
        else:
            profit = -bet
            consecutive_losses += 1
        
        results.append({
            'hit': hit,
            'multiplier': final_mult,
            'bet': bet,
            'profit': profit
        })
        
        recent_results.append(1 if hit else 0)
    
    # 计算指标
    total_bet = sum(r['bet'] for r in results)
    total_profit = sum(r['profit'] for r in results)
    roi = total_profit / total_bet * 100 if total_bet > 0 else 0
    
    # 最大回撤
    cumsum = 0
    peak = 0
    max_drawdown = 0
    for r in results:
        cumsum += r['profit']
        peak = max(peak, cumsum)
        drawdown = peak - cumsum
        max_drawdown = max(max_drawdown, drawdown)
    
    return {
        'roi': roi,
        'profit': total_profit,
        'drawdown': max_drawdown,
        'total_bet': total_bet
    }

# 搜索范围
lookbacks = [5, 6, 8, 10, 12, 15, 20]
good_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
bad_thresholds = [0.10, 0.15, 0.20, 0.25]
boost_mults = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
reduce_mults = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
max_mults = [10, 12, 15, 20]

total_combos = len(lookbacks) * len(good_thresholds) * len(bad_thresholds) * len(boost_mults) * len(reduce_mults) * len(max_mults)
print(f"搜索 {total_combos} 种参数组合...")

results_list = []
for lb, gt, bt, bm, rm, mm in product(lookbacks, good_thresholds, bad_thresholds, boost_mults, reduce_mults, max_mults):
    if gt <= bt:
        continue
    
    result = run_backtest(lb, gt, bt, bm, rm, mm)
    result.update({
        'lookback': lb,
        'good_thresh': gt,
        'bad_thresh': bt,
        'boost_mult': bm,
        'reduce_mult': rm,
        'max_mult': mm
    })
    results_list.append(result)

print(f"有效组合: {len(results_list)}")
print()

# 筛选ROI >= 20%
high_roi = [r for r in results_list if r['roi'] >= 20]
print(f"找到 {len(high_roi)} 个ROI >= 20% 的配置")
print()

if high_roi:
    high_roi.sort(key=lambda x: x['roi'], reverse=True)
    
    print("【Top 15 高ROI配置】")
    print("-" * 110)
    print(f"{'排名':<4} {'ROI':<10} {'净收益':<12} {'回撤':<10} {'回看':<6} {'增强阈值':<10} {'降低阈值':<10} {'增强倍':<8} {'降低倍':<8} {'上限':<6}")
    print("-" * 110)
    
    for i, r in enumerate(high_roi[:15], 1):
        print(f"{i:<4} {r['roi']:.2f}%     {r['profit']:+.0f}元      {r['drawdown']:.0f}元     "
              f"{r['lookback']:<6} {r['good_thresh']:.2f}       {r['bad_thresh']:.2f}       "
              f"{r['boost_mult']:.1f}x     {r['reduce_mult']:.1f}x     {r['max_mult']}")
    
    # 风险调整
    print()
    print("【风险调整后最佳配置（ROI/回撤比）】")
    for r in high_roi:
        r['risk_score'] = r['roi'] / (r['drawdown'] / 1000 + 0.1)
    high_roi.sort(key=lambda x: x['risk_score'], reverse=True)
    
    print("-" * 80)
    for i, r in enumerate(high_roi[:5], 1):
        print(f"{i}. ROI={r['roi']:.2f}%, 收益={r['profit']:+.0f}元, 回撤={r['drawdown']:.0f}元")
        print(f"   参数: lookback={r['lookback']}, good={r['good_thresh']:.2f}, bad={r['bad_thresh']:.2f}, "
              f"boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
        print()

else:
    # 显示最高ROI
    results_list.sort(key=lambda x: x['roi'], reverse=True)
    
    print("未找到ROI >= 20%的配置")
    print()
    print("【最高ROI配置 Top 20】")
    print("-" * 110)
    print(f"{'排名':<4} {'ROI':<10} {'净收益':<12} {'回撤':<10} {'回看':<6} {'增强阈值':<10} {'降低阈值':<10} {'增强倍':<8} {'降低倍':<8} {'上限':<6}")
    print("-" * 110)
    
    for i, r in enumerate(results_list[:20], 1):
        print(f"{i:<4} {r['roi']:.2f}%     {r['profit']:+.0f}元      {r['drawdown']:.0f}元     "
              f"{r['lookback']:<6} {r['good_thresh']:.2f}       {r['bad_thresh']:.2f}       "
              f"{r['boost_mult']:.1f}x     {r['reduce_mult']:.1f}x     {r['max_mult']}")

# 分析
print()
print("=" * 70)
print("【ROI理论分析】")
print("=" * 70)
print(f"基础命中率: {base_hit_rate:.1f}%")
print(f"盈亏平衡点: 33.3%")
print()
print("理论ROI计算 (1倍投注):")
for hr in [30, 33, 35, 37, 40, 45, 50]:
    profit_per_100 = hr * 30 - (100 - hr) * 15
    roi = profit_per_100 / (100 * 15) * 100
    print(f"  命中率 {hr}%: 理论ROI = {roi:.1f}%")

print()
print("结论:")
if base_hit_rate < 33.3:
    needed = (20 * 1.5 + 100) / 3  # 反算需要的命中率
    print(f"  当前命中率{base_hit_rate:.1f}%低于盈亏平衡点33.3%")
    print(f"  要达到20% ROI，理论上需要40%命中率")
    print(f"  策略优化空间有限，核心是提升预测准确率")
else:
    print(f"  当前命中率{base_hit_rate:.1f}%高于盈亏平衡点")
    print(f"  通过策略优化可进一步提升ROI")
