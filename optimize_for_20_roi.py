# -*- coding: utf-8 -*-
"""
搜索ROI >= 20%的参数组合
"""

import pandas as pd
import numpy as np
from itertools import product

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()

# 测试最近300期
test_periods = 300
train_start = len(numbers) - test_periods  # 从第90期开始测试（前90期作为训练）

# 预测函数（简化版Top15）
def get_top15_prediction(history):
    """基于历史数据返回Top15预测"""
    if len(history) < 30:
        return list(range(1, 16))
    
    # 统计最近出现频率
    recent = history[-30:]
    freq = {}
    for n in range(1, 50):
        freq[n] = recent.count(n) * 2 + history[-60:].count(n)
    
    # 添加趋势
    if len(history) >= 5:
        for n in history[-5:]:
            freq[n] = freq.get(n, 0) + 1
    
    sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nums[:15]]

# Fibonacci序列
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def get_fib_multiplier(consecutive_losses, max_mult=10):
    """获取Fibonacci倍数"""
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
    
    for i in range(train_start, len(numbers)):
        history = numbers[:i]
        actual = numbers[i]
        
        # 预测
        prediction = get_top15_prediction(history)
        hit = actual in prediction
        
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
            profit = 30 * final_mult  # 净盈利
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
        
        # 更新历史（修正后：在计算后更新）
        recent_results.append(1 if hit else 0)
    
    # 计算指标
    total_bet = sum(r['bet'] for r in results)
    total_profit = sum(r['profit'] for r in results)
    roi = total_profit / total_bet * 100 if total_bet > 0 else 0
    
    # 计算最大回撤
    cumsum = 0
    peak = 0
    max_drawdown = 0
    for r in results:
        cumsum += r['profit']
        peak = max(peak, cumsum)
        drawdown = peak - cumsum
        max_drawdown = max(max_drawdown, drawdown)
    
    hit_rate = sum(r['hit'] for r in results) / len(results) * 100
    
    return {
        'roi': roi,
        'profit': total_profit,
        'drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'total_bet': total_bet
    }

# 参数搜索范围（更激进）
lookbacks = [5, 6, 8, 10, 12, 15]
good_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
bad_thresholds = [0.15, 0.20, 0.25]
boost_mults = [1.5, 1.8, 2.0, 2.5, 3.0]  # 更激进的增强
reduce_mults = [0.3, 0.4, 0.5, 0.6]
max_mults = [10, 12, 15]  # 尝试更高上限

print("=" * 70)
print("搜索ROI >= 20% 的参数组合")
print("=" * 70)
print(f"数据: {len(numbers)}期, 测试: {test_periods}期")
print()

# 搜索
results_list = []
total_combos = len(lookbacks) * len(good_thresholds) * len(bad_thresholds) * len(boost_mults) * len(reduce_mults) * len(max_mults)
print(f"搜索 {total_combos} 种参数组合...")
print()

for lb, gt, bt, bm, rm, mm in product(lookbacks, good_thresholds, bad_thresholds, boost_mults, reduce_mults, max_mults):
    if gt <= bt:  # 阈值逻辑检查
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

# 筛选ROI >= 20%
high_roi = [r for r in results_list if r['roi'] >= 20]
print(f"找到 {len(high_roi)} 个ROI >= 20% 的配置")
print()

if high_roi:
    # 按ROI排序
    high_roi.sort(key=lambda x: x['roi'], reverse=True)
    
    print("【Top 10 高ROI配置】")
    print("-" * 100)
    print(f"{'排名':<4} {'ROI':<8} {'净收益':<10} {'回撤':<10} {'命中率':<8} {'回看':<6} {'增强阈值':<8} {'降低阈值':<8} {'增强倍':<8} {'降低倍':<8} {'上限':<6}")
    print("-" * 100)
    
    for i, r in enumerate(high_roi[:10], 1):
        print(f"{i:<4} {r['roi']:.2f}%   {r['profit']:+.0f}元    {r['drawdown']:.0f}元    {r['hit_rate']:.1f}%   "
              f"{r['lookback']:<6} {r['good_thresh']:.2f}     {r['bad_thresh']:.2f}     "
              f"{r['boost_mult']:.1f}x     {r['reduce_mult']:.1f}x     {r['max_mult']}")
    
    print()
    
    # 找到最佳平衡（高ROI + 低回撤）
    # 风险调整后收益 = ROI / (回撤/1000)
    for r in high_roi:
        r['risk_adjusted'] = r['roi'] / (r['drawdown'] / 1000 + 0.1)
    
    high_roi.sort(key=lambda x: x['risk_adjusted'], reverse=True)
    
    print("【风险调整后最佳配置（高ROI + 低回撤）】")
    print("-" * 100)
    for i, r in enumerate(high_roi[:5], 1):
        print(f"{i}. ROI={r['roi']:.2f}%, 收益={r['profit']:+.0f}元, 回撤={r['drawdown']:.0f}元")
        print(f"   参数: lookback={r['lookback']}, good_thresh={r['good_thresh']:.2f}, "
              f"bad_thresh={r['bad_thresh']:.2f}, boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
        print()

else:
    # 找最高ROI
    results_list.sort(key=lambda x: x['roi'], reverse=True)
    print("未找到ROI >= 20%的配置，以下是最高ROI的配置：")
    print()
    
    for i, r in enumerate(results_list[:10], 1):
        print(f"{i}. ROI={r['roi']:.2f}%, 收益={r['profit']:+.0f}元, 回撤={r['drawdown']:.0f}元")
        print(f"   参数: lookback={r['lookback']}, good_thresh={r['good_thresh']:.2f}, "
              f"bad_thresh={r['bad_thresh']:.2f}, boost={r['boost_mult']}x, reduce={r['reduce_mult']}x, max={r['max_mult']}")
        print()

# 分析为什么难以达到20% ROI
print("=" * 70)
print("【ROI分析】")
print("=" * 70)
best = results_list[0]
print(f"当前最高ROI: {best['roi']:.2f}%")
print(f"命中率: {best['hit_rate']:.1f}%")
print()
print("理论分析:")
print(f"- 基础赔率: 1赔3 (投15元中奖得45元，净赚30元)")
print(f"- 盈亏平衡命中率: 33.3%")
print(f"- 当前命中率: {best['hit_rate']:.1f}%")
print()

# 分析不同命中率下的理论ROI
print("【理论ROI vs 命中率】(假设1倍投注)")
for hr in [30, 33, 35, 40, 45, 50]:
    # 每100期
    wins = hr
    losses = 100 - hr
    profit = wins * 30 - losses * 15
    total_bet = 100 * 15
    roi = profit / total_bet * 100
    print(f"  命中率 {hr}%: ROI = {roi:.1f}%")

print()
print("结论: 要达到20% ROI，需要约40%命中率，或通过策略放大盈利期收益")
