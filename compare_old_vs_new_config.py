# -*- coding: utf-8 -*-
"""
对比 smart_dynamic_300periods_detail.csv 方案 vs 当前最新方案
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

# 加载数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
numbers = df['number'].tolist()
dates = df['date'].tolist()

# 使用实际预测器
from top15_statistical_predictor import Top15StatisticalPredictor
predictor = Top15StatisticalPredictor()

test_periods = 300
test_start = len(numbers) - test_periods

print("=" * 80)
print("方案对比: smart_dynamic方案 vs 当前最新方案")
print("=" * 80)
print(f"数据: {len(numbers)}期, 测试: {test_periods}期")
print(f"时间范围: {dates[test_start]} ~ {dates[-1]}")
print()

# Fibonacci序列
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def get_fib_mult(losses, max_mult=10):
    if losses >= len(FIB):
        return min(FIB[-1], max_mult)
    return min(FIB[losses], max_mult)

def run_backtest(config_name, lookback, good_thresh, bad_thresh, boost_mult, reduce_mult, max_mult=10):
    """运行回测"""
    results = []
    consecutive_losses = 0
    recent_results = []
    
    for i in range(test_start, len(numbers)):
        history = numbers[:i]
        actual = numbers[i]
        date = dates[i]
        
        # 预测
        prediction = predictor.predict(history)
        hit = actual in prediction
        
        # 获取基础Fibonacci倍数
        base_mult = get_fib_mult(consecutive_losses, max_mult)
        
        # 动态调整 - 关键：先计算倍数，再更新历史
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
            profit = 32 * final_mult  # 净盈利 (47元奖励 - 15元投注 = 32元)
            consecutive_losses = 0
        else:
            profit = -bet
            consecutive_losses += 1
        
        results.append({
            'date': date,
            'number': actual,
            'hit': hit,
            'multiplier': final_mult,
            'bet': bet,
            'profit': profit,
            'is_10x': final_mult >= 10
        })
        
        # 更新历史（在计算后）
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
    
    hit_rate = sum(r['hit'] for r in results) / len(results) * 100
    times_10x = sum(1 for r in results if r['is_10x'])
    hits_10x = sum(1 for r in results if r['is_10x'] and r['hit'])
    
    return {
        'config_name': config_name,
        'roi': roi,
        'profit': total_profit,
        'drawdown': max_drawdown,
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'times_10x': times_10x,
        'hits_10x': hits_10x,
        'results': results
    }

# 配置1: smart_dynamic_300periods_detail.csv 的方案（旧版）
config_old = {
    'name': 'smart_dynamic旧版',
    'lookback': 12,
    'good_thresh': 0.35,
    'bad_thresh': 0.20,
    'boost_mult': 1.2,
    'reduce_mult': 0.8,
    'max_mult': 10
}

# 配置2: 当前最新方案（优化后）
config_new = {
    'name': '当前最新优化版',
    'lookback': 10,
    'good_thresh': 0.30,
    'bad_thresh': 0.20,
    'boost_mult': 1.2,
    'reduce_mult': 0.5,
    'max_mult': 10
}

print("【配置参数对比】")
print("-" * 60)
print(f"{'参数':<15} {'smart_dynamic旧版':<20} {'当前最新优化版':<20}")
print("-" * 60)
print(f"{'回看窗口':<15} {config_old['lookback']}期{'':<16} {config_new['lookback']}期")
print(f"{'增强阈值':<15} {config_old['good_thresh']*100:.0f}%{'':<17} {config_new['good_thresh']*100:.0f}%")
print(f"{'降低阈值':<15} {config_old['bad_thresh']*100:.0f}%{'':<17} {config_new['bad_thresh']*100:.0f}%")
print(f"{'增强倍数':<15} {config_old['boost_mult']}x{'':<17} {config_new['boost_mult']}x")
print(f"{'降低倍数':<15} {config_old['reduce_mult']}x{'':<17} {config_new['reduce_mult']}x")
print(f"{'最大倍数':<15} {config_old['max_mult']}x{'':<17} {config_new['max_mult']}x")
print()

# 运行回测
print("运行回测...")
result_old = run_backtest(
    config_old['name'],
    config_old['lookback'], config_old['good_thresh'], config_old['bad_thresh'],
    config_old['boost_mult'], config_old['reduce_mult'], config_old['max_mult']
)

result_new = run_backtest(
    config_new['name'],
    config_new['lookback'], config_new['good_thresh'], config_new['bad_thresh'],
    config_new['boost_mult'], config_new['reduce_mult'], config_new['max_mult']
)

print()
print("=" * 80)
print("【性能指标对比】")
print("=" * 80)
print()
print(f"{'指标':<15} {'smart_dynamic旧版':<20} {'当前最新优化版':<20} {'差异':<15}")
print("-" * 70)

# ROI
diff_roi = result_new['roi'] - result_old['roi']
diff_sign = '+' if diff_roi >= 0 else ''
print(f"{'ROI':<15} {result_old['roi']:.2f}%{'':<15} {result_new['roi']:.2f}%{'':<15} {diff_sign}{diff_roi:.2f}%")

# 净收益
diff_profit = result_new['profit'] - result_old['profit']
diff_sign = '+' if diff_profit >= 0 else ''
print(f"{'净收益':<15} {result_old['profit']:+.0f}元{'':<14} {result_new['profit']:+.0f}元{'':<14} {diff_sign}{diff_profit:.0f}元")

# 最大回撤
diff_dd = result_new['drawdown'] - result_old['drawdown']
diff_sign = '+' if diff_dd >= 0 else ''
print(f"{'最大回撤':<15} {result_old['drawdown']:.0f}元{'':<15} {result_new['drawdown']:.0f}元{'':<15} {diff_sign}{diff_dd:.0f}元")

# 总投注
diff_bet = result_new['total_bet'] - result_old['total_bet']
diff_sign = '+' if diff_bet >= 0 else ''
print(f"{'总投注':<15} {result_old['total_bet']:.0f}元{'':<14} {result_new['total_bet']:.0f}元{'':<14} {diff_sign}{diff_bet:.0f}元")

# 命中率
diff_hr = result_new['hit_rate'] - result_old['hit_rate']
diff_sign = '+' if diff_hr >= 0 else ''
print(f"{'命中率':<15} {result_old['hit_rate']:.1f}%{'':<15} {result_new['hit_rate']:.1f}%{'':<15} {diff_sign}{diff_hr:.1f}%")

# 10倍投注
print(f"{'10倍投注次数':<15} {result_old['times_10x']}次{'':<16} {result_new['times_10x']}次")
hit_rate_10x_old = result_old['hits_10x'] / result_old['times_10x'] * 100 if result_old['times_10x'] > 0 else 0
hit_rate_10x_new = result_new['hits_10x'] / result_new['times_10x'] * 100 if result_new['times_10x'] > 0 else 0
print(f"{'10倍命中次数':<15} {result_old['hits_10x']}次 ({hit_rate_10x_old:.1f}%){'':<8} {result_new['hits_10x']}次 ({hit_rate_10x_new:.1f}%)")

print()
print("=" * 80)
print("【结论】")
print("=" * 80)

# 判断哪个方案更优
score_old = 0
score_new = 0

# ROI权重最大
if result_new['roi'] > result_old['roi']:
    score_new += 3
    print(f"✅ ROI: 当前最新方案更优 ({result_new['roi']:.2f}% vs {result_old['roi']:.2f}%)")
else:
    score_old += 3
    print(f"✅ ROI: smart_dynamic旧版更优 ({result_old['roi']:.2f}% vs {result_new['roi']:.2f}%)")

# 收益
if result_new['profit'] > result_old['profit']:
    score_new += 2
    print(f"✅ 净收益: 当前最新方案更优 ({result_new['profit']:+.0f}元 vs {result_old['profit']:+.0f}元)")
else:
    score_old += 2
    print(f"✅ 净收益: smart_dynamic旧版更优 ({result_old['profit']:+.0f}元 vs {result_new['profit']:+.0f}元)")

# 回撤（越低越好）
if result_new['drawdown'] < result_old['drawdown']:
    score_new += 2
    print(f"✅ 最大回撤: 当前最新方案更优 ({result_new['drawdown']:.0f}元 vs {result_old['drawdown']:.0f}元)")
else:
    score_old += 2
    print(f"✅ 最大回撤: smart_dynamic旧版更优 ({result_old['drawdown']:.0f}元 vs {result_new['drawdown']:.0f}元)")

# 风险调整收益
risk_adj_old = result_old['roi'] / (result_old['drawdown'] / 1000 + 0.1)
risk_adj_new = result_new['roi'] / (result_new['drawdown'] / 1000 + 0.1)
if risk_adj_new > risk_adj_old:
    score_new += 1
    print(f"✅ 风险调整收益: 当前最新方案更优 ({risk_adj_new:.2f} vs {risk_adj_old:.2f})")
else:
    score_old += 1
    print(f"✅ 风险调整收益: smart_dynamic旧版更优 ({risk_adj_old:.2f} vs {risk_adj_new:.2f})")

print()
print(f"综合评分: smart_dynamic旧版 {score_old} 分 vs 当前最新方案 {score_new} 分")
print()

if score_new > score_old:
    print("🏆 【当前最新优化方案更优】")
    print(f"   ROI提升: {result_old['roi']:.2f}% → {result_new['roi']:.2f}% (+{diff_roi:.2f}%)")
    print(f"   回撤降低: {result_old['drawdown']:.0f}元 → {result_new['drawdown']:.0f}元 ({diff_dd:.0f}元)")
elif score_old > score_new:
    print("🏆 【smart_dynamic旧版方案更优】")
    print(f"   建议：考虑使用旧版配置参数")
else:
    print("🤝 【两方案表现相当】")
