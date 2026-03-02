#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""详细分析2026/2/26这一期的投注数据"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# 读取两个策略的CSV
base_df = pd.read_csv('validate_optimal_smart_base_300periods.csv')
pause_df = pd.read_csv('validate_optimal_smart_pause_300periods.csv')

# Fib序列
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

print("=" * 100)
print("2026/2/26投注详细分析")
print("=" * 100)

# 基础策略
print("\n【基础策略】2026/2/23 ~ 2026/2/28")
print("-" * 100)
base_sub = base_df[(base_df['period'] >= 294) & (base_df['period'] <= 299)]
for _, row in base_sub.iterrows():
    period = int(row['period'])
    date = row['date']
    actual = int(row['actual'])
    hit = row['hit']
    fib_idx = int(row['fib_index'])
    fib_val = fib[fib_idx] if fib_idx < len(fib) else fib[-1]
    mult = row['multiplier']
    bet = row['bet']
    recent = row['recent_rate']
    cumul = row['cumulative_profit']
    
    # 计算smart adjustment
    if recent >= 0.35:
        adj_type = "增强×1.5"
    elif recent <= 0.20:
        adj_type = "降低×0.5"
    else:
        adj_type = "正常×1.0"
    
    result = "✅命中" if hit else "❌未中"
    
    print(f"期数{period} {date} 实际{actual:2d} {result}")
    print(f"  投注前状态: Fib索引={fib_idx} → 基础倍数={fib_val}")
    print(f"  最近12期命中率: {recent:.2%} → {adj_type}")
    print(f"  最终倍数: {mult:.1f}倍 投注: {bet:.0f}元")
    print(f"  累计收益: {cumul:+.0f}元")
    print()

print("\n" + "=" * 100)
print("【暂停策略】2026/2/23 ~ 2026/2/28")
print("-" * 100)
pause_sub = pause_df[(pause_df['period'] >= 294) & (pause_df['period'] <= 299)]
for _, row in pause_sub.iterrows():
    period = int(row['period'])
    date = row['date']
    actual = int(row['actual'])
    hit = row['hit']
    fib_idx = int(row['fib_index'])
    fib_val = fib[fib_idx] if fib_idx < len(fib) else fib[-1]
    mult = row['multiplier']
    bet = row['bet']
    recent = row['recent_rate']
    cumul = row['cumulative_profit']
    paused = row['paused']
    result_str = row['result']
    
    if paused:
        print(f"期数{period} {date} 实际{actual:2d} {'✅命中' if hit else '❌未中'}")
        print(f"  ⏸️  暂停期（命中后休息）不投注")
        print(f"  累计收益: {cumul:+.0f}元")
    else:
        # 计算smart adjustment
        if recent >= 0.35:
            adj_type = "增强×1.5"
        elif recent <= 0.20:
            adj_type = "降低×0.5"
        else:
            adj_type = "正常×1.0"
        
        result = "✅命中" if hit else "❌未中"
        
        print(f"期数{period} {date} 实际{actual:2d} {result}")
        print(f"  投注前状态: Fib索引={fib_idx} → 基础倍数={fib_val}")
        print(f"  最近12期命中率: {recent:.2%} → {adj_type}")
        print(f"  最终倍数: {mult:.1f}倍 投注: {bet:.0f}元")
        print(f"  累计收益: {cumul:+.0f}元")
    print()

print("\n" + "=" * 100)
print("【关键对比】2026/2/26这一期")
print("=" * 100)

base_row = base_df[base_df['date'] == '2026/2/26'].iloc[0]
pause_row = pause_df[pause_df['date'] == '2026/2/26'].iloc[0]

print(f"\n基础策略:")
print(f"  Fib索引: {int(base_row['fib_index'])} → 基础倍数: {fib[int(base_row['fib_index'])]}")
print(f"  最近命中率: {base_row['recent_rate']:.2%}")
print(f"  最终倍数: {base_row['multiplier']:.1f}倍")
print(f"  实际投注: {base_row['bet']:.0f}元")
print(f"  命中结果: {'✅命中' if base_row['hit'] else '❌未中'}")

print(f"\n暂停策略:")
print(f"  Fib索引: {int(pause_row['fib_index'])} → 基础倍数: {fib[int(pause_row['fib_index'])]}")
print(f"  最近命中率: {pause_row['recent_rate']:.2%}")
print(f"  最终倍数: {pause_row['multiplier']:.1f}倍")
print(f"  实际投注: {pause_row['bet']:.0f}元")
print(f"  命中结果: {'✅命中' if pause_row['hit'] else '❌未中'}")

print("\n" + "=" * 100)
print("【差异原因】")
print("=" * 100)
print("基础策略在2026/2/21、2026/2/23都是继续投注（未中）")
print("暂停策略在这两期都是暂停期（之前命中），不投注")
print("因此到2026/2/26时，两个策略的Fib索引状态不同：")
print("  • 基础策略: 索引3（连续亏损累积）")
print("  • 暂停策略: 索引2（暂停后从低倍开始）")
print("\n这是暂停策略的设计目的：通过暂停避免连续小额亏损，保持在低倍区")
print("=" * 100)
