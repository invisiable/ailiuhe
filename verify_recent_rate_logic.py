#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证暂停策略中recent_results的更新逻辑"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# 读取CSV
df = pd.read_csv('validate_optimal_smart_pause_300periods.csv')

print("=" * 100)
print("验证2026/2/26的命中率计算")
print("=" * 100)

# 找到2026/2/26
target_idx = df[df['date'] == '2026/2/26'].index[0]

# 回溯12期（包括暂停期）
lookback = 12
start_idx = max(0, target_idx - lookback)

print(f"\n2026/2/26投注前的最近{lookback}期记录：")
print("-" * 100)

recent_df = df.iloc[start_idx:target_idx]

bet_count = 0
hit_count = 0
all_count = 0

print(f"{'期数':<6} {'日期':<12} {'开奖':<6} {'命中':<6} {'暂停':<6} 说明")
print("-" * 100)

for _, row in recent_df.iterrows():
    period = int(row['period'])
    date = row['date']
    actual = int(row['actual'])
    hit = '✅' if row['hit'] else '❌'
    paused = '⏸️' if row['paused'] else '🎲'
    
    all_count += 1
    
    if not row['paused']:
        bet_count += 1
        if row['hit']:
            hit_count += 1
        note = "投注期"
    else:
        note = "暂停期（不计入）"
    
    print(f"{period:<6} {date:<12} {actual:<6} {hit:<6} {paused:<6} {note}")

print("-" * 100)
print(f"\n统计（按投注期计算）：")
print(f"  总期数: {all_count}期")
print(f"  投注期数: {bet_count}期")
print(f"  暂停期数: {all_count - bet_count}期")
print(f"  投注期命中: {hit_count}期")
print(f"  投注期命中率: {hit_count}/{bet_count} = {hit_count/bet_count:.2%}")

target_row = df.iloc[target_idx]
csv_rate = target_row['recent_rate']
print(f"\nCSV记录的命中率: {csv_rate:.2%}")

print("\n" + "=" * 100)
print("【问题分析】")
print("=" * 100)
print("\n当前逻辑：")
print("  • 暂停期不调用process_period()，不更新recent_results")
print("  • recent_results只记录投注期的结果")
print("  • 命中率 = 投注期命中数 / lookback窗口大小(12)")
print("\n问题：")
print("  如果最近12个'自然期'中有4期是暂停期，")
print("  那么recent_results中只有8期的数据，")
print("  但计算命中率时除数还是12，这会导致：")
print("  • 如果recent_results长度<12，命中率计算基于不足12期的数据")
print("  • 这会使命中率偏高（分母变小）")

print("\n实际应该如何计算？")
print("  方案1: 基于投注期计算（投注期命中/投注期总数）")
print("  方案2: 基于自然期计算（投注期命中/自然期总数，暂停期算miss）")
print("  方案3: 只在投注期更新，但滑动窗口基于投注期而非自然期")

print("\n当前实现可能的问题：")
print("  • recent_results在暂停期不更新")
print("  • 但get_recent_rate()直接计算 sum(recent_results)/len(recent_results)")
print("  • 这意味着命中率是基于'投注期'而非'自然期'")
print("  • 如果暂停期较多，投注期的命中率会虚高")

print("\n" + "=" * 100)
print("【建议】")
print("=" * 100)
print("用户反馈'应该是2倍'可能意味着：")
print("  1. 命中率不应该≥35%（不应触发增强）")
print("  2. 暂停期应该计入命中率计算，作为'未命中'")
print("  3. 或者暂停策略不应使用Smart增强（已有暂停机制）")
