#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试2026/2/26暂停策略倍数计算"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# 读取暂停策略CSV
df = pd.read_csv('validate_optimal_smart_pause_300periods.csv')

# Fib序列
fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

print("=" * 100)
print("追踪2026/2/26前后的完整状态变化")
print("=" * 100)

# 查看2/20-2/28的数据
target_df = df[(df['period'] >= 291) & (df['period'] <= 299)]

print("\n逐期分析：")
print("-" * 100)

for _, row in target_df.iterrows():
    period = int(row['period'])
    date = row['date']
    actual = int(row['actual'])
    hit = row['hit']
    paused = row['paused']
    fib_idx = int(row['fib_index'])
    mult = row['multiplier']
    bet = row['bet']
    recent_rate = row['recent_rate']
    result = row['result']
    
    print(f"\n【期数 {period}】{date} 开奖号: {actual}")
    
    if paused:
        print(f"  状态: ⏸️ 暂停期（不投注）")
        print(f"  命中: {'✅' if hit else '❌'}")
        print(f"  Fib索引: {fib_idx}")
        print(f"  最近命中率: {recent_rate:.2%}")
    else:
        fib_base = fib[fib_idx] if fib_idx < len(fib) else fib[-1]
        
        # 计算理论倍数
        if recent_rate >= 0.35:
            adj_mult = fib_base * 1.5
            adj_desc = "增强 (≥35%)"
        elif recent_rate <= 0.20:
            adj_mult = fib_base * 0.5
            adj_desc = "降低 (≤20%)"
        else:
            adj_mult = fib_base * 1.0
            adj_desc = "正常"
        
        print(f"  状态: 🎲 正常投注")
        print(f"  投注前Fib索引: {fib_idx} → 基础倍数 = {fib_base}")
        print(f"  投注前命中率: {recent_rate:.2%} → Smart调整: {adj_desc}")
        print(f"  理论倍数: {fib_base} × {adj_mult/fib_base:.1f} = {adj_mult:.1f}")
        print(f"  实际倍数: {mult:.1f}")
        print(f"  实际投注: {bet:.0f}元")
        print(f"  命中结果: {'✅ 命中' if hit else '❌ 未中'} ({result})")
        
        if abs(mult - adj_mult) > 0.01:
            print(f"  ⚠️ 警告: 理论倍数 ({adj_mult:.1f}) ≠ 实际倍数 ({mult:.1f})")

print("\n" + "=" * 100)
print("【重点：2026/2/26分析】")
print("=" * 100)

row_2_26 = df[df['date'] == '2026/2/26'].iloc[0]
fib_idx = int(row_2_26['fib_index'])
fib_base = fib[fib_idx]
recent = row_2_26['recent_rate']
mult = row_2_26['multiplier']
bet = row_2_26['bet']

print(f"\n投注前状态:")
print(f"  Fib索引: {fib_idx}")
print(f"  Fib基础倍数: {fib_base}")
print(f"  最近12期命中率: {recent:.4f} ({recent:.2%})")
print(f"\nSmart调整判断:")
print(f"  good_thresh = 0.35 (35%)")
print(f"  bad_thresh = 0.20 (20%)")
print(f"  当前命中率 {recent:.4f} vs 阈值 0.35: {'≥' if recent >= 0.35 else '<'}")

if recent >= 0.35:
    theory_mult = fib_base * 1.5
    print(f"  结论: 触发增强 → {fib_base} × 1.5 = {theory_mult}")
elif recent <= 0.20:
    theory_mult = fib_base * 0.5
    print(f"  结论: 触发降低 → {fib_base} × 0.5 = {theory_mult}")
else:
    theory_mult = fib_base * 1.0
    print(f"  结论: 保持正常 → {fib_base} × 1.0 = {theory_mult}")

print(f"\n实际投注:")
print(f"  CSV记录倍数: {mult}")
print(f"  CSV记录投注: {bet}元")
print(f"  理论倍数: {theory_mult}")
print(f"  理论投注: {theory_mult * 15}元")

if abs(mult - theory_mult) > 0.01:
    print(f"\n❌ 发现差异: 理论倍数 {theory_mult} ≠ 实际倍数 {mult}")
    print(f"   差异原因需要进一步调查代码逻辑")
else:
    print(f"\n✅ 倍数计算一致")

print("\n" + "=" * 100)
print("【用户反馈】")
print("=" * 100)
print(f"用户认为: 2026/2/26应该是2倍")
print(f"系统显示: {mult}倍")
print(f"Fib基础: {fib_base}倍")
print(f"\n可能的理解:")
print(f"1. 用户期望基础倍数 = {fib_base}倍（不应用Smart增强）")
print(f"2. 用户期望暂停后恢复时不立即触发增强")
print(f"3. 用户认为命中率计算有误（当前{recent:.2%}）")
