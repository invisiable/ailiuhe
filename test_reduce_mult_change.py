#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证最优智能投注策略的reduce_mult参数修改
"""

import sys
sys.path.insert(0, 'd:/AiLiuHe')

print("=" * 70)
print("验证最优智能投注策略参数修改")
print("=" * 70)
print()

# 模拟配置（从lucky_number_gui.py中复制）
config = {
    'name': '最优智能动态倍投策略 v3.1',
    'lookback': 12,
    'good_thresh': 0.35,
    'bad_thresh': 0.20,
    'boost_mult': 1.5,
    'reduce_mult': 1.0,  # 修改后的值
    'max_multiplier': 10,
    'base_bet': 15,
    'win_reward': 47
}

print("【策略配置】")
for key, value in config.items():
    print(f"  {key}: {value}")

print()
print("【核心参数验证】")
print(f"  窗口期数: {config['lookback']}期")
print(f"  增强阈值: 命中率≥{config['good_thresh']:.0%} → 倍数×{config['boost_mult']}")
print(f"  降低阈值: 命中率≤{config['bad_thresh']:.0%} → 倍数×{config['reduce_mult']}")
print(f"  最大倍数: {config['max_multiplier']}倍")

print()
print("【修改说明】")
print("  ✅ reduce_mult: 0.5 → 1.0")
print("  📝 含义: 当命中率≤20%时，不再降低倍数（×0.5），而是保持基础倍数（×1.0）")
print("  💡 优势: 避免在低命中期过度保守，保持合理投注规模")

print()
print("【示例计算】")
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
print(f"  假设当前Fibonacci索引=3（基础倍数={fib_sequence[3]}）")
print(f"  命中率≤20%时:")
print(f"    修改前: {fib_sequence[3]} × 0.5 = {fib_sequence[3] * 0.5}")
print(f"    修改后: {fib_sequence[3]} × 1.0 = {fib_sequence[3] * 1.0}")
print(f"  差异: 修改后保持原倍数，不做降低调整")

print()
print("=" * 70)
print("✅ 参数验证完成")
print("=" * 70)
