#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""追踪recent_results的长度变化"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# Fibonacci数列  
fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

# 配置
config = {
    'base_bet': 15,
    'win_reward': 47,
    'max_multiplier': 10,
    'lookback': 12,
    'good_thresh': 0.35,
    'bad_thresh': 0.20,
    'boost_mult': 1.5,
    'reduce_mult': 0.5
}

# 简化版SmartDynamicStrategy
class SimpleStrategy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fib_index = 0
        self.recent_results = []
    
    def get_base_multiplier(self):
        if self.fib_index >= len(fib_sequence):
            return min(fib_sequence[-1], self.cfg['max_multiplier'])
        return min(fib_sequence[self.fib_index], self.cfg['max_multiplier'])
    
    def get_recent_rate(self):
        if len(self.recent_results) == 0:
            return 0.33
        return sum(self.recent_results) / len(self.recent_results)
    
    def process_period(self, hit):
        base_mult = self.get_base_multiplier()
        
        # 关键判断：只有当recent_results长度≥lookback时才应用Smart调整
        if len(self.recent_results) >= self.cfg['lookback']:
            rate = self.get_recent_rate()
            if rate >= self.cfg['good_thresh']:
                multiplier = min(base_mult * self.cfg['boost_mult'], self.cfg['max_multiplier'])
            elif rate <= self.cfg['bad_thresh']:
                multiplier = max(base_mult * self.cfg['reduce_mult'], 1)
            else:
                multiplier = base_mult
        else:
            multiplier = base_mult
        
        if hit:
            self.fib_index = 0
        else:
            self.fib_index += 1
        
        self.recent_results.append(1 if hit else 0)
        if len(self.recent_results) > self.cfg['lookback']:
            self.recent_results.pop(0)
        
        return multiplier
    
    def add_pause_period(self):
        """暂停期添加0"""
        self.recent_results.append(0)
        if len(self.recent_results) > self.cfg['lookback']:
            self.recent_results.pop(0)

# 读取CSV
df = pd.read_csv('validate_optimal_smart_pause_300periods.csv')

# 模拟到2026/2/26
strategy = SimpleStrategy(config)

print("=" * 100)
print("模拟追踪到2026/2/26，查看recent_results长度")
print("=" * 100)

target_df = df[(df['period'] >= 285) & (df['period'] <= 297)]

for _, row in target_df.iterrows():
    period = int(row['period'])
    date = row['date']
    hit = row['hit']
    paused = row['paused']
    
    # 投注前状态
    len_before = len(strategy.recent_results)
    rate_before = strategy.get_recent_rate()
    fib_idx = strategy.fib_index
    
    if period == 297:  # 2026/2/26
        print(f"\n{'='*100}")
        print(f"【关键期数 {period}】{date}")
        print(f"{'='*100}")
        print(f"投注前状态:")
        print(f"  recent_results长度: {len_before}")
        print(f"  lookback要求: {config['lookback']}")
        print(f"  是否满足Smart条件: {'✅ 是' if len_before >= config['lookback'] else '❌ 否'}")
        print(f"  命中率: {rate_before:.2%}")
        print(f"  Fib索引: {fib_idx}")
        print(f"  基础倍数: {fib_sequence[fib_idx]}")
        
        if len_before >= config['lookback']:
            if rate_before >= config['good_thresh']:
                theory_mult = fib_sequence[fib_idx] * config['boost_mult']
                print(f"  Smart调整: 增强 ({rate_before:.2%} ≥ {config['good_thresh']:.0%})")
                print(f"  理论倍数: {fib_sequence[fib_idx]} × {config['boost_mult']} = {theory_mult}")
            elif rate_before <= config['bad_thresh']:
                theory_mult = fib_sequence[fib_idx] * config['reduce_mult']
                print(f"  Smart调整: 降低 ({rate_before:.2%} ≤ {config['bad_thresh']:.0%})")
                print(f"  理论倍数: {fib_sequence[fib_idx]} × {config['reduce_mult']} = {theory_mult}")
            else:
                theory_mult = fib_sequence[fib_idx]
                print(f"  Smart调整: 正常 ({config['bad_thresh']:.0%} < {rate_before:.2%} < {config['good_thresh']:.0%})")
                print(f"  理论倍数: {fib_sequence[fib_idx]}")
        else:
            theory_mult = fib_sequence[fib_idx]
            print(f"  Smart调整: ❌ 不应用（数据不足）")
            print(f"  理论倍数: {fib_sequence[fib_idx]} (仅基础倍数)")
        
        print(f"\nCSV记录:")
        print(f"  实际倍数: {row['multiplier']}")
        print(f"  实际投注: {row['bet']}元")
        print(f"{'='*100}\n")
    
    # 处理这一期
    if paused:
        strategy.add_pause_period()
    else:
        strategy.process_period(hit)
    
    len_after = len(strategy.recent_results)
    
    if period <= 297:
        status = "⏸️ 暂停" if paused else ("✅ 命中" if hit else "❌ 未中")
        print(f"期数{period:3d} {date} {status:<10} recent_results长度: {len_before} → {len_after}")

print("\n" + "=" * 100)
print("结论")
print("=" * 100)
print("如果recent_results长度不足12，Smart调整不会生效，")
print("倍数将直接使用Fibonacci基础倍数。")
print("这解释了为什么2026/2/26是2倍而不是3倍。")
