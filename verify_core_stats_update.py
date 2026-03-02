#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证核心统计数据是否基于暂停策略"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 100)
print("验证核心统计数据更新")
print("=" * 100)

with open('d:\\AiLiuHe\\lucky_number_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找核心统计数据部分
start_marker = "第二步：核心统计数据"
end_marker = "第三步：下期投注建议"

start_idx = content.find(start_marker)
if start_idx != -1:
    # 从start_idx之后查找end_marker
    end_idx = content.find(end_marker, start_idx)
else:
    end_idx = -1

if start_idx == -1 or end_idx == -1:
    print("❌ 找不到核心统计数据部分")
    sys.exit(1)

core_stats_section = content[start_idx:end_idx]

print("\n✅ 找到核心统计数据部分")
print(f"   起始位置: {start_idx}")
print(f"   结束位置: {end_idx}")
print(f"   长度: {len(core_stats_section)} 字符")

# 检查是否使用了暂停策略的变量
pause_vars = [
    "pause_variant['total_periods']",
    "pause_variant['bet_periods']",
    "pause_variant['pause_periods']",
    "pause_variant['wins']",
    "pause_variant['hit_rate']",
    "pause_variant['total_cost']",
    "pause_variant['total_reward']",
    "pause_variant['total_profit']",
    "pause_variant['roi']",
    "pause_variant['max_drawdown']",
    "pause_variant['hit_10x_count']",
    "pause_variant['max_consecutive_losses']",
]

print("\n检查暂停策略变量使用情况：")
print("-" * 100)

all_found = True
for var in pause_vars:
    if var in core_stats_section:
        print(f"  ✅ {var:<45} 已使用")
    else:
        print(f"  ❌ {var:<45} 未找到")
        all_found = False

# 检查是否还有基础策略的变量（应该被替换掉）
old_vars = [
    ("len(results)", "应改为 pause_variant['total_periods'] 或 pause_variant['bet_periods']"),
    ("hits", "应改为 pause_variant['wins']"),
    ("hit_rate*100", "应改为 pause_variant['hit_rate']*100"),
    ("total_cost", "应改为 pause_variant['total_cost']"),
    ("strategy.total_win", "应改为 pause_variant['total_reward']"),
    ("total_profit", "应改为 pause_variant['total_profit']"),
    ("roi", "应改为 pause_variant['roi']"),
    ("strategy.max_drawdown", "应改为 pause_variant['max_drawdown']"),
    ("hit_10x_count", "应改为 pause_variant['hit_10x_count']"),
    ("max_consecutive_losses", "应改为 pause_variant['max_consecutive_losses']"),
]

print("\n检查是否还有基础策略变量（应避免）：")
print("-" * 100)

has_old_vars = False
for var, suggestion in old_vars:
    # 排除已经是pause_variant开头的行
    lines_with_var = [line for line in core_stats_section.split('\n') if var in line and 'pause_variant' not in line]
    if lines_with_var:
        print(f"  ⚠️  发现 '{var}':")
        for line in lines_with_var[:3]:  # 只显示前3个
            print(f"      {line.strip()}")
        print(f"      建议: {suggestion}")
        has_old_vars = True

if not has_old_vars:
    print("  ✅ 未发现基础策略变量")

print("\n" + "=" * 100)
print("检查标题是否更新：")
print("-" * 100)

if "命中1停1期暂停策略" in core_stats_section:
    print("  ✅ 标题已更新为包含'命中1停1期暂停策略'")
elif "第二步：核心统计数据" in core_stats_section:
    print("  ⚠️  标题未明确标注暂停策略")
else:
    print("  ❌ 标题异常")

print("\n" + "=" * 100)
if all_found and not has_old_vars:
    print("✅ 核心统计数据已成功更新为基于暂停策略！")
else:
    print("⚠️  核心统计数据部分更新，但可能还有遗留问题")
print("=" * 100)
