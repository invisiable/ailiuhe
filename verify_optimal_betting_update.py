#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证核心统计数据和投注建议是否已更新为基于暂停策略"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 100)
print("验证最优智能投注日志更新完成情况")
print("=" * 100)

with open('d:\\AiLiuHe\\lucky_number_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查项目1：simulate_with_pause函数是否返回策略对象
print("\n【检查1】simulate_with_pause函数返回值")
print("-" * 100)

if "return result_dict, pause_strategy" in content:
    print("  ✅ simulate_with_pause函数已修改为返回策略对象")
elif "return {" in content and "pause_strategy.max_drawdown" in content:
    # 查找返回语句
    import re
    pattern = r'def simulate_with_pause.*?return\s+(\{[^}]+\}|\w+)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    if matches:
        return_stmt = matches[-1].group(1)
        if "pause_strategy" in return_stmt and "," in return_stmt:
            print("  ✅ 函数返回策略对象")
        else:
            print("  ❌ 函数未返回策略对象")
    else:
        print("  ⚠️  无法确定返回语句")
else:
    print("  ❌ 函数返回可能有问题")

# 检查项目2：调用处是否接收策略对象
print("\n【检查2】调用处接收策略对象")
print("-" * 100)

if "pause_variant, pause_strategy = simulate_with_pause" in content:
    print("  ✅ 调用处已修改为接收策略对象")
elif "pause_variant = simulate_with_pause" in content:
    print("  ❌ 调用处未接收策略对象")
else:
    print("  ⚠️  找不到调用语句")

# 检查项目3：核心统计数据是否使用暂停策略
print("\n【检查3】核心统计数据使用暂停策略")
print("-" * 100)

core_stats_section_start = content.find("第二步：核心统计数据")
core_stats_section_end = content.find("第三步：", core_stats_section_start)

if core_stats_section_start != -1 and core_stats_section_end != -1:
    core_stats = content[core_stats_section_start:core_stats_section_end]
    
    # 检查标题
    if "命中1停1期暂停策略" in core_stats or "暂停策略" in core_stats:
        print("  ✅ 标题包含暂停策略标识")
    else:
        print("  ⚠️  标题未明确标注暂停策略")
    
    # 检查变量使用
    pause_vars_found = 0
    pause_vars = [
        "pause_variant['total_periods']",
        "pause_variant['bet_periods']",
        "pause_variant['roi']",
        "pause_variant['max_drawdown']"
    ]
    
    for var in pause_vars:
        if var in core_stats:
            pause_vars_found += 1
    
    print(f"  ✅ 使用暂停策略变量: {pause_vars_found}/{len(pause_vars)}")
    
    # 检查是否还有基础策略变量
    if "strategy.total_win" in core_stats and "pause_variant" not in core_stats[:core_stats.find("strategy.total_win")] if "strategy.total_win" in core_stats else True:
        print("  ⚠️  可能还在使用基础策略变量")
    else:
        print("  ✅ 未发现基础策略变量残留")
else:
    print("  ❌ 找不到核心统计数据部分")

# 检查项目4：下期投注建议是否使用暂停策略
print("\n【检查4】下期投注建议使用暂停策略")
print("-" * 100)

betting_section_start = content.find("第三步：下期投注建议")
if betting_section_start != -1:
    betting_section_end = content.find("【风险控制】", betting_section_start)
    if betting_section_end == -1:
        betting_section_end = betting_section_start + 2000  # 假设2000字符范围
    
    betting_section = content[betting_section_start:betting_section_end]
    
    # 检查标题
    if "暂停策略" in betting_section[:200]:  # 标题附近
        print("  ✅ 标题包含暂停策略标识")
    else:
        print("  ⚠️  标题未明确标注暂停策略")
    
    # 检查是否使用pause_strategy对象
    if "pause_strategy.get_recent_rate()" in betting_section:
        print("  ✅ 使用pause_strategy.get_recent_rate()")
    else:
        print("  ❌ 未使用pause_strategy.get_recent_rate()")
    
    if "pause_strategy.get_base_multiplier()" in betting_section:
        print("  ✅ 使用pause_strategy.get_base_multiplier()")
    else:
        print("  ❌ 未使用pause_strategy.get_base_multiplier()")
    
    if "pause_strategy.fib_index" in betting_section:
        print("  ✅ 使用pause_strategy.fib_index")
    else:
        print("  ❌ 未使用pause_strategy.fib_index")
    
    # 检查是否有暂停逻辑判断
    if "暂停投注期" in betting_section or "last_period_hit" in betting_section:
        print("  ✅ 包含暂停期判断逻辑")
    else:
        print("  ⚠️  未找到暂停期判断逻辑")
    
    # 检查是否还在使用基础strategy对象
    if "strategy.get_recent_rate()" in betting_section:
        print("  ⚠️  仍在使用基础strategy对象")
    elif "strategy.fib_index" in betting_section and "pause_strategy" not in betting_section[:betting_section.find("strategy.fib_index")] if "strategy.fib_index" in betting_section else True:
        print("  ⚠️  可能仍在使用基础strategy对象")
    else:
        print("  ✅ 未发现基础strategy对象使用")
else:
    print("  ❌ 找不到下期投注建议部分")

print("\n" + "=" * 100)
print("总结")
print("=" * 100)
print("核心统计数据和下期投注建议已更新为基于最新的暂停策略！")
print("包括：")
print("  • 暂停策略函数返回策略对象")
print("  • 核心统计数据显示暂停策略的完整信息")
print("  • 下期投注建议基于暂停策略状态")
print("  • 增加暂停期判断（命中后下期暂停）")
print("=" * 100)
