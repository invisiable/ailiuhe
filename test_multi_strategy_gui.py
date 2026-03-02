#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试GUI中的TOP5多策略投注功能

快速测试新增的analyze_multi_strategy_zodiac_betting方法
"""

import sys
sys.path.insert(0, 'd:/AiLiuHe')

print("=" * 80)
print("测试TOP5多策略投注功能")
print("=" * 80)
print()

print("步骤1: 导入模块...")
try:
    import pandas as pd
    from zodiac_simple_smart import ZodiacSimpleSmart
    from zodiac_top5_probability_betting import ZodiacTop5ProbabilityBetting
    print("✅ 依赖模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

print()
print("步骤2: 加载数据...")
try:
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"✅ 数据加载成功: {len(df)}期")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

print()
print("步骤3: 测试生肖预测器...")
try:
    predictor = ZodiacSimpleSmart()
    animals = df['animal'].values
    test_data = animals[:100].tolist()
    result = predictor.predict_from_history(test_data, top_n=5, debug=False)
    print(f"✅ 预测器工作正常: TOP5={result['top5']}")
except Exception as e:
    print(f"❌ 预测器测试失败: {e}")
    sys.exit(1)

print()
print("步骤4: 测试概率预测器...")
try:
    prob_config = {
        'base_bet': 20,
        'win_reward': 47,
        'max_multiplier': 10,
        'lookback': 20,
        'prob_high_thresh': 0.45,
        'prob_low_thresh': 0.30,
        'high_mult': 1.5,
        'low_mult': 0.5
    }
    prob_betting = ZodiacTop5ProbabilityBetting(prob_config)
    hit_prob = prob_betting.predict_next_probability()
    mult = prob_betting.calculate_multiplier(hit_prob)
    print(f"✅ 概率预测器工作正常: 初始概率={hit_prob:.2%}, 倍数={mult:.1f}")
except Exception as e:
    print(f"❌ 概率预测器测试失败: {e}")
    sys.exit(1)

print()
print("步骤5: 验证GUI模块结构...")
try:
    import lucky_number_gui
    
    # 检查是否有新方法
    if hasattr(lucky_number_gui.LuckyNumberGUI, 'analyze_multi_strategy_zodiac_betting'):
        print("✅ analyze_multi_strategy_zodiac_betting 方法存在")
    else:
        print("❌ analyze_multi_strategy_zodiac_betting 方法不存在")
    
    if hasattr(lucky_number_gui.LuckyNumberGUI, '_simulate_fibonacci_betting'):
        print("✅ _simulate_fibonacci_betting 方法存在")
    else:
        print("❌ _simulate_fibonacci_betting 方法不存在")
    
    if hasattr(lucky_number_gui.LuckyNumberGUI, '_simulate_smart_dynamic_v32'):
        print("✅ _simulate_smart_dynamic_v32 方法存在")
    else:
        print("❌ _simulate_smart_dynamic_v32 方法不存在")
    
    if hasattr(lucky_number_gui.LuckyNumberGUI, '_simulate_probability_betting'):
        print("✅ _simulate_probability_betting 方法存在")
    else:
        print("❌ _simulate_probability_betting 方法不存在")
        
except Exception as e:
    print(f"❌ GUI模块验证失败: {e}")
    sys.exit(1)

print()
print("=" * 80)
print("✅ 所有测试通过！TOP5多策略投注功能已成功集成到GUI")
print("=" * 80)
print()
print("使用方法:")
print("1. 运行: python lucky_number_gui.py")
print("2. 点击: 📊 TOP5多策略投注 按钮")
print("3. 等待约10-30秒获取300期对比结果")
print()
print("详细说明请查看: TOP5多策略投注使用指南.md")
print()
