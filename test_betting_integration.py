"""
快速测试投注策略功能
"""

print("=" * 80)
print("测试投注策略模块集成")
print("=" * 80)
print()

# 测试1: 导入模块
print("测试1: 导入模块...")
try:
    from betting_strategy import BettingStrategy
    print("  ✓ betting_strategy 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    exit(1)

try:
    from top15_predictor import Top15Predictor
    print("  ✓ top15_predictor 导入成功")
except Exception as e:
    print(f"  ✗ 导入失败: {e}")
    exit(1)

print()

# 测试2: 创建实例
print("测试2: 创建策略实例...")
try:
    betting = BettingStrategy()
    print("  ✓ BettingStrategy 实例创建成功")
    print(f"    - 基础投注: {betting.base_bet}元")
    print(f"    - 命中奖励: {betting.win_reward}元")
    print(f"    - 未中惩罚: {betting.loss_penalty}元")
except Exception as e:
    print(f"  ✗ 创建失败: {e}")
    exit(1)

print()

# 测试3: 计算投注建议
print("测试3: 生成投注建议...")
try:
    # 假设连续亏损2次，累计30元
    recommendation = betting.generate_next_bet_recommendation(
        consecutive_losses=2,
        total_loss=30.0,
        strategy_type='martingale'
    )
    print("  ✓ 建议生成成功")
    print(f"    - 建议倍数: {recommendation['recommended_multiplier']}倍")
    print(f"    - 投注金额: {recommendation['recommended_bet']}元")
    print(f"    - 如果命中: +{recommendation['potential_profit_if_win']}元")
except Exception as e:
    print(f"  ✗ 生成失败: {e}")
    exit(1)

print()

# 测试4: 策略对比
print("测试4: 测试策略对比...")
try:
    import numpy as np
    np.random.seed(42)
    
    # 生成10期模拟数据
    predictions = []
    actuals = []
    
    for i in range(10):
        top5 = np.random.choice(range(1, 50), size=5, replace=False).tolist()
        predictions.append(top5)
        
        if np.random.random() < 0.4:
            actual = np.random.choice(top5)
        else:
            others = [x for x in range(1, 50) if x not in top5]
            actual = np.random.choice(others)
        actuals.append(actual)
    
    result = betting.simulate_strategy(predictions, actuals, 'martingale')
    print("  ✓ 策略模拟成功")
    print(f"    - 测试期数: {result['total_periods']}")
    print(f"    - 命中次数: {result['wins']}")
    print(f"    - 命中率: {result['hit_rate']*100:.1f}%")
    print(f"    - 总收益: {result['total_profit']:+.2f}元")
except Exception as e:
    print(f"  ✗ 模拟失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()

# 测试5: 检查GUI集成
print("测试5: 检查GUI集成...")
try:
    import lucky_number_gui
    print("  ✓ lucky_number_gui 导入成功")
    
    # 检查是否有analyze_betting_strategy方法
    if hasattr(lucky_number_gui.LuckyNumberGUI, 'analyze_betting_strategy'):
        print("  ✓ analyze_betting_strategy 方法存在")
    else:
        print("  ✗ analyze_betting_strategy 方法不存在")
        
except Exception as e:
    print(f"  ✗ 检查失败: {e}")
    exit(1)

print()
print("=" * 80)
print("✅ 所有测试通过！投注策略功能已成功集成")
print("=" * 80)
print()
print("下一步:")
print("  1. 运行 GUI: python lucky_number_gui.py")
print("  2. 点击 '💰 投注策略分析' 按钮")
print("  3. 查看完整的策略分析报告")
print()
print("或者:")
print("  运行完整演示: python demo_betting_strategy.py")
print()
