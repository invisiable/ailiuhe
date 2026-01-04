"""
测试投注策略按钮点击
"""

import sys
import traceback

print("=" * 80)
print("测试投注策略功能")
print("=" * 80)
print()

# 测试1: 检查数据文件
print("1. 检查数据文件...")
try:
    import pandas as pd
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    print(f"   ✓ 数据文件存在: {len(df)}期")
except Exception as e:
    print(f"   ✗ 数据文件错误: {e}")
    sys.exit(1)

print()

# 测试2: 导入必需模块
print("2. 检查模块导入...")
try:
    from betting_strategy import BettingStrategy
    print("   ✓ BettingStrategy导入成功")
except Exception as e:
    print(f"   ✗ BettingStrategy导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from top15_predictor import Top15Predictor
    print("   ✓ Top15Predictor导入成功")
except Exception as e:
    print(f"   ✗ Top15Predictor导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print()

# 测试3: 模拟按钮点击的核心逻辑
print("3. 模拟按钮点击逻辑...")
try:
    from datetime import datetime
    
    print("   读取数据...")
    file_path = 'data/lucky_numbers.csv'
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    if len(df) < 50:
        print(f"   ✗ 数据不足: 只有{len(df)}期，需要至少50期")
        sys.exit(1)
    
    print(f"   ✓ 数据充足: {len(df)}期")
    
    # 使用最近30期进行快速测试
    test_periods = min(30, len(df))
    start_idx = len(df) - test_periods
    
    print(f"   测试期数: {test_periods}期")
    
    predictor = Top15Predictor()
    predictions_top5 = []
    actuals = []
    
    print("   生成预测（使用与综合预测相同方法）...")
    for i in range(start_idx, len(df)):
        train_data = df.iloc[:i]['number'].values
        analysis = predictor.get_analysis(train_data)
        top15 = analysis['top15']
        top5 = top15[:5]
        predictions_top5.append(top5)
        actual = df.iloc[i]['number']
        actuals.append(actual)
        
        if (i - start_idx + 1) % 10 == 0:
            print(f"   处理进度: {i - start_idx + 1}/{test_periods}")
    
    print(f"   ✓ 预测生成完成")
    
    # 创建投注策略
    betting = BettingStrategy()
    
    # 测试一个策略
    print("   运行策略模拟...")
    result = betting.simulate_strategy(predictions_top5, actuals, 'martingale')
    
    print(f"   ✓ 策略模拟成功")
    print(f"   命中率: {result['hit_rate']*100:.1f}%")
    print(f"   总收益: {result['total_profit']:+.2f}元")
    
except Exception as e:
    print(f"   ✗ 执行失败: {e}")
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✅ 所有测试通过！按钮功能正常")
print("=" * 80)
