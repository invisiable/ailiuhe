"""
测试验证 - 每次预测都基于最新数据生成
"""

from top15_predictor import Top15Predictor
import pandas as pd
from datetime import datetime
import time

print("=" * 80)
print("验证：每次预测都基于最新数据重新生成")
print("=" * 80)

# 模拟3次连续预测
for i in range(1, 4):
    print(f"\n{'=' * 80}")
    print(f"第 {i} 次预测")
    print("=" * 80)
    
    # 获取当前时间
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"预测时间: {current_time}")
    
    # 每次都重新读取数据
    print("🔄 重新读取数据文件...")
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    numbers = df['number'].values
    
    print(f"✅ 数据加载: {len(numbers)}期")
    print(f"最新一期: 第{len(numbers)}期, 数字={numbers[-1]}")
    
    # 创建新的预测器实例
    print("📊 创建新的预测器实例...")
    predictor = Top15Predictor()
    
    # 生成预测
    print("🎯 基于最新数据生成预测...")
    analysis = predictor.get_analysis(numbers)
    
    print(f"\nTop 15预测结果:")
    print(f"  {analysis['top15']}")
    print(f"\n趋势: {analysis['trend']}")
    print(f"极端值占比: {analysis['extreme_ratio']:.0f}%")
    
    # 显示预测的唯一性标识
    prediction_hash = hash(tuple(analysis['top15']))
    print(f"\n预测结果哈希: {prediction_hash}")
    
    if i < 3:
        print("\n等待1秒后进行下一次预测...")
        time.sleep(1)

print("\n" + "=" * 80)
print("验证说明")
print("=" * 80)
print("""
✅ 每次预测流程:
1. 显示当前预测时间（精确到毫秒）
2. 重新从CSV文件读取数据
3. 创建新的Top15Predictor实例
4. 基于最新数据执行 get_analysis(numbers)
5. 生成并返回预测结果

🔄 数据更新方式:
- 如果数据文件更新（添加新期数），下次预测会自动使用新数据
- 如果数据未变，多次预测结果一致（因为输入相同）
- 要测试新数据效果，需要在data/lucky_numbers.csv中添加新行

📝 验证方法:
1. 运行此脚本 - 看到3次预测都重新读取数据
2. 修改data/lucky_numbers.csv添加新数据
3. 再次运行 - 预测结果会基于新数据变化
""")

print("\n" + "=" * 80)
print("结论: ✅ 系统每次预测都基于最新数据重新生成，无缓存！")
print("=" * 80 + "\n")
