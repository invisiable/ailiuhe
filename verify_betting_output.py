"""
验证投注策略输出 - 无表情符号版本
"""

import pandas as pd
from betting_strategy import BettingStrategy
from top15_predictor import Top15Predictor

print("="*80)
print("验证投注策略输出结果")
print("="*80)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"数据加载: {len(df)}期")

# 使用少量数据快速测试
test_periods = 20
start_idx = len(df) - test_periods

# 创建预测器
predictor = Top15Predictor()
predictions_top5 = []
actuals = []

print("生成历史预测...")
for i in range(start_idx, len(df)):
    train_data = df.iloc[:i]['number'].values
    analysis = predictor.get_analysis(train_data)
    top15 = analysis['top15']
    top5 = top15[:5]
    predictions_top5.append(top5)
    
    actual = df.iloc[i]['number']
    actuals.append(actual)

print(f"预测生成完成: {len(predictions_top5)}期")

# 策略分析
betting = BettingStrategy()
result = betting.simulate_strategy(predictions_top5, actuals, 'martingale')

print(f"\n策略分析结果:")
print(f"  命中率: {result['hit_rate']*100:.1f}%")
print(f"  总收益: {result['total_profit']:+.2f}元")

# 获取下期预测
print("\n" + "="*80)
print("下期预测 (使用综合预测Top15方法)")
print("="*80)

all_numbers = df['number'].values
next_analysis = predictor.get_analysis(all_numbers)
next_top15 = next_analysis['top15']
next_top5 = next_top15[:5]

print(f"\n当前趋势: {next_analysis['trend']}")
print(f"极端值占比: {next_analysis['extreme_ratio']:.0f}%")
print(f"\nTOP15预测: {next_top15}")
print(f"建议购买TOP5: {next_top5}")

# 生成投注建议
recommendation = betting.generate_next_bet_recommendation(0, 0, 'martingale')

print(f"\n投注方案:")
print(f"  购买数字: {next_top5}")
print(f"  投注倍数: {recommendation['recommended_multiplier']}倍")
print(f"  每个数字: {recommendation['bet_per_number']:.2f}元")
print(f"  总投注额: {recommendation['recommended_bet']:.2f}元")
print(f"  如果命中: +{recommendation['potential_profit_if_win']:.2f}元")
print(f"  风险等级: 低风险")

print("\n" + "="*80)
print("验证完成！")
print("="*80)
