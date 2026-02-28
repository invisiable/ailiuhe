"""
测试TOP15投注策略的300期回测功能
"""

import pandas as pd
from top15_predictor import Top15Predictor
from betting_strategy import BettingStrategy

print("="*120)
print("测试 TOP15投注策略 - 300期回测")
print("="*120)
print()

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"✓ 总数据量: {len(df)}期")

# 检查是否有足够的数据
if len(df) < 300:
    print(f"⚠️ 数据不足300期，实际可用: {len(df)}期")
    test_periods = len(df)
else:
    print(f"✓ 数据充足，测试300期")
    test_periods = 300

print()

# 导入Top15预测器
predictor = Top15Predictor()

predictions_top15 = []
actuals = []
dates = []

start_idx = len(df) - test_periods

print("正在生成每期的TOP15预测...")

for i in range(start_idx, len(df)):
    # 使用i之前的数据进行预测
    train_data = df.iloc[:i]['number'].values
    
    # 获取top15预测
    analysis = predictor.get_analysis(train_data)
    top15 = analysis['top15']
    
    predictions_top15.append(top15)
    
    # 实际结果
    actual = df.iloc[i]['number']
    actuals.append(actual)
    
    # 记录日期
    date = df.iloc[i]['date']
    dates.append(date)
    
    if (i - start_idx + 1) % 50 == 0:
        print(f"  已处理 {i - start_idx + 1}/{test_periods} 期...")

print(f"\n✓ 预测生成完成！共 {len(predictions_top15)} 期\n")

# 计算命中率
actual_hit_rate = sum(1 for i in range(len(actuals)) if actuals[i] in predictions_top15[i]) / len(actuals)
print(f"实际命中率: {actual_hit_rate*100:.2f}%")

# 创建投注策略实例
betting = BettingStrategy(base_bet=15, win_reward=47, loss_penalty=15)

# 使用斐波那契策略进行回测
print("\n正在使用斐波那契策略进行回测...")
result = betting.simulate_strategy(predictions_top15, actuals, 'fibonacci', hit_rate=actual_hit_rate)

# 添加日期信息
for i, period_data in enumerate(result['history']):
    if i < len(dates):
        period_data['date'] = dates[i]

print("\n回测结果:")
print(f"  总期数: {result['total_periods']}")
print(f"  命中次数: {result['wins']}")
print(f"  命中率: {result['hit_rate']*100:.2f}%")
print(f"  总投注: {result['total_cost']:.2f}元")
print(f"  总收益: {result['total_profit']:+.2f}元")
print(f"  ROI: {result['roi']:+.2f}%")
print(f"  最大连续亏损: {result['max_consecutive_losses']}期")

# 检查历史记录数量
print(f"\n历史记录总数: {len(result['history'])}期")

# 检查是否能显示最近300期
if len(result['history']) >= 300:
    print(f"✓ 数据足够显示最近300期")
    display_periods = 300
else:
    print(f"⚠️ 数据不足300期，只能显示最近{len(result['history'])}期")
    display_periods = len(result['history'])

# 显示最近300期的分析
print(f"\n{'='*120}")
print(f"最近{display_periods}期详情分析")
print(f"{'='*120}")

recent_data = result['history'][-display_periods:]
recent_hits = sum(1 for p in recent_data if p['result'] == 'WIN')
recent_hit_rate = recent_hits / len(recent_data)
recent_profit = sum(p['profit'] for p in recent_data)
recent_cost = sum(p['bet_amount'] for p in recent_data)
recent_roi = (recent_profit / recent_cost * 100) if recent_cost > 0 else 0

print(f"命中次数: {recent_hits}/{len(recent_data)} = {recent_hit_rate*100:.2f}%")
print(f"总投注: {recent_cost:.2f}元")
print(f"总收益: {recent_profit:+.2f}元")
print(f"ROI: {recent_roi:+.2f}%")

# 显示最后10期作为样本
print(f"\n{'='*120}")
print(f"最近300期详情（显示最后10期作为样本）")
print(f"{'='*120}")
print(f"{'日期':<12} {'中奖号码':<8} {'预测号码':<50} {'倍数':<6} {'投注':<10} {'结果':<6} {'盈亏':<12} {'累计':<12}")
print("-" * 140)

# 显示最后10期
for period in result['history'][-10:]:
    pred_str = str(period.get('prediction', []))
    if len(pred_str) > 48:
        pred_str = pred_str[:45] + "..."
    
    date_str = period.get('date', f"第{period['period']}期")
    
    print(
        f"{str(date_str):<12} "
        f"{period.get('actual', 'N/A'):<8} "
        f"{pred_str:<50} "
        f"{period['multiplier']:<6} "
        f"{period['bet_amount']:<10.2f} "
        f"{period['result']:<6} "
        f"{period['profit']:>+12.2f} "
        f"{period['total_profit']:>12.2f}"
    )

print()
print("="*120)
print("✅ 测试完成！")
print("="*120)
print()
print("总结:")
print(f"  • 成功测试了{test_periods}期数据")
print(f"  • 历史记录包含{len(result['history'])}期详细数据")
if len(result['history']) >= 300:
    print(f"  • ✓ GUI中将显示最近300期的完整详情")
else:
    print(f"  • ⚠️ GUI中将显示最近{len(result['history'])}期的详情（数据不足300期）")
print(f"  • 命中率: {result['hit_rate']*100:.2f}%")
print(f"  • 总收益: {result['total_profit']:+.2f}元")
print(f"  • ROI: {result['roi']:+.2f}%")
