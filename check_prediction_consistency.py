"""
检查生肖TOP4预测记录一致性
对比GUI中两个地方的预测是否一致
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
print(f"总数据期数: {len(df)}")
print(f"最新一期: {df.iloc[-1]['date']} - {df.iloc[-1]['animal']}")
print(f"{'='*80}\n")

# 模拟GUI中的回测过程
test_periods = min(200, len(df))
start_idx = len(df) - test_periods
print(f"回测设置: 测试{test_periods}期，从第{start_idx}期开始")
print(f"回测范围: {df.iloc[start_idx]['date']} ~ {df.iloc[-1]['date']}\n")

# 创建策略实例（与GUI相同）
strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)

# 回测过程（与GUI相同）
predictions_top4 = []
hit_records = []

print("开始回测...")
for i in range(start_idx, len(df)):
    # 使用i之前的数据进行预测
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    
    # 预测
    prediction = strategy.predict_top4(train_animals)
    top4 = prediction['top4']
    predictions_top4.append(top4)
    
    # 实际结果
    actual = str(df.iloc[i]['animal']).strip()
    hit = actual in top4
    hit_records.append(hit)
    
    # 更新性能
    strategy.update_performance(hit)
    if (i - start_idx + 1) % 10 == 0:
        strategy.check_and_switch_model()

print(f"回测完成！共{len(predictions_top4)}期\n")
print(f"{'='*80}\n")

# 检查最近100期
if len(predictions_top4) >= 100:
    print("检查最近100期预测记录...\n")
    
    recent_100_start_idx = len(hit_records) - 100
    
    # 显示最后5期的详细信息（作为样本）
    print("【最近100期的最后5期详细信息】")
    print(f"{'期号':<6} {'日期':<12} {'用于训练的期数':<15} {'预测TOP4':<35} {'实际':<8}")
    print(f"{'-'*90}")
    
    for i in range(95, 100):  # 最后5期
        record_idx = recent_100_start_idx + i
        period_idx = start_idx + record_idx
        
        date_str = df.iloc[period_idx]['date']
        predicted_top4 = predictions_top4[record_idx]
        actual_animal = df.iloc[period_idx]['animal']
        
        # 计算这一期使用了多少期数据
        train_data_count = period_idx
        
        top4_str = ', '.join(predicted_top4)
        print(f"{i+1:<6} {date_str:<12} {train_data_count:<15} {top4_str:<35} {actual_animal:<8}")
    
    print(f"\n{'='*80}\n")

# 预测下一期（与GUI相同）
print("【下期投注建议的预测】")
all_animals = [str(a).strip() for a in df['animal'].tolist()]
prediction_result = strategy.predict_top4(all_animals)
next_top4 = prediction_result['top4']
current_model_name = strategy.get_current_model_name()

print(f"用于训练的期数: {len(all_animals)}")
print(f"预测TOP4: {', '.join(next_top4)}")
print(f"当前模型: {current_model_name}")
print(f"当前模型状态: {strategy.current_model}")

print(f"\n{'='*80}\n")

# 关键对比：最近100期的最后一期 vs 如果重新用全部数据预测最后一期
print("【关键对比】\n")

# 最后一期的实际数据
last_period_idx = len(df) - 1
last_date = df.iloc[last_period_idx]['date']
last_actual = df.iloc[last_period_idx]['animal']

# 回测中最后一期的预测（用前387期预测第388期）
backtest_last_prediction = predictions_top4[-1]
print(f"1. 回测中的最后一期预测:")
print(f"   期数: 第{test_periods}期")
print(f"   日期: {last_date}")
print(f"   训练数据: 前{last_period_idx}期 (不包括自己)")
print(f"   预测TOP4: {', '.join(backtest_last_prediction)}")
print(f"   实际生肖: {last_actual}")
print(f"   是否命中: {'✓' if last_actual in backtest_last_prediction else '✗'}\n")

# 用全部数据重新预测这一期（这是不合理的，因为包括了自己）
# 但如果GUI有bug，可能就是这样做的
print(f"2. 如果用全部数据重新预测最后一期（包括自己）:")
# 创建新的strategy实例，避免状态污染
fresh_strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)
all_animals_including_last = [str(a).strip() for a in df['animal'].tolist()]
fresh_prediction = fresh_strategy.predict_top4(all_animals_including_last)
print(f"   训练数据: 全部{len(all_animals_including_last)}期 (包括自己)")
print(f"   预测TOP4: {', '.join(fresh_prediction['top4'])}")
print(f"   注意: 这个预测是用来预测下一期(第{len(df)+1}期)的，不是预测当期\n")

print(f"3. 下期投注建议中的预测:")
print(f"   预测的是: 第{len(df)+1}期（未来）")
print(f"   训练数据: {len(all_animals)}期")
print(f"   预测TOP4: {', '.join(next_top4)}")

print(f"\n{'='*80}\n")

# 结论
print("【分析结论】\n")
print("正常情况下：")
print("✓ 回测最后一期的预测 ≠ 下期投注建议的预测（因为预测的目标不同）")
print("✓ 回测第i期：用前i-1期数据预测第i期")
print("✓ 下期建议：用全部数据预测下一期（未来）\n")

if backtest_last_prediction != next_top4:
    print("✅ 两个预测不相同是正常的！它们预测的目标不同。")
else:
    print("⚠️ 两个预测相同，这可能表示模型有问题或者数据不足。")

print(f"\n如果GUI显示不一致，可能的原因：")
print("1. 最近100期列表显示的是回测预测（正确）")
print("2. 但某处可能误用了'下期预测'的结果")
print("3. 或者最近100期列表中某些预测没有正确记录")
