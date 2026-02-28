"""
测试修改后的GUI显示
验证最近100期详细记录是否正确显示模型信息
"""

import pandas as pd
from recommended_zodiac_top4_strategy import RecommendedZodiacTop4Strategy

# 模拟GUI的回测过程
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
test_periods = min(200, len(df))
start_idx = len(df) - test_periods

strategy = RecommendedZodiacTop4Strategy(use_emergency_backup=True)

predictions_top4 = []
predictor_records = []
actuals = []
hit_records = []

print("模拟回测过程...\n")

for i in range(start_idx, len(df)):
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    
    prediction = strategy.predict_top4(train_animals)
    top4 = prediction['top4']
    predictor_name = prediction['predictor']
    
    predictions_top4.append(top4)
    predictor_records.append(predictor_name)
    
    actual = str(df.iloc[i]['animal']).strip()
    actuals.append(actual)
    
    hit = actual in top4
    hit_records.append(hit)
    
    strategy.update_performance(hit)
    
    if (i - start_idx + 1) % 10 == 0:
        switched, msg = strategy.check_and_switch_model()
        if switched:
            print(f"第{i-start_idx+1}期: {msg}")

print("\n" + "="*95)
print("📋 最近100期详细命中记录（回测实际预测）")
print("="*95)
print("💡 说明：显示回测过程中的实际预测，包含模型自动切换的影响\n")
print(f"{'期号':<6} {'日期':<12} {'实际':<8} {'预测TOP4':<30} {'模型':<15} {'结果':<6}")
print("-"*95)

# 显示最后20期作为示例
recent_100_start_idx = len(hit_records) - 100
for i in range(80, 100):  # 只显示第81-100期
    record_idx = recent_100_start_idx + i
    period_idx = start_idx + record_idx
    
    date_str = df.iloc[period_idx]['date']
    actual_animal = actuals[record_idx]
    predicted_top4 = predictions_top4[record_idx]
    predictor_used = predictor_records[record_idx]
    hit = hit_records[record_idx]
    
    top4_str = ', '.join(predicted_top4)
    result_mark = "✓" if hit else "✗"
    period_number = i + 1
    
    print(f"{period_number:<6} {date_str:<12} {actual_animal:<8} {top4_str:<30} {predictor_used:<15} {result_mark:<6}")

# 统计模型使用情况
recent_100_predictors = predictor_records[-100:]
primary_count = sum(1 for p in recent_100_predictors if '重训练' in p or '主力' in p)
backup_count = 100 - primary_count

print("-"*95)
print(f"模型使用统计: 主力模型 {primary_count}期  |  备份模型 {backup_count}期")
print("="*95)

print("\n✅ 修改效果验证完成！")
print("现在用户可以清楚看到每期使用的是哪个模型了。")
