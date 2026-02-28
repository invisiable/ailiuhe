"""
快速测试动态择优功能的详细预测输出
"""

import pandas as pd
from ensemble_select_best_predictor import EnsembleSelectBestPredictor

# 读取数据
df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')

# 只测试最后10期
test_periods = 10
start_idx = len(df) - test_periods

predictor = EnsembleSelectBestPredictor(window_size=20)

print("="*105)
print("测试动态择优预测详细输出格式")
print("="*105)
print(f"\n{'期号':<6} {'日期':<12} {'实际':<8} {'预测TOP4':<30} {'使用模型':<25} {'结果':<6}")
print("-"*105)

predictions_top4 = []
actuals = []
hit_records = []
predictor_records = []

for i in range(start_idx, len(df)):
    train_animals = [str(a).strip() for a in df['animal'].iloc[:i].tolist()]
    
    # 预测
    result = predictor.predict_top4(train_animals)
    top4 = result['top4']
    predictor_name = result['predictor']
    
    predictions_top4.append(top4)
    predictor_records.append(predictor_name)
    
    # 实际
    actual = str(df.iloc[i]['animal']).strip()
    actuals.append(actual)
    
    # 命中
    hit = actual in top4
    hit_records.append(hit)
    
    # 更新性能
    details = result.get('details', {})
    predictor.update_performance(actual, details)
    
    # 格式化输出
    period_number = i - start_idx + 1
    date_str = df.iloc[i]['date']
    top4_str = ', '.join(top4)
    result_mark = "✓" if hit else "✗"
    
    print(f"{period_number:<6} {date_str:<12} {actual:<8} {top4_str:<30} {predictor_name:<25} {result_mark:<6}")

print("-"*105)
print(f"\n✅ 格式测试完成！")
print(f"命中数: {sum(hit_records)}/{len(hit_records)}")
print(f"各预测器使用次数:")
predictor_counts = {}
for pred in predictor_records:
    predictor_counts[pred] = predictor_counts.get(pred, 0) + 1
for pred, count in sorted(predictor_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {pred}: {count}次")
