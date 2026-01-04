"""
测试模型的保存和加载功能
"""
from lucky_number_predictor import LuckyNumberPredictor

print("="*70)
print("测试模型保存和加载")
print("="*70)

# 1. 训练一个新模型
print("\n1. 训练新模型...")
predictor = LuckyNumberPredictor()
predictor.load_data(
    'data/lucky_numbers.csv',
    number_column='number',
    date_column='date',
    animal_column='animal',
    element_column='element'
)

train_results = predictor.train_model('random_forest', test_size=0.2)
print(f"✓ 训练完成 - MAE: {train_results['test_mae']:.4f}")

# 测试预测
pred1 = predictor.predict_next(1)[0]
print(f"✓ 预测测试: 数字{pred1['number']}, 生肖{pred1['animal']}, 五行{pred1['element']}")

# 2. 保存模型
print("\n2. 保存模型...")
filepath = predictor.save_model('models')
print(f"✓ 模型已保存到: {filepath}")

# 3. 创建新的预测器并加载模型
print("\n3. 加载模型...")
predictor2 = LuckyNumberPredictor()
predictor2.load_model(filepath)
print(f"✓ 模型加载成功")
print(f"  - 模型类型: {predictor2.model_type}")
print(f"  - 历史数据: {len(predictor2.raw_numbers)} 个")
print(f"  - 序列长度: {predictor2.sequence_length}")
print(f"  - 特征数量: {len(predictor2.feature_names)}")
print(f"  - 生肖数据: {'有' if predictor2.raw_animals is not None else '无'}")
print(f"  - 五行数据: {'有' if predictor2.raw_elements is not None else '无'}")
print(f"  - 五行映射: {'有' if predictor2.number_to_element else '无'}")

# 4. 使用加载的模型进行预测
print("\n4. 使用加载的模型预测...")
pred2 = predictor2.predict_next(1)[0]
print(f"✓ 预测结果: 数字{pred2['number']}, 生肖{pred2['animal']}, 五行{pred2['element']}")

# 5. 对比两次预测结果
print("\n5. 对比预测结果...")
if pred1['number'] == pred2['number'] and pred1['animal'] == pred2['animal'] and pred1['element'] == pred2['element']:
    print("✓ 预测结果一致！模型保存加载功能正常")
else:
    print("✗ 预测结果不一致")
    print(f"  原始: {pred1}")
    print(f"  加载: {pred2}")

# 6. 测试Top 3预测
print("\n6. 测试Top 3预测...")
try:
    top3 = predictor2.predict_top_probabilities(top_k=3)
    print("✓ Top 3预测成功:")
    for i, p in enumerate(top3, 1):
        print(f"  {i}. 数字{p['number']:2d} 生肖{p['animal']} 五行{p['element']} 概率{p['probability']*100:.2f}%")
except Exception as e:
    print(f"✗ Top 3预测失败: {e}")

# 7. 测试分别预测
print("\n7. 测试分别预测...")
try:
    sep_pred = predictor2.predict_separately(top_k=3)
    print("✓ 分别预测成功:")
    print("  数字Top 3:", [f"{p['value']}({p['probability']*100:.1f}%)" for p in sep_pred['numbers']])
    print("  生肖Top 3:", [f"{p['value']}({p['probability']*100:.1f}%)" for p in sep_pred['animals']])
    print("  五行Top 3:", [f"{p['value']}({p['probability']*100:.1f}%)" for p in sep_pred['elements']])
except Exception as e:
    print(f"✗ 分别预测失败: {e}")

print("\n" + "="*70)
print("测试完成！")
print("="*70)
