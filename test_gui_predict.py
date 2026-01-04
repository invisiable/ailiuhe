"""
测试GUI预测功能（无界面模式）
"""

from lucky_number_predictor import LuckyNumberPredictor

print("="*60)
print("测试GUI预测流程")
print("="*60)

# 模拟GUI加载数据过程
predictor = LuckyNumberPredictor()

print("\n1. 加载数据...")
try:
    predictor.load_data(
        'data/lucky_numbers.csv',
        number_column='number',
        date_column='date',
        animal_column='animal',
        element_column='element'
    )
    print(f"✅ 加载成功: {len(predictor.X)}个训练样本")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit(1)

print("\n2. 训练模型...")
try:
    results = predictor.train_model('random_forest', test_size=0.2)
    print(f"✅ 训练成功: MAE={results['test_mae']:.2f}")
except Exception as e:
    print(f"❌ 训练失败: {e}")
    exit(1)

print("\n3. 预测未来数字...")
try:
    predictions = predictor.predict_next(n_predictions=5)
    print(f"✅ 预测成功!\n")
    
    # 模拟GUI显示格式
    result_parts = []
    for pred in predictions:
        if isinstance(pred, dict):
            result_parts.append(f"【{pred['number']}】{pred['animal']}/{pred['element']}")
        else:
            result_parts.append(f"【{int(pred)}】")
    
    result_text = "  ".join(result_parts)
    print(f"显示结果: {result_text}\n")
    
    # 详细日志
    print(f"{'期数':<6} {'幸运数字':<10} {'生肖':<6} {'五行':<6}")
    print("-" * 35)
    for i, pred in enumerate(predictions, 1):
        if isinstance(pred, dict):
            print(f"第{i}期   {pred['number']:<10} {pred['animal']:<6} {pred['element']:<6}")
        else:
            print(f"第{i}期   {int(pred):<10}")
    
except Exception as e:
    print(f"❌ 预测失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✅ 所有测试通过！GUI应该可以正常使用了")
print("="*60)
print("\n现在可以运行: python main.py")
