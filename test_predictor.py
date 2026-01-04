"""
测试幸运数字预测模块
"""

from lucky_number_predictor import LuckyNumberPredictor

print("=" * 60)
print("幸运数字预测系统测试")
print("=" * 60)

# 创建预测器
predictor = LuckyNumberPredictor()

# 加载数据
print("\n1. 加载训练数据...")
try:
    predictor.load_data('data/lucky_numbers.csv')
    print(f"✅ 数据加载成功！")
    print(f"   - 训练样本数: {len(predictor.X)}")
    print(f"   - 特征维度: {predictor.X.shape[1]}")
    print(f"   - 序列长度: {predictor.sequence_length}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit(1)

# 训练模型
print("\n2. 训练预测模型...")
try:
    results = predictor.train_model(model_type='random_forest', test_size=0.2)
    print(f"✅ 模型训练成功！")
    print(f"   - 模型类型: {results['model_type']}")
    print(f"   - 训练样本: {results['train_samples']}")
    print(f"   - 测试样本: {results['test_samples']}")
    print(f"   - 测试集MAE: {results['test_mae']:.4f}")
    print(f"   - 测试集RMSE: {results['test_rmse']:.4f}")
    print(f"   - 测试集R²: {results['test_r2']:.4f}")
except Exception as e:
    print(f"❌ 模型训练失败: {e}")
    exit(1)

# 预测未来数字
print("\n3. 预测未来幸运数字...")
try:
    predictions = predictor.predict_next(n_predictions=5)
    print(f"✅ 预测成功！未来5期预测结果：\n")
    print(f"{'期数':<6} {'幸运数字':<10} {'生肖':<6} {'五行':<6}")
    print("-" * 35)
    for i, pred in enumerate(predictions, 1):
        print(f"第{i}期   {pred['number']:<10} {pred['animal']:<6} {pred['element']:<6}")
except Exception as e:
    print(f"❌ 预测失败: {e}")
    exit(1)

# 显示特征重要性
print("\n4. 特征重要性分析...")
importance = predictor.get_feature_importance()
if importance:
    print("✅ Top 10 重要特征：")
    sorted_importance = sorted(importance, key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in sorted_importance:
        print(f"   {feat:<15}: {imp:.4f}")
else:
    print("⚠️  当前模型不支持特征重要性分析")

# 保存模型
print("\n5. 保存模型...")
try:
    filepath = predictor.save_model()
    print(f"✅ 模型已保存: {filepath}")
except Exception as e:
    print(f"❌ 保存模型失败: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
