"""
测试Top 3概率预测功能
"""

from lucky_number_predictor import LuckyNumberPredictor

print("=" * 70)
print("🎯 幸运数字 Top 3 概率预测测试")
print("=" * 70)

# 创建预测器
predictor = LuckyNumberPredictor()

# 加载数据
print("\n1. 加载训练数据...")
try:
    predictor.load_data('data/lucky_numbers.csv')
    print(f"✅ 数据加载成功！")
    print(f"   - 训练样本数: {len(predictor.X)}")
    print(f"   - 特征维度: {predictor.X.shape[1]}")
except Exception as e:
    print(f"❌ 失败: {e}")
    exit(1)

# 训练模型
print("\n2. 训练随机森林模型...")
try:
    results = predictor.train_model(model_type='random_forest', test_size=0.2)
    print(f"✅ 模型训练成功！")
    print(f"   - 测试集MAE: {results['test_mae']:.4f}")
    print(f"   - 测试集R²: {results['test_r2']:.4f}")
except Exception as e:
    print(f"❌ 失败: {e}")
    exit(1)

# Top 3 概率预测
print("\n3. 预测下一期最可能的幸运数字 (Top 3)...")
try:
    top_predictions = predictor.predict_top_probabilities(top_k=3)
    print(f"✅ 预测成功！\n")
    
    print("┌─────────────────────────────────────────────────────────┐")
    print("│              🎲 Top 3 最可能的幸运数字                 │")
    print("├─────────────────────────────────────────────────────────┤")
    
    for i, pred in enumerate(top_predictions, 1):
        prob_percent = pred['probability'] * 100
        bar_length = int(prob_percent / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        
        print(f"│ 第 {i} 名:                                                │")
        print(f"│   数字: {pred['number']:>2}   生肖: {pred['animal']}   五行: {pred['element']}                     │")
        print(f"│   概率: {prob_percent:>6.2f}%                                        │")
        print(f"│   {bar} │")
        print("├─────────────────────────────────────────────────────────┤")
    
    print("└─────────────────────────────────────────────────────────┘")
    
    # 显示历史数据参考
    print(f"\n📊 基于历史数据:")
    print(f"   最近10期: {list(predictor.raw_numbers[-10:])}")
    print(f"   平均值: {sum(predictor.raw_numbers[-10:])/10:.2f}")
    
except Exception as e:
    print(f"❌ 预测失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
print("\n💡 提示: 运行 'python main.py' 在图形界面中查看更丰富的展示")
