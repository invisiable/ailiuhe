"""
快速测试脚本 - 验证幸运数字预测功能
"""

from lucky_number_predictor import LuckyNumberPredictor
import os

def test_lucky_number_prediction():
    """测试幸运数字预测功能"""
    
    print("="*70)
    print("🎲 幸运数字预测系统 - 快速测试")
    print("="*70)
    
    # 创建预测器
    predictor = LuckyNumberPredictor()
    
    # 测试数据文件
    data_file = os.path.join('data', 'lucky_numbers.csv')
    
    print(f"\n1️⃣ 加载测试数据: {data_file}")
    try:
        predictor.load_data(data_file, 'number', 'date')
        print(f"   ✅ 数据加载成功")
        print(f"   📊 历史数据点: {len(predictor.raw_numbers)}")
        print(f"   📊 训练样本: {len(predictor.X)}")
        print(f"   📊 特征维度: {len(predictor.feature_names)}")
        print(f"   📊 最近10个: {list(predictor.raw_numbers[-10:])}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    print(f"\n2️⃣ 训练随机森林模型")
    try:
        results = predictor.train_model('random_forest', test_size=0.2)
        print(f"   ✅ 训练完成")
        print(f"   📈 模型类型: {results['model_type']}")
        print(f"   📈 训练样本: {results['train_samples']}")
        print(f"   📈 测试样本: {results['test_samples']}")
        print(f"   📈 测试集MAE: {results['test_mae']:.4f}")
        print(f"   📈 测试集RMSE: {results['test_rmse']:.4f}")
        print(f"   📈 测试集R²: {results['test_r2']:.4f}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    print(f"\n3️⃣ 预测未来5个幸运数字")
    try:
        predictions = predictor.predict_next(5)
        print(f"   ✅ 预测完成")
        print(f"   🔮 预测结果: {[int(p) for p in predictions]}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    print(f"\n4️⃣ 保存模型")
    try:
        filepath = predictor.save_model()
        print(f"   ✅ 模型已保存")
        print(f"   💾 文件路径: {filepath}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    print(f"\n5️⃣ 测试加载模型")
    try:
        new_predictor = LuckyNumberPredictor()
        new_predictor.load_model(filepath)
        print(f"   ✅ 模型加载成功")
        
        # 再次预测验证
        new_predictions = new_predictor.predict_next(3)
        print(f"   🔮 验证预测: {[int(p) for p in new_predictions]}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return
    
    print(f"\n{'='*70}")
    print(f"✅ 所有测试通过！系统运行正常。")
    print(f"{'='*70}")
    print(f"\n💡 提示: 运行 'python main.py' 启动图形界面")
    print(f"📖 详细说明请查看 '使用指南.md'")
    print()

if __name__ == "__main__":
    test_lucky_number_prediction()
