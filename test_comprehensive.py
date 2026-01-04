"""
测试综合预测功能
"""
import sys
sys.path.insert(0, 'd:\\AIagent')

try:
    print("=" * 70)
    print("测试综合预测功能")
    print("=" * 70)
    
    print("\n1. 导入模块...")
    from enhanced_predictor_v2 import EnhancedPredictor
    print("✓ 导入成功")
    
    print("\n2. 创建并训练3个预测器...")
    from lucky_number_predictor import LuckyNumberPredictor
    
    data_file = 'data/lucky_numbers.csv'
    predictors = []
    
    for model_type in ['gradient_boosting', 'lightgbm', 'xgboost']:
        print(f"   - 训练 {model_type}...")
        pred = LuckyNumberPredictor()
        pred.load_data(data_file, 'number', 'date', 'animal', 'element')
        pred.train_model(model_type, test_size=0.2)
        predictors.append(pred)
    
    print(f"✓ 3个模型训练完成，共 {len(predictors[0].raw_numbers)} 条记录")
    
    print("\n3. 创建增强预测器...")
    enhanced = EnhancedPredictor(predictors)
    print("✓ 创建成功")
    
    print("\n4. 准备预测...")
    
    print("\n5. 执行综合预测（可能需要几秒钟）...")
    predictions = enhanced.comprehensive_predict_v2(top_k=10)
    print("✓ 预测完成")
    
    print("\n" + "=" * 70)
    print("综合预测 Top 10 结果")
    print("=" * 70)
    print(f"{'排名':<6} {'数字':<6} {'综合概率':<12} {'概率条'}")
    print("-" * 70)
    for i, pred in enumerate(predictions[:10], 1):
        prob = pred['probability']
        bar = '█' * int(prob * 50)
        print(f"{i:>2}.    {pred['number']:>2}     {prob:>6.4f}       {bar}")
    
    print("\n" + "=" * 70)
    print("✅ 测试通过！综合预测功能正常工作")
    print("=" * 70)
    
except ImportError as e:
    print(f"\n❌ 导入错误: {e}")
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ 运行错误: {e}")
    import traceback
    traceback.print_exc()
