"""
测试生肖机器学习预测模型
"""

from zodiac_ml_predictor import ZodiacMLPredictor
import pandas as pd


def test_basic_prediction():
    """测试基本预测功能"""
    print("\n" + "="*80)
    print("测试1: 基本预测功能")
    print("="*80)
    
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict()
    
    print(f"✓ 预测完成")
    print(f"  模型: {result['model']}")
    print(f"  ML状态: {result['ml_enabled']}")
    print(f"  TOP6生肖: {[z for z, s in result['top6_zodiacs']]}")
    print(f"  TOP6号码: {result['top18_numbers'][:6]}")
    
    return result


def test_different_weights():
    """测试不同权重配比"""
    print("\n" + "="*80)
    print("测试2: 不同权重配比")
    print("="*80)
    
    weights = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for w in weights:
        predictor = ZodiacMLPredictor(ml_weight=w)
        result = predictor.predict()
        
        top3 = [z for z, s in result['top6_zodiacs'][:3]]
        print(f"  ML权重={w:.1f}: TOP3={top3}")


def test_model_training():
    """测试模型训练"""
    print("\n" + "="*80)
    print("测试3: 模型训练过程")
    print("="*80)
    
    predictor = ZodiacMLPredictor(ml_weight=0.5)
    
    # 显式训练
    predictor.train_models()
    
    print(f"✓ 训练完成")
    print(f"  模型数量: {len(predictor.models)}")
    print(f"  模型列表: {list(predictor.models.keys())}")
    
    # 预测
    result = predictor.predict()
    print(f"  ML预测概率示例: {list(result['ml_probs'].items())[:3]}")


def test_validation():
    """简单验证测试"""
    print("\n" + "="*80)
    print("测试4: 简单验证（最近10期）")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    
    # 使用倒数第11-20期训练，测试倒数第1-10期
    total = len(df)
    test_periods = 10
    
    correct_top3 = 0
    correct_top6 = 0
    
    for i in range(test_periods):
        # 截取到倒数第i+1期为止的数据
        test_df = df.iloc[:total-test_periods+i]
        actual = df.iloc[total-test_periods+i]['animal']
        
        # 保存临时数据
        test_df.to_csv('data/temp_zodiac_test.csv', index=False, encoding='utf-8-sig')
        
        # 预测
        predictor = ZodiacMLPredictor(ml_weight=0.4)
        result = predictor.predict(csv_file='data/temp_zodiac_test.csv')
        
        top6_zodiacs = [z for z, s in result['top6_zodiacs']]
        top3_zodiacs = top6_zodiacs[:3]
        
        if actual in top3_zodiacs:
            correct_top3 += 1
        if actual in top6_zodiacs:
            correct_top6 += 1
        
        status = "✓" if actual in top6_zodiacs else "✗"
        print(f"  期数 {total-test_periods+i+1}: 实际={actual:2s}  TOP3={top3_zodiacs}  {status}")
    
    print(f"\n验证结果:")
    print(f"  TOP3命中率: {correct_top3}/{test_periods} = {correct_top3/test_periods*100:.1f}%")
    print(f"  TOP6命中率: {correct_top6}/{test_periods} = {correct_top6/test_periods*100:.1f}%")
    print(f"  理论TOP3: 25.0%  理论TOP6: 50.0%")


def compare_with_pure_statistical():
    """对比纯统计模型"""
    print("\n" + "="*80)
    print("测试5: 对比纯统计 vs 混合模型")
    print("="*80)
    
    # 纯统计（ML权重=0）
    predictor_stat = ZodiacMLPredictor(ml_weight=0.0)
    result_stat = predictor_stat.predict()
    
    # 混合模型（ML权重=0.4）
    predictor_hybrid = ZodiacMLPredictor(ml_weight=0.4)
    result_hybrid = predictor_hybrid.predict()
    
    print("纯统计模型 TOP6:")
    for i, (z, s) in enumerate(result_stat['top6_zodiacs'], 1):
        print(f"  {i}. {z} (评分: {s:.2f})")
    
    print("\n混合模型 TOP6:")
    for i, (z, s) in enumerate(result_hybrid['top6_zodiacs'], 1):
        print(f"  {i}. {z} (评分: {s:.2f})")
    
    # 对比
    stat_top3 = set([z for z, s in result_stat['top6_zodiacs'][:3]])
    hybrid_top3 = set([z for z, s in result_hybrid['top6_zodiacs'][:3]])
    
    print(f"\nTOP3差异:")
    print(f"  相同: {stat_top3 & hybrid_top3}")
    print(f"  仅统计: {stat_top3 - hybrid_top3}")
    print(f"  仅混合: {hybrid_top3 - stat_top3}")


if __name__ == "__main__":
    print("\n🤖 生肖机器学习预测模型 - 测试套件")
    
    # 运行所有测试
    try:
        test_basic_prediction()
        test_different_weights()
        test_model_training()
        compare_with_pure_statistical()
        test_validation()
        
        print("\n" + "="*80)
        print("✓ 所有测试完成")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
