"""
奇偶性预测快速使用示例
"""

from odd_even_predictor import OddEvenPredictor


def predict_next_odd_even():
    """预测下一期数字的奇偶性"""
    print("=" * 80)
    print("幸运数字奇偶性预测")
    print("=" * 80)
    
    # 创建预测器
    predictor = OddEvenPredictor()
    
    # 1. 显示历史统计
    print("\n📊 历史奇偶统计:")
    print("-" * 80)
    stats = predictor.get_statistics('data/lucky_numbers.csv')
    
    print(f"总期数: {stats['total_count']}")
    print(f"奇数: {stats['odd_count']} 期 ({stats['odd_ratio']*100:.2f}%)")
    print(f"偶数: {stats['even_count']} 期 ({stats['even_ratio']*100:.2f}%)")
    print(f"最长连续奇数: {stats['max_odd_streak']} 期")
    print(f"最长连续偶数: {stats['max_even_streak']} 期")
    
    print(f"\n最近5期数字: {' -> '.join(map(str, stats['last_5_numbers']))}")
    print(f"奇偶分布:     {' -> '.join(stats['last_5_odd_even'])}")
    
    # 显示最近N期统计
    for key, value in stats['recent_stats'].items():
        n = key.split('_')[1]
        print(f"\n最近{n}期: 奇数 {value['odd_count']} ({value['odd_ratio']*100:.1f}%), "
              f"偶数 {value['even_count']} ({value['even_ratio']*100:.1f}%)")
    
    # 2. 训练并预测
    print("\n" + "=" * 80)
    print("🔮 训练模型并预测下一期...")
    print("-" * 80)
    
    predictor.train_model('data/lucky_numbers.csv', 
                         model_type='gradient_boosting',
                         test_size=0.2)
    
    # 3. 进行预测
    print("\n" + "=" * 80)
    print("🎯 预测结果:")
    print("-" * 80)
    
    prediction = predictor.predict()
    
    print(f"\n预测下一期数字为: 【{prediction['prediction']}】")
    print(f"置信度: {prediction['confidence']*100:.2f}%")
    print(f"\n详细概率:")
    print(f"  奇数概率: {prediction['odd_probability']*100:.2f}%")
    print(f"  偶数概率: {prediction['even_probability']*100:.2f}%")
    
    # 4. 建议
    print("\n" + "=" * 80)
    print("💡 建议:")
    print("-" * 80)
    
    if prediction['confidence'] >= 0.8:
        confidence_level = "高"
        advice = "模型对预测结果非常有信心"
    elif prediction['confidence'] >= 0.6:
        confidence_level = "中"
        advice = "模型对预测结果较有信心"
    else:
        confidence_level = "低"
        advice = "建议谨慎参考，可能需要结合其他因素"
    
    print(f"置信度等级: {confidence_level}")
    print(f"使用建议: {advice}")
    
    # 根据历史趋势给出额外建议
    last_5 = stats['last_5_odd_even']
    odd_count_last_5 = last_5.count('奇')
    even_count_last_5 = last_5.count('偶')
    
    print(f"\n趋势分析:")
    if odd_count_last_5 >= 4:
        print(f"  ⚠️ 最近5期已连续出现{odd_count_last_5}次奇数，可能会回调")
    elif even_count_last_5 >= 4:
        print(f"  ⚠️ 最近5期已连续出现{even_count_last_5}次偶数，可能会回调")
    else:
        print(f"  ✓ 最近5期奇偶分布较为均衡")
    
    print("\n" + "=" * 80)


def load_and_predict_with_saved_model(model_file):
    """使用已保存的模型进行预测"""
    print("=" * 80)
    print("使用已保存的模型预测")
    print("=" * 80)
    
    predictor = OddEvenPredictor()
    predictor.load_model(model_file)
    predictor.load_data('data/lucky_numbers.csv')
    
    prediction = predictor.predict()
    
    print(f"\n预测结果: {prediction['prediction']}")
    print(f"置信度: {prediction['confidence']*100:.2f}%")
    print(f"奇数概率: {prediction['odd_probability']*100:.2f}%")
    print(f"偶数概率: {prediction['even_probability']*100:.2f}%")


if __name__ == "__main__":
    # 方式1: 训练新模型并预测
    predict_next_odd_even()
    
    # 方式2: 使用已保存的模型（如果有）
    # model_file = 'models/OddEven_gradient_boosting_20251215_092859.joblib'
    # load_and_predict_with_saved_model(model_file)
