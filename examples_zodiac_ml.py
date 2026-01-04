"""
生肖ML预测模型 - 综合示例
展示所有主要功能的使用方法
"""

from zodiac_ml_predictor import ZodiacMLPredictor
import pandas as pd


def example_1_basic_usage():
    """示例1: 基础使用"""
    print("\n" + "="*80)
    print("示例1: 基础使用")
    print("="*80)
    
    # 创建预测器
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    
    # 获取预测
    result = predictor.predict()
    
    # 显示结果
    print(f"\nTOP6生肖: {[z for z, s in result['top6_zodiacs']]}")
    print(f"推荐号码: {result['top18_numbers'][:12]}")


def example_2_different_weights():
    """示例2: 对比不同权重"""
    print("\n" + "="*80)
    print("示例2: 对比不同权重配置")
    print("="*80)
    
    weights = {
        "纯统计": 0.0,
        "统计为主": 0.3,
        "平衡模式": 0.4,
        "ML为主": 0.6,
        "纯ML": 1.0
    }
    
    for name, weight in weights.items():
        predictor = ZodiacMLPredictor(ml_weight=weight)
        result = predictor.predict()
        top3 = [z for z, s in result['top6_zodiacs'][:3]]
        print(f"{name:8s} (ML={weight:.1f}): {top3}")


def example_3_detailed_info():
    """示例3: 获取详细信息"""
    print("\n" + "="*80)
    print("示例3: 获取详细预测信息")
    print("="*80)
    
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict()
    
    print(f"\n模型信息:")
    print(f"  模型名称: {result['model']}")
    print(f"  版本: {result['version']}")
    print(f"  ML状态: {result['ml_enabled']}")
    print(f"  权重配比: 统计{result['stat_weight']*100:.0f}% + ML{result['ml_weight']*100:.0f}%")
    
    print(f"\n最新一期:")
    print(f"  期数: {result['total_periods']}")
    print(f"  日期: {result['last_date']}")
    print(f"  开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\nTOP3生肖详细:")
    for i, (zodiac, final_score) in enumerate(result['top6_zodiacs'][:3], 1):
        stat_score = result['stat_scores'][zodiac]
        ml_prob = result['ml_probs'][zodiac] if result['ml_probs'] else 0
        
        print(f"  {i}. {zodiac}")
        print(f"     综合评分: {final_score:.2f}")
        print(f"     统计评分: {stat_score:.2f}")
        print(f"     ML概率: {ml_prob*100:.1f}%")
        print(f"     号码: {predictor.zodiac_numbers[zodiac]}")


def example_4_manual_training():
    """示例4: 手动训练模型"""
    print("\n" + "="*80)
    print("示例4: 手动训练和查看模型信息")
    print("="*80)
    
    predictor = ZodiacMLPredictor(ml_weight=0.5)
    
    # 显式训练
    print("\n开始训练...")
    predictor.train_models()
    
    # 查看模型信息
    print(f"\n训练完成:")
    print(f"  训练状态: {predictor.is_trained}")
    print(f"  模型数量: {len(predictor.models)}")
    print(f"  模型列表: {list(predictor.models.keys())}")
    
    # 预测
    result = predictor.predict()
    print(f"\n预测结果: {[z for z, s in result['top6_zodiacs'][:3]]}")


def example_5_comparison():
    """示例5: 统计vs混合模型对比"""
    print("\n" + "="*80)
    print("示例5: 纯统计 vs 混合模型对比")
    print("="*80)
    
    # 纯统计
    stat_predictor = ZodiacMLPredictor(ml_weight=0.0)
    stat_result = stat_predictor.predict()
    
    # 混合模型
    hybrid_predictor = ZodiacMLPredictor(ml_weight=0.4)
    hybrid_result = hybrid_predictor.predict()
    
    print("\n纯统计模型 TOP6:")
    for i, (z, s) in enumerate(stat_result['top6_zodiacs'], 1):
        print(f"  {i}. {z:2s} (评分: {s:6.2f})")
    
    print("\n混合模型 TOP6:")
    for i, (z, s) in enumerate(hybrid_result['top6_zodiacs'], 1):
        print(f"  {i}. {z:2s} (评分: {s:6.2f})")
    
    # 对比差异
    stat_top3 = set([z for z, s in stat_result['top6_zodiacs'][:3]])
    hybrid_top3 = set([z for z, s in hybrid_result['top6_zodiacs'][:3]])
    
    print(f"\nTOP3对比:")
    print(f"  相同: {stat_top3 & hybrid_top3}")
    print(f"  仅统计有: {stat_top3 - hybrid_top3}")
    print(f"  仅混合有: {hybrid_top3 - stat_top3}")


def example_6_number_recommendation():
    """示例6: 号码推荐详解"""
    print("\n" + "="*80)
    print("示例6: 号码推荐策略")
    print("="*80)
    
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict()
    
    top18 = result['top18_numbers']
    
    print("\n推荐号码分级:")
    print(f"  强推 (TOP 1-6):   {top18[0:6]}")
    print(f"  推荐 (TOP 7-12):  {top18[6:12]}")
    print(f"  备选 (TOP 13-18): {top18[12:18]}")
    
    print("\n选号策略建议:")
    print("  保守型: 选择 TOP2生肖 的号码")
    top2_nums = []
    for zodiac, _ in result['top6_zodiacs'][:2]:
        top2_nums.extend(predictor.zodiac_numbers[zodiac])
    print(f"    → {sorted(top2_nums)}")
    
    print("\n  平衡型: 选择 TOP3生肖 的号码 ⭐")
    top3_nums = []
    for zodiac, _ in result['top6_zodiacs'][:3]:
        top3_nums.extend(predictor.zodiac_numbers[zodiac])
    print(f"    → {sorted(top3_nums)}")
    
    print("\n  进取型: 选择 TOP12号码")
    print(f"    → {top18[:12]}")


def example_7_validation():
    """示例7: 简单验证"""
    print("\n" + "="*80)
    print("示例7: 最近5期简单验证")
    print("="*80)
    
    df = pd.read_csv('data/lucky_numbers.csv', encoding='utf-8-sig')
    total = len(df)
    
    correct_top3 = 0
    correct_top6 = 0
    
    print()
    for i in range(5):
        # 使用前N期数据
        train_df = df.iloc[:total-5+i]
        actual = df.iloc[total-5+i]['animal']
        
        # 保存并预测
        train_df.to_csv('data/temp_val.csv', index=False, encoding='utf-8-sig')
        
        predictor = ZodiacMLPredictor(ml_weight=0.4)
        result = predictor.predict(csv_file='data/temp_val.csv')
        
        top6 = [z for z, s in result['top6_zodiacs']]
        top3 = top6[:3]
        
        if actual in top3:
            correct_top3 += 1
        if actual in top6:
            correct_top6 += 1
        
        status = "✓" if actual in top6 else "✗"
        print(f"  期 {total-5+i+1}: 实际={actual:2s}  预测TOP3={top3}  {status}")
    
    print(f"\n验证结果:")
    print(f"  TOP3命中: {correct_top3}/5 = {correct_top3/5*100:.0f}%")
    print(f"  TOP6命中: {correct_top6}/5 = {correct_top6/5*100:.0f}%")


def example_8_all_scores():
    """示例8: 查看所有生肖评分"""
    print("\n" + "="*80)
    print("示例8: 所有生肖评分排名")
    print("="*80)
    
    predictor = ZodiacMLPredictor(ml_weight=0.4)
    result = predictor.predict()
    
    print("\n所有12生肖评分（从高到低）:")
    print(f"{'排名':<4} {'生肖':<4} {'综合评分':<10} {'统计评分':<10} {'ML概率':<10}")
    print("-" * 50)
    
    # 获取所有生肖的评分
    all_scores = result['all_scores']
    sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (zodiac, final_score) in enumerate(sorted_all, 1):
        stat_score = result['stat_scores'][zodiac]
        ml_prob = result['ml_probs'][zodiac] if result['ml_probs'] else 0
        
        marker = "⭐" if i <= 6 else ""
        print(f"{i:<4} {zodiac:<4} {final_score:<10.2f} {stat_score:<10.2f} {ml_prob*100:<9.1f}% {marker}")


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("生肖ML预测模型 - 综合示例展示")
    print("="*80)
    
    try:
        # 运行所有示例
        example_1_basic_usage()
        example_2_different_weights()
        example_3_detailed_info()
        example_4_manual_training()
        example_5_comparison()
        example_6_number_recommendation()
        example_7_validation()
        example_8_all_scores()
        
        print("\n" + "="*80)
        print("✅ 所有示例运行完成")
        print("="*80)
        print("\n💡 提示:")
        print("  - 可以单独运行某个示例，如: example_1_basic_usage()")
        print("  - 可以修改ML权重参数进行实验")
        print("  - 可以根据示例代码编写自己的预测逻辑")
        print()
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
