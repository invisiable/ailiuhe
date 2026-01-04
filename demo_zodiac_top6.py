"""
生肖TOP6预测模型 - 使用演示
展示如何使用生肖TOP6模型进行预测
"""

from zodiac_top6_predictor import ZodiacTop6Predictor


def demo_simple_usage():
    """演示1: 简单使用"""
    print("=" * 80)
    print("演示1: 快速预测")
    print("=" * 80)
    
    # 创建预测器
    predictor = ZodiacTop6Predictor()
    
    # 获取预测结果
    result = predictor.predict()
    
    # 显示预测
    print(f"\n📅 最新一期: 第{result['total_periods']}期")
    print(f"   开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n🔮 下一期预测（第{result['total_periods']+1}期）:")
    print(f"\n   推荐生肖TOP6: ", end="")
    print([z for z, s in result['top6_zodiacs']])
    
    print(f"\n   推荐号码TOP18: ", end="")
    print(result['top18_numbers'][:18])
    
    print("\n" + "=" * 80 + "\n")


def demo_detailed_output():
    """演示2: 详细输出"""
    print("=" * 80)
    print("演示2: 详细预测信息")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    print(f"\n📊 模型信息:")
    print(f"   名称: {result['model_info']['name']}")
    print(f"   版本: {result['model_info']['version']}")
    print(f"   描述: {result['model_info']['description']}")
    
    print(f"\n📅 历史数据:")
    print(f"   总期数: {result['total_periods']}")
    print(f"   最新日期: {result['last_date']}")
    print(f"   最新结果: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n🎯 生肖预测 TOP6:")
    print("-" * 80)
    print(f"{'排名':<6} {'生肖':<6} {'评分':<10} {'对应号码':<30}")
    print("-" * 80)
    
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
        numbers = predictor.zodiac_numbers[zodiac]
        emoji = "⭐⭐" if i <= 2 else "⭐" if i <= 4 else "✓"
        print(f"{emoji} {i:<4} {zodiac:<6} {score:>6.2f}     {str(numbers):<30}")
    
    print(f"\n📋 号码推荐（基于TOP6生肖权重排序）:")
    print("-" * 80)
    top18 = result['top18_numbers']
    print(f"   TOP  1-6:  {top18[0:6]}")
    print(f"   TOP  7-12: {top18[6:12]}")
    print(f"   TOP 13-18: {top18[12:18]}")
    
    print("\n" + "=" * 80 + "\n")


def demo_validation_results():
    """演示3: 验证结果"""
    print("=" * 80)
    print("演示3: 模型验证（最近20期）")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    validation = predictor.validate(test_periods=20)
    
    print(f"\n📊 总体统计:")
    print(f"   测试期数: {validation['test_periods']}")
    
    print(f"\n   生肖TOP6:")
    print(f"      命中次数: {validation['zodiac_top6_hits']}")
    print(f"      命中率: {validation['zodiac_top6_rate']:.1f}%")
    print(f"      理论值: 50.0% (6/12)")
    
    zodiac_diff = validation['zodiac_top6_rate'] - 50.0
    if zodiac_diff > 0:
        print(f"      性能: ⬆️ 优于理论 {zodiac_diff:.1f}%")
    else:
        print(f"      性能: ⬇️ 低于理论 {abs(zodiac_diff):.1f}%")
    
    print(f"\n   号码TOP18:")
    print(f"      命中次数: {validation['number_top18_hits']}")
    print(f"      命中率: {validation['number_top18_rate']:.1f}%")
    print(f"      理论值: 36.7% (18/49)")
    
    number_diff = validation['number_top18_rate'] - 36.7
    if number_diff > 0:
        print(f"      性能: ⬆️ 优于理论 {number_diff:.1f}%")
    else:
        print(f"      性能: ⬇️ 低于理论 {abs(number_diff):.1f}%")
    
    # 显示详细验证结果（最近5期）
    print(f"\n📋 详细验证结果（最近5期）:")
    print("-" * 80)
    
    for detail in validation['details'][-5:]:
        print(f"\n第{detail['期号']}期 ({detail['日期']}):")
        print(f"   实际开出: {detail['实际号码']} - {detail['实际生肖']}")
        print(f"   预测生肖: {detail['预测生肖TOP6'][:3]} ... (共6个)")
        print(f"   预测号码: {detail['预测号码TOP18'][:6]} ... (共18个)")
        
        result_text = []
        if detail['生肖命中'] == '✓':
            result_text.append("✅ 生肖命中")
        else:
            result_text.append("❌ 生肖未中")
        
        if detail['号码命中'] == '✓':
            result_text.append("✅ 号码命中")
        else:
            result_text.append("❌ 号码未中")
        
        print(f"   结果: {' | '.join(result_text)}")
    
    print("\n" + "=" * 80 + "\n")


def demo_strategy_recommendation():
    """演示4: 策略建议"""
    print("=" * 80)
    print("演示4: 使用策略建议")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    print("\n💡 推荐使用策略:\n")
    
    # 策略1: 保守型
    print("【策略1 - 保守型】")
    print("   目标: 高命中率，低风险")
    print("   选择: TOP2生肖 + TOP6号码")
    
    top2_zodiacs = result['top6_zodiacs'][:2]
    top2_numbers = set()
    for zodiac, score in top2_zodiacs:
        top2_numbers.update(predictor.zodiac_numbers[zodiac])
    
    print(f"   生肖: {[z for z, s in top2_zodiacs]}")
    print(f"   号码: {sorted(list(top2_numbers))[:6]}")
    print(f"   覆盖: 约{len(top2_numbers)}个号码")
    
    # 策略2: 平衡型
    print("\n【策略2 - 平衡型】")
    print("   目标: 平衡命中率和覆盖面")
    print("   选择: TOP4生肖 + TOP12号码")
    
    top4_zodiacs = result['top6_zodiacs'][:4]
    top4_numbers = set()
    for zodiac, score in top4_zodiacs:
        top4_numbers.update(predictor.zodiac_numbers[zodiac])
    
    print(f"   生肖: {[z for z, s in top4_zodiacs]}")
    print(f"   号码: {result['top18_numbers'][:12]}")
    print(f"   覆盖: 约{len(top4_numbers)}个号码")
    
    # 策略3: 进取型
    print("\n【策略3 - 进取型】")
    print("   目标: 最大覆盖面")
    print("   选择: TOP6生肖 + TOP18号码")
    
    top6_numbers = set()
    for zodiac, score in result['top6_zodiacs']:
        top6_numbers.update(predictor.zodiac_numbers[zodiac])
    
    print(f"   生肖: {[z for z, s in result['top6_zodiacs']]}")
    print(f"   号码: {result['top18_numbers'][:18]}")
    print(f"   覆盖: 约{len(top6_numbers)}个号码")
    
    # 策略4: 组合型
    print("\n【策略4 - 组合型】⭐ 推荐")
    print("   目标: 结合其他模型")
    print("   选择: TOP6生肖 + 其他号码模型的交集")
    
    print(f"   步骤:")
    print(f"   1. 获取生肖TOP6推荐的号码")
    print(f"   2. 获取其他模型（如TOP15）的号码")
    print(f"   3. 取交集或按权重合并")
    print(f"   4. 优先选择高权重号码")
    
    print("\n" + "=" * 80 + "\n")


def demo_real_time_predict():
    """演示5: 实时预测展示"""
    print("=" * 80)
    print("演示5: 实时预测展示")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    print(f"\n{'='*80}")
    print("🐉 生肖TOP6预测模型 - 下一期预测")
    print("=" * 80)
    
    print(f"\n📅 最新一期（第{result['total_periods']}期）:")
    print(f"   日期: {result['last_date']}")
    print(f"   开出: {result['last_number']} - {result['last_zodiac']}")
    
    print(f"\n🔮 下一期预测（第{result['total_periods']+1}期）:\n")
    
    # 生肖预测
    print("⭐ 推荐生肖 TOP 6:")
    print("-" * 80)
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        
        if i <= 2:
            emoji = "⭐⭐"
            level = "强推"
        elif i <= 4:
            emoji = "⭐"
            level = "推荐"
        else:
            emoji = "✓"
            level = "备选"
        
        print(f"{emoji} {i}. {zodiac:2s} [{level}] (评分: {score:6.2f})  号码: {nums}")
    
    # 号码推荐
    print(f"\n📋 推荐号码 TOP 18（按生肖权重排序）:")
    print("-" * 80)
    top18 = result['top18_numbers']
    
    print(f"   ⭐⭐ TOP 1-6:   {top18[0:6]}")
    print(f"   ⭐  TOP 7-12:  {top18[6:12]}")
    print(f"   ✓  TOP 13-18: {top18[12:18]}")
    
    # 模型性能
    print(f"\n{'='*80}")
    print("📊 模型性能说明")
    print("=" * 80)
    print("   生肖TOP6 理论命中率: 50.0% (6/12)")
    print("   号码TOP18 理论命中率: 36.7% (18/49)")
    print("\n   本模型通过多维度分析优化，实际命中率可能高于理论值")
    
    # 使用建议
    print(f"\n{'='*80}")
    print("💡 使用建议")
    print("=" * 80)
    print("   1. ⭐⭐ 优先选择强推生肖（TOP2），成功率最高")
    print("   2. ⭐  搭配推荐生肖（TOP3-4）扩大范围")
    print("   3. ✓  备选生肖作为保险")
    print("   4. 📋  号码推荐已按权重排序，建议从TOP1-6开始选择")
    print("   5. 🔄  可与其他模型（如TOP15）组合使用，取交集")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    print("\n🎯 生肖TOP6预测模型 - 使用演示\n")
    
    demo_simple_usage()
    demo_detailed_output()
    demo_validation_results()
    demo_strategy_recommendation()
    demo_real_time_predict()
    
    print("=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print("\n提示: 运行 test_zodiac_top6.py 可进行完整的模型测试和验证\n")


if __name__ == '__main__':
    main()
