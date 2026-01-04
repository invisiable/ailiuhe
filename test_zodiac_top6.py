"""
测试生肖TOP6预测模型
验证模型性能和预测效果
"""

from zodiac_top6_predictor import ZodiacTop6Predictor
import pandas as pd


def test_basic_prediction():
    """测试基本预测功能"""
    print("=" * 80)
    print("测试1: 基本预测功能")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    print(f"\n✓ 模型名称: {result['model_info']['name']}")
    print(f"✓ 版本: {result['model_info']['version']}")
    print(f"✓ 数据期数: {result['total_periods']}")
    print(f"✓ 最新一期: {result['last_date']} - {result['last_number']} ({result['last_zodiac']})")
    
    print(f"\n✓ TOP6生肖预测:")
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
        print(f"   {i}. {zodiac} (评分: {score:.2f})")
    
    print(f"\n✓ TOP18号码推荐:")
    print(f"   {result['top18_numbers']}")
    
    assert len(result['top6_zodiacs']) == 6, "应该返回6个生肖"
    assert len(result['top18_numbers']) == 18, "应该返回18个号码"
    
    print("\n✅ 基本预测功能测试通过！\n")


def test_validation_accuracy():
    """测试验证功能和准确率"""
    print("=" * 80)
    print("测试2: 模型验证（最近20期）")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    validation = predictor.validate(test_periods=20)
    
    print(f"\n测试期数: {validation['test_periods']}")
    print(f"\n生肖TOP6命中情况:")
    print(f"   命中次数: {validation['zodiac_top6_hits']}")
    print(f"   命中率: {validation['zodiac_top6_rate']:.1f}%")
    
    print(f"\n号码TOP18命中情况:")
    print(f"   命中次数: {validation['number_top18_hits']}")
    print(f"   命中率: {validation['number_top18_rate']:.1f}%")
    
    # 显示前5期详细结果
    print(f"\n前5期详细验证结果:")
    print("-" * 80)
    for detail in validation['details'][:5]:
        print(f"\n第{detail['期号']}期 ({detail['日期']}):")
        print(f"   实际: {detail['实际号码']} - {detail['实际生肖']}")
        print(f"   预测生肖: {detail['预测生肖TOP6']}")
        print(f"   结果: 生肖{detail['生肖命中']} 号码{detail['号码命中']}")
    
    # 性能评估
    print(f"\n{'='*80}")
    print("📊 性能评估")
    print("=" * 80)
    
    zodiac_rate = validation['zodiac_top6_rate']
    number_rate = validation['number_top18_rate']
    
    # 理论命中率：TOP6生肖 = 6/12 = 50%, TOP18号码 = 18/49 = 36.7%
    print(f"   生肖TOP6理论命中率: 50.0% (6/12)")
    print(f"   生肖TOP6实际命中率: {zodiac_rate:.1f}%", end="")
    if zodiac_rate > 50:
        print(f" ⬆️ 超过理论值 {zodiac_rate - 50:.1f}%")
    else:
        print(f" ⬇️ 低于理论值 {50 - zodiac_rate:.1f}%")
    
    print(f"\n   号码TOP18理论命中率: 36.7% (18/49)")
    print(f"   号码TOP18实际命中率: {number_rate:.1f}%", end="")
    if number_rate > 36.7:
        print(f" ⬆️ 超过理论值 {number_rate - 36.7:.1f}%")
    else:
        print(f" ⬇️ 低于理论值 {36.7 - number_rate:.1f}%")
    
    print("\n✅ 模型验证测试完成！\n")


def test_validation_different_periods():
    """测试不同期数的验证"""
    print("=" * 80)
    print("测试3: 不同期数验证对比")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    
    test_configs = [10, 20, 30, 50]
    results = []
    
    print("\n测试不同期数的模型表现...\n")
    
    for periods in test_configs:
        validation = predictor.validate(test_periods=periods)
        results.append({
            'periods': periods,
            'zodiac_rate': validation['zodiac_top6_rate'],
            'number_rate': validation['number_top18_rate']
        })
        print(f"✓ {periods}期验证完成")
    
    # 显示对比结果
    print(f"\n{'='*80}")
    print("期数对比结果:")
    print("=" * 80)
    print(f"{'期数':<10} {'生肖TOP6命中率':<20} {'号码TOP18命中率':<20}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['periods']:<10} {r['zodiac_rate']:>6.1f}% {'':<13} {r['number_rate']:>6.1f}%")
    
    print("\n✅ 不同期数验证测试完成！\n")


def test_zodiac_coverage():
    """测试生肖覆盖情况"""
    print("=" * 80)
    print("测试4: 生肖覆盖分析")
    print("=" * 80)
    
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    # 统计TOP6生肖覆盖的号码数量
    covered_numbers = set()
    for zodiac, score in result['top6_zodiacs']:
        numbers = predictor.zodiac_numbers[zodiac]
        covered_numbers.update(numbers)
    
    print(f"\nTOP6生肖覆盖情况:")
    print(f"   覆盖号码数量: {len(covered_numbers)}/49")
    print(f"   覆盖率: {len(covered_numbers)/49*100:.1f}%")
    
    print(f"\n各生肖号码数量:")
    for zodiac, score in result['top6_zodiacs']:
        numbers = predictor.zodiac_numbers[zodiac]
        print(f"   {zodiac}: {len(numbers)}个号码 - {numbers}")
    
    print(f"\n理论覆盖范围:")
    # 12生肖覆盖1-48号，49号只有鼠生肖
    print(f"   6个生肖理论上可以覆盖约25个号码")
    print(f"   实际覆盖: {len(covered_numbers)}个号码")
    
    print("\n✅ 生肖覆盖分析完成！\n")


def test_comparison_with_top5():
    """与TOP5模型对比"""
    print("=" * 80)
    print("测试5: 与生肖TOP5模型对比")
    print("=" * 80)
    
    # 导入TOP5模型（如果存在）
    try:
        from zodiac_predictor import ZodiacPredictor
        
        top5_predictor = ZodiacPredictor()
        top6_predictor = ZodiacTop6Predictor()
        
        # TOP5验证
        print("\n正在验证TOP5模型...")
        top5_result = top5_predictor.predict()
        
        # TOP6验证
        print("正在验证TOP6模型...")
        top6_result = top6_predictor.predict()
        
        print(f"\n{'='*80}")
        print("模型对比（下一期预测）:")
        print("=" * 80)
        
        print(f"\nTOP5模型:")
        print(f"   推荐生肖数: 5个")
        for i, (zodiac, score) in enumerate(top5_result['top5_zodiacs'], 1):
            print(f"   {i}. {zodiac} ({score:.2f})")
        
        print(f"\nTOP6模型:")
        print(f"   推荐生肖数: 6个")
        for i, (zodiac, score) in enumerate(top6_result['top6_zodiacs'], 1):
            print(f"   {i}. {zodiac} ({score:.2f})")
        
        # 计算重叠度
        top5_set = set([z for z, s in top5_result['top5_zodiacs']])
        top6_set = set([z for z, s in top6_result['top6_zodiacs']])
        overlap = top5_set & top6_set
        
        print(f"\n重叠分析:")
        print(f"   重叠生肖数: {len(overlap)}/5")
        print(f"   重叠生肖: {list(overlap)}")
        print(f"   TOP6独有: {list(top6_set - top5_set)}")
        
        print("\n✅ 模型对比完成！\n")
        
    except ImportError:
        print("\n⚠️  未找到ZodiacPredictor模型，跳过对比测试\n")


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("🚀 开始测试生肖TOP6预测模型")
    print("=" * 80)
    
    test_basic_prediction()
    test_validation_accuracy()
    test_validation_different_periods()
    test_zodiac_coverage()
    test_comparison_with_top5()
    
    print("=" * 80)
    print("🎉 所有测试完成！")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()
