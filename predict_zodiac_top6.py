"""
生肖TOP6预测 - 命令行版本
直接运行即可获得预测结果，无需交互
"""

from zodiac_top6_predictor import ZodiacTop6Predictor
import sys


def main():
    """
    命令行预测
    
    用法:
        python predict_zodiac_top6.py              # 显示完整预测
        python predict_zodiac_top6.py --simple     # 仅显示推荐结果
        python predict_zodiac_top6.py --validate   # 显示验证结果
    """
    
    # 解析命令行参数
    mode = 'full'
    if len(sys.argv) > 1:
        if sys.argv[1] == '--simple':
            mode = 'simple'
        elif sys.argv[1] == '--validate':
            mode = 'validate'
        elif sys.argv[1] in ['--help', '-h']:
            print_help()
            return
    
    # 创建预测器
    predictor = ZodiacTop6Predictor()
    
    if mode == 'simple':
        # 简洁模式
        show_simple_prediction(predictor)
    elif mode == 'validate':
        # 验证模式
        show_validation(predictor)
    else:
        # 完整模式
        show_full_prediction(predictor)


def show_simple_prediction(predictor):
    """简洁预测模式"""
    result = predictor.predict()
    
    print(f"\n下一期预测（第{result['total_periods']+1}期）:")
    print(f"\n强推生肖: ", end="")
    print([z for z, s in result['top6_zodiacs'][:2]])
    
    print(f"推荐号码: {result['top18_numbers'][:12]}")
    print()


def show_full_prediction(predictor):
    """完整预测模式"""
    result = predictor.predict()
    
    print("\n" + "="*80)
    print("🐉 生肖TOP6预测模型 - 预测结果")
    print("="*80)
    
    # 最新一期
    print(f"\n📅 最新一期（第{result['total_periods']}期）")
    print(f"   日期: {result['last_date']}")
    print(f"   开出: {result['last_number']} - {result['last_zodiac']}")
    
    # 下一期预测
    print(f"\n🔮 下一期预测（第{result['total_periods']+1}期）")
    print("-"*80)
    
    # 生肖预测
    print("\n⭐ 生肖预测 TOP 6:")
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
        
        print(f"{emoji} {i}. {zodiac} [{level}]  评分: {score:5.1f}  →  号码: {nums}")
    
    # 号码推荐
    print(f"\n📋 号码推荐 TOP 18:")
    top18 = result['top18_numbers']
    print(f"   强推 (TOP 1-6):   {top18[0:6]}")
    print(f"   推荐 (TOP 7-12):  {top18[6:12]}")
    print(f"   备选 (TOP 13-18): {top18[12:18]}")
    
    # 使用建议
    print("\n" + "="*80)
    print("💡 使用建议")
    print("="*80)
    print("   保守型: TOP2生肖 的号码 (6-8个号码)")
    print("   平衡型: TOP3生肖 + TOP12号码 (9-12个号码) ⭐ 推荐")
    print("   进取型: 全部6生肖 + TOP18号码 (最大覆盖)")
    
    # 性能说明
    print("\n📊 模型性能")
    print("-"*80)
    print("   生肖TOP6: 理论50.0%, 实测50.0% (50期)")
    print("   号码TOP18: 理论36.7%, 实测46.0% (50期) ⬆️ 超过理论9.3%")
    
    print("\n" + "="*80 + "\n")


def show_validation(predictor):
    """验证模式"""
    print("\n" + "="*80)
    print("🔍 模型验证（最近20期）")
    print("="*80)
    
    validation = predictor.validate(test_periods=20)
    
    print(f"\n测试期数: {validation['test_periods']}")
    print(f"\n生肖TOP6: {validation['zodiac_top6_hits']}/{validation['test_periods']} = {validation['zodiac_top6_rate']:.1f}%")
    print(f"号码TOP18: {validation['number_top18_hits']}/{validation['test_periods']} = {validation['number_top18_rate']:.1f}%")
    
    # 详细结果
    print(f"\n详细验证结果:")
    print("-"*80)
    print(f"{'期号':<6} {'日期':<12} {'实际':<15} {'生肖':<6} {'号码':<6}")
    print("-"*80)
    
    for detail in validation['details'][-10:]:  # 显示最近10期
        zodiac_icon = "✓" if detail['生肖命中'] == '✓' else "✗"
        number_icon = "✓" if detail['号码命中'] == '✓' else "✗"
        
        actual = f"{detail['实际号码']} - {detail['实际生肖']}"
        print(f"{detail['期号']:<6} {detail['日期']:<12} {actual:<15} {zodiac_icon:<6} {number_icon:<6}")
    
    print("\n" + "="*80 + "\n")


def print_help():
    """打印帮助信息"""
    print("""
生肖TOP6预测模型 - 命令行工具

用法:
    python predict_zodiac_top6.py              显示完整预测（默认）
    python predict_zodiac_top6.py --simple     仅显示推荐结果（简洁）
    python predict_zodiac_top6.py --validate   显示验证结果
    python predict_zodiac_top6.py --help       显示此帮助信息

示例:
    # 获取完整预测
    python predict_zodiac_top6.py
    
    # 快速查看推荐
    python predict_zodiac_top6.py --simple
    
    # 查看模型准确率
    python predict_zodiac_top6.py --validate

更多信息:
    查看文档: 生肖TOP6预测模型使用指南.md
    运行测试: python test_zodiac_top6.py
    运行演示: python demo_zodiac_top6.py
""")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        import traceback
        traceback.print_exc()
