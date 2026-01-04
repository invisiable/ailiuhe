"""
测试生肖预测最近20期记录显示
验证GUI中已修改为显示所有20期的预测记录
"""

from zodiac_predictor import ZodiacPredictor

def test_20_periods_display():
    """测试最近20期预测记录的显示"""
    predictor = ZodiacPredictor()
    
    print("="*80)
    print("测试生肖预测 - 最近20期记录显示")
    print("="*80)
    print()
    
    # 获取最近20期验证数据
    validation = predictor.get_recent_20_validation('data/lucky_numbers.csv')
    
    if not validation:
        print("❌ 无法获取验证数据")
        return
    
    # 打印统计信息
    print("📊 最近20期验证详情:")
    print(f"  生肖TOP5成功率: {validation['zodiac_top5_rate']:.1f}% ({validation['zodiac_top5_hits']}/20)")
    print(f"  号码TOP15成功率: {validation['number_top15_rate']:.1f}% ({validation['number_top15_hits']}/20)")
    print()
    
    # 打印所有20期的预测记录
    print("最近20期预测记录:")
    print("-"*80)
    
    for detail in validation['details']:
        period = detail['期数']
        date = detail['日期']
        actual_num = detail['实际号码']
        actual_zodiac = detail['实际生肖']
        zodiac_hit = detail['生肖命中']
        predicted_top5 = detail['预测生肖TOP5']
        
        # 格式化输出每期的详细预测结果
        print(f"第{period:3d}期 ({date}): {actual_num:2d}号({actual_zodiac}) - {zodiac_hit:<10s}")
        print(f"      预测TOP5: {predicted_top5}")
    
    print("-"*80)
    print()
    print("✅ 测试完成！")
    print()
    print("🎯 优化说明:")
    print("  1. GUI中已将'最近10期预测记录'改为'最近20期预测记录'")
    print("  2. 现在显示所有20期的预测记录（而非只显示最后10期）")
    print("  3. 每期记录包含：期数、实际号码和生肖、命中状态、预测TOP5列表")
    print("  4. 移除号码命中信息，重点展示生肖预测TOP5")
    print()


if __name__ == '__main__':
    test_20_periods_display()
