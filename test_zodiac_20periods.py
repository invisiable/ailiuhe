"""测试生肖预测器的最近20期验证功能"""
from zodiac_predictor import ZodiacPredictor
import pandas as pd

print("="*70)
print("测试生肖预测器 - 最近20期验证")
print("="*70)

predictor = ZodiacPredictor()

print("\n正在获取最近20期验证数据...")
validation = predictor.get_recent_20_validation()

if validation:
    print(f"\n✅ 最近20期验证结果:")
    print(f"   生肖TOP5成功率: {validation['zodiac_top5_rate']:.1f}% ({validation['zodiac_top5_hits']}/20)")
    print(f"   号码TOP15成功率: {validation['number_top15_rate']:.1f}% ({validation['number_top15_hits']}/20)")
    
    print(f"\n📊 详细预测记录 (最近10期):")
    print("-"*70)
    
    recent_10 = validation['details'][-10:]
    for detail in recent_10:
        period = detail['期数']
        date = detail['日期']
        actual_num = detail['实际号码']
        actual_zodiac = detail['实际生肖']
        zodiac_hit = detail['生肖命中']
        number_hit = detail['号码命中']
        
        print(f"第{period:3d}期 ({date}): {actual_num:2d}号 ({actual_zodiac}) - 生肖{zodiac_hit} 号码{number_hit}")
    
    print("\n" + "="*70)
    print("✅ 测试完成")
else:
    print("\n❌ 数据不足20期，无法验证")
