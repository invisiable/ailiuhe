"""
生肖预测演示 - 包含最近20期验证数据
"""
from zodiac_predictor import ZodiacPredictor
from datetime import datetime

def main():
    print("=" * 80)
    print("🐉 生肖预测模型 - 下一期预测（含最近20期验证）")
    print("=" * 80)
    
    predictor = ZodiacPredictor()
    
    # 1. 获取下一期预测
    print("\n📊 正在生成预测...")
    result = predictor.predict()
    
    print(f"\n【基础信息】")
    print(f"  预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据期数: {result['total_periods']}期")
    print(f"  最新一期: 第{result['total_periods']}期 ({result['last_date']})")
    print(f"  开出结果: {result['last_number']}号 ({result['last_zodiac']})")
    
    print(f"\n{'='*80}")
    print(f"🎯 下一期预测（第{result['total_periods']+1}期）")
    print(f"{'='*80}")
    
    # 2. 显示生肖预测
    print(f"\n⭐ 推荐生肖 TOP 5:")
    print("-" * 80)
    for i, (zodiac, score) in enumerate(result['top5_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        nums_str = ', '.join(map(str, nums))
        
        if i <= 2:
            marker = "⭐⭐"
        elif i == 3:
            marker = "⭐"
        else:
            marker = "○"
        
        print(f"  {marker} {i}. {zodiac:2s} (评分: {score:5.2f})")
        print(f"      对应号码: {nums_str}")
    
    # 3. 显示号码推荐
    print(f"\n📋 推荐号码 TOP 15 (基于生肖):")
    print("-" * 80)
    top5 = result['top15_numbers'][:5]
    top10 = result['top15_numbers'][5:10]
    top15 = result['top15_numbers'][10:15]
    print(f"  TOP 1-5:   {top5}")
    print(f"  TOP 6-10:  {top10}")
    print(f"  TOP 11-15: {top15}")
    
    # 4. 获取最近20期验证数据
    print(f"\n{'='*80}")
    print("📊 最近20期验证数据")
    print(f"{'='*80}")
    
    print("\n正在验证最近20期预测准确率...")
    validation = predictor.get_recent_20_validation()
    
    if validation:
        zodiac_rate = validation['zodiac_top5_rate']
        number_rate = validation['number_top15_rate']
        zodiac_hits = validation['zodiac_top5_hits']
        number_hits = validation['number_top15_hits']
        
        print(f"\n【验证结果统计】")
        print(f"  验证期数: 20期")
        print(f"  生肖TOP5成功率: {zodiac_rate:.1f}% ({zodiac_hits}/20) {'✅ 优秀' if zodiac_rate >= 50 else '✓ 良好'}")
        print(f"  号码TOP15成功率: {number_rate:.1f}% ({number_hits}/20) {'✅ 优秀' if number_rate >= 30 else '✓ 良好'}")
        
        # 显示全部20期的详细数据
        print(f"\n【详细预测记录】（最近20期）")
        print("-" * 80)
        print(f"{'期数':>4s} | {'日期':^10s} | {'号码':>2s} | {'生肖':^2s} | {'生肖预测':^10s} | {'号码预测':^10s}")
        print("-" * 80)
        
        for detail in validation['details']:
            period = detail['期数']
            date = detail['日期']
            actual_num = detail['实际号码']
            actual_zodiac = detail['实际生肖']
            zodiac_hit = detail['生肖命中']
            number_hit = detail['号码命中']
            
            print(f"{period:4d} | {date:10s} | {actual_num:2d} | {actual_zodiac:2s} | {zodiac_hit:^12s} | {number_hit:^12s}")
        
        print("-" * 80)
        
        # 统计分析
        print(f"\n【验证分析】")
        details = validation['details']
        
        # 前10期 vs 后10期
        first_10 = details[:10]
        last_10 = details[10:]
        
        first_10_zodiac = sum(1 for d in first_10 if '✅' in d['生肖命中'])
        last_10_zodiac = sum(1 for d in last_10 if '✅' in d['生肖命中'])
        
        first_10_number = sum(1 for d in first_10 if '✅' in d['号码命中'])
        last_10_number = sum(1 for d in last_10 if '✅' in d['号码命中'])
        
        print(f"  前10期 (第{first_10[0]['期数']}-{first_10[-1]['期数']}期):")
        print(f"    生肖TOP5: {first_10_zodiac}/10 ({first_10_zodiac*10}%)")
        print(f"    号码TOP15: {first_10_number}/10 ({first_10_number*10}%)")
        
        print(f"  后10期 (第{last_10[0]['期数']}-{last_10[-1]['期数']}期):")
        print(f"    生肖TOP5: {last_10_zodiac}/10 ({last_10_zodiac*10}%)")
        print(f"    号码TOP15: {last_10_number}/10 ({last_10_number*10}%)")
        
        # 趋势分析
        if last_10_zodiac > first_10_zodiac:
            trend = "📈 上升趋势"
        elif last_10_zodiac < first_10_zodiac:
            trend = "📉 下降趋势"
        else:
            trend = "➡️ 稳定"
        
        print(f"\n  近期表现: {trend}")
        
    else:
        print("\n⚠️ 数据不足20期，无法进行验证")
    
    # 5. 使用建议
    print(f"\n{'='*80}")
    print("💡 使用建议")
    print(f"{'='*80}")
    print("\n  推荐策略:")
    print("    1. 主要考虑 TOP1-2 生肖（高置信度）")
    print("    2. TOP3 生肖作为重要备选")
    print("    3. 从 TOP3 生肖对应的约12个号码中精选")
    print("    4. 结合其他预测模型（如TOP20）进一步优化")
    print("\n  预期效果:")
    print(f"    • 生肖TOP5命中率: ~{zodiac_rate:.0f}%（基于最近20期）")
    print(f"    • 号码TOP15命中率: ~{number_rate:.0f}%（基于最近20期）")
    print(f"    • 生肖预测稳定性高，建议作为主要参考指标")
    
    print(f"\n{'='*80}")
    print("✅ 预测完成")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
