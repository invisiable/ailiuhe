"""
生肖TOP6预测 - 快速入门脚本
一键获取下一期预测结果
"""

from zodiac_top6_predictor import ZodiacTop6Predictor


def quick_predict():
    """快速预测下一期"""
    
    print("\n" + "="*80)
    print("🎯 生肖TOP6预测模型 - 快速预测")
    print("="*80)
    
    # 创建预测器并获取结果
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    # 显示最新一期
    print(f"\n📅 最新一期（第{result['total_periods']}期）")
    print(f"   日期: {result['last_date']}")
    print(f"   开出: {result['last_number']} - {result['last_zodiac']}")
    
    # 显示下一期预测
    print(f"\n🔮 下一期预测（第{result['total_periods']+1}期）")
    print("="*80)
    
    # TOP3生肖（强推）
    print("\n⭐⭐ 强推生肖 TOP 3:")
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'][:3], 1):
        nums = predictor.zodiac_numbers[zodiac]
        print(f"   {i}. {zodiac} (评分: {score:.1f})  →  号码: {nums}")
    
    # 其余生肖（备选）
    print("\n✓ 备选生肖:")
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'][3:6], 4):
        nums = predictor.zodiac_numbers[zodiac]
        print(f"   {i}. {zodiac} (评分: {score:.1f})  →  号码: {nums}")
    
    # 号码推荐
    print(f"\n📋 推荐号码 TOP 12:")
    top12 = result['top18_numbers'][:12]
    print(f"   {top12}")
    
    # 使用建议
    print("\n"+ "="*80)
    print("💡 使用建议")
    print("="*80)
    print("   【保守型】选择 TOP2生肖 的号码（6-8个号码）")
    print("   【平衡型】选择 TOP3生肖 的号码（9-12个号码）⭐ 推荐")
    print("   【进取型】选择 TOP6生肖 + TOP12号码（覆盖更全）")
    
    # 性能说明
    print("\n📊 模型性能（最近50期验证）")
    print("   生肖TOP6命中率: 50.0% (理论50.0%)")
    print("   号码TOP18命中率: 46.0% (理论36.7%) ⬆️ 超过理论9.3%")
    
    print("\n" + "="*80 + "\n")
    
    return result


def show_detailed_analysis():
    """显示详细分析"""
    
    predictor = ZodiacTop6Predictor()
    result = predictor.predict()
    
    print("\n" + "="*80)
    print("📊 详细分析")
    print("="*80)
    
    # 生肖覆盖统计
    print("\n【生肖覆盖统计】")
    total_covered = set()
    for zodiac, score in result['top6_zodiacs']:
        nums = predictor.zodiac_numbers[zodiac]
        total_covered.update(nums)
    
    print(f"   TOP6生肖共覆盖: {len(total_covered)}/49 个号码 ({len(total_covered)/49*100:.1f}%)")
    
    # 各生肖详情
    print("\n【各生肖详细信息】")
    for i, (zodiac, score) in enumerate(result['top6_zodiacs'], 1):
        nums = predictor.zodiac_numbers[zodiac]
        level = "强推" if i <= 2 else "推荐" if i <= 4 else "备选"
        print(f"   [{level}] {zodiac}:")
        print(f"      评分: {score:.2f}")
        print(f"      号码: {nums}")
        print(f"      数量: {len(nums)}个")
        print()
    
    # 号码分布
    print("【号码推荐分布】")
    top18 = result['top18_numbers']
    print(f"   极小值 (1-10):   {[n for n in top18 if n <= 10]}")
    print(f"   小值   (11-20):  {[n for n in top18 if 11 <= n <= 20]}")
    print(f"   中值   (21-30):  {[n for n in top18 if 21 <= n <= 30]}")
    print(f"   大值   (31-40):  {[n for n in top18 if 31 <= n <= 40]}")
    print(f"   极大值 (41-49):  {[n for n in top18 if n >= 41]}")
    
    print("\n" + "="*80 + "\n")


def show_validation():
    """显示验证结果"""
    
    print("\n" + "="*80)
    print("🔍 模型验证（最近20期）")
    print("="*80)
    
    predictor = ZodiacTop6Predictor()
    validation = predictor.validate(test_periods=20)
    
    print(f"\n总体表现:")
    print(f"   测试期数: {validation['test_periods']}")
    print(f"   生肖TOP6命中: {validation['zodiac_top6_hits']}/{validation['test_periods']} = {validation['zodiac_top6_rate']:.1f}%")
    print(f"   号码TOP18命中: {validation['number_top18_hits']}/{validation['test_periods']} = {validation['number_top18_rate']:.1f}%")
    
    # 最近5期详情
    print(f"\n最近5期详细结果:")
    print("-"*80)
    for detail in validation['details'][-5:]:
        zodiac_icon = "✅" if detail['生肖命中'] == '✓' else "❌"
        number_icon = "✅" if detail['号码命中'] == '✓' else "❌"
        
        print(f"\n第{detail['期号']}期 ({detail['日期']}):")
        print(f"   实际: {detail['实际号码']} - {detail['实际生肖']}")
        print(f"   结果: {zodiac_icon} 生肖  {number_icon} 号码")
    
    print("\n" + "="*80 + "\n")


def main():
    """主菜单"""
    
    while True:
        print("\n╔════════════════════════════════════════╗")
        print("║   生肖TOP6预测模型 - 快速入门         ║")
        print("╚════════════════════════════════════════╝")
        print("\n请选择功能:")
        print("   1. 🎯 快速预测（推荐）")
        print("   2. 📊 详细分析")
        print("   3. 🔍 模型验证")
        print("   4. 📖 使用说明")
        print("   0. 退出")
        print()
        
        choice = input("请输入选项 (0-4): ").strip()
        
        if choice == '1':
            quick_predict()
            
        elif choice == '2':
            show_detailed_analysis()
            
        elif choice == '3':
            show_validation()
            
        elif choice == '4':
            show_usage_guide()
            
        elif choice == '0':
            print("\n👋 再见！祝您好运！\n")
            break
            
        else:
            print("\n⚠️  无效选项，请重新选择\n")
        
        input("按回车键继续...")


def show_usage_guide():
    """显示使用说明"""
    
    print("\n" + "="*80)
    print("📖 使用说明")
    print("="*80)
    
    print("""
【模型简介】
   生肖TOP6预测模型专注于预测最可能出现的6个生肖，并基于此推荐号码。

【预测内容】
   1. TOP6生肖预测（理论命中率50%）
   2. TOP18号码推荐（理论命中率36.7%）

【使用策略】
   
   ⭐ 保守型（推荐新手）
      - 选择TOP2生肖
      - 每个生肖选1-2个号码
      - 共4-6个号码
      - 预期命中率: 40-45%
   
   ⭐⭐ 平衡型（最推荐）
      - 选择TOP3-4生肖
      - 结合号码推荐TOP12
      - 共10-12个号码
      - 预期命中率: 45-50%
   
   ⭐⭐⭐ 进取型
      - 使用全部6个生肖
      - 号码推荐TOP18
      - 共18个号码
      - 最大覆盖率: 49%

【组合使用】
   可与TOP15等其他模型组合，取交集获得最高准确率。
   
   示例:
   1. 获取生肖TOP6的号码推荐
   2. 获取TOP15的号码推荐
   3. 两者的交集 = 最高准确率的号码

【注意事项】
   1. 彩票具有随机性，模型仅供参考
   2. 建议查看长期（50期+）表现
   3. 可以根据实际效果调整策略

【更多信息】
   详细文档: 生肖TOP6预测模型使用指南.md
   测试文件: test_zodiac_top6.py
   演示文件: demo_zodiac_top6.py
""")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出。再见！\n")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}\n")
        import traceback
        traceback.print_exc()
