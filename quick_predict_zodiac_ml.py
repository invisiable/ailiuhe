"""
生肖ML预测 - 快速命令行版本
一键获取混合模型预测结果
"""

from zodiac_ml_predictor import ZodiacMLPredictor
import sys


def main():
    """
    快速预测
    
    用法:
        python quick_predict_zodiac_ml.py              # 默认配置(ML=40%)
        python quick_predict_zodiac_ml.py 0.5          # 自定义ML权重
        python quick_predict_zodiac_ml.py --pure-stat  # 纯统计模式
        python quick_predict_zodiac_ml.py --pure-ml    # 纯ML模式
    """
    
    # 解析参数
    ml_weight = 0.4  # 默认ML权重40%
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--pure-stat':
            ml_weight = 0.0
        elif arg == '--pure-ml':
            ml_weight = 1.0
        elif arg in ['--help', '-h']:
            print(__doc__)
            return
        else:
            try:
                ml_weight = float(arg)
                if not 0 <= ml_weight <= 1:
                    print("错误: ML权重必须在0-1之间")
                    return
            except:
                print(f"错误: 无效参数 '{arg}'")
                print("使用 --help 查看帮助")
                return
    
    print("\n" + "="*80)
    print("🤖 生肖预测 - 机器学习混合模型")
    print("="*80)
    
    # 创建预测器
    predictor = ZodiacMLPredictor(ml_weight=ml_weight)
    
    # 获取预测
    print("\n加载数据并训练模型...")
    result = predictor.predict()
    
    # 显示配置
    print(f"\n⚙️  模型配置")
    print(f"   模式: {result['model']}")
    print(f"   ML状态: {'✓ 已启用' if result['ml_enabled'] else '✗ 未启用'}")
    print(f"   权重配比: 统计{result['stat_weight']*100:.0f}% + ML{result['ml_weight']*100:.0f}%")
    
    # 显示最新一期
    print(f"\n📅 最新一期（第{result['total_periods']}期）")
    print(f"   日期: {result['last_date']}")
    print(f"   开出: {result['last_number']} - {result['last_zodiac']}")
    
    # 显示预测
    print(f"\n🔮 下一期预测（第{result['total_periods']+1}期）")
    print("="*80)
    
    # TOP6生肖
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
        
        # 显示统计评分和ML概率（如果可用）
        stat_score = result['stat_scores'][zodiac]
        extra_info = f"(统计:{stat_score:5.1f}"
        
        if result['ml_probs']:
            ml_prob = result['ml_probs'][zodiac]
            extra_info += f", ML:{ml_prob*100:4.1f}%"
        
        extra_info += ")"
        
        print(f"{emoji} {i}. {zodiac} [{level:4s}]  综合评分: {score:6.2f}  {extra_info}")
        print(f"      → 号码: {nums}")
    
    # TOP18号码
    print(f"\n📋 推荐号码 TOP 18:")
    top18 = result['top18_numbers']
    print(f"   强推 (1-6):   {top18[0:6]}")
    print(f"   推荐 (7-12):  {top18[6:12]}")
    print(f"   备选 (13-18): {top18[12:18]}")
    
    # 使用建议
    print("\n" + "="*80)
    print("💡 使用建议")
    print("="*80)
    print("   【保守型】选择 TOP2生肖 的号码")
    print("   【平衡型】选择 TOP3生肖 的号码 ⭐ 推荐")
    print("   【进取型】选择 TOP6生肖 + TOP12号码")
    
    # 模型说明
    if result['ml_enabled']:
        print("\n📊 模型说明")
        print(f"   ✓ 使用 {len(predictor.models)} 个机器学习模型")
        print(f"   ✓ 提取 100+ 维特征")
        print(f"   ✓ 统计评分 + ML预测概率 智能融合")
    else:
        print("\n📊 模型说明")
        print("   ✓ 使用纯统计分析模式")
        print("   ✓ 多维度频率、轮转、周期分析")
    
    print("\n" + "="*80)
    
    # 显示快捷命令提示
    print("\n💡 快捷命令:")
    print("   python quick_predict_zodiac_ml.py           # 平衡模式(ML=40%)")
    print("   python quick_predict_zodiac_ml.py 0.5       # 自定义权重")
    print("   python quick_predict_zodiac_ml.py --pure-stat  # 纯统计")
    print("   python quick_predict_zodiac_ml.py --pure-ml    # 纯ML")
    print()


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("\n❌ 错误: 找不到数据文件 data/lucky_numbers.csv")
        print("   请确保数据文件存在")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
